"""Bandwidth-lean pseudo-spectral RHS and RK stepping for the 3D Euler equations.

Every hot kernel of the reference 3D path (cuFFT and elementwise alike) runs
close to the DRAM roofline, so speed comes from moving fewer bytes:

* real-to-complex transforms (half-spectrum storage);
* the flux divergence is formed in spectral space, so each RHS needs one
  inverse transform per conserved variable instead of one per flux component;
* the momentum flux tensor is symmetric and the mass flux is the momentum, so
  only 9 physical flux fields are transformed (rho*u_i*u_j + p*delta_ij and
  (E+p)*u_i), with the mass-flux spectrum taken from the stage state;
* RK stages are combined in spectral space, which folds the Fourier split-step
  dissipation and the dealias projection into the final update.

The result is the reference step to round-off: ``grid.apply_spectral_dissipation``
for half a step, RK2/RK4 with ``_compute_rhs_3d`` (2/3-masked derivatives) and
the positivity clamp on every stage, then ``grid.apply_accepted_state_stabilization``.
The clamp leaves a positive state untouched, so the fast path counts floor
violations on device and returns ``None`` when a clamp would act; the solver
then repeats that step on the reference path.

Execution tiers: NumPy/SciPy or eager Torch everywhere; on CUDA and MPS the
pointwise kernels are fused with ``torch.compile`` (``PHRIKE_COMPILE=0``
disables it); on CUDA each RK stage is one Triton kernel (``spectral3d_triton``)
that visits only the modes kept by the 2/3 rule once the state is band-limited.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Sequence, Tuple

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch may not be installed
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    import pyfftw.interfaces.scipy_fft as scipy_fft  # type: ignore
except Exception:
    from scipy import fft as scipy_fft  # type: ignore

from .grid import _build_exponential_filter_rate, _build_filter_mask, _spectral_dissipation_settings


Array = Any

# Classical RK schemes written as (w_acc, w_next) per stage: after stage i,
# acc += w_acc[i]*dt*k_i and the next stage state is U0 + w_next[i]*dt*k_i.
# The final update is U0 + acc + w_acc[-1]*dt*k_last.
_RK_TABLES = {
    "rk2": ((0.0, 0.5), (1.0, None)),
    "rk4": ((1.0 / 6.0, 0.5), (1.0 / 3.0, 0.5), (1.0 / 3.0, 1.0), (1.0 / 6.0, None)),
}


def _is_torch(a: Any) -> bool:
    return _TORCH_AVAILABLE and isinstance(a, torch.Tensor)


def _stack(items: Sequence[Array]) -> Array:
    if _is_torch(items[0]):
        return torch.stack(list(items), dim=0)
    return np.stack(items, axis=0)


def _compile_default(device: str) -> bool:
    flag = os.environ.get("PHRIKE_COMPILE")
    if flag is not None:
        return flag.strip().lower() not in ("0", "false", "no", "off", "")
    return device.startswith(("cuda", "mps"))


# --- Pointwise kernels (backend-agnostic; compiled with torch.compile on CUDA/MPS) ---


def _primitive(U, gamma):
    rho, mx, my, mz, E = U[0], U[1], U[2], U[3], U[4]
    ux = mx / rho
    uy = my / rho
    uz = mz / rho
    # Same expression order as EulerEquations3D.primitive.
    p = (gamma - 1.0) * (E - 0.5 * rho * (ux * ux + uy * uy + uz * uz))
    return rho, mx, my, mz, E, ux, uy, uz, p


def euler3d_symmetric_fluxes(U: Array, gamma: float) -> Array:
    """Return the 9 independent flux fields of the 3D Euler equations.

    Order: [Txx, Tyy, Tzz, Txy, Txz, Tyz, Hx, Hy, Hz] with
    T_ij = rho*u_i*u_j + p*delta_ij and H_i = (E + p)*u_i.
    """
    rho, mx, my, mz, E, ux, uy, uz, p = _primitive(U, gamma)
    H = E + p
    return _stack(
        [mx * ux + p, my * uy + p, mz * uz + p, mx * uy, mx * uz, my * uz, H * ux, H * uy, H * uz]
    )


def _count_below(U, gamma, rho_min, p_min):
    """Values a positivity clamp would raise (NaN never counts, as in the clamp)."""
    rho, _, _, _, _, _, _, _, p = _primitive(U, gamma)
    if _is_torch(rho):
        return (rho < rho_min).sum() + (p < p_min).sum()
    return int(np.count_nonzero(rho < rho_min) + np.count_nonzero(p < p_min))


def _fluxes_and_count(U, gamma, rho_min, p_min):
    return euler3d_symmetric_fluxes(U, gamma), _count_below(U, gamma, rho_min, p_min)


def _embedded_checks(U_high, E, gamma, rho_min, p_min):
    """Floor violations of U_high and U_low = U_high - E, and max|E|."""
    count = _count_below(U_high, gamma, rho_min, p_min) + _count_below(
        U_high - E, gamma, rho_min, p_min
    )
    return count, torch.max(torch.abs(E))


def _finite_and_speed(U, gamma):
    """(all finite, max |u_i| + c) in one pass, NaN-propagating like the reference."""
    rho, _, _, _, _, ux, uy, uz, p = _primitive(U, gamma)
    a = torch.sqrt(gamma * p / rho)
    comp = torch.maximum(torch.maximum(torch.abs(ux), torch.abs(uy)), torch.abs(uz)) + a
    return torch.isfinite(U).all(), torch.max(comp)


def _div_hat_complex(ikx, iky, ikz, M, ax, ay, az):
    return -M * (ikx * ax + iky * ay + ikz * az)


def _rhs_hat_complex(ikx, iky, ikz, M, Fh, Mh):
    """Spectral RHS from flux spectra Fh (9 fields) and momentum spectra Mh (3)."""
    d = _div_hat_complex
    return _stack(
        [
            d(ikx, iky, ikz, M, Mh[0], Mh[1], Mh[2]),
            d(ikx, iky, ikz, M, Fh[0], Fh[3], Fh[4]),
            d(ikx, iky, ikz, M, Fh[3], Fh[1], Fh[5]),
            d(ikx, iky, ikz, M, Fh[4], Fh[5], Fh[2]),
            d(ikx, iky, ikz, M, Fh[6], Fh[7], Fh[8]),
        ]
    )


def _rhs_hat_real(kx, ky, kz, M, Fr, Mr):
    """``_rhs_hat_complex`` on view_as_real arrays (Inductor does not fuse complex math)."""

    def d(ax, ay, az):
        sr = kx * ax[..., 0] + ky * ay[..., 0] + kz * az[..., 0]
        si = kx * ax[..., 1] + ky * ay[..., 1] + kz * az[..., 1]
        # -M * i * (sr + i si) = M*si - i*M*sr
        return torch.stack([M * si, -M * sr], dim=-1)

    return torch.stack(
        [d(Mr[0], Mr[1], Mr[2]), d(Fr[0], Fr[3], Fr[4]), d(Fr[3], Fr[1], Fr[5]),
         d(Fr[4], Fr[5], Fr[2]), d(Fr[6], Fr[7], Fr[8])]
    )


class Euler3DSpectral:
    """Real-to-complex spectral operators and RK stepping for ``Grid3D`` + Euler.

    Works with NumPy and Torch grids.  Spectral arrays use the ``rfftn`` layout
    (Nz, Ny, Nx//2 + 1).  ``Euler2DSpectral`` reuses everything for 2D grids
    by viewing (Ny, Nx//2 + 1) spectra as (1, Ny, Nx//2 + 1) in the kernels.
    """

    NVAR = 5
    MOM = slice(1, 4)

    @staticmethod
    def _grid_shape(grid):
        return (grid.Nz, grid.Ny, grid.Nx)

    def _pointwise(self):
        return dict(fluxes=euler3d_symmetric_fluxes, fluxes_count=_fluxes_and_count,
                    count=_count_below, checks=_embedded_checks, speed=_finite_and_speed,
                    rhs_real=_rhs_hat_real)

    def __init__(self, grid, equations, rho_floor: float, p_floor: float,
                 compile: Optional[bool] = None):
        self.grid = grid
        self.gamma = float(equations.gamma)
        self.rho_floor = float(rho_floor)
        self.p_floor = float(p_floor)
        self.shape = self._grid_shape(grid)
        self._dims = tuple(range(-len(self.shape), 0))
        self._kshape = (1,) * (3 - len(self.shape)) + self.shape
        self._fn = self._pointwise()
        self._torch = bool(getattr(grid, "_use_torch", False))

        Nz, Ny, Nx = self._kshape
        Xh = Nx // 2 + 1
        nd = len(self.shape)
        kx = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=grid.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=grid.dy) if nd >= 2 else np.zeros(1)
        kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=grid.dz) if nd == 3 else np.zeros(1)
        # The Nyquist mode has no real-valued derivative; zeroing it matches the
        # complex-FFT path, which drops that contribution when taking .real.
        if Nx % 2 == 0:
            kx[-1] = 0.0
        if Ny % 2 == 0:
            ky[Ny // 2] = 0.0
        if Nz % 2 == 0:
            kz[Nz // 2] = 0.0
        # Native (unbatched) spectral shapes: (Nz, Ny, Xh) in 3D, (Ny, Xh) in 2D.
        mask = np.ascontiguousarray(np.asarray(_to_numpy(grid.dealias_mask))[..., :Xh])
        self.dissipation = bool(getattr(grid, "spectral_dissipation_enabled", False))
        lam = np.ascontiguousarray(np.asarray(_to_numpy(grid.filter_rate))[..., :Xh])
        bshape = {3: ((1, 1, Xh), (1, Ny, 1), (Nz, 1, 1)), 2: ((1, Xh), (Ny, 1), (1, 1)),
                  1: ((Xh,), (1,), (1,))}[len(self.shape)]
        kx3, ky3, kz3 = (a.reshape(sh) for a, sh in zip((kx, ky, kz), bshape))

        if self._torch:
            ref = _ref_tensor(grid)
            dev = ref.device
            rdt = ref.dtype
            cdt = torch.complex64 if rdt == torch.float32 else torch.complex128
            self.device = str(dev)
            self.rdtype = rdt
            to_r = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=rdt, device=dev)
            self.kx, self.ky, self.kz = to_r(kx3), to_r(ky3), to_r(kz3)
            self.M, self.lam = to_r(mask), to_r(lam)
            self.ikx = (1j * self.kx).to(cdt)
            self.iky = (1j * self.ky).to(cdt)
            self.ikz = (1j * self.kz).to(cdt)
        else:
            self.device = "cpu"
            self.kx, self.ky, self.kz = kx3, ky3, kz3
            self.M, self.lam = mask, lam
            self.ikx, self.iky, self.ikz = 1j * self.kx, 1j * self.ky, 1j * self.kz

        if compile is None:
            compile = self._torch and _compile_default(self.device)
        self._compiled = False
        self._triton = None
        self._cache = None  # (U, U._version) whose band-limited spectrum is in self._uhat
        self._pending = None  # (U, U._version) of an uncommitted embedded step
        self._kbuf = None  # compact stage RHS for embedded RK
        if compile and self._torch:
            self._setup_compiled()
            if self.device.startswith("cuda"):
                self._setup_triton(kx, ky, kz, mask, lam)

    # --- transforms -----------------------------------------------------

    def rfftn(self, f: Array) -> Array:
        if self._torch:
            return torch.fft.rfftn(f, dim=self._dims)
        try:
            return scipy_fft.rfftn(f, axes=self._dims, workers=self.grid.fft_workers)
        except TypeError:
            return scipy_fft.rfftn(f, axes=self._dims)

    def irfftn(self, F: Array) -> Array:
        if self._torch:
            return torch.fft.irfftn(F, s=self.shape, dim=self._dims)
        try:
            return scipy_fft.irfftn(F, s=self.shape, axes=self._dims, workers=self.grid.fft_workers)
        except TypeError:
            return scipy_fft.irfftn(F, s=self.shape, axes=self._dims)

    def _irfftn_fwd(self, F):
        """Inverse of a ``norm="forward"`` spectrum (no 1/N pass)."""
        return torch.fft.irfftn(F, s=self.shape, dim=self._dims, norm="forward")

    def _decay(self, h: float):
        """exp(-h * lambda) on the half spectrum."""
        if self._torch:
            return torch.exp(-float(h) * self.lam)
        return np.exp(-float(h) * self.lam)

    # --- compiled kernels -------------------------------------------------

    def _setup_compiled(self) -> None:
        try:
            opts = dict(fullgraph=True, dynamic=False)
            fn = self._fn
            self._c_fluxes = torch.compile(fn["fluxes"], **opts)
            self._c_fluxes_count = torch.compile(fn["fluxes_count"], **opts)
            self._c_count = torch.compile(fn["count"], **opts)
            self._c_embedded_checks = torch.compile(fn["checks"], **opts)
            self._c_finite_speed = torch.compile(fn["speed"], **opts)
            self._c_rhs = torch.compile(fn["rhs_real"], **opts) if fn["rhs_real"] else None
            self._compiled = True
        except Exception as exc:  # pragma: no cover - depends on toolchain
            warnings.warn(f"torch.compile unavailable for 3D Euler kernels ({exc}); using eager.")
            self._compiled = False

    def _setup_triton(self, kx, ky, kz, mask, lam) -> None:
        try:
            from . import spectral3d_triton as st
        except Exception as exc:  # pragma: no cover - depends on toolchain
            warnings.warn(f"Triton unavailable for 3D Euler RK stages ({exc}); using torch kernels.")
            return
        # The kernels rebuild the dealias mask and the dissipation decay from
        # their separable 1D factors (built in float64, as Grid3D does).
        g = self.grid
        Nz, Ny, Nx = self._kshape
        Xh = Nx // 2 + 1
        mz, my, mx = (_build_filter_mask(n, g.dealias) for n in (Nz, Ny, Nx))
        mx = mx[:Xh]
        if self.dissipation:
            _, p, rate, onset = _spectral_dissipation_settings(g.filter_params)
            lz, ly, lx = (
                _build_exponential_filter_rate(n, p, rate, g.dealias, onset) for n in (Nz, Ny, Nx)
            )
            lx = lx[:Xh]
        else:
            lz, ly, lx = np.zeros(Nz), np.zeros(Ny), np.zeros(Xh)
        if len(self.shape) <= 2:  # no z axis: unit mask, zero decay rate
            mz, lz = np.ones(1), np.zeros(1)
        if len(self.shape) == 1:  # no y axis
            my, ly = np.ones(1), np.zeros(1)
        rtol = 1e-5 if self.rdtype == torch.float32 else 1e-12
        lam_scale = max(float(np.max(np.abs(lam))), 1.0)
        prod = (mz[:, None, None] * my[None, :, None] * mx[None, None, :]).reshape(mask.shape)
        total = (lz[:, None, None] + ly[None, :, None] + lx[None, None, :]).reshape(lam.shape)
        separable = np.allclose(prod, mask, rtol=rtol, atol=0.0) and np.allclose(
            total, lam, rtol=rtol, atol=rtol * lam_scale
        )
        if not separable:
            warnings.warn("3D Euler: spectral operators are not separable; Triton RK stages disabled.")
            return
        to_r = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=self.rdtype, device=self.device)
        self._triton = st
        self._waves = tuple(to_r(a) for a in (kx, ky, kz, mx, my, mz))
        self._mask1d = tuple(to_r(a) for a in (mx, my, mz))
        self._lam1d = tuple(to_r(a) for a in (lx, ly, lz))
        self._modes = st.RetainedModes(self._kshape, bool(g.dealias))
        self._modes_all = st.RetainedModes(self._kshape, False)
        self._inv_n = 1.0 / float(np.prod(self.shape))
        self._bufs = None

    def _k(self, X):
        """Spectral array in the kernels' (C, Nz, Ny, Xh) layout (a view)."""
        nd = len(self.shape)
        return X if nd == 3 else (X[:, None] if nd == 2 else X[:, None, None])

    def _launch_stage(self, Fh, Sin, Sout, U0, acc, w, ia, inx, modes, first, final, high, factors):
        k = self._k
        self._triton.stage(k(Fh), k(Sin), k(Sout), k(U0), acc if acc.dim() == 2 else k(acc), w, ia, inx,
                           self._waves, modes, first=first, final=final, high=k(high),
                           final_factors=factors)

    def _launch_rhs(self, Fh, S, K, w, modes):
        self._triton.rhs(self._k(Fh), self._k(S), K, w, self._waves, modes)

    def _combine(self, K, nk, base, out, w, w0, factors, modes, masked):
        self._triton.combine(K, nk, None if base is None else self._k(base), self._k(out), w, w0,
                             factors, modes, masked=masked)

    def _full(self, factors):
        fx, fy, fz = factors
        if len(self.shape) == 3:
            return fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
        if len(self.shape) == 2:
            return fy[:, None] * fx[None, :]
        return fx * fy[0] * fz[0]

    def _guard(self, fn, *args):
        """Run a compiled kernel; on toolchain failure fall back to eager for good."""
        try:
            return fn(*args)
        except Exception as exc:  # pragma: no cover - depends on toolchain
            if isinstance(exc, torch.OutOfMemoryError):
                raise
            if self._compiled:
                warnings.warn(f"torch.compile failed for 3D Euler kernels ({exc}); using eager.")
            self._compiled = False
            return None

    def disable_triton(self, exc: Exception) -> None:
        """Fall back to the torch kernels for good after a Triton failure."""
        if isinstance(exc, torch.OutOfMemoryError):
            raise exc  # a resource problem, not a toolchain one
        warnings.warn(f"Triton 3D Euler kernels failed ({exc}); using torch kernels.")
        self._triton = None
        self._cache = None
        self._pending = None
        self._kbuf = None

    # --- building blocks --------------------------------------------------

    FLUX_KIND = "euler3d"

    def _triton_fluxes(self, U, count):
        st = self._triton
        if st is None or not U.is_cuda:
            return None
        try:
            return st.fluxes(self.FLUX_KIND, U, self.gamma, self.rho_floor, self.p_floor, count=count)
        except Exception as exc:  # pragma: no cover - depends on toolchain
            self.disable_triton(exc)
            return None

    def fluxes(self, U: Array) -> Array:
        out = self._triton_fluxes(U, False) if self._torch else None
        if out is not None:
            return out
        if self._compiled:
            out = self._guard(self._c_fluxes, U, self.gamma)
            if out is not None:
                return out
        return self._fn["fluxes"](U, self.gamma)

    def fluxes_and_count(self, U: Array):
        out = self._triton_fluxes(U, True) if self._torch else None
        if out is not None:
            return out
        args = (U, self.gamma, self.rho_floor, self.p_floor)
        if self._compiled:
            out = self._guard(self._c_fluxes_count, *args)
            if out is not None:
                return out
        return self._fn["fluxes_count"](*args)

    def count_below(self, U: Array):
        args = (U, self.gamma, self.rho_floor, self.p_floor)
        if self._compiled:
            out = self._guard(self._c_count, *args)
            if out is not None:
                return out
        return self._fn["count"](*args)

    def finite_and_speed(self, U: Array) -> Tuple[bool, float]:
        """(state all finite, max wave speed) with a single host sync (Torch only)."""
        out = None
        if self._compiled:
            out = self._guard(self._c_finite_speed, U, self.gamma)
        if out is None:
            out = self._fn["speed"](U, self.gamma)
        finite, speed = torch.stack([out[0].to(out[1].dtype), out[1]]).tolist()
        return bool(finite), float(speed)

    def rhs_hat(self, Fh: Array, Mh: Array) -> Array:
        """Spectral RHS from flux spectra (9) and momentum spectra (3)."""
        if self._compiled and self._c_rhs is not None:
            out = self._guard(
                self._c_rhs, self.kx, self.ky, self.kz, self.M,
                torch.view_as_real(Fh), torch.view_as_real(Mh),
            )
            if out is not None:
                return torch.view_as_complex(out)
        return self._rhs_hat_generic(Fh, Mh)

    def _rhs_hat_generic(self, Fh, Mh):
        return _rhs_hat_complex(self.ikx, self.iky, self.ikz, self.M, Fh, Mh)

    def rhs(self, U: Array) -> Array:
        """Physical-space RHS -div(F(U)) with 2/3-masked derivatives (as ``_compute_rhs_3d``)."""
        Fh = self.rfftn(self.fluxes(U))
        Mh = self.rfftn(U[self.MOM])
        return self.irfftn(self.rhs_hat(Fh, Mh))

    # --- split-step dissipation -------------------------------------------

    def _is_cached(self, U) -> bool:
        c = self._cache
        return c is not None and c[0] is U and c[1] == U._version

    def dissipate(self, U: Array, h: float, project: bool = False) -> Array:
        """``grid.apply_spectral_dissipation(U, h, project=project)``.

        Like the grid method, returns ``U`` itself when there is nothing to do.
        """
        if not self.dissipation and (not project or self._is_cached(U)):
            return U  # a cached state is already band-limited
        if self._triton is not None:
            return self._dissipate_triton(U, h, project)
        Uh = self.rfftn(U)
        if self.dissipation:
            Uh = Uh * self._decay(h)
        if project:
            Uh = Uh * self.M
        return self.irfftn(Uh)

    def _factors(self, h: float, project: bool):
        """Per-axis 1D factors of [mask *] exp(-h * lambda) for the Triton kernels."""
        if self.dissipation:
            f = tuple(torch.exp(-float(h) * lam) for lam in self._lam1d)
        else:
            f = tuple(torch.ones_like(m) for m in self._mask1d)
        if project:
            f = tuple(a * m for a, m in zip(f, self._mask1d))
        return f

    def _dissipate_triton(self, U, h, project):
        uhat, _, _ = self._triton_buffers(U)
        self._pending = None
        factors = self._factors(h, project)
        if self._is_cached(U):
            # Band-limited state: scale the cached spectrum in place.
            w = torch.tensor([self._inv_n], dtype=U.dtype, device=U.device)
            self._combine(None, 0, uhat, uhat, w, 0, factors, self._modes, masked=True)
        else:
            scale = self._full(factors)
            Uh = torch.fft.rfftn(U, dim=self._dims, norm="forward")
            if not project:
                # The result may carry modes above the cutoff; it cannot be cached.
                self._cache = None
                return self._irfftn_fwd(Uh * scale)
            torch.mul(Uh, scale, out=uhat)
        Un = self._irfftn_fwd(uhat)
        self._cache = (Un, Un._version)
        return Un

    # --- fixed RK step ----------------------------------------------------

    def rk_step(self, U: Array, dt: float, scheme: str, h: float) -> Optional[Array]:
        """RK2/RK4 step of the split state ``U``, then dissipation(h) and projection.

        Equivalent to ``_rk{2,4}_step_3d`` followed by
        ``grid.apply_accepted_state_stabilization(., h)``.  Returns ``None``
        when the positivity clamp would change a stage or the final state.
        """
        table = _RK_TABLES[scheme]
        if self._triton is not None:
            try:
                return self._rk_step_triton(U, dt, table, h)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self.disable_triton(exc)
        return self._rk_step_generic(U, dt, table, h)

    def _rk_step_generic(self, U, dt, table, h) -> Optional[Array]:
        U0h = self.rfftn(U)
        acc = 0.0 * U0h
        Sh, Us = U0h, U
        violations = 0
        for i, (w_acc, w_next) in enumerate(table):
            if i == 0:
                F = self.fluxes(Us)
            else:
                F, c = self.fluxes_and_count(Us)
                violations = violations + c
            k = self.rhs_hat(self.rfftn(F), Sh[1:4])
            if w_next is None:
                Uh_high = U0h + acc + (w_acc * dt) * k
                violations = violations + self.count_below(self.irfftn(Uh_high))
                if int(violations) > 0:
                    return None
                scale = self.M * self._decay(h) if self.dissipation else self.M
                return self.irfftn(Uh_high * scale)
            acc = acc + (w_acc * dt) * k
            Sh = U0h + (w_next * dt) * k
            Us = self.irfftn(Sh)
        raise RuntimeError("RK table must end with a final stage")  # pragma: no cover

    def _triton_buffers(self, U):
        cdt = torch.complex64 if U.dtype == torch.float32 else torch.complex128
        if self._bufs is None or self._bufs[0].dtype != cdt:
            spec = (self.NVAR,) + self.shape[:-1] + (self.shape[-1] // 2 + 1,)
            self._uhat = torch.zeros(spec, dtype=cdt, device=U.device)
            self._shat = torch.zeros(spec, dtype=cdt, device=U.device)
            acc = torch.empty((self.NVAR, self._modes.total), dtype=cdt, device=U.device)
            self._bufs = (self._uhat, self._shat, acc)
            self._cache = None
            self._kbuf = None
        return self._bufs

    def _rk_step_triton(self, U, dt, table, h) -> Optional[Array]:
        """One fused Triton kernel per stage.

        When ``U`` is a band-limited output of this object (its spectrum is
        cached in ``self._uhat``), only the retained modes are visited.  Any
        other input is stepped over all modes, exactly as the reference path,
        so content above the dealias cutoff still enters the stage states.
        """
        uhat, shat, acc = self._triton_buffers(U)
        self._pending = None
        retained = self._is_cached(U)
        if retained:
            U0h, modes = uhat, self._modes
        else:
            U0h, modes = torch.fft.rfftn(U, dim=self._dims, norm="forward"), self._modes_all
            acc = uhat  # free until the final stage writes it (in place)
        self._cache = None
        # [1/N, (w_acc*dt, w_next*dt) per stage] as one device array (no host sync).
        weights = [self._inv_n] + [c * dt for wa, wn in table for c in (wa, 0.0 if wn is None else wn)]
        w = torch.tensor(weights, dtype=U.dtype, device=U.device)
        final_factors = self._factors(h, project=True)
        violations = torch.zeros((), dtype=torch.int64, device=U.device)
        Us, Sin = U, U0h
        last = len(table) - 1
        for i in range(len(table)):
            if i == 0:
                F = self.fluxes(Us)
            else:
                F, c = self.fluxes_and_count(Us)
                violations += c
            Fh = torch.fft.rfftn(F, dim=self._dims)
            del F
            final = i == last
            self._launch_stage(Fh, Sin, uhat if final else shat, U0h, acc, w, 1 + 2 * i, 2 + 2 * i,
                               modes, i == 0, final, shat, final_factors)
            del Fh  # release before the next transforms allocate (peak memory)
            Us = None
            if final:
                break
            Sin = shat
            Us = self._irfftn_fwd(shat)
        # The clamp acts on the state before dissipation and projection (in shat).
        if self.dissipation or modes is not self._modes:
            U_high = self._irfftn_fwd(shat)
            violations += self.count_below(U_high)
            del U_high
            Un = self._irfftn_fwd(uhat)
        else:
            Un = self._irfftn_fwd(uhat)  # identical to the pre-projection state here
            violations += self.count_below(Un)
        if modes is not self._modes:
            shat.zero_()  # retained-mode steps rely on zeros above the cutoff
        if int(violations.item()) > 0:
            return None
        self._cache = (Un, Un._version)
        return Un

    # --- embedded (adaptive) RK on the retained modes --------------------

    def embedded_ready(self, U: Array) -> bool:
        """True if ``U`` is the cached band-limited state, so ``embedded_step`` applies."""
        return self._triton is not None and self._is_cached(U)

    def embedded_step(self, U: Array, dt: float, a, b_high, b_low, h: float):
        """One embedded RK attempt in spectral space for a band-limited ``U``.

        Returns ``(U_new, error)``: the higher-order solution after
        dissipation(h) and projection (the reference accepted state) and
        max|U_high - U_low|.  Returns ``None`` when the stage limiter would act.
        The cached spectrum is left untouched; call ``commit`` with ``U_new``
        if the step is accepted.
        """
        if not self.embedded_ready(U):
            raise ValueError("embedded_step needs the cached band-limited state (see embedded_ready)")
        st = self._triton
        uhat, shat, _ = self._triton_buffers(U)
        s = len(b_high)
        if self._kbuf is None or self._kbuf.shape[0] < s or self._kbuf.dtype != uhat.dtype:
            self._kbuf = torch.empty((s, self.NVAR, self._modes.total), dtype=uhat.dtype, device=U.device)
        K = self._kbuf
        b_high = np.asarray(b_high, dtype=float)
        coefs = [self._inv_n]
        offsets = [0]
        for i in range(1, s):
            offsets.append(len(coefs))
            coefs += [dt * float(a[i][j]) for j in range(i)]
        off_err = len(coefs)
        coefs += list(dt * (b_high - np.asarray(b_low, dtype=float)))
        off_new = len(coefs)
        coefs += list(dt * b_high)
        w = torch.tensor(coefs, dtype=U.dtype, device=U.device)
        modes, waves = self._modes, self._waves
        violations = torch.zeros((), dtype=torch.int64, device=U.device)

        Us, S = U, uhat
        for i in range(s):
            if i == 0:
                F = self.fluxes(Us)
            else:
                self._combine(K, i, uhat, shat, w, offsets[i], None, modes, masked=False)
                S = shat
                Us = self._irfftn_fwd(shat)
                F, c = self.fluxes_and_count(Us)
                violations += c
            Fh = torch.fft.rfftn(F, dim=self._dims)
            del F
            self._launch_rhs(Fh, S, K[i], w, modes)
            del Fh
            Us = None
        self._combine(K, s, None, shat, w, off_err, None, modes, masked=False)
        E = self._irfftn_fwd(shat)
        self._combine(K, s, uhat, shat, w, off_new, None, modes, masked=False)
        U_high = self._irfftn_fwd(shat)
        args = (U_high, E, self.gamma, self.rho_floor, self.p_floor)
        out = self._guard(self._c_embedded_checks, *args) if self._compiled else None
        c, err = out if out is not None else self._fn["checks"](*args)
        violations += c
        del E
        nviol, error = torch.stack([violations.to(err.dtype), err]).tolist()
        if nviol > 0:
            return None
        if self.dissipation:
            del U_high
            self._combine(None, 0, shat, shat, w, 0, self._factors(h, project=True), modes, masked=True)
            Un = self._irfftn_fwd(shat)
        else:
            Un = U_high  # band-limited, so the projection is the identity
        self._pending = (Un, Un._version)
        return Un, float(error)

    def commit(self, Un: Array) -> None:
        """Adopt an accepted ``embedded_step`` result as the cached state."""
        p = self._pending
        if p is None or p[0] is not Un or p[1] != Un._version:
            return
        # The candidate spectrum lives in shat; swap roles so it becomes uhat.
        # The old uhat (zero above the cutoff) serves as the next scratch buffer.
        uhat, shat, acc = self._bufs
        self._uhat, self._shat = shat, uhat
        self._bufs = (shat, uhat, acc)
        self._cache = (Un, Un._version)
        self._pending = None


# --- 2D Euler -------------------------------------------------------------------


def _primitive2d(U, gamma):
    rho, mx, my, E = U[0], U[1], U[2], U[3]
    ux = mx / rho
    uy = my / rho
    # Same expression order as EulerEquations2D.primitive.
    p = (gamma - 1.0) * (E - 0.5 * rho * (ux * ux + uy * uy))
    return rho, mx, my, E, ux, uy, p


def euler2d_symmetric_fluxes(U: Array, gamma: float) -> Array:
    """The 5 independent 2D Euler flux fields [Txx, Tyy, Txy, Hx, Hy]."""
    rho, mx, my, E, ux, uy, p = _primitive2d(U, gamma)
    H = E + p
    return _stack([mx * ux + p, my * uy + p, mx * uy, H * ux, H * uy])


def _count_below2d(U, gamma, rho_min, p_min):
    rho, _, _, _, _, _, p = _primitive2d(U, gamma)
    if _is_torch(rho):
        return (rho < rho_min).sum() + (p < p_min).sum()
    return int(np.count_nonzero(rho < rho_min) + np.count_nonzero(p < p_min))


def _fluxes_and_count2d(U, gamma, rho_min, p_min):
    return euler2d_symmetric_fluxes(U, gamma), _count_below2d(U, gamma, rho_min, p_min)


def _embedded_checks2d(U_high, E, gamma, rho_min, p_min):
    count = _count_below2d(U_high, gamma, rho_min, p_min) + _count_below2d(
        U_high - E, gamma, rho_min, p_min
    )
    return count, torch.max(torch.abs(E))


def _finite_and_speed2d(U, gamma):
    rho, _, _, _, ux, uy, p = _primitive2d(U, gamma)
    a = torch.sqrt(gamma * p / rho)
    speed = torch.maximum(torch.abs(ux), torch.abs(uy)) + a
    return torch.isfinite(U).all(), torch.max(speed)


class Euler2DSpectral(Euler3DSpectral):
    """``Euler3DSpectral`` for periodic ``Grid2D`` + ``EulerEquations2D``.

    Each stage transforms 5 flux fields [Txx, Tyy, Txy, Hx, Hy] instead of the
    reference's 8 per direction; the mass flux is the stage momentum.
    """

    NVAR = 4
    MOM = slice(1, 3)
    FLUX_KIND = "euler2d"

    @staticmethod
    def _grid_shape(grid):
        return (grid.Ny, grid.Nx)

    def _pointwise(self):
        return dict(fluxes=euler2d_symmetric_fluxes, fluxes_count=_fluxes_and_count2d,
                    count=_count_below2d, checks=_embedded_checks2d, speed=_finite_and_speed2d,
                    rhs_real=None)

    def _rhs_hat_generic(self, Fh, Mh):
        ikx, iky, M = self.ikx, self.iky, self.M
        d = lambda ax, ay: -M * (ikx * ax + iky * ay)
        return _stack([d(Mh[0], Mh[1]), d(Fh[0], Fh[2]), d(Fh[2], Fh[1]), d(Fh[3], Fh[4])])

    def _launch_stage(self, Fh, Sin, Sout, U0, acc, w, ia, inx, modes, first, final, high, factors):
        k = self._k
        self._triton.euler2d_stage(k(Fh), k(Sin), k(Sout), k(U0), acc if acc.dim() == 2 else k(acc),
                                   w, ia, inx, self._waves, modes, first=first, final=final,
                                   high=k(high), final_factors=factors)

    def _launch_rhs(self, Fh, S, K, w, modes):
        self._triton.euler2d_rhs(self._k(Fh), self._k(S), K, w, self._waves, modes)


# --- 1D Euler -------------------------------------------------------------------


def _primitive1d(U, gamma):
    rho, m, E = U[0], U[1], U[2]
    u = m / rho
    # Same expression order as EulerEquations1D.primitive.
    p = (gamma - 1.0) * (E - 0.5 * rho * u * u)
    return rho, m, E, u, p


def euler1d_fluxes(U: Array, gamma: float) -> Array:
    """The 2 independent 1D Euler flux fields [m u + p, (E + p) u]."""
    rho, m, E, u, p = _primitive1d(U, gamma)
    return _stack([m * u + p, (E + p) * u])


def _count_below1d(U, gamma, rho_min, p_min):
    rho, _, _, _, p = _primitive1d(U, gamma)
    if _is_torch(rho):
        return (rho < rho_min).sum() + (p < p_min).sum()
    return int(np.count_nonzero(rho < rho_min) + np.count_nonzero(p < p_min))


def _fluxes_and_count1d(U, gamma, rho_min, p_min):
    return euler1d_fluxes(U, gamma), _count_below1d(U, gamma, rho_min, p_min)


def _embedded_checks1d(U_high, E, gamma, rho_min, p_min):
    count = _count_below1d(U_high, gamma, rho_min, p_min) + _count_below1d(
        U_high - E, gamma, rho_min, p_min
    )
    return count, torch.max(torch.abs(E))


def _finite_and_speed1d(U, gamma):
    rho, _, _, u, p = _primitive1d(U, gamma)
    return torch.isfinite(U).all(), torch.max(torch.abs(u) + torch.sqrt(gamma * p / rho))


class Euler1DSpectral(Euler3DSpectral):
    """``Euler3DSpectral`` for periodic Fourier ``Grid1D`` + ``EulerEquations1D``.

    Two flux transforms per stage; on small grids the gain is fewer kernel
    launches and host synchronizations rather than bandwidth.
    """

    NVAR = 3
    MOM = slice(1, 2)
    FLUX_KIND = "euler1d"

    @staticmethod
    def _grid_shape(grid):
        return (grid.N,)

    def _pointwise(self):
        return dict(fluxes=euler1d_fluxes, fluxes_count=_fluxes_and_count1d,
                    count=_count_below1d, checks=_embedded_checks1d, speed=_finite_and_speed1d,
                    rhs_real=None)

    def _rhs_hat_generic(self, Fh, Mh):
        d = lambda a: -self.M * (self.ikx * a)
        return _stack([d(Mh[0]), d(Fh[0]), d(Fh[1])])

    def _launch_stage(self, Fh, Sin, Sout, U0, acc, w, ia, inx, modes, first, final, high, factors):
        k = self._k
        self._triton.euler1d_stage(k(Fh), k(Sin), k(Sout), k(U0), acc if acc.dim() == 2 else k(acc),
                                   w, ia, inx, self._waves, modes, first=first, final=final,
                                   high=k(high), final_factors=factors)

    def _launch_rhs(self, Fh, S, K, w, modes):
        self._triton.euler1d_rhs(self._k(Fh), self._k(S), K, w, self._waves, modes)


def _ref_tensor(grid):
    """A real tensor of the grid's Torch device and dtype."""
    for name in ("kx", "k", "x"):
        ref = getattr(grid, name, None)
        if _is_torch(ref):
            return ref
    raise ValueError("Torch grid has no wavenumber or coordinate tensor")


def _to_numpy(a: Any) -> np.ndarray:
    if _is_torch(a):
        return a.detach().cpu().numpy()
    return np.asarray(a)
