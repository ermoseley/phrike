"""Bandwidth-lean spectral RK stepping for 3D ideal/resistive MHD (Torch grids).

The same restructuring as ``spectral3d`` for the 8-component MHD state:
real-to-complex FFTs, divergence and RK stages in spectral space, and 12
transformed flux fields per stage instead of 24.  Those are the symmetric
Maxwell-stress momentum flux (6), the energy flux (3) and the antisymmetric
induction tensor (3); the mass flux is the stage momentum.  Resistive and
viscous Laplacians enter the spectral RHS directly.

The reference step (``SpectralSolverMHD._fixed_step`` without Hall or
ambipolar terms) projects B after every substage and clamps every substage.
The fast path relies on two facts:

* the spectral RHS of B is ``i k_j A_ji`` with A antisymmetric, so it is
  solenoidal, and a stage state built from a solenoidal band-limited state is
  already projected (to round-off); the Leray projection is applied once, in
  the final stage, so div B cannot drift;
* the clamps leave a positive state unchanged (the stage clamp's rebuild only
  to round-off), so floor violations are counted on device and the step is
  repeated on the reference path when one would act.

Only the state produced by the previous fast step (cached spectrum, hence
solenoidal and band-limited) is stepped here; any other state takes one
reference step and its output seeds the cache.
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch may not be installed
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

from .grid import _build_exponential_filter_rate, _build_filter_mask, _spectral_dissipation_settings
from .spectral3d import _RK_TABLES, _compile_default, _ref_tensor, _to_numpy


# --- pointwise kernels (compiled with torch.compile on CUDA/MPS) ---------------


def _mhd_primitive(U, gamma):
    rho = U[0]
    ux = U[1] / rho
    uy = U[2] / rho
    uz = U[3] / rho
    E = U[4]
    Bx, By, Bz = U[5], U[6], U[7]
    # Same expression order as MHDEquations3D.primitive.
    kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
    b2h = 0.5 * (Bx * Bx + By * By + Bz * Bz)
    p = (gamma - 1.0) * (E - kinetic - b2h)
    return rho, ux, uy, uz, E, Bx, By, Bz, p, b2h


def mhd3d_fluxes(U, gamma):
    """The 12 independent MHD flux fields.

    Order: [Txx, Tyy, Tzz, Txy, Txz, Tyz, Hx, Hy, Hz, Axy, Axz, Ayz] with
    T_ij = rho u_i u_j + p* delta_ij - B_i B_j, H_i = (E + p*) u_i - B_i (u.B),
    A_ij = u_i B_j - u_j B_i and p* = p + |B|^2/2.
    """
    rho, ux, uy, uz, E, Bx, By, Bz, p, b2h = _mhd_primitive(U, gamma)
    mx, my, mz = U[1], U[2], U[3]
    pstar = p + b2h
    uB = ux * Bx + uy * By + uz * Bz
    Ep = E + pstar
    return torch.stack([
        mx * ux + pstar - Bx * Bx, my * uy + pstar - By * By, mz * uz + pstar - Bz * Bz,
        mx * uy - Bx * By, mx * uz - Bx * Bz, my * uz - By * Bz,
        Ep * ux - Bx * uB, Ep * uy - By * uB, Ep * uz - Bz * uB,
        ux * By - uy * Bx, ux * Bz - uz * Bx, uy * Bz - uz * By,
    ])


def _mhd_count_below(U, gamma, rho_min, p_min):
    rho, _, _, _, _, _, _, _, p, _ = _mhd_primitive(U, gamma)
    return (rho < rho_min).sum() + (p < p_min).sum()


def _mhd_fluxes_and_count(U, gamma, rho_min, p_min):
    return mhd3d_fluxes(U, gamma), _mhd_count_below(U, gamma, rho_min, p_min)


def _mhd_embedded_checks(U_high, E, gamma, rho_min, p_min):
    """Floor violations of U_high (the accepted-state clamp input) and max|E|."""
    return _mhd_count_below(U_high, gamma, rho_min, p_min), torch.max(torch.abs(E))


def _mhd_finite_and_speed(U, gamma):
    """(all finite, MHDEquations3D.max_wave_speed) in one pass."""
    rho, ux, uy, uz, _, Bx, By, Bz, p, _ = _mhd_primitive(U, gamma)
    a2 = gamma * p / rho
    b2 = (Bx * Bx + By * By + Bz * Bz) / rho
    fast = torch.sqrt(torch.clamp(a2 + b2, min=0.0))
    comp = torch.maximum(torch.maximum(torch.abs(ux), torch.abs(uy)), torch.abs(uz)) + fast
    return torch.isfinite(U).all(), torch.max(comp)


class MHD3DSpectral:
    """Spectral RK stepping for ``Grid3D`` + ``MHDEquations3D`` on Torch grids.

    Spectral arrays use the ``rfftn`` layout divided by Nx*Ny*Nz
    (``norm="forward"``); ``self._uhat`` holds the spectrum of the cached state.
    ``MHD2DSpectral`` runs 2.5D MHD through the same kernels with a single z mode.
    """

    @staticmethod
    def _grid_shape(grid):
        return (grid.Nz, grid.Ny, grid.Nx)

    _speed_fn = staticmethod(lambda U, gamma: _mhd_finite_and_speed(U, gamma))

    def __init__(self, grid, equations, rho_min: float, p_min: float, eta: float, nu: float,
                 compile: Optional[bool] = None):
        self.grid = grid
        self.gamma = float(equations.gamma)
        self.rho_min, self.p_min = float(rho_min), float(p_min)
        self.eta, self.nu = float(eta), float(nu)
        self.shape = self._grid_shape(grid)
        self._dims = tuple(range(-len(self.shape), 0))
        self._kshape = (1,) * (3 - len(self.shape)) + self.shape
        Nz, Ny, Nx = self._kshape
        Xh = Nx // 2 + 1
        ref = _ref_tensor(grid)
        dev, rdt = ref.device, ref.dtype
        self.device, self.rdtype = str(dev), rdt
        self.cdtype = torch.complex64 if rdt == torch.float32 else torch.complex128
        to_r = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=rdt, device=dev)

        nd = len(self.shape)
        kx = 2.0 * np.pi * np.fft.rfftfreq(Nx, d=grid.dx)
        ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=grid.dy) if nd >= 2 else np.zeros(1)
        kz = 2.0 * np.pi * np.fft.fftfreq(Nz, d=grid.dz) if nd == 3 else np.zeros(1)
        # With the 2/3 rule (required here) Nyquist modes are never retained.
        mz, my, mx = (_build_filter_mask(n, True) for n in (Nz, Ny, Nx))
        mx = mx[:Xh]
        self.dissipation = bool(getattr(grid, "spectral_dissipation_enabled", False))
        if self.dissipation:
            _, p, rate, onset = _spectral_dissipation_settings(grid.filter_params)
            lz, ly, lx = (_build_exponential_filter_rate(n, p, rate, True, onset) for n in (Nz, Ny, Nx))
            lx = lx[:Xh]
        else:
            lz, ly, lx = np.zeros(Nz), np.zeros(Ny), np.zeros(Xh)
        if nd <= 2:  # no z axis
            mz, lz = np.ones(1), np.zeros(1)
        if nd == 1:  # no y axis
            my, ly = np.ones(1), np.zeros(1)
        mask = np.asarray(_to_numpy(grid.dealias_mask))[..., :Xh]
        lam = np.asarray(_to_numpy(grid.filter_rate))[..., :Xh]
        rtol = 1e-5 if rdt == torch.float32 else 1e-12
        prod = (mz[:, None, None] * my[None, :, None] * mx[None, None, :]).reshape(mask.shape)
        total = (lz[:, None, None] + ly[None, :, None] + lx[None, None, :]).reshape(lam.shape)
        if not (np.allclose(prod, mask, rtol=rtol, atol=0.0)
                and np.allclose(total, lam, rtol=rtol, atol=rtol * max(float(np.max(np.abs(lam))), 1.0))):
            raise ValueError("MHD fast path needs the separable dealias mask and dissipation of the grid")
        bshape = {3: ((1, 1, Xh), (1, Ny, 1), (Nz, 1, 1)), 2: ((1, Xh), (Ny, 1), (1, 1)),
                  1: ((Xh,), (1,), (1,))}[nd]
        self.kx3, self.ky3, self.kz3 = (to_r(a.reshape(sh)) for a, sh in zip((kx, ky, kz), bshape))
        self.M3 = to_r(mask)
        self._mask1d = tuple(to_r(a) for a in (mx, my, mz))
        self._lam1d = tuple(to_r(a) for a in (lx, ly, lz))
        self._waves = tuple(to_r(a) for a in (kx, ky, kz, mx, my, mz))
        self._inv_n = 1.0 / float(np.prod(self.shape))
        self._cache = None
        self._pending = None
        self._uhat = self._shat = None
        self._acc = self._kbuf = None

        if compile is None:
            compile = _compile_default(self.device)
        self._compiled = False
        self._triton = None
        if compile:
            try:
                opts = dict(fullgraph=True, dynamic=False)
                self._c_fluxes = torch.compile(mhd3d_fluxes, **opts)
                self._c_fluxes_count = torch.compile(_mhd_fluxes_and_count, **opts)
                self._c_count = torch.compile(_mhd_count_below, **opts)
                self._c_checks = torch.compile(_mhd_embedded_checks, **opts)
                self._c_speed = torch.compile(self._speed_fn, **opts)
                self._compiled = True
            except Exception as exc:  # pragma: no cover - depends on toolchain
                warnings.warn(f"torch.compile unavailable for 3D MHD kernels ({exc}); using eager.")
            if self.device.startswith("cuda"):
                try:
                    from . import spectral3d_triton as st

                    self._triton = st
                    self._modes = st.RetainedModes(self._kshape, True)
                except Exception as exc:  # pragma: no cover - depends on toolchain
                    warnings.warn(f"Triton unavailable for 3D MHD stages ({exc}); using torch kernels.")

    # --- helpers ------------------------------------------------------------

    def _call(self, name, fallback, *args):
        if self._compiled:
            try:
                return getattr(self, name)(*args)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                if isinstance(exc, torch.OutOfMemoryError):
                    raise
                warnings.warn(f"torch.compile failed for 3D MHD kernels ({exc}); using eager.")
                self._compiled = False
        return fallback(*args)

    def _triton_fluxes(self, U, count):
        if self._triton is None or not U.is_cuda:
            return None
        try:
            return self._triton.fluxes("mhd3d", U, self.gamma, self.rho_min, self.p_min, count=count)
        except Exception as exc:  # pragma: no cover - depends on toolchain
            self._disable_triton(exc)
            return None

    def fluxes(self, U):
        out = self._triton_fluxes(U, False)
        if out is not None:
            return out
        return self._call("_c_fluxes", mhd3d_fluxes, U, self.gamma)

    def fluxes_and_count(self, U):
        out = self._triton_fluxes(U, True)
        if out is not None:
            return out
        return self._call("_c_fluxes_count", _mhd_fluxes_and_count, U, self.gamma, self.rho_min, self.p_min)

    def count_below(self, U):
        return self._call("_c_count", _mhd_count_below, U, self.gamma, self.rho_min, self.p_min)

    def finite_and_speed(self, U) -> Tuple[bool, float]:
        finite, speed = self._call("_c_speed", self._speed_fn, U, self.gamma)
        finite, speed = torch.stack([finite.to(speed.dtype), speed]).tolist()
        return bool(finite), float(speed)

    def _irfftn(self, F):
        return torch.fft.irfftn(F, s=self.shape, dim=self._dims, norm="forward")

    def _rfftn_fwd(self, U):
        return torch.fft.rfftn(U, dim=self._dims, norm="forward")

    def _rfftn(self, F):
        return torch.fft.rfftn(F, dim=self._dims)

    def _k(self, X):
        """Spectral array in the kernels' (C, Nz, Ny, Xh) layout (a view)."""
        nd = len(self.shape)
        return X if nd == 3 else (X[:, None] if nd == 2 else X[:, None, None])

    def _factors(self, h: float, project: bool):
        if self.dissipation:
            f = tuple(torch.exp(-float(h) * lam) for lam in self._lam1d)
        else:
            f = tuple(torch.ones_like(m) for m in self._mask1d)
        if project:
            f = tuple(a * m for a, m in zip(f, self._mask1d))
        return f

    def _full(self, factors):
        fx, fy, fz = factors
        if len(self.shape) == 3:
            return fz[:, None, None] * fy[None, :, None] * fx[None, None, :]
        if len(self.shape) == 2:
            return fy[:, None] * fx[None, :]
        return fx * fy[0] * fz[0]

    def _buffers(self):
        if self._uhat is None:
            spec = (8,) + self.shape[:-1] + (self.shape[-1] // 2 + 1,)
            self._uhat = torch.zeros(spec, dtype=self.cdtype, device=self.device)
            self._shat = torch.zeros_like(self._uhat)
        return self._uhat, self._shat

    def _is_cached(self, U) -> bool:
        c = self._cache
        return c is not None and c[0] is U and c[1] == U._version

    def ready(self, U) -> bool:
        return self._is_cached(U)

    def invalidate(self) -> None:
        self._cache = None
        self._pending = None

    def seed(self, U):
        """Cache the spectrum of a solenoidal state; returns its dealiased version."""
        uhat, _ = self._buffers()
        self._pending = None
        torch.mul(self._rfftn_fwd(U), self.M3, out=uhat)
        Un = self._irfftn(uhat)
        self._cache = (Un, Un._version)
        return Un

    def dissipate(self, U, h: float):
        """Split-step dissipation of the cached state (``None`` if not cached)."""
        if not self._is_cached(U):
            return None
        if not self.dissipation:
            return U
        uhat, _ = self._buffers()
        self._pending = None
        factors = self._factors(h, project=False)
        if self._triton is not None:  # only the retained modes are nonzero
            w = torch.tensor([self._inv_n], dtype=self.rdtype, device=uhat.device)
            self._triton.combine(None, 0, self._k(uhat), self._k(uhat), w, 0, factors, self._modes, masked=True)
        else:
            uhat.mul_(self._full(factors))
        Un = self._irfftn(uhat)
        self._cache = (Un, Un._version)
        return Un

    # --- spectral RHS (generic torch path) ---------------------------------

    def _rhs_hat(self, Fh, S):
        """8-component spectral RHS; Fh are normalized flux spectra (12), S the stage state."""
        ikx, iky, ikz = 1j * self.kx3, 1j * self.ky3, 1j * self.kz3
        M = self.M3
        k2m = (self.kx3 ** 2 + self.ky3 ** 2 + self.kz3 ** 2) * M

        def d(ax, ay, az):
            s = 0.0
            if ax is not None:
                s = s + ikx * ax
            if ay is not None:
                s = s + iky * ay
            if az is not None:
                s = s + ikz * az
            return -M * s

        T, H, A = Fh[0:6], Fh[6:9], Fh[9:12]
        k = [
            d(S[1], S[2], S[3]),
            d(T[0], T[3], T[4]) - self.nu * k2m * S[1],
            d(T[3], T[1], T[5]) - self.nu * k2m * S[2],
            d(T[4], T[5], T[2]) - self.nu * k2m * S[3],
            d(H[0], H[1], H[2]),
            d(None, -A[0], -A[1]) - self.eta * k2m * S[5],
            d(A[0], None, -A[2]) - self.eta * k2m * S[6],
            d(A[1], A[2], None) - self.eta * k2m * S[7],
        ]
        return torch.stack(k)

    def _leray(self, X):
        kx, ky, kz = self.kx3, self.ky3, self.kz3
        k2 = kx * kx + ky * ky + kz * kz
        inv = torch.where(k2 == 0, torch.zeros_like(k2), 1.0 / torch.where(k2 == 0, torch.ones_like(k2), k2))
        c = (kx * X[5] + ky * X[6] + kz * X[7]) * inv
        X = X.clone()
        X[5] = X[5] - kx * c
        X[6] = X[6] - ky * c
        X[7] = X[7] - kz * c
        return X

    # --- fixed RK step ------------------------------------------------------

    def rk_step(self, U, dt: float, scheme: str, h: float, check_stages: bool = True) -> Optional[Any]:
        """RK2/RK4 step of the cached split state ``U``, then projection, dissipation(h)
        and the dealias mask.  ``None`` when a positivity clamp would act."""
        assert self._is_cached(U)
        table = _RK_TABLES[scheme]
        if self._triton is not None:
            try:
                return self._rk_step_triton(U, dt, table, h, check_stages)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self._disable_triton(exc)
        return self._rk_step_generic(U, dt, table, h, check_stages)

    def _disable_triton(self, exc):
        if isinstance(exc, torch.OutOfMemoryError):
            raise exc
        warnings.warn(f"Triton 3D MHD kernels failed ({exc}); using torch kernels.")
        self._triton = None

    def _rk_step_generic(self, U, dt, table, h, check_stages):
        uhat, shat = self._buffers()
        U0h = uhat.clone()
        acc = torch.zeros_like(U0h)
        Sh, Us = U0h, U
        violations = torch.zeros((), dtype=torch.int64, device=U.device)
        for i, (wa, wn) in enumerate(table):
            if i > 0 and check_stages:
                F, c = self.fluxes_and_count(Us)
                violations += c
            else:
                F = self.fluxes(Us)
            k = self._rhs_hat(self._rfftn(F) * self._inv_n, Sh)
            if wn is None:
                high = U0h + acc + (wa * dt) * k
                return self._finish(high, h, violations)
            acc = acc + (wa * dt) * k
            Sh = U0h + (wn * dt) * k
            Us = self._irfftn(Sh)
        raise RuntimeError("RK table must end with a final stage")  # pragma: no cover

    def _finish(self, high, h, violations):
        """Clamp check on the pre-projection state, then project, dissipate and mask."""
        if self.dissipation:  # otherwise it equals the accepted state up to round-off
            violations = violations + self.count_below(self._irfftn(high))
        un = self._leray(high) * self._full(self._factors(h, project=True))
        Un = self._irfftn(un)
        violations = violations + self.count_below(Un)
        if int(violations.item()) > 0:
            self.invalidate()
            return None
        self._uhat.copy_(un)
        self._cache = (Un, Un._version)
        return Un

    def _rk_step_triton(self, U, dt, table, h, check_stages):
        st = self._triton
        uhat, shat = self._buffers()
        modes = self._modes
        if self._acc is None:
            self._acc = torch.empty((8, modes.total), dtype=self.cdtype, device=U.device)
        acc = self._acc
        self._pending = None
        self._cache = None
        w = torch.tensor(
            [self._inv_n, self.eta, self.nu]
            + [c * dt for wa, wn in table for c in (wa, 0.0 if wn is None else wn)],
            dtype=U.dtype, device=U.device,
        )
        fin = self._factors(h, project=True)
        violations = torch.zeros((), dtype=torch.int64, device=U.device)
        Us, Sin = U, uhat
        last = len(table) - 1
        for i in range(len(table)):
            if i > 0 and check_stages:
                F, c = self.fluxes_and_count(Us)
                violations += c
            else:
                F = self.fluxes(Us)
            Fh = self._rfftn(F)
            del F
            final = i == last
            k = self._k
            st.mhd_stage(k(Fh), k(Sin), k(uhat if final else shat), k(uhat), acc, w, 3 + 2 * i, 4 + 2 * i,
                         self._waves, modes, first=(i == 0), final=final,
                         high=k(shat), final_factors=fin)
            del Fh
            Us = None
            if final:
                break
            Sin = shat
            Us = self._irfftn(shat)
        # shat: pre-projection state (the final clamp input); uhat: accepted spectrum.
        # Without dissipation the two differ only by the round-off projection.
        if self.dissipation:
            violations += self.count_below(self._irfftn(shat))
        Un = self._irfftn(uhat)
        violations += self.count_below(Un)
        if int(violations.item()) > 0:
            self.invalidate()
            return None
        self._cache = (Un, Un._version)
        return Un

    # --- embedded RK (stages are not limited, as in the reference) ----------

    def embedded_step(self, U, dt, a, b_high, b_low, h):
        """(accepted state candidate, max|U_high - U_low|) or ``None`` when the
        accepted-state clamp would act.  Call ``commit`` if it is accepted."""
        assert self._is_cached(U)
        if self._triton is None:
            return self._embedded_generic(U, dt, a, b_high, b_low, h)
        st = self._triton
        uhat, shat = self._buffers()
        modes = self._modes
        s = len(b_high)
        if self._kbuf is None or self._kbuf.shape[0] < s:
            self._kbuf = torch.empty((s, 8, modes.total), dtype=self.cdtype, device=U.device)
        K = self._kbuf
        b_high = np.asarray(b_high, dtype=float)
        coefs = [self._inv_n, self.eta, self.nu]
        offsets = [0]
        for i in range(1, s):
            offsets.append(len(coefs))
            coefs += [dt * float(a[i][j]) for j in range(i)]
        off_err = len(coefs)
        coefs += list(dt * (b_high - np.asarray(b_low, dtype=float)))
        off_new = len(coefs)
        coefs += list(dt * b_high)
        w = torch.tensor(coefs, dtype=U.dtype, device=U.device)
        Us, S = U, uhat
        for i in range(s):
            if i > 0:
                st.combine(K, i, self._k(uhat), self._k(shat), w, offsets[i], None, modes, masked=False)
                S = shat
                Us = self._irfftn(shat)
            Fh = self._rfftn(self.fluxes(Us))
            Sk = self._k(S)
            st.mhd_stage(self._k(Fh), Sk, Sk, Sk, K[i], w, 0, 0, self._waves, modes, first=False,
                         final=False, rhs_only=True)
            del Fh
            Us = None
        st.combine(K, s, None, self._k(shat), w, off_err, None, modes, masked=False)
        E = self._irfftn(shat)
        st.combine(K, s, self._k(uhat), self._k(shat), w, off_new, None, modes, masked=False)
        U_high = self._irfftn(shat)
        c, err = self._call("_c_checks", _mhd_embedded_checks, U_high, E,
                            self.gamma, self.rho_min, self.p_min)
        del E, U_high
        st.leray_scale(self._k(shat), self._waves, self._factors(h, project=True), modes)
        Un = self._irfftn(shat)
        c = c + self.count_below(Un)
        nviol, error = torch.stack([c.to(err.dtype), err]).tolist()
        if nviol > 0:
            return None
        self._pending = (Un, Un._version)
        return Un, float(error)

    def _embedded_generic(self, U, dt, a, b_high, b_low, h):
        uhat, shat = self._buffers()
        U0h = uhat
        s = len(b_high)
        k = []
        Us, Sh = U, U0h
        for i in range(s):
            if i > 0:
                Sh = U0h + sum(dt * float(a[i][j]) * k[j] for j in range(i) if a[i][j] != 0.0)
                Us = self._irfftn(Sh)
            k.append(self._rhs_hat(self._rfftn(self.fluxes(Us)) * self._inv_n, Sh))
        Eh = sum(dt * float(b_high[i] - b_low[i]) * k[i] for i in range(s) if b_high[i] != b_low[i])
        high = U0h + sum(dt * float(b_high[i]) * k[i] for i in range(s) if b_high[i] != 0.0)
        c, err = _mhd_embedded_checks(self._irfftn(high), self._irfftn(Eh),
                                      self.gamma, self.rho_min, self.p_min)
        un = self._leray(high) * self._full(self._factors(h, project=True))
        Un = self._irfftn(un)
        c = c + self.count_below(Un)
        nviol, error = torch.stack([c.to(err.dtype), err]).tolist()
        if nviol > 0:
            return None
        shat.copy_(un)
        self._pending = (Un, Un._version)
        return Un, float(error)

    def commit(self, Un) -> None:
        p = self._pending
        if p is None or p[0] is not Un or p[1] != Un._version:
            return
        self._uhat, self._shat = self._shat, self._uhat
        self._cache = (Un, Un._version)
        self._pending = None


def _mhd2d_finite_and_speed(U, gamma):
    """(all finite, MHDEquations2D.max_wave_speed) in one pass."""
    rho, ux, uy, uz, _, Bx, By, Bz, p, _ = _mhd_primitive(U, gamma)
    a2 = gamma * p / rho
    b2 = (Bx * Bx + By * By + Bz * Bz) / rho
    fast = torch.sqrt(torch.clamp(a2 + b2, min=0.0))
    comp = torch.maximum(torch.abs(ux), torch.abs(uy)) + fast
    return torch.isfinite(U).all(), torch.max(comp)


class MHD2DSpectral(MHD3DSpectral):
    """2.5D MHD (``Grid2D`` + ``MHDEquations2D``): 3D MHD with a single z mode.

    With kz = 0 every z derivative vanishes, and the Leray projection leaves Bz
    alone, as ``Grid2D.project_solenoidal`` does.
    """

    @staticmethod
    def _grid_shape(grid):
        return (grid.Ny, grid.Nx)

    _speed_fn = staticmethod(lambda U, gamma: _mhd2d_finite_and_speed(U, gamma))


def _mhd1d_finite_and_speed(U, gamma):
    """(all finite, MHDEquations1D.max_wave_speed: |ux| + exact fast speed in x)."""
    rho, ux, _, _, _, Bx, By, Bz, p, _ = _mhd_primitive(U, gamma)
    a2 = gamma * p / rho
    b2 = (Bx * Bx + By * By + Bz * Bz) / rho
    cax2 = (Bx * Bx) / rho
    disc = torch.clamp((a2 + b2) * (a2 + b2) - 4.0 * a2 * cax2, min=0.0)
    cf = torch.sqrt(0.5 * (a2 + b2 + torch.sqrt(disc)))
    return torch.isfinite(U).all(), torch.max(torch.abs(ux) + cf)


class MHD1DSpectral(MHD3DSpectral):
    """1D MHD on a periodic Fourier ``Grid1D``: 3D MHD with single y and z modes.

    The Leray projection then removes every k != 0 mode of Bx, which is the 1D
    projection (Bx constant); By and Bz are untouched.
    """

    @staticmethod
    def _grid_shape(grid):
        return (grid.N,)

    _speed_fn = staticmethod(lambda U, gamma: _mhd1d_finite_and_speed(U, gamma))
