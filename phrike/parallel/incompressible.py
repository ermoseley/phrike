"""Slab-decomposed pseudo-spectral incompressible (visco-resistive) MHD.

Equations, with unit density and the magnetic field in Alfven units
(``b = B / sqrt(mu0 rho)``, so ``|b|`` is the Alfven speed)::

    du/dt = -div(u u - b b) - grad(p + b^2/2) + nu lap(u) + f,   div u = 0
    db/dt =  curl(u x b)                      + eta lap(b),      div b = 0

A uniform mean field is the ``k = 0`` mode of ``b``; it needs no special
treatment and may be zero.  With ``mhd=False`` the magnetic field is dropped
and the same code is incompressible Navier-Stokes.

Method
------
* The unknowns are the retained Fourier modes (2/3 rule) of ``u`` and ``b``.
  The nonlinearity is exactly quadratic, so the 2/3 rule removes aliasing
  exactly and the truncated system conserves energy and cross helicity when
  ``nu = eta = 0``.
* Per stage: 6 inverse transforms (``u``, ``b``), one pass forming 8 products
  in physical space, 8 forward transforms.  The momentum flux
  ``T = u u - b b`` is symmetric and its isotropic part is a pressure, so 5
  fields carry it (``Tzz`` is subtracted from the diagonal); the induction
  tensor ``A = u b - b u`` is antisymmetric (3 fields), which makes ``db/dt``
  solenoidal to round-off.  Pressure is eliminated by the Leray projection.
* Viscosity and resistivity are integrated exactly by an integrating factor
  (Lawson RK): there is no diffusive time-step limit for any ``nu``, ``eta``
  or their ratio.  The explicit part is RK2 (midpoint), RK3 (Heun) or RK4.
* The step is ``dt = cfl / max_x sum_i (|u_i| + |b_i|) / dx_i``, evaluated on
  the fields the first stage transforms anyway (the stage-0 products do not
  depend on ``dt``), so the CFL condition costs no extra transform.

Linear stability on the retained modes requires ``cfl <= 1.35`` for RK4 and
``0.83`` for RK3; RK2 (midpoint) has no stable interval on the imaginary axis
and relies on the physical diffusion, so keep its ``cfl`` small.
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .forcing import BandForcing
from .grid import SlabGrid3D

# (w_acc, w_next) per stage, as ``spectral3d._RK_TABLES``: stage i is evaluated
# at node c_i = w_next of the previous stage and uses only the previous slope.
LAWSON_TABLES = {
    "rk2": ((0.0, 0.5), (1.0, None)),
    "rk3": ((0.25, 1.0 / 3.0), (0.0, 2.0 / 3.0), (0.75, None)),
    "rk4": ((1.0 / 6.0, 0.5), (1.0 / 3.0, 0.5), (1.0 / 3.0, 1.0), (1.0 / 6.0, None)),
}


def lawson_coefficients(table, dt: float):
    """Per stage ``(wa*dt, wn*dt, tau_0, tau_n, tau_a)`` of the integrating-factor RK.

    ``tau_0`` multiplies the base state in the stage (or final) state,
    ``tau_n`` the slope in the next stage and ``tau_a`` the slope in the final
    update, each as ``exp(-D k^2 tau)``.
    """
    out, c = [], 0.0
    for wa, wn in table:
        if wn is None:
            out.append((wa * dt, 0.0, dt, 0.0, (1.0 - c) * dt))
        else:
            out.append((wa * dt, wn * dt, wn * dt, (wn - c) * dt, (1.0 - c) * dt))
            c = wn
    return out


def _triton_default() -> bool:
    flag = os.environ.get("PHRIKE_TRITON")
    if flag is None:
        return True
    return flag.strip().lower() not in ("0", "false", "no", "off", "")


class IncompressibleSlabStepper:
    """Distributed spectral stepping of incompressible MHD (or Navier-Stokes).

    ``uhat`` is ``(6, Kz, nky, Kx)`` (``u`` then ``b``; 3 components without a
    magnetic field): the retained spectrum divided by N, so the physical
    fields are ``grid.inverse(uhat)``.
    """

    def __init__(self, grid: SlabGrid3D, nu: float, eta: float = 0.0, mhd: bool = True,
                 scheme: str = "rk4", cfl: float = 0.5, triton: Optional[bool] = None,
                 spherical_truncation: bool = False, forcing: Optional[BandForcing] = None):
        self.grid = grid
        self.comm = grid.comm
        self.mhd = bool(mhd)
        self.ncomp = 6 if self.mhd else 3
        self.nprod = 8 if self.mhd else 5
        self.nu, self.eta = float(nu), float(eta) if self.mhd else 0.0
        if self.nu < 0.0 or self.eta < 0.0:
            raise ValueError("viscosity and resistivity must be non-negative")
        scheme = str(scheme).lower()
        if scheme not in LAWSON_TABLES:
            raise ValueError(f"incompressible scheme must be one of {sorted(LAWSON_TABLES)}, got {scheme!r}")
        self.scheme = scheme
        self.table = LAWSON_TABLES[scheme]
        self.cfl = float(cfl)
        self.forcing = forcing
        self.inv_n = grid.inv_n
        self.inv_spacing = (1.0 / grid.dx, 1.0 / grid.dy, 1.0 / grid.dz)
        d = grid.decomp
        # Optional isotropic truncation at the smallest per-axis cutoff.
        self.spherical = bool(spherical_truncation)
        if self.spherical:
            kcut = min(2.0 * np.pi * d.cx / grid.Lx, 2.0 * np.pi * d.cy / grid.Ly,
                       2.0 * np.pi * d.cz / grid.Lz)
            self.kcut2 = float(kcut * kcut) * (1.0 + 1e-12)
        else:
            self.kcut2 = float("inf")

        self.uhat = grid.zeros_spectral(self.ncomp)
        self.shat = grid.zeros_spectral(self.ncomp)
        self.acc = grid.zeros_spectral(self.ncomp)
        self.t = 0.0
        self.steps = 0
        self.rate: Optional[float] = None  # max_x sum_i (|u_i| + |b_i|) / dx_i of the last stage 0
        self.last_dt: Optional[float] = None
        self.injected = 0.0  # this rank's part of the energy added by white-noise kicks

        self._triton = None
        self._strict_kernels = False  # raise on a kernel failure instead of falling back
        if triton is None:
            triton = _triton_default()
        # "force" skips the device test (the test suite runs the kernels on an emulator).
        if triton and (grid.device.type == "cuda" or triton == "force"):
            try:
                from . import kernels_triton as kt

                self._triton = kt
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self._note(f"Triton unavailable for the incompressible kernels ({exc}); using torch ops.")

    # --- helpers ------------------------------------------------------------

    def _note(self, message: str) -> None:
        if self.comm.is_root:
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _disable_triton(self, exc: Exception) -> None:
        if isinstance(exc, torch.OutOfMemoryError) or self._strict_kernels:
            raise exc
        self._note(f"Triton incompressible kernels failed ({exc!r}); using torch ops.")
        self._triton = None

    def warm_up(self) -> bool:
        """Build the Triton kernels this stepper uses and settle the tier for all ranks.

        Every kernel variant is run on a tiny serial problem (no MPI inside),
        rank 0 first so that it fills Triton's on-disk cache for the others.
        The Triton tier is kept only if it works on every rank: ranks that
        disagreed would transform different arrays in the middle of a step.
        Returns whether it is in use.
        """
        if self._triton is None:
            return False
        comm = self.comm
        problem = [""]

        def run() -> float:
            try:
                self._warm_up_local()
                return 1.0
            except Exception as exc:  # pragma: no cover - depends on toolchain
                if isinstance(exc, torch.OutOfMemoryError):
                    raise
                problem[0] = repr(exc)
                return 0.0

        ok = run() if comm.is_root else 1.0
        comm.barrier()
        if not comm.is_root:
            ok = run()
        if comm.min(ok) < 1.0:
            if problem[0]:
                warnings.warn(f"[rank {comm.rank}] Triton kernels failed to build or run: {problem[0]}",
                              RuntimeWarning, stacklevel=2)
            self._note("Triton kernels are not usable on every rank; all ranks use torch ops.")
            self._triton = None
        return self._triton is not None

    def _warm_up_local(self) -> None:
        from .comm import Comm

        g = self.grid
        tiny = SlabGrid3D(8, 8, 8, g.Lx, g.Ly, g.Lz, comm=Comm.serial(), device=g.device,
                          precision=g.precision, fft_batch=g.fft.batch)
        st = IncompressibleSlabStepper(tiny, self.nu, self.eta, mhd=self.mhd, scheme=self.scheme,
                                       cfl=self.cfl, triton="force", spherical_truncation=self.spherical)
        if st._triton is None:
            raise RuntimeError("the Triton kernel module could not be imported")
        st._strict_kernels = True
        X, Y, Z = tiny.xyz_mesh()
        kx, ky, kz = 2.0 * np.pi / g.Lx, 2.0 * np.pi / g.Ly, 2.0 * np.pi / g.Lz
        fields = [torch.sin(ky * Y), torch.sin(kz * Z), torch.sin(kx * X)]
        if self.mhd:
            fields += [0.5 * torch.cos(kz * Z), 0.5 * torch.cos(kx * X), 0.5 * torch.cos(ky * Y)]
        st.seed(torch.stack(fields))
        st.step()
        st.step(dt=1e-3)

    def _inv_k2(self) -> torch.Tensor:
        k2 = self.grid.k2()
        return torch.where(k2 == 0, torch.zeros_like(k2), 1.0 / torch.where(k2 == 0, torch.ones_like(k2), k2))

    def _keep(self) -> Optional[torch.Tensor]:
        """Mask of the evolved modes (None for the full 2/3 box)."""
        if not self.spherical:
            return None
        return (self.grid.k2() <= self.kcut2).to(self.grid.rdtype)

    def _project_(self, X: torch.Tensor) -> None:
        """Leray-project a 3-vector spectrum in place."""
        g = self.grid
        c = (g.kx3 * X[0] + g.ky3 * X[1] + g.kz3 * X[2]) * self._inv_k2()
        X[0] -= g.kx3 * c
        X[1] -= g.ky3 * c
        X[2] -= g.kz3 * c

    def _decay(self, tau: float) -> Optional[torch.Tensor]:
        """exp(-D k^2 tau) per component, (ncomp, Kz, nky, Kx), or None for the identity."""
        if tau == 0.0 or (self.nu == 0.0 and self.eta == 0.0):
            return None
        k2 = self.grid.k2()
        e = torch.exp(-(self.nu * float(tau)) * k2).expand(3, *k2.shape)
        if not self.mhd:
            return e
        return torch.cat((e, torch.exp(-(self.eta * float(tau)) * k2).expand(3, *k2.shape)), dim=0)

    # --- state --------------------------------------------------------------

    def seed(self, V: torch.Tensor) -> None:
        """Adopt physical fields ``(ncomp, nz, Ny, Nx)``: dealias and make them solenoidal."""
        g = self.grid
        if tuple(V.shape) != (self.ncomp,) + g.decomp.physical_shape:
            raise ValueError(f"fields must be {(self.ncomp,) + g.decomp.physical_shape}, got {tuple(V.shape)}")
        self.set_spectrum(g.forward(V.to(dtype=g.rdtype, device=g.device), normalized=True))

    def set_spectrum(self, S: torch.Tensor) -> None:
        """Adopt a retained spectrum (divided by N); it is projected and truncated."""
        if S is not self.uhat:
            self.uhat.copy_(S)
        self._project_(self.uhat[0:3])
        if self.mhd:
            self._project_(self.uhat[3:6])
        keep = self._keep()
        if keep is not None:
            self.uhat.mul_(keep)
        self.rate = None

    def physical(self) -> torch.Tensor:
        """``u`` (and ``b``) on this rank's slab, ``(ncomp, nz, Ny, Nx)``."""
        return self.grid.inverse(self.uhat)

    # --- kernels (torch tier) -----------------------------------------------

    def _products_torch(self, V: torch.Tensor, stats: bool):
        ux, uy, uz = V[0], V[1], V[2]
        if self.mhd:
            bx, by, bz = V[3], V[4], V[5]
            tzz = uz * uz - bz * bz
            G = torch.stack([
                ux * ux - bx * bx - tzz, uy * uy - by * by - tzz,
                ux * uy - bx * by, ux * uz - bx * bz, uy * uz - by * bz,
                ux * by - uy * bx, ux * bz - uz * bx, uy * bz - uz * by,
            ])
        else:
            tzz = uz * uz
            G = torch.stack([ux * ux - tzz, uy * uy - tzz, ux * uy, ux * uz, uy * uz])
        if not stats:
            return G
        ix, iy, iz = self.inv_spacing
        if self.mhd:
            rate = (ux.abs() + bx.abs()) * ix + (uy.abs() + by.abs()) * iy + (uz.abs() + bz.abs()) * iz
        else:
            rate = ux.abs() * ix + uy.abs() * iy + uz.abs() * iz
        finite = torch.isfinite(rate)
        bad = (~finite).sum()
        return G, torch.where(finite, rate, torch.zeros_like(rate)).max(), bad

    def _products(self, V: torch.Tensor, stats: bool):
        kt = self._triton
        if kt is not None:
            try:
                return kt.imhd_products(V, self.inv_spacing, self.mhd, stats)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self._disable_triton(exc)
        return self._products_torch(V, stats)

    def nonlinear(self, Gh: torch.Tensor) -> torch.Tensor:
        """Projected nonlinear term from normalized product spectra ``Gh``."""
        g = self.grid
        kx, ky, kz = g.kx3, g.ky3, g.kz3
        N = torch.empty((self.ncomp,) + tuple(Gh.shape[1:]), dtype=Gh.dtype, device=Gh.device)
        N[0] = -1j * (kx * Gh[0] + ky * Gh[2] + kz * Gh[3])
        N[1] = -1j * (kx * Gh[2] + ky * Gh[1] + kz * Gh[4])
        N[2] = -1j * (kx * Gh[3] + ky * Gh[4])
        self._project_(N[0:3])
        if self.mhd:
            N[3] = 1j * (ky * Gh[5] + kz * Gh[6])
            N[4] = 1j * (kz * Gh[7] - kx * Gh[5])
            N[5] = -1j * (kx * Gh[6] + ky * Gh[7])
        keep = self._keep()
        if keep is not None:
            N.mul_(keep)
        return N

    def _stage_torch(self, Gh: torch.Tensor, coef, first: bool, final: bool) -> None:
        wa, wn, tau0, taun, taua = coef
        N = self.nonlinear(Gh)

        def scaled(tau, X):
            e = self._decay(tau)
            return X if e is None else e * X

        update = wa * scaled(taua, N)
        if final:
            base = scaled(tau0, self.uhat)
            self.uhat.copy_(base + update if first else base + self.acc + update)
            return
        if first:
            self.acc.copy_(update)
        else:
            self.acc.add_(update)
        self.shat.copy_(scaled(tau0, self.uhat) + wn * scaled(taun, N))

    def _stage(self, G: torch.Tensor, coef, first: bool, final: bool) -> None:
        """Transform the products and apply one RK stage."""
        g = self.grid
        Gh = g.forward(G, normalized=False)
        kt = self._triton
        if kt is not None:
            try:
                wa, wn, tau0, taun, taua = coef
                huge = 1e300 if g.rdtype == torch.float64 else 3e38
                params = torch.tensor(
                    [self.inv_n, self.nu, self.eta, min(self.kcut2, huge), wa, wn, tau0, taun, taua],
                    dtype=g.rdtype, device=g.device)
                kt.imhd_stage(Gh, self.uhat, self.uhat if final else self.shat, self.acc,
                              (g.kx, g.ky, g.kz), params, self.mhd, first, final)
                return
            except Exception as exc:  # pragma: no cover - depends on toolchain
                # A kernel variant that fails to build has written nothing.
                self._disable_triton(exc)
        Gh.mul_(self.inv_n)
        self._stage_torch(Gh, coef, first, final)

    def _add_forcing(self, values: Optional[torch.Tensor], coef, final: bool) -> None:
        """Add a frozen band acceleration to the stage, as a term of the nonlinear slope."""
        if values is None:
            return
        f = self.forcing
        wa, wn, _, taun, taua = coef
        idx, k2 = f.index, f.local_k2
        acc = self.acc.reshape(self.ncomp, -1)
        if final:
            out = self.uhat.reshape(self.ncomp, -1)
            out[:3, idx] += (wa * torch.exp(-self.nu * taua * k2)) * values
            return
        acc[:3, idx] += (wa * torch.exp(-self.nu * taua * k2)) * values
        out = self.shat.reshape(self.ncomp, -1)
        out[:3, idx] += (wn * torch.exp(-self.nu * taun * k2)) * values

    # --- time stepping ------------------------------------------------------

    def step(self, dt_max: float = float("inf"), dt: Optional[float] = None) -> float:
        """Advance one step and return the step taken.

        The step is the CFL step capped by ``dt_max`` (the time to the next
        output, say); pass ``dt`` to impose it instead.
        """
        g = self.grid
        V = g.inverse(self.uhat)
        G, rate, bad = self._products(V, True)
        del V
        values = torch.stack([rate.to(torch.float64), bad.to(torch.float64)]).tolist()
        rate, bad = self.comm.allreduce(values, "max")  # the one collective of a step
        if bad > 0:
            raise FloatingPointError(f"Non-finite solution at step {self.steps}, t={self.t}")
        rate = float(rate)
        self.rate = rate
        if dt is None:
            dt = self.cfl / rate if rate > 0.0 else float("inf")
            dt = min(dt, float(dt_max))
        dt = float(dt)
        if not (math.isfinite(dt) and dt > 0.0):
            raise RuntimeError(
                f"No finite positive time step at step {self.steps}, t={self.t}: the fields "
                "vanish and no output or end time bounds the step.")

        forcing = self.forcing
        frozen = forcing.frozen(self.uhat) if forcing is not None else None
        coefs = lawson_coefficients(self.table, dt)
        last = len(coefs) - 1
        for i, coef in enumerate(coefs):
            if i > 0:
                V = g.inverse(self.shat)
                G = self._products(V, False)
                del V
            self._stage(G, coef, i == 0, i == last)
            del G
            self._add_forcing(frozen, coef, i == last)
        if forcing is not None:
            kick = forcing.kick(dt)
            if kick is not None:
                flat = self.uhat.reshape(self.ncomp, -1)
                u = flat[:3, forcing.index]
                self.injected += float((((u.conj() * kick).real + 0.5 * (kick.conj() * kick).real).sum(dim=0)
                                        * forcing.local_weight).sum().item())
                flat[:3, forcing.index] = u + kick
            forcing.advance(dt)
        self.t += dt
        self.steps += 1
        self.last_dt = dt
        return dt

    def rhs(self, S: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Nonlinear term N(S) (without diffusion or forcing); for tests and diagnostics."""
        g = self.grid
        V = g.inverse(self.uhat if S is None else S)
        return self.nonlinear(g.forward(self._products_torch(V, False), normalized=True))

    # --- diagnostics --------------------------------------------------------

    def diagnostics(self) -> Dict[str, float]:
        """Global energies, dissipation rates and derived turbulence numbers."""
        g = self.grid
        w, k2 = g.weight3, g.k2()
        u = self.uhat[0:3]
        eu = (u.real ** 2 + u.imag ** 2).sum(dim=0)
        sums = [float((w * eu).sum().item()), float((w * k2 * eu).sum().item())]
        div = (g.kx3 * u[0] + g.ky3 * u[1] + g.kz3 * u[2]).abs().max()
        maxima = [float(div.item())]
        if self.mhd:
            b = self.uhat[3:6]
            eb = (b.real ** 2 + b.imag ** 2).sum(dim=0)
            cross = (u.real * b.real + u.imag * b.imag).sum(dim=0)
            # Magnetic helicity <a . b> with a = i k x b / k^2.
            ax = 1j * (g.ky3 * b[2] - g.kz3 * b[1])
            ay = 1j * (g.kz3 * b[0] - g.kx3 * b[2])
            az = 1j * (g.kx3 * b[1] - g.ky3 * b[0])
            hel = (ax.real * b[0].real + ax.imag * b[0].imag + ay.real * b[1].real + ay.imag * b[1].imag
                   + az.real * b[2].real + az.imag * b[2].imag) * self._inv_k2()
            mean = b[:, 0, 0, 0].real if (g.decomp.ky0 == 0) else torch.zeros(3, dtype=g.rdtype, device=g.device)
            mean_e = (mean ** 2).sum()
            sums += [float((w * eb).sum().item()), float((w * k2 * eb).sum().item()),
                     float((w * cross).sum().item()), float((w * hel).sum().item()),
                     float(mean_e.item())] + [float(v) for v in mean.tolist()]
            maxima.append(float((g.kx3 * b[0] + g.ky3 * b[1] + g.kz3 * b[2]).abs().max().item()))
        forcing = self.forcing
        power = 0.0
        if forcing is not None and forcing.kind != "white":
            power = forcing.local_power(self.uhat, forcing._to_local(forcing.f) if forcing.n_local else None)
        sums += [power, self.injected]
        sums = self.comm.allreduce(sums, "sum")
        maxima = self.comm.allreduce(maxima, "max")

        out: Dict[str, float] = {"time": self.t, "step": float(self.steps)}
        ke, wu = 0.5 * sums[0], sums[1]
        out["kinetic_energy"] = ke
        out["viscous_dissipation"] = self.nu * wu
        out["enstrophy"] = 0.5 * wu
        out["max_div_u_hat"] = float(maxima[0])
        urms = math.sqrt(max(2.0 * ke, 0.0))
        out["u_rms"] = urms
        if self.mhd:
            me, wb, hc, hm, mean_e = 0.5 * sums[2], sums[3], sums[4], sums[5], sums[6]
            out["magnetic_energy"] = me
            out["mean_field_energy"] = 0.5 * mean_e
            out["resistive_dissipation"] = self.eta * wb
            out["cross_helicity"] = hc
            out["magnetic_helicity"] = hm
            out["mean_field_x"], out["mean_field_y"], out["mean_field_z"] = sums[7], sums[8], sums[9]
            out["max_div_b_hat"] = float(maxima[1])
            brms = math.sqrt(max(2.0 * me, 0.0))
            out["b_rms"] = brms
            # Alfven Mach numbers: against the total field and against the mean field.
            out["alfven_mach"] = urms / brms if brms > 0.0 else float("inf")
            b0 = math.sqrt(max(mean_e, 0.0))
            out["alfven_mach_mean_field"] = urms / b0 if b0 > 0.0 else float("inf")
        out["total_energy"] = ke + (out.get("magnetic_energy", 0.0))
        out["forcing_power"] = float(sums[-2])
        out["injected_energy"] = float(sums[-1])
        eps = out["viscous_dissipation"]
        if self.nu > 0.0 and eps > 0.0:
            kmax = math.sqrt(g.k2_max / 3.0) if not self.spherical else math.sqrt(self.kcut2)
            eta_k = (self.nu ** 3 / eps) ** 0.25
            out["kolmogorov_scale"] = eta_k
            out["kmax_eta"] = kmax * eta_k
            uprime = urms / math.sqrt(3.0)
            lam = math.sqrt(15.0 * self.nu * uprime * uprime / eps)
            out["taylor_reynolds"] = uprime * lam / self.nu
        if self.mhd and self.eta > 0.0:
            out["magnetic_prandtl"] = self.nu / self.eta
        if self.rate is not None:
            out["cfl_rate"] = self.rate
        return out

    def spectra(self) -> Dict[str, np.ndarray]:
        """Shell-summed energy spectra: ``sum_k E(k) = <|u|^2>/2`` (and the same for ``b``).

        Shells are unit-wide in ``|k| L_max / (2 pi)``, centred on integers.
        """
        g = self.grid
        dk = 2.0 * np.pi / max(g.Lx, g.Ly, g.Lz)
        shell = torch.floor(torch.sqrt(g.k2()) / dk + 0.5).to(torch.int64)
        nbins = int(math.floor(math.sqrt(g.k2_max) / dk + 0.5)) + 1
        w = g.weight3

        def binned(X):
            density = 0.5 * w * (X.real ** 2 + X.imag ** 2).sum(dim=0)
            local = torch.bincount(shell.reshape(-1), weights=density.reshape(-1).to(torch.float64),
                                   minlength=nbins)
            return self.comm.allreduce(local.cpu().numpy(), "sum")

        out = {"k": np.arange(nbins) * dk, "kinetic": binned(self.uhat[0:3])}
        if self.mhd:
            out["magnetic"] = binned(self.uhat[3:6])
        return out
