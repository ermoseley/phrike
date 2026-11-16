"""Slab-decomposed spectral RK stepping for 3D compressible Euler and MHD.

This is the single-GPU fast path (``spectral3d`` / ``spectral_mhd3d``) on a
slab decomposition: real-to-complex transforms, the flux divergence and the RK
stages in spectral space, 9 (Euler) or 12 (MHD) transformed flux fields per
stage, and one Leray projection of B in the final stage.  Two things change:

* spectral arrays hold only the retained (2/3-rule) modes of this rank's ky
  slab, ``(C, Kz, nky, Kx)``, so a transform communicates 4/9 of the grid's
  data and the dealias mask is the identity on everything stored;
* scalars that decide the step (floor violations, max wave speed, the embedded
  error) are reduced over all ranks.

On CUDA the stage updates and the flux evaluation are the existing Triton
kernels of ``spectral3d_triton``, launched on the compact arrays (the kernels
take strides and per-axis 1D arrays, so a compact array is just a grid with
no modes dropped).  Everywhere else the same step runs on plain Torch ops.

With one rank the step equals ``SpectralSolverMHD3D`` / ``SpectralSolver3D``
to round-off (see ``tests/test_parallel_compressible.py``).

Positivity: stages run without the limiter while floor violations are counted;
when the count over all ranks is non-zero the step is repeated with the
reference limiter acting stage by stage (``_rk_step_limited``).
"""

from __future__ import annotations

import math
import os
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..spectral3d import _RK_TABLES, euler3d_symmetric_fluxes
from ..spectral3d import _count_below as _euler_count_below
from ..spectral_mhd3d import _mhd_count_below, mhd3d_fluxes
from .grid import SlabGrid3D


class _CompactModes:
    """Mode geometry of a compact local spectrum for the ``spectral3d_triton`` kernels.

    With ``cy >= Kyr`` and ``cz >= Kzr`` the kernels' index map is the identity.
    """

    def __init__(self, Kz: int, nky: int, Kx: int):
        self.Kx, self.Kyr, self.Kzr = int(Kx), int(nky), int(Kz)
        self.cy, self.cz = int(nky), int(Kz)
        self.Ny, self.Nz = int(nky), int(Kz)
        self.total = int(Kz) * int(nky) * int(Kx)


def _triton_default() -> bool:
    flag = os.environ.get("PHRIKE_TRITON")
    if flag is None:
        return True
    return flag.strip().lower() not in ("0", "false", "no", "off", "")


def mhd_clamp(U: torch.Tensor, gamma: float, rho_min: float, p_min: float) -> torch.Tensor:
    """Floor density and thermal pressure and rebuild the state (``SpectralSolverMHD._clamp``)."""
    rho = U[0]
    ux, uy, uz = U[1] / rho, U[2] / rho, U[3] / rho
    Bx, By, Bz = U[5], U[6], U[7]
    b2h = 0.5 * (Bx * Bx + By * By + Bz * Bz)
    p = (gamma - 1.0) * (U[4] - 0.5 * rho * (ux * ux + uy * uy + uz * uz) - b2h)
    rho_c = torch.clamp(rho, min=rho_min)
    p_c = torch.clamp(p, min=p_min)
    E = p_c / (gamma - 1.0) + 0.5 * rho_c * (ux * ux + uy * uy + uz * uz) + b2h
    return torch.stack([rho_c, rho_c * ux, rho_c * uy, rho_c * uz, E, Bx, By, Bz])


def euler_clamp(U: torch.Tensor, gamma: float, rho_min: float, p_min: float) -> torch.Tensor:
    """Floor density and pressure and rebuild the state (``_clamp_conserved``)."""
    rho = U[0]
    ux, uy, uz = U[1] / rho, U[2] / rho, U[3] / rho
    p = (gamma - 1.0) * (U[4] - 0.5 * rho * (ux * ux + uy * uy + uz * uz))
    rho_c = torch.clamp(rho, min=rho_min)
    p_c = torch.clamp(p, min=p_min)
    E = p_c / (gamma - 1.0) + 0.5 * rho_c * (ux * ux + uy * uy + uz * uz)
    return torch.stack([rho_c, rho_c * ux, rho_c * uy, rho_c * uz, E])


class CompressibleSlabStepper:
    """Distributed spectral RK stepping of ``EulerEquations3D`` or ``MHDEquations3D``.

    The state is the pair ``(U, uhat)``: this rank's slab of the conserved
    variables and the retained spectrum of the whole state divided by N (the
    ``norm="forward"`` convention), with ``U == inverse(uhat)``.
    """

    def __init__(self, grid: SlabGrid3D, system: str, gamma: float, rho_min: float,
                 p_min: float, eta: float = 0.0, nu: float = 0.0,
                 triton: Optional[bool] = None, strict_clamp: bool = False):
        """``strict_clamp``: also count floor violations of the final state
        before the modal dissipation, as the reference limiter does.  It costs
        one more inverse transform of the state per step and only matters when
        split-step dissipation is on and the undissipated state dips below a
        floor that the dissipated one stays above."""
        if system not in ("euler", "mhd"):
            raise ValueError("system must be 'euler' or 'mhd'")
        self.grid = grid
        self.comm = grid.comm
        self.system = system
        self.mhd = system == "mhd"
        self.nvar, self.nflux = (8, 12) if self.mhd else (5, 9)
        self.kind = "mhd3d" if self.mhd else "euler3d"
        self.gamma = float(gamma)
        self.rho_min, self.p_min = float(rho_min), float(p_min)
        self.eta, self.nu = float(eta), float(nu)
        if not self.mhd and (self.eta != 0.0 or self.nu != 0.0):
            raise ValueError("explicit viscosity/resistivity belong to the MHD system")
        self.strict_clamp = bool(strict_clamp)
        self.dissipation = bool(grid.spectral_dissipation_enabled)
        self.inv_n = grid.inv_n

        self.uhat = grid.zeros_spectral(self.nvar)
        self.shat = grid.zeros_spectral(self.nvar)
        self.unew = grid.zeros_spectral(self.nvar)
        self.U: Optional[torch.Tensor] = None
        self.speed: Optional[float] = None  # global max wave speed of the current state
        self.band_limited = True  # False after the accepted-state limiter acted
        self.events = 0  # limiter calls that raised at least one value
        self.cells = 0  # density and pressure values raised, over all ranks
        self._warned = False
        self._acc = None
        self._kbuf = None
        self._pending = None
        self._backup = None

        self._triton = None
        self._stats_kernel = None
        self._strict_kernels = False  # raise on a kernel failure instead of falling back
        if triton is None:
            triton = _triton_default()
        # "force" skips the device test (the test suite runs the kernels on an emulator).
        if triton and (grid.device.type == "cuda" or triton == "force"):
            try:
                from .. import spectral3d_triton as st

                self._triton = st
                d = grid.decomp
                self._modes = _CompactModes(d.Kz, d.nky, d.Kx)
                self._waves = (grid.kx, grid.ky, grid.kz) + tuple(grid.ones1d)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self._note(f"Triton unavailable for the distributed stages ({exc}); using torch ops.")
            try:
                from . import kernels_triton as kt

                self._stats_kernel = kt
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self._note(f"Triton state-statistics kernel unavailable ({exc}); using torch ops.")

    # --- kernel tier --------------------------------------------------------

    def warm_up(self, embedded: bool = False) -> bool:
        """Build the Triton kernels this stepper uses and settle the tier for all ranks.

        A kernel that failed on one rank in the middle of a distributed step
        would leave the ranks out of step with each other.  So every kernel
        variant is run here on a tiny serial problem, with no MPI inside:
        rank 0 first, which also fills Triton's on-disk cache for the others,
        then the rest.  The Triton tier is kept only if it works on every
        rank.  Returns whether it is in use.
        """
        if self._triton is None and self._stats_kernel is None:
            return False
        comm = self.comm
        problem = [""]

        def run() -> float:
            try:
                self._warm_up_local(embedded)
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
            self._stats_kernel = None
            self._acc = None
            self._kbuf = None
        return self._triton is not None

    def _warm_up_local(self, embedded: bool) -> None:
        from .comm import Comm

        g = self.grid
        tiny = SlabGrid3D(8, 8, 8, g.Lx, g.Ly, g.Lz, comm=Comm.serial(), device=g.device,
                          precision=g.precision, filter_params=g.filter_params, fft_batch=g.fft.batch)
        st = CompressibleSlabStepper(tiny, self.system, self.gamma, self.rho_min, self.p_min,
                                     eta=self.eta, nu=self.nu, triton="force",
                                     strict_clamp=self.strict_clamp)
        if (st._triton is None) != (self._triton is None) or (st._stats_kernel is None) != (
                self._stats_kernel is None):
            raise RuntimeError("the Triton kernel modules could not be imported")
        st._strict_kernels = True
        X, Y, Z = tiny.xyz_mesh()
        kx, ky, kz = 2.0 * np.pi / g.Lx, 2.0 * np.pi / g.Ly, 2.0 * np.pi / g.Lz
        rho = 1.0 + 0.1 * torch.cos(kx * X)
        u = [0.1 * torch.sin(ky * Y), 0.1 * torch.sin(kz * Z), 0.1 * torch.sin(kx * X)]
        fields = [rho, rho * u[0], rho * u[1], rho * u[2]]
        energy = 1.0 / (self.gamma - 1.0) + 0.5 * rho * (u[0] ** 2 + u[1] ** 2 + u[2] ** 2)
        if self.mhd:
            B = [0.1 * torch.cos(ky * Y), 0.1 * torch.cos(kz * Z), 0.1 * torch.cos(kx * X)]
            energy = energy + 0.5 * (B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
            fields = fields + [energy] + B
        else:
            fields = fields + [energy]
        st.seed(torch.stack(fields))
        dt = 1e-3 * tiny.min_spacing
        for scheme in ("rk2", "rk4"):
            st.step(dt, scheme)
        if embedded:
            from ..adaptive import create_adaptive_method

            method = create_adaptive_method("rk23")
            st.attempt(dt, method.a, method.b_high, method.b_low)
            st.commit()
        if st.events:
            raise RuntimeError("the warm-up state needed the positivity limiter")

    # --- small helpers ------------------------------------------------------

    def _note(self, message: str) -> None:
        if self.comm.is_root:
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _disable_triton(self, exc: Exception) -> None:
        if isinstance(exc, torch.OutOfMemoryError) or self._strict_kernels:
            raise exc
        self._note(f"Triton kernels failed on the distributed path ({exc!r}); using torch ops.")
        self._triton = None
        self._acc = None
        self._kbuf = None
        self._pending = None

    def _table(self, scheme: str):
        try:
            return _RK_TABLES[scheme.lower()]
        except KeyError:
            raise ValueError(f"fixed-step scheme must be rk2 or rk4, got {scheme!r}") from None

    def _count(self, U: torch.Tensor) -> torch.Tensor:
        fn = _mhd_count_below if self.mhd else _euler_count_below
        return fn(U, self.gamma, self.rho_min, self.p_min)

    def _fluxes(self, U: torch.Tensor, count: bool):
        st = self._triton
        if st is not None:
            return st.fluxes(self.kind, U, self.gamma, self.rho_min, self.p_min, count=count)
        F = mhd3d_fluxes(U, self.gamma) if self.mhd else euler3d_symmetric_fluxes(U, self.gamma)
        return (F, self._count(U)) if count else F

    def _local_stats(self, U: torch.Tensor) -> Tuple[float, float, float, float]:
        """This rank's (values below a floor, non-finite state values, non-finite
        wave speeds, max finite wave speed)."""
        kt = self._stats_kernel
        if kt is not None:
            try:
                return kt.state_stats(self.kind, U, self.gamma, self.rho_min, self.p_min)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                if isinstance(exc, torch.OutOfMemoryError) or self._strict_kernels:
                    raise
                self._note(f"Triton state-statistics kernel failed ({exc!r}); using torch ops.")
                self._stats_kernel = None
        return self._local_stats_torch(U)

    def _local_stats_torch(self, U: torch.Tensor) -> Tuple[float, float, float, float]:
        rho = U[0]
        ux, uy, uz = U[1] / rho, U[2] / rho, U[3] / rho
        kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
        if self.mhd:
            b2 = U[5] * U[5] + U[6] * U[6] + U[7] * U[7]
            p = (self.gamma - 1.0) * (U[4] - kinetic - 0.5 * b2)
            fast = torch.sqrt(torch.clamp((self.gamma * p + b2) / rho, min=0.0))
        else:
            p = (self.gamma - 1.0) * (U[4] - kinetic)
            fast = torch.sqrt(self.gamma * p / rho)
        comp = torch.maximum(torch.maximum(ux.abs(), uy.abs()), uz.abs()) + fast
        usable = torch.isfinite(comp)
        f64 = torch.float64
        values = torch.stack([
            ((rho < self.rho_min).sum() + (p < self.p_min).sum()).to(f64),
            (~torch.isfinite(U)).sum().to(f64),
            (~usable).sum().to(f64),
            torch.where(usable, comp, torch.zeros_like(comp)).max().to(f64),
        ]).tolist()
        return float(values[0]), float(values[1]), float(values[2]), float(values[3])

    def _reduce_stats(self, count: float, bad: float, unusable: float, speed: float):
        """Global (floor violations, non-finite state values, max wave speed), in one collective.

        The first two are the largest per-rank counts: only whether they are
        zero is used.  The speed is NaN when it is not finite somewhere
        (negative pressure, say); the state itself may still be finite, so
        that is left to the caller that needs a time step.
        """
        count, bad, unusable, speed = self.comm.allreduce([count, bad, unusable, speed], "max")
        return float(count), float(bad), (float("nan") if unusable > 0 else float(speed))

    def _require_finite(self, bad: float, where: str) -> None:
        if bad > 0:
            raise FloatingPointError(f"Non-finite solution {where}")

    def _note_events(self, count: float) -> None:
        if count <= 0:
            return
        self.events += 1
        self.cells += int(count)
        if not self._warned:
            self._warned = True
            self._note("Positivity clamp raised density or pressure to the floor "
                       f"({int(count)} values on the first event this run).")

    # --- spectral operators on the compact arrays (torch tier) --------------

    def _rhs_hat(self, Fh: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        """Spectral RHS from normalized flux spectra ``Fh`` and the stage spectrum ``S``."""
        g = self.grid
        ikx, iky, ikz = 1j * g.kx3, 1j * g.ky3, 1j * g.kz3

        def d(ax, ay, az):
            s = 0.0
            if ax is not None:
                s = s + ikx * ax
            if ay is not None:
                s = s + iky * ay
            if az is not None:
                s = s + ikz * az
            return -s

        if not self.mhd:
            return torch.stack([
                d(S[1], S[2], S[3]),
                d(Fh[0], Fh[3], Fh[4]), d(Fh[3], Fh[1], Fh[5]), d(Fh[4], Fh[5], Fh[2]),
                d(Fh[6], Fh[7], Fh[8]),
            ])
        k2 = g.k2()
        T, H, A = Fh[0:6], Fh[6:9], Fh[9:12]
        nu, eta = self.nu, self.eta
        return torch.stack([
            d(S[1], S[2], S[3]),
            d(T[0], T[3], T[4]) - nu * k2 * S[1],
            d(T[3], T[1], T[5]) - nu * k2 * S[2],
            d(T[4], T[5], T[2]) - nu * k2 * S[3],
            d(H[0], H[1], H[2]),
            d(None, -A[0], -A[1]) - eta * k2 * S[5],
            d(A[0], None, -A[2]) - eta * k2 * S[6],
            d(A[1], A[2], None) - eta * k2 * S[7],
        ])

    def _leray(self, X: torch.Tensor) -> torch.Tensor:
        """Leray-project the magnetic components of a spectrum (new tensor)."""
        if not self.mhd:
            return X
        g = self.grid
        kx, ky, kz = g.kx3, g.ky3, g.kz3
        k2 = g.k2()
        inv = torch.where(k2 == 0, torch.zeros_like(k2), 1.0 / torch.where(k2 == 0, torch.ones_like(k2), k2))
        c = (kx * X[5] + ky * X[6] + kz * X[7]) * inv
        X = X.clone()
        X[5] -= kx * c
        X[6] -= ky * c
        X[7] -= kz * c
        return X

    def _stabilize(self, high: torch.Tensor, h: float) -> torch.Tensor:
        """Project B, then apply the second half of the modal dissipation."""
        new = self._leray(high)
        if self.dissipation:
            new = new * self.grid.product3(self.grid.decay_factors(h))
        return new

    # --- state --------------------------------------------------------------

    def seed(self, U: torch.Tensor) -> torch.Tensor:
        """Adopt a physical state: dealias it, project B and hold thermal pressure.

        For a band-limited solenoidal ``U`` this is the reference
        initialization (``_project_B`` then ``project_dealiased``) to
        round-off.  Content above the dealias cutoff is dropped before the
        projection rather than after it.
        """
        g = self.grid
        if tuple(U.shape) != (self.nvar,) + g.decomp.physical_shape:
            raise ValueError(f"state must be {(self.nvar,) + g.decomp.physical_shape}, got {tuple(U.shape)}")
        U = U.to(dtype=g.rdtype, device=g.device)
        g.forward(U, normalized=True, out=self.uhat)
        if self.mhd:
            projected = self._leray(self.uhat)
            Bn = g.inverse(projected[5:8])
            energy = U[4] + 0.5 * ((Bn * Bn).sum(dim=0) - (U[5:8] * U[5:8]).sum(dim=0))
            projected[4] = g.forward(energy[None], normalized=True)[0]
            self.uhat.copy_(projected)
        return self.set_spectrum(self.uhat)

    def set_spectrum(self, uhat: torch.Tensor) -> torch.Tensor:
        """Adopt a retained spectrum (normalized by 1/N) as the state."""
        if uhat is not self.uhat:
            self.uhat.copy_(uhat)
        self.U = self.grid.inverse(self.uhat)
        self.band_limited = True
        self._pending = None
        self._refresh_stats("in the initial state")
        return self.U

    def _refresh_stats(self, where: str) -> None:
        count, bad, speed = self._reduce_stats(*self._local_stats(self.U))
        self._require_finite(bad, where)
        self.speed = speed

    def physical(self) -> torch.Tensor:
        """This rank's slab of the conserved variables."""
        if self.U is None:
            self.U = self.grid.inverse(self.uhat)
        return self.U

    def max_div_b(self) -> float:
        """Global max |div B| in physical space (one inverse transform)."""
        if not self.mhd:
            return 0.0
        g = self.grid
        div = 1j * (g.kx3 * self.uhat[5] + g.ky3 * self.uhat[6] + g.kz3 * self.uhat[7])
        return self.comm.max(float(g.inverse(div[None]).abs().max().item()))

    # --- split-step dissipation ---------------------------------------------

    def dissipate(self, h: float) -> None:
        """First half of the modal dissipation, ``exp(-h * lambda)``, on the state."""
        if not self.dissipation:
            return
        g = self.grid
        factors = g.decay_factors(h)
        self._pending = None
        st = self._triton
        if st is not None:
            try:
                w = torch.tensor([self.inv_n], dtype=g.rdtype, device=g.device)
                st.combine(None, 0, self.uhat, self.uhat, w, 0, factors, self._modes, masked=True)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self._disable_triton(exc)
                st = None
                self.uhat.mul_(g.product3(factors))
        else:
            self.uhat.mul_(g.product3(factors))
        self.U = g.inverse(self.uhat)

    # --- fixed RK step ------------------------------------------------------

    def step(self, dt: float, scheme: str = "rk4") -> None:
        """One step: dissipation(dt/2), RK2/RK4, projection, dissipation(dt/2)."""
        table = self._table(scheme)
        h = 0.5 * float(dt)
        self.dissipate(h)
        accepted = False
        if self._triton is not None:
            try:
                accepted = self._rk_step_triton(dt, table, h)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self._disable_triton(exc)
                accepted = self._rk_step_generic(dt, table, h)
        else:
            accepted = self._rk_step_generic(dt, table, h)
        if not accepted:
            self._rk_step_limited(dt, table, h)

    def _adopt(self, new_spec: torch.Tensor, Un: torch.Tensor, speed: float) -> None:
        if new_spec is self.unew:
            self.uhat, self.unew = self.unew, self.uhat
        else:
            self.uhat.copy_(new_spec)
        self.U = Un
        self.speed = speed
        self.band_limited = True
        self._pending = None

    def _judge(self, new_spec: torch.Tensor, stage_count, high: Optional[torch.Tensor]) -> bool:
        """Transform the candidate, reduce its checks and adopt it if no floor is hit."""
        g = self.grid
        Un = g.inverse(new_spec)
        count, bad, unusable, speed = self._local_stats(Un)
        count += float(stage_count)
        if self.strict_clamp and self.dissipation and high is not None:
            count += float(self._count(g.inverse(high)))
        count, bad, speed = self._reduce_stats(count, bad, unusable, speed)
        if count > 0:
            return False
        self._require_finite(bad, "after a step")
        self._adopt(new_spec, Un, speed)
        return True

    def _rk_step_generic(self, dt: float, table, h: float) -> bool:
        g = self.grid
        U0h = self.uhat
        acc = None
        Sh, Us = U0h, self.physical()
        count = torch.zeros((), dtype=torch.int64, device=g.device)
        high = None
        for i, (wa, wn) in enumerate(table):
            if i > 0:
                F, c = self._fluxes(Us, True)
                count = count + c
            else:
                F = self._fluxes(Us, False)
                self.U = None  # rebuilt from uhat if the step has to be repeated
            Us = None
            Fh = g.forward(F, normalized=True)
            del F
            k = self._rhs_hat(Fh, Sh)
            del Fh
            if wn is None:
                high = U0h + (wa * dt) * k
                if acc is not None:
                    high = high + acc
                break
            if wa != 0.0:
                acc = (wa * dt) * k if acc is None else acc + (wa * dt) * k
            Sh = U0h + (wn * dt) * k
            Us = g.inverse(Sh)
        return self._judge(self._stabilize(high, h), int(count.item()), high)

    def _rk_step_triton(self, dt: float, table, h: float) -> bool:
        g, st, modes = self.grid, self._triton, self._modes
        uhat, shat, unew = self.uhat, self.shat, self.unew
        if self._acc is None:
            self._acc = torch.empty((self.nvar, modes.total), dtype=g.cdtype, device=g.device)
        acc = self._acc
        self._pending = None
        head = [self.inv_n, self.eta, self.nu] if self.mhd else [self.inv_n]
        w = torch.tensor(head + [c * dt for wa, wn in table for c in (wa, 0.0 if wn is None else wn)],
                         dtype=g.rdtype, device=g.device)
        base = len(head)
        factors = g.decay_factors(h)
        count = torch.zeros((), dtype=torch.int64, device=g.device)
        Us, Sin = self.physical(), uhat
        last = len(table) - 1
        for i in range(len(table)):
            if i > 0:
                F, c = self._fluxes(Us, True)
                count += c
            else:
                F = self._fluxes(Us, False)
                self.U = None  # rebuilt from uhat if the step has to be repeated
            Us = None
            Fh = g.forward(F, normalized=False)
            del F
            final = i == last
            launch = st.mhd_stage if self.mhd else st.stage
            launch(Fh, Sin, unew if final else shat, uhat, acc, w, base + 2 * i, base + 1 + 2 * i,
                   self._waves, modes, first=(i == 0), final=final, high=shat, final_factors=factors)
            del Fh
            if final:
                break
            Sin = shat
            Us = g.inverse(shat)
        # shat: state before projection and dissipation; unew: the candidate.
        return self._judge(unew, int(count.item()), shat)

    # --- the same step with the positivity limiter acting -------------------

    def _limit(self, Us: torch.Tensor, Sh: torch.Tensor):
        """Reference limiter on a stage: clamp where needed and re-transform."""
        count = self.comm.sum(float(self._count(Us).item()))
        if count <= 0:
            return Us, Sh, False
        self._note_events(count)
        clamp = mhd_clamp if self.mhd else euler_clamp
        Us = clamp(Us, self.gamma, self.rho_min, self.p_min)
        return Us, self.grid.forward(Us, normalized=True), True

    def _rk_step_limited(self, dt: float, table, h: float) -> None:
        """RK2/RK4 with the reference limiter on every stage (rare, torch ops).

        Stage states are ``inverse(spectrum)``; a clamped stage is transformed
        back so the mass flux and the diffusion terms see the limited state on
        the retained modes, as the reference's masked derivatives do.
        """
        g = self.grid
        U0h = self.uhat
        acc = torch.zeros_like(U0h)
        Sh, Us = U0h, self.physical()
        self.U = None
        high = None
        for wa, wn in table:
            k = self._rhs_hat(g.forward(self._fluxes_torch(Us), normalized=True), Sh)
            if wn is None:
                high = U0h + acc + (wa * dt) * k
                break
            acc = acc + (wa * dt) * k
            Sh = U0h + (wn * dt) * k
            Us, Sh, _ = self._limit(g.inverse(Sh), Sh)
        _, high, _ = self._limit(g.inverse(high), high)
        self._finish_limited(high, h)

    def _fluxes_torch(self, U: torch.Tensor) -> torch.Tensor:
        return mhd3d_fluxes(U, self.gamma) if self.mhd else euler3d_symmetric_fluxes(U, self.gamma)

    def _finish_limited(self, high: torch.Tensor, h: float) -> None:
        g = self.grid
        new = self._stabilize(high, h)
        Un = g.inverse(new)
        clamped = False
        if self.mhd:  # only the MHD reference limits the accepted state
            Un, new, clamped = self._limit(Un, new)
        self.uhat.copy_(new)
        self.U = Un
        self.band_limited = not clamped
        self._pending = None
        self._refresh_stats("after a limited step")

    # --- embedded (adaptive) RK ---------------------------------------------

    def attempt(self, dt: float, a, b_high, b_low) -> Tuple[float, float]:
        """One adaptive attempt: dissipation(dt/2), then an embedded RK step.

        Returns ``(error, solution_scale)`` reduced over all ranks, where
        ``solution_scale`` is ``max|U|`` of the split state.  Follow with
        ``commit`` to accept the candidate or ``reject`` to restore the state.
        """
        h = 0.5 * float(dt)
        self._backup = self.uhat.clone() if self.dissipation else None
        self.dissipate(h)
        scale = self.comm.max(float(self.physical().abs().max().item()))
        return self.embedded_step(dt, a, b_high, b_low, h), scale

    def reject(self) -> None:
        """Discard the candidate and return to the state before ``attempt``."""
        self._pending = None
        backup = getattr(self, "_backup", None)
        if backup is not None:
            self.uhat.copy_(backup)
            self.U = None  # rebuilt from the restored spectrum on demand
        self._backup = None

    def embedded_step(self, dt: float, a, b_high, b_low, h: float) -> float:
        """One embedded-RK attempt from the current state (after ``dissipate(h)``).

        Returns the global error estimate ``max|U_high - U_low|``.  The
        candidate (higher-order solution, projected, dissipated) is held
        until ``commit``; calling anything else discards it.
        """
        if self._triton is not None:
            try:
                out = self._embedded_triton(dt, a, b_high, b_low, h)
            except Exception as exc:  # pragma: no cover - depends on toolchain
                self._disable_triton(exc)
                out = self._embedded_generic(dt, a, b_high, b_low, h)
        else:
            out = self._embedded_generic(dt, a, b_high, b_low, h)
        if out is None:
            out = self._embedded_limited(dt, a, b_high, b_low, h)
        return out

    def commit(self) -> None:
        """Adopt the candidate of the last ``embedded_step``."""
        if self._pending is None:
            raise RuntimeError("no embedded step is pending")
        spec, Un, speed, band_limited = self._pending
        if spec is self.shat:
            self.uhat, self.shat = self.shat, self.uhat
        else:
            self.uhat.copy_(spec)
        self.U, self.speed = Un, speed
        self.band_limited = band_limited
        self._pending = None
        self._backup = None

    def _embedded_judge(self, spec, Un, count, err_local, extra_bad=0.0):
        """Reduce the checks of an embedded candidate; ``None`` if a floor is hit."""
        c2, bad, unusable, speed = self._local_stats(Un)
        if self.mhd:
            count += c2  # the accepted-state limiter input
        count, bad, speed = self._reduce_stats(count, bad + extra_bad, unusable, speed)
        error = self.comm.max(err_local if np.isfinite(err_local) else float("inf"))
        if count > 0:
            return None
        if bad > 0:
            error = float("nan")  # the controller rejects and shrinks the step
        self._pending = (spec, Un, speed, True)
        return error

    def _embedded_generic(self, dt, a, b_high, b_low, h):
        g = self.grid
        U0h = self.uhat
        s = len(b_high)
        ks = []
        Us, Sh = self.physical(), U0h
        count = torch.zeros((), dtype=torch.int64, device=g.device)
        for i in range(s):
            if i > 0:
                Sh = U0h + sum(dt * float(a[i][j]) * ks[j] for j in range(i) if a[i][j] != 0.0)
                Us = g.inverse(Sh)
                if not self.mhd:  # Euler stages are limited in the reference; MHD ones are not
                    count = count + self._count(Us)
            ks.append(self._rhs_hat(g.forward(self._fluxes(Us, False), normalized=True), Sh))
        Eh = sum(dt * float(b_high[i] - b_low[i]) * ks[i] for i in range(s) if b_high[i] != b_low[i])
        high = U0h + sum(dt * float(b_high[i]) * ks[i] for i in range(s) if b_high[i] != 0.0)
        del ks
        E = g.inverse(Eh)
        U_high = g.inverse(high)
        err = float(E.abs().max().item())
        count = float(count.item()) + float(self._count(U_high).item())
        if not self.mhd:
            count += float(self._count(U_high - E).item())
        del E
        if self.mhd or self.dissipation:
            new = self._stabilize(high, h)
            Un = g.inverse(new)
        else:
            new, Un = high, U_high  # band-limited: the projection is the identity
        return self._embedded_judge(new, Un, count, err)

    def _embedded_triton(self, dt, a, b_high, b_low, h):
        g, st, modes = self.grid, self._triton, self._modes
        uhat, shat = self.uhat, self.shat
        s = len(b_high)
        if self._kbuf is None or self._kbuf.shape[0] < s:
            self._kbuf = torch.empty((s, self.nvar, modes.total), dtype=g.cdtype, device=g.device)
        K = self._kbuf
        b_high = np.asarray(b_high, dtype=float)
        coefs = [self.inv_n, self.eta, self.nu] if self.mhd else [self.inv_n]
        offsets = [0]
        for i in range(1, s):
            offsets.append(len(coefs))
            coefs += [dt * float(a[i][j]) for j in range(i)]
        off_err = len(coefs)
        coefs += list(dt * (b_high - np.asarray(b_low, dtype=float)))
        off_new = len(coefs)
        coefs += list(dt * b_high)
        w = torch.tensor(coefs, dtype=g.rdtype, device=g.device)
        count = torch.zeros((), dtype=torch.int64, device=g.device)
        self._pending = None
        Us, S = self.physical(), uhat
        for i in range(s):
            if i > 0:
                st.combine(K, i, uhat, shat, w, offsets[i], None, modes, masked=False)
                S = shat
                Us = g.inverse(shat)
            if i > 0 and not self.mhd:
                F, c = self._fluxes(Us, True)
                count += c
            else:
                F = self._fluxes(Us, False)
            Us = None
            Fh = g.forward(F, normalized=False)
            del F
            if self.mhd:
                st.mhd_stage(Fh, S, S, S, K[i], w, 0, 0, self._waves, modes, first=False,
                             final=False, rhs_only=True)
            else:
                st.rhs(Fh, S, K[i], w, self._waves, modes)
            del Fh
        st.combine(K, s, None, shat, w, off_err, None, modes, masked=False)
        E = g.inverse(shat)
        st.combine(K, s, uhat, shat, w, off_new, None, modes, masked=False)
        U_high = g.inverse(shat)
        err = float(E.abs().max().item())
        total = float(count.item()) + float(self._count(U_high).item())
        if not self.mhd:
            total += float(self._count(U_high - E).item())
        del E
        factors = g.decay_factors(h)
        if self.mhd:
            del U_high
            st.leray_scale(shat, self._waves, factors, modes)
            Un = g.inverse(shat)
        elif self.dissipation:
            del U_high
            st.combine(None, 0, shat, shat, w, 0, factors, modes, masked=True)
            Un = g.inverse(shat)
        else:
            Un = U_high
        return self._embedded_judge(shat, Un, total, err)

    def _embedded_limited(self, dt, a, b_high, b_low, h) -> float:
        """Embedded attempt with the reference limiter acting (rare, torch ops)."""
        g = self.grid
        U0h = self.uhat
        s = len(b_high)
        ks = []
        Us, Sh = self.physical(), U0h
        for i in range(s):
            if i > 0:
                Sh = U0h + sum(dt * float(a[i][j]) * ks[j] for j in range(i) if a[i][j] != 0.0)
                Us = g.inverse(Sh)
                if not self.mhd:
                    Us, Sh, _ = self._limit(Us, Sh)
            ks.append(self._rhs_hat(g.forward(self._fluxes_torch(Us), normalized=True), Sh))
        high = U0h + sum(dt * float(b_high[i]) * ks[i] for i in range(s) if b_high[i] != 0.0)
        if self.mhd:
            Eh = sum(dt * float(b_high[i] - b_low[i]) * ks[i] for i in range(s) if b_high[i] != b_low[i])
            err = float(g.inverse(Eh).abs().max().item())
            _, high, _ = self._limit(g.inverse(high), high)
        else:
            low = U0h + sum(dt * float(b_low[i]) * ks[i] for i in range(s) if b_low[i] != 0.0)
            U_high, high, _ = self._limit(g.inverse(high), high)
            U_low, _, _ = self._limit(g.inverse(low), low)
            err = float((U_high - U_low).abs().max().item())
            del U_high, U_low
        del ks
        new = self._stabilize(high, h)
        Un = g.inverse(new)
        clamped = False
        if self.mhd:
            Un, new, clamped = self._limit(Un, new)
        _, bad, speed = self._reduce_stats(*self._local_stats(Un))
        error = self.comm.max(err if np.isfinite(err) else float("inf"))
        if bad > 0:
            error = float("nan")
        self._pending = (new, Un, speed, not clamped)
        return error

    # --- forcing ------------------------------------------------------------

    def kick(self, dv: torch.Tensor) -> None:
        """Add a velocity increment ``dv`` (3, nz, Ny, Nx) at fixed density and internal energy.

        Momentum gains ``rho dv`` and the total energy gains the kinetic energy
        change.  The products carry a little content above the dealias cutoff,
        so the momentum and energy are projected back onto the retained modes
        (four forward and four inverse transforms): the state stays
        band-limited and equal to its stored spectrum, which keeps restarts
        exact.
        """
        g = self.grid
        U = self.physical()
        rho = U[0]
        U[4] += (U[1:4] * dv).sum(dim=0) + 0.5 * rho * (dv * dv).sum(dim=0)
        U[1:4] += rho * dv
        g.forward(U[1:5], normalized=True, out=self.uhat[1:5])
        g.inverse(self.uhat[1:5], out=U[1:5])
        self._pending = None
        self._refresh_stats("after a forcing kick")

    # --- diagnostics --------------------------------------------------------

    def diagnostics(self) -> Dict[str, float]:
        """Volume-averaged conserved quantities and derived turbulence numbers."""
        U = self.physical()
        gamma = self.gamma
        rho = U[0]
        kin = 0.5 * (U[1:4] * U[1:4]).sum(dim=0) / rho
        if self.mhd:
            mag = 0.5 * (U[5:8] * U[5:8]).sum(dim=0)
            cross = (U[1:4] * U[5:8]).sum(dim=0) / rho
        else:
            mag = torch.zeros_like(rho)
            cross = torch.zeros_like(rho)
        p = (gamma - 1.0) * (U[4] - kin - mag)
        f64 = torch.float64
        sums = torch.stack([rho.sum(dtype=f64), U[1].sum(dtype=f64), U[2].sum(dtype=f64),
                            U[3].sum(dtype=f64), U[4].sum(dtype=f64), kin.sum(dtype=f64),
                            mag.sum(dtype=f64), cross.sum(dtype=f64), p.sum(dtype=f64)]).tolist()
        mins = torch.stack([rho.min().to(f64), p.min().to(f64)]).tolist()
        sums = self.comm.allreduce(sums, "sum") / float(self.grid.n_points)
        mins = self.comm.allreduce(mins, "min")
        out = {
            "mass": sums[0], "momentum_x": sums[1], "momentum_y": sums[2], "momentum_z": sums[3],
            "energy": sums[4], "kinetic_energy": sums[5], "magnetic_energy": sums[6],
            "cross_helicity": sums[7], "mean_pressure": sums[8],
            "min_density": float(mins[0]), "min_pressure": float(mins[1]),
            "positivity_events": float(self.events),
        }
        out["internal_energy"] = sums[8] / (gamma - 1.0)
        urms = math.sqrt(max(2.0 * sums[5] / sums[0], 0.0)) if sums[0] > 0.0 else float("nan")
        out["u_rms"] = urms
        if sums[8] > 0.0 and sums[0] > 0.0:
            out["sonic_mach"] = urms / math.sqrt(gamma * sums[8] / sums[0])
        if self.mhd:
            out["alfven_mach"] = math.sqrt(sums[5] / sums[6]) if sums[6] > 0.0 else float("inf")
            out["plasma_beta"] = sums[8] / sums[6] if sums[6] > 0.0 else float("inf")
        if self.speed is not None:
            out["max_wave_speed"] = float(self.speed)
        return out

    def spectra(self) -> Dict[str, np.ndarray]:
        """Shell-summed spectra of the velocity (and B): ``sum_k E(k) = <|u|^2>/2``."""
        g = self.grid
        U = self.physical()
        dk = 2.0 * np.pi / max(g.Lx, g.Ly, g.Lz)
        shell = torch.floor(torch.sqrt(g.k2()) / dk + 0.5).to(torch.int64).reshape(-1)
        nbins = int(math.floor(math.sqrt(g.k2_max) / dk + 0.5)) + 1

        def binned(X):
            density = 0.5 * g.weight3 * (X.real ** 2 + X.imag ** 2).sum(dim=0)
            local = torch.bincount(shell, weights=density.reshape(-1).to(torch.float64), minlength=nbins)
            return self.comm.allreduce(local.cpu().numpy(), "sum")

        out = {"k": np.arange(nbins) * dk,
               "kinetic": binned(g.forward(U[1:4] / U[0], normalized=True))}
        if self.mhd:
            out["magnetic"] = binned(self.uhat[5:8])
        return out
