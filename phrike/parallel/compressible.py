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

The MHD accepted-state limiter acts after the 2/3 projection, so a state it
clamps carries content above the cutoff.  The reference keeps that content
for exactly one more step: the next step dissipates the whole spectrum of the
clamped state, evaluates every stage on the full physical state and removes
the out-of-band part again with the accepted-state projection.  The stepper
keeps it as ``residual`` ``R`` (physical space, this rank's slab, ``None``
while the state is band-limited) next to the retained spectrum: ``R`` is damped
with the full-spectrum factors by a full distributed transform that is built
only then, every stage state is ``inverse(stage spectrum) + R``, and ``R`` is
dropped when the step is accepted (unless the limiter clamps again).
"""

from __future__ import annotations

import copy
import math
import os
import sys
import traceback
import warnings
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from ..spectral3d import _RK_TABLES, euler3d_symmetric_fluxes
from ..spectral3d import _count_below as _euler_count_below
from ..spectral_mhd3d import (_mhd_count_below, derivative_spectra, mhd3d_fluxes,
                              mhd3d_transport_fluxes, mhd_temperature)
from ..transport import Transport, flux_buffer_rows
from .comm import Comm
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


class _ScratchFFT:
    """Shape-correct stand-in for ``SlabFFT`` with no communication (warm-up only).

    ``forward`` returns zeros and ``inverse`` a fixed positive state, so a
    scratch stepper runs its kernels on this rank's real local shapes, strides
    and totals without touching MPI.
    """

    def __init__(self, grid: SlabGrid3D, nvar: int, gamma: float):
        d = grid.decomp
        self.d, self.cdtype, self.device = d, grid.cdtype, grid.device
        # Rows past nvar stand for the transport derivative fields (up to 9 go back at once).
        U = torch.zeros((max(nvar, 12),) + d.physical_shape, dtype=grid.rdtype, device=grid.device)
        U[0] = 1.0
        U[4] = 1.0 / (gamma - 1.0)
        if nvar == 8:
            U[5:8] = 0.1
            U[4] += 0.5 * 3 * 0.1 ** 2
        self.state = U

    def forward(self, f, normalized=False, out=None):
        if out is None:
            return torch.zeros((f.shape[0],) + self.d.spectral_shape, dtype=self.cdtype, device=self.device)
        out.zero_()
        return out

    def inverse(self, S, out=None):
        if out is None:
            return self.state[:S.shape[0]]
        out.copy_(self.state[:S.shape[0]])
        return out


class CompressibleSlabStepper:
    """Distributed spectral RK stepping of ``EulerEquations3D`` or ``MHDEquations3D``.

    The state is the pair ``(U, uhat)``: this rank's slab of the conserved
    variables and the retained spectrum of the whole state divided by N (the
    ``norm="forward"`` convention), with ``U == inverse(uhat) + R``, where the
    out-of-band residual ``R`` (``self.residual``) exists only on the step
    after the MHD accepted-state limiter has clamped (see the module notes).
    """

    def __init__(self, grid: SlabGrid3D, system: str, gamma: float, rho_min: float,
                 p_min: float, transport: Optional[Transport] = None,
                 triton: Optional[bool] = None, strict_clamp: bool = True):
        """``strict_clamp``: also count floor violations of the final state
        before the modal dissipation, as the reference limiter does (the default).
        It costs one more inverse transform of the state per step and only
        matters when split-step dissipation is on and the undissipated state
        dips below a floor that the dissipated one stays above.  ``False``
        skips that count: faster, but a step the reference would limit can pass.

        ``transport``: viscosity, resistivity and conduction of the MHD system
        (``phrike.transport``); the Euler system is inviscid."""
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
        self.transport = transport if transport is not None else Transport()
        if not self.mhd and self.transport.active:
            raise ValueError("viscosity, resistivity and conduction belong to the MHD system")
        self.strict_clamp = bool(strict_clamp)
        self.dissipation = bool(grid.spectral_dissipation_enabled)
        self.inv_n = grid.inv_n

        self.uhat = grid.zeros_spectral(self.nvar)
        self.shat = grid.zeros_spectral(self.nvar)
        self.unew = grid.zeros_spectral(self.nvar)
        self._has_state = False  # set by seed/set_spectrum; warm_up may borrow the buffers before
        self.U: Optional[torch.Tensor] = None
        self.speed: Optional[float] = None  # global max wave speed of the current state
        self._residual: Optional[torch.Tensor] = None  # out-of-band part after a clamp (5, nz, Ny, Nx)
        self.events = 0  # limiter calls that raised at least one value
        self.cells = 0  # density and pressure values raised, over all ranks
        self._warned = False
        self._acc = None
        self._kbuf = None
        self._pending = None
        self._backup = None

        self._triton = None
        self._stats_kernel = None
        self._strict_kernels = False  # raise on a kernel failure instead of falling back (set by warm_up)
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
        """See ``_warm_up``.  A stepper with no state yet (the driver warms up before it
        seeds) lends its three spectral buffers to the full-size scratch stepper."""
        fresh = not self._has_state
        if fresh:
            self.uhat = self.shat = self.unew = None
        try:
            return self._warm_up(embedded)
        finally:
            if fresh:
                g = self.grid
                self.uhat, self.shat, self.unew = (g.zeros_spectral(self.nvar) for _ in range(3))

    def _warm_up(self, embedded: bool) -> bool:
        """Build the Triton kernels this stepper uses and settle the tier for all ranks.

        A kernel that failed on one rank in the middle of a distributed step
        would leave the ranks out of step with each other.  So every kernel
        variant is run here, with no MPI inside, on a tiny serial problem and
        then on scratch buffers of this rank's real local shapes (Triton
        specializes on sizes such as a row count of 1 or a multiple of 16, and
        slabs differ between ranks): rank 0 first, which also fills Triton's
        on-disk cache for the others, then the rest.  The Triton tier is kept
        only if it works on every rank, and from then on a kernel failure
        raises (the driver aborts all ranks) instead of switching one rank to
        torch ops.  Returns whether the Triton tier is in use.
        """
        comm = self.comm
        # Settle which modules every rank has before anyone returns or enters a barrier.
        if comm.min(0.0 if self._triton is None else 1.0) < 1.0:
            self._triton = None
        if comm.min(0.0 if self._stats_kernel is None else 1.0) < 1.0:
            self._stats_kernel = None
        if self._triton is None and self._stats_kernel is None:
            return False
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
        elif self._triton is not None or self._stats_kernel is not None:
            self._strict_kernels = True
        return self._triton is not None

    def _warm_up_local(self, embedded: bool) -> None:
        self._warm_up_tiny(embedded)
        self._warm_up_shapes(embedded)

    def _warm_up_shapes(self, embedded: bool) -> None:
        """Run the kernels on this rank's real local shapes without any communication.

        A scratch stepper over a copy of the grid whose transforms are
        ``_ScratchFFT`` takes the same RK steps as the real one, so the kernels
        see the real (Kz, nky, Kx) spectra, (C, nz, Ny, Nx) fields, totals and
        strides.  Nothing of this stepper is touched.
        """
        g = self.grid
        shim = copy.copy(g)
        shim.comm = Comm.serial()
        shim.fft = _ScratchFFT(g, self.nvar, self.gamma)
        st = CompressibleSlabStepper(shim, self.system, self.gamma, self.rho_min, self.p_min,
                                     transport=self.transport, triton="force",
                                     strict_clamp=self.strict_clamp)
        self._match_tiers(st)
        st._strict_kernels = True
        st.set_spectrum(st.uhat)
        dt = self._warm_up_dt(g, 1.0, 1.0)
        for scheme in ("rk2", "rk4"):
            st.step(dt, scheme)
        if embedded:
            from ..adaptive import create_adaptive_method

            method = create_adaptive_method("rk23")
            st.attempt(dt, method.a, method.b_high, method.b_low)
            st.commit()
        if st.events:
            raise RuntimeError("the shape warm-up state needed the positivity limiter")

    def _warm_up_tiny(self, embedded: bool) -> None:
        g = self.grid
        tiny = SlabGrid3D(8, 8, 8, g.Lx, g.Ly, g.Lz, comm=Comm.serial(), device=g.device,
                          precision=g.precision, filter_params=g.filter_params, fft_batch=g.fft.batch)
        st = CompressibleSlabStepper(tiny, self.system, self.gamma, self.rho_min, self.p_min,
                                     transport=self.transport, triton="force",
                                     strict_clamp=self.strict_clamp)
        self._match_tiers(st)
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
        dt = self._warm_up_dt(tiny, 0.9, 1.1)
        for scheme in ("rk2", "rk4"):
            st.step(dt, scheme)
        if embedded:
            from ..adaptive import create_adaptive_method

            method = create_adaptive_method("rk23")
            st.attempt(dt, method.a, method.b_high, method.b_low)
            st.commit()
        if st.events:
            raise RuntimeError("the warm-up state needed the positivity limiter")

    def _match_tiers(self, st: "CompressibleSlabStepper") -> None:
        """Give a scratch stepper this stepper's kernel modules: a module that the collective
        check removed on every rank stays off, and one that the scratch could not import fails."""
        if self._triton is None:
            st._triton = None
        if self._stats_kernel is None:
            st._stats_kernel = None
        if (st._triton is None) != (self._triton is None) or (st._stats_kernel is None) != (
                self._stats_kernel is None):
            raise RuntimeError("the Triton kernel modules could not be imported")

    def _warm_up_dt(self, grid: SlabGrid3D, rho_min: float, rho_max: float) -> float:
        """A step the warm-up state takes without the limiter: below its diffusive bound too."""
        dt = 1e-3 * grid.min_spacing
        rate = self.transport.max_diffusivity(self.gamma, rho_min, rho_max) * grid.k2_max
        return min(dt, 0.1 / rate) if rate > 0.0 else dt

    # --- small helpers ------------------------------------------------------

    def _note(self, message: str) -> None:
        if self.comm.is_root:
            warnings.warn(message, RuntimeWarning, stacklevel=3)

    def _disable_triton(self, exc: Exception) -> None:
        if isinstance(exc, FloatingPointError):
            raise exc  # a non-finite state, judged on reduced statistics: the same on every rank
        if self.comm.size > 1:
            # The other ranks are in, or about to enter, this stage's transposes: one rank that
            # switched to torch ops (or raised) alone would leave them waiting or mismatched.
            sys.stderr.write(f"[phrike.parallel] rank {self.comm.rank}: a kernel failed during a "
                             f"distributed step ({exc!r}); aborting all ranks\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            sys.stderr.flush()
            self.comm.abort(1)
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
        F = self._fluxes_torch(U)
        return (F, self._count(U)) if count else F

    @staticmethod
    def _planes(n: int):
        """Slices of the local z planes for pointwise torch-tier passes: bounded temporaries."""
        step = max(1, -(-n // 8))
        return [slice(i, min(n, i + step)) for i in range(0, n, step)]

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
        T, H, A = Fh[0:6], Fh[6:9], Fh[9:12]
        k = [
            d(S[1], S[2], S[3]),
            d(T[0], T[3], T[4]), d(T[3], T[1], T[5]), d(T[4], T[5], T[2]),
            d(H[0], H[1], H[2]),
            d(None, -A[0], -A[1]), d(A[0], None, -A[2]), d(A[1], A[2], None),
        ]
        eta = self.transport.eta
        if eta > 0.0:  # -curl(eta J) = -eta (k^2 B - k (k . B))
            k2 = g.k2()
            kb = g.kx3 * S[5] + g.ky3 * S[6] + g.kz3 * S[7]
            k[5] = k[5] - eta * (k2 * S[5] - g.kx3 * kb)
            k[6] = k[6] - eta * (k2 * S[6] - g.ky3 * kb)
            k[7] = k[7] - eta * (k2 * S[7] - g.kz3 * kb)
        return torch.stack(k)

    # --- transport (viscosity, resistivity, conduction) ---------------------

    def _stage_fluxes(self, Us: torch.Tensor, Sspec: torch.Tensor, count: bool, kernels: bool = True):
        """``(F, count or None, T spectrum or None)`` of a stage, transport terms included.

        ``Us`` is the full stage state (the out-of-band residual included),
        ``Sspec`` its retained spectrum (for the current).  With viscosity the
        velocity is transformed (3 forward), the trace-free strain (5, plus
        div u with bulk viscosity) and the current (3) are built on the retained
        modes and transformed back into the flux buffer, and one pass turns them
        into the 12 fluxes in place.  The T spectrum (normalized) carries the heat
        flux, added by ``_conduct``.  ``kernels=False`` keeps to torch ops.
        """
        tr = self.transport
        if not (self.mhd and tr.active):
            if kernels:
                F, c = self._fluxes(Us, True) if count else (self._fluxes(Us, False), None)
            else:
                F, c = self._fluxes_torch(Us), (self._count(Us) if count else None)
            return F, c, None
        g = self.grid
        if not tr.gradients:  # conduction alone: the ideal fluxes and T
            if kernels:
                F, c = self._fluxes(Us, True) if count else (self._fluxes(Us, False), None)
            else:
                F, c = self._fluxes_torch(Us), (self._count(Us) if count else None)
            T = mhd_temperature(Us, self.gamma).unsqueeze(0)
            return F, c, g.forward(T, normalized=True)[0]
        F = torch.empty((12,) + tuple(Us.shape[1:]), dtype=Us.dtype, device=Us.device)
        That = uh = None
        if tr.viscous or tr.conductive:
            # T (row 8) and the velocity (rows 9-11) are transformed first (flux_buffer_rows).
            if tr.viscous:
                torch.div(Us[1:4], Us[0], out=F[9:12])
            if tr.conductive:
                F[8] = mhd_temperature(Us, self.gamma)
            spec = g.forward(F[8 if tr.conductive else 9:12 if tr.viscous else 9], normalized=True)
            if tr.conductive:  # a copy, so the velocity spectra are freed once consumed
                That = spec[0].clone() if tr.viscous else spec[0]
            if tr.viscous:
                uh = spec[-3:]
            del spec
        # One inverse transform (one batch of transposes) for all the derivative fields.
        X = derivative_spectra(uh, Sspec[5:8] if tr.resistive else None, g.kx3, g.ky3, g.kz3, tr)
        del uh
        g.inverse(X, out=F[0:X.shape[0]])
        del X
        st = self._triton
        if kernels and st is not None:
            c = st.transport_fluxes(Us, F, self.gamma, self.rho_min, self.p_min, tr, count=count)
            return F, c, That
        for sl in self._planes(Us.shape[1]):  # in place, as the fused kernel: inputs read first
            F[:, sl] = mhd3d_transport_fluxes(Us[:, sl], F[:, sl], self.gamma, tr)
        return F, (self._count(Us) if count else None), That

    def _conduct(self, Fh: torch.Tensor, That: Optional[torch.Tensor], scale: float) -> None:
        """Heat flux ``-kappa grad T`` added to the energy-flux spectra (in place)."""
        if That is None:
            return
        g = self.grid
        c = -self.transport.kappa * scale
        Fh[6] += (c * 1j) * g.kx3 * That
        Fh[7] += (c * 1j) * g.ky3 * That
        Fh[8] += (c * 1j) * g.kz3 * That

    def _stage_rhs(self, Us: torch.Tensor, Sh: torch.Tensor, count: bool = False, kernels: bool = True):
        """Torch-tier spectral RHS of a stage and its floor count (or None)."""
        F, c, That = self._stage_fluxes(Us, Sh, count, kernels)
        Fh = self.grid.forward(F, normalized=True)
        del F
        self._conduct(Fh, That, 1.0)
        return self._rhs_hat(Fh, Sh), c

    def min_density(self) -> float:
        """Global minimum density of the current state (collective)."""
        return self.density_extrema()[0]

    def density_extrema(self) -> Tuple[float, float]:
        """Global minimum and maximum density of the current state (one collective)."""
        low, high = torch.aminmax(self.physical()[0])
        worst = self.comm.allreduce(torch.stack((-low, high)).tolist(), "max")
        return -float(worst[0]), float(worst[1])

    def dissipation_rates(self) -> Dict[str, float]:
        """Volume means of the viscous and Ohmic heating of the current state (collective).

        ``transport.heating``, summed a few derivative rows at a time so that the
        diagnostics fit beside the buffers of a step at the largest sizes.
        """
        tr = self.transport
        if not (self.mhd and tr.gradients):
            return {}
        g = self.grid
        U = self.physical()
        rho = U[0]
        uh = g.forward(U[1:4] / rho, normalized=True) if tr.viscous else None
        X = derivative_spectra(uh, self.uhat[5:8] if tr.resistive else None, g.kx3, g.ky3, g.kz3, tr)
        del uh
        rowD, rowT, rowJ, _ = flux_buffer_rows(tr)
        f64 = torch.float64
        viscous = torch.zeros((), dtype=f64, device=g.device)
        ohmic = torch.zeros((), dtype=f64, device=g.device)
        if tr.shear:  # 2 rho nu D:D with Dzz = -Dxx - Dyy
            P = g.inverse(X[rowD:rowD + 2])
            viscous += 2.0 * tr.nu * (rho * (P[0] * P[0] + P[1] * P[1] + (P[0] + P[1]) ** 2)).sum(dtype=f64)
            P = g.inverse(X[rowD + 2:rowD + 5])
            viscous += 4.0 * tr.nu * (rho * (P * P).sum(dim=0)).sum(dtype=f64)
            del P
        if tr.bulk:
            P = g.inverse(X[rowT:rowT + 1])[0]
            viscous += tr.nu_bulk * (rho * P * P).sum(dtype=f64)
            del P
        if tr.resistive:
            P = g.inverse(X[rowJ:rowJ + 3])
            ohmic += tr.eta * (P * P).sum(dtype=f64)
            del P
        del X
        mean = self.comm.allreduce(torch.stack((viscous, ohmic)).tolist(), "sum") / float(g.n_points)
        out = {}
        if tr.viscous:
            out["viscous_dissipation"] = float(mean[0])
        if tr.resistive:
            out["resistive_dissipation"] = float(mean[1])
        return out

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
        """Adopt a retained spectrum (normalized by 1/N) as the band-limited state."""
        self._has_state = True
        if uhat is not self.uhat:
            self.uhat.copy_(uhat)
        self._drop_residual()
        self.U = self.grid.inverse(self.uhat)
        self._pending = None
        self._refresh_stats("in the initial state")
        return self.U

    # --- out-of-band residual of a clamped state ------------------------------

    @property
    def residual(self) -> Optional[torch.Tensor]:
        """Part of the physical state above the dealias cutoff (rho, momentum, E), or ``None``."""
        return self._residual

    @property
    def band_limited(self) -> bool:
        """False while the state carries ``residual`` (the step after an accepted-state clamp)."""
        return self._residual is None

    def _drop_residual(self) -> None:
        self._residual = None
        self.grid.release_full()

    def _add_residual(self, U: torch.Tensor) -> torch.Tensor:
        """Add the residual to a physical state that was just transformed (in place)."""
        R = self._residual
        if R is not None:
            U[:R.shape[0]] += R
        return U

    def _inverse_full(self, S: torch.Tensor) -> torch.Tensor:
        """Physical state of a stage spectrum: ``inverse(S) + R``."""
        return self._add_residual(self.grid.inverse(S))

    def set_residual(self, residual: Optional[torch.Tensor]) -> None:
        """Adopt the out-of-band residual of a restart (after ``set_spectrum``); ``None`` clears it."""
        if residual is None:
            if self._residual is not None:
                self._drop_residual()
                self.U = None
                self._refresh_stats("after clearing the residual")
            return
        if tuple(residual.shape) != (5,) + self.grid.decomp.physical_shape:
            raise ValueError(f"residual must be {(5,) + self.grid.decomp.physical_shape}, "
                             f"got {tuple(residual.shape)}")
        if not self.mhd:
            raise ValueError("only the MHD accepted-state limiter leaves a residual")
        self._residual = residual.to(dtype=self.grid.rdtype, device=self.grid.device)
        self._pending = None
        self.U = None
        self.physical()
        self._refresh_stats("after restoring the residual")

    def _new_residual(self, Un: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """``Un - inverse(forward(Un))`` for a clamped state: its content above the cutoff."""
        return Un[:5] - self.grid.inverse(new[:5])

    def _refresh_stats(self, where: str) -> None:
        count, bad, speed = self._reduce_stats(*self._local_stats(self.U))
        self._require_finite(bad, where)
        self.speed = speed

    def physical(self) -> torch.Tensor:
        """This rank's slab of the conserved variables."""
        if self.U is None:
            self.U = self._inverse_full(self.uhat)
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
        if self._residual is not None:  # the reference damps the whole spectrum of a clamped state
            self._residual = g.dissipate_out_of_band(self._residual, h)
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
        self.U = self._inverse_full(self.uhat)

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
        self._drop_residual()
        self._pending = None

    def _judge(self, new_spec: torch.Tensor, stage_count, high: Optional[torch.Tensor]) -> bool:
        """Transform the candidate, reduce its checks and adopt it if no floor is hit."""
        g = self.grid
        Un = g.inverse(new_spec)
        count, bad, unusable, speed = self._local_stats(Un)
        count += float(stage_count)
        if high is not None and (self._residual is not None or (self.strict_clamp and self.dissipation)):
            # The reference clamps the final substage before projection and
            # dissipation; the residual of a clamped state is still part of it.
            count += float(self._count(self._inverse_full(high)))
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
            F, c, That = self._stage_fluxes(Us, Sh, i > 0)
            if c is not None:
                count = count + c
            if i == 0:
                self.U = None  # rebuilt from uhat if the step has to be repeated
            Us = None
            Fh = g.forward(F, normalized=True)
            del F
            self._conduct(Fh, That, 1.0)
            del That
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
            Us = self._inverse_full(Sh)
        return self._judge(self._stabilize(high, h), int(count.item()), high)

    def _rk_step_triton(self, dt: float, table, h: float) -> bool:
        g, st, modes = self.grid, self._triton, self._modes
        uhat, shat, unew = self.uhat, self.shat, self.unew
        if self._acc is None:
            self._acc = torch.empty((self.nvar, modes.total), dtype=g.cdtype, device=g.device)
        acc = self._acc
        self._pending = None
        head = [self.inv_n, self.transport.eta] if self.mhd else [self.inv_n]
        w = torch.tensor(head + [c * dt for wa, wn in table for c in (wa, 0.0 if wn is None else wn)],
                         dtype=g.rdtype, device=g.device)
        base = len(head)
        factors = g.decay_factors(h)
        count = torch.zeros((), dtype=torch.int64, device=g.device)
        Us, Sin = self.physical(), uhat
        last = len(table) - 1
        for i in range(len(table)):
            F, c, That = self._stage_fluxes(Us, Sin, i > 0)
            if c is not None:
                count += c
            if i == 0:
                self.U = None  # rebuilt from uhat if the step has to be repeated
            Us = None
            Fh = g.forward(F, normalized=False)
            del F
            self._conduct(Fh, That, 1.0 / self.inv_n)
            del That
            final = i == last
            launch = st.mhd_stage if self.mhd else st.stage
            extra = {"resistive": self.transport.resistive} if self.mhd else {}
            launch(Fh, Sin, unew if final else shat, uhat, acc, w, base + 2 * i, base + 1 + 2 * i,
                   self._waves, modes, first=(i == 0), final=final, high=shat, final_factors=factors,
                   **extra)
            del Fh
            if final:
                break
            Sin = shat
            Us = self._inverse_full(shat)
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
        U0h = self.uhat
        acc = torch.zeros_like(U0h)
        Sh, Us = U0h, self.physical()
        self.U = None
        high = None
        for wa, wn in table:
            k, _ = self._stage_rhs(Us, Sh, kernels=False)
            if wn is None:
                high = U0h + acc + (wa * dt) * k
                break
            acc = acc + (wa * dt) * k
            Sh = U0h + (wn * dt) * k
            Us, Sh, _ = self._limit(self._inverse_full(Sh), Sh)
        _, high, _ = self._limit(self._inverse_full(high), high)
        self._finish_limited(high, h)

    def _fluxes_torch(self, U: torch.Tensor) -> torch.Tensor:
        """The flux fields with torch ops, a few z planes at a time (the same values as one pass)."""
        fn = mhd3d_fluxes if self.mhd else euler3d_symmetric_fluxes
        F = torch.empty((self.nflux,) + tuple(U.shape[1:]), dtype=U.dtype, device=U.device)
        for sl in self._planes(U.shape[1]):
            F[:, sl] = fn(U[:, sl], self.gamma)
        return F

    def _finish_limited(self, high: torch.Tensor, h: float) -> None:
        g = self.grid
        new = self._stabilize(high, h)
        Un = g.inverse(new)
        clamped = False
        if self.mhd:  # only the MHD reference limits the accepted state
            Un, new, clamped = self._limit(Un, new)
        self.uhat.copy_(new)
        self.U = Un
        if clamped:  # Un is the clamped full state; keep what the retained modes cannot hold
            self._residual = self._new_residual(Un, new)
        else:
            self._drop_residual()
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
        self._backup = (self.uhat.clone(), self._residual) if self.dissipation else None
        self.dissipate(h)
        scale = self.comm.max(float(self.physical().abs().max().item()))
        return self.embedded_step(dt, a, b_high, b_low, h), scale

    def reject(self) -> None:
        """Discard the candidate and return to the state before ``attempt``."""
        self._pending = None
        backup = getattr(self, "_backup", None)
        if backup is not None:
            self.uhat.copy_(backup[0])
            self._residual = backup[1]
            self.U = None  # rebuilt from the restored spectrum and residual on demand
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
        spec, Un, speed, residual = self._pending
        if spec is self.shat:
            self.uhat, self.shat = self.shat, self.uhat
        else:
            self.uhat.copy_(spec)
        self.U, self.speed = Un, speed
        if residual is None:
            self._drop_residual()
        else:
            self._residual = residual
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
        self._pending = (spec, Un, speed, None)
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
                Us = self._inverse_full(Sh)
                if not self.mhd:  # Euler stages are limited in the reference; MHD ones are not
                    count = count + self._count(Us)
            ks.append(self._stage_rhs(Us, Sh)[0])
        Eh = sum(dt * float(b_high[i] - b_low[i]) * ks[i] for i in range(s) if b_high[i] != b_low[i])
        high = U0h + sum(dt * float(b_high[i]) * ks[i] for i in range(s) if b_high[i] != 0.0)
        del ks
        E = g.inverse(Eh)
        U_high = self._inverse_full(high)
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
        coefs = [self.inv_n, self.transport.eta] if self.mhd else [self.inv_n]
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
                Us = self._inverse_full(shat)
            F, c, That = self._stage_fluxes(Us, S, i > 0 and not self.mhd)
            if c is not None:
                count += c
            Us = None
            Fh = g.forward(F, normalized=False)
            del F
            self._conduct(Fh, That, 1.0 / self.inv_n)
            del That
            if self.mhd:
                st.mhd_stage(Fh, S, S, S, K[i], w, 0, 0, self._waves, modes, first=False,
                             final=False, rhs_only=True, resistive=self.transport.resistive)
            else:
                st.rhs(Fh, S, K[i], w, self._waves, modes)
            del Fh
        st.combine(K, s, None, shat, w, off_err, None, modes, masked=False)
        E = g.inverse(shat)
        st.combine(K, s, uhat, shat, w, off_new, None, modes, masked=False)
        U_high = self._inverse_full(shat)
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
                Us = self._inverse_full(Sh)
                if not self.mhd:
                    Us, Sh, _ = self._limit(Us, Sh)
            ks.append(self._stage_rhs(Us, Sh, kernels=False)[0])
        high = U0h + sum(dt * float(b_high[i]) * ks[i] for i in range(s) if b_high[i] != 0.0)
        if self.mhd:
            Eh = sum(dt * float(b_high[i] - b_low[i]) * ks[i] for i in range(s) if b_high[i] != b_low[i])
            err = float(g.inverse(Eh).abs().max().item())
            _, high, _ = self._limit(self._inverse_full(high), high)
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
        self._pending = (new, Un, speed, self._new_residual(Un, new) if clamped else None)
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
        if self._residual is not None:  # the out-of-band part rides along unchanged
            U[1:5] += self._residual[1:5]
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
        del rho, kin, mag, cross, p  # dissipation_rates needs the memory
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
        out.update(self.dissipation_rates())
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
