from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

from .grid import Grid1D, Grid2D, Grid3D
from .equations import EulerEquations1D, EulerEquations2D, EulerEquations3D


Array = np.ndarray


def _compute_rhs(grid: Grid1D, eqs: EulerEquations1D, U: Array) -> Array:
    # Pseudo-spectral: compute flux in physical space, then differentiate spectrally
    F = eqs.flux(U)
    # Batched spectral derivative across components (shape (3, N))
    dFdx = grid.dx1(F)
    # Euler in conservation form: dU/dt = - dF/dx
    return -dFdx


def _rk2_step(grid: Grid1D, eqs: EulerEquations1D, U: Array, dt: float) -> Array:
    k1 = _compute_rhs(grid, eqs, U)
    U1 = U + dt * 0.5 * k1
    k2 = _compute_rhs(grid, eqs, U1)
    return U + dt * k2


def _rk4_step(grid: Grid1D, eqs: EulerEquations1D, U: Array, dt: float) -> Array:
    k1 = _compute_rhs(grid, eqs, U)
    U2 = U + 0.5 * dt * k1
    k2 = _compute_rhs(grid, eqs, U2)
    U3 = U + 0.5 * dt * k2
    k3 = _compute_rhs(grid, eqs, U3)
    U4 = U + dt * k3
    k4 = _compute_rhs(grid, eqs, U4)
    return U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _apply_physical_filters(grid: Grid1D, U: Array) -> Array:
    # Apply optional spectral filter to each component to suppress Gibbs/aliasing
    return grid.apply_spectral_filter(U)


@dataclass
class SpectralSolver1D:
    grid: Grid1D
    equations: EulerEquations1D
    scheme: str = "rk4"  # "rk2" or "rk4"
    cfl: float = 0.4

    # Runtime state
    t: float = 0.0
    U: Optional[Array] = None

    def compute_dt(self, U: Array) -> float:
        max_speed = self.equations.max_wave_speed(U)
        # CFL for spectral (heuristic): dt <= cfl * dx / max_speed
        if max_speed <= 0.0:
            return 1e-6
        return self.cfl * self.grid.dx / max_speed

    def step(self, U: Array, dt: float) -> Array:
        if self.scheme.lower() == "rk2":
            Un = _rk2_step(self.grid, self.equations, U, dt)
        else:
            Un = _rk4_step(self.grid, self.equations, U, dt)
        Un = _apply_physical_filters(self.grid, Un)
        return Un

    def run(
        self,
        U0: Array,
        t0: float,
        t_end: float,
        output_interval: float = 0.05,
        checkpoint_interval: float = 0.0,
        outdir: Optional[str] = None,
        on_output: Optional[Callable[[float, Array], None]] = None,
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        self.U = U0.copy()
        self.t = float(t0)
        next_output = self.t + output_interval
        next_checkpoint = self.t + checkpoint_interval if checkpoint_interval and checkpoint_interval > 0 else np.inf

        history: Dict[str, List[float]] = {"time": [], "mass": [], "momentum": [], "energy": []}

        def record() -> None:
            cons = self.equations.conserved_quantities(self.U)  # type: ignore[arg-type]
            history["time"].append(self.t)
            history["mass"].append(cons["mass"])   # type: ignore[index]
            history["momentum"].append(cons["momentum"])  # type: ignore[index]
            history["energy"].append(cons["energy"])  # type: ignore[index]

        record()
        if on_output is not None:
            on_output(self.t, self.U)

        while self.t < t_end - 1e-12:
            dt = min(self.compute_dt(self.U), t_end - self.t)  # type: ignore[arg-type]
            self.U = self.step(self.U, dt)  # type: ignore[arg-type]
            self.t += dt

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(outdir, self.t, U=self.U, grid=self.grid, equations=self.equations)
                next_checkpoint += checkpoint_interval

        record()
        if on_output is not None:
            on_output(self.t, self.U)
        return history


# 2D solver

def _compute_rhs_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array) -> Array:
    # Fluxes in x and y
    Fx, Fy = eqs.flux(U)
    dFdx = grid.dx1(Fx)
    dFdy = grid.dy1(Fy)
    # dU/dt = -(dFx/dx + dFy/dy)
    return -(dFdx + dFdy)


def _rk2_step_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array, dt: float) -> Array:
    k1 = _compute_rhs_2d(grid, eqs, U)
    U1 = U + dt * 0.5 * k1
    k2 = _compute_rhs_2d(grid, eqs, U1)
    return U + dt * k2


def _rk4_step_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array, dt: float) -> Array:
    k1 = _compute_rhs_2d(grid, eqs, U)
    U2 = U + 0.5 * dt * k1
    k2 = _compute_rhs_2d(grid, eqs, U2)
    U3 = U + 0.5 * dt * k2
    k3 = _compute_rhs_2d(grid, eqs, U3)
    U4 = U + dt * k3
    k4 = _compute_rhs_2d(grid, eqs, U4)
    return U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _apply_physical_filters_2d(grid: Grid2D, U: Array) -> Array:
    return grid.apply_spectral_filter(U)


@dataclass
class SpectralSolver2D:
    grid: Grid2D
    equations: EulerEquations2D
    scheme: str = "rk4"
    cfl: float = 0.3

    t: float = 0.0
    U: Optional[Array] = None

    def compute_dt(self, U: Array) -> float:
        max_speed = self.equations.max_wave_speed(U)
        if max_speed <= 0.0:
            return 1e-6
        # CFL for 2D: use min(dx, dy)
        return self.cfl * min(self.grid.dx, self.grid.dy) / max_speed

    def step(self, U: Array, dt: float) -> Array:
        if self.scheme.lower() == "rk2":
            Un = _rk2_step_2d(self.grid, self.equations, U, dt)
        else:
            Un = _rk4_step_2d(self.grid, self.equations, U, dt)
        Un = _apply_physical_filters_2d(self.grid, Un)
        return Un

    def run(
        self,
        U0: Array,
        t0: float,
        t_end: float,
        output_interval: float = 0.1,
        checkpoint_interval: float = 0.0,
        outdir: Optional[str] = None,
        on_output: Optional[Callable[[float, Array], None]] = None,
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        self.t = float(t0)
        next_output = self.t + output_interval
        next_checkpoint = self.t + checkpoint_interval if checkpoint_interval and checkpoint_interval > 0 else np.inf

        history: Dict[str, List[float]] = {"time": [], "mass": [], "momentum_x": [], "momentum_y": [], "energy": []}

        def record() -> None:
            rho, ux, uy, p = self.equations.primitive(self.U)  # type: ignore[arg-type]
            if _TORCH_AVAILABLE and isinstance(rho, (torch.Tensor,)):
                mass = float(torch.sum(rho).item())
                momx = float(torch.sum(rho * ux).item())
                momy = float(torch.sum(rho * uy).item())
                energy = float(torch.sum(p / (self.equations.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy)).item())
            else:
                mass = float(np.sum(rho))
                momx = float(np.sum(rho * ux))
                momy = float(np.sum(rho * uy))
                energy = float(np.sum(p / (self.equations.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy)))
            history["time"].append(self.t)
            history["mass"].append(mass)
            history["momentum_x"].append(momx)
            history["momentum_y"].append(momy)
            history["energy"].append(energy)

        record()
        if on_output is not None:
            on_output(self.t, self.U)

        while self.t < t_end - 1e-12:
            dt = min(self.compute_dt(self.U), t_end - self.t)  # type: ignore[arg-type]
            self.U = self.step(self.U, dt)  # type: ignore[arg-type]
            self.t += dt

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(outdir, self.t, U=self.U, grid=self.grid, equations=self.equations)
                next_checkpoint += checkpoint_interval

        record()
        if on_output is not None:
            on_output(self.t, self.U)
        return history


def _compute_rhs_3d(grid: Grid3D, eqs: EulerEquations3D, U: Array) -> Array:
    Fx, Fy, Fz = eqs.flux(U)
    dFdx = grid.dx1(Fx)
    dFdy = grid.dy1(Fy)
    dFdz = grid.dz1(Fz)
    return -(dFdx + dFdy + dFdz)


def _rk2_step_3d(grid: Grid3D, eqs: EulerEquations3D, U: Array, dt: float) -> Array:
    k1 = _compute_rhs_3d(grid, eqs, U)
    U1 = U + dt * 0.5 * k1
    k2 = _compute_rhs_3d(grid, eqs, U1)
    return U + dt * k2


def _rk4_step_3d(grid: Grid3D, eqs: EulerEquations3D, U: Array, dt: float) -> Array:
    k1 = _compute_rhs_3d(grid, eqs, U)
    U2 = U + 0.5 * dt * k1
    k2 = _compute_rhs_3d(grid, eqs, U2)
    U3 = U + 0.5 * dt * k2
    k3 = _compute_rhs_3d(grid, eqs, U3)
    U4 = U + dt * k3
    k4 = _compute_rhs_3d(grid, eqs, U4)
    return U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _apply_physical_filters_3d(grid: Grid3D, U: Array) -> Array:
    return grid.apply_spectral_filter(U)


@dataclass
class SpectralSolver3D:
    grid: Grid3D
    equations: EulerEquations3D
    scheme: str = "rk4"
    cfl: float = 0.25

    t: float = 0.0
    U: Optional[Array] = None

    def compute_dt(self, U: Array) -> float:
        max_speed = self.equations.max_wave_speed(U)
        if max_speed <= 0.0:
            return 1e-6
        return self.cfl * min(self.grid.dx, self.grid.dy, self.grid.dz) / max_speed

    def step(self, U: Array, dt: float) -> Array:
        if self.scheme.lower() == "rk2":
            Un = _rk2_step_3d(self.grid, self.equations, U, dt)
        else:
            Un = _rk4_step_3d(self.grid, self.equations, U, dt)
        Un = _apply_physical_filters_3d(self.grid, Un)
        return Un

    def run(
        self,
        U0: Array,
        t0: float,
        t_end: float,
        output_interval: float = 0.2,
        checkpoint_interval: float = 0.0,
        outdir: Optional[str] = None,
        on_output: Optional[Callable[[float, Array], None]] = None,
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        try:
            import torch  # type: ignore
            _TORCH_AVAILABLE = True
        except Exception:
            _TORCH_AVAILABLE = False
            torch = None  # type: ignore

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        self.t = float(t0)
        next_output = self.t + output_interval
        next_checkpoint = self.t + checkpoint_interval if checkpoint_interval and checkpoint_interval > 0 else np.inf

        history: Dict[str, List[float]] = {"time": [], "mass": [], "momentum_x": [], "momentum_y": [], "momentum_z": [], "energy": []}

        def record() -> None:
            rho, ux, uy, uz, p = self.equations.primitive(self.U)  # type: ignore[arg-type]
            if _TORCH_AVAILABLE and isinstance(rho, (torch.Tensor,)):
                mass = float(torch.sum(rho).item())
                momx = float(torch.sum(rho * ux).item())
                momy = float(torch.sum(rho * uy).item())
                momz = float(torch.sum(rho * uz).item())
                energy = float(torch.sum(p / (self.equations.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy + uz * uz)).item())
            else:
                mass = float(np.sum(rho))
                momx = float(np.sum(rho * ux))
                momy = float(np.sum(rho * uy))
                momz = float(np.sum(rho * uz))
                energy = float(np.sum(p / (self.equations.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy + uz * uz)))
            history["time"].append(self.t)
            history["mass"].append(mass)
            history["momentum_x"].append(momx)
            history["momentum_y"].append(momy)
            history["momentum_z"].append(momz)
            history["energy"].append(energy)

        record()
        if on_output is not None:
            on_output(self.t, self.U)

        while self.t < t_end - 1e-12:
            dt = min(self.compute_dt(self.U), t_end - self.t)  # type: ignore[arg-type]
            self.U = self.step(self.U, dt)  # type: ignore[arg-type]
            self.t += dt

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(outdir, self.t, U=self.U, grid=self.grid, equations=self.equations)
                next_checkpoint += checkpoint_interval

        record()
        if on_output is not None:
            on_output(self.t, self.U)
        return history

