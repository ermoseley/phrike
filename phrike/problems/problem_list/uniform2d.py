"""Uniform 2D box test: constant rho, p and zero velocity."""

from typing import Optional
import numpy as np

from phrike.grid import Grid2D
from phrike.equations import EulerEquations2D
from phrike.solver import SpectralSolver2D
from ..base import BaseProblem
from phrike.io import save_solution_snapshot
import matplotlib.pyplot as plt


def uniform2d_initial_conditions(x: np.ndarray, y: np.ndarray, rho0: float, p0: float, gamma: float):
    """Create uniform fields, torch-aware if x/y are tensors."""
    try:
        import torch  # type: ignore
        is_torch = isinstance(x, torch.Tensor)
    except Exception:
        torch = None  # type: ignore
        is_torch = False

    if is_torch:
        u0 = torch.zeros_like(x)
        v0 = torch.zeros_like(x)
        rho = torch.full_like(x, float(rho0))
        p = torch.full_like(x, float(p0))
    else:
        u0 = np.zeros_like(x)
        v0 = np.zeros_like(x)
        rho = np.full_like(x, rho0)
        p = np.full_like(x, p0)

    eqs = EulerEquations2D(gamma=gamma)
    return eqs.conservative(rho, u0, v0, p)


class Uniform2DProblem(BaseProblem):
    """Uniform 2D field with Dirichlet BCs on all sides, gravity off."""

    def create_grid(self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False) -> Grid2D:
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        basis_x = str(self.config["grid"].get("basis_x", "legendre")).lower()
        basis_y = str(self.config["grid"].get("basis_y", "legendre")).lower()
        bc = "dirichlet"

        return Grid2D(
            Nx=Nx,
            Ny=Ny,
            Lx=Lx,
            Ly=Ly,
            dealias=False,
            filter_params=self.filter_config,
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
            basis_x=basis_x,
            basis_y=basis_y,
            bc=bc,
        )

    def create_equations(self) -> EulerEquations2D:
        return EulerEquations2D(gamma=self.gamma)

    def get_solver_class(self):
        return SpectralSolver2D

    def create_initial_conditions(self, grid: Grid2D):
        X, Y = grid.xy_mesh()
        ic = self.config.get("initial_conditions", {})
        rho0 = float(ic.get("rho", 1.0))
        p0 = float(ic.get("p", 1.0))
        return uniform2d_initial_conditions(X, Y, rho0, p0, self.gamma)

    def apply_boundary_conditions(self, U):
        # Rely on Grid2D BCs (Dirichlet)
        return U

    def create_visualization(self, solver, t: float, U):
        # Minimal: save snapshot; optional quick diagnostic lineout of rho mean
        try:
            save_solution_snapshot(self.outdir, t, U=U, grid=solver.grid, equations=solver.equations)
        except Exception:
            pass

    def create_final_visualization(self, solver) -> None:
        try:
            save_solution_snapshot(self.outdir, solver.t, U=solver.U, grid=solver.grid, equations=solver.equations)
        except Exception:
            pass

