"""2D Kelvin-Helmholtz instability problem."""

import os
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from phrike.grid import Grid2D
from phrike.equations import EulerEquations2D
from phrike.initial_conditions import kelvin_helmholtz_2d
from phrike.solver import SpectralSolver2D
from phrike.io import save_solution_snapshot
from .base import BaseProblem


class KHI2DProblem(BaseProblem):
    """2D Kelvin-Helmholtz instability problem."""
    
    def create_grid(self, backend: str = "numpy", device: Optional[str] = None) -> Grid2D:
        """Create 2D grid."""
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        dealias = bool(self.config["grid"].get("dealias", True))
        
        return Grid2D(
            Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, 
            dealias=dealias, 
            filter_params=self.filter_config, 
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device
        )
    
    def create_equations(self) -> EulerEquations2D:
        """Create 2D Euler equations."""
        return EulerEquations2D(gamma=self.gamma)
    
    def create_initial_conditions(self, grid: Grid2D):
        """Create KHI initial conditions."""
        X, Y = grid.xy_mesh()
        icfg = self.config["initial_conditions"]
        
        rho_outer = float(icfg.get("rho_outer", 1.0))
        rho_inner = float(icfg.get("rho_inner", 2.0))
        u0 = float(icfg.get("u0", 1.0))
        shear_thickness = float(icfg.get("shear_thickness", 0.02))
        pressure_outer = float(icfg.get("pressure_outer", 1.0))
        pressure_inner = float(icfg.get("pressure_inner", 1.0))
        perturb_eps = float(icfg.get("perturb_eps", 0.01))
        perturb_sigma = float(icfg.get("perturb_sigma", 0.02))
        perturb_kx = int(icfg.get("perturb_kx", 2))
        
        return kelvin_helmholtz_2d(
            X, Y, 
            rho_outer=rho_outer, rho_inner=rho_inner,
            u0=u0, shear_thickness=shear_thickness,
            pressure_outer=pressure_outer, pressure_inner=pressure_inner,
            perturb_eps=perturb_eps, perturb_sigma=perturb_sigma,
            perturb_kx=perturb_kx, gamma=self.gamma
        )
    
    def create_visualization(self, solver, t: float, U) -> None:
        """Create visualization for current state."""
        rho, ux, uy, p = solver.equations.primitive(U)
        rho = self.convert_torch_to_numpy(rho)[0]
        
        # Create frame
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        ax.imshow(rho, origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], aspect='equal')
        ax.set_title(f"Density t={t:.3f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Save frame
        frames_dir = os.path.join(self.outdir, "frames")
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=120)
        plt.close(fig)
        
        # Save snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir, t, U=U, grid=solver.grid, equations=solver.equations
        )
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")
    
    def create_final_visualization(self, solver) -> None:
        """Create final visualization plots."""
        rho, ux, uy, p = solver.equations.primitive(solver.U)
        rho = self.convert_torch_to_numpy(rho)[0]
        
        # Final density plot
        plt.figure(figsize=(8, 8))
        plt.imshow(rho, origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], aspect='equal')
        plt.title(f"Density t={solver.t:.3f}")
        plt.xlabel('x')
        plt.ylabel('y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"khi2d_t{solver.t:.3f}.png"), dpi=150)
        plt.close()
    
    def get_solver_class(self):
        """Get 2D solver class."""
        return SpectralSolver2D
