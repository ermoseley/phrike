"""1D Sod shock tube problem."""

import os
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt

from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.initial_conditions import sod_shock_tube
from phrike.solver import SpectralSolver1D
from phrike.visualization import plot_fields, plot_conserved_time_series
from phrike.io import save_solution_snapshot
from .base import BaseProblem


class SodProblem(BaseProblem):
    """1D Sod shock tube problem."""
    
    def create_grid(self, backend: str = "numpy", device: Optional[str] = None) -> Grid1D:
        """Create 1D grid."""
        N = int(self.config["grid"]["N"])
        Lx = float(self.config["grid"]["Lx"])
        dealias = bool(self.config["grid"].get("dealias", True))
        
        return Grid1D(
            N=N, 
            Lx=Lx, 
            dealias=dealias, 
            filter_params=self.filter_config, 
            fft_workers=self.fft_workers
        )
    
    def create_equations(self) -> EulerEquations1D:
        """Create 1D Euler equations."""
        return EulerEquations1D(gamma=self.gamma)
    
    def create_initial_conditions(self, grid: Grid1D):
        """Create Sod shock tube initial conditions."""
        x0 = float(self.config["grid"].get("x0", 0.5 * grid.Lx))
        left = self.config["initial_conditions"]["left"]
        right = self.config["initial_conditions"]["right"]
        
        return sod_shock_tube(grid.x, x0, left, right, self.gamma)
    
    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        # For 1D problems, we don't need frame-by-frame visualization
        # The final plot will be created in create_final_visualization
        pass
    
    def create_final_visualization(self, solver) -> None:
        """Create final visualization plots."""
        # Save final snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir, solver.t, U=solver.U, grid=solver.grid, equations=solver.equations
        )
        print(f"Saved final snapshot: {snapshot_path}")
        
        # Plot fields
        plot_fields(
            grid=solver.grid, 
            U=solver.U, 
            equations=solver.equations, 
            title=f"Sod at t={solver.t:.3f}", 
            outpath=os.path.join(self.outdir, f"fields_t{solver.t:.3f}.png")
        )
        
        # Plot conserved quantities if history is available
        if hasattr(solver, 'history') and solver.history and len(solver.history.get("time", [])) > 0:
            plot_conserved_time_series(
                solver.history, 
                outpath=os.path.join(self.outdir, "conserved.png")
            )
    
    def get_solver_class(self):
        """Get 1D solver class."""
        return SpectralSolver1D
