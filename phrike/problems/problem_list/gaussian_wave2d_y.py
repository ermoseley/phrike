"""
2D Gaussian Wave Packet Problem - Y-direction only
A 1D Gaussian wave packet copied along x-direction, traveling purely in y-direction.
This tests the 2D Legendre implementation with a problem that should behave identically to 1D.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional

from phrike.problems.base import BaseProblem
from phrike.grid import Grid2D
from phrike.equations import EulerEquations2D


class GaussianWave2DYProblem(BaseProblem):
    """2D Gaussian wave packet problem - traveling in y-direction only."""

    def __init__(self, config: Dict = None, config_path: str = None, restart_from: str = None):
        super().__init__(config=config, config_path=config_path, restart_from=restart_from)
        self.gamma = float(self.config["physics"]["gamma"])

    def create_grid(self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False) -> Grid2D:
        """Create 2D grid with Legendre basis."""
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        dealias = bool(self.config["grid"].get("dealias", True))
        basis_x = str(self.config["grid"].get("basis_x", "legendre")).lower()
        basis_y = str(self.config["grid"].get("basis_y", "legendre")).lower()
        bc = self.config["grid"].get("bc", "dirichlet")

        return Grid2D(
            Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly,
            basis_x=basis_x, basis_y=basis_y, bc=bc,
            dealias=dealias, filter_params=self.filter_config,
            fft_workers=self.fft_workers, backend=backend,
            torch_device=device, precision=self.precision,
        )

    def create_equations(self) -> EulerEquations2D:
        """Create 2D Euler equations."""
        return EulerEquations2D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid2D):
        """Create 2D Gaussian wave packet initial conditions - y-direction only.
        
        This creates a 1D Gaussian wave packet that is copied along the x-direction,
        so d/dx = 0 for all variables. The wave travels purely in the y-direction.
        """
        # Get parameters from config
        ic_config = self.config["initial_conditions"]
        rho0 = float(ic_config.get("rho0", 1.0))
        u0 = float(ic_config.get("u0", 0.0))
        v0 = float(ic_config.get("v0", 0.0))
        p0 = float(ic_config.get("p0", 1.0))
        amplitude = float(ic_config.get("amplitude", 0.1))  # Match 1D amplitude
        sigma = float(ic_config.get("sigma", 0.05))        # Match 1D sigma
        y0 = float(ic_config.get("y0", 0.5))               # Center of domain in y
        direction = ic_config.get("direction", "up")       # "up" or "down"

        x, y = grid.x, grid.y
        X, Y = np.meshgrid(x, y, indexing='ij')

        # Create 1D Gaussian perturbation in y-direction only
        gaussian_y = amplitude * np.exp(-((y - y0) ** 2) / (2.0 * sigma**2))
        
        # Broadcast to 2D (constant along x)
        gaussian_2d = np.tile(gaussian_y, (len(x), 1)).T

        # Calculate sound speed
        c_s = np.sqrt(self.gamma * p0 / rho0)
        
        # Wave propagation direction in y
        if direction == "up":
            ny = 1.0   # positive y direction
        else:
            ny = -1.0  # negative y direction
        
        # Apply proper linear acoustic wave relations (same as 1D)
        # For a wave traveling in y-direction:
        # δp = c_s² δρ, δu = 0, δv = -c_s * ny * δρ/ρ₀
        delta_rho = gaussian_2d
        delta_p = c_s**2 * delta_rho
        delta_u = np.zeros_like(delta_rho)  # No x-velocity perturbation
        delta_v = -c_s * ny * delta_rho / rho0  # y-velocity perturbation

        # Set up the fields
        rho = rho0 + delta_rho
        u = u0 + delta_u  # No x-velocity
        v = v0 + delta_v  # y-velocity perturbation
        p = p0 + delta_p

        equations = EulerEquations2D(gamma=self.gamma)
        return equations.conservative(rho, u, v, p)

    def get_analytical_solution(
        self, grid: Grid2D, t: float
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get analytical solution at time t (without reflections for simplicity)."""
        ic_config = self.config["initial_conditions"]
        rho0 = float(ic_config.get("rho0", 1.0))
        u0 = float(ic_config.get("u0", 0.0))
        v0 = float(ic_config.get("v0", 0.0))
        p0 = float(ic_config.get("p0", 1.0))
        amplitude = float(ic_config.get("amplitude", 0.1))
        sigma = float(ic_config.get("sigma", 0.05))
        y0 = float(ic_config.get("y0", 0.5))
        direction = ic_config.get("direction", "up")
        
        c_s = np.sqrt(self.gamma * p0 / rho0)
        y = grid.y
        
        # Wave propagation direction
        if direction == "up":
            ny = 1.0
        else:
            ny = -1.0
        
        # Translate the wave packet
        y_translated = y + c_s * ny * t
        gaussian_y = amplitude * np.exp(-((y_translated - y0) ** 2) / (2.0 * sigma**2))
        
        # Broadcast to 2D
        gaussian_2d = np.tile(gaussian_y, (len(grid.x), 1)).T
        
        # Apply acoustic relations
        delta_rho = gaussian_2d
        delta_p = c_s**2 * delta_rho
        delta_u = np.zeros_like(delta_rho)
        delta_v = -c_s * ny * delta_rho / rho0

        rho_analytical = rho0 + delta_rho
        u_analytical = u0 + delta_u
        v_analytical = v0 + delta_v
        p_analytical = p0 + delta_p

        equations = EulerEquations2D(gamma=self.gamma)
        return equations.conservative(rho_analytical, u_analytical, v_analytical, p_analytical)

    def create_visualization(self, grid: Grid2D, U: np.ndarray, t: float) -> None:
        """Create visualization of the 2D Gaussian wave packet."""
        equations = EulerEquations2D(gamma=self.gamma)
        rho, u, v, p = equations.primitive(U)
        
        x, y = grid.x, grid.y
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'2D Gaussian Wave Packet (Y-direction) at t={t:.3f}', fontsize=14)
        
        # Density
        im1 = axes[0, 0].imshow(rho, extent=[x.min(), x.max(), y.min(), y.max()], 
                               origin='lower', cmap='viridis', aspect='equal')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        axes[0, 0].set_title('Density')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Pressure
        im2 = axes[0, 1].imshow(p, extent=[x.min(), x.max(), y.min(), y.max()], 
                               origin='lower', cmap='coolwarm', aspect='equal')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        axes[0, 1].set_title('Pressure')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Velocity magnitude
        v_mag = np.sqrt(u**2 + v**2)
        im3 = axes[1, 0].imshow(v_mag, extent=[x.min(), x.max(), y.min(), y.max()], 
                               origin='lower', cmap='plasma', aspect='equal')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        axes[1, 0].set_title('Velocity Magnitude')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # Y-velocity
        im4 = axes[1, 1].imshow(v, extent=[x.min(), x.max(), y.min(), y.max()], 
                               origin='lower', cmap='RdBu_r', aspect='equal')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        axes[1, 1].set_title('Y-Velocity')
        plt.colorbar(im4, ax=axes[1, 1])
        
        plt.tight_layout()
        return fig

    def create_final_visualization(self, solver) -> None:
        """Create final visualization with analysis."""
        return self.create_visualization(solver.grid, solver.U, solver.t)

    def get_solver_class(self):
        """Get the solver class for this problem."""
        from phrike.solver import SpectralSolver2D
        return SpectralSolver2D
