"""
1D Periodic Shu-Osher Test Problem for Compressible Euler Equations

This problem tests shock-capturing and wave propagation in a high-order solver.
The initial condition consists of a shock region (x < 0.5) and a sinusoidal
density region (x >= 0.5) with periodic boundary conditions.

Reference: Shu, C.-W. & Osher, S. (1988) "Efficient implementation of essentially
non-oscillatory shock-capturing schemes"
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Tuple
from phrike.problems.base import BaseProblem
from phrike.equations import EulerEquations1D
from phrike.grid import Grid1D
from phrike.solver import SpectralSolver1D
from phrike.visualization import plot_fields
from phrike.io import save_solution_snapshot


class ShuOsher1DProblem(BaseProblem):
    """1D Periodic Shu-Osher Test Problem for Compressible Euler Equations.
    
    This problem tests shock-capturing and wave propagation capabilities.
    The initial condition features a shock region and a sinusoidal density
    region with periodic boundary conditions.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, restart_from: Optional[str] = None):
        """Initialize the Shu-Osher 1D problem.
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary containing problem parameters
            restart_from: Path to restart file
        """
        super().__init__(config_path, config, restart_from)
        
        # Problem parameters
        self.gamma = self.config.get("physics", {}).get("gamma", 1.4)
        self.Lx = self.config.get("grid", {}).get("Lx", 1.0)  # Domain length
        self.N = self.config.get("grid", {}).get("N", 256)    # Grid resolution
        
        # Shock region parameters (x < 0.5)
        self.rho_L = 3.857143
        self.u_L = 2.629369
        self.p_L = 10.33333
        
        # Sinusoidal region parameters (x >= 0.5)
        self.rho_R = 1.0
        self.u_R = 0.0
        self.p_R = 1.0
        self.rho_amplitude = 0.2
        self.rho_frequency = 2.0 * np.pi
        
        # Compute reference solution parameters
        self._compute_reference_solution()
    
    def _compute_reference_solution(self):
        """Compute reference solution parameters using Rankine-Hugoniot conditions."""
        # Shock speed from Rankine-Hugoniot condition
        # S = (rho_L * u_L - rho_R * u_R) / (rho_L - rho_R)
        self.shock_speed = (self.rho_L * self.u_L - self.rho_R * self.u_R) / (self.rho_L - self.rho_R)
        
        # Post-shock state (right side of shock)
        # Using Rankine-Hugoniot jump conditions
        self.rho_post = self.rho_R * (self.shock_speed - self.u_R) / (self.shock_speed - self.u_L)
        self.u_post = self.shock_speed
        self.p_post = self.p_R + self.rho_R * (self.shock_speed - self.u_R) * (self.u_post - self.u_R)
        
        # Sound speed in post-shock region
        self.c_post = np.sqrt(self.gamma * self.p_post / self.rho_post)
    
    def create_grid(self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False):
        """Create the computational grid with optional basis/BC from config."""
        grid_cfg = self.config.get("grid", {})
        N = int(grid_cfg.get("N", self.N))
        Lx = float(grid_cfg.get("Lx", self.Lx))
        dealias = bool(grid_cfg.get("dealias", True))
        basis = str(grid_cfg.get("basis", "fourier")).lower()
        bc = grid_cfg.get("bc", None)
        return Grid1D(
            N=N,
            Lx=Lx,
            basis=basis,
            bc=bc,
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
        )
    
    def create_equations(self):
        """Create the equation system.
        
        Returns:
            EulerEquations1D instance
        """
        return EulerEquations1D(gamma=self.gamma)
    
    def get_solver_class(self):
        """Get the appropriate solver class for this problem.
        
        Returns:
            SpectralSolver1D class
        """
        return SpectralSolver1D
    
    def create_initial_conditions(self, grid) -> np.ndarray:
        """Create initial conditions for the Shu-Osher problem.
        
        Args:
            grid: Grid instance
            
        Returns:
            Initial conservative variables U = [rho, rho*u, E]
        """
        x = grid.x
        
        # Initialize arrays
        rho = np.zeros_like(x)
        u = np.zeros_like(x)
        p = np.zeros_like(x)
        
        # Shock region (x < 0.5)
        shock_mask = x < 0.5
        rho[shock_mask] = self.rho_L
        u[shock_mask] = self.u_L
        p[shock_mask] = self.p_L
        
        # Sinusoidal region (x >= 0.5)
        sinusoidal_mask = x >= 0.5
        rho[sinusoidal_mask] = self.rho_R + self.rho_amplitude * np.sin(self.rho_frequency * x[sinusoidal_mask])
        u[sinusoidal_mask] = self.u_R
        p[sinusoidal_mask] = self.p_R
        
        # Apply initial conditions smoothing if configured
        rho, u, p = self.apply_initial_conditions_smoothing(rho, u, p, grid)
        
        # Convert to conservative variables
        equations = self.create_equations()
        return equations.conservative(rho, u, p)
    
    def get_analytical_solution(self, t: float, grid) -> np.ndarray:
        """Compute analytical solution at time t.
        
        Args:
            t: Time at which to compute the solution
            grid: Grid instance
            
        Returns:
            Analytical conservative variables U = [rho, rho*u, E]
        """
        x = grid.x
        
        # Initialize arrays
        rho = np.zeros_like(x)
        u = np.zeros_like(x)
        p = np.zeros_like(x)
        
        # Shock position at time t
        shock_position = 0.5 + self.shock_speed * t
        
        # Apply periodic boundary conditions to shock position
        shock_position = shock_position % self.Lx
        
        # Left side of shock (post-shock state)
        left_mask = x < shock_position
        rho[left_mask] = self.rho_post
        u[left_mask] = self.u_post
        p[left_mask] = self.p_post
        
        # Right side of shock (sinusoidal region)
        right_mask = x >= shock_position
        
        # Advect sinusoidal profile along characteristics
        # For the right-going characteristic: x - c*t = constant
        # For the left-going characteristic: x + c*t = constant
        c_R = np.sqrt(self.gamma * self.p_R / self.rho_R)
        
        # Characteristic coordinates
        x_char = x[right_mask] - c_R * t
        
        # Apply periodic boundary conditions to characteristic coordinates
        x_char = x_char % self.Lx
        
        # Sinusoidal density profile
        rho[right_mask] = self.rho_R + self.rho_amplitude * np.sin(self.rho_frequency * x_char)
        u[right_mask] = self.u_R
        p[right_mask] = self.p_R
        
        # Convert to conservative variables
        equations = self.create_equations()
        return equations.conservative(rho, u, p)
    
    def create_visualization(self, solver, t: float, U) -> None:
        """Create visualization of the solution.
        
        Args:
            solver: The solver containing the current solution
            t: Current time
            U: Current solution state
        """
        # Get current solution
        U = solver.U
        
        # Convert to primitive variables
        equations = self.create_equations()
        rho, u, p, _ = equations.primitive(U)
        
        # Get analytical solution
        U_analytical = self.get_analytical_solution(t, solver.grid)
        rho_analytical, u_analytical, p_analytical, _ = equations.primitive(U_analytical)
        
        # Create plots
        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        
        # Density
        axs[0, 0].plot(solver.grid.x, rho, 'b-', label='Numerical', linewidth=2)
        axs[0, 0].plot(solver.grid.x, rho_analytical, 'r--', label='Analytical', linewidth=2)
        axs[0, 0].set_title(f'Density at t = {t:.3f}')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('ρ')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)
        
        # Velocity
        axs[0, 1].plot(solver.grid.x, u, 'b-', label='Numerical', linewidth=2)
        axs[0, 1].plot(solver.grid.x, u_analytical, 'r--', label='Analytical', linewidth=2)
        axs[0, 1].set_title(f'Velocity at t = {t:.3f}')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('u')
        axs[0, 1].legend()
        axs[0, 1].grid(True, alpha=0.3)
        
        # Pressure
        axs[1, 0].plot(solver.grid.x, p, 'b-', label='Numerical', linewidth=2)
        axs[1, 0].plot(solver.grid.x, p_analytical, 'r--', label='Analytical', linewidth=2)
        axs[1, 0].set_title(f'Pressure at t = {t:.3f}')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('p')
        axs[1, 0].legend()
        axs[1, 0].grid(True, alpha=0.3)
        
        # Error
        error = np.abs(rho - rho_analytical)
        axs[1, 1].plot(solver.grid.x, error, 'g-', linewidth=2)
        axs[1, 1].set_title(f'Density Error at t = {t:.3f}')
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('|ρ - ρ_analytical|')
        axs[1, 1].grid(True, alpha=0.3)
        axs[1, 1].set_yscale('log')
        
        plt.suptitle(f'Shu-Osher 1D Problem - t = {t:.3f}')
        plt.tight_layout()
        
        # Save frame to the correct output directory
        import os
        frames_dir = os.path.join(self.outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        frame_path = os.path.join(frames_dir, f"frame_{t:08.3f}.png")
        plt.savefig(frame_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def compute_diagnostics(self, solver: SpectralSolver1D, t: float) -> Dict[str, float]:
        """Compute diagnostic quantities.
        
        Args:
            solver: The solver containing the current solution
            t: Current time
            
        Returns:
            Dictionary of diagnostic quantities
        """
        U = solver.U
        equations = self.create_equations()
        rho, u, p, _ = equations.primitive(U)
        
        # Conservation errors
        cons_errors = self.compute_conservation_errors(solver, t)
        
        # L2 and L∞ errors against analytical solution
        U_analytical = self.get_analytical_solution(t, solver.grid)
        rho_analytical, u_analytical, p_analytical, _ = equations.primitive(U_analytical)
        
        # L2 errors
        l2_rho = np.sqrt(np.mean((rho - rho_analytical)**2))
        l2_u = np.sqrt(np.mean((u - u_analytical)**2))
        l2_p = np.sqrt(np.mean((p - p_analytical)**2))
        
        # L∞ errors
        linf_rho = np.max(np.abs(rho - rho_analytical))
        linf_u = np.max(np.abs(u - u_analytical))
        linf_p = np.max(np.abs(p - p_analytical))
        
        return {
            **cons_errors,
            "l2_rho_error": l2_rho,
            "l2_u_error": l2_u,
            "l2_p_error": l2_p,
            "linf_rho_error": linf_rho,
            "linf_u_error": linf_u,
            "linf_p_error": linf_p,
            "shock_position": (0.5 + self.shock_speed * t) % self.Lx,
            "shock_speed": self.shock_speed
        }
    
    def get_wave_parameters(self) -> Dict[str, float]:
        """Get wave parameters for the problem.
        
        Returns:
            Dictionary of wave parameters
        """
        return {
            "gamma": self.gamma,
            "shock_speed": self.shock_speed,
            "rho_L": self.rho_L,
            "u_L": self.u_L,
            "p_L": self.p_L,
            "rho_R": self.rho_R,
            "u_R": self.u_R,
            "p_R": self.p_R,
            "rho_post": self.rho_post,
            "u_post": self.u_post,
            "p_post": self.p_post,
            "c_post": self.c_post
        }
