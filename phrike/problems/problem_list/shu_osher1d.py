"""
1D Shu-Osher shock-entropy interaction.

The post-shock state is the Mach-3 Rankine-Hugoniot state (u_post = u_L,
p_post = p_L). The pre-shock gas is an entropy wave at rest. There is no
closed-form solution once the shock enters that wave. A periodic grid adds a
second discontinuity at the seam; the shipped setup uses a non-periodic basis.

Reference: Shu, C.-W. & Osher, S. (1988).
"""

import warnings

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional
from ..base import BaseProblem
from phrike.equations import EulerEquations1D
from phrike.grid import Grid1D
from phrike.solver import SpectralSolver1D
from phrike.visualization import plot_fields
from phrike.io import save_solution_snapshot


class ShuOsher1DProblem(BaseProblem):
    """Shu-Osher shock-entropy interaction.

    The left state is the post-shock Rankine-Hugoniot state. Gamma is taken
    from the base class so ``equations.gamma`` and ``physics.gamma`` agree.
    """
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None, restart_from: Optional[str] = None):
        """Initialize the Shu-Osher 1D problem.
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary containing problem parameters
            restart_from: Path to restart file
        """
        super().__init__(config_path, config, restart_from)

        grid_cfg = self.config.get("grid", {})
        ic_cfg = self.config.get("initial_conditions", {})
        self.Lx = float(grid_cfg.get("Lx", 1.0))
        self.N = int(grid_cfg.get("N", 256))
        self.x_shock = float(ic_cfg.get("x_shock", 0.5))

        # Post-shock (left) state for a Mach-3 shock into (rho, u, p) = (1, 0, 1)
        # at gamma = 1.4. These are the Rankine-Hugoniot values: u_post = u_L.
        self.rho_L = float(ic_cfg.get("rho_L", 3.857143))
        self.u_L = float(ic_cfg.get("u_L", 2.629369))
        self.p_L = float(ic_cfg.get("p_L", 10.33333))

        self.rho_R = float(ic_cfg.get("rho_R", 1.0))
        self.u_R = float(ic_cfg.get("u_R", 0.0))
        self.p_R = float(ic_cfg.get("p_R", 1.0))
        self.rho_amplitude = float(ic_cfg.get("rho_amplitude", 0.2))
        self.rho_frequency = float(ic_cfg.get("rho_frequency", 2.0 * np.pi))

        self._compute_reference_solution()
    
    def _compute_reference_solution(self):
        """Shock speed from the Rankine-Hugoniot mass flux.

        The left state is already the post-shock state, so u_post = u_L and
        p_post = p_L. Setting u_post to the shock speed does not satisfy the
        jump conditions.
        """
        denominator = self.rho_L - self.rho_R
        if denominator == 0.0:
            raise ValueError("Shu-Osher left and right densities must differ")
        self.shock_speed = (
            self.rho_L * self.u_L - self.rho_R * self.u_R
        ) / denominator
        self.rho_post = self.rho_L
        self.u_post = self.u_L
        self.p_post = self.p_L
        self.c_post = np.sqrt(self.gamma * self.p_post / self.rho_post)
    
    def create_grid(self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False):
        """Create the computational grid with optional basis/BC from config."""
        grid_cfg = self.config.get("grid", {})
        N = int(grid_cfg.get("N", self.N))
        Lx = float(grid_cfg.get("Lx", self.Lx))
        dealias = bool(grid_cfg.get("dealias", True))
        basis = str(grid_cfg.get("basis", "fourier")).lower()
        bc = grid_cfg.get("bc", None)
        if basis == "fourier":
            warnings.warn(
                "Shu-Osher on a periodic Fourier grid has a second "
                "discontinuity where the post-shock state meets the "
                "pre-shock state at the domain seam. Use chebyshev or "
                "legendre with fixed endpoints.",
                UserWarning,
                stacklevel=2,
            )
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
        
        shock_mask = x < self.x_shock
        rho[shock_mask] = self.rho_L
        u[shock_mask] = self.u_L
        p[shock_mask] = self.p_L

        sinusoidal_mask = x >= self.x_shock
        rho[sinusoidal_mask] = self.rho_R + self.rho_amplitude * np.sin(self.rho_frequency * x[sinusoidal_mask])
        u[sinusoidal_mask] = self.u_R
        p[sinusoidal_mask] = self.p_R
        
        # Apply initial conditions smoothing if configured
        rho, u, p = self.apply_initial_conditions_smoothing(rho, u, p, grid)
        
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
        equations = self.create_equations()
        rho, u, p, _ = equations.primitive(U)
        x, rho, u, p = self.convert_torch_to_numpy(solver.grid.x, rho, u, p)

        fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

        axs[0, 0].plot(x, rho, 'b-', linewidth=2)
        axs[0, 0].set_title(f'Density at t = {t:.3f}')
        axs[0, 0].set_xlabel('x')
        axs[0, 0].set_ylabel('ρ')
        axs[0, 0].grid(True, alpha=0.3)

        axs[0, 1].plot(x, u, 'b-', linewidth=2)
        axs[0, 1].set_title(f'Velocity at t = {t:.3f}')
        axs[0, 1].set_xlabel('x')
        axs[0, 1].set_ylabel('u')
        axs[0, 1].grid(True, alpha=0.3)

        axs[1, 0].plot(x, p, 'b-', linewidth=2)
        axs[1, 0].set_title(f'Pressure at t = {t:.3f}')
        axs[1, 0].set_xlabel('x')
        axs[1, 0].set_ylabel('p')
        axs[1, 0].grid(True, alpha=0.3)

        # No closed-form reference: show the entropy-wave density ahead of
        # the shock at t = 0, which is stationary until the shock arrives.
        axs[1, 1].plot(x, rho, 'b-', linewidth=2)
        axs[1, 1].axvline(self.x_shock, color='k', linestyle='--', linewidth=1)
        axs[1, 1].set_title('Density (no closed-form reference)')
        axs[1, 1].set_xlabel('x')
        axs[1, 1].set_ylabel('ρ')
        axs[1, 1].grid(True, alpha=0.3)
        
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
        """Primitive ranges and conservation errors.

        ``compute_conservation_errors`` takes the state, not the time. There
        is no closed-form error to report for this shock-entropy interaction.
        """
        U = solver.U
        equations = self.create_equations()
        rho, u, p, _ = equations.primitive(U)
        rho, u, p = self.convert_torch_to_numpy(rho, u, p)
        cons_errors = self.compute_conservation_errors(solver, U)
        return {
            **cons_errors,
            "t": float(t),
            "rho_min": float(np.min(rho)),
            "rho_max": float(np.max(rho)),
            "u_min": float(np.min(u)),
            "u_max": float(np.max(u)),
            "p_min": float(np.min(p)),
            "p_max": float(np.max(p)),
            "shock_speed": float(self.shock_speed),
            "shock_position": float(self.x_shock + self.shock_speed * t),
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
