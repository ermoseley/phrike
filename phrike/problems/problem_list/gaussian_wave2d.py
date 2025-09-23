"""2D Gaussian wave packet problem for testing wave propagation and boundary reflections.

This problem implements a 2D Gaussian wave packet that:
- Starts at 2/3 of the way to the right and 2/3 of the way up
- Travels at 45° angle towards the left and up
- Reflects off walls with proper boundary conditions
- Tests 2D wave propagation, dispersion, and boundary condition handling
"""

import os
import numpy as np
from typing import Optional, Dict

import matplotlib.pyplot as plt

from phrike.grid import Grid2D
from phrike.equations import EulerEquations2D
from phrike.solver import SpectralSolver2D
from phrike.visualization import plot_fields, plot_conserved_time_series
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


class GaussianWave2DProblem(BaseProblem):
    """2D Gaussian wave packet problem for testing wave propagation and boundary reflections.

    This problem is designed to test:
    - 2D wave propagation accuracy
    - Boundary condition handling (reflections)
    - Dispersion and diffusion characteristics
    - Conservation properties in 2D
    - Long-time stability with reflections
    """

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid2D:
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
            Nx=Nx,
            Ny=Ny,
            Lx=Lx,
            Ly=Ly,
            basis_x=basis_x,
            basis_y=basis_y,
            bc=bc,
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
        )

    def create_equations(self) -> EulerEquations2D:
        """Create 2D Euler equations."""
        return EulerEquations2D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid2D):
        """Create 2D Gaussian wave packet initial conditions."""
        # Get parameters from config
        ic_config = self.config["initial_conditions"]
        rho0 = float(ic_config.get("rho0", 1.0))
        u0 = float(ic_config.get("u0", 0.0))
        v0 = float(ic_config.get("v0", 0.0))
        p0 = float(ic_config.get("p0", 1.0))
        amplitude = float(ic_config.get("amplitude", 0.1))  # Match 1D amplitude
        sigma = float(ic_config.get("sigma", 0.05))        # Match 1D sigma
        x0 = float(ic_config.get("x0", 0.5))               # Center of domain
        y0 = float(ic_config.get("y0", 0.5))               # Center of domain
        
        # Wave direction (45° towards left and up)
        angle = float(ic_config.get("angle", 45.0))  # degrees
        angle_rad = np.pi * angle / 180.0
        
        # Create 2D Gaussian perturbation using the same physical center and width across bases
        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
        
        # Adjust center and sigma for different coordinate systems
        if hasattr(grid, '_legendre_basis') and grid._legendre_basis is not None:
            # Legendre grid: use provided center as-is (domain is [0, Lx])
            center_x, center_y = x0, y0
            # Scale sigma to account for Legendre's coarser spacing in the center
            # Legendre spacing at center is ~1.58x larger than Fourier spacing
            effective_sigma = sigma * 1.58
        else:
            # Fourier grid: adjust center to account for endpoint=False
            # Fourier domain is [0, Lx*(N-1)/N], so center is at Lx*(N-1)/(2*N)
            center_x = grid.x.max() / 2.0
            center_y = grid.y.max() / 2.0
            effective_sigma = sigma
        
        gaussian = amplitude * np.exp(-((X - center_x)**2 + (Y - center_y)**2) / (2.0 * effective_sigma**2))

        # Calculate sound speed
        c_s = np.sqrt(self.gamma * p0 / rho0)
        
        # Check if we want zero initial velocity (for centered profile evolution)
        if angle == 0.0 and u0 == 0.0 and v0 == 0.0:
            # Zero initial velocity case - just density and pressure perturbations
            delta_rho = gaussian
            delta_p = c_s**2 * delta_rho
            delta_u = np.zeros_like(gaussian)  # Zero velocity perturbation
            delta_v = np.zeros_like(gaussian)  # Zero velocity perturbation
        else:
            # Wave propagation case - apply acoustic wave relations
            # Wave propagation direction (unit vector)
            nx = -np.cos(angle_rad)  # x-component (towards left)
            ny = np.sin(angle_rad)   # y-component (towards up)
            
            # Apply proper linear acoustic wave relations (same as 1D)
            # For a wave traveling in direction (nx, ny):
            # δp = c_s² δρ, δu = -c_s * nx * δρ/ρ₀, δv = -c_s * ny * δρ/ρ₀
            delta_rho = gaussian
            delta_p = c_s**2 * delta_rho
            delta_u = -c_s * nx * delta_rho / rho0  # Note: negative sign for proper acoustic relation
            delta_v = -c_s * ny * delta_rho / rho0  # Note: negative sign for proper acoustic relation

        rho = rho0 + delta_rho
        u = u0 + delta_u
        v = v0 + delta_v
        p = p0 + delta_p

        # Convert to conservative variables
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
        sigma = float(ic_config.get("sigma", 0.1))
        x0 = float(ic_config.get("x0", 2.0 * grid.Lx / 3.0))
        y0 = float(ic_config.get("y0", 2.0 * grid.Ly / 3.0))
        angle = float(ic_config.get("angle", 45.0))
        
        angle_rad = np.pi * angle / 180.0
        c_s = np.sqrt(self.gamma * p0 / rho0)
        
        # Wave velocity components
        u_wave = -c_s * np.cos(angle_rad)
        v_wave = c_s * np.sin(angle_rad)
        
        # Translate wave packet (ignoring boundary reflections for analytical)
        X, Y = np.meshgrid(grid.x, grid.y, indexing='ij')
        x_translated = X + u_wave * t
        y_translated = Y + v_wave * t
        
        gaussian = amplitude * np.exp(-((x_translated - x0)**2 + (y_translated - y0)**2) / (2.0 * sigma**2))
        
        # Apply linear acoustic relations
        delta_rho = gaussian
        delta_p = c_s**2 * delta_rho
        delta_u = u_wave * delta_rho / rho0
        delta_v = v_wave * delta_rho / rho0

        rho_analytical = rho0 + delta_rho
        u_analytical = u0 + delta_u
        v_analytical = v0 + delta_v
        p_analytical = p0 + delta_p

        return {"rho": rho_analytical, "u": u_analytical, "v": v_analytical, "p": p_analytical}

    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        # Save frame for video generation
        frames_dir = os.path.join(self.outdir, "frames")
        if os.path.exists(frames_dir):
            U_primitive = solver.equations.primitive(U)
            analytical = self.get_analytical_solution(solver.grid, t)

            # Create frame plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"2D Gaussian Wave Packet at t={t:.3f}", fontsize=14)

            X, Y = np.meshgrid(solver.grid.x, solver.grid.y, indexing='ij')
            rho, u, v, p = U_primitive

            # Density
            im1 = axes[0, 0].contourf(X, Y, rho, levels=20, cmap='viridis')
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("y")
            axes[0, 0].set_title("Density")
            axes[0, 0].set_aspect('equal')
            plt.colorbar(im1, ax=axes[0, 0])

            # Velocity magnitude
            v_mag = np.sqrt(u**2 + v**2)
            im2 = axes[0, 1].contourf(X, Y, v_mag, levels=20, cmap='plasma')
            axes[0, 1].set_xlabel("x")
            axes[0, 1].set_ylabel("y")
            axes[0, 1].set_title("Velocity Magnitude")
            axes[0, 1].set_aspect('equal')
            plt.colorbar(im2, ax=axes[0, 1])

            # Pressure
            im3 = axes[1, 0].contourf(X, Y, p, levels=20, cmap='coolwarm')
            axes[1, 0].set_xlabel("x")
            axes[1, 0].set_ylabel("y")
            axes[1, 0].set_title("Pressure")
            axes[1, 0].set_aspect('equal')
            plt.colorbar(im3, ax=axes[1, 0])

            # Velocity field
            skip = max(1, min(X.shape[0]//10, X.shape[1]//10))
            axes[1, 1].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                             u[::skip, ::skip], v[::skip, ::skip], 
                             v_mag[::skip, ::skip], cmap='plasma')
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("y")
            axes[1, 1].set_title("Velocity Field")
            axes[1, 1].set_aspect('equal')

            plt.tight_layout()

            # Save frame with timestamp-based naming
            frame_path = os.path.join(frames_dir, f"frame_{t:.6f}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches="tight")
            plt.close()

    def create_final_visualization(self, solver) -> None:
        """Create final visualization plots."""
        # Save final snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir,
            solver.t,
            U=solver.U,
            grid=solver.grid,
            equations=solver.equations,
        )
        print(f"Saved final snapshot: {snapshot_path}")

        # Create comprehensive visualization
        self._create_wave_analysis_plots(solver)

    def _create_wave_analysis_plots(self, solver) -> None:
        """Create comprehensive analysis plots for the 2D wave packet."""
        U_primitive = solver.equations.primitive(solver.U)
        X, Y = np.meshgrid(solver.grid.x, solver.grid.y, indexing='ij')
        rho, u, v, p = U_primitive

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f"2D Gaussian Wave Packet Analysis at t={solver.t:.3f}", fontsize=16)

        # Plot 1: Density contour
        im1 = axes[0, 0].contourf(X, Y, rho, levels=20, cmap='viridis')
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        axes[0, 0].set_title("Density")
        axes[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axes[0, 0])

        # Plot 2: Pressure contour
        im2 = axes[0, 1].contourf(X, Y, p, levels=20, cmap='coolwarm')
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")
        axes[0, 1].set_title("Pressure")
        axes[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axes[0, 1])

        # Plot 3: Velocity magnitude
        v_mag = np.sqrt(u**2 + v**2)
        im3 = axes[0, 2].contourf(X, Y, v_mag, levels=20, cmap='plasma')
        axes[0, 2].set_xlabel("x")
        axes[0, 2].set_ylabel("y")
        axes[0, 2].set_title("Velocity Magnitude")
        axes[0, 2].set_aspect('equal')
        plt.colorbar(im3, ax=axes[0, 2])

        # Plot 4: Velocity field
        skip = max(1, min(X.shape[0]//15, X.shape[1]//15))
        axes[1, 0].quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                         u[::skip, ::skip], v[::skip, ::skip], 
                         v_mag[::skip, ::skip], cmap='plasma')
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        axes[1, 0].set_title("Velocity Field")
        axes[1, 0].set_aspect('equal')

        # Plot 5: Density along center lines
        mid_x = X.shape[0] // 2
        mid_y = Y.shape[1] // 2
        axes[1, 1].plot(solver.grid.x, rho[mid_y, :], 'b-', linewidth=2, label='y-center')
        axes[1, 1].plot(solver.grid.y, rho[:, mid_x], 'r-', linewidth=2, label='x-center')
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Density Along Center Lines")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Conservation errors over time
        if hasattr(solver, "history") and solver.history and len(solver.history.get("time", [])) > 0:
            times = np.array(solver.history["time"])
            if "mass" in solver.history and "momentum" in solver.history and "energy" in solver.history:
                mass = np.array(solver.history["mass"])
                momentum = np.array(solver.history["momentum"])
                energy = np.array(solver.history["energy"])

                mass_error = np.abs(mass - mass[0]) / np.abs(mass[0])
                momentum_error = np.abs(momentum - momentum[0]) / (np.abs(momentum[0]) + 1e-16)
                energy_error = np.abs(energy - energy[0]) / np.abs(energy[0])

                axes[1, 2].semilogy(times, mass_error, label="Mass", marker="o", markersize=3)
                axes[1, 2].semilogy(times, momentum_error, label="Momentum", marker="s", markersize=3)
                axes[1, 2].semilogy(times, energy_error, label="Energy", marker="^", markersize=3)
                axes[1, 2].set_xlabel("Time")
                axes[1, 2].set_ylabel("Relative Error")
                axes[1, 2].set_title("Conservation Errors")
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        else:
            axes[1, 2].text(0.5, 0.5, "No history data", ha="center", va="center", 
                           transform=axes[1, 2].transAxes)
            axes[1, 2].set_title("Conservation Errors")

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.outdir, "gaussian_wave2d_analysis.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def get_solver_class(self):
        """Get 2D solver class."""
        return SpectralSolver2D

    def get_wave_parameters(self) -> Dict[str, float]:
        """Get wave packet parameters for analysis."""
        ic_config = self.config["initial_conditions"]
        rho0 = float(ic_config.get("rho0", 1.0))
        p0 = float(ic_config.get("p0", 1.0))
        amplitude = float(ic_config.get("amplitude", 0.1))
        sigma = float(ic_config.get("sigma", 0.1))
        x0 = float(ic_config.get("x0", 0.0))
        y0 = float(ic_config.get("y0", 0.0))
        angle = float(ic_config.get("angle", 45.0))

        # Get domain size from grid config
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])

        # Calculate derived parameters
        c_s = np.sqrt(self.gamma * p0 / rho0)  # Sound speed
        angle_rad = np.pi * angle / 180.0
        u_wave = -c_s * np.cos(angle_rad)
        v_wave = c_s * np.sin(angle_rad)
        
        # Time to reach boundaries
        time_to_left = (x0 - 0) / abs(u_wave) if u_wave < 0 else np.inf
        time_to_right = (Lx - x0) / u_wave if u_wave > 0 else np.inf
        time_to_bottom = (y0 - 0) / abs(v_wave) if v_wave < 0 else np.inf
        time_to_top = (Ly - y0) / v_wave if v_wave > 0 else np.inf

        return {
            "rho0": rho0,
            "p0": p0,
            "amplitude": amplitude,
            "sigma": sigma,
            "x0": x0,
            "y0": y0,
            "angle": angle,
            "sound_speed": c_s,
            "u_wave": u_wave,
            "v_wave": v_wave,
            "time_to_left": time_to_left,
            "time_to_right": time_to_right,
            "time_to_bottom": time_to_bottom,
            "time_to_top": time_to_top,
        }
