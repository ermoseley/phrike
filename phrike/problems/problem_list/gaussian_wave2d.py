"""2D Gaussian acoustic packet.

The packet travels along the configured angle. 45 degrees is left and up.
A Fourier grid is periodic. The analytic translate does not include wall
reflections on a Legendre grid.
"""

import os
import warnings
import numpy as np
from typing import Optional, Dict

import matplotlib.pyplot as plt

from phrike.grid import Grid2D
from phrike.equations import EulerEquations2D
from phrike.solver import SpectralSolver2D
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
        if basis_x == "fourier" and basis_y == "fourier":
            if str(bc).strip().lower() == "periodic":
                warnings.warn(
                    "bc: periodic is not accepted by Grid2D. A Fourier-Fourier "
                    "grid is already periodic, so this boundary name is ignored.",
                    UserWarning,
                    stacklevel=2,
                )
                bc = "dirichlet"

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

    def _packet_parameters(self) -> Dict[str, float]:
        """Shared centre, width and direction for the IC and the translate."""
        ic = self.config["initial_conditions"]
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        return {
            "rho0": float(ic.get("rho0", 1.0)),
            "u0": float(ic.get("u0", 0.0)),
            "v0": float(ic.get("v0", 0.0)),
            "p0": float(ic.get("p0", 1.0)),
            "amplitude": float(ic.get("amplitude", 0.1)),
            "sigma": float(ic.get("sigma", 0.05)),
            "x0": float(ic.get("x0", 0.5 * Lx)),
            "y0": float(ic.get("y0", 0.5 * Ly)),
            "angle": float(ic.get("angle", 45.0)),
        }

    def _direction(self, angle_deg: float):
        """Unit vector for ``angle`` degrees: 0 is -x, 90 is +y (left and up at 45)."""
        angle_rad = np.pi * float(angle_deg) / 180.0
        return -np.cos(angle_rad), np.sin(angle_rad)

    def _mesh(self, grid: Grid2D):
        """Return (X, Y) with shape (Ny, Nx), matching Grid2D.xy_mesh."""
        X, Y = grid.xy_mesh()
        try:
            import torch

            if isinstance(X, torch.Tensor):
                backend = {"dtype": X.dtype, "device": X.device}
                return X.detach().cpu().numpy(), Y.detach().cpu().numpy(), backend
        except Exception:
            pass
        return np.asarray(X), np.asarray(Y), None

    def _to_backend(self, array: np.ndarray, backend):
        if backend is None:
            return array
        import torch

        return torch.as_tensor(array, dtype=backend["dtype"], device=backend["device"])

    def _gaussian(self, X, Y, x0, y0, sigma, amplitude, t, u_wave, v_wave):
        """Packet translated by (u_wave, v_wave). Fourier axes use the nearest image."""
        x_rel = X - x0 - u_wave * t
        y_rel = Y - y0 - v_wave * t
        basis_x = str(self.config["grid"].get("basis_x", "legendre")).lower()
        basis_y = str(self.config["grid"].get("basis_y", "legendre")).lower()
        if basis_x == "fourier":
            Lx = float(self.config["grid"]["Lx"])
            x_rel = (x_rel + 0.5 * Lx) % Lx - 0.5 * Lx
        if basis_y == "fourier":
            Ly = float(self.config["grid"]["Ly"])
            y_rel = (y_rel + 0.5 * Ly) % Ly - 0.5 * Ly
        return amplitude * np.exp(-(x_rel**2 + y_rel**2) / (2.0 * sigma**2))

    def create_initial_conditions(self, grid: Grid2D):
        """Create a 2D acoustic Gaussian packet on the grid's (Ny, Nx) layout."""
        params = self._packet_parameters()
        rho0 = params["rho0"]
        u0 = params["u0"]
        v0 = params["v0"]
        p0 = params["p0"]
        X, Y, backend = self._mesh(grid)
        if X.shape != (grid.Ny, grid.Nx):
            raise ValueError(
                f"coordinate mesh has shape {X.shape}, expected {(grid.Ny, grid.Nx)}"
            )

        c_s = np.sqrt(self.gamma * p0 / rho0)
        nx, ny = self._direction(params["angle"])
        # angle 0 with a fluid at rest is the centred standing packet.
        if params["angle"] == 0.0 and u0 == 0.0 and v0 == 0.0:
            u_wave = 0.0
            v_wave = 0.0
        else:
            u_wave = c_s * nx
            v_wave = c_s * ny

        delta_rho = self._gaussian(
            X, Y, params["x0"], params["y0"], params["sigma"], params["amplitude"],
            0.0, u_wave, v_wave,
        )
        delta_p = c_s**2 * delta_rho
        delta_u = u_wave * delta_rho / rho0
        delta_v = v_wave * delta_rho / rho0

        rho = self._to_backend(rho0 + delta_rho, backend)
        u = self._to_backend(u0 + delta_u, backend)
        v = self._to_backend(v0 + delta_v, backend)
        p = self._to_backend(p0 + delta_p, backend)
        state = EulerEquations2D(gamma=self.gamma).conservative(rho, u, v, p)
        if tuple(state.shape[-2:]) != (grid.Ny, grid.Nx):
            raise ValueError(
                f"Gaussian wave state has shape {tuple(state.shape)}, "
                f"expected (*, {grid.Ny}, {grid.Nx})"
            )
        return state

    def get_analytical_solution(
        self, grid: Grid2D, t: float
    ) -> Optional[Dict[str, np.ndarray]]:
        """Linear translate of the packet. Wall reflections are not included.

        Fourier axes wrap to the nearest periodic image. A Legendre axis is
        translated in the open plane, so the reference is only meaningful
        before the packet reaches a wall.
        """
        params = self._packet_parameters()
        rho0 = params["rho0"]
        p0 = params["p0"]
        X, Y, _backend = self._mesh(grid)
        c_s = np.sqrt(self.gamma * p0 / rho0)
        nx, ny = self._direction(params["angle"])
        if params["angle"] == 0.0 and params["u0"] == 0.0 and params["v0"] == 0.0:
            u_wave = 0.0
            v_wave = 0.0
        else:
            u_wave = c_s * nx
            v_wave = c_s * ny
        delta_rho = self._gaussian(
            X, Y, params["x0"], params["y0"], params["sigma"], params["amplitude"],
            t, u_wave, v_wave,
        )
        return {
            "rho": rho0 + delta_rho,
            "u": params["u0"] + u_wave * delta_rho / rho0,
            "v": params["v0"] + v_wave * delta_rho / rho0,
            "p": p0 + c_s**2 * delta_rho,
        }

    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        # Save frame for video generation
        frames_dir = os.path.join(self.outdir, "frames")
        if os.path.exists(frames_dir):
            rho, u, v, p = self.convert_torch_to_numpy(
                *solver.equations.primitive(U)
            )
            X, Y, _backend = self._mesh(solver.grid)

            # Create frame plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"2D Gaussian Wave Packet at t={t:.3f}", fontsize=14)

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

        # Create visualization
        self._create_wave_analysis_plots(solver)

    def _create_wave_analysis_plots(self, solver) -> None:
        """Create analysis plots for the 2D wave packet."""
        rho, u, v, p = self.convert_torch_to_numpy(
            *solver.equations.primitive(solver.U)
        )
        X, Y, _backend = self._mesh(solver.grid)

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

        # Plot 5: Density along center lines. Arrays are (Ny, Nx).
        x_line = np.asarray(self.convert_torch_to_numpy(solver.grid.x)[0])
        y_line = np.asarray(self.convert_torch_to_numpy(solver.grid.y)[0])
        mid_y = rho.shape[0] // 2
        mid_x = rho.shape[1] // 2
        axes[1, 1].plot(x_line, rho[mid_y, :], 'b-', linewidth=2, label='y = mid')
        axes[1, 1].plot(y_line, rho[:, mid_x], 'r-', linewidth=2, label='x = mid')
        axes[1, 1].set_xlabel("Position")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Density Along Center Lines")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: velocity through the packet. Conservation history is returned
        # by solver.run and is not stored on the solver.
        axes[1, 2].plot(x_line, u[mid_y, :], 'b-', linewidth=2, label='u')
        axes[1, 2].plot(x_line, v[mid_y, :], 'r-', linewidth=2, label='v')
        axes[1, 2].set_xlabel("x")
        axes[1, 2].set_ylabel("Velocity")
        axes[1, 2].set_title("Velocity along y = mid")
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)

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
        params = self._packet_parameters()
        rho0 = params["rho0"]
        p0 = params["p0"]
        amplitude = params["amplitude"]
        sigma = params["sigma"]
        x0 = params["x0"]
        y0 = params["y0"]
        angle = params["angle"]

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
