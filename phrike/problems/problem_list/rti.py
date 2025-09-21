"""2D Rayleigh-Taylor Instability problem following Athena++ conventions.

This implementation closely follows the standard RTI setup used in Athena++ with:
- Heavy fluid on top, light fluid on bottom (unstable configuration)
- Hydrostatic equilibrium initial conditions
- Small amplitude perturbations to trigger instability
- Dirichlet boundary conditions on all four sides
- Compatible with both NumPy and PyTorch backends
"""

import os
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from phrike.grid import Grid2D
from phrike.equations import EulerEquations2D
from phrike.solver import SpectralSolver2D
from phrike.visualization import plot_fields, plot_conserved_time_series
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def rti_initial_conditions(
    x: np.ndarray,
    y: np.ndarray,
    rho_heavy: float,
    rho_light: float,
    pressure_top: float,
    gamma: float,
    gravity: float,
    interface_thickness: float,
    perturb_amplitude: float,
    perturb_wavelength: float,
    grid=None,
) -> np.ndarray:
    """Create Rayleigh-Taylor instability initial conditions following Athena++ conventions.
    
    Standard RTI setup:
    - Heavy fluid (high density) on top, light fluid (low density) on bottom
    - Gravity points downward (negative y direction)
    - Hydrostatic equilibrium: dp/dy = -rho * g
    - Small amplitude perturbations at the interface
    
    Args:
        x, y: 2D coordinate arrays (Ny, Nx)
        rho_heavy: Density of heavy fluid (top)
        rho_light: Density of light fluid (bottom)
        pressure_top: Pressure at top boundary
        gamma: Adiabatic index
        gravity: Gravitational acceleration (positive value, points downward)
        interface_thickness: Thickness of density transition layer
        perturb_amplitude: Amplitude of initial perturbation
        perturb_wavelength: Wavelength of perturbation
        grid: Grid object (unused but kept for compatibility)
        
    Returns:
        Conservative variables array U with shape (4, Ny, Nx)
    """
    # Support both NumPy arrays and Torch tensors
    try:
        import torch
        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False
        torch = None

    # Get domain dimensions
    if _TORCH_AVAILABLE and isinstance(y, torch.Tensor):
        Ymin = float(torch.min(y).item())
        Ymax = float(torch.max(y).item())
        Xmin = float(torch.min(x).item())
        Xmax = float(torch.max(x).item())
    else:
        Ymin = float(np.min(y))
        Ymax = float(np.max(y))
        Xmin = float(np.min(x))
        Xmax = float(np.max(x))
    
    Ly = Ymax - Ymin
    Lx = Xmax - Xmin
    
    # Interface location (typically at mid-height)
    y_interface = Ymin + 0.5 * Ly
    
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        # Torch path
        # Create smooth density stratification using tanh
        # Heavy fluid on top (y > y_interface), light fluid on bottom (y < y_interface)
        y_centered = y - y_interface
        tanh_arg = y_centered / interface_thickness
        
        # Tensor constants for precision
        half = torch.tensor(0.5, dtype=x.dtype, device=x.device)
        one = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        
        # Density profile: rho = rho_light + 0.5 * (rho_heavy - rho_light) * (1 + tanh(y/delta))
        rho = rho_light + half * (rho_heavy - rho_light) * (one + torch.tanh(tanh_arg))
        
        # Initial velocity perturbations (small amplitude)
        u = torch.zeros_like(x)
        two_pi = torch.tensor(2.0 * np.pi, dtype=x.dtype, device=x.device)
        v = perturb_amplitude * torch.sin(two_pi * x / perturb_wavelength) * \
            torch.exp(-half * (y_centered / interface_thickness) ** 2)
        
        # Hydrostatic pressure integration: dp/dy = -rho * g
        # Integrate from top to bottom: p(y) = p_top - integral(rho * g * dy)
        Ny, Nx = y.shape
        p = torch.zeros_like(x)
        g_tensor = torch.tensor(gravity, dtype=x.dtype, device=x.device)
        
        # Integrate pressure from top boundary downward
        y_line = y[:, 0]
        dy_line = y_line[1:] - y_line[:-1]  # Forward differences
        p_line = torch.zeros(Ny, dtype=x.dtype, device=x.device)
        p_line[-1] = pressure_top  # Start from top
        
        rho_line = rho[:, 0]
        for j in range(Ny - 2, -1, -1):  # Integrate downward from top
            rho_mid = half * (rho_line[j] + rho_line[j + 1])
            p_line[j] = p_line[j + 1] - rho_mid * g_tensor * dy_line[j]
        
        p = p_line[:, None].repeat(1, Nx)
        
    else:
        # NumPy path
        # Create smooth density stratification using tanh
        y_centered = y - y_interface
        tanh_arg = y_centered / interface_thickness
        
        # Density profile: heavy on top, light on bottom
        rho = rho_light + 0.5 * (rho_heavy - rho_light) * (1.0 + np.tanh(tanh_arg))
        
        # Initial velocity perturbations
        u = np.zeros_like(x)
        v = perturb_amplitude * np.sin(2.0 * np.pi * x / perturb_wavelength) * \
            np.exp(-0.5 * (y_centered / interface_thickness) ** 2)
        
        # Hydrostatic pressure integration: dp/dy = -rho * g
        Ny, Nx = y.shape
        p = np.zeros_like(x)
        
        # Integrate pressure from top boundary downward
        y_line = y[:, 0]
        dy_line = np.diff(y_line)
        p_line = np.zeros(Ny, dtype=x.dtype)
        p_line[-1] = pressure_top  # Start from top
        
        rho_line = rho[:, 0]
        for j in range(Ny - 2, -1, -1):  # Integrate downward from top
            rho_mid = 0.5 * (rho_line[j] + rho_line[j + 1])
            p_line[j] = p_line[j + 1] - rho_mid * gravity * dy_line[j]
        
        p = np.repeat(p_line[:, None], Nx, axis=1)

    # Convert to conservative variables
    eqs = EulerEquations2D(gamma=gamma)
    return eqs.conservative(rho, u, v, p)


class RTIProblem(BaseProblem):
    """2D Rayleigh-Taylor instability problem following Athena++ conventions."""

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid2D:
        """Create 2D grid for RTI problem with Dirichlet boundary conditions."""
        # Debug mode: validate backend and device availability
        if debug:
            if backend == "torch":
                try:
                    import torch
                    if device == "cuda" and not torch.cuda.is_available():
                        raise RuntimeError(f"Debug mode: CUDA requested but not available. PyTorch was not compiled with CUDA support.")
                    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                        raise RuntimeError(f"Debug mode: MPS requested but not available. MPS is only available on Apple Silicon Macs.")
                    elif device not in [None, "cpu", "cuda", "mps"]:
                        raise RuntimeError(f"Debug mode: Unknown device '{device}' requested. Valid devices are: cpu, cuda, mps")
                except ImportError:
                    raise RuntimeError(f"Debug mode: Torch backend requested but PyTorch is not installed.")
        
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        dealias = bool(self.config["grid"].get("dealias", True))
        fft_workers = int(self.config["grid"].get("fft_workers", 1))
        
        # Use Legendre basis for non-periodic boundary conditions
        basis_x = str(self.config["grid"].get("basis_x", "legendre")).lower()
        basis_y = str(self.config["grid"].get("basis_y", "legendre")).lower()

        # Dirichlet boundary conditions on all four sides
        bc = self.config["grid"].get("bc", "dirichlet")
        
        grid = Grid2D(
            Nx=Nx,
            Ny=Ny,
            Lx=Lx,
            Ly=Ly,
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
            basis_x=basis_x,
            basis_y=basis_y,
            bc=bc,
        )

        return grid

    def create_equations(self) -> EulerEquations2D:
        """Create 2D Euler equations."""
        return EulerEquations2D(gamma=self.gamma)

    def apply_boundary_conditions(self, U: np.ndarray) -> np.ndarray:
        """Apply boundary conditions for RTI problem.
        
        With Legendre basis and Dirichlet BCs, the boundary conditions are handled
        naturally by the spectral method. No manual application needed.
        """
        return U

    def get_solver_class(self):
        """Get the solver class for this problem."""
        return SpectralSolver2D

    def create_initial_conditions(self, grid: Grid2D):
        """Create RTI initial conditions following Athena++ conventions."""
        # Extract parameters from configuration
        ic_config = self.config["initial_conditions"]
        rho_heavy = float(ic_config.get("rho_heavy", 2.0))
        rho_light = float(ic_config.get("rho_light", 1.0))
        pressure_top = float(ic_config.get("pressure_top", 2.5))
        
        # Perturbation parameters
        perturb_amplitude = float(ic_config.get("perturb_amplitude", 0.01))
        perturb_wavelength = float(ic_config.get("perturb_wavelength", 1.0))
        interface_thickness = float(ic_config.get("interface_thickness", 0.05))
        
        # Gravity configuration
        gravity = 0.0
        if self.gravity_config and self.gravity_config.get("enabled", False):
            gravity = float(self.gravity_config.get("gy", 1.0))
        
        # Get coordinate arrays
        X, Y = grid.xy_mesh()
        
        U = rti_initial_conditions(
            x=X,
            y=Y,
            rho_heavy=rho_heavy,
            rho_light=rho_light,
            pressure_top=pressure_top,
            gamma=self.gamma,
            gravity=gravity,
            interface_thickness=interface_thickness,
            perturb_amplitude=perturb_amplitude,
            perturb_wavelength=perturb_wavelength,
            grid=grid,
        )
        
        return U

    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        frames_dir = os.path.join(self.outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get video configuration for frame settings
        video_config = self.config.get("video", {})
        frame_dpi = int(video_config.get("frame_dpi", 150))
        frame_scale = float(video_config.get("scale", 1.0))
        
        # Create frame plot
        fig, axs = plt.subplots(2, 2, figsize=(12 * frame_scale, 10 * frame_scale), constrained_layout=True)
        fig.suptitle(f"RTI at t={t:.3f}", fontsize=14)
        
        x, y = solver.grid.xy_mesh()
        rho, u, v, p = solver.equations.primitive(U)
        
        # Convert to numpy if needed
        x = self.convert_torch_to_numpy(x)[0]
        y = self.convert_torch_to_numpy(y)[0]
        rho = self.convert_torch_to_numpy(rho)[0]
        u = self.convert_torch_to_numpy(u)[0]
        v = self.convert_torch_to_numpy(v)[0]
        p = self.convert_torch_to_numpy(p)[0]
        
        # Density
        im1 = axs[0, 0].contourf(x, y, rho, levels=20, cmap='viridis')
        axs[0, 0].set_xlabel("x")
        axs[0, 0].set_ylabel("y")
        axs[0, 0].set_title("Density")
        axs[0, 0].set_aspect('equal')
        plt.colorbar(im1, ax=axs[0, 0])
        
        # Velocity magnitude
        vel_mag = np.sqrt(u**2 + v**2)
        im2 = axs[0, 1].contourf(x, y, vel_mag, levels=20, cmap='plasma')
        axs[0, 1].set_xlabel("x")
        axs[0, 1].set_ylabel("y")
        axs[0, 1].set_title("Velocity Magnitude")
        axs[0, 1].set_aspect('equal')
        plt.colorbar(im2, ax=axs[0, 1])
        
        # Pressure
        im3 = axs[1, 0].contourf(x, y, p, levels=20, cmap='coolwarm')
        axs[1, 0].set_xlabel("x")
        axs[1, 0].set_ylabel("y")
        axs[1, 0].set_title("Pressure")
        axs[1, 0].set_aspect('equal')
        plt.colorbar(im3, ax=axs[1, 0])
        
        # Velocity vectors (subsampled for clarity)
        skip = max(1, min(x.shape[0]//16, x.shape[1]//16))
        axs[1, 1].quiver(x[::skip, ::skip], y[::skip, ::skip], 
                        u[::skip, ::skip], v[::skip, ::skip], 
                        scale=50, alpha=0.7)
        axs[1, 1].set_xlabel("x")
        axs[1, 1].set_ylabel("y")
        axs[1, 1].set_title("Velocity Vectors")
        axs[1, 1].set_aspect('equal')
        
        # Save frame
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=frame_dpi)
        plt.close(fig)
        
        # Save snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir, t, U=U, grid=solver.grid, equations=solver.equations
        )
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

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

        # Final 2D visualization (contours), compatible with 2D fields
        try:
            import matplotlib.pyplot as plt
            import numpy as _np

            x, y = solver.grid.xy_mesh()
            rho, u, v, p = solver.equations.primitive(solver.U)

            # Convert to numpy if needed
            x = self.convert_torch_to_numpy(x)[0]
            y = self.convert_torch_to_numpy(y)[0]
            rho = self.convert_torch_to_numpy(rho)[0]
            u = self.convert_torch_to_numpy(u)[0]
            v = self.convert_torch_to_numpy(v)[0]
            p = self.convert_torch_to_numpy(p)[0]

            # Create figure and axes
            fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
            fig.suptitle(f"Final RTI State at t={solver.t:.3f}", fontsize=14)

            # Density
            im1 = axs[0, 0].contourf(x, y, rho, levels=20, cmap='viridis')
            axs[0, 0].set_xlabel("x")
            axs[0, 0].set_ylabel("y")
            axs[0, 0].set_title("Density")
            axs[0, 0].set_aspect('equal')
            plt.colorbar(im1, ax=axs[0, 0])

            # Velocity magnitude
            vel_mag = _np.sqrt(u**2 + v**2)
            im2 = axs[0, 1].contourf(x, y, vel_mag, levels=20, cmap='plasma')
            axs[0, 1].set_xlabel("x")
            axs[0, 1].set_ylabel("y")
            axs[0, 1].set_title("Velocity Magnitude")
            axs[0, 1].set_aspect('equal')
            plt.colorbar(im2, ax=axs[0, 1])

            # Pressure
            im3 = axs[1, 0].contourf(x, y, p, levels=20, cmap='coolwarm')
            axs[1, 0].set_xlabel("x")
            axs[1, 0].set_ylabel("y")
            axs[1, 0].set_title("Pressure")
            axs[1, 0].set_aspect('equal')
            plt.colorbar(im3, ax=axs[1, 0])

            # Velocity vectors (subsampled for clarity)
            skip = max(1, min(x.shape[0]//16, x.shape[1]//16))
            axs[1, 1].quiver(x[::skip, ::skip], y[::skip, ::skip], 
                            u[::skip, ::skip], v[::skip, ::skip], 
                            scale=50, alpha=0.7)
            axs[1, 1].set_xlabel("x")
            axs[1, 1].set_ylabel("y")
            axs[1, 1].set_title("Velocity Vectors")
            axs[1, 1].set_aspect('equal')

            final_plot_path = os.path.join(self.outdir, "final_state.png")
            fig.savefig(final_plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved final visualization: {final_plot_path}")

        except Exception as e:
            print(f"Error creating final visualization: {e}")