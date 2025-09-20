"""2D Rayleigh-Taylor instability problem with gravity and mixed boundary conditions."""

import os
from typing import Optional, Dict

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
    rho_bottom: float,
    rho_top: float,
    pressure_bottom: float,
    gamma: float,
    shear_thickness: float,
    perturb_eps: float,
    perturb_sigma: float,
    perturb_kx: int,
    perturb_ky: int,
    gy: float,
) -> np.ndarray:
    """Create Rayleigh-Taylor instability initial conditions with smooth density transition.
    
    Args:
        x, y: 2D coordinate arrays (Ny, Nx)
        rho_bottom: Density at bottom of box
        rho_top: Density at top of box
        pressure_bottom: Pressure at bottom of box
        gamma: Adiabatic index
        shear_thickness: Thickness of density transition layer
        perturb_eps: Amplitude of initial perturbation
        perturb_sigma: Width of perturbation
        perturb_kx, perturb_ky: Wavenumbers for perturbation
        
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

    # Create smooth density stratification with tanh transition centered at mid-height
    # Heavy-over-light for RTI: rho increases with height if rho_top > rho_bottom
    Ymin = float(np.min(y))
    Ymax = float(np.max(y))
    Ly = Ymax - Ymin
    y0 = Ymin + 0.5 * Ly
    delta = max(1e-6, shear_thickness * Ly)

    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        # Torch path
        y_centered = y - y0
        tanh_arg = y_centered / delta
        rho = rho_bottom + 0.5 * (rho_top - rho_bottom) * (1.0 + torch.tanh(tanh_arg))
        
        # Velocity perturbation: vertical component localized near interface
        u = torch.zeros_like(x)
        v = perturb_eps * torch.sin(perturb_kx * 2.0 * np.pi * x / (x.max() - x.min() + 1e-12)) * \
            torch.exp(-0.5 * ((y_centered) / max(1e-12, perturb_sigma * Ly)) ** 2)
        
        # Hydrostatic pressure integration: dp/dy = rho * gy
        # Work along y for each x column using trapezoidal rule
        Ny, Nx = y.shape
        p = torch.zeros_like(x)
        # Build base y-line and dy (uniform assumed)
        y_line = y[:, 0]
        dy_line = y_line[1:] - y_line[:-1]
        p_line = torch.zeros(Ny, dtype=x.dtype, device=x.device)
        p_line[0] = float(pressure_bottom)
        rho_line = rho[:, 0]
        for j in range(1, Ny):
            rho_mid = 0.5 * (rho_line[j - 1] + rho_line[j])
            p_line[j] = p_line[j - 1] + rho_mid * float(gy) * dy_line[j - 1]
        p = p_line[:, None].repeat(1, Nx)
    else:
        # Numpy path
        y_centered = y - y0
        tanh_arg = y_centered / delta
        rho = rho_bottom + 0.5 * (rho_top - rho_bottom) * (1.0 + np.tanh(tanh_arg))
        
        # Velocity perturbation
        u = np.zeros_like(x)
        v = perturb_eps * np.sin(perturb_kx * 2.0 * np.pi * x / (np.max(x) - np.min(x) + 1e-12)) * \
            np.exp(-0.5 * ((y_centered) / max(1e-12, perturb_sigma * Ly)) ** 2)
        
        # Hydrostatic pressure integration: dp/dy = rho * gy
        Ny, Nx = y.shape
        y_line = y[:, 0]
        dy_line = np.diff(y_line)
        p_line = np.zeros(Ny, dtype=x.dtype)
        p_line[0] = float(pressure_bottom)
        rho_line = rho[:, 0]
        for j in range(1, Ny):
            rho_mid = 0.5 * (rho_line[j - 1] + rho_line[j])
            p_line[j] = p_line[j - 1] + rho_mid * float(gy) * dy_line[j - 1]
        p = np.repeat(p_line[:, None], Nx, axis=1)

    # Convert to conservative variables
    eqs = EulerEquations2D(gamma=gamma)
    return eqs.conservative(rho, u, v, p)


class RTIProblem(BaseProblem):
    """2D Rayleigh-Taylor instability problem with gravity and mixed boundary conditions."""

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid2D:
        """Create 2D grid for RTI problem."""
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
        # Basis selection (default to Legendre for RTI)
        basis_x = str(self.config["grid"].get("basis_x", "legendre")).lower()
        basis_y = str(self.config["grid"].get("basis_y", "legendre")).lower()

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
        )

        # With Legendre basis in y, boundary conditions are naturally enforced
        # No manual BC application needed

        return grid

    def create_equations(self) -> EulerEquations2D:
        """Create 2D Euler equations."""
        return EulerEquations2D(gamma=self.gamma)

    def apply_boundary_conditions(self, U: np.ndarray) -> np.ndarray:
        """Apply boundary conditions for RTI problem.
        
        With Legendre basis in y, boundary conditions are naturally enforced.
        This method is kept for compatibility but does nothing.
        """
        return U

    def create_initial_conditions(self, grid: Grid2D):
        """Create Rayleigh-Taylor instability initial conditions."""
        # Get configuration parameters
        ic_config = self.config["initial_conditions"]
        rho_bottom = float(ic_config["rho_bottom"])
        rho_top = float(ic_config["rho_top"])
        pressure_bottom = float(ic_config["pressure_bottom"])
        shear_thickness = float(ic_config.get("shear_thickness", 0.02))
        perturb_eps = float(ic_config.get("perturb_eps", 0.01))
        perturb_sigma = float(ic_config.get("perturb_sigma", 0.02))
        perturb_kx = int(ic_config.get("perturb_kx", 2))
        perturb_ky = int(ic_config.get("perturb_ky", 1))

        X, Y = grid.xy_mesh()
        # Gravity from config
        gy = 0.0
        if self.gravity_config and self.gravity_config.get("enabled", False):
            gy = float(self.gravity_config.get("gy", 0.0))
        U = rti_initial_conditions(
            x=X,
            y=Y,
            rho_bottom=rho_bottom,
            rho_top=rho_top,
            pressure_bottom=pressure_bottom,
            gamma=self.gamma,
            shear_thickness=shear_thickness,
            perturb_eps=perturb_eps,
            perturb_sigma=perturb_sigma,
            perturb_kx=perturb_kx,
            perturb_ky=perturb_ky,
            gy=gy,
        )
        
        # With Legendre basis, boundary conditions are naturally enforced
        # No manual BC application needed
        
        return U

    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        # With Legendre basis, boundary conditions are naturally enforced
        # No manual BC application needed
        
        # Save frame for video generation
        frames_dir = os.path.join(self.outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get video configuration for frame settings
        video_config = self.config.get("video", {})
        frame_dpi = int(video_config.get("frame_dpi", 150))
        frame_scale = float(video_config.get("scale", 1.0))
        
        # Calculate figure size based on scale
        base_size = 12
        figsize = (base_size * frame_scale, 8 * frame_scale)
        
        # Create frame plot
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        fig.suptitle(f"Rayleigh-Taylor Instability at t={t:.3f}", fontsize=14)
        
        x, y = solver.grid.xy_mesh()
        rho, u, v, p, _ = solver.equations.primitive(U)
        
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

        # Plot fields
        plot_fields(
            grid=solver.grid,
            U=solver.U,
            equations=solver.equations,
            title=f"Rayleigh-Taylor Instability at t={solver.t:.3f}",
            outpath=os.path.join(self.outdir, f"fields_t{solver.t:.3f}.png"),
        )

        # Plot conserved quantities if history is available
        if (
            hasattr(solver, "history")
            and solver.history
            and len(solver.history.get("time", [])) > 0
        ):
            plot_conserved_time_series(
                solver.history, outpath=os.path.join(self.outdir, "conserved.png")
            )

    def get_solver_class(self):
        """Get 2D solver class."""
        return SpectralSolver2D
