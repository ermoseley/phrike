"""1D Rayleigh-Taylor Instability problem following Athena++ conventions.

This implementation closely follows the standard RTI setup used in Athena++ with:
- Heavy fluid on top, light fluid on bottom (unstable configuration)
- Hydrostatic equilibrium initial conditions
- Small amplitude perturbations to trigger instability
- Dirichlet boundary conditions on both sides
- Compatible with both NumPy and PyTorch backends
"""

import os
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.solver import SpectralSolver1D
from phrike.visualization import plot_fields, plot_conserved_time_series
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def rti_1d_initial_conditions(
    x: np.ndarray,
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
    """Create 1D Rayleigh-Taylor instability initial conditions following Athena++ conventions.
    
    Standard RTI setup in 1D:
    - Heavy fluid (high density) on top, light fluid (low density) on bottom
    - Gravity points downward (negative y direction in 2D, but here we use x as vertical)
    - Hydrostatic equilibrium: dp/dx = -rho * g
    - Small amplitude perturbations at the interface
    
    Args:
        x: 1D coordinate array (N,)
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
        Conservative variables array U with shape (3, N)
    """
    # Support both NumPy arrays and Torch tensors
    try:
        import torch
        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False
        torch = None

    # Get domain dimensions
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        Xmin = float(torch.min(x).item())
        Xmax = float(torch.max(x).item())
    else:
        Xmin = float(np.min(x))
        Xmax = float(np.max(x))
    
    Lx = Xmax - Xmin
    
    # Interface location (typically at mid-height)
    x_interface = Xmin + 0.5 * Lx
    
    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        # Torch path
        # Create smooth density stratification using tanh
        # Heavy fluid on top (x > x_interface), light fluid on bottom (x < x_interface)
        x_centered = x - x_interface
        tanh_arg = x_centered / interface_thickness
        
        # Tensor constants for precision
        half = torch.tensor(0.5, dtype=x.dtype, device=x.device)
        one = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        
        # Density profile: rho = rho_light + 0.5 * (rho_heavy - rho_light) * (1 + tanh(x/delta))
        rho = rho_light + half * (rho_heavy - rho_light) * (one + torch.tanh(tanh_arg))
        
        # Initial velocity perturbations (small amplitude)
        # In 1D, we only have u (horizontal velocity), no vertical component
        u = perturb_amplitude * torch.sin(2.0 * np.pi * x / perturb_wavelength) * \
            torch.exp(-half * (x_centered / interface_thickness) ** 2)
        
        # Hydrostatic pressure integration: dp/dx = -rho * g
        # Integrate from top to bottom: p(x) = p_top - integral(rho * g * dx)
        Nx = x.shape[0]
        p = torch.zeros_like(x)
        g_tensor = torch.tensor(gravity, dtype=x.dtype, device=x.device)
        
        # Integrate pressure from top boundary downward
        x_line = x
        dx_line = x_line[1:] - x_line[:-1]  # Forward differences
        p_line = torch.zeros(Nx, dtype=x.dtype, device=x.device)
        p_line[-1] = pressure_top  # Start from top
        
        for i in range(Nx - 2, -1, -1):  # Integrate downward from top
            rho_mid = half * (rho[i] + rho[i + 1])
            p_line[i] = p_line[i + 1] - rho_mid * g_tensor * dx_line[i]
        
        p = p_line
        
    else:
        # NumPy path
        # Create smooth density stratification using tanh
        x_centered = x - x_interface
        tanh_arg = x_centered / interface_thickness
        
        # Density profile: heavy on top, light on bottom
        rho = rho_light + 0.5 * (rho_heavy - rho_light) * (1.0 + np.tanh(tanh_arg))
        
        # Initial velocity perturbations
        u = perturb_amplitude * np.sin(2.0 * np.pi * x / perturb_wavelength) * \
            np.exp(-0.5 * (x_centered / interface_thickness) ** 2)
        
        # Hydrostatic pressure integration: dp/dx = -rho * g
        Nx = x.shape[0]
        p = np.zeros_like(x)
        
        # Integrate pressure from top boundary downward
        x_line = x
        dx_line = np.diff(x_line)
        p_line = np.zeros(Nx, dtype=x.dtype)
        p_line[-1] = pressure_top  # Start from top
        
        for i in range(Nx - 2, -1, -1):  # Integrate downward from top
            rho_mid = 0.5 * (rho[i] + rho[i + 1])
            p_line[i] = p_line[i + 1] - rho_mid * gravity * dx_line[i]
        
        p = p_line

    # Convert to conservative variables
    eqs = EulerEquations1D(gamma=gamma)
    return eqs.conservative(rho, u, p)


class RTI1DProblem(BaseProblem):
    """1D Rayleigh-Taylor instability problem following Athena++ conventions."""

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid1D:
        """Create 1D grid for RTI problem with Dirichlet boundary conditions."""
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
        
        N = int(self.config["grid"]["N"])
        Lx = float(self.config["grid"]["Lx"])
        dealias = bool(self.config["grid"].get("dealias", True))
        fft_workers = int(self.config["grid"].get("fft_workers", 1))
        
        # Use Legendre basis for non-periodic boundary conditions
        basis = str(self.config["grid"].get("basis", "legendre")).lower()
        bc = self.config["grid"].get("bc", None)
        
        grid = Grid1D(
            N=N,
            Lx=Lx,
            basis=basis,
            bc=bc,
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
            problem_config=self.config,
        )

        return grid

    def create_equations(self) -> EulerEquations1D:
        """Create 1D Euler equations."""
        return EulerEquations1D(gamma=self.gamma)

    def apply_boundary_conditions(self, U: np.ndarray) -> np.ndarray:
        """Apply boundary conditions for RTI problem.
        
        With Legendre basis and Dirichlet BCs, the boundary conditions are handled
        naturally by the spectral method. No manual application needed.
        """
        return U

    def get_solver_class(self):
        """Get the solver class for this problem."""
        return SpectralSolver1D

    def create_initial_conditions(self, grid: Grid1D):
        """Create 1D RTI initial conditions following Athena++ conventions."""
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
            gravity = float(self.gravity_config.get("gx", 1.0))  # Use gx for 1D
        
        U = rti_1d_initial_conditions(
            x=grid.x,
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
        fig, axs = plt.subplots(3, 1, figsize=(10 * frame_scale, 12 * frame_scale), constrained_layout=True)
        fig.suptitle(f"1D RTI at t={t:.3f}", fontsize=14)
        
        x = solver.grid.x
        rho, u, p, _ = solver.equations.primitive(U)
        
        # Convert to numpy if needed
        x = self.convert_torch_to_numpy(x)
        rho = self.convert_torch_to_numpy(rho)
        u = self.convert_torch_to_numpy(u)
        p = self.convert_torch_to_numpy(p)
        
        # Density
        axs[0].plot(x, rho, 'b-', linewidth=2)
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("Density")
        axs[0].set_title("Density")
        axs[0].grid(True, alpha=0.3)
        
        # Velocity
        axs[1].plot(x, u, 'r-', linewidth=2)
        axs[1].set_xlabel("x")
        axs[1].set_ylabel("Velocity")
        axs[1].set_title("Velocity")
        axs[1].grid(True, alpha=0.3)
        
        # Pressure
        axs[2].plot(x, p, 'g-', linewidth=2)
        axs[2].set_xlabel("x")
        axs[2].set_ylabel("Pressure")
        axs[2].set_title("Pressure")
        axs[2].grid(True, alpha=0.3)
        
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

        # Final 1D visualization
        try:
            import matplotlib.pyplot as plt
            import numpy as _np

            x = solver.grid.x
            rho, u, p, _ = solver.equations.primitive(solver.U)

            # Convert to numpy if needed
            x = self.convert_torch_to_numpy(x)
            rho = self.convert_torch_to_numpy(rho)
            u = self.convert_torch_to_numpy(u)
            p = self.convert_torch_to_numpy(p)

            fig, axs = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True)
            fig.suptitle(f"Final 1D RTI State at t={solver.t:.3f}", fontsize=14)

            # Density
            axs[0].plot(x, rho, 'b-', linewidth=2)
            axs[0].set_xlabel("x")
            axs[0].set_ylabel("Density")
            axs[0].set_title("Density")
            axs[0].grid(True, alpha=0.3)

            # Velocity
            axs[1].plot(x, u, 'r-', linewidth=2)
            axs[1].set_xlabel("x")
            axs[1].set_ylabel("Velocity")
            axs[1].set_title("Velocity")
            axs[1].grid(True, alpha=0.3)

            # Pressure
            axs[2].plot(x, p, 'g-', linewidth=2)
            axs[2].set_xlabel("x")
            axs[2].set_ylabel("Pressure")
            axs[2].set_title("Pressure")
            axs[2].grid(True, alpha=0.3)

            final_plot_path = os.path.join(self.outdir, "final_state.png")
            fig.savefig(final_plot_path, dpi=150)
            plt.close(fig)
            print(f"Saved final visualization: {final_plot_path}")

        except Exception as e:
            print(f"Error creating final visualization: {e}")
