"""1D Hydrostatic Equilibrium test problem."""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

from phrike.equations import EulerEquations1D
from phrike.grid import Grid1D
from phrike.visualization import plot_fields, plot_conserved_time_series
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def hse_initial_conditions(
    x: np.ndarray,
    rho_left: float,
    rho_right: float,
    pressure_left: float,
    gamma: float,
    gx: float,
    x0: float,
    sigma: float = 0.01,
) -> np.ndarray:
    """Create hydrostatic equilibrium initial conditions.
    
    Args:
        x: 1D coordinate array
        rho_left: Density on the left side
        rho_right: Density on the right side
        pressure_left: Pressure at left boundary
        gamma: Adiabatic index
        gx: Gravity in x-direction (negative for gravity pointing left)
        x0: Discontinuity location
        sigma: Smoothing parameter for tanh transition
        
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

    if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
        # Torch path
        xl = x
        # Create smooth density transition
        tanh_step = 0.5 * (1.0 + torch.tanh((x0 - xl) / sigma))
        rho = float(rho_right) + (float(rho_left) - float(rho_right)) * tanh_step
        u = torch.zeros_like(xl)
        
        # Hydrostatic pressure integration: dp/dx = rho * gx
        # Simple trapezoidal integration
        N = len(xl)
        p = torch.zeros_like(xl)
        p[0] = float(pressure_left)
        
        # Get grid spacing (assuming uniform)
        dx = xl[1] - xl[0]
        gx_tensor = torch.tensor(float(gx), dtype=xl.dtype, device=xl.device)
        
        for j in range(1, N):
            rho_mid = 0.5 * (rho[j - 1] + rho[j])
            p[j] = p[j - 1] + rho_mid * gx_tensor * dx
            
    else:
        # NumPy path
        x_arr = np.asarray(x)
        # Create smooth density transition
        tanh_step = 0.5 * (1.0 + np.tanh((x0 - x_arr) / sigma))
        rho = float(rho_right) + (float(rho_left) - float(rho_right)) * tanh_step
        u = np.zeros_like(x_arr)
        
        # Hydrostatic pressure integration: dp/dx = rho * gx
        # Simple trapezoidal integration
        N = len(x_arr)
        p = np.zeros_like(x_arr)
        p[0] = float(pressure_left)
        
        # Get grid spacing (assuming uniform)
        dx = x_arr[1] - x_arr[0]
        
        for j in range(1, N):
            rho_mid = 0.5 * (rho[j - 1] + rho[j])
            p[j] = p[j - 1] + rho_mid * float(gx) * dx

    # Convert to conservative variables
    eq = EulerEquations1D(gamma=gamma)
    return eq.conservative(rho, u, p)


class HSE1DProblem(BaseProblem):
    """1D Hydrostatic Equilibrium test problem."""

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid1D:
        """Create 1D grid."""
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
        basis = str(self.config["grid"].get("basis", "fourier")).lower()
        bc = self.config["grid"].get("bc", None)

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
            problem_config=self.config,
        )

    def create_equations(self) -> EulerEquations1D:
        """Create 1D Euler equations."""
        return EulerEquations1D(gamma=self.gamma)

    def get_solver_class(self):
        """Get the solver class for this problem."""
        from phrike.solver import SpectralSolver1D
        return SpectralSolver1D

    def create_initial_conditions(self, grid: Grid1D):
        """Create HSE initial conditions."""
        import numpy as np

        # Get configuration parameters
        ic_config = self.config["initial_conditions"]
        rho_left = float(ic_config["left"]["rho"])
        rho_right = float(ic_config["right"]["rho"])
        pressure_left = float(ic_config["left"]["p"])
        
        # Get gravity configuration
        gravity_config = self.config.get("gravity", {})
        gx = 0.0
        if gravity_config.get("enabled", False):
            gx = float(gravity_config.get("gx", 0.0))
        
        # Discontinuity location (middle of domain)
        x0 = float(self.config["grid"].get("x0", 0.5 * grid.Lx))
        
        # Smoothing parameter
        sigma = float(self.config.get("initial_conditions", {}).get("smoothing", {}).get("sigma", 0.01))

        U0 = hse_initial_conditions(
            x=grid.x,
            rho_left=rho_left,
            rho_right=rho_right,
            pressure_left=pressure_left,
            gamma=self.gamma,
            gx=gx,
            x0=x0,
            sigma=sigma,
        )

        return U0

    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
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
        fig.suptitle(f"HSE Test at t={t:.3f}", fontsize=14)
        
        x = solver.grid.x
        rho, u, p = solver.equations.primitive(U)
        
        # Convert to numpy if needed
        x = self.convert_torch_to_numpy(x)[0]
        rho = self.convert_torch_to_numpy(rho)[0]
        u = self.convert_torch_to_numpy(u)[0]
        p = self.convert_torch_to_numpy(p)[0]
        
        # Density
        axs[0, 0].plot(x, rho, 'b-', linewidth=2)
        axs[0, 0].set_xlabel("x")
        axs[0, 0].set_ylabel("Density")
        axs[0, 0].set_title("Density")
        axs[0, 0].grid(True)
        
        # Velocity
        axs[0, 1].plot(x, u, 'r-', linewidth=2)
        axs[0, 1].set_xlabel("x")
        axs[0, 1].set_ylabel("Velocity")
        axs[0, 1].set_title("Velocity")
        axs[0, 1].grid(True)
        
        # Pressure
        axs[1, 0].plot(x, p, 'g-', linewidth=2)
        axs[1, 0].set_xlabel("x")
        axs[1, 0].set_ylabel("Pressure")
        axs[1, 0].set_title("Pressure")
        axs[1, 0].grid(True)
        
        # Total energy
        E = U[2]
        E = self.convert_torch_to_numpy(E)[0]
        axs[1, 1].plot(x, E, 'm-', linewidth=2)
        axs[1, 1].set_xlabel("x")
        axs[1, 1].set_ylabel("Energy")
        axs[1, 1].set_title("Total Energy")
        axs[1, 1].grid(True)
        
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
            title=f"HSE Test at t={solver.t:.3f}",
            outpath=os.path.join(self.outdir, f"fields_t{solver.t:.3f}.png"),
        )
