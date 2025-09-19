"""2D Kelvin-Helmholtz instability problem."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from phrike.grid import Grid2D
from phrike.equations import EulerEquations2D
# Initial condition function moved from phrike.initial_conditions
from phrike.solver import SpectralSolver2D
from ..base import BaseProblem


def kelvin_helmholtz_2d(
    X: np.ndarray,
    Y: np.ndarray,
    rho_outer: float = 1.0,
    rho_inner: float = 1.0,
    u0: float = 1.0,
    shear_thickness: float = 0.02,
    pressure_outer: float = 1.0,
    pressure_inner: float = 1.0,
    perturb_eps: float = 0.01,
    perturb_sigma: float = 0.02,
    perturb_kx: int = 2,
    gamma: float = 1.4,
) -> np.ndarray:
    """2D Kelvin-Helmholtz initial condition matching RAMSES khi.py profile.

    Creates two shear layers at y=0.25 and y=0.75 with:
    - Velocity: outer regions at -0.5*u0, middle region at +0.5*u0
    - Density/pressure: can differ between outer and inner regions
    - Perturbation: Gaussian-modulated sinusoidal vy at both interfaces
    """
    # Check if we're using torch tensors
    try:
        import torch  # type: ignore

        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False
        torch = None  # type: ignore

    is_torch = _TORCH_AVAILABLE and isinstance(X, (torch.Tensor,))

    if is_torch:
        # Normalize coordinates to [0,1) like RAMSES script
        Lx = X.max() - X.min()
        Ly = Y.max() - Y.min()
        yn = Y / Ly  # normalized y coordinates

        # Velocity profile: two tanh transitions at y=0.25 and y=0.75
        U1 = -0.5 * u0  # outer regions
        U2 = +0.5 * u0  # middle region
        a = float(shear_thickness)
        tanh_low = torch.tanh((yn - 0.25) / a)
        tanh_high = torch.tanh((yn - 0.75) / a)
        ux = U1 + 0.5 * (U2 - U1) * (tanh_low - tanh_high)

        # Density profile: same tanh structure
        rho_outer_val = float(rho_outer)
        rho_inner_val = float(rho_inner)
        rho = rho_outer_val + 0.5 * (rho_inner_val - rho_outer_val) * (
            tanh_low - tanh_high
        )

        # Pressure profile: same tanh structure
        p0 = float(pressure_outer)
        p_inner = float(pressure_inner)
        p = p0 + 0.5 * (p_inner - p0) * (tanh_low - tanh_high)

        # vy perturbation: Gaussian-modulated sinusoidal at both interfaces
        sig = float(perturb_sigma)
        gauss = torch.exp(-((yn - 0.25) ** 2) / (2.0 * sig * sig)) + torch.exp(
            -((yn - 0.75) ** 2) / (2.0 * sig * sig)
        )
        sinus = torch.sin(2.0 * torch.pi * perturb_kx * X / Lx)
        uy = perturb_eps * sinus * gauss
    else:
        # Normalize coordinates to [0,1) like RAMSES script
        Lx = X.max() - X.min()
        Ly = Y.max() - Y.min()
        yn = Y / Ly  # normalized y coordinates

        # Velocity profile: two tanh transitions at y=0.25 and y=0.75
        U1 = -0.5 * u0  # outer regions
        U2 = +0.5 * u0  # middle region
        a = float(shear_thickness)
        tanh_low = np.tanh((yn - 0.25) / a)
        tanh_high = np.tanh((yn - 0.75) / a)
        ux = U1 + 0.5 * (U2 - U1) * (tanh_low - tanh_high)

        # Density profile: same tanh structure
        rho_outer_val = float(rho_outer)
        rho_inner_val = float(rho_inner)
        rho = rho_outer_val + 0.5 * (rho_inner_val - rho_outer_val) * (
            tanh_low - tanh_high
        )

        # Pressure profile: same tanh structure
        p0 = float(pressure_outer)
        p_inner = float(pressure_inner)
        p = p0 + 0.5 * (p_inner - p0) * (tanh_low - tanh_high)

        # vy perturbation: Gaussian-modulated sinusoidal at both interfaces
        sig = float(perturb_sigma)
        gauss = np.exp(-((yn - 0.25) ** 2) / (2.0 * sig * sig)) + np.exp(
            -((yn - 0.75) ** 2) / (2.0 * sig * sig)
        )
        sinus = np.sin(2.0 * np.pi * perturb_kx * X / Lx)
        uy = perturb_eps * sinus * gauss

    eqs = EulerEquations2D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, p)


class KHI2DProblem(BaseProblem):
    """2D Kelvin-Helmholtz instability problem."""

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid2D:
        """Create 2D grid."""
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        dealias = bool(self.config["grid"].get("dealias", True))

        return Grid2D(
            Nx=Nx,
            Ny=Ny,
            Lx=Lx,
            Ly=Ly,
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
        """Create KHI initial conditions."""
        X, Y = grid.xy_mesh()
        icfg = self.config["initial_conditions"]

        rho_outer = float(icfg.get("rho_outer", 1.0))
        rho_inner = float(icfg.get("rho_inner", 2.0))
        u0 = float(icfg.get("u0", 1.0))
        shear_thickness = float(icfg.get("shear_thickness", 0.02))
        pressure_outer = float(icfg.get("pressure_outer", 1.0))
        pressure_inner = float(icfg.get("pressure_inner", 1.0))
        perturb_eps = float(icfg.get("perturb_eps", 0.01))
        perturb_sigma = float(icfg.get("perturb_sigma", 0.02))
        perturb_kx = int(icfg.get("perturb_kx", 2))

        return kelvin_helmholtz_2d(
            X,
            Y,
            rho_outer=rho_outer,
            rho_inner=rho_inner,
            u0=u0,
            shear_thickness=shear_thickness,
            pressure_outer=pressure_outer,
            pressure_inner=pressure_inner,
            perturb_eps=perturb_eps,
            perturb_sigma=perturb_sigma,
            perturb_kx=perturb_kx,
            gamma=self.gamma,
        )

    def create_visualization(self, solver, t: float, U) -> None:
        """Create visualization for current state."""
        rho, ux, uy, p = solver.equations.primitive(U)
        rho = self.convert_torch_to_numpy(rho)[0]

        # Create frame
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        ax.imshow(
            rho,
            origin="lower",
            extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
            aspect="equal",
        )
        ax.set_title(f"Density t={t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save frame
        frames_dir = os.path.join(self.outdir, "frames")
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=120)
        plt.close(fig)

    def create_final_visualization(self, solver) -> None:
        """Create final visualization plots."""
        rho, ux, uy, p = solver.equations.primitive(solver.U)
        rho = self.convert_torch_to_numpy(rho)[0]

        # Final density plot
        plt.figure(figsize=(8, 8))
        plt.imshow(
            rho,
            origin="lower",
            extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
            aspect="equal",
        )
        plt.title(f"Density t={solver.t:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"khi2d_t{solver.t:.3f}.png"), dpi=150)
        plt.close()

    def get_solver_class(self):
        """Get 2D solver class."""
        return SpectralSolver2D
