"""3D Taylor-Green vortex problem."""

import os
import warnings
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from phrike.grid import Grid3D
from phrike.equations import EulerEquations3D
from phrike.solver import SpectralSolver3D
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def taylor_green_vortex_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    rho0: float = 1.0,
    p0: float = 100.0,
    U0: float = 1.0,
    k: int = 1,
    gamma: float = 1.4,
) -> np.ndarray:
    """3D Taylor-Green vortex initial condition.

    Velocity field (Brachet et al. 1983):
        u =  U0 * sin(k x) cos(k y) cos(k z)
        v = -U0 * cos(k x) sin(k y) cos(k z)
        w =  0
    The pressure that cancels the divergent part of the initial advection is
        p = p0 + (rho0 * U0**2 / 16) * (cos(2kx) + cos(2ky)) * (cos(2kz) + 2)
    ``p0`` is the mean pressure. The Mach number is
    ``Ma = U0 / sqrt(gamma * p0 / rho0)``.
    """
    # Torch interop if needed
    try:
        import torch  # type: ignore

        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False
        torch = None  # type: ignore

    is_torch = _TORCH_AVAILABLE and any(
        isinstance(a, (torch.Tensor,)) for a in (X, Y, Z)
    )

    def _pressure(cos):
        # Mean p0 plus the Brachet fluctuation. The fluctuation averages to 0.
        return (
            float(p0)
            + (float(rho0) * float(U0) ** 2 / 16.0)
            * (cos(2.0 * k * X) + cos(2.0 * k * Y))
            * (cos(2.0 * k * Z) + 2.0)
        )

    if is_torch:
        assert torch is not None
        ux = U0 * torch.sin(k * X) * torch.cos(k * Y) * torch.cos(k * Z)
        uy = -U0 * torch.cos(k * X) * torch.sin(k * Y) * torch.cos(k * Z)
        uz = torch.zeros_like(X)
        rho = torch.full_like(X, float(rho0))
        p = _pressure(torch.cos)
    else:
        ux = U0 * np.sin(k * X) * np.cos(k * Y) * np.cos(k * Z)
        uy = -U0 * np.cos(k * X) * np.sin(k * Y) * np.cos(k * Z)
        uz = np.zeros_like(X)
        rho = rho0 * np.ones_like(X)
        p = _pressure(np.cos)

    eqs = EulerEquations3D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, uz, p)


class TGV3DProblem(BaseProblem):
    """3D Taylor-Green vortex problem."""

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid3D:
        """Create 3D grid."""
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Nz = int(self.config["grid"]["Nz"])
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        Lz = float(self.config["grid"]["Lz"])
        dealias = bool(self.config["grid"].get("dealias", True))

        return Grid3D(
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
        )

    def create_equations(self) -> EulerEquations3D:
        """Create 3D Euler equations."""
        return EulerEquations3D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid3D):
        """Create TGV initial conditions."""
        X, Y, Z = grid.xyz_mesh()
        icfg = self.config["initial_conditions"]

        rho0 = float(icfg.get("rho0", 1.0))
        U0 = float(icfg.get("U0", 1.0))
        k = int(icfg.get("k", 1))
        # mach, when set, fixes the mean pressure so Ma = U0/c0 holds.
        if "mach" in icfg:
            mach = float(icfg["mach"])
            if mach <= 0.0:
                raise ValueError("initial_conditions.mach must be positive")
            p0 = rho0 * U0 * U0 / (self.gamma * mach * mach)
            if "p0" in icfg:
                stated = float(icfg["p0"])
                if abs(stated - p0) > 1e-8 * max(1.0, abs(p0)):
                    warnings.warn(
                        "initial_conditions.p0 does not match mach; "
                        f"using p0={p0:.8g} from Ma={mach:g}",
                        UserWarning,
                        stacklevel=2,
                    )
        else:
            p0 = float(icfg.get("p0", 100.0))

        return taylor_green_vortex_3d(
            X, Y, Z, rho0=rho0, p0=p0, U0=U0, k=k, gamma=self.gamma
        )

    def create_visualization(self, solver, t: float, U) -> None:
        """Create visualization for current state."""
        rho, ux, uy, uz, p = solver.equations.primitive(U)
        rho = self.convert_torch_to_numpy(rho)[0]

        # Take mid-plane slice
        mid = rho.shape[0] // 2

        # Create frame
        fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
        ax.imshow(
            rho[mid],
            origin="lower",
            extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
            aspect="equal",
        )
        ax.set_title(f"Density z=mid t={t:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save frame
        frames_dir = os.path.join(self.outdir, "frames")
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=120)
        plt.close(fig)

        # Save snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir, t, U=U, grid=solver.grid, equations=solver.equations
        )
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

    def create_final_visualization(self, solver) -> None:
        """Create final visualization plots."""
        rho, ux, uy, uz, p = solver.equations.primitive(solver.U)
        rho = self.convert_torch_to_numpy(rho)[0]

        # Final mid-plane slice plot
        mid = rho.shape[0] // 2
        plt.figure(figsize=(7, 6))
        plt.imshow(
            rho[mid],
            origin="lower",
            extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
            aspect="equal",
        )
        plt.title(f"Density z=mid t={solver.t:.3f}")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(os.path.join(self.outdir, f"tgv3d_t{solver.t:.3f}.png"), dpi=150)
        plt.close()

    def get_solver_class(self):
        """Get 3D solver class."""
        return SpectralSolver3D
