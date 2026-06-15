"""2D Orszag-Tang vortex — classic nonlinear MHD benchmark.

Standard setup (Stone et al. / Athena), periodic on [0,Lx]x[0,Ly], gamma = 5/3:

    rho = 25/(36 pi),   p = 5/(12 pi)
    u   = (-sin(2 pi y/Ly),  sin(2 pi x/Lx), 0)
    B   = (-B0 sin(2 pi y/Ly), B0 sin(4 pi x/Lx), 0),   B0 = 1/sqrt(4 pi)

The flow is transonic (sound speed = 1, |u| = 1); density develops a contrast of
~6 (peaks near rho ~ 0.45-0.5). The initial field is analytically
divergence-free (Bx depends only on y, By only on x); the Helmholtz projection
keeps it div-free to machine precision as small-scale current sheets form. The
vortex develops MHD turbulence/shocks, so a small explicit
resistivity/viscosity (physics.mhd.resistivity / .viscosity) may be needed for
stability at high resolution.
"""

import os
import numpy as np
from typing import Optional, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phrike.grid import Grid2D
from phrike.equations import MHDEquations2D
from phrike.solver import SpectralSolverMHD2D
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def orszag_tang_2d(X, Y, gamma: float = 5.0 / 3.0, Lx: float = 1.0, Ly: float = 1.0,
                   rho0: Optional[float] = None, p0: Optional[float] = None,
                   B0: Optional[float] = None):
    is_torch = hasattr(X, "cpu")
    if rho0 is None:
        rho0 = 25.0 / (36.0 * np.pi)
    if p0 is None:
        p0 = 5.0 / (12.0 * np.pi)
    if B0 is None:
        B0 = 1.0 / np.sqrt(4.0 * np.pi)
    kx = 2.0 * np.pi / Lx
    ky = 2.0 * np.pi / Ly
    if is_torch:
        import torch

        rho = torch.full_like(X, float(rho0))
        p = torch.full_like(X, float(p0))
        ux = -torch.sin(ky * Y)
        uy = torch.sin(kx * X)
        uz = torch.zeros_like(X)
        Bx = -B0 * torch.sin(ky * Y)
        By = B0 * torch.sin(2.0 * kx * X)
        Bz = torch.zeros_like(X)
    else:
        rho = np.full_like(X, float(rho0))
        p = np.full_like(X, float(p0))
        ux = -np.sin(ky * Y)
        uy = np.sin(kx * X)
        uz = np.zeros_like(X)
        Bx = -B0 * np.sin(ky * Y)
        By = B0 * np.sin(2.0 * kx * X)
        Bz = np.zeros_like(X)
    eqs = MHDEquations2D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, uz, p, Bx, By, Bz)


class OrszagTang2DProblem(BaseProblem):
    """2D Orszag-Tang vortex (MHD)."""

    def create_grid(self, backend: str = "numpy", device: Optional[str] = None,
                    debug: bool = False) -> Grid2D:
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Lx = float(self.config["grid"].get("Lx", 1.0))
        Ly = float(self.config["grid"].get("Ly", 1.0))
        dealias = bool(self.config["grid"].get("dealias", True))
        grid = Grid2D(
            Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dealias=dealias,
            filter_params=self.filter_config, fft_workers=self.fft_workers,
            backend=backend, torch_device=device, precision=self.precision,
            debug=debug, basis_x="fourier", basis_y="fourier",
        )
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        return grid

    def create_equations(self) -> MHDEquations2D:
        return MHDEquations2D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid2D):
        X, Y = grid.xy_mesh()
        icfg = self.config.get("initial_conditions", {})
        return orszag_tang_2d(
            X, Y, gamma=self.gamma, Lx=grid.Lx, Ly=grid.Ly,
            rho0=icfg.get("rho0"), p0=icfg.get("p0"), B0=icfg.get("B0"),
        )

    def create_visualization(self, solver, t: float, U) -> None:
        rho = solver.equations.primitive(U)[0]
        rho = self.convert_torch_to_numpy(rho)[0]
        fig, ax = plt.subplots(1, 1, figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(rho, origin="lower",
                       extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
                       aspect="equal", cmap="inferno")
        ax.set_title(f"Orszag-Tang density  t={t:.3f}")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.85)
        frames_dir = os.path.join(self.outdir, "frames")
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.4f}.png"),
                    dpi=int(self.config.get("video", {}).get("frame_dpi", 120)),
                    bbox_inches="tight")
        plt.close(fig)

    def create_final_visualization(self, solver) -> None:
        rho, ux, uy, uz, p, Bx, By, Bz = solver.equations.primitive(solver.U)
        rho, ux, uy, p, Bx, By = self.convert_torch_to_numpy(rho, ux, uy, p, Bx, By)
        ext = [0, solver.grid.Lx, 0, solver.grid.Ly]
        fig, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
        for ax, data, title, cmap in (
            (axes[0, 0], rho, "Density", "inferno"),
            (axes[0, 1], p, "Pressure", "viridis"),
            (axes[1, 0], np.sqrt(ux ** 2 + uy ** 2), "|v|", "magma"),
            (axes[1, 1], np.sqrt(Bx ** 2 + By ** 2), "|B_perp|", "plasma"),
        ):
            im = ax.imshow(data, origin="lower", extent=ext, aspect="equal", cmap=cmap)
            ax.set_title(f"{title}  t={solver.t:.3f}")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, shrink=0.85)
        plt.savefig(os.path.join(self.outdir, f"orszag_tang2d_t{solver.t:.3f}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    def get_solver_class(self):
        return SpectralSolverMHD2D
