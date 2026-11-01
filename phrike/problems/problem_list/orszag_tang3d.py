"""3D Orszag-Tang vortex of Helzel, Rossmanith & Taetz (2011).

A sinusoidal z perturbation of amplitude epsilon modulates the planar velocity
and adds a vertical velocity. At epsilon=0 this is the 2D vortex extruded in z.
"""

import os
import numpy as np
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phrike.grid import Grid3D
from phrike.equations import MHDEquations3D
from phrike.solver import SpectralSolverMHD3D
from ..base import BaseProblem
from .orszag_tang2d import orszag_tang_2d


def orszag_tang_3d(X, Y, Z, gamma: float = 5.0 / 3.0, Lx: float = 1.0,
                   Ly: float = 1.0, Lz: float = 1.0, epsilon: float = 0.2,
                   rho0: Optional[float] = None, p0: Optional[float] = None,
                   B0: Optional[float] = None):
    U = orszag_tang_2d(X, Y, gamma=gamma, Lx=Lx, Ly=Ly,
                       rho0=rho0, p0=p0, B0=B0)
    if epsilon == 0.0:
        return U
    if hasattr(X, "cpu"):
        import torch
        perturbation = epsilon * torch.sin((2.0 * np.pi / Lz) * Z)
    else:
        perturbation = epsilon * np.sin((2.0 * np.pi / Lz) * Z)
    eqs = MHDEquations3D(gamma=gamma)
    rho, ux, uy, _, p, Bx, By, Bz = eqs.primitive(U)
    return eqs.conservative(rho, (1.0 + perturbation) * ux,
                            (1.0 + perturbation) * uy, perturbation, p, Bx, By, Bz)


class OrszagTang3DProblem(BaseProblem):
    """3D Orszag-Tang vortex (MHD)."""

    def create_grid(self, backend="numpy", device=None, debug=False) -> Grid3D:
        grid = Grid3D(
            Nx=int(self.config["grid"]["Nx"]), Ny=int(self.config["grid"]["Ny"]),
            Nz=int(self.config["grid"]["Nz"]),
            Lx=float(self.config["grid"].get("Lx", 1.0)),
            Ly=float(self.config["grid"].get("Ly", 1.0)),
            Lz=float(self.config["grid"].get("Lz", 1.0)),
            dealias=bool(self.config["grid"].get("dealias", True)),
            filter_params=self.filter_config, fft_workers=self.fft_workers,
            backend=backend, torch_device=device, precision=self.precision, debug=debug,
        )
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        return grid

    def create_equations(self) -> MHDEquations3D:
        return MHDEquations3D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid3D):
        icfg = self.config.get("initial_conditions", {})
        return orszag_tang_3d(
            *grid.xyz_mesh(), gamma=self.gamma, Lx=grid.Lx, Ly=grid.Ly, Lz=grid.Lz,
            epsilon=float(icfg.get("epsilon", 0.2)),
            rho0=icfg.get("rho0"), p0=icfg.get("p0"), B0=icfg.get("B0"),
        )

    def _save_slice(self, solver, t, U, path, dpi):
        rho, _, _, _, _, Bx, By, Bz = solver.equations.primitive(U)
        mid = solver.grid.Nz // 2
        rho, Bx, By, Bz = self.convert_torch_to_numpy(rho[mid], Bx[mid], By[mid], Bz[mid])
        magnetic_pressure = 0.5 * (Bx ** 2 + By ** 2 + Bz ** 2)
        fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
        for ax, data, title, cmap in (
            (axes[0], rho, "Density", "inferno"),
            (axes[1], magnetic_pressure, "Magnetic pressure", "plasma"),
        ):
            im = ax.imshow(data, origin="lower",
                           extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
                           aspect="equal", cmap=cmap)
            ax.set_title(f"{title} (z=mid)  t={t:.3f}")
            ax.set_xlabel("x"); ax.set_ylabel("y")
            plt.colorbar(im, ax=ax, shrink=0.85)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    def create_visualization(self, solver, t, U) -> None:
        self._save_slice(
            solver, t, U, os.path.join(self.outdir, "frames", f"frame_{t:08.4f}.png"),
            int(self.config.get("video", {}).get("frame_dpi", 120)),
        )

    def create_final_visualization(self, solver) -> None:
        self._save_slice(
            solver, solver.t, solver.U,
            os.path.join(self.outdir, f"orszag_tang3d_t{solver.t:.3f}.png"), 150,
        )

    def get_solver_class(self):
        return SpectralSolverMHD3D
