"""3D circularly polarized Alfven wave (propagating along x in a 3D box).

Same exact nonlinear solution as the 1D problem, but run through the full 3D MHD
machinery (Grid3D.project_solenoidal, MHDEquations3D flux). Primary 3D
convergence / div(B) validation from mhd_plan.md.
"""

import os
import numpy as np
from typing import Optional, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phrike.grid import Grid3D
from phrike.equations import MHDEquations3D
from phrike.solver import SpectralSolverMHD3D
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def circularly_polarized_alfven_3d(X, t=0.0, rho0=1.0, p0=0.1, B0=1.0, b1=0.1,
                                   m=1, Lx=1.0, direction=1, gamma=5.0 / 3.0):
    """CP Alfven wave varying along x (uniform in y, z)."""
    is_torch = hasattr(X, "cpu")
    cA = B0 / np.sqrt(rho0)
    k = 2.0 * np.pi * m / Lx
    phase = k * direction * cA * t
    sign = float(direction)
    inv = 1.0 / np.sqrt(rho0)
    if is_torch:
        import torch

        phi = k * X - phase
        cos = torch.cos(phi); sin = torch.sin(phi)
        rho = torch.full_like(X, float(rho0)); p = torch.full_like(X, float(p0))
        Bx = torch.full_like(X, float(B0)); By = b1 * cos; Bz = b1 * sin
        ux = torch.zeros_like(X); uy = -sign * By * inv; uz = -sign * Bz * inv
    else:
        phi = k * X - phase
        cos = np.cos(phi); sin = np.sin(phi)
        rho = np.full_like(X, float(rho0)); p = np.full_like(X, float(p0))
        Bx = np.full_like(X, float(B0)); By = b1 * cos; Bz = b1 * sin
        ux = np.zeros_like(X); uy = -sign * By * inv; uz = -sign * Bz * inv
    eqs = MHDEquations3D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, uz, p, Bx, By, Bz)


class Alfven3DProblem(BaseProblem):
    """3D circularly polarized Alfven wave (MHD validation)."""

    def _ic_params(self) -> Dict[str, Any]:
        icfg = self.config["initial_conditions"]
        return dict(
            rho0=float(icfg.get("rho0", 1.0)), p0=float(icfg.get("p0", 0.1)),
            B0=float(icfg.get("B0", 1.0)), b1=float(icfg.get("b1", 0.1)),
            m=int(icfg.get("m", 1)), direction=int(icfg.get("direction", 1)),
        )

    def create_grid(self, backend="numpy", device=None, debug=False) -> Grid3D:
        g = Grid3D(
            Nx=int(self.config["grid"]["Nx"]), Ny=int(self.config["grid"]["Ny"]),
            Nz=int(self.config["grid"]["Nz"]),
            Lx=float(self.config["grid"]["Lx"]), Ly=float(self.config["grid"]["Ly"]),
            Lz=float(self.config["grid"]["Lz"]),
            dealias=bool(self.config["grid"].get("dealias", True)),
            filter_params=self.filter_config, fft_workers=self.fft_workers,
            backend=backend, torch_device=device, precision=self.precision, debug=debug,
        )
        g.mhd_config = self.config.get("physics", {}).get("mhd", {})
        return g

    def create_equations(self) -> MHDEquations3D:
        return MHDEquations3D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid3D):
        X, _, _ = grid.xyz_mesh()
        return circularly_polarized_alfven_3d(
            X, t=0.0, Lx=grid.Lx, gamma=self.gamma, **self._ic_params()
        )

    def create_visualization(self, solver, t, U):
        pass

    def create_final_visualization(self, solver) -> None:
        snap = save_solution_snapshot(self.outdir, solver.t, U=solver.U,
                                      grid=solver.grid, equations=solver.equations)
        print(f"Saved final snapshot: {snap}")
        X, _, _ = solver.grid.xyz_mesh()
        x = X[0, 0, :]
        if hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        _, _, _, _, _, _, By, Bz = solver.equations.primitive(solver.U)
        By, Bz = self.convert_torch_to_numpy(By, Bz)
        line_y = By[0, 0, :]
        line_z = Bz[0, 0, :]
        Ua = circularly_polarized_alfven_3d(np.asarray(x), t=solver.t, Lx=solver.grid.Lx,
                                            gamma=self.gamma, **self._ic_params())
        eqs = MHDEquations3D(gamma=self.gamma)
        _, _, _, _, _, _, Bya, Bza = eqs.primitive(Ua)
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        ax.plot(x, line_y, "b-", label="By num"); ax.plot(x, Bya, "k--", label="By exact")
        ax.plot(x, line_z, "r-", label="Bz num"); ax.plot(x, Bza, "g--", label="Bz exact")
        ax.set_title(f"3D CP Alfven (x-line) t={solver.t:.4f}")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.outdir, f"alfven3d_t{solver.t:.4f}.png"), dpi=150)
        plt.close()

    def get_solver_class(self):
        return SpectralSolverMHD3D
