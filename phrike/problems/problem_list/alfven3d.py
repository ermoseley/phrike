"""3D circularly polarized Alfven wave along x or the box diagonal.

Same exact nonlinear solution as the 1D problem, but run through the full 3D MHD
machinery (Grid3D.project_solenoidal, MHDEquations3D flux). Used as the 3D
convergence and div(B) check.
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
                                   m=1, Lx=1.0, direction=1, gamma=5.0 / 3.0,
                                   *, orientation="x", Y=None, Z=None,
                                   Ly=1.0, Lz=1.0):
    """Exact CP wave; diagonal uses n=(1,1,1)/sqrt(3) in a periodic cube."""
    if orientation not in ("x", "diagonal"):
        raise ValueError("alfven3d orientation must be 'x' or 'diagonal'")
    if orientation == "diagonal":
        if not (Lx == Ly == Lz):
            raise ValueError("alfven3d diagonal requires a cubic box (Lx = Ly = Lz)")
        if Y is None or Z is None:
            raise ValueError("alfven3d diagonal requires X, Y and Z coordinates")
        if hasattr(X, "cpu"):
            import torch
            xp = torch
        else:
            xp = np
        phi = (2.0 * np.pi * m / Lx) * (
            X + Y + Z - np.sqrt(3.0) * direction * B0 / np.sqrt(rho0) * t
        )
        c = b1 * xp.cos(phi); s = b1 * xp.sin(phi)
        bx = c / np.sqrt(2.0) + s / np.sqrt(6.0)
        by = -c / np.sqrt(2.0) + s / np.sqrt(6.0)
        bz = -2.0 * s / np.sqrt(6.0)
        guide = B0 / np.sqrt(3.0)
        scale = -float(direction) / np.sqrt(rho0)
        return MHDEquations3D(gamma=gamma).conservative(
            xp.full_like(X, float(rho0)), scale * bx, scale * by, scale * bz,
            xp.full_like(X, float(p0)), guide + bx, guide + by, guide + bz,
        )
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
            orientation=str(icfg.get("orientation", "x")),
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
        return self._exact_state(grid, 0.0)

    def _exact_state(self, grid: Grid3D, t: float):
        X, Y, Z = grid.xyz_mesh()
        return circularly_polarized_alfven_3d(
            X, t=t, Y=Y, Z=Z, Lx=grid.Lx, Ly=grid.Ly, Lz=grid.Lz,
            gamma=self.gamma, **self._ic_params()
        )

    def create_visualization(self, solver, t, U):
        pass

    def create_final_visualization(self, solver) -> None:
        snap = save_solution_snapshot(self.outdir, solver.t, U=solver.U,
                                      grid=solver.grid, equations=solver.equations)
        print(f"Saved final snapshot: {snap}")
        actual = solver.equations.primitive(solver.U)
        exact = solver.equations.primitive(self._exact_state(solver.grid, solver.t))
        for name, indices in (("B", (5, 6, 7)), ("u", (1, 2, 3))):
            fields = np.array(self.convert_torch_to_numpy(*(actual[i] for i in indices)))
            reference = np.array(self.convert_torch_to_numpy(*(exact[i] for i in indices)))
            error = fields - reference
            norm = np.sqrt(np.sum(reference ** 2))
            relative = np.sqrt(np.sum(error ** 2)) / (norm if norm > 0.0 else 1.0)
            print(f"Alfven {name}: L2={np.sqrt(np.mean(error ** 2)):.6e}, "
                  f"relative L2={relative:.6e}, max={np.max(np.abs(error)):.6e}")
        X, _, _ = solver.grid.xyz_mesh()
        x = X[0, 0, :]
        if hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        _, _, _, _, _, _, By, Bz = solver.equations.primitive(solver.U)
        By, Bz = self.convert_torch_to_numpy(By, Bz)
        line_y = By[0, 0, :]
        line_z = Bz[0, 0, :]
        Ua = circularly_polarized_alfven_3d(np.asarray(x), t=solver.t, Lx=solver.grid.Lx,
                                            Y=np.zeros_like(x), Z=np.zeros_like(x),
                                            Ly=solver.grid.Ly, Lz=solver.grid.Lz,
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
