"""1D Brio-Wu magnetohydrodynamic shock tube."""

import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from phrike.equations import MHDEquations1D
from phrike.grid import Grid1D
from phrike.io import save_solution_snapshot
from phrike.solver import SpectralSolverMHD1D
from ..base import BaseProblem


class BrioWu1DProblem(BaseProblem):
    """Coplanar MHD shock tube with a uniform longitudinal magnetic field."""

    def create_grid(self, backend: str = "numpy", device: Optional[str] = None,
                    debug: bool = False) -> Grid1D:
        cfg = self.config["grid"]
        grid = Grid1D(
            N=int(cfg["N"]), Lx=float(cfg["Lx"]),
            basis=str(cfg.get("basis", "fourier")).lower(), bc=cfg.get("bc"),
            dealias=bool(cfg.get("dealias", True)),
            filter_params=self.filter_config, fft_workers=self.fft_workers,
            backend=backend, torch_device=device, precision=self.precision,
            debug=debug,
        )
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        if cfg.get("bc_values"):
            grid.set_dirichlet_bc_values(cfg["bc_values"])
        return grid

    def create_equations(self) -> MHDEquations1D:
        return MHDEquations1D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid1D):
        ic = self.config.get("initial_conditions", {})
        left = dict(rho=1.0, u=0.0, v=0.0, w=0.0, p=1.0,
                    bx=0.75, by=1.0, bz=0.0)
        right = dict(rho=0.125, u=0.0, v=0.0, w=0.0, p=0.1,
                     bx=0.75, by=-1.0, bz=0.0)
        left.update(ic.get("left", {}))
        right.update(ic.get("right", {}))
        bx_left, bx_right = float(left["bx"]), float(right["bx"])
        if not np.isfinite(bx_left) or not np.isfinite(bx_right) or bx_left != bx_right:
            raise ValueError("Brio-Wu requires matching finite Bx on both sides")

        x = grid.x
        x0 = float(self.config["grid"].get("x0", 0.5 * grid.Lx))
        smoothing = ic.get("smoothing", {})
        sigma = float(smoothing.get("sigma", 0.0))
        if hasattr(x, "cpu"):
            import torch

            if smoothing.get("enabled", False) and sigma > 0.0:
                step = 0.5 * (1.0 + torch.tanh((x0 - x) / sigma))
            else:
                step = (x < x0).to(dtype=x.dtype)
            bx = torch.full_like(x, bx_left)
        else:
            if smoothing.get("enabled", False) and sigma > 0.0:
                step = 0.5 * (1.0 + np.tanh((x0 - x) / sigma))
            else:
                step = (x < x0).astype(x.dtype)
            bx = np.full_like(x, bx_left)

        def profile(key):
            return float(right[key]) + (float(left[key]) - float(right[key])) * step

        return self.create_equations().conservative(
            profile("rho"), profile("u"), profile("v"), profile("w"),
            profile("p"), bx, profile("by"), profile("bz"),
        )

    def create_visualization(self, solver, t: float, U):
        frames_dir = os.path.join(self.outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        x = self.convert_torch_to_numpy(solver.grid.x)[0]
        rho, ux, uy, uz, p, _, by, _ = self.convert_torch_to_numpy(
            *solver.equations.primitive(U)
        )
        video = self.config.get("video", {})
        scale = float(video.get("scale", 1.0))
        fig, axes = plt.subplots(2, 2, figsize=(10 * scale, 6 * scale),
                                 constrained_layout=True)
        fig.suptitle(f"Brio-Wu at t={t:.3f}")
        axes[0, 0].plot(x, rho)
        axes[0, 0].set_ylabel("Density")
        axes[0, 1].plot(x, p)
        axes[0, 1].set_ylabel("Pressure")
        for velocity, label in ((ux, "ux"), (uy, "uy"), (uz, "uz")):
            axes[1, 0].plot(x, velocity, label=label)
        axes[1, 0].set_ylabel("Velocity")
        axes[1, 0].legend()
        axes[1, 1].plot(x, by)
        axes[1, 1].set_ylabel("By")
        for ax in axes.flat:
            ax.set_xlabel("x")
            ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"),
                    dpi=int(video.get("frame_dpi", 150)))
        plt.close(fig)
        snapshot_path = save_solution_snapshot(
            self.outdir, t, U=U, grid=solver.grid, equations=solver.equations,
        )
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

    def create_final_visualization(self, solver) -> None:
        self.create_visualization(solver, solver.t, solver.U)

    def get_solver_class(self):
        return SpectralSolverMHD1D
