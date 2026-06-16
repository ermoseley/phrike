"""1D Brio-Wu MHD shock tube — classic coplanar MHD Riemann problem.

Brio & Wu (1988). Standard setup on ``[0, Lx]`` with a discontinuity at ``x0``
and ``gamma = 2``:

    left  (x < x0):  rho = 1.0,   u = 0, p = 1.0, By = +1.0
    right (x >= x0): rho = 0.125, u = 0, p = 0.1, By = -1.0
    Bx = 0.75 (constant),  Bz = 0

The solution develops the characteristic Brio-Wu structure: a fast rarefaction,
a slow compound wave, a contact discontinuity, a slow shock, and a fast
rarefaction. ``Bx`` is constant in 1D (the solenoidal constraint dBx/dx = 0), so
the fast/slow waves only modulate the transverse field ``By``.

This problem uses a periodic (Fourier) basis. The run time is kept short enough
that the waves launched from ``x0`` do not interact with the waves launched from
the periodic image of the discontinuity at the domain edge, so the central
structure matches the standard (open-boundary) Brio-Wu reference solution.
Shocks are stabilised by the spectral filter together with a small explicit
resistivity/viscosity (``physics.mhd.resistivity`` / ``.viscosity``).
"""

import os
from typing import Optional, Dict, Any

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phrike.grid import Grid1D
from phrike.equations import MHDEquations1D
from phrike.solver import SpectralSolverMHD1D
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def brio_wu_shock_tube(
    x,
    x0: float,
    left: Dict[str, float],
    right: Dict[str, float],
    Bx: float,
    gamma: float,
):
    """Return the conservative MHD state for the Brio-Wu initial condition.

    ``left`` / ``right`` dictionaries provide the primitive variables on each
    side of ``x0``: ``rho``, ``p``, ``By`` (required) and ``ux``, ``uy``, ``uz``,
    ``Bz`` (optional, default 0). ``Bx`` is uniform across the domain.
    """
    is_torch = hasattr(x, "cpu")

    def _g(side: Dict[str, float], key: str, default: float = 0.0) -> float:
        return float(side.get(key, default))

    if is_torch:
        import torch

        def step(vl: float, vr: float):
            return torch.where(
                x < x0,
                torch.tensor(float(vl), dtype=x.dtype, device=x.device),
                torch.tensor(float(vr), dtype=x.dtype, device=x.device),
            )

        Bx_field = torch.full_like(x, float(Bx))
    else:
        def step(vl: float, vr: float):
            return np.where(x < x0, float(vl), float(vr))

        Bx_field = np.full_like(x, float(Bx))

    rho = step(_g(left, "rho", 1.0), _g(right, "rho", 0.125))
    ux = step(_g(left, "ux"), _g(right, "ux"))
    uy = step(_g(left, "uy"), _g(right, "uy"))
    uz = step(_g(left, "uz"), _g(right, "uz"))
    p = step(_g(left, "p", 1.0), _g(right, "p", 0.1))
    By = step(_g(left, "By", 1.0), _g(right, "By", -1.0))
    Bz = step(_g(left, "Bz"), _g(right, "Bz"))

    eqs = MHDEquations1D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, uz, p, Bx_field, By, Bz)


class BrioWu1DProblem(BaseProblem):
    """1D Brio-Wu MHD shock tube."""

    def _ic_params(self) -> Dict[str, Any]:
        icfg = self.config.get("initial_conditions", {})
        return dict(
            left=dict(icfg.get("left", {})),
            right=dict(icfg.get("right", {})),
            Bx=float(icfg.get("Bx", 0.75)),
        )

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid1D:
        N = int(self.config["grid"]["N"])
        Lx = float(self.config["grid"]["Lx"])
        dealias = bool(self.config["grid"].get("dealias", True))
        grid = Grid1D(
            N=N,
            Lx=Lx,
            basis="fourier",
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
            debug=debug,
        )
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        return grid

    def create_equations(self) -> MHDEquations1D:
        return MHDEquations1D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid1D):
        x0 = float(self.config["grid"].get("x0", 0.5 * grid.Lx))
        return brio_wu_shock_tube(grid.x, x0=x0, gamma=self.gamma, **self._ic_params())

    def _plot_panels(self, fig, axes, x, rho, ux, uy, p, By, Bx) -> None:
        panels = (
            (axes[0, 0], rho, "Density", "b"),
            (axes[0, 1], p, "Pressure", "m"),
            (axes[0, 2], By, "By (transverse field)", "purple"),
            (axes[1, 0], ux, "Velocity x", "g"),
            (axes[1, 1], uy, "Velocity y", "darkorange"),
            (axes[1, 2], Bx, "Bx (constant in 1D)", "teal"),
        )
        for ax, data, title, color in panels:
            ax.plot(x, data, color=color, linewidth=1.5)
            ax.set_xlabel("x")
            ax.set_title(title)
            ax.grid(True, alpha=0.3)

    def create_visualization(self, solver, t: float, U) -> None:
        frames_dir = os.path.join(self.outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        video_config = self.config.get("video", {})
        frame_dpi = int(video_config.get("frame_dpi", 150))
        frame_scale = float(video_config.get("scale", 1.0))
        figsize = (15 * frame_scale, 8 * frame_scale)

        rho, ux, uy, uz, p, Bx, By, Bz = solver.equations.primitive(U)
        x = solver.grid.x
        x, rho, ux, uy, p, Bx, By = self.convert_torch_to_numpy(
            x, rho, ux, uy, p, Bx, By
        )

        fig, axes = plt.subplots(2, 3, figsize=figsize, constrained_layout=True)
        fig.suptitle(f"Brio-Wu MHD shock tube at t={t:.3f}", fontsize=14)
        self._plot_panels(fig, axes, x, rho, ux, uy, p, By, Bx)
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=frame_dpi)
        plt.close(fig)

        snapshot_path = save_solution_snapshot(
            self.outdir, t, U=U, grid=solver.grid, equations=solver.equations
        )
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

    def create_final_visualization(self, solver) -> None:
        snapshot_path = save_solution_snapshot(
            self.outdir,
            solver.t,
            U=solver.U,
            grid=solver.grid,
            equations=solver.equations,
        )
        print(f"Saved final snapshot: {snapshot_path}")

        rho, ux, uy, uz, p, Bx, By, Bz = solver.equations.primitive(solver.U)
        x = solver.grid.x
        x, rho, ux, uy, p, Bx, By = self.convert_torch_to_numpy(
            x, rho, ux, uy, p, Bx, By
        )

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        fig.suptitle(f"Brio-Wu MHD shock tube at t={solver.t:.3f}", fontsize=14)
        self._plot_panels(fig, axes, x, rho, ux, uy, p, By, Bx)
        fig.savefig(
            os.path.join(self.outdir, f"fields_t{solver.t:.3f}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

    def get_solver_class(self):
        return SpectralSolverMHD1D
