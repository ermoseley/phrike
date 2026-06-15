"""1D circularly polarized Alfven wave — primary MHD validation problem.

A circularly polarized Alfven wave propagating along x is an exact nonlinear
solution of the ideal MHD equations: density and (thermal + magnetic) pressure
are uniform and the transverse field/velocity rotate while translating at the
Alfven speed c_A = B0 / sqrt(rho0). It validates:

- the magnetic pressure / pressure recovery (E -= 1/2 |B|^2),
- the induction flux rows (dB/dt = curl(u x B)),
- the solenoidal constraint (div B = dBx/dx = 0 to machine precision),
- spectral spatial convergence and RK temporal order,
- the phase speed (= c_A).

For a wave travelling in +x the eigenmode relation is  du_perp = -dB_perp/sqrt(rho0).
"""

import os
import numpy as np
from typing import Optional, Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phrike.grid import Grid1D
from phrike.equations import MHDEquations1D
from phrike.solver import SpectralSolverMHD1D
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def circularly_polarized_alfven_1d(
    x,
    t: float = 0.0,
    rho0: float = 1.0,
    p0: float = 0.1,
    B0: float = 1.0,
    b1: float = 0.1,
    m: int = 1,
    Lx: float = 1.0,
    direction: int = 1,
    gamma: float = 5.0 / 3.0,
):
    """Conservative state of a CP Alfven wave at time ``t``.

    direction = +1 : wave travels in +x (du_perp = -dB_perp/sqrt(rho0)).
    direction = -1 : wave travels in -x (du_perp = +dB_perp/sqrt(rho0)).
    """
    is_torch = hasattr(x, "cpu")
    cA = B0 / np.sqrt(rho0)
    k = 2.0 * np.pi * m / Lx
    phase_shift = k * direction * cA * t
    sign = float(direction)
    inv_sqrt_rho = 1.0 / np.sqrt(rho0)
    if is_torch:
        import torch

        phi = k * x - phase_shift
        cos = torch.cos(phi)
        sin = torch.sin(phi)
        rho = torch.full_like(x, float(rho0))
        p = torch.full_like(x, float(p0))
        Bx = torch.full_like(x, float(B0))
        By = b1 * cos
        Bz = b1 * sin
        ux = torch.zeros_like(x)
        uy = -sign * By * inv_sqrt_rho
        uz = -sign * Bz * inv_sqrt_rho
    else:
        phi = k * x - phase_shift
        cos = np.cos(phi)
        sin = np.sin(phi)
        rho = np.full_like(x, float(rho0))
        p = np.full_like(x, float(p0))
        Bx = np.full_like(x, float(B0))
        By = b1 * cos
        Bz = b1 * sin
        ux = np.zeros_like(x)
        uy = -sign * By * inv_sqrt_rho
        uz = -sign * Bz * inv_sqrt_rho
    eqs = MHDEquations1D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, uz, p, Bx, By, Bz)


class Alfven1DProblem(BaseProblem):
    """1D circularly polarized Alfven wave (MHD validation)."""

    def _ic_params(self) -> Dict[str, Any]:
        icfg = self.config["initial_conditions"]
        return dict(
            rho0=float(icfg.get("rho0", 1.0)),
            p0=float(icfg.get("p0", 0.1)),
            B0=float(icfg.get("B0", 1.0)),
            b1=float(icfg.get("b1", 0.1)),
            m=int(icfg.get("m", 1)),
            direction=int(icfg.get("direction", 1)),
        )

    def create_grid(self, backend: str = "numpy", device: Optional[str] = None,
                    debug: bool = False) -> Grid1D:
        N = int(self.config["grid"]["N"])
        Lx = float(self.config["grid"]["Lx"])
        dealias = bool(self.config["grid"].get("dealias", True))
        grid = Grid1D(
            N=N, Lx=Lx, basis="fourier", dealias=dealias,
            filter_params=self.filter_config, fft_workers=self.fft_workers,
            backend=backend, torch_device=device, precision=self.precision,
            debug=debug,
        )
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        return grid

    def create_equations(self) -> MHDEquations1D:
        return MHDEquations1D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid1D):
        return circularly_polarized_alfven_1d(
            grid.x, t=0.0, Lx=grid.Lx, gamma=self.gamma, **self._ic_params()
        )

    def analytic_state(self, grid: Grid1D, t: float):
        """Conservative analytic state at time t (numpy)."""
        x = grid.x
        if hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        return circularly_polarized_alfven_1d(
            np.asarray(x), t=t, Lx=grid.Lx, gamma=self.gamma, **self._ic_params()
        )

    def create_visualization(self, solver, t: float, U):
        pass

    def create_final_visualization(self, solver) -> None:
        snapshot_path = save_solution_snapshot(
            self.outdir, solver.t, U=solver.U, grid=solver.grid,
            equations=solver.equations,
        )
        print(f"Saved final snapshot: {snapshot_path}")

        x = solver.grid.x
        if hasattr(x, "cpu"):
            x = x.detach().cpu().numpy()
        _, _, _, _, _, Bx, By, Bz = solver.equations.primitive(solver.U)
        By, Bz = self.convert_torch_to_numpy(By, Bz)
        Ua = self.analytic_state(solver.grid, solver.t)
        eqs = MHDEquations1D(gamma=self.gamma)
        _, _, _, _, _, _, Bya, Bza = eqs.primitive(Ua)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        axes[0].plot(x, By, "b-", lw=1.5, label="By (num)")
        axes[0].plot(x, Bya, "k--", lw=1.0, label="By (analytic)")
        axes[0].plot(x, Bz, "r-", lw=1.5, label="Bz (num)")
        axes[0].plot(x, Bza, "g--", lw=1.0, label="Bz (analytic)")
        axes[0].set_xlabel("x"); axes[0].set_ylabel("B_perp")
        axes[0].set_title(f"CP Alfven wave at t={solver.t:.4f}")
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        err = np.sqrt(np.mean((By - Bya) ** 2 + (Bz - Bza) ** 2))
        axes[1].semilogy(x, np.abs(By - Bya) + 1e-300, "b-", label="|By err|")
        axes[1].semilogy(x, np.abs(Bz - Bza) + 1e-300, "r-", label="|Bz err|")
        axes[1].set_xlabel("x"); axes[1].set_ylabel("abs error")
        axes[1].set_title(f"L2(B_perp) error = {err:.3e}")
        axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.outdir, f"alfven1d_t{solver.t:.4f}.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()

    def get_solver_class(self):
        return SpectralSolverMHD1D
