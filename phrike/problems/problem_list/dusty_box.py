"""Uniform coupled box and shared periodic dusty-problem setup."""

from pathlib import Path

import numpy as np

from phrike.dusty_hydro import SGSEulerEquations1D, SGSEulerEquations2D, SGSEulerEquations3D
from phrike.grid import Grid1D, Grid2D, Grid3D
from ..dusty_base import DustyProblemBase


def dust_state(closure, rho, u, sigma0=0.0, S3=0.0, S4=3.0):
    """Isotropic dispersion, with optional realizable HyQMOM skew/kurtosis."""
    if sigma0 < 0:
        raise ValueError("sigma0 must be non-negative")
    if closure.name == "hyqmom":
        if S4 < 1 + S3*S3:
            raise ValueError("HyQMOM requires S4 >= 1 + S3**2")
        return closure.from_primitives(rho, u, sigma0**2, S3*sigma0**3,
                                       S4*sigma0**4, sigma0**2)
    C = np.zeros((3, 3, *rho.shape))
    for d in range(3):
        C[d, d] = sigma0**2
    return closure.from_primitives(rho, u, C)


class DustyBoxProblem(DustyProblemBase):
    """Uniform gas/dust velocities and per-component dust dispersion.

    IC keys: rho0, p0, velocity (resolved gas components), dust_to_gas,
    dust_velocity (three components), sigma0, S3, S4, e_s0 (absolute) or
    e_s_fraction (fraction of the mean gas internal energy density).
    """

    @property
    def ndim(self):
        g = self.config["grid"]
        return 3 if "Nz" in g else 2 if "Ny" in g else 1

    def create_grid(self, backend="numpy", device=None, debug=False):
        g = self.config["grid"]
        common = dict(backend=backend, torch_device=device, precision=self.precision,
                      dealias=g.get("dealias", True), filter_params=self.filter_config,
                      fft_workers=self.fft_workers)
        if self.ndim == 1:
            return Grid1D(int(g["N"]), float(g["Lx"]), basis=g.get("basis", "fourier"),
                          bc=g.get("bc"), **common)
        if self.ndim == 2:
            return Grid2D(int(g["Nx"]), int(g["Ny"]), float(g["Lx"]), float(g["Ly"]),
                          basis_x=g.get("basis_x", g.get("basis", "fourier")),
                          basis_y=g.get("basis_y", g.get("basis", "fourier")), **common)
        if g.get("basis", "fourier") != "fourier":
            raise ValueError("dusty problems require periodic Fourier grids")
        return Grid3D(*(int(g[k]) for k in ("Nx", "Ny", "Nz")),
                      *(float(g[k]) for k in ("Lx", "Ly", "Lz")), **common)

    def create_equations(self):
        return (SGSEulerEquations1D, SGSEulerEquations2D, SGSEulerEquations3D)[self.ndim-1](self.gamma)

    def create_initial_conditions(self, grid):
        ic = self.config.get("initial_conditions", {})
        shape = tuple(getattr(grid, k) for k in (("N",) if self.ndim == 1 else
                                               ("Ny", "Nx") if self.ndim == 2 else ("Nz", "Ny", "Nx")))
        rho = np.full(shape, float(ic.get("rho0", 1.0)))
        p = np.full(shape, float(ic.get("p0", 1.0)))
        velocity = ic.get("velocity", [0.0]*self.ndim)
        if len(velocity) != self.ndim:
            raise ValueError("gas velocity must contain one component per resolved dimension")
        return self.create_equations().conservative(rho, *(np.full(shape, float(v)) for v in velocity), p)

    def create_dust_initial_conditions(self, grid, U_gas, closure):
        ic = self.config.get("initial_conditions", {})
        velocity = np.asarray(ic.get("dust_velocity", [0.0]*3), dtype=np.float64)
        if velocity.shape != (3,):
            raise ValueError("dust_velocity must contain three components")
        u = np.broadcast_to(velocity.reshape((3,) + (1,)*self.ndim), (3, *U_gas[0].shape))
        rho = float(ic.get("dust_to_gas", 0.01))*np.full_like(U_gas[0], float(ic.get("rho0", 1.0)))
        return dust_state(closure, rho, u, float(ic.get("sigma0", 0.0)),
                          float(ic.get("S3", 0.0)), float(ic.get("S4", 3.0)))

    def create_sgs_initial_conditions(self, grid, U_gas):
        ic = self.config.get("initial_conditions", {})
        internal = U_gas[-1] - 0.5*np.sum(U_gas[1:-1]**2, axis=0)/U_gas[0]
        value = ic.get("e_s0", float(ic.get("e_s_fraction", 0.0))*np.mean(internal))
        return np.full_like(U_gas[0], float(value))

    def create_final_visualization(self, solver):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, 5), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(2, 2).ravel()
        p = solver.closure.primitives(solver.U_dust)
        variance = (np.trace(p["C"], axis1=0, axis2=1)/3 if solver.closure.name == "gaussian"
                    else (p["v"] + 2*p["sperp2"])/3)
        if solver.ndim == 1:
            axes[0].plot(solver.grid.x, solver.U_gas[0], label="gas")
            axes[0].plot(solver.grid.x, p["rho"], label="dust")
            axes[0].legend()
            axes[1].plot(solver.grid.x, solver.U_gas[1]/solver.U_gas[0], label="gas")
            axes[1].plot(solver.grid.x, p["u"][0], label="dust")
            axes[1].legend()
            axes[2].plot(solver.grid.x, solver.e_s)
            axes[3].plot(solver.grid.x, variance)
            labels = ("Density", "x velocity", "SGS energy density", "Dust variance")
        else:
            for ax, data in zip(axes, (solver.U_gas[0], p["rho"], solver.e_s, variance)):
                image = ax.imshow(data[data.shape[0]//2] if solver.ndim == 3 else data,
                                  origin="lower", aspect="auto")
                fig.colorbar(image, ax=ax)
            labels = ("Gas density", "Dust density", "SGS energy density", "Dust variance")
        for ax, label in zip(axes, labels):
            ax.set_title(label)
        fig.suptitle(f"{self._problem_label()}, t={solver.t:.4g}")
        fig.savefig(Path(self.outdir)/"diagnostics.png", dpi=100)
