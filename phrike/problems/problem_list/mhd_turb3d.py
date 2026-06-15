"""3D MHD turbulence: turbulent velocity + banded random solenoidal B field.

Reuses the turbulent velocity generator from turb3d for u, builds a band-limited
random magnetic field, projects it to be exactly divergence-free, adds a uniform
mean field b0, and normalizes the fluctuations to a target b_rms.
"""

import os
import sys
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
from .turb3d import turbulent_velocity_3d


def banded_random_b_3d(grid: Grid3D, b_rms=0.1, kmin=1.0, kmax=5.0,
                       b0=(0.0, 0.0, 1.0), seed=123):
    """Band-limited, divergence-free random B with mean field b0, target b_rms.

    Returns numpy arrays (Bx, By, Bz) of shape (Nz, Ny, Nx).
    """
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    rng = np.random.default_rng(seed)
    # FFT-grid integer wavenumbers (cycles per box). The Leray projection is
    # scale-invariant so integer k (vs physical 2*pi/L * k) is fine here.
    kx = np.fft.fftfreq(Nx, d=1.0 / Nx).reshape(1, 1, Nx)
    ky = np.fft.fftfreq(Ny, d=1.0 / Ny).reshape(1, Ny, 1)
    kz = np.fft.fftfreq(Nz, d=1.0 / Nz).reshape(Nz, 1, 1)
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    band = (kmag >= kmin) & (kmag <= kmax)
    envelope = np.where(band, 1.0, 0.0)
    k2 = kx * kx + ky * ky + kz * kz
    k2_safe = np.where(k2 == 0.0, 1.0, k2)

    def field_hat():
        F = np.fft.fftn(rng.standard_normal((Nz, Ny, Nx))) * envelope
        F[0, 0, 0] = 0.0
        return F

    # Build band-limited spectral field, then project to div-free in NumPy
    # (independent of the grid backend so this works for torch/MPS grids too).
    FBx, FBy, FBz = field_hat(), field_hat(), field_hat()
    kdot = kx * FBx + ky * FBy + kz * FBz
    coef = kdot / k2_safe
    FBx = FBx - kx * coef
    FBy = FBy - ky * coef
    FBz = FBz - kz * coef
    Bx = np.fft.ifftn(FBx).real
    By = np.fft.ifftn(FBy).real
    Bz = np.fft.ifftn(FBz).real
    # normalize fluctuation rms before adding the (div-free) uniform mean field
    rms = float(np.sqrt(np.mean(Bx ** 2 + By ** 2 + Bz ** 2)))
    if rms > 0:
        s = float(b_rms) / rms
        Bx *= s; By *= s; Bz *= s
    Bx = Bx + float(b0[0]); By = By + float(b0[1]); Bz = Bz + float(b0[2])
    return Bx, By, Bz


class MHDTurb3DProblem(BaseProblem):
    """3D compressible MHD turbulence."""

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
        X, Y, Z = grid.xyz_mesh()
        icfg = self.config["initial_conditions"]
        # Hydro (velocity) part via the existing turbulence generator
        U_hydro = turbulent_velocity_3d(
            X, Y, Z,
            rho0=float(icfg.get("rho0", 1.0)), p0=float(icfg.get("p0", 1.0)),
            vrms=float(icfg.get("vrms", 0.1)), kmin=float(icfg.get("kmin", 2.0)),
            kmax=float(icfg.get("kmax", 16.0)), alpha=float(icfg.get("alpha", 0.3333)),
            spectrum_type=str(icfg.get("spectrum_type", "parabolic")),
            seed=int(icfg.get("seed", 42)), gamma=self.gamma,
        )
        rho = U_hydro[0]
        ux = U_hydro[1] / rho; uy = U_hydro[2] / rho; uz = U_hydro[3] / rho
        # Magnetic field
        Bx, By, Bz = banded_random_b_3d(
            grid, b_rms=float(icfg.get("b_rms", 0.1)),
            kmin=float(icfg.get("b_kmin", 1.0)), kmax=float(icfg.get("b_kmax", 5.0)),
            b0=tuple(icfg.get("b0", [0.0, 0.0, 1.0])), seed=int(icfg.get("b_seed", 123)),
        )
        p = float(icfg.get("p0", 1.0)) * np.ones_like(np.asarray(rho) if not hasattr(rho, "cpu") else rho.detach().cpu().numpy())
        eqs = MHDEquations3D(gamma=self.gamma)
        if hasattr(rho, "cpu"):
            import torch
            dev, dt = rho.device, rho.dtype
            to = lambda a: torch.from_numpy(np.asarray(a)).to(device=dev, dtype=dt)
            return eqs.conservative(rho, ux, uy, uz, to(p), to(Bx), to(By), to(Bz))
        return eqs.conservative(rho, ux, uy, uz, p, np.asarray(Bx), np.asarray(By), np.asarray(Bz))

    def create_visualization(self, solver, t, U) -> None:
        rho = self.convert_torch_to_numpy(solver.equations.primitive(U)[0])[0]
        mid = rho.shape[0] // 2
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
        im = ax.imshow(rho[mid], origin="lower",
                       extent=[0, solver.grid.Lx, 0, solver.grid.Ly], cmap="inferno")
        ax.set_title(f"MHD turb density (z=mid) t={t:.3f}")
        plt.colorbar(im, ax=ax, shrink=0.85)
        frames_dir = os.path.join(self.outdir, "frames")
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=120,
                    bbox_inches="tight")
        plt.close(fig)

    def create_final_visualization(self, solver) -> None:
        save_solution_snapshot(self.outdir, solver.t, U=solver.U, grid=solver.grid,
                               equations=solver.equations)

    def get_solver_class(self):
        return SpectralSolverMHD3D
