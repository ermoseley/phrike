"""Initial conditions of the distributed solvers.

Each problem names the system it runs on and builds this rank's part of the
initial state from the grid and the ``initial_conditions`` block:

``euler`` / ``mhd``
    compressible; the builder returns the conserved variables on this rank's
    slab, ``(5 | 8, nz, Ny, Nx)``.
``ins`` / ``imhd``
    incompressible; the builder returns ``("physical", fields)`` or
    ``("spectral", spectrum)`` for ``u`` (and ``b``).

Analytic problems reuse the functions of ``problems/problem_list`` on the
local mesh.  Random fields come from ``parallel.fields`` and are identical for
any rank count, which the full-grid NumPy generators of the serial problems
cannot be.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np
import torch

from ..equations import EulerEquations3D, MHDEquations3D
from .fields import random_vector_spectrum
from .grid import SlabGrid3D

COMPRESSIBLE = ("euler", "mhd")
INCOMPRESSIBLE = ("ins", "imhd")


@dataclass
class ParallelProblem:
    name: str
    system: str
    build: Callable[[SlabGrid3D, Dict[str, Any], float], Any]
    description: str


_REGISTRY: Dict[str, ParallelProblem] = {}


def register(name: str, system: str, description: str):
    def wrap(fn):
        _REGISTRY[name] = ParallelProblem(name, system, fn, description)
        return fn

    return wrap


def get_problem(name: str) -> ParallelProblem:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown distributed problem {name!r}. Available: {', '.join(list_problems())}") from None


def list_problems() -> List[str]:
    return sorted(_REGISTRY)


def _velocity_spectrum(grid: SlabGrid3D, icfg: Dict[str, Any], compressive: float) -> torch.Tensor:
    return random_vector_spectrum(
        grid, seed=int(icfg.get("seed", 42)), kmin=float(icfg.get("kmin", 2.0)),
        kmax=float(icfg.get("kmax", 16.0)), rms=float(icfg.get("vrms", 0.1)),
        spectrum=str(icfg.get("spectrum_type", "parabolic")),
        slope=float(icfg.get("power_law_slope", -5.0 / 3.0)), compressive_fraction=compressive,
    )


def _magnetic_spectrum(grid: SlabGrid3D, icfg: Dict[str, Any]) -> torch.Tensor:
    """Solenoidal random field of rms ``b_rms`` plus the uniform field ``b0``."""
    S = random_vector_spectrum(
        grid, seed=int(icfg.get("b_seed", 123)), kmin=float(icfg.get("b_kmin", 1.0)),
        kmax=float(icfg.get("b_kmax", 5.0)), rms=float(icfg.get("b_rms", 0.1)),
        spectrum=str(icfg.get("b_spectrum_type", "flat")),
        slope=float(icfg.get("b_power_law_slope", -5.0 / 3.0)), compressive_fraction=0.0,
    )
    b0 = [float(v) for v in icfg.get("b0", [0.0, 0.0, 0.0])]
    if len(b0) != 3:
        raise ValueError("initial_conditions.b0 must have three components (x, y, z)")
    if grid.decomp.ky0 == 0:  # the rank that owns the k = 0 mode
        for c in range(3):
            S[c, 0, 0, 0] = b0[c]
    return S


# --- compressible ---------------------------------------------------------------


@register("orszag_tang3d", "mhd", "3D Orszag-Tang vortex (Helzel, Rossmanith & Taetz 2011)")
def _orszag_tang3d(grid, icfg, gamma):
    from ..problems.problem_list.orszag_tang3d import orszag_tang_3d

    return orszag_tang_3d(*grid.xyz_mesh(), gamma=gamma, Lx=grid.Lx, Ly=grid.Ly, Lz=grid.Lz,
                          epsilon=float(icfg.get("epsilon", 0.2)), rho0=icfg.get("rho0"),
                          p0=icfg.get("p0"), B0=icfg.get("B0"))


@register("alfven3d", "mhd", "circularly polarized Alfven wave along x or the box diagonal")
def _alfven3d(grid, icfg, gamma):
    from ..problems.problem_list.alfven3d import circularly_polarized_alfven_3d

    X, Y, Z = grid.xyz_mesh()
    return circularly_polarized_alfven_3d(
        X, t=0.0, rho0=float(icfg.get("rho0", 1.0)), p0=float(icfg.get("p0", 0.1)),
        B0=float(icfg.get("B0", 1.0)), b1=float(icfg.get("b1", 0.1)), m=int(icfg.get("m", 1)),
        Lx=grid.Lx, direction=int(icfg.get("direction", 1)), gamma=gamma,
        orientation=str(icfg.get("orientation", "x")), Y=Y, Z=Z, Ly=grid.Ly, Lz=grid.Lz)


@register("tgv3d", "euler", "compressible Taylor-Green vortex")
def _tgv3d(grid, icfg, gamma):
    from ..problems.problem_list.tgv3d import taylor_green_vortex_3d

    rho0, U0 = float(icfg.get("rho0", 1.0)), float(icfg.get("U0", 1.0))
    if "mach" in icfg:
        p0 = rho0 * U0 * U0 / (gamma * float(icfg["mach"]) ** 2)
    else:
        p0 = float(icfg.get("p0", 100.0))
    return taylor_green_vortex_3d(*grid.xyz_mesh(), rho0=rho0, p0=p0, U0=U0,
                                  k=int(icfg.get("k", 1)), gamma=gamma)


@register("turb3d", "euler", "compressible turbulence from a random velocity field")
def _turb3d(grid, icfg, gamma):
    u = grid.inverse(_velocity_spectrum(grid, icfg, float(icfg.get("alpha", 0.3333))))
    rho = torch.full_like(u[0], float(icfg.get("rho0", 1.0)))
    p = torch.full_like(u[0], float(icfg.get("p0", 1.0)))
    return EulerEquations3D(gamma=gamma).conservative(rho, u[0], u[1], u[2], p)


@register("mhd_turb3d", "mhd", "compressible MHD turbulence: random velocity and banded solenoidal B")
def _mhd_turb3d(grid, icfg, gamma):
    u = grid.inverse(_velocity_spectrum(grid, icfg, float(icfg.get("alpha", 0.3333))))
    B = grid.inverse(_magnetic_spectrum(grid, dict({"b0": [0.0, 0.0, 1.0]}, **icfg)))
    rho = torch.full_like(u[0], float(icfg.get("rho0", 1.0)))
    p = torch.full_like(u[0], float(icfg.get("p0", 1.0)))
    return MHDEquations3D(gamma=gamma).conservative(rho, u[0], u[1], u[2], p, B[0], B[1], B[2])


# --- incompressible -------------------------------------------------------------


@register("imhd_turb", "imhd",
          "incompressible MHD turbulence: random u, seed or mean magnetic field, optional forcing")
def _imhd_turb(grid, icfg, gamma):
    cfg = dict({"kmax": 4.0, "kmin": 1.0, "vrms": 1.0, "b_rms": 1e-3, "b_kmax": 4.0}, **icfg)
    return "spectral", torch.cat([_velocity_spectrum(grid, cfg, 0.0), _magnetic_spectrum(grid, cfg)])


@register("ins_turb", "ins", "incompressible hydrodynamic turbulence: random u, optional forcing")
def _ins_turb(grid, icfg, gamma):
    cfg = dict({"kmax": 4.0, "kmin": 1.0, "vrms": 1.0}, **icfg)
    return "spectral", _velocity_spectrum(grid, cfg, 0.0)


@register("imhd_orszag_tang", "imhd", "incompressible 3D Orszag-Tang vortex (Politano, Pouquet & Sulem 1995)")
def _imhd_orszag_tang(grid, icfg, gamma):
    X, Y, Z = grid.xyz_mesh()
    x, y, z = (2.0 * np.pi / grid.Lx) * X, (2.0 * np.pi / grid.Ly) * Y, (2.0 * np.pi / grid.Lz) * Z
    beta = float(icfg.get("beta", 0.8))
    u = [-2.0 * torch.sin(y), 2.0 * torch.sin(x), torch.zeros_like(x)]
    b = [beta * (-2.0 * torch.sin(2.0 * y) + torch.sin(z)), beta * (2.0 * torch.sin(x) + torch.sin(z)),
         beta * (torch.sin(x) + torch.sin(y))]
    return "physical", torch.stack(u + b)


@register("imhd_alfven", "imhd", "circularly polarized Alfven wave along x on a uniform field")
def _imhd_alfven(grid, icfg, gamma):
    X, _, _ = grid.xyz_mesh()
    B0, b1 = float(icfg.get("B0", 1.0)), float(icfg.get("b1", 0.1))
    phase = (2.0 * np.pi * int(icfg.get("m", 1)) / grid.Lx) * X
    sign = -float(int(icfg.get("direction", 1)))
    by, bz = b1 * torch.cos(phase), b1 * torch.sin(phase)
    zero = torch.zeros_like(X)
    return "physical", torch.stack([zero, sign * by, sign * bz, torch.full_like(X, B0), by, bz])


@register("ins_taylor_green", "ins", "incompressible Taylor-Green vortex (Brachet et al. 1983)")
def _ins_taylor_green(grid, icfg, gamma):
    X, Y, Z = grid.xyz_mesh()
    k, U0 = int(icfg.get("k", 1)), float(icfg.get("U0", 1.0))
    x, y, z = (2.0 * np.pi * k / grid.Lx) * X, (2.0 * np.pi * k / grid.Ly) * Y, (2.0 * np.pi * k / grid.Lz) * Z
    return "physical", torch.stack([U0 * torch.sin(x) * torch.cos(y) * torch.cos(z),
                                    -U0 * torch.cos(x) * torch.sin(y) * torch.cos(z),
                                    torch.zeros_like(x)])
