from __future__ import annotations

from typing import Dict

import numpy as np

from .equations import EulerEquations1D


def sod_shock_tube(
    x: np.ndarray,
    x0: float,
    left: Dict[str, float],
    right: Dict[str, float],
    gamma: float,
) -> np.ndarray:
    """Return conservative state U for the Sod shock tube initial condition.

    left/right dictionaries should include keys: rho, u, p
    """
    rho = np.where(x < x0, left["rho"], right["rho"])  # type: ignore[index]
    u = np.where(x < x0, left["u"], right["u"])  # type: ignore[index]
    p = np.where(x < x0, left["p"], right["p"])  # type: ignore[index]
    eqs = EulerEquations1D(gamma=gamma)
    return eqs.conservative(rho, u, p)


def sinusoidal_density(
    x: np.ndarray,
    rho0: float = 1.0,
    u0: float = 0.0,
    p0: float = 1.0,
    amplitude: float = 1e-3,
    k: int = 1,
    Lx: float = 1.0,
    gamma: float = 1.4,
) -> np.ndarray:
    rho = rho0 * (1.0 + amplitude * np.sin(2.0 * np.pi * k * x / Lx))
    u = u0 * np.ones_like(x)
    p = p0 * np.ones_like(x)
    eqs = EulerEquations1D(gamma=gamma)
    return eqs.conservative(rho, u, p)


