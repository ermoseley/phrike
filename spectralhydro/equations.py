from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):  # type: ignore
        def inner(func):
            return func
        return inner
    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)


# --- Numba-accelerated kernels ---

@njit(cache=True, fastmath=True)
def _primitive_kernel(U: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = U[0]
    mom = U[1]
    E = U[2]
    u = mom / rho
    kinetic = 0.5 * rho * u * u
    p = (gamma - 1.0) * (E - kinetic)
    a = np.sqrt(gamma * p / rho)
    return rho, u, p, a


@njit(cache=True, fastmath=True)
def _flux_kernel(U: np.ndarray, gamma: float) -> np.ndarray:
    rho, u, p, _ = _primitive_kernel(U, gamma)
    mom = rho * u
    F = np.empty_like(U)
    F[0] = mom
    F[1] = mom * u + p
    F[2] = (U[2] + p) * u
    return F


@njit(cache=True, fastmath=True)
def _max_wave_speed_kernel(U: np.ndarray, gamma: float) -> float:
    # U shape (3, N)
    rho, u, p, a = _primitive_kernel(U, gamma)
    c = np.abs(u) + a
    max_c = 0.0
    for i in range(c.shape[0]):
        if c[i] > max_c:
            max_c = c[i]
    return max_c


Array = np.ndarray


@dataclass
class EulerEquations1D:
    """Compressible Euler equations in 1D, conservative form.

    State vector U = [rho, mom, E], where:
      - rho: density
      - mom: momentum density (rho * u)
      - E: total energy density = p/(gamma-1) + 0.5*rho*u^2
    """

    gamma: float = 1.4

    def primitive(self, U: Array) -> Tuple[Array, Array, Array, Array]:
        return _primitive_kernel(U, self.gamma)

    def conservative(self, rho: Array, u: Array, p: Array) -> Array:
        mom = rho * u
        E = p / (self.gamma - 1.0) + 0.5 * rho * u * u
        return np.array([rho, mom, E])

    def flux(self, U: Array) -> Array:
        return _flux_kernel(U, self.gamma)

    def max_wave_speed(self, U: Array) -> float:
        return float(_max_wave_speed_kernel(U, self.gamma))

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        rho, u, p, _ = self.primitive(U)
        total_mass = float(np.sum(rho))
        total_momentum = float(np.sum(rho * u))
        total_energy = float(np.sum(p / (self.gamma - 1.0) + 0.5 * rho * u * u))
        return {
            "mass": total_mass,
            "momentum": total_momentum,
            "energy": total_energy,
        }


