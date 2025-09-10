from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


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
        rho = U[0]
        mom = U[1]
        E = U[2]
        u = mom / rho
        kinetic = 0.5 * rho * u * u
        p = (self.gamma - 1.0) * (E - kinetic)
        a = np.sqrt(self.gamma * p / rho)
        return rho, u, p, a

    def conservative(self, rho: Array, u: Array, p: Array) -> Array:
        mom = rho * u
        E = p / (self.gamma - 1.0) + 0.5 * rho * u * u
        return np.array([rho, mom, E])

    def flux(self, U: Array) -> Array:
        rho, u, p, _ = self.primitive(U)
        mom = rho * u
        F1 = mom
        F2 = mom * u + p
        F3 = (U[2] + p) * u
        return np.array([F1, F2, F3])

    def max_wave_speed(self, U: Array) -> float:
        _, u, _, a = self.primitive(U)
        return float(np.max(np.abs(u) + a))

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


