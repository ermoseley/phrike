"""Small gas acoustic wave with cold, initially co-moving dust."""

import numpy as np

from .acoustic1d import acoustic_primitives
from .dusty_box import DustyBoxProblem, dust_state


class DustyWave1DProblem(DustyBoxProblem):
    def create_initial_conditions(self, grid):
        if self.ndim != 1:
            raise ValueError("dustywave1d requires a 1D grid")
        ic = self.config.get("initial_conditions", {})
        rho, u, p = acoustic_primitives(grid.x, float(ic.get("rho0", 1.0)), 0.0,
            float(ic.get("p0", 1.0)), float(ic.get("A", 1e-4)), int(ic.get("k", 1)),
            grid.Lx, self.gamma)
        return self.create_equations().conservative(rho, u, p)

    def create_dust_initial_conditions(self, grid, U_gas, closure):
        ic = self.config.get("initial_conditions", {})
        u = np.zeros((3, *U_gas[0].shape))
        u[0] = U_gas[1]/U_gas[0]
        return dust_state(closure, float(ic.get("dust_to_gas", 1.0))*U_gas[0], u)
