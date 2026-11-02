"""Nearly collisionless opposing cold beams or one Gaussian-velocity blob."""

import numpy as np

from .dusty_box import DustyBoxProblem, dust_state


class DustBeams1DProblem(DustyBoxProblem):
    def create_dust_initial_conditions(self, grid, U_gas, closure):
        if self.ndim != 1:
            raise ValueError("dust_beams1d requires a 1D grid")
        ic = self.config.get("initial_conditions", {})
        v0, width = float(ic.get("v0", 0.5)), float(ic.get("width", 0.07))*grid.Lx
        if width <= 0:
            raise ValueError("beam width must be positive")
        profile = ic.get("profile", "two_streams")
        scale = float(ic.get("dust_to_gas", 0.01))*float(ic.get("rho0", 1.0))

        def blob(centre, velocity, dispersion):
            distance = (grid.x-centre+0.5*grid.Lx) % grid.Lx - 0.5*grid.Lx
            rho = scale*(float(ic.get("background", 1e-4)) + np.exp(-0.5*(distance/width)**2))
            u = np.zeros((3, grid.N))
            u[0] = velocity
            return dust_state(closure, rho, u, dispersion)

        if profile == "two_streams":
            return blob(0.25*grid.Lx, v0, 0.0) + blob(0.75*grid.Lx, -v0, 0.0)
        if profile == "gaussian":
            return blob(float(ic.get("centre", 0.5))*grid.Lx, v0, float(ic.get("sigma0", 0.1)))
        raise ValueError("beam profile must be 'two_streams' or 'gaussian'")
