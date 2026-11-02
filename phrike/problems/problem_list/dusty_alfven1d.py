"""Small circularly polarized Alfven wave with cold, co-moving charged dust."""

import numpy as np

from .charged_dust_box import ChargedDustBoxProblem
from .dusty_box import dust_state


class DustyAlfven1DProblem(ChargedDustBoxProblem):
    """A is the transverse velocity amplitude; B0 is the longitudinal field.

    The gas starts in the ideal-MHD travelling mode, with co-moving dust.
    Dust back-reaction then excites the coupled linear modes. Gaussian closure
    is required because the perturbed magnetic field is oblique.
    """

    def create_initial_conditions(self, grid):
        if self.ndim != 1:
            raise ValueError("dusty_alfven1d requires a 1D grid")
        ic = self.config.get("initial_conditions", {})
        rho0 = float(ic.get("rho0", 1.0))
        phase = 2*np.pi*int(ic.get("k", 1))*grid.x/grid.Lx
        amplitude = float(ic.get("A", 1e-6))
        direction = int(ic.get("direction", 1))
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 or -1")
        rho = np.full_like(grid.x, rho0)
        p = np.full_like(grid.x, float(ic.get("p0", 1.0)))
        zero = np.zeros_like(grid.x)
        uy, uz = amplitude*np.cos(phase), amplitude*np.sin(phase)
        Bx = np.full_like(grid.x, float(ic.get("B0", 1.0)))
        return self.create_equations().conservative(
            rho, zero, uy, uz, p, Bx, -direction*np.sqrt(rho0)*uy, -direction*np.sqrt(rho0)*uz)

    def create_dust_initial_conditions(self, grid, U_gas, closure):
        ic = self.config.get("initial_conditions", {})
        return dust_state(closure, float(ic.get("dust_to_gas", 0.01))*U_gas[0],
                          U_gas[1:4]/U_gas[0])
