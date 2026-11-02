"""Canonical Orszag-Tang vortex with co-moving charged dust and active SGS."""

from .charged_dust_box import ChargedDustBoxProblem
from .dusty_box import dust_state
from .orszag_tang2d import orszag_tang_2d


class DustyOrszagTang2DProblem(ChargedDustBoxProblem):
    def create_initial_conditions(self, grid):
        if self.ndim != 2:
            raise ValueError("dusty_orszag_tang2d requires a 2D grid")
        X, Y = grid.xy_mesh()
        ic = self.config.get("initial_conditions", {})
        return orszag_tang_2d(X, Y, gamma=self.gamma, Lx=grid.Lx, Ly=grid.Ly,
                              rho0=ic.get("rho0"), p0=ic.get("p0"), B0=ic.get("B0"))

    def create_dust_initial_conditions(self, grid, U_gas, closure):
        ic = self.config.get("initial_conditions", {})
        return dust_state(closure, float(ic.get("dust_to_gas", 0.01))*U_gas[0],
                          U_gas[1:4]/U_gas[0], float(ic.get("sigma0", 0.01)))
