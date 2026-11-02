"""Decaying 3D gas turbulence using PHRIKE's existing velocity generator."""

from .dusty_box import DustyBoxProblem
from .turb3d import turbulent_velocity_3d


class DustyTurb3DProblem(DustyBoxProblem):
    def create_initial_conditions(self, grid):
        if self.ndim != 3:
            raise ValueError("dusty_turb3d requires a 3D grid")
        ic = self.config.get("initial_conditions", {})
        X, Y, Z = grid.xyz_mesh()
        return turbulent_velocity_3d(X, Y, Z, gamma=self.gamma,
            **{key: ic[key] for key in ("rho0", "p0", "vrms", "kmin", "kmax", "alpha",
                                      "spectrum_type", "power_law_slope", "seed") if key in ic})
