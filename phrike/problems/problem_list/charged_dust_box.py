"""Uniform MHD gas, charged dust and active SGS in one to three dimensions."""

import numpy as np

from .dusty_box import DustyBoxProblem


class ChargedDustBoxProblem(DustyBoxProblem):
    """Box ICs use three-component velocity and magnetic_field vectors.

    Dust uses the inherited dust_velocity, sigma0 and dust_to_gas keys.
    charge_to_mass is configured under dust and defaults to zero.
    """

    def create_grid(self, backend="numpy", device=None, debug=False):
        grid = super().create_grid(backend=backend, device=device, debug=debug)
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        return grid

    def create_equations(self):
        from phrike.dusty_mhd import SGSMHDEquations1D, SGSMHDEquations2D, SGSMHDEquations3D

        return (SGSMHDEquations1D, SGSMHDEquations2D, SGSMHDEquations3D)[self.ndim-1](self.gamma)

    def get_solver_class(self):
        from phrike.dusty_mhd import DustyMHDSolver

        return DustyMHDSolver

    def create_initial_conditions(self, grid):
        ic = self.config.get("initial_conditions", {})
        shape = tuple(getattr(grid, k) for k in (("N",) if self.ndim == 1 else
                                               ("Ny", "Nx") if self.ndim == 2 else ("Nz", "Ny", "Nx")))
        velocity = np.asarray(ic.get("velocity", [0.0]*3), dtype=np.float64)
        magnetic = np.asarray(ic.get("magnetic_field", [1.0, 0.0, 0.0]), dtype=np.float64)
        if velocity.shape != (3,) or magnetic.shape != (3,):
            raise ValueError("velocity and magnetic_field must contain three components")
        rho = np.full(shape, float(ic.get("rho0", 1.0)))
        p = np.full(shape, float(ic.get("p0", 1.0)))
        u = [np.full(shape, v) for v in velocity]
        B = [np.full(shape, b) for b in magnetic]
        return self.create_equations().conservative(rho, *u, p, *B)

    def create_sgs_initial_conditions(self, grid, U_gas):
        ic = self.config.get("initial_conditions", {})
        internal = (U_gas[4] - 0.5*np.sum(U_gas[1:4]**2, axis=0)/U_gas[0]
                    - 0.5*np.sum(U_gas[5:8]**2, axis=0))
        value = ic.get("e_s0", float(ic.get("e_s_fraction", 0.0))*np.mean(internal))
        return np.full_like(U_gas[0], float(value))
