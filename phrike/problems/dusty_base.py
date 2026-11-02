"""NumPy-only problem runner for coupled gas, dust and active SGS.

``dust`` defaults: enabled=True, closure='gaussian', cfl=0.8,
backreaction=True, stopping={model: constant, t_s: 1},
charge_to_mass=0 (neutral; code units),
coupling_integrator='strang' ('exponential' selects the Gaussian ideal-MHD
averaged-flux step; its failed attempts use counted resolved Strang recovery),
eddy_diffusion={enabled: False, c_kappa: 1}. Epstein stopping instead takes
``grain_rho_a`` (default 1). HyQMOM is supported only in one dimension.

``sgs`` defaults: enabled=True, pressure=True, production='filter_drain',
decay=True, d_model='mt25', c_eps=1, c_ell=1, f_cap=0.5,
diffusion={enabled: False, c_kappa: 1}. Diffusion accepts a constant ``kappa``
or uses c_kappa*sigma*Delta. Forcing models: mt25, mt25_magnetized, white,
finite_tau, none. Charged MHD dust defaults to mt25_magnetized when omitted.
Turning SGS off requires zero initial e_s; turning dust off uses vacuum moments.
Fixed rk2/rk4, float64, periodic Fourier grids only. Snapshots are outputs,
not restart checkpoints. Video, tracers, gravity and artificial viscosity are
not implemented by this runner.
"""

from abc import abstractmethod
from pathlib import Path

import numpy as np

from phrike.dusty_hydro import DustyHydroSolver
from .base import BaseProblem


class DustyProblemBase(BaseProblem):
    @abstractmethod
    def create_dust_initial_conditions(self, grid, U_gas, closure):
        """Return realizable raw moments on the gas collocation grid."""

    def create_sgs_initial_conditions(self, grid, U_gas):
        return np.zeros_like(U_gas[0])

    def get_solver_class(self):
        return DustyHydroSolver

    def create_visualization(self, solver, t, U):
        pass

    def run(self, backend="auto", device=None, generate_video=False, debug=False):
        if str(backend).lower() not in ("auto", "numpy", "cpu") or device not in (None, "cpu"):
            raise ValueError("dusty problems require NumPy float64 on CPU; torch is unsupported")
        if self.adaptive_config and self.adaptive_config.get("enabled", False):
            raise ValueError("dusty problems do not support adaptive integration")
        if self.restart_from is not None:
            raise ValueError("coupled dust/SGS restart is not implemented")
        if generate_video or self.config.get("tracers", {}).get("enabled", False):
            raise ValueError("dusty problems do not support video or tracers")
        for name in ("gravity", "artificial_viscosity", "initial_conditions_smoothing"):
            if self.config.get(name, {}).get("enabled", False):
                raise ValueError(f"{name} is unsupported by the dusty runner")
        grid = self.create_grid(backend="numpy", device=None, debug=debug)
        solver = self.get_solver_class()(grid, self.create_equations(), scheme=self.scheme, cfl=self.cfl,
                                         dust_config=self.config.get("dust", {}),
                                         sgs_config=self.config.get("sgs", {}))
        solver.U_gas = np.asarray(self.create_initial_conditions(grid), dtype=np.float64)
        if solver.dust_enabled:
            solver.U_dust = np.asarray(self.create_dust_initial_conditions(
                grid, solver.U_gas, solver.closure), dtype=np.float64)
        if solver.sgs_enabled:
            solver.e_s = np.asarray(self.create_sgs_initial_conditions(grid, solver.U_gas), dtype=np.float64)
        gas_names = (("rho", "rho_ux", "rho_uy", "rho_uz", "E", "Bx", "By", "Bz")
                     if solver.U_gas.shape[0] == 8 else
                     ("rho", *(f"rho_u{a}" for a in "xyz"[:solver.ndim]), "E"))
        snapshot_index = 0

        def output(t, U):
            nonlocal snapshot_index
            self.on_output(solver, t, U)
            np.savez(Path(self.outdir)/f"snapshot_{snapshot_index:05d}.npz", t=t,
                     gas=U, dust=solver.U_dust, e_s=solver.e_s, closure=solver.closure.name,
                     field_names=solver.closure.field_names, dust_field_names=solver.closure.field_names,
                     gas_field_names=gas_names)
            snapshot_index += 1

        history = solver.run(self.t0, self.t_end, self.output_interval, on_output=output)
        np.savez(Path(self.outdir)/"history.npz", **history)
        self.create_final_visualization(solver)
        return solver, history
