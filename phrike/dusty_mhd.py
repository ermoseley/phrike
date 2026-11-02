"""Charged raw-moment dust and an active SGS reservoir in periodic MHD.

Only kinetic energy removed by the spectral filter feeds SGS. Magnetic energy
removed by that filter remains in the gas total energy as thermal energy.
There is no magnetic SGS reservoir. The default Strang step resolves coherent
gyration; the optional exponential step uses averaged raw states for fluxes.
"""

from dataclasses import dataclass, field

import numpy as np

from .dusty_hydro import DustyHydroSolver, _SGSFlux
from .equations import MHDEquations1D, MHDEquations2D, MHDEquations3D
from .solver import SpectralSolverMHD


class _SGSMHDFlux(_SGSFlux):
    energy_index = 4

    def max_wave_speed(self, U):
        if self.p_sgs is None:
            return super().max_wave_speed(U)
        p = self.primitive(U)[4]
        # sqrt(a^2 + b^2 + c_sgs^2) bounds every fast magnetosonic speed.
        fast = np.sqrt((self.gamma*p + np.sum(U[5:8]**2, axis=0)
                        + (5/3)*self.p_sgs)/U[0])
        return float(np.max(np.max(np.abs(U[1:1+self.ndim]/U[0]), axis=0) + fast))


@dataclass
class SGSMHDEquations1D(_SGSMHDFlux, MHDEquations1D):
    ndim = 1
    p_sgs: np.ndarray | None = field(default=None, kw_only=True)


@dataclass
class SGSMHDEquations2D(_SGSMHDFlux, MHDEquations2D):
    ndim = 2
    p_sgs: np.ndarray | None = field(default=None, kw_only=True)


@dataclass
class SGSMHDEquations3D(_SGSMHDFlux, MHDEquations3D):
    ndim = 3
    p_sgs: np.ndarray | None = field(default=None, kw_only=True)


class DustyMHDSolver(DustyHydroSolver):
    """Fixed-step ideal/resistive MHD with exact local drag and gyration.

    The MHD state is [rho, mx, my, mz, E, Bx, By, Bz] in every spatial
    dimension. ``dust.charge_to_mass`` is signed, in code units (default 0).
    Charged dust defaults to ``sgs.d_model='mt25_magnetized'``. Gaussian
    moments allow arbitrary fields; HyQMOM requires a straight field along x.
    ``dust.coupling_integrator='exponential'`` requires Gaussian ideal MHD.
    Its acceptance/recovery counters must accompany any AP accuracy claim.
    """

    _is_mhd = True
    energy_index = 4
    _equation_types = (SGSMHDEquations1D, SGSMHDEquations2D, SGSMHDEquations3D)
    _gas_solver_types = (SpectralSolverMHD,)*3

    def __init__(self, grid, equations, **kwargs):
        cfg = getattr(grid, "mhd_config", None) or {}
        for name in ("hall", "ambipolar", "shearing_sheet"):
            option = cfg.get(name) or {}
            if option and option.get("enabled", name == "shearing_sheet"):
                raise ValueError(f"DustyMHDSolver does not support {name}")
        super().__init__(grid, equations, **kwargs)
        if self.coupling_integrator == "exponential" and self.gas_solver.resistivity != 0:
            raise ValueError("exponential coupling requires ideal MHD (zero resistivity)")
        charge = float(self.dust_config.get("charge_to_mass", 0.0))
        if not np.isfinite(charge):
            raise ValueError("dust.charge_to_mass must be finite")

    def _default_d_model(self):
        return "mt25_magnetized" if self.dust_config.get("charge_to_mass", 0.0) else "mt25"

    def _internal_energy(self, U):
        return super()._internal_energy(U) - 0.5*np.sum(U[5:8]**2, axis=0)

    def _validate_state(self):
        super()._validate_state()
        if self.closure.name == "hyqmom":
            magnitude = np.sqrt(np.sum(self.U_gas[5:8]**2, axis=0))
            if np.any(np.abs(self.U_gas[6:8]) > 1e-12*magnitude):
                raise ValueError("field-aligned HyQMOM requires B parallel to x")

    def _gas_step(self, dt):
        s, U = self.gas_solver, self.U_gas
        if s.hall is not None or s.ambipolar is not None or s.shearing_sheet is not None:
            raise ValueError("DustyMHDSolver does not support Hall, ambipolar or shearing-sheet sources")
        split = self.grid.apply_spectral_dissipation(U, 0.5*dt)
        un = s._project_B(s._rk_substages(split, dt))
        accepted = s._limit_accepted(self.grid.apply_accepted_state_stabilization(un, 0.5*dt))
        drop = ((self._kinetic(U) - self._kinetic(split))
                + (self._kinetic(un) - self._kinetic(accepted)))
        if self.ndim == 1:
            accepted = self.grid.apply_boundary_conditions(accepted, self.equations)
        self.U_gas = accepted
        return drop

    def diagnostics(self, dt=0.0):
        result = super().diagnostics(dt)
        result["E_magnetic"] = self._integral(0.5*np.sum(self.U_gas[5:8]**2, axis=0))
        result["div_b_max"] = self.gas_solver.max_div_b(self.U_gas)
        return result
