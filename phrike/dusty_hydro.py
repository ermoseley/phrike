"""Periodic NumPy gas, realizable raw dust moments and active SGS evolution.

See ``problems.dusty_base`` for configuration defaults. Assign ``U_gas``,
``U_dust`` and ``e_s`` before stepping; dust and SGS default to vacuum/zero.
Callbacks follow SpectralSolver: output(t, U_gas), step(index, dt, U_gas).
"""

from dataclasses import dataclass, field

import numpy as np

from . import dust_moments, dust_sources, sgs
from .equations import EulerEquations1D, EulerEquations2D, EulerEquations3D
from .grid import Grid1D, Grid2D, Grid3D
from . import solver as gas_solver


class _SGSFlux:
    energy_index = -1

    def flux(self, U):
        base = super().flux(U)
        if self.p_sgs is None:
            return base
        fluxes = (base,) if self.ndim == 1 else base
        result = []
        for d, flux in enumerate(fluxes):
            flux = flux.copy()
            flux[1+d] += self.p_sgs
            flux[self.energy_index] += self.p_sgs * U[1+d]/U[0]
            result.append(flux)
        return result[0] if self.ndim == 1 else tuple(result)

    def max_wave_speed(self, U):
        if self.p_sgs is None:
            return super().max_wave_speed(U)
        p = self.primitive(U)[2 if self.ndim == 1 else -1]
        c = np.sqrt((self.gamma*p + (5/3)*self.p_sgs)/U[0])
        return float(np.max(np.max(np.abs(U[1:1+self.ndim]/U[0]), axis=0) + c))


@dataclass
class SGSEulerEquations1D(_SGSFlux, EulerEquations1D):
    ndim = 1
    p_sgs: np.ndarray | None = field(default=None, kw_only=True)


@dataclass
class SGSEulerEquations2D(_SGSFlux, EulerEquations2D):
    ndim = 2
    p_sgs: np.ndarray | None = field(default=None, kw_only=True)


@dataclass
class SGSEulerEquations3D(_SGSFlux, EulerEquations3D):
    ndim = 3
    p_sgs: np.ndarray | None = field(default=None, kw_only=True)


class DustyHydroSolver:
    """Strang-split fixed-step gas, dust and SGS solver (float64, Fourier only)."""

    _is_mhd = False
    energy_index = -1
    _equation_types = (SGSEulerEquations1D, SGSEulerEquations2D, SGSEulerEquations3D)
    _gas_solver_types = (gas_solver.SpectralSolver1D, gas_solver.SpectralSolver2D,
                         gas_solver.SpectralSolver3D)

    def __init__(self, grid, equations, *, scheme="rk4", cfl,
                 dust_config, sgs_config):
        if scheme.lower() not in ("rk2", "rk4"):
            raise ValueError("DustyHydroSolver requires fixed rk2 or rk4; adaptive schemes are unsupported")
        if not np.isfinite(cfl) or not 0 < cfl <= 1:
            raise ValueError("cfl must lie in (0, 1]")
        if getattr(grid, "_use_torch", False) or getattr(grid, "backend", "numpy") != "numpy":
            raise TypeError("DustyHydroSolver requires NumPy float64, not torch")
        if isinstance(grid, Grid3D):
            self.ndim, self.shape = 3, (grid.Nz, grid.Ny, grid.Nx)
        elif isinstance(grid, Grid2D):
            self.ndim, self.shape = 2, (grid.Ny, grid.Nx)
        elif isinstance(grid, Grid1D):
            self.ndim, self.shape = 1, (grid.N,)
        else:
            raise TypeError("expected Grid1D, Grid2D or Grid3D")
        bases = [getattr(grid, name, "fourier") for name in ("basis", "basis_x", "basis_y", "basis_z")]
        if any(str(b).lower() != "fourier" for b in bases):
            raise ValueError("DustyHydroSolver supports only periodic Fourier grids")
        if getattr(grid, "precision", "double") != "double":
            raise ValueError("DustyHydroSolver requires float64 (grid.precision='double')")
        self.nmom = 3 if self._is_mhd else self.ndim
        eq_type = self._equation_types[self.ndim-1]
        if not isinstance(equations, eq_type):
            raise TypeError(f"expected {eq_type.__name__} for this grid")
        self.grid, self.equations = grid, equations
        self.scheme, self.cfl = scheme.lower(), float(cfl)
        self.dust_config, self.sgs_config = dict(dust_config or {}), dict(sgs_config or {})
        self.dust_enabled = bool(self.dust_config.get("enabled", True))
        self.sgs_enabled = bool(self.sgs_config.get("enabled", True))
        name = self.dust_config.get("closure", "gaussian")
        if name not in ("gaussian", "hyqmom"):
            raise ValueError(f"unknown dust closure: {name}")
        self.closure = (dust_moments.GaussianClosure if name == "gaussian" else
                        dust_moments.HyqmomClosure)(self.ndim)
        if self.sgs_config.get("production", "filter_drain") not in ("none", "filter_drain"):
            raise ValueError("sgs.production must be 'none' or 'filter_drain'")
        self.dxs = tuple(float(getattr(grid, a)) for a in ("dx", "dy", "dz")[:self.ndim])
        self.cell_volume = float(np.prod(self.dxs))
        self.Delta = self.cell_volume**(1/self.ndim)
        cls = self._gas_solver_types[self.ndim-1]
        self.gas_solver = cls(grid, equations, scheme=self.scheme, cfl=self.cfl)
        self.positivity = self.gas_solver.positivity
        self.U_gas = None
        self.U_dust = np.zeros((self.closure.nfields, *self.shape), dtype=np.float64)
        self.e_s = np.zeros(self.shape, dtype=np.float64)
        self.t = 0.0
        self.ledger = dict.fromkeys(("production", "sgs_decay", "dust_work", "drag_heat_mean",
                                    "drag_heat_random", "compression_to_sgs", "filter_kinetic_drop",
                                    "projection_heat"), 0.0)
        self.transport_stats = dict(substeps=0, first_order_cells=0, max_doublings=0,
                                    halvings=0, repaired_cells=0, roundoff_projections=0,
                                    sgs_substeps=0, diffusion_substeps=0)

    @property
    def U(self):
        """Gas state, for compatibility with problem visualization hooks."""
        return self.U_gas

    def _validate_state(self):
        nfields = 8 if self._is_mhd else self.ndim+2
        for name, shape in (("U_gas", (nfields, *self.shape)),
                            ("U_dust", (self.closure.nfields, *self.shape)), ("e_s", self.shape)):
            value = getattr(self, name)
            if not isinstance(value, np.ndarray):
                raise TypeError(f"{name} must be a NumPy array, not torch or None")
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            if value.dtype != np.float64:
                raise TypeError(f"{name} must have dtype float64")
        if np.any(self.U_gas[0] <= 0) or np.any(self._internal_energy(self.U_gas) <= 0):
            raise ValueError("gas density and internal energy must be positive")
        if np.any(self.e_s < 0) or (not self.sgs_enabled and np.any(self.e_s != 0)):
            raise ValueError("SGS energy must be non-negative, and zero when SGS is disabled")
        if not self.dust_enabled and np.any(self.U_dust != 0):
            raise ValueError("dust must be zero when dust is disabled")
        if self.dust_enabled and not np.all(self.closure.realizable(self.U_dust)):
            raise ValueError("dust state is not realizable")

    def _integral(self, field):
        return float(np.sum(field))*self.cell_volume

    def _kinetic(self, U):
        return 0.5*np.sum(U[1:1+self.nmom]**2, axis=0)/U[0]

    def _internal_energy(self, U):
        return U[self.energy_index] - self._kinetic(U)

    def _set_pressure(self):
        self.equations.p_sgs = ((2/3)*self.e_s if self.sgs_enabled and
                                self.sgs_config.get("pressure", True) else None)

    def _nu(self):
        stopping = self.dust_config.get("stopping", {})
        model = stopping.get("model", "constant")
        if model == "constant":
            ts = float(stopping.get("t_s", 1.0))
            if ts <= 0 or np.isnan(ts):
                raise ValueError("dust.stopping.t_s must be positive")
            return np.full(self.shape, 1/ts)
        if model == "epstein":
            grain = float(stopping.get("grain_rho_a", 1.0))
            if not np.isfinite(grain) or grain <= 0:
                raise ValueError("dust.stopping.grain_rho_a must be finite and positive")
            return dust_sources.epstein_nu(self.U_gas, self.equations.gamma, grain, self.ndim)
        raise ValueError(f"unknown stopping model: {model}")

    def _local_step(self, h):
        if not (self.dust_enabled or self.sgs_enabled):
            return
        self.U_gas, self.U_dust, self.e_s, ledger = dust_sources.local_coupling_step(
            self.U_gas, self.U_dust, self.e_s, h, closure=self.closure,
            gamma=self.equations.gamma, nu=self._nu() if self.dust_enabled else 0.0,
            Delta=self.Delta, ndim=self.ndim,
            d_model=self.sgs_config.get("d_model", self._default_d_model()) if self.sgs_enabled else "none",
            c_eps=float(self.sgs_config.get("c_eps", 1.0)),
            sgs_decay=self.sgs_enabled and self.sgs_config.get("decay", True),
            backreaction=self.dust_config.get("backreaction", True),
            charge_to_mass=self.dust_config.get("charge_to_mass", 0.0))
        for source, target in (("Q_mean", "drag_heat_mean"), ("Q_rand", "drag_heat_random"),
                               ("L_dust", "dust_work"), ("L_decay", "sgs_decay"),
                               ("Q_projection", "projection_heat")):
            self.ledger[target] += self._integral(ledger[source])

    def _default_d_model(self):
        return "mt25"

    def compute_dt(self):
        self._validate_state()
        self._set_pressure()
        bounds = [self.gas_solver._stability_limit(self.U_gas)]
        if self.dust_enabled:
            bounds.append(dust_moments.max_stable_dt(self.U_dust, self.closure, self.dxs,
                                                     self.dust_config.get("cfl", 0.8)))
        if self.sgs_enabled:
            u = self.U_gas[1:1+self.ndim]/self.U_gas[0]
            rate = sum(float(np.max(np.abs(u[d])))/dx for d, dx in enumerate(self.dxs))
            bounds.append(0.5*self.cfl/rate if rate > 0 else np.inf)
        return min(bounds)

    def _gas_step(self, dt):
        U = self.U_gas
        split = self.grid.apply_spectral_dissipation(U, 0.5*dt)
        suffix = ("", "_2d", "_3d")[self.ndim-1]
        rk = getattr(gas_solver, f"_{self.scheme}_step{suffix}")
        args = (self.grid, self.equations, split, dt, None, None, self.positivity)
        if self.ndim == 3:
            args += (self.gas_solver.rho_floor, self.gas_solver.p_floor)
        un = rk(*args)
        stabilize = getattr(gas_solver, f"_apply_accepted_state_stabilization{suffix}")
        if self.ndim == 3:
            accepted = stabilize(self.grid, un, dt=0.5*dt)
        else:
            accepted = stabilize(self.grid, un, self.equations, dt=0.5*dt)
        drop = ((self._kinetic(U) - self._kinetic(split))
                + (self._kinetic(un) - self._kinetic(accepted)))
        self.U_gas = accepted
        return drop

    def step(self, dt):
        """Advance one accepted step and update state, time and cumulative ledgers."""
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        self._validate_state()
        # Freeze SGS pressure from the start of the full split step.
        self._set_pressure()
        self._local_step(0.5*dt)
        velocity_before = self.U_gas[1:1+self.ndim]/self.U_gas[0]
        drop = self._gas_step(dt)
        self.ledger["filter_kinetic_drop"] += self._integral(drop)
        if self.sgs_enabled:
            if self.sgs_config.get("production", "filter_drain") == "filter_drain":
                production = sgs.drain_production(
                    drop, self.dxs, self.Delta, self._internal_energy(self.U_gas),
                    c_ell=self.sgs_config.get("c_ell", 1.0), f_cap=self.sgs_config.get("f_cap", 0.5))
                self.U_gas[self.energy_index] -= production
                self.e_s += production
                self.ledger["production"] += self._integral(production)
            velocity = 0.5*(velocity_before + self.U_gas[1:1+self.ndim]/self.U_gas[0])
            self.e_s, nsub = sgs.advect_scalar(self.e_s, velocity, dt, self.dxs, cfl=self.cfl)
            self.transport_stats["sgs_substeps"] += nsub
            divu = sum(getattr(self.grid, name)(velocity[d])
                       for d, name in enumerate(("dx1", "dy1", "dz1")[:self.ndim]))
            compressed = self.e_s*sgs.compression_factor(divu, dt)
            change = compressed - self.e_s
            self.e_s = compressed
            self.U_gas[self.energy_index] -= change
            self.ledger["compression_to_sgs"] += self._integral(change)
            diffusion = self.sgs_config.get("diffusion", {})
            if diffusion.get("enabled", False):
                kappa = diffusion.get("kappa", float(diffusion.get("c_kappa", 1.0))*
                                      np.sqrt(2*self.e_s/(3*self.U_gas[0]))*self.Delta)
                self.e_s, nsub = sgs.diffuse_scalar(self.e_s, kappa, dt, self.dxs)
                self.transport_stats["diffusion_substeps"] += nsub
        if self.dust_enabled:
            self.U_dust, stats = dust_moments.transport(
                self.U_dust, self.closure, dt, self.dxs, cfl=self.dust_config.get("cfl", 0.8))
            for key in ("substeps", "first_order_cells", "halvings", "repaired_cells",
                        "roundoff_projections"):
                self.transport_stats[key] = self.transport_stats.get(key, 0) + stats[key]
            # Vacuum/round-off regularisation keeps mass and momentum; its tiny
            # energy change is returned to the gas so the total stays exact.
            self.U_gas[self.energy_index] -= stats["regularization_dE"]
            self.transport_stats["max_doublings"] = max(self.transport_stats["max_doublings"], stats["max_doublings"])
            diffusion = self.dust_config.get("eddy_diffusion", {})
            if self.sgs_enabled and diffusion.get("enabled", False) and np.any(self.e_s > 0):
                coeff = sgs.sgs_coefficients(self.e_s, self.U_gas[0], self._nu(), self.Delta,
                                             self.sgs_config.get("d_model", self._default_d_model()))
                # x/(1+x) evaluated without inf/inf at zero SGS energy.
                fraction = np.divide(coeff["x"], 1 + coeff["x"],
                                     out=np.ones_like(coeff["x"]), where=np.isfinite(coeff["x"]))
                kappa = float(diffusion.get("c_kappa", 1.0))*coeff["sigma"]*self.Delta*fraction
                self.U_dust, nsub = dust_moments.diffuse(self.U_dust, kappa, dt, self.dxs)
                self.transport_stats["diffusion_substeps"] += nsub
        self._local_step(0.5*dt)
        self.t += dt
        self.gas_solver.t, self.gas_solver.U = self.t, self.U_gas
        return self.U_gas

    def diagnostics(self, dt=0.0):
        mean = self._integral(self.closure.mean_kinetic_energy(self.U_dust))
        random = self._integral(self.closure.random_energy(self.U_dust))
        gas, es = self._integral(self.U_gas[self.energy_index]), self._integral(self.e_s)
        mass_dust = self._integral(self.U_dust[0])
        result = dict(time=self.t, dt=dt, E_gas=gas, E_sgs=es, E_dust_mean=mean,
                      E_dust_random=random, E_total=gas+es+mean+random,
                      mass_gas=self._integral(self.U_gas[0]), mass_dust=mass_dust,
                      dust_var_mean=2*random/(3*mass_dust) if mass_dust > 0 else 0.0,
                      e_s_mean=float(np.mean(self.e_s)), positivity_cells=self.positivity.cells,
                      positivity_events=self.positivity.events, **self.ledger, **self.transport_stats)
        momentum = self.U_dust[1:4] if self.closure.name == "gaussian" else self.U_dust[[1, 5, 6]]
        for d, axis in enumerate("xyz"[:self.nmom]):
            result[f"momentum_{axis}"] = self._integral(self.U_gas[1+d] + momentum[d])
        return result

    def run(self, t0, t_end, output_interval, on_output=None, on_step=None, max_steps=None):
        """Return initial and every accepted-step diagnostic; output at exact times.

        ``max_steps`` may stop a run early; its final accepted state is output.
        Ledgers accumulate over the solver lifetime, including resumed runs.
        """
        if not np.isfinite(t0) or not np.isfinite(t_end) or t_end < t0:
            raise ValueError("require finite t_end >= t0")
        if not np.isfinite(output_interval) or output_interval <= 0:
            raise ValueError("output_interval must be finite and positive")
        self._validate_state()
        self.t = float(t0)
        history = {key: [value] for key, value in self.diagnostics().items()}
        if on_output is not None:
            on_output(self.t, self.U_gas)
        last_output = self.t
        output_index, count = 1, 0
        while self.t < t_end and (max_steps is None or count < max_steps):
            next_output = min(t_end, t0 + output_index*output_interval)
            dt = min(self.compute_dt(), next_output - self.t)
            if dt <= 0 or self.t + dt == self.t:
                raise RuntimeError("coupled timestep stalled")
            self.step(dt)
            count += 1
            for key, value in self.diagnostics(dt).items():
                history[key].append(value)
            if on_step is not None:
                on_step(count, dt, self.U_gas)
            if self.t >= next_output:
                if on_output is not None:
                    on_output(self.t, self.U_gas)
                last_output = self.t
                output_index += 1
        if on_output is not None and self.t != last_output:
            on_output(self.t, self.U_gas)
        self.history = history
        return history
