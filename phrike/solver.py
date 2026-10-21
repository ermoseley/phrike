from __future__ import annotations

import inspect
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

from .grid import Grid1D, Grid2D, Grid3D
from .equations import (
    EulerEquations1D,
    EulerEquations2D,
    EulerEquations3D,
    MHDEquations1D,
    MHDEquations2D,
    MHDEquations3D,
)
from .adaptive import AdaptiveTimeStepper, create_adaptive_stepper
from .artificial_viscosity import SpectralArtificialViscosity, ArtificialViscosityConfig


Array = np.ndarray

# Fixed-step schemes and embedded pairs. An unknown name is an error.
# rk23/rk45/rk78 are not silent aliases of RK4.
_FIXED_SCHEMES = frozenset({"rk2", "rk4"})
_EMBEDDED_SCHEMES = frozenset({"rk23", "rk45", "rk78"})
_KNOWN_SCHEMES = _FIXED_SCHEMES | _EMBEDDED_SCHEMES
_UNUSED_FALLBACK_WARNED = False

# Historical positivity floors. 2D is looser than 1D; 3D follows 1D.
# Changing the 2D numbers would move any shock that actually hits the floor.
_RHO_FLOOR_1D = 1e-10
_P_FLOOR_1D = 1e-12
_RHO_FLOOR_2D = 1e-8
_P_FLOOR_2D = 1e-10


class UnboundedTimeStep(RuntimeError):
    """The step is not bounded by a wave speed, a parabolic limit, or a horizon."""


class StalledTimeStep(RuntimeError):
    """The stability limit collapsed far below its initial value."""


# A step bound this far below the initial one, or one that is 1e3 smaller and
# leaves more than 1e6 steps to t_end, is a stalled run, not a slow one.
_STALL_COLLAPSE_FRACTION = 1e-8
_STALL_SLOW_FRACTION = 1e-3
_STALL_MAX_REMAINING_STEPS = 1e6


class PositivityMonitor:
    """Count positivity-floor hits and warn once per run.

    ``events`` is the number of clamp calls that raised at least one value.
    ``cells`` sums the density and pressure values that were below the floor.
    """

    def __init__(self) -> None:
        self.events = 0
        self.cells = 0
        self._warned = False

    def reset_run(self) -> None:
        self.events = 0
        self.cells = 0
        self._warned = False

    def note(self, n_values: int) -> None:
        if n_values <= 0:
            return
        self.events += 1
        self.cells += int(n_values)
        if self._warned:
            return
        self._warned = True
        warnings.warn(
            "Positivity clamp raised density or pressure to the floor "
            f"({int(n_values)} values on the first event this run).",
            RuntimeWarning,
            stacklevel=3,
        )


def _init_monitors(solver, rho_floor: float, p_floor: float) -> None:
    solver.positivity = PositivityMonitor()
    solver.rho_floor = float(rho_floor)
    solver.p_floor = float(p_floor)
    solver._accepted_steps = 0


def _is_torch_array(value) -> bool:
    return bool(_TORCH_AVAILABLE and isinstance(value, torch.Tensor))


def _array_all_finite(U) -> bool:
    if _is_torch_array(U):
        return bool(torch.isfinite(U).all().item())
    return bool(np.isfinite(np.asarray(U)).all())


def _require_finite_state(solver, where: str, U=None) -> None:
    """Raise if the conserved state contains NaN or Inf.

    The check looks at the state itself. It does not trust ``max_wave_speed``,
    which can report 0 when every cell is non-finite.
    """
    state = solver.U if U is None else U
    if state is None or _array_all_finite(state):
        return
    step = int(getattr(solver, "_accepted_steps", 0))
    raise FloatingPointError(
        f"Non-finite solution {where} at step {step}, t={solver.t}"
    )


def _count_below(values, floor: float) -> int:
    if _is_torch_array(values):
        return int(torch.count_nonzero(values < float(floor)).item())
    return int(np.count_nonzero(np.asarray(values) < float(floor)))


def _floor_values(values, floor: float):
    if _is_torch_array(values):
        return torch.clamp(values, min=float(floor))
    return np.maximum(values, floor)


def _clamp_conserved(
    U,
    equations,
    rho_min: float,
    p_min: float,
    monitor: Optional[PositivityMonitor] = None,
    *,
    rebuild_always: bool = False,
):
    """Floor density and pressure.

    ``rebuild_always`` reproduces the historical 1D/2D fixed-step clamp, which
    rebuilds the conservative state at every stage. Adaptive steps and the 3D
    fixed stages leave a positive state untouched, so a smooth update stays
    bit-for-bit the embedded or classical Runge-Kutta step.

    MHD (8 components) is not handled here. Fixed MHD substages use
    ``SpectralSolverMHD._clamp``; embedded MHD stages are not limited.
    """
    primitive = equations.primitive(U)
    ncomp = int(U.shape[0])
    if ncomp == 3:
        rho, velocity, pressure = primitive[0], primitive[1], primitive[2]
        below = _count_below(rho, rho_min) + _count_below(pressure, p_min)
        if monitor is not None:
            monitor.note(below)
        if below == 0 and not rebuild_always:
            return U
        return equations.conservative(
            _floor_values(rho, rho_min), velocity, _floor_values(pressure, p_min)
        )
    if ncomp == 4:
        rho, ux, uy, pressure = primitive
        below = _count_below(rho, rho_min) + _count_below(pressure, p_min)
        if monitor is not None:
            monitor.note(below)
        if below == 0 and not rebuild_always:
            return U
        return equations.conservative(
            _floor_values(rho, rho_min), ux, uy, _floor_values(pressure, p_min)
        )
    if ncomp == 5:
        rho, ux, uy, uz, pressure = primitive
        below = _count_below(rho, rho_min) + _count_below(pressure, p_min)
        if monitor is not None:
            monitor.note(below)
        if below == 0 and not rebuild_always:
            return U
        return equations.conservative(
            _floor_values(rho, rho_min),
            ux,
            uy,
            uz,
            _floor_values(pressure, p_min),
        )
    raise ValueError(
        f"Positivity clamp expected a 1D, 2D, or 3D Euler state, got {ncomp} components"
    )


def _limit_euler_stage(solver, U):
    return _clamp_conserved(
        U,
        solver.equations,
        solver.rho_floor,
        solver.p_floor,
        solver.positivity,
        rebuild_always=False,
    )


def _call_adaptive_step(solver, stepper, rhs, state, dt, scale, limit_state=None):
    """Step, and attach the solver step and time to a non-finite failure.

    ``limit_state`` is forwarded only when ``stepper.step`` accepts it. Tests
    replace the stepper with a duck-typed ``(rhs, state, dt, scale)`` callable.
    """
    try:
        parameters = inspect.signature(stepper.step).parameters
    except (TypeError, ValueError):
        parameters = {}
    accepts_limit = limit_state is not None and (
        "limit_state" in parameters
        or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    )
    try:
        if accepts_limit:
            return stepper.step(rhs, state, dt, scale, limit_state=limit_state)
        return stepper.step(rhs, state, dt, scale)
    except FloatingPointError as exc:
        message = str(exc)
        if "t=" in message:
            raise
        raise FloatingPointError(
            f"{message} At step {int(getattr(solver, '_accepted_steps', 0))}, "
            f"t={solver.t}."
        ) from exc


def _positivity_clamp(U: Array, eqs: EulerEquations1D, rho_min: float = _RHO_FLOOR_1D,
                      p_min: float = _P_FLOOR_1D, monitor: Optional[PositivityMonitor] = None) -> Array:
    """Clamp 1D density and pressure, always rebuilding the conservative state.

    Floors default to rho_min=1e-10 and p_min=1e-12. The 2D fixed stages use
    looser historical floors (1e-8 and 1e-10). 3D uses these 1D floors.
    Rebuilding even when the floor does not bind matches the previous 1D
    fixed-step update.
    """
    return _clamp_conserved(
        U, eqs, rho_min, p_min, monitor, rebuild_always=True
    )


def _parse_scheme(value, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a scheme name, got {value!r}")
    name = value.strip().lower()
    if name not in _KNOWN_SCHEMES:
        raise ValueError(
            f"Unknown {label} {value!r}. "
            "Supported schemes: rk2, rk4, rk23, rk45, rk78."
        )
    return name


def _warn_unused_fallback(fallback_name: str) -> None:
    """``fallback_scheme`` does not run while an embedded method is active."""
    global _UNUSED_FALLBACK_WARNED
    if _UNUSED_FALLBACK_WARNED:
        return
    _UNUSED_FALLBACK_WARNED = True
    warnings.warn(
        "fallback_scheme is unused while adaptive.scheme is an embedded method "
        f"(rk23, rk45, or rk78); ignoring fallback_scheme={fallback_name!r}. "
        "It is applied only when adaptive stepping is requested with rk2 or rk4.",
        RuntimeWarning,
        stacklevel=3,
    )


def _configure_time_integration(scheme, adaptive_config):
    """Resolve the fixed or embedded scheme shared by every solver class.

    Policy:
    * ``rk2`` and ``rk4`` are the fixed schemes. ``rk23``, ``rk45``, and
      ``rk78`` are embedded methods and require adaptive stepping.
    * An unknown name raises ``ValueError`` instead of running RK4.
    * When adaptive stepping is on and both ``scheme`` and
      ``adaptive.scheme`` name embedded methods, they must be the same
      method. The adaptive block wins only when ``scheme`` is a fixed
      method (the historical ``scheme: rk4`` plus ``adaptive.scheme: rk45``
      combination). If ``adaptive.scheme`` is omitted, an embedded
      ``scheme`` is used; otherwise the default embedded method is ``rk45``.
    * ``fallback_scheme`` must be ``rk2`` or ``rk4``. It is the fixed method
      used when adaptive stepping is requested with a non-embedded scheme.
      It is not a recovery path after rejected steps, and a process-wide
      warning says so when an embedded method is selected anyway.
    """
    scheme_name = _parse_scheme(scheme if scheme is not None else "rk4", "integration scheme")
    enabled = bool(adaptive_config and adaptive_config.get("enabled", False))
    if not enabled:
        if scheme_name not in _FIXED_SCHEMES:
            raise ValueError(
                f"Integration scheme {scheme_name!r} is an embedded method. "
                "Enable adaptive time stepping, or choose rk2 or rk4."
            )
        return scheme_name, False, None

    if adaptive_config.get("scheme") is None:
        adaptive_method = scheme_name if scheme_name in _EMBEDDED_SCHEMES else "rk45"
    else:
        adaptive_method = _parse_scheme(adaptive_config.get("scheme"), "adaptive scheme")

    if (
        scheme_name in _EMBEDDED_SCHEMES
        and adaptive_method in _EMBEDDED_SCHEMES
        and scheme_name != adaptive_method
    ):
        raise ValueError(
            f"integration scheme {scheme_name!r} and adaptive scheme "
            f"{adaptive_method!r} name different embedded methods. "
            "Set them to the same method."
        )

    fallback_raw = adaptive_config.get("fallback_scheme", None)
    if fallback_raw is None:
        fallback_name = "rk4"
    else:
        fallback_name = _parse_scheme(fallback_raw, "fallback_scheme")
        if fallback_name not in _FIXED_SCHEMES:
            raise ValueError(
                f"fallback_scheme must be rk2 or rk4, got {fallback_raw!r}"
            )

    if adaptive_method in _EMBEDDED_SCHEMES:
        if fallback_raw is not None:
            _warn_unused_fallback(fallback_name)
        stepper = create_adaptive_stepper(
            adaptive_method,
            rtol=float(adaptive_config.get("rtol", 1e-6)),
            atol=float(adaptive_config.get("atol", 1e-8)),
            safety_factor=float(adaptive_config.get("safety_factor", 0.9)),
            min_dt_factor=float(adaptive_config.get("min_dt_factor", 0.1)),
            max_dt_factor=float(adaptive_config.get("max_dt_factor", 5.0)),
            max_rejections=adaptive_config.get("max_rejections", 10),
            dt_min=adaptive_config.get("dt_min", None),
        )
        return adaptive_method, True, stepper

    warnings.warn(
        f"Adaptive scheme {adaptive_method!r} is not an embedded Runge-Kutta "
        f"method; using fixed-step {fallback_name!r} from fallback_scheme.",
        RuntimeWarning,
        stacklevel=3,
    )
    return fallback_name, False, None


def _reject_unsupported_artificial_viscosity(config, solver_name: str) -> None:
    if config and config.get("enabled", False):
        raise NotImplementedError(
            f"Artificial viscosity is enabled but {solver_name} does not apply it. "
            "Disable artificial_viscosity.enabled."
        )


def _install_adaptive_floor(solver, t0: float, t_end: float) -> None:
    stepper = getattr(solver, "adaptive_stepper", None)
    controller = getattr(stepper, "controller", None)
    if controller is None or getattr(controller, "dt_min", None) is not None:
        return
    span = abs(float(t_end) - float(t0))
    controller.dt_min = 1e-12 * span if span > 0.0 else 1e-14


def _begin_run(solver, t0: float, t_end: float) -> None:
    solver.t = float(t0)
    solver._accepted_steps = 0
    solver._dt_reference = None
    monitor = getattr(solver, "positivity", None)
    if monitor is not None:
        monitor.reset_run()
    _install_adaptive_floor(solver, t0, t_end)
    _require_finite_state(solver, "before time integration")


def _checked_wave_speed(solver, U) -> float:
    _require_finite_state(solver, "while computing the time step", U)
    speed = solver.equations.max_wave_speed(U)
    try:
        speed = float(speed)
    except (TypeError, ValueError) as exc:
        raise FloatingPointError(
            f"Maximum wave speed is not a real number ({speed!r}) at "
            f"step {solver._accepted_steps}, t={solver.t}"
        ) from exc
    if not np.isfinite(speed):
        raise FloatingPointError(
            f"Non-finite maximum wave speed ({speed}) at "
            f"step {solver._accepted_steps}, t={solver.t}"
        )
    if speed < 0.0:
        raise FloatingPointError(
            f"Negative maximum wave speed ({speed}) at "
            f"step {solver._accepted_steps}, t={solver.t}"
        )
    return speed


def _min_spacing(grid) -> float:
    """Smallest positive node spacing on the axes the grid actually has."""
    values = []
    for fallback, primary in (("dx", "dx_min"), ("dy", "dy_min"), ("dz", "dz_min")):
        if not hasattr(grid, fallback):
            continue
        value = getattr(grid, primary, None)
        if value is None:
            value = getattr(grid, fallback)
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"Grid {fallback} spacing is not positive ({value}).")
        values.append(value)
    if not values:
        raise RuntimeError("Cannot form a CFL length: the grid has no dx, dy, or dz.")
    return min(values)


def _hyperbolic_dt(cfl: float, length: float, speed: float) -> float:
    if speed == 0.0:
        return float("inf")
    return float(cfl) * float(length) / float(speed)


def _smallest_finite(bounds) -> float:
    """Minimum finite positive bound, or inf when every bound is absent."""
    finite = []
    for bound in bounds:
        if bound is None:
            continue
        bound = float(bound)
        if not np.isfinite(bound):
            if bound == float("inf"):
                continue
            raise FloatingPointError(f"Non-finite time-step bound ({bound})")
        if bound <= 0.0:
            raise RuntimeError(f"Non-positive time-step bound ({bound})")
        finite.append(bound)
    if not finite:
        return float("inf")
    return min(finite)


def _event_horizon(t, t_end, next_output, next_checkpoint) -> float:
    horizon = float(t_end) - float(t)
    for event in (next_output, next_checkpoint):
        if event is None:
            continue
        event = float(event)
        if np.isfinite(event) and event > float(t):
            horizon = min(horizon, event - float(t))
    return horizon


def _check_stalled_step(solver, free_dt: float, t_end) -> None:
    """Raise when the step bound has collapsed against the first one.

    ``free_dt`` is the proposal or stability limit before any output-time cap,
    which can legitimately be tiny. The first finite value is the reference.
    """
    reference = getattr(solver, "_dt_reference", None)
    if reference is None:
        if np.isfinite(free_dt) and free_dt > 0.0:
            solver._dt_reference = float(free_dt)
        return
    if not np.isfinite(free_dt):
        return
    ratio = free_dt / reference
    remaining = max(float(t_end) - float(solver.t), 0.0)
    if ratio < _STALL_COLLAPSE_FRACTION or (
        ratio < _STALL_SLOW_FRACTION
        and remaining > _STALL_MAX_REMAINING_STEPS * free_dt
    ):
        monitor = getattr(solver, "positivity", None)
        floored = (
            f" The positivity floor was hit for {monitor.cells} values."
            if monitor is not None and monitor.cells
            else ""
        )
        raise StalledTimeStep(
            f"Time step collapsed to {free_dt:.3e} ({ratio:.1e} of the initial "
            f"{reference:.3e}) at step {solver._accepted_steps}, t={solver.t}; "
            f"the run would not reach t_end={float(t_end):g}.{floored} "
            "The solution has likely lost resolution, for example a vacuum-like "
            "state with an extreme wave speed. Increase resolution or "
            "dissipation."
        )


def _cap_proposal(solver, proposal, t_end, next_output, next_checkpoint) -> float:
    """Cap a proposed step by the stability limit and the next output time.

    A genuinely zero wave speed contributes no hyperbolic bound. The step is
    then the viscous/diffusive limit, or the time remaining until the next
    output, checkpoint, or ``t_end``. If none of those is finite and positive,
    raise rather than inventing a ``1e-6`` step.
    """
    limit = solver._stability_limit(solver.U)
    horizon = _event_horizon(solver.t, t_end, next_output, next_checkpoint)
    finite = [
        float(value)
        for value in (proposal, limit, horizon)
        if np.isfinite(float(value))
    ]
    if not finite:
        raise UnboundedTimeStep(
            "No finite time step at "
            f"step {solver._accepted_steps}, t={solver.t}: the maximum wave "
            "speed is zero and no viscous, diffusive, output, checkpoint, or "
            "end-time limit applies."
        )
    dt = min(finite)
    if dt <= 0.0:
        raise UnboundedTimeStep(
            f"No positive time step at step {solver._accepted_steps}, t={solver.t}."
        )
    free = [float(v) for v in (proposal, limit) if np.isfinite(float(v))]
    _check_stalled_step(solver, min(free) if free else float("inf"), t_end)
    return dt


def _public_dt(solver, U) -> float:
    dt = solver._stability_limit(U)
    if not np.isfinite(dt):
        raise UnboundedTimeStep(
            "No finite time-step bound at "
            f"step {solver._accepted_steps}, t={solver.t}: the maximum wave "
            "speed is zero and no viscous or diffusive stability limit applies."
        )
    return dt


def _emit_final_output(history, t, record, on_output, U) -> None:
    """Record the end state once. Skip when the last step already landed on it."""
    times = history.get("time") or []
    if times and abs(float(times[-1]) - float(t)) <= 1e-12:
        return
    record()
    if on_output is not None:
        on_output(t, U)


def _note_accepted(solver) -> None:
    solver._accepted_steps = int(getattr(solver, "_accepted_steps", 0)) + 1
    _require_finite_state(solver, "after accepted step")


def nodal_quadrature_weights(grid, U):
    """Quadrature weights for a 1D non-Fourier grid, on ``U``'s backend.

    Returns ``None`` for Fourier grids, for grids without weights, and for
    states that are not a 1D ``(components, N)`` array. Callers then keep the
    plain grid sums of ``equations.conserved_quantities``.
    """
    if getattr(grid, "_basis_name", "fourier") == "fourier":
        return None
    weights = getattr(grid, "wx", None)
    if weights is None or getattr(U, "ndim", 0) != 2:
        return None
    if _is_torch_array(U):
        return torch.as_tensor(weights, dtype=U.dtype, device=U.device)
    return np.asarray(weights, dtype=U.dtype)


def conserved_diagnostics(grid, equations, U) -> Dict[str, float]:
    """Mass, momentum and energy diagnostics of a state.

    Fourier grids return ``equations.conserved_quantities(U)``, the plain grid
    sums, unchanged. Chebyshev and Legendre grids integrate with the grid's
    quadrature weights: node sums are not integrals there, and they drift by
    the node-clustering imbalance even when the weighted integral is
    conserved. The keys match those of ``equations.conserved_quantities``.
    """
    weights = nodal_quadrature_weights(grid, U)
    ncomp = int(U.shape[0])
    if weights is None or ncomp not in (3, 8):
        return equations.conserved_quantities(U)

    def integral(q) -> float:
        if _is_torch_array(q):
            return float(torch.sum(q * weights).item())
        return float(np.sum(q * weights))

    primitive = equations.primitive(U)
    if ncomp == 3:
        rho, u, p, _ = primitive
        return {
            "mass": integral(rho),
            "momentum": integral(rho * u),
            "energy": integral(p / (equations.gamma - 1.0) + 0.5 * rho * u * u),
        }
    rho, ux, uy, uz, p, Bx, By, Bz = primitive
    b2 = Bx * Bx + By * By + Bz * Bz
    energy = (
        p / (equations.gamma - 1.0)
        + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
        + 0.5 * b2
    )
    return {
        "mass": integral(rho),
        "momentum_x": integral(rho * ux),
        "momentum_y": integral(rho * uy),
        "momentum_z": integral(rho * uz),
        "energy": integral(energy),
        "magnetic_energy": integral(0.5 * b2),
        "cross_helicity": integral(ux * Bx + uy * By + uz * Bz),
    }


def _diffusion(grid, f: Array) -> Array:
    """Diffusion operator: zero-flux endpoint closure on 1D polynomial grids."""
    diffusion = getattr(grid, "diffusion", None)
    return diffusion(f) if diffusion is not None else grid.laplacian(f)


def _compute_rhs(grid: Grid1D, eqs: EulerEquations1D, U: Array, 
                artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                gravity_config: Optional[Dict] = None) -> Array:
    # Pseudo-spectral: compute flux in physical space, then differentiate spectrally
    # Enforce boundary conditions for non-periodic bases before computing flux
    if hasattr(grid, "apply_boundary_conditions"):
        U = grid.apply_boundary_conditions(U, eqs)
    if hasattr(grid, "apply_boundary_flux"):
        dFdx, boundary_flux = grid.flux_derivative(U, eqs, return_boundary_flux=True)
        rhs = grid.apply_boundary_flux(-dFdx, U, eqs, boundary_flux)
    else:
        rhs = -grid.flux_derivative(U, eqs)
    
    # Add artificial viscosity if enabled
    if artificial_viscosity is not None:
        viscosity_terms = artificial_viscosity.compute_viscosity_terms(U, grid, eqs)
        for i, term in enumerate(viscosity_terms):
            if i < len(rhs):
                rhs[i] += term
    
    # Add gravity source terms if enabled
    if gravity_config and gravity_config.get("enabled", False):
        rho, u, p, _ = eqs.primitive(U)
        gx = gravity_config.get("gx", 0.0)
        
        # Momentum equation: d(ρu)/dt += ρgx
        rhs[1] += rho * gx
        
        # Energy equation: dE/dt += ρgx * u (work done by gravity)
        rhs[2] += rho * gx * u
    
    return rhs


def _rk2_step(grid: Grid1D, eqs: EulerEquations1D, U: Array, dt: float, 
              artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
              gravity_config: Optional[Dict] = None,
              monitor: Optional[PositivityMonitor] = None) -> Array:
    k1 = _compute_rhs(grid, eqs, U, artificial_viscosity, gravity_config)
    U1 = U + dt * 0.5 * k1
    U1 = _positivity_clamp(U1, eqs, monitor=monitor)
    k2 = _compute_rhs(grid, eqs, U1, artificial_viscosity, gravity_config)
    Unew = U + dt * k2
    return _positivity_clamp(Unew, eqs, monitor=monitor)


def _rk4_step(grid: Grid1D, eqs: EulerEquations1D, U: Array, dt: float,
              artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
              gravity_config: Optional[Dict] = None,
              monitor: Optional[PositivityMonitor] = None) -> Array:
    k1 = _compute_rhs(grid, eqs, U, artificial_viscosity, gravity_config)
    U2 = _positivity_clamp(U + 0.5 * dt * k1, eqs, monitor=monitor)
    k2 = _compute_rhs(grid, eqs, U2, artificial_viscosity, gravity_config)
    U3 = _positivity_clamp(U + 0.5 * dt * k2, eqs, monitor=monitor)
    k3 = _compute_rhs(grid, eqs, U3, artificial_viscosity, gravity_config)
    U4 = _positivity_clamp(U + dt * k3, eqs, monitor=monitor)
    k4 = _compute_rhs(grid, eqs, U4, artificial_viscosity, gravity_config)
    Unew = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _positivity_clamp(Unew, eqs, monitor=monitor)


def _apply_accepted_state_stabilization(
    grid: Grid1D,
    U: Array,
    equations: Optional[EulerEquations1D] = None,
    dt: float = 0.0,
) -> Array:
    # Complete the split Fourier step, or apply the independent modal filter.
    U_filtered = grid.apply_accepted_state_stabilization(U, dt)
    # Re-apply boundary conditions after filtering to prevent endpoint drift
    if equations is not None and hasattr(grid, "apply_boundary_conditions"):
        U_filtered = grid.apply_boundary_conditions(U_filtered, equations)
    return U_filtered


def _artificial_viscosity_dt(grid, viscosity) -> float:
    if viscosity is None:
        return float("inf")
    config = viscosity.config
    nu = (config.nu_constant or config.nu_max) if config.mode == "constant" else config.nu_max
    weight = max([1.0, *config.variable_weights.values()])
    if nu <= 0.0:
        return float("inf")
    # 1D fixed-endpoint diffusion matrix. Multi-D grids have no `_basis`.
    basis = getattr(grid, "_basis_name", None)
    if (config.mode == "constant" and basis not in (None, "fourier")
            and getattr(grid, "_bc_global", None) == "fixed"
            and getattr(grid, "N", 0) > 2
            and hasattr(grid, "_basis")):
        if not hasattr(grid, "_fixed_diffusion_bound"):
            laplacian = grid._basis.d2x(np.eye(grid.N)).T
            grid._fixed_diffusion_bound = float(np.max(np.sum(np.abs(laplacian[1:-1, 1:-1]), axis=1)))
        # Bound the fixed-endpoint diffusion matrix, with a unit RK stability margin.
        return 1.0 / (nu * weight * grid._fixed_diffusion_bound)
    dx = _min_spacing(grid)
    return 0.2 * dx ** 2 / (nu * weight)


def _spacing_wavenumber_bound(grid) -> float:
    """Conservative explicit-diffusion bound ``sum_i 4 / dx_i**2``."""
    terms = []
    for fallback, primary in (("dx", "dx_min"), ("dy", "dy_min"), ("dz", "dz_min")):
        if not hasattr(grid, fallback):
            continue
        value = getattr(grid, primary, None)
        if value is None:
            value = getattr(grid, fallback)
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise RuntimeError(f"Grid {fallback} spacing is not positive ({value}).")
        terms.append(4.0 / (value * value))
    if not terms:
        raise RuntimeError(
            "Cannot bound the diffusive CFL: grid wavenumbers and spacings are unavailable."
        )
    return float(sum(terms))


class SpectralSolver1D:
    grid: Grid1D
    equations: EulerEquations1D
    scheme: str = "rk4"  # "rk2", "rk4", "rk23", "rk45", "rk78"
    cfl: float = 0.4

    # Runtime state
    t: float = 0.0
    U: Optional[Array] = None
    
    # Adaptive time-stepping
    adaptive_stepper: Optional[AdaptiveTimeStepper] = None
    adaptive_enabled: bool = False
    
    # Artificial viscosity
    artificial_viscosity: Optional[SpectralArtificialViscosity] = None
    # Gravity configuration
    gravity_config: Optional[Dict] = None
    def __init__(self, grid: Grid1D, equations: EulerEquations1D, U0: Optional[Array] = None, 
                 scheme: str = "rk4", cfl: float = 0.4, adaptive_config: Optional[Dict] = None,
                 artificial_viscosity_config: Optional[Dict] = None, gravity_config: Optional[Dict] = None):
        """Initialize the 1D spectral solver.
        
        Args:
            grid: 1D grid
            equations: 1D Euler equations
            U0: Initial solution (optional)
            scheme: Time integration scheme ("rk2", "rk4", "rk23", "rk45", "rk78")
            cfl: CFL number
            adaptive_config: Configuration for adaptive time-stepping
            artificial_viscosity_config: Configuration for artificial viscosity
            gravity_config: Configuration for gravity
        """
        self.grid = grid
        self.equations = equations
        self.cfl = cfl
        self.t = 0.0
        self.U = U0
        self.scheme, self.adaptive_enabled, self.adaptive_stepper = (
            _configure_time_integration(scheme, adaptive_config)
        )
        _init_monitors(self, _RHO_FLOOR_1D, _P_FLOOR_1D)
        
        # Setup artificial viscosity if configured
        self.artificial_viscosity = None
        if artificial_viscosity_config and artificial_viscosity_config.get("enabled", False):
            from .artificial_viscosity import create_artificial_viscosity
            self.artificial_viscosity = create_artificial_viscosity(artificial_viscosity_config)

        # Setup gravity if configured
        self.gravity_config = gravity_config


    def _stability_limit(self, U: Array) -> float:
        speed = _checked_wave_speed(self, U)
        # CFL for spectral (heuristic): dt <= cfl * dx / max_speed.
        # A zero wave speed is unbounded here. Artificial viscosity can still
        # supply a finite limit. A uniform state at rest with p > 0 is not a
        # zero-speed state: the sound speed remains.
        return _smallest_finite((
            _hyperbolic_dt(self.cfl, _min_spacing(self.grid), speed),
            _artificial_viscosity_dt(self.grid, self.artificial_viscosity),
        ))

    def compute_dt(self, U: Array) -> float:
        return _public_dt(self, U)

    def step(self, U: Array, dt: float) -> Array:
        """Perform one time step using the configured scheme.
        
        Args:
            U: Current solution vector
            dt: Time step
            
        Returns:
            New solution vector
        """
        if self.adaptive_enabled and self.adaptive_stepper is not None:
            # Use adaptive time-stepping
            return self._adaptive_step(U, dt)
        else:
            # Use fixed time-stepping
            return self._fixed_step(U, dt)
    
    def _fixed_step(self, U: Array, dt: float) -> Array:
        """Perform one fixed time step."""
        U_split = self.grid.apply_spectral_dissipation(U, 0.5 * dt)
        if self.scheme.lower() == "rk2":
            Un = _rk2_step(self.grid, self.equations, U_split, dt, self.artificial_viscosity, self.gravity_config, self.positivity)
        else:
            Un = _rk4_step(self.grid, self.equations, U_split, dt, self.artificial_viscosity, self.gravity_config, self.positivity)
        return _apply_accepted_state_stabilization(
            self.grid, Un, self.equations, dt=0.5 * dt
        )
    
    def _adaptive_step(self, U: Array, dt: float) -> Array:
        """Perform one adaptive time step."""
        def rhs_func(U_current: Array) -> Array:
            """RHS function for adaptive stepper."""
            return _compute_rhs(self.grid, self.equations, U_current, self.artificial_viscosity, self.gravity_config)
        
        U_split = self.grid.apply_spectral_dissipation(U, 0.5 * dt)

        # Compute solution scale for relative error
        if _TORCH_AVAILABLE and isinstance(U_split, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U_split)).item())
        else:
            solution_scale = float(np.max(np.abs(U_split)))
        
        # Perform adaptive step. Stages after the first are positivity limited.
        result = _call_adaptive_step(
            self, self.adaptive_stepper, rhs_func, U_split, dt, solution_scale,
            limit_state=lambda state: _limit_euler_stage(self, state),
        )
        
        # Complete the split dissipation and projection for an accepted solution.
        if result.accepted:
            Un = _apply_accepted_state_stabilization(
                self.grid, result.U_new, self.equations, dt=0.5 * dt
            )
        else:
            # Return original solution if step was rejected
            Un = U
            
        return Un
    
    def _adaptive_run_step(self, dt: float, step_count: int, 
                          on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in adaptive run loop."""
        def rhs_func(U_current: Array) -> Array:
            return _compute_rhs(self.grid, self.equations, U_current, self.artificial_viscosity, self.gravity_config)
        
        U_split = self.grid.apply_spectral_dissipation(self.U, 0.5 * dt)

        # Compute solution scale for relative error
        if _TORCH_AVAILABLE and isinstance(U_split, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U_split)).item())
        else:
            solution_scale = float(np.max(np.abs(U_split)))
        
        # Perform adaptive step. Stages after the first are positivity limited.
        result = _call_adaptive_step(
            self, self.adaptive_stepper, rhs_func, U_split, dt, solution_scale,
            limit_state=lambda state: _limit_euler_stage(self, state),
        )
        
        if result.accepted:
            # Step accepted - update solution and time
            self.U = _apply_accepted_state_stabilization(
                self.grid, result.U_new, self.equations, dt=0.5 * dt
            )
            self.t += dt
            _note_accepted(self)
            
            # Call monitoring callback
            if on_step is not None:
                on_step(step_count, dt, self.U)
            
            # Return recommended next step size
            return result.dt_next
        else:
            # Step rejected - return recommended smaller step size
            return result.dt_next
    
    def _fixed_run_step(self, dt: float, step_count: int,
                       on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in fixed run loop."""
        self.U = self.step(self.U, dt)
        self.t += dt
        _note_accepted(self)
        
        # Call monitoring callback
        if on_step is not None:
            on_step(step_count, dt, self.U)
        
        return self._stability_limit(self.U)

    def run(
        self,
        U0: Array,
        t0: float,
        t_end: float,
        output_interval: float = 0.05,
        checkpoint_interval: float = 0.0,
        outdir: Optional[str] = None,
        on_output: Optional[Callable[[float, Array], None]] = None,
        on_step: Optional[Callable[[int, float, Array], None]] = None,
        tracers: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        # Start Fourier runs in the same retained subspace enforced after steps.
        self.U = self.grid.project_dealiased(self.U)
        self.U = self.grid.initialize_boundary_conditions(self.U, self.equations)
        _begin_run(self, t0, t_end)
        next_output = self.t + output_interval
        next_checkpoint = (
            self.t + checkpoint_interval
            if checkpoint_interval and checkpoint_interval > 0
            else np.inf
        )

        history: Dict[str, List[float]] = {
            "time": [],
            "mass": [],
            "momentum": [],
            "energy": [],
        }
        step_count = 0

        def record() -> None:
            cons = conserved_diagnostics(self.grid, self.equations, self.U)  # type: ignore[arg-type]
            history["time"].append(self.t)
            history["mass"].append(cons["mass"])  # type: ignore[index]
            history["momentum"].append(cons["momentum"])  # type: ignore[index]
            history["energy"].append(cons["energy"])  # type: ignore[index]

        record()
        if on_output is not None:
            on_output(self.t, self.U)

        dt = float("inf")
        
        while self.t < t_end - 1e-12:
            dt = _cap_proposal(self, dt, t_end, next_output, next_checkpoint)
            dt_used = dt
            state_before = self.U
            time_before = self.t
            
            if self.adaptive_enabled and self.adaptive_stepper is not None:
                # Adaptive time-stepping
                dt = self._adaptive_run_step(dt, step_count, on_step)
            else:
                # Fixed time-stepping (original behavior)
                dt = self._fixed_run_step(dt, step_count, on_step)
            
            if tracers is not None and self.t > time_before:
                tracers.step(
                    self.grid,
                    self.equations,
                    state_before,
                    dt_used,
                    U_new=self.U,
                )
            step_count += 1

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(
                    outdir,
                    self.t,
                    U=self.U,
                    grid=self.grid,
                    equations=self.equations,
                    tracers=tracers,
                )
                next_checkpoint += checkpoint_interval

        _emit_final_output(history, self.t, record, on_output, self.U)
        return history


# 2D solver


def _compute_rhs_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array, 
                   artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                   gravity_config: Optional[Dict] = None) -> Array:
    # Mirror 1D: pre-apply BCs via grid if available (Fourier or hybrid).
    if hasattr(grid, "apply_boundary_conditions"):
        U = grid.apply_boundary_conditions(U, eqs)
    # Fluxes in x and y
    Fx, Fy = eqs.flux(U)
    dFdx = grid.dx1(Fx)
    dFdy = grid.dy1(Fy)
    # dU/dt = -(dFx/dx + dFy/dy)
    rhs = -(dFdx + dFdy)

    if getattr(grid, "_mixed_open_y", False):
        rhs = grid.apply_split_advection(rhs, U)
        rhs = grid.apply_boundary_flux(rhs, U, eqs, Fy)
        rhs = grid.project_dealiased(rhs)
        if grid.sbp_dissipation_strength > 0.0:
            rhs += grid.sbp_dissipation(U, eqs.max_wave_speed(U))
    
    # Add artificial viscosity if enabled
    if artificial_viscosity is not None:
        viscosity_terms = artificial_viscosity.compute_viscosity_terms(U, grid, eqs)
        for i, term in enumerate(viscosity_terms):
            if i < len(rhs):
                rhs[i] += term
    
    # Add gravity source terms if enabled
    if gravity_config and gravity_config.get("enabled", False):
        rho, ux, uy, p = eqs.primitive(U)
        gx = gravity_config.get("gx", 0.0)
        gy = gravity_config.get("gy", 0.0)
        
        # Momentum equations: d(ρux)/dt += ρgx, d(ρuy)/dt += ρgy
        rhs[1] += rho * gx
        rhs[2] += rho * gy
        
        # Energy equation: dE/dt += ρ(gx*ux + gy*uy) (work done by gravity)
        rhs[3] += rho * (gx * ux + gy * uy)
    
    return rhs


def _positivity_clamp_2d(U: Array, eqs: EulerEquations2D, rho_min: float = _RHO_FLOOR_2D,
                         p_min: float = _P_FLOOR_2D,
                         monitor: Optional[PositivityMonitor] = None) -> Array:
    """Clamp 2D density and pressure, always rebuilding the conservative state.

    Floors stay at the historical 2D values rho_min=1e-8 and p_min=1e-10.
    """
    return _clamp_conserved(
        U, eqs, rho_min, p_min, monitor, rebuild_always=True
    )



def _rk2_step_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array, dt: float,
                 artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                 gravity_config: Optional[Dict] = None,
                 monitor: Optional[PositivityMonitor] = None) -> Array:
    k1 = _compute_rhs_2d(grid, eqs, U, artificial_viscosity, gravity_config)
    U1 = _positivity_clamp_2d(U + dt * 0.5 * k1, eqs, monitor=monitor)
    k2 = _compute_rhs_2d(grid, eqs, U1, artificial_viscosity, gravity_config)
    Unew = _positivity_clamp_2d(U + dt * k2, eqs, monitor=monitor)
    return Unew


def _rk4_step_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array, dt: float,
                 artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                 gravity_config: Optional[Dict] = None,
                 monitor: Optional[PositivityMonitor] = None) -> Array:
    k1 = _compute_rhs_2d(grid, eqs, U, artificial_viscosity, gravity_config)
    U2 = _positivity_clamp_2d(U + 0.5 * dt * k1, eqs, monitor=monitor)
    k2 = _compute_rhs_2d(grid, eqs, U2, artificial_viscosity, gravity_config)
    U3 = _positivity_clamp_2d(U + 0.5 * dt * k2, eqs, monitor=monitor)
    k3 = _compute_rhs_2d(grid, eqs, U3, artificial_viscosity, gravity_config)
    U4 = _positivity_clamp_2d(U + dt * k3, eqs, monitor=monitor)
    k4 = _compute_rhs_2d(grid, eqs, U4, artificial_viscosity, gravity_config)
    Unew = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _positivity_clamp_2d(Unew, eqs, monitor=monitor)


def _apply_accepted_state_stabilization_2d(
    grid: Grid2D,
    U: Array,
    equations: Optional[EulerEquations2D] = None,
    dt: float = 0.0,
) -> Array:
    U_filtered = grid.apply_accepted_state_stabilization(U, dt)
    if equations is not None and hasattr(grid, "apply_boundary_conditions"):
        return grid.apply_boundary_conditions(U_filtered, equations)
    return U_filtered


@dataclass
class SpectralSolver2D:
    grid: Grid2D
    equations: EulerEquations2D
    scheme: str = "rk4"
    cfl: float = 0.3

    t: float = 0.0
    U: Optional[Array] = None
    
    # Adaptive time-stepping
    adaptive_stepper: Optional[AdaptiveTimeStepper] = None
    adaptive_enabled: bool = False
    
    # Artificial viscosity
    artificial_viscosity: Optional[SpectralArtificialViscosity] = None
    # Gravity configuration
    gravity_config: Optional[Dict] = None

    def __init__(self, grid: Grid2D, equations: EulerEquations2D, U0: Optional[Array] = None, 
                 scheme: str = "rk4", cfl: float = 0.3, adaptive_config: Optional[Dict] = None,
                 artificial_viscosity_config: Optional[Dict] = None, gravity_config: Optional[Dict] = None):
        """Initialize the 2D spectral solver.
        
        Args:
            grid: 2D grid
            equations: 2D Euler equations
            U0: Initial solution (optional)
            scheme: Time integration scheme ("rk2", "rk4", "rk23", "rk45", "rk78")
            cfl: CFL number
            adaptive_config: Configuration for adaptive time-stepping
            gravity_config: Configuration for gravity
        """
        self.grid = grid
        self.equations = equations
        self.cfl = cfl
        self.t = 0.0
        self.U = U0
        self.scheme, self.adaptive_enabled, self.adaptive_stepper = (
            _configure_time_integration(scheme, adaptive_config)
        )
        _init_monitors(self, _RHO_FLOOR_2D, _P_FLOOR_2D)

        # Setup artificial viscosity if configured
        self.artificial_viscosity = None
        if artificial_viscosity_config and artificial_viscosity_config.get("enabled", False):
            from .artificial_viscosity import create_artificial_viscosity
            self.artificial_viscosity = create_artificial_viscosity(artificial_viscosity_config)
        
        # Setup gravity if configured
        self.gravity_config = gravity_config

    def _stability_limit(self, U: Array) -> float:
        speed = _checked_wave_speed(self, U)
        bounds = [
            _hyperbolic_dt(self.cfl, _min_spacing(self.grid), speed),
            _artificial_viscosity_dt(self.grid, self.artificial_viscosity),
        ]
        if getattr(self.grid, "_mixed_open_y", False) and speed > 0.0:
            rate = self.grid.sbp_dissipation_rate_bound(speed)
            if rate > 0.0:
                bounds.append(1.0 / rate)
        return _smallest_finite(bounds)

    def compute_dt(self, U: Array) -> float:
        return _public_dt(self, U)

    def step(self, U: Array, dt: float) -> Array:
        """Perform one time step using the configured scheme.
        
        Args:
            U: Current solution vector
            dt: Time step
            
        Returns:
            New solution vector
        """
        if self.adaptive_enabled and self.adaptive_stepper is not None:
            # Use adaptive time-stepping
            return self._adaptive_step(U, dt)
        else:
            # Use fixed time-stepping
            return self._fixed_step(U, dt)
    
    def _fixed_step(self, U: Array, dt: float) -> Array:
        """Perform one fixed time step."""
        U_split = self.grid.apply_spectral_dissipation(U, 0.5 * dt)
        if self.scheme.lower() == "rk2":
            Un = _rk2_step_2d(self.grid, self.equations, U_split, dt, self.artificial_viscosity, self.gravity_config, self.positivity)
        else:
            Un = _rk4_step_2d(self.grid, self.equations, U_split, dt, self.artificial_viscosity, self.gravity_config, self.positivity)
        return _apply_accepted_state_stabilization_2d(
            self.grid, Un, self.equations, dt=0.5 * dt
        )
    
    def _adaptive_step(self, U: Array, dt: float) -> Array:
        """Perform one adaptive time step."""
        def rhs_func(U_current: Array) -> Array:
            """RHS function for adaptive stepper."""
            return _compute_rhs_2d(self.grid, self.equations, U_current, self.artificial_viscosity, self.gravity_config)
        
        U_split = self.grid.apply_spectral_dissipation(U, 0.5 * dt)

        # Compute solution scale for relative error
        if _TORCH_AVAILABLE and isinstance(U_split, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U_split)).item())
        else:
            solution_scale = float(np.max(np.abs(U_split)))
        
        result = _call_adaptive_step(
            self, self.adaptive_stepper, rhs_func, U_split, dt, solution_scale,
            limit_state=lambda state: _limit_euler_stage(self, state),
        )
        
        if result.accepted:
            Un = _apply_accepted_state_stabilization_2d(
                self.grid, result.U_new, self.equations, dt=0.5 * dt
            )
        else:
            Un = U
            
        return Un

    def _adaptive_run_step(self, dt: float, step_count: int, 
                          on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in adaptive run loop."""
        def rhs_func(U_current: Array) -> Array:
            return _compute_rhs_2d(
                self.grid,
                self.equations,
                U_current,
                self.artificial_viscosity,
                self.gravity_config,
            )
        
        U_split = self.grid.apply_spectral_dissipation(self.U, 0.5 * dt)

        if _TORCH_AVAILABLE and isinstance(U_split, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U_split)).item())
        else:
            solution_scale = float(np.max(np.abs(U_split)))
        
        result = _call_adaptive_step(
            self, self.adaptive_stepper, rhs_func, U_split, dt, solution_scale,
            limit_state=lambda state: _limit_euler_stage(self, state),
        )
        
        if result.accepted:
            self.U = _apply_accepted_state_stabilization_2d(
                self.grid, result.U_new, self.equations, dt=0.5 * dt
            )
            self.t += dt
            _note_accepted(self)
            if on_step is not None:
                on_step(step_count, dt, self.U)
            return result.dt_next
        else:
            return result.dt_next
    
    def _fixed_run_step(self, dt: float, step_count: int,
                       on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in fixed run loop."""
        self.U = self.step(self.U, dt)
        self.t += dt
        _note_accepted(self)
        
        # Call monitoring callback
        if on_step is not None:
            on_step(step_count, dt, self.U)
        
        return self._stability_limit(self.U)

    def run(
        self,
        U0: Array,
        t0: float,
        t_end: float,
        output_interval: float = 0.1,
        checkpoint_interval: float = 0.0,
        outdir: Optional[str] = None,
        on_output: Optional[Callable[[float, Array], None]] = None,
        on_step: Optional[Callable[[int, float, Array], None]] = None,
        tracers: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        # Start Fourier runs in the same retained subspace enforced after steps.
        self.U = self.grid.project_dealiased(self.U)
        if getattr(self.grid, "_mixed_open_y", False):
            self.U = self.grid.initialize_boundary_conditions(self.U, self.equations)
        _begin_run(self, t0, t_end)
        next_output = self.t + output_interval
        next_checkpoint = (
            self.t + checkpoint_interval
            if checkpoint_interval and checkpoint_interval > 0
            else np.inf
        )

        history: Dict[str, List[float]] = {
            "time": [],
            "mass": [],
            "momentum_x": [],
            "momentum_y": [],
            "energy": [],
        }
        step_count = 0

        def record() -> None:
            if getattr(self.grid, "_mixed_open_y", False):
                cons = self.grid.conserved_quantities(self.U, self.equations)
                history["time"].append(self.t)
                for key in ("mass", "momentum_x", "momentum_y", "energy"):
                    history[key].append(cons[key])
                return
            rho, ux, uy, p = self.equations.primitive(self.U)  # type: ignore[arg-type]
            if _TORCH_AVAILABLE and isinstance(rho, (torch.Tensor,)):
                mass = float(torch.sum(rho).item())
                momx = float(torch.sum(rho * ux).item())
                momy = float(torch.sum(rho * uy).item())
                energy = float(
                    torch.sum(
                        p / (self.equations.gamma - 1.0)
                        + 0.5 * rho * (ux * ux + uy * uy)
                    ).item()
                )
            else:
                mass = float(rho.sum())
                momx = float((rho * ux).sum())
                momy = float((rho * uy).sum())
                energy = float(
                    (
                        p / (self.equations.gamma - 1.0)
                        + 0.5 * rho * (ux * ux + uy * uy)
                    ).sum()
                )
            history["time"].append(self.t)
            history["mass"].append(mass)
            history["momentum_x"].append(momx)
            history["momentum_y"].append(momy)
            history["energy"].append(energy)

        record()
        if on_output is not None:
            on_output(self.t, self.U)

        dt = float("inf")
        
        while self.t < t_end - 1e-12:
            dt = _cap_proposal(self, dt, t_end, next_output, next_checkpoint)
            dt_used = dt
            state_before = self.U
            time_before = self.t
            
            if self.adaptive_enabled and self.adaptive_stepper is not None:
                # Adaptive time-stepping
                dt = self._adaptive_run_step(dt, step_count, on_step)
            else:
                # Fixed time-stepping (original behavior)
                dt = self._fixed_run_step(dt, step_count, on_step)
            
            if tracers is not None and self.t > time_before:
                tracers.step(
                    self.grid,
                    self.equations,
                    state_before,
                    dt_used,
                    U_new=self.U,
                )
            step_count += 1

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(
                    outdir,
                    self.t,
                    U=self.U,
                    grid=self.grid,
                    equations=self.equations,
                    tracers=tracers,
                )
                next_checkpoint += checkpoint_interval

        _emit_final_output(history, self.t, record, on_output, self.U)
        return history


# ===================== Ideal/Resistive MHD solvers =====================
#
# 8-component conservative state U = [rho, rho*ux, rho*uy, rho*uz, E, Bx, By, Bz].
# The RHS is the same conservative flux-divergence structure as Euler; the
# magnetic solenoidal constraint div(B)=0 is enforced by Helmholtz (Leray)
# projection of B after every RK substage and (last) after spectral dissipation.
# Optional explicit resistivity/viscosity are applied as
# spectral Laplacians on the B and momentum rows; the energy row is left to the
# ideal flux so that dissipated magnetic/kinetic energy is converted to thermal
# energy through the pressure recovery (total energy conserving, div(B)-safe
# since div(laplacian B) = laplacian(div B) = 0).


def _mhd_clone(U: Array) -> Array:
    if _TORCH_AVAILABLE and isinstance(U, torch.Tensor):
        return U.clone()
    return U.copy()


def shearing_sheet_axes(vertical: int, radial: int) -> Tuple[int, int]:
    """Component index and sign of phi = z_v x r in the right-handed (x, y, z) basis."""
    if vertical == radial or not {vertical, radial} <= {0, 1, 2}:
        raise ValueError("Shearing-sheet vertical_axis and radial_axis must be distinct in {0, 1, 2}")
    return 3 - vertical - radial, 1 if (radial - vertical) % 3 == 1 else -1


def _max_abs_float(a: Array) -> float:
    if _TORCH_AVAILABLE and isinstance(a, torch.Tensor):
        return float(torch.max(torch.abs(a)).item())
    return float(np.max(np.abs(a)))


class SpectralSolverMHD:
    """Pseudo-spectral ideal/resistive MHD solver (1D/2D/3D autodetected).

    Mirrors the Euler solver interface (constructor signature, ``run`` signature)
    so it is a drop-in for ``BaseProblem.run``. The dimensionality is inferred
    from the grid. Runtime MHD options (resistivity, viscosity,
    project_each_substage, divergence_clean) are read from ``grid.mhd_config`` if
    present, and can also be set as attributes after construction.

    Artificial viscosity is applied only in 1D. Enabling it for 2D or 3D raises
    ``NotImplementedError``. Fixed Runge-Kutta substages clamp density and
    pressure and count those events. Embedded adaptive stages are not
    positivity-limited; only the accepted state is clamped, together with the
    magnetic projection. The hyperbolic CFL length is the minimum node spacing.
    """

    def __init__(self, grid, equations, U0: Optional[Array] = None,
                 scheme: str = "rk4", cfl: float = 0.3,
                 adaptive_config: Optional[Dict] = None,
                 artificial_viscosity_config: Optional[Dict] = None,
                 gravity_config: Optional[Dict] = None):
        self.grid = grid
        self.equations = equations
        self.cfl = cfl
        self.t = 0.0
        self.U = U0
        self.gravity_config = gravity_config

        # Infer dimensionality from the grid
        if hasattr(grid, "Nz") and getattr(grid, "Nz", None) is not None:
            self.dim = 3
        elif hasattr(grid, "Ny") and getattr(grid, "Ny", None) is not None:
            self.dim = 2
        else:
            self.dim = 1

        # MHD runtime options (from grid.mhd_config, with safe defaults)
        cfg = getattr(grid, "mhd_config", None) or {}
        self.resistivity = float(cfg.get("resistivity", 0.0))   # eta
        self.viscosity = float(cfg.get("viscosity", 0.0))       # nu
        self.project_each_substage = bool(cfg.get("project_each_substage", True))
        self.divergence_clean = str(cfg.get("divergence_clean", "projection"))
        if getattr(grid, "_mixed_open_y", False) and self.divergence_clean != "projection":
            raise ValueError("Mixed open MHD boundaries require magnetic projection")
        self.rho_min = float(cfg.get("rho_min", 1e-10))
        self.p_min = float(cfg.get("p_min", 1e-12))
        self.ambipolar = None
        ambipolar_config = cfg.get("ambipolar", {})
        if ambipolar_config.get("enabled", False):
            from .nonideal import MidpointAmbipolar

            self.ambipolar = MidpointAmbipolar(grid, ambipolar_config)
            if adaptive_config and adaptive_config.get("enabled", False):
                raise ValueError("Ambipolar splitting currently requires fixed-step RK integration")
        self.hall = None
        hall_config = cfg.get("hall", {})
        if hall_config.get("enabled", False):
            from .nonideal import Hall

            self.hall = Hall(grid, hall_config)
            if adaptive_config and adaptive_config.get("enabled", False):
                raise ValueError("Hall splitting currently requires fixed-step RK integration")
        self.shearing_sheet = None
        shear_config = cfg.get("shearing_sheet") or {}
        if shear_config.get("enabled", True) and shear_config:
            omega = float(shear_config.get("omega", 0.0))
            shear = (float(shear_config["shear"]) if "shear" in shear_config
                     else float(shear_config.get("q", 1.5)) * omega)
            vertical = int(shear_config.get("vertical_axis", 0))
            radial = int(shear_config.get("radial_axis", 1))
            if not (np.isfinite(omega) and np.isfinite(shear)):
                raise ValueError("Shearing-sheet omega and shear must be finite")
            if self.dim > 1:
                raise NotImplementedError(
                    "Shearing-sheet sources assume fields independent of the radial axis "
                    "and are implemented for 1D grids only")
            if vertical != 0 or radial not in (1, 2):
                raise ValueError("1D shearing sheet requires vertical_axis 0 and radial_axis 1 or 2")
            azimuthal, sign = shearing_sheet_axes(vertical, radial)
            self.shearing_sheet = dict(omega=omega, shear=shear, radial=radial,
                                       azimuthal=azimuthal, sign=sign)
        # Parabolic-CFL safety factor for explicit diffusion (RK4 stable region on
        # the negative real axis extends to ~2.78; 1.0 leaves a comfortable margin).
        self.cfl_diff = float(cfg.get("cfl_diff", 1.0))
        self._k2_max = self._grid_k2_max()
        self.artificial_viscosity = None
        if artificial_viscosity_config and artificial_viscosity_config.get("enabled", False):
            if self.dim != 1:
                _reject_unsupported_artificial_viscosity(
                    artificial_viscosity_config, f"{self.dim}D MHD"
                )
            from .artificial_viscosity import create_artificial_viscosity

            self.artificial_viscosity = create_artificial_viscosity(artificial_viscosity_config)
        self.scheme, self.adaptive_enabled, self.adaptive_stepper = (
            _configure_time_integration(scheme, adaptive_config)
        )
        _init_monitors(self, self.rho_min, self.p_min)

    def _grid_k2_max(self) -> float:
        """Max squared wavenumber on the grid (for the parabolic diffusion CFL).

        A missing multi-D wavenumber array uses ``sum 4/dx_i**2``. Any other
        failure propagates; returning 0 would silently drop the diffusive CFL.
        """
        g = self.grid
        if getattr(g, "_mixed_open_y", False):
            D = g._sbp_dy(np.eye(g.Ny))
            root = np.sqrt(g.wy)
            D = root[:, None]*D/root[None, :]
            ky2_bound = np.linalg.norm(D, 1)*np.linalg.norm(D, np.inf)
            kx = g.kx.detach().cpu().numpy() if g._use_torch else g.kx
            return float(np.max(kx*kx)+ky2_bound)
        if self.dim == 1:
            if getattr(g, "_basis_name", "fourier") != "fourier":
                return 4.0 / float(g.dx) ** 2
            k2 = g.k * g.k
        else:
            k2 = getattr(g, "k2", None)
            if k2 is None:
                return _spacing_wavenumber_bound(g)
        if _TORCH_AVAILABLE and isinstance(k2, torch.Tensor):
            return float(torch.max(k2).item())
        return float(np.max(np.asarray(k2)))

    # --- physics helpers ---
    def _flux_divergence(self, U: Array) -> Array:
        if self.dim == 1:
            dFdx, boundary_flux = self.grid.flux_derivative(
                U, self.equations, return_boundary_flux=True
            )
            return self.grid.apply_boundary_flux(-dFdx, U, self.equations, boundary_flux)
        F = self.equations.flux(U)
        if self.dim == 2:
            Fx, Fy = F
            rhs = -(self.grid.dx1(Fx) + self.grid.dy1(Fy))
            if getattr(self.grid, "_mixed_open_y", False):
                rhs = self.grid.apply_split_advection(rhs, U)
                rhs = self.grid.apply_boundary_flux(rhs, U, self.equations, Fy)
                rhs = self.grid.project_dealiased(rhs)
            return rhs
        Fx, Fy, Fz = F
        return -(self.grid.dx1(Fx) + self.grid.dy1(Fy) + self.grid.dz1(Fz))

    def _compute_rhs(self, U: Array) -> Array:
        if self.dim == 1 or getattr(self.grid, "_mixed_open_y", False):
            U = self.grid.apply_boundary_conditions(U, self.equations)
        rhs = self._flux_divergence(U)
        if (getattr(self.grid, "_mixed_open_y", False)
                and self.grid.sbp_dissipation_strength > 0.0):
            rhs += self.grid.sbp_dissipation(U, self.equations.max_wave_speed(U))
        if self.artificial_viscosity is not None:
            terms = self.artificial_viscosity.compute_viscosity_terms(U, self.grid, self.equations)
            for i, term in enumerate(terms):
                if i != 5:  # Preserve the longitudinal magnetic field in 1D.
                    rhs[i] += term
        eta = self.resistivity
        nu = self.viscosity
        if eta != 0.0:
            rhs[5] = rhs[5] + eta * _diffusion(self.grid, U[5])
            rhs[6] = rhs[6] + eta * _diffusion(self.grid, U[6])
            rhs[7] = rhs[7] + eta * _diffusion(self.grid, U[7])
        if nu != 0.0:
            rhs[1] = rhs[1] + nu * _diffusion(self.grid, U[1])
            rhs[2] = rhs[2] + nu * _diffusion(self.grid, U[2])
            rhs[3] = rhs[3] + nu * _diffusion(self.grid, U[3])
        if self.shearing_sheet is not None:
            self._add_shearing_sheet(U, rhs)
        return rhs

    def _add_shearing_sheet(self, U: Array, rhs: Array) -> None:
        """Axisymmetric shearing-sheet terms for the velocity u relative to v0 = -S r phi.

        d(rho u_r)/dt += 2 Omega rho u_phi, d(rho u_phi)/dt -= (2 Omega - S) rho u_r,
        dB_phi/dt -= S B_r, dE/dt += S (rho u_r u_phi - B_r B_phi); internal energy is
        unchanged and B_phi is generated along phi, so div B is untouched.
        """
        s = self.shearing_sheet
        r, f, sign = s["radial"], s["azimuthal"], s["sign"]
        mr, mf, Br, Bf = U[1 + r], U[1 + f], U[5 + r], U[5 + f]
        rhs[1 + r] += 2.0 * s["omega"] * sign * mf
        rhs[1 + f] -= (2.0 * s["omega"] - s["shear"]) * sign * mr
        rhs[5 + f] -= s["shear"] * sign * Br
        rhs[4] += s["shear"] * sign * (mr * mf / U[0] - Br * Bf)

    def _clamp(self, U: Array, *, rebuild_always: bool = True) -> Array:
        """Floor density and thermal pressure, rebuilding the total energy.

        Stages always rebuild, matching the previous MHD stage update. The
        accepted-state limiter passes ``rebuild_always=False`` so a positive
        state is returned untouched.
        """
        rho, ux, uy, uz, p, Bx, By, Bz = self.equations.primitive(U)
        below = _count_below(rho, self.rho_min) + _count_below(p, self.p_min)
        monitor = getattr(self, "positivity", None)
        if monitor is not None:
            monitor.note(below)
        if below == 0 and not rebuild_always:
            return U
        if _TORCH_AVAILABLE and isinstance(U, torch.Tensor):
            rho_c = torch.clamp(rho, min=self.rho_min)
            p_c = torch.clamp(p, min=self.p_min)
        else:
            rho_c = np.maximum(rho, self.rho_min)
            p_c = np.maximum(p, self.p_min)
        return self.equations.conservative(rho_c, ux, uy, uz, p_c, Bx, By, Bz)

    def _limit_accepted(self, U: Array) -> Array:
        """Restore positivity after the accepted-state stabilization.

        The hard 2/3 Fourier projection and the modal filter act on each
        conserved field separately and can ring below zero near steep fronts.
        Density and thermal pressure are floored as in the stage clamp, and
        the total energy is rebuilt from the stabilized B and velocity. B is
        not touched, so div B stays at the projection's round-off. A state
        that is already positive is returned unchanged.
        """
        return self._clamp(U, rebuild_always=False)

    def _project_B(self, U: Array) -> Array:
        """Helmholtz-project B and recompute E holding thermal pressure fixed."""
        if self.divergence_clean == "none":
            return U
        rho, ux, uy, uz, p, Bx, By, Bz = self.equations.primitive(U)
        if self.dim == 1:
            Bx = self.grid.project_solenoidal_x(U[5])
        elif self.dim == 2:
            Bx, By = self.grid.project_solenoidal(U[5], U[6])
        else:
            Bx, By, Bz = self.grid.project_solenoidal(U[5], U[6], U[7])
        U = _mhd_clone(U)
        U[5] = Bx
        U[6] = By
        U[7] = Bz
        U[4] = (
            p / (self.equations.gamma - 1.0)
            + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            + 0.5 * (Bx * Bx + By * By + Bz * Bz)
        )
        return U

    def _rk_substages(self, U: Array, dt: float) -> Array:
        proj = self._project_B if self.project_each_substage else (lambda x: x)
        rhs = self._compute_rhs
        clamp = self._clamp
        if self.scheme.lower() == "rk2":
            k1 = rhs(U)
            U1 = proj(clamp(U + 0.5 * dt * k1))
            k2 = rhs(U1)
            Un = proj(clamp(U + dt * k2))
        else:  # rk4
            k1 = rhs(U)
            U2 = proj(clamp(U + 0.5 * dt * k1))
            k2 = rhs(U2)
            U3 = proj(clamp(U + 0.5 * dt * k2))
            k3 = rhs(U3)
            U4 = proj(clamp(U + dt * k3))
            k4 = rhs(U4)
            Un = proj(clamp(U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)))
        return Un

    def max_div_b(self, U: Array) -> float:
        if self.dim == 1:
            d = self.grid.divergence_x(U[5])
        elif self.dim == 2:
            d = self.grid.divergence(U[5], U[6])
        else:
            d = self.grid.divergence(U[5], U[6], U[7])
        return _max_abs_float(d)

    # --- time stepping ---
    def _stability_limit(self, U: Array) -> float:
        speed = _checked_wave_speed(self, U)
        bounds = [_hyperbolic_dt(self.cfl, _min_spacing(self.grid), speed)]
        if self.dim == 1:
            bounds.append(_artificial_viscosity_dt(self.grid, self.artificial_viscosity))
        if self.shearing_sheet is not None:
            rate_shear = 2.0 * abs(self.shearing_sheet["omega"]) + abs(self.shearing_sheet["shear"])
            if rate_shear > 0.0:
                bounds.append(self.cfl / rate_shear)
        # Parabolic CFL for explicit resistivity/viscosity: dt <= cfl_diff/(D k^2_max)
        diff = max(self.resistivity, self.viscosity)
        rate = diff * self._k2_max
        if getattr(self.grid, "_mixed_open_y", False):
            rate += self.grid.sbp_dissipation_rate_bound(speed)
        if rate > 0.0:
            if not np.isfinite(rate):
                raise FloatingPointError(
                    f"Non-finite diffusive rate ({rate}) at "
                    f"step {self._accepted_steps}, t={self.t}"
                )
            bounds.append(self.cfl_diff / rate)
        return _smallest_finite(bounds)

    def compute_dt(self, U: Array) -> float:
        return _public_dt(self, U)

    def step(self, U: Array, dt: float) -> Array:
        if self.adaptive_enabled and self.adaptive_stepper is not None:
            return self._adaptive_step(U, dt)
        return self._fixed_step(U, dt)

    def _fixed_step(self, U: Array, dt: float) -> Array:
        if getattr(self.grid, "_mixed_open_y", False):
            U = self.grid.apply_boundary_conditions(U, self.equations)
        U_split = self.grid.apply_spectral_dissipation(U, 0.5 * dt)
        # Hall and ambipolar halves mirror each other around the ideal step.
        if self.hall is not None:
            U_split = self.hall.step(U_split, self.equations, 0.5 * dt)
        if self.ambipolar is not None:
            U_split = self.ambipolar.step(U_split, self.equations, 0.5 * dt)
        Un = self._rk_substages(U_split, dt)
        # Project B before the scalar modal operator; the latter commutes with
        # divergence, so the stored state remains both band-limited and div-free.
        Un = self._project_B(Un)
        if self.ambipolar is not None:
            Un = self.ambipolar.step(Un, self.equations, 0.5 * dt)
        if self.hall is not None:
            Un = self.hall.step(Un, self.equations, 0.5 * dt)
        Un = self._limit_accepted(self.grid.apply_accepted_state_stabilization(Un, 0.5 * dt))
        if self.dim == 1:
            Un = self.grid.apply_boundary_conditions(Un, self.equations)
        elif getattr(self.grid, "_mixed_open_y", False):
            # Finite-tolerance CG need not preserve the SBP divergence exactly.
            Un = self._project_B(Un)
        return Un

    def _adaptive_step(self, U: Array, dt: float) -> Array:
        def rhs_func(Uc: Array) -> Array:
            return self._compute_rhs(Uc)

        U_split = self.grid.apply_spectral_dissipation(U, 0.5 * dt)
        if _TORCH_AVAILABLE and isinstance(U_split, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U_split)).item())
        else:
            solution_scale = float(np.max(np.abs(U_split)))
        result = _call_adaptive_step(
            self, self.adaptive_stepper, rhs_func, U_split, dt, solution_scale
        )
        if result.accepted:
            Un = self._project_B(self._clamp(result.U_new))
            Un = self._limit_accepted(
                self.grid.apply_accepted_state_stabilization(Un, 0.5 * dt)
            )
            if self.dim == 1:
                Un = self.grid.apply_boundary_conditions(Un, self.equations)
        else:
            Un = U
        return Un

    def _adaptive_run_step(self, dt: float, step_count: int,
                           on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        def rhs_func(Uc: Array) -> Array:
            return self._compute_rhs(Uc)

        U_split = self.grid.apply_spectral_dissipation(self.U, 0.5 * dt)
        if _TORCH_AVAILABLE and isinstance(U_split, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U_split)).item())
        else:
            solution_scale = float(np.max(np.abs(U_split)))
        # Embedded MHD stages are not positivity-limited. The accepted state is.
        result = _call_adaptive_step(
            self, self.adaptive_stepper, rhs_func, U_split, dt, solution_scale
        )
        if result.accepted:
            Un = self._project_B(self._clamp(result.U_new))
            Un = self._limit_accepted(
                self.grid.apply_accepted_state_stabilization(Un, 0.5 * dt)
            )
            if self.dim == 1:
                Un = self.grid.apply_boundary_conditions(Un, self.equations)
            self.U = Un
            self.t += dt
            _note_accepted(self)
            if on_step is not None:
                on_step(step_count, dt, self.U)
            return result.dt_next
        return result.dt_next

    def _fixed_run_step(self, dt: float, step_count: int,
                        on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        self.U = self.step(self.U, dt)
        self.t += dt
        _note_accepted(self)
        if on_step is not None:
            on_step(step_count, dt, self.U)
        return self._stability_limit(self.U)

    def run(self, U0: Array, t0: float, t_end: float,
            output_interval: float = 0.1, checkpoint_interval: float = 0.0,
            outdir: Optional[str] = None,
            on_output: Optional[Callable[[float, Array], None]] = None,
            on_step: Optional[Callable[[int, float, Array], None]] = None,
            tracers: Optional[Any] = None) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        # Project the initial state so div(B)=0 from the first stored step.
        self.U = self._project_B(self.U)
        self.U = self.grid.project_dealiased(self.U)
        if self.dim == 1 or getattr(self.grid, "_mixed_open_y", False):
            self.U = self.grid.initialize_boundary_conditions(self.U, self.equations)
        _begin_run(self, t0, t_end)
        next_output = self.t + output_interval
        next_checkpoint = (
            self.t + checkpoint_interval
            if checkpoint_interval and checkpoint_interval > 0
            else np.inf
        )

        history: Dict[str, List[float]] = {
            "time": [], "mass": [], "momentum_x": [], "momentum_y": [],
            "momentum_z": [], "energy": [], "magnetic_energy": [],
            "cross_helicity": [], "div_b_max": [],
        }
        step_count = 0

        def record() -> None:
            cons = (self.grid.conserved_quantities(self.U, self.equations)
                    if getattr(self.grid, "_mixed_open_y", False)
                    else conserved_diagnostics(self.grid, self.equations, self.U))
            history["time"].append(self.t)
            for key in ("mass", "momentum_x", "momentum_y", "momentum_z",
                        "energy", "magnetic_energy", "cross_helicity"):
                history[key].append(cons[key])
            history["div_b_max"].append(self.max_div_b(self.U))

        record()
        if on_output is not None:
            on_output(self.t, self.U)

        dt = float("inf")
        while self.t < t_end - 1e-12:
            dt = _cap_proposal(self, dt, t_end, next_output, next_checkpoint)
            dt_used = dt
            state_before = self.U
            time_before = self.t
            if self.adaptive_enabled and self.adaptive_stepper is not None:
                dt = self._adaptive_run_step(dt, step_count, on_step)
            else:
                dt = self._fixed_run_step(dt, step_count, on_step)
            if tracers is not None and self.t > time_before:
                tracers.step(
                    self.grid,
                    self.equations,
                    state_before,
                    dt_used,
                    U_new=self.U,
                )
            step_count += 1

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(outdir, self.t, U=self.U, grid=self.grid,
                                       equations=self.equations, tracers=tracers)
                next_checkpoint += checkpoint_interval

        _emit_final_output(history, self.t, record, on_output, self.U)
        return history


class SpectralSolverMHD1D(SpectralSolverMHD):
    """1D ideal/resistive MHD spectral solver."""


class SpectralSolverMHD2D(SpectralSolverMHD):
    """2D (2.5D) ideal/resistive MHD spectral solver."""


class SpectralSolverMHD3D(SpectralSolverMHD):
    """3D ideal/resistive MHD spectral solver."""


def _compute_rhs_3d(grid: Grid3D, eqs: EulerEquations3D, U: Array,
                   artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                   gravity_config: Optional[Dict] = None) -> Array:
    Fx, Fy, Fz = eqs.flux(U)
    dFdx = grid.dx1(Fx)
    dFdy = grid.dy1(Fy)
    dFdz = grid.dz1(Fz)
    rhs = -(dFdx + dFdy + dFdz)
    
    # Add artificial viscosity if enabled
    if artificial_viscosity is not None:
        viscosity_terms = artificial_viscosity.compute_viscosity_terms(U, grid, eqs)
        for i, term in enumerate(viscosity_terms):
            if i < len(rhs):
                rhs[i] += term
    
    # Add gravity source terms if enabled
    if gravity_config and gravity_config.get("enabled", False):
        rho, ux, uy, uz, p = eqs.primitive(U)
        gx = gravity_config.get("gx", 0.0)
        gy = gravity_config.get("gy", 0.0)
        gz = gravity_config.get("gz", 0.0)
        
        # Momentum equations: d(ρux)/dt += ρgx, d(ρuy)/dt += ρgy, d(ρuz)/dt += ρgz
        rhs[1] += rho * gx
        rhs[2] += rho * gy
        rhs[3] += rho * gz
        
        # Energy equation: dE/dt += ρ(gx*ux + gy*uy + gz*uz) (work done by gravity)
        rhs[4] += rho * (gx * ux + gy * uy + gz * uz)
    
    return rhs


def _rk2_step_3d(grid: Grid3D, eqs: EulerEquations3D, U: Array, dt: float,
                 artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                 gravity_config: Optional[Dict] = None,
                 monitor: Optional[PositivityMonitor] = None,
                 rho_min: float = _RHO_FLOOR_1D,
                 p_min: float = _P_FLOOR_1D) -> Array:
    def clamp(state):
        return _clamp_conserved(
            state, eqs, rho_min, p_min, monitor, rebuild_always=False
        )

    k1 = _compute_rhs_3d(grid, eqs, U, artificial_viscosity, gravity_config)
    U1 = clamp(U + dt * 0.5 * k1)
    k2 = _compute_rhs_3d(grid, eqs, U1, artificial_viscosity, gravity_config)
    return clamp(U + dt * k2)


def _rk4_step_3d(grid: Grid3D, eqs: EulerEquations3D, U: Array, dt: float,
                 artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                 gravity_config: Optional[Dict] = None,
                 monitor: Optional[PositivityMonitor] = None,
                 rho_min: float = _RHO_FLOOR_1D,
                 p_min: float = _P_FLOOR_1D) -> Array:
    def clamp(state):
        return _clamp_conserved(
            state, eqs, rho_min, p_min, monitor, rebuild_always=False
        )

    k1 = _compute_rhs_3d(grid, eqs, U, artificial_viscosity, gravity_config)
    U2 = clamp(U + 0.5 * dt * k1)
    k2 = _compute_rhs_3d(grid, eqs, U2, artificial_viscosity, gravity_config)
    U3 = clamp(U + 0.5 * dt * k2)
    k3 = _compute_rhs_3d(grid, eqs, U3, artificial_viscosity, gravity_config)
    U4 = clamp(U + dt * k3)
    k4 = _compute_rhs_3d(grid, eqs, U4, artificial_viscosity, gravity_config)
    return clamp(U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4))


def _apply_accepted_state_stabilization_3d(
    grid: Grid3D, U: Array, dt: float = 0.0
) -> Array:
    return grid.apply_accepted_state_stabilization(U, dt)


@dataclass
class SpectralSolver3D:
    grid: Grid3D
    equations: EulerEquations3D
    scheme: str = "rk4"
    cfl: float = 0.25

    t: float = 0.0
    U: Optional[Array] = None
    
    # Adaptive time-stepping
    adaptive_stepper: Optional[AdaptiveTimeStepper] = None
    adaptive_enabled: bool = False
    
    # Gravity configuration
    gravity_config: Optional[Dict] = None

    def __init__(self, grid: Grid3D, equations: EulerEquations3D, U0: Optional[Array] = None,
                 scheme: str = "rk4", cfl: float = 0.25, adaptive_config: Optional[Dict] = None,
                 artificial_viscosity_config: Optional[Dict] = None, gravity_config: Optional[Dict] = None):
        """Initialize the 3D spectral solver.

        Args:
            grid: 3D grid
            equations: 3D Euler equations
            U0: Initial solution (optional)
            scheme: Time integration scheme ("rk2", "rk4", "rk23", "rk45", "rk78")
            cfl: CFL number
            adaptive_config: Configuration for adaptive time-stepping
            artificial_viscosity_config: Not implemented. ``enabled: true`` raises.
            gravity_config: Configuration for gravity
        """
        self.grid = grid
        self.equations = equations
        self.cfl = cfl
        self.t = 0.0
        self.U = U0
        _reject_unsupported_artificial_viscosity(
            artificial_viscosity_config, "SpectralSolver3D"
        )
        self.scheme, self.adaptive_enabled, self.adaptive_stepper = (
            _configure_time_integration(scheme, adaptive_config)
        )
        _init_monitors(self, _RHO_FLOOR_1D, _P_FLOOR_1D)

        # Setup gravity if configured
        self.gravity_config = gravity_config

    def _stability_limit(self, U: Array) -> float:
        speed = _checked_wave_speed(self, U)
        return _smallest_finite((
            _hyperbolic_dt(self.cfl, _min_spacing(self.grid), speed),
        ))

    def compute_dt(self, U: Array) -> float:
        return _public_dt(self, U)

    def step(self, U: Array, dt: float) -> Array:
        """Perform one time step using the configured scheme.
        
        Args:
            U: Current solution vector
            dt: Time step
            
        Returns:
            New solution vector
        """
        if self.adaptive_enabled and self.adaptive_stepper is not None:
            # Use adaptive time-stepping
            return self._adaptive_step(U, dt)
        else:
            # Use fixed time-stepping
            return self._fixed_step(U, dt)
    
    def _fixed_step(self, U: Array, dt: float) -> Array:
        """Perform one fixed time step."""
        U_split = self.grid.apply_spectral_dissipation(U, 0.5 * dt)
        if self.scheme.lower() == "rk2":
            Un = _rk2_step_3d(
                self.grid, self.equations, U_split, dt, None, self.gravity_config,
                self.positivity, self.rho_floor, self.p_floor,
            )
        else:
            Un = _rk4_step_3d(
                self.grid, self.equations, U_split, dt, None, self.gravity_config,
                self.positivity, self.rho_floor, self.p_floor,
            )
        return _apply_accepted_state_stabilization_3d(
            self.grid, Un, dt=0.5 * dt
        )
    
    def _adaptive_step(self, U: Array, dt: float) -> Array:
        """Perform one adaptive time step."""
        def rhs_func(U_current: Array) -> Array:
            """RHS function for adaptive stepper."""
            return _compute_rhs_3d(self.grid, self.equations, U_current, None, self.gravity_config)
        
        U_split = self.grid.apply_spectral_dissipation(U, 0.5 * dt)

        # Compute solution scale for relative error
        if _TORCH_AVAILABLE and isinstance(U_split, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U_split)).item())
        else:
            solution_scale = float(np.max(np.abs(U_split)))
        
        result = _call_adaptive_step(
            self, self.adaptive_stepper, rhs_func, U_split, dt, solution_scale,
            limit_state=lambda state: _limit_euler_stage(self, state),
        )
        
        if result.accepted:
            Un = _apply_accepted_state_stabilization_3d(
                self.grid, result.U_new, dt=0.5 * dt
            )
        else:
            Un = U
            
        return Un

    def _adaptive_run_step(self, dt: float, step_count: int, 
                          on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in adaptive run loop."""
        def rhs_func(U_current: Array) -> Array:
            return _compute_rhs_3d(self.grid, self.equations, U_current, None, self.gravity_config)
        
        U_split = self.grid.apply_spectral_dissipation(self.U, 0.5 * dt)

        if _TORCH_AVAILABLE and isinstance(U_split, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U_split)).item())
        else:
            solution_scale = float(np.max(np.abs(U_split)))
        
        result = _call_adaptive_step(
            self, self.adaptive_stepper, rhs_func, U_split, dt, solution_scale,
            limit_state=lambda state: _limit_euler_stage(self, state),
        )
        
        if result.accepted:
            self.U = _apply_accepted_state_stabilization_3d(
                self.grid, result.U_new, dt=0.5 * dt
            )
            self.t += dt
            _note_accepted(self)
            if on_step is not None:
                on_step(step_count, dt, self.U)
            return result.dt_next
        else:
            return result.dt_next
    
    def _fixed_run_step(self, dt: float, step_count: int,
                       on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in fixed run loop."""
        self.U = self.step(self.U, dt)
        self.t += dt
        _note_accepted(self)
        
        # Call monitoring callback
        if on_step is not None:
            on_step(step_count, dt, self.U)
        
        return self._stability_limit(self.U)

    def run(
        self,
        U0: Array,
        t0: float,
        t_end: float,
        output_interval: float = 0.2,
        checkpoint_interval: float = 0.0,
        outdir: Optional[str] = None,
        on_output: Optional[Callable[[float, Array], None]] = None,
        on_step: Optional[Callable[[int, float, Array], None]] = None,
        tracers: Optional[Any] = None,
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        # Start Fourier runs in the same retained subspace enforced after steps.
        self.U = self.grid.project_dealiased(self.U)
        _begin_run(self, t0, t_end)
        next_output = self.t + output_interval
        next_checkpoint = (
            self.t + checkpoint_interval
            if checkpoint_interval and checkpoint_interval > 0
            else np.inf
        )

        history: Dict[str, List[float]] = {
            "time": [],
            "mass": [],
            "momentum_x": [],
            "momentum_y": [],
            "momentum_z": [],
            "energy": [],
        }
        step_count = 0

        def record() -> None:
            rho, ux, uy, uz, p = self.equations.primitive(self.U)  # type: ignore[arg-type]
            if _TORCH_AVAILABLE and isinstance(rho, (torch.Tensor,)):
                mass = float(torch.sum(rho).item())
                momx = float(torch.sum(rho * ux).item())
                momy = float(torch.sum(rho * uy).item())
                momz = float(torch.sum(rho * uz).item())
                energy = float(
                    torch.sum(
                        p / (self.equations.gamma - 1.0)
                        + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
                    ).item()
                )
            else:
                mass = float(rho.sum())
                momx = float((rho * ux).sum())
                momy = float((rho * uy).sum())
                momz = float((rho * uz).sum())
                energy = float(
                    (
                        p / (self.equations.gamma - 1.0)
                        + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
                    ).sum()
                )
            history["time"].append(self.t)
            history["mass"].append(mass)
            history["momentum_x"].append(momx)
            history["momentum_y"].append(momy)
            history["momentum_z"].append(momz)
            history["energy"].append(energy)

        record()
        if on_output is not None:
            on_output(self.t, self.U)

        dt = float("inf")
        
        while self.t < t_end - 1e-12:
            dt = _cap_proposal(self, dt, t_end, next_output, next_checkpoint)
            dt_used = dt
            state_before = self.U
            time_before = self.t
            
            if self.adaptive_enabled and self.adaptive_stepper is not None:
                # Adaptive time-stepping
                dt = self._adaptive_run_step(dt, step_count, on_step)
            else:
                # Fixed time-stepping (original behavior)
                dt = self._fixed_run_step(dt, step_count, on_step)
            
            if tracers is not None and self.t > time_before:
                tracers.step(
                    self.grid,
                    self.equations,
                    state_before,
                    dt_used,
                    U_new=self.U,
                )
            step_count += 1

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(
                    outdir,
                    self.t,
                    U=self.U,
                    grid=self.grid,
                    equations=self.equations,
                    tracers=tracers,
                )
                next_checkpoint += checkpoint_interval

        _emit_final_output(history, self.t, record, on_output, self.U)
        return history
