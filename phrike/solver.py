from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

from .grid import Grid1D, Grid2D, Grid3D
from .equations import EulerEquations1D, EulerEquations2D, EulerEquations3D
from .adaptive import AdaptiveTimeStepper, create_adaptive_stepper
from .artificial_viscosity import SpectralArtificialViscosity, ArtificialViscosityConfig


Array = np.ndarray


def _positivity_clamp(U: Array, eqs: EulerEquations1D, rho_min: float = 1e-10, p_min: float = 1e-12) -> Array:
    """Clamp density and pressure to be positive (lightweight limiter).

    Args:
        U: Conservative variables (3, N)
        eqs: Euler equations object
        rho_min: Minimum allowed density
        p_min: Minimum allowed pressure
    Returns:
        Clamped conservative variables of same shape.
    """
    try:
        import torch  # type: ignore
        is_torch = isinstance(U, torch.Tensor)
    except Exception:
        is_torch = False
        torch = None  # type: ignore

    rho, u, p, _ = eqs.primitive(U)
    if is_torch:  # type: ignore[truthy-bool]
        rho_clamped = torch.clamp(rho, min=float(rho_min))
        p_clamped = torch.clamp(p, min=float(p_min))
        Uc = eqs.conservative(rho_clamped, u, p_clamped)
    else:
        rho_clamped = np.maximum(rho, rho_min)
        p_clamped = np.maximum(p, p_min)
        Uc = eqs.conservative(rho_clamped, u, p_clamped)
    return Uc


def _compute_rhs(grid: Grid1D, eqs: EulerEquations1D, U: Array, 
                artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                gravity_config: Optional[Dict] = None) -> Array:
    # Pseudo-spectral: compute flux in physical space, then differentiate spectrally
    # Enforce boundary conditions for non-periodic bases before computing flux
    if hasattr(grid, "apply_boundary_conditions"):
        U = grid.apply_boundary_conditions(U, eqs)
    F = eqs.flux(U)
    # Batched spectral derivative across components (shape (3, N))
    dFdx = grid.dx1(F)
    # Euler in conservation form: dU/dt = - dF/dx
    rhs = -dFdx
    
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
              gravity_config: Optional[Dict] = None) -> Array:
    k1 = _compute_rhs(grid, eqs, U, artificial_viscosity, gravity_config)
    U1 = U + dt * 0.5 * k1
    U1 = _positivity_clamp(U1, eqs)
    k2 = _compute_rhs(grid, eqs, U1, artificial_viscosity, gravity_config)
    Unew = U + dt * k2
    Unew = _positivity_clamp(Unew, eqs)
    return _apply_physical_filters(grid, Unew, eqs)


def _rk4_step(grid: Grid1D, eqs: EulerEquations1D, U: Array, dt: float,
              artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
              gravity_config: Optional[Dict] = None) -> Array:
    k1 = _compute_rhs(grid, eqs, U, artificial_viscosity, gravity_config)
    U2 = _positivity_clamp(U + 0.5 * dt * k1, eqs)
    k2 = _compute_rhs(grid, eqs, U2, artificial_viscosity, gravity_config)
    U3 = _positivity_clamp(U + 0.5 * dt * k2, eqs)
    k3 = _compute_rhs(grid, eqs, U3, artificial_viscosity, gravity_config)
    U4 = _positivity_clamp(U + dt * k3, eqs)
    k4 = _compute_rhs(grid, eqs, U4, artificial_viscosity, gravity_config)
    Unew = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    Unew = _positivity_clamp(Unew, eqs)
    return _apply_physical_filters(grid, Unew, eqs)


def _apply_physical_filters(grid: Grid1D, U: Array, equations: Optional[EulerEquations1D] = None) -> Array:
    # Apply optional spectral filter to each component to suppress Gibbs/aliasing
    U_filtered = grid.apply_spectral_filter(U)
    # Re-apply boundary conditions after filtering to prevent endpoint drift
    if equations is not None and hasattr(grid, "apply_boundary_conditions"):
        try:
            U_filtered = grid.apply_boundary_conditions(U_filtered, equations)
        except Exception:
            pass
    return U_filtered


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
    # Filter cadence
    filter_interval: int = 1

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
        self.scheme = scheme
        self.cfl = cfl
        self.t = 0.0
        self.U = U0
        
        # Setup adaptive time-stepping if configured
        self.adaptive_enabled = False
        self.adaptive_stepper = None
        
        if adaptive_config and adaptive_config.get("enabled", False):
            adaptive_method = adaptive_config.get("scheme", "rk45")
            rtol = adaptive_config.get("rtol", 1e-6)
            atol = adaptive_config.get("atol", 1e-8)
            
            # Extract controller parameters
            controller_kwargs = {
                "safety_factor": adaptive_config.get("safety_factor", 0.9),
                "min_dt_factor": adaptive_config.get("min_dt_factor", 0.1),
                "max_dt_factor": adaptive_config.get("max_dt_factor", 5.0),
                "max_rejections": adaptive_config.get("max_rejections", 10)
            }
            
            # Check if scheme is adaptive
            if adaptive_method in ["rk23", "rk45", "rk78"]:
                self.adaptive_stepper = create_adaptive_stepper(
                    adaptive_method, rtol=rtol, atol=atol, **controller_kwargs
                )
                self.adaptive_enabled = True
                self.scheme = adaptive_method
            else:
                # Fall back to non-adaptive scheme
                self.scheme = adaptive_config.get("fallback_scheme", "rk4")
        
        # Setup artificial viscosity if configured
        self.artificial_viscosity = None
        if artificial_viscosity_config and artificial_viscosity_config.get("enabled", False):
            from .artificial_viscosity import create_artificial_viscosity
            self.artificial_viscosity = create_artificial_viscosity(artificial_viscosity_config)

        # Setup gravity if configured
        self.gravity_config = gravity_config

        # Filter interval: use integration.spectral_filter.interval if present
        # If AV is enabled, default to 5; else 1
        try:
            from .problems.base import BaseProblem  # type: ignore
        except Exception:
            BaseProblem = None  # type: ignore
        # Pull interval from grid.filter_params if provided via Problem
        cfg = getattr(self.grid, "filter_params", None)
        interval_cfg = None
        if isinstance(cfg, dict):
            interval_cfg = cfg.get("interval", None)
        if interval_cfg is not None:
            try:
                self.filter_interval = max(1, int(interval_cfg))
            except Exception:
                self.filter_interval = 1
        else:
            self.filter_interval = 5 if self.artificial_viscosity is not None else 1


    def compute_dt(self, U: Array) -> float:
        max_speed = self.equations.max_wave_speed(U)
        # CFL for spectral (heuristic): dt <= cfl * dx / max_speed
        if max_speed <= 0.0:
            return 1e-6
        return self.cfl * self.grid.dx / max_speed

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
        if self.scheme.lower() == "rk2":
            Un = _rk2_step(self.grid, self.equations, U, dt, self.artificial_viscosity, self.gravity_config)
        else:
            Un = _rk4_step(self.grid, self.equations, U, dt, self.artificial_viscosity, self.gravity_config)
        # Apply spectral filter with cadence
        if (getattr(self, "_step_counter", 0) % self.filter_interval) == 0:
            Un = _apply_physical_filters(self.grid, Un, self.equations)
        self._step_counter = getattr(self, "_step_counter", 0) + 1
        return Un
    
    def _adaptive_step(self, U: Array, dt: float) -> Array:
        """Perform one adaptive time step."""
        def rhs_func(U_current: Array) -> Array:
            """RHS function for adaptive stepper."""
            return _compute_rhs(self.grid, self.equations, U_current, self.artificial_viscosity, self.gravity_config)
        
        # Compute solution scale for relative error
        if _TORCH_AVAILABLE and isinstance(U, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U)).item())
        else:
            solution_scale = float(np.max(np.abs(U)))
        
        # Perform adaptive step
        result = self.adaptive_stepper.step(rhs_func, U, dt, solution_scale)
        
        # Apply spectral filtering to accepted solution
        if result.accepted:
            Un = _apply_physical_filters(self.grid, result.U_new, self.equations)
        else:
            # Return original solution if step was rejected
            Un = U
            
        return Un
    
    def _adaptive_run_step(self, dt: float, step_count: int, 
                          on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in adaptive run loop."""
        def rhs_func(U_current: Array) -> Array:
            return _compute_rhs(self.grid, self.equations, U_current, self.artificial_viscosity, self.gravity_config)
        
        # Compute solution scale for relative error
        if _TORCH_AVAILABLE and isinstance(self.U, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(self.U)).item())
        else:
            solution_scale = float(np.max(np.abs(self.U)))
        
        # Perform adaptive step
        result = self.adaptive_stepper.step(rhs_func, self.U, dt, solution_scale)
        
        if result.accepted:
            # Step accepted - update solution and time
            self.U = _apply_physical_filters(self.grid, result.U_new)
            self.t += dt
            
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
        
        # Call monitoring callback
        if on_step is not None:
            on_step(step_count, dt, self.U)
        
        # Return next CFL-based step size
        return min(self.compute_dt(self.U), 1e-6 if self.equations.max_wave_speed(self.U) <= 0.0 else float('inf'))

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
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        self.t = float(t0)
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
            cons = self.equations.conserved_quantities(self.U)  # type: ignore[arg-type]
            history["time"].append(self.t)
            history["mass"].append(cons["mass"])  # type: ignore[index]
            history["momentum"].append(cons["momentum"])  # type: ignore[index]
            history["energy"].append(cons["energy"])  # type: ignore[index]

        record()
        if on_output is not None:
            on_output(self.t, self.U)

        # Initialize time step
        dt = self.compute_dt(self.U) if self.U is not None else 1e-6
        
        while self.t < t_end - 1e-12:
            # Ensure we don't overshoot the end time
            dt = min(dt, t_end - self.t)
            
            if self.adaptive_enabled and self.adaptive_stepper is not None:
                # Adaptive time-stepping
                dt = self._adaptive_run_step(dt, step_count, on_step)
            else:
                # Fixed time-stepping (original behavior)
                dt = self._fixed_run_step(dt, step_count, on_step)
            
            step_count += 1

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(
                    outdir, self.t, U=self.U, grid=self.grid, equations=self.equations
                )
                next_checkpoint += checkpoint_interval

        record()
        if on_output is not None:
            on_output(self.t, self.U)
        return history


# 2D solver


def _compute_rhs_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array, 
                   artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                   gravity_config: Optional[Dict] = None) -> Array:
    # Mirror 1D: pre-apply BCs via grid if available (Fourier or hybrid). 2D Legendre disabled.
    if hasattr(grid, "apply_boundary_conditions"):
        try:
            U = grid.apply_boundary_conditions(U, eqs)
        except Exception:
            pass
    # Fluxes in x and y
    Fx, Fy = eqs.flux(U)
    dFdx = grid.dx1(Fx)
    dFdy = grid.dy1(Fy)
    # dU/dt = -(dFx/dx + dFy/dy)
    rhs = -(dFdx + dFdy)

    # Remove SAT/Rusanov boundary fluxes to mirror 1D behavior
    
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
def _positivity_clamp_2d(U: Array, eqs: EulerEquations2D, rho_min: float = 1e-8, p_min: float = 1e-10) -> Array:
    """Clamp density and pressure to small positive bounds in 2D.

    Rebuilds conservative state from clamped (rho, ux, uy, p) to avoid negative pressure blow-ups.
    """
    try:
        import torch  # type: ignore
        is_torch = isinstance(U, torch.Tensor)
    except Exception:
        is_torch = False
        torch = None  # type: ignore

    rho, ux, uy, p = eqs.primitive(U)
    if is_torch:  # type: ignore[truthy-bool]
        rho_c = torch.clamp(rho, min=float(rho_min))
        p_c = torch.clamp(p, min=float(p_min))
        Uc = eqs.conservative(rho_c, ux, uy, p_c)
    else:
        rho_c = np.maximum(rho, rho_min)
        p_c = np.maximum(p, p_min)
        Uc = eqs.conservative(rho_c, ux, uy, p_c)
    return Uc



def _rk2_step_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array, dt: float,
                 artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                 gravity_config: Optional[Dict] = None) -> Array:
    k1 = _compute_rhs_2d(grid, eqs, U, artificial_viscosity, gravity_config)
    U1 = _positivity_clamp_2d(U + dt * 0.5 * k1, eqs)
    k2 = _compute_rhs_2d(grid, eqs, U1, artificial_viscosity, gravity_config)
    Unew = _positivity_clamp_2d(U + dt * k2, eqs)
    return Unew


def _rk4_step_2d(grid: Grid2D, eqs: EulerEquations2D, U: Array, dt: float,
                 artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                 gravity_config: Optional[Dict] = None) -> Array:
    k1 = _compute_rhs_2d(grid, eqs, U, artificial_viscosity, gravity_config)
    U2 = _positivity_clamp_2d(U + 0.5 * dt * k1, eqs)
    k2 = _compute_rhs_2d(grid, eqs, U2, artificial_viscosity, gravity_config)
    U3 = _positivity_clamp_2d(U + 0.5 * dt * k2, eqs)
    k3 = _compute_rhs_2d(grid, eqs, U3, artificial_viscosity, gravity_config)
    U4 = _positivity_clamp_2d(U + dt * k3, eqs)
    k4 = _compute_rhs_2d(grid, eqs, U4, artificial_viscosity, gravity_config)
    Unew = U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return _positivity_clamp_2d(Unew, eqs)


def _apply_physical_filters_2d(grid: Grid2D, U: Array) -> Array:
    U_filtered = grid.apply_spectral_filter(U)
    # Re-apply BCs after filtering if available
    try:
        if hasattr(grid, "apply_boundary_conditions"):
            return grid.apply_boundary_conditions(U_filtered, None)
    except Exception:
        pass
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
        self.scheme = scheme
        self.cfl = cfl
        self.t = 0.0
        self.U = U0
        
        # Setup adaptive time-stepping if configured
        self.adaptive_enabled = False
        self.adaptive_stepper = None
        
        if adaptive_config and adaptive_config.get("enabled", False):
            adaptive_method = adaptive_config.get("scheme", "rk45")
            rtol = adaptive_config.get("rtol", 1e-6)
            atol = adaptive_config.get("atol", 1e-8)
            
            # Extract controller parameters
            controller_kwargs = {
                "safety_factor": adaptive_config.get("safety_factor", 0.9),
                "min_dt_factor": adaptive_config.get("min_dt_factor", 0.1),
                "max_dt_factor": adaptive_config.get("max_dt_factor", 5.0),
                "max_rejections": adaptive_config.get("max_rejections", 10)
            }
            
            # Check if scheme is adaptive
            if adaptive_method in ["rk23", "rk45", "rk78"]:
                self.adaptive_stepper = create_adaptive_stepper(
                    adaptive_method, rtol=rtol, atol=atol, **controller_kwargs
                )
                self.adaptive_enabled = True
                self.scheme = adaptive_method
            else:
                # Fall back to non-adaptive scheme
                self.scheme = adaptive_config.get("fallback_scheme", "rk4")

        # Setup artificial viscosity if configured
        self.artificial_viscosity = None
        if artificial_viscosity_config and artificial_viscosity_config.get("enabled", False):
            from .artificial_viscosity import create_artificial_viscosity
            self.artificial_viscosity = create_artificial_viscosity(artificial_viscosity_config)
        
        # Setup gravity if configured
        self.gravity_config = gravity_config

        # Filter interval cadence (default 1; if AV enabled, can be more frequent)
        self.filter_interval: int = 1
        cfg = getattr(self.grid, "filter_params", None)
        if isinstance(cfg, dict):
            try:
                self.filter_interval = max(1, int(cfg.get("interval", 1)))
            except Exception:
                self.filter_interval = 1

    def compute_dt(self, U: Array) -> float:
        max_speed = self.equations.max_wave_speed(U)
        if max_speed <= 0.0:
            return 1e-6
        # Use true minimum node spacing if available (Legendre/nonuniform)
        dx_eff = getattr(self.grid, "dx_min", None) or self.grid.dx
        dy_eff = getattr(self.grid, "dy_min", None) or self.grid.dy
        return self.cfl * min(dx_eff, dy_eff) / max_speed

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
        if self.scheme.lower() == "rk2":
            Un = _rk2_step_2d(self.grid, self.equations, U, dt, self.artificial_viscosity, self.gravity_config)
        else:
            Un = _rk4_step_2d(self.grid, self.equations, U, dt, self.artificial_viscosity, self.gravity_config)
        if (getattr(self, "_step_counter", 0) % self.filter_interval) == 0:
            Un = _apply_physical_filters_2d(self.grid, Un)
        self._step_counter = getattr(self, "_step_counter", 0) + 1
        return Un
    
    def _adaptive_step(self, U: Array, dt: float) -> Array:
        """Perform one adaptive time step."""
        def rhs_func(U_current: Array) -> Array:
            """RHS function for adaptive stepper."""
            return _compute_rhs_2d(self.grid, self.equations, U_current, self.artificial_viscosity, self.gravity_config)
        
        # Compute solution scale for relative error
        if _TORCH_AVAILABLE and isinstance(U, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U)).item())
        else:
            solution_scale = float(np.max(np.abs(U)))
        
        result = self.adaptive_stepper.step(rhs_func, U, dt, solution_scale, self.adaptive_stepper.controller)
        
        if result.accepted:
            Un = _apply_physical_filters_2d(self.grid, result.U_new)
        else:
            Un = U
            
        return Un

    def _adaptive_run_step(self, dt: float, step_count: int, 
                          on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in adaptive run loop."""
        def rhs_func(U_current: Array) -> Array:
            return _compute_rhs_2d(self.grid, self.equations, U_current, None, self.gravity_config)
        
        if _TORCH_AVAILABLE and isinstance(self.U, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(self.U)).item())
        else:
            solution_scale = float(np.max(np.abs(self.U)))
        
        result = self.adaptive_stepper.step(rhs_func, self.U, dt, solution_scale)
        
        if result.accepted:
            self.U = _apply_physical_filters_2d(self.grid, result.U_new)
            self.t += dt
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
        
        # Call monitoring callback
        if on_step is not None:
            on_step(step_count, dt, self.U)
        
        # Return next CFL-based step size
        return min(self.compute_dt(self.U), 1e-6 if self.equations.max_wave_speed(self.U) <= 0.0 else float('inf'))

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
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        self.t = float(t0)
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

        # Initialize time step
        dt = self.compute_dt(self.U) if self.U is not None else 1e-6
        
        while self.t < t_end - 1e-12:
            # Ensure we don't overshoot the end time
            dt = min(dt, t_end - self.t)
            
            if self.adaptive_enabled and self.adaptive_stepper is not None:
                # Adaptive time-stepping
                dt = self._adaptive_run_step(dt, step_count, on_step)
            else:
                # Fixed time-stepping (original behavior)
                dt = self._fixed_run_step(dt, step_count, on_step)
            
            step_count += 1

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(
                    outdir, self.t, U=self.U, grid=self.grid, equations=self.equations
                )
                next_checkpoint += checkpoint_interval

        record()
        if on_output is not None:
            on_output(self.t, self.U)
        return history


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
                 gravity_config: Optional[Dict] = None) -> Array:
    k1 = _compute_rhs_3d(grid, eqs, U, artificial_viscosity, gravity_config)
    U1 = U + dt * 0.5 * k1
    k2 = _compute_rhs_3d(grid, eqs, U1, artificial_viscosity, gravity_config)
    return U + dt * k2


def _rk4_step_3d(grid: Grid3D, eqs: EulerEquations3D, U: Array, dt: float,
                 artificial_viscosity: Optional[SpectralArtificialViscosity] = None,
                 gravity_config: Optional[Dict] = None) -> Array:
    k1 = _compute_rhs_3d(grid, eqs, U, artificial_viscosity, gravity_config)
    U2 = U + 0.5 * dt * k1
    k2 = _compute_rhs_3d(grid, eqs, U2, artificial_viscosity, gravity_config)
    U3 = U + 0.5 * dt * k2
    k3 = _compute_rhs_3d(grid, eqs, U3, artificial_viscosity, gravity_config)
    U4 = U + dt * k3
    k4 = _compute_rhs_3d(grid, eqs, U4, artificial_viscosity, gravity_config)
    return U + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _apply_physical_filters_3d(grid: Grid3D, U: Array) -> Array:
    return grid.apply_spectral_filter(U)


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
                 gravity_config: Optional[Dict] = None):
        """Initialize the 3D spectral solver.
        
        Args:
            grid: 3D grid
            equations: 3D Euler equations
            U0: Initial solution (optional)
            scheme: Time integration scheme ("rk2", "rk4", "rk23", "rk45", "rk78")
            cfl: CFL number
            adaptive_config: Configuration for adaptive time-stepping
            gravity_config: Configuration for gravity
        """
        self.grid = grid
        self.equations = equations
        self.scheme = scheme
        self.cfl = cfl
        self.t = 0.0
        self.U = U0
        
        # Setup adaptive time-stepping if configured
        self.adaptive_enabled = False
        self.adaptive_stepper = None
        
        if adaptive_config and adaptive_config.get("enabled", False):
            adaptive_method = adaptive_config.get("scheme", "rk45")
            rtol = adaptive_config.get("rtol", 1e-6)
            atol = adaptive_config.get("atol", 1e-8)
            
            # Extract controller parameters
            controller_kwargs = {
                "safety_factor": adaptive_config.get("safety_factor", 0.9),
                "min_dt_factor": adaptive_config.get("min_dt_factor", 0.1),
                "max_dt_factor": adaptive_config.get("max_dt_factor", 5.0),
                "max_rejections": adaptive_config.get("max_rejections", 10)
            }
            
            # Check if scheme is adaptive
            if adaptive_method in ["rk23", "rk45", "rk78"]:
                self.adaptive_stepper = create_adaptive_stepper(
                    adaptive_method, rtol=rtol, atol=atol, **controller_kwargs
                )
                self.adaptive_enabled = True
                self.scheme = adaptive_method
            else:
                # Fall back to non-adaptive scheme
                self.scheme = adaptive_config.get("fallback_scheme", "rk4")

        # Setup gravity if configured
        self.gravity_config = gravity_config

    def compute_dt(self, U: Array) -> float:
        max_speed = self.equations.max_wave_speed(U)
        if max_speed <= 0.0:
            return 1e-6
        return self.cfl * min(self.grid.dx, self.grid.dy, self.grid.dz) / max_speed

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
        if self.scheme.lower() == "rk2":
            Un = _rk2_step_3d(self.grid, self.equations, U, dt, None, self.gravity_config)
        else:
            Un = _rk4_step_3d(self.grid, self.equations, U, dt, None, self.gravity_config)
        Un = _apply_physical_filters_3d(self.grid, Un)
        return Un
    
    def _adaptive_step(self, U: Array, dt: float) -> Array:
        """Perform one adaptive time step."""
        def rhs_func(U_current: Array) -> Array:
            """RHS function for adaptive stepper."""
            return _compute_rhs_3d(self.grid, self.equations, U_current, None, self.gravity_config)
        
        # Compute solution scale for relative error
        if _TORCH_AVAILABLE and isinstance(U, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(U)).item())
        else:
            solution_scale = float(np.max(np.abs(U)))
        
        result = self.adaptive_stepper.step(rhs_func, U, dt, solution_scale, self.adaptive_stepper.controller)
        
        if result.accepted:
            Un = _apply_physical_filters_3d(self.grid, result.U_new)
        else:
            Un = U
            
        return Un

    def _adaptive_run_step(self, dt: float, step_count: int, 
                          on_step: Optional[Callable[[int, float, Array], None]]) -> float:
        """Perform one step in adaptive run loop."""
        def rhs_func(U_current: Array) -> Array:
            return _compute_rhs_3d(self.grid, self.equations, U_current, None, self.gravity_config)
        
        if _TORCH_AVAILABLE and isinstance(self.U, torch.Tensor):
            solution_scale = float(torch.max(torch.abs(self.U)).item())
        else:
            solution_scale = float(np.max(np.abs(self.U)))
        
        result = self.adaptive_stepper.step(rhs_func, self.U, dt, solution_scale)
        
        if result.accepted:
            self.U = _apply_physical_filters_3d(self.grid, result.U_new)
            self.t += dt
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
        
        # Call monitoring callback
        if on_step is not None:
            on_step(step_count, dt, self.U)
        
        # Return next CFL-based step size
        return min(self.compute_dt(self.U), 1e-6 if self.equations.max_wave_speed(self.U) <= 0.0 else float('inf'))

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
    ) -> Dict[str, List[float]]:
        from .io import save_solution_snapshot

        try:
            import torch  # type: ignore

            _TORCH_AVAILABLE = True
        except Exception:
            _TORCH_AVAILABLE = False
            torch = None  # type: ignore

        if _TORCH_AVAILABLE and isinstance(U0, (torch.Tensor,)):
            self.U = U0.clone()
        else:
            self.U = U0.copy()
        self.t = float(t0)
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

        # Initialize time step
        dt = self.compute_dt(self.U) if self.U is not None else 1e-6
        
        while self.t < t_end - 1e-12:
            # Ensure we don't overshoot the end time
            dt = min(dt, t_end - self.t)
            
            if self.adaptive_enabled and self.adaptive_stepper is not None:
                # Adaptive time-stepping
                dt = self._adaptive_run_step(dt, step_count, on_step)
            else:
                # Fixed time-stepping (original behavior)
                dt = self._fixed_run_step(dt, step_count, on_step)
            
            step_count += 1

            if self.t + 1e-12 >= next_output:
                record()
                if on_output is not None:
                    on_output(self.t, self.U)
                next_output += output_interval

            if outdir and self.t + 1e-12 >= next_checkpoint:
                save_solution_snapshot(
                    outdir, self.t, U=self.U, grid=self.grid, equations=self.equations
                )
                next_checkpoint += checkpoint_interval

        record()
        if on_output is not None:
            on_output(self.t, self.U)
        return history
