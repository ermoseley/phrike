"""Adaptive time-stepping infrastructure for pseudospectral solvers.

This module provides embedded Runge-Kutta methods and adaptive step size control
for the spectral Euler solvers. It supports RK23, RK45, and RK78 methods with
automatic error estimation and step size adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _copy_state(state: Any) -> Any:
    """Copy a NumPy array or clone a Torch tensor without changing backends."""
    if _TORCH_AVAILABLE and torch is not None and isinstance(state, torch.Tensor):
        return state.clone()
    return state.copy()


@dataclass
class AdaptiveStepController:
    """Controller for adaptive time-stepping with error estimation and step size control.

    This class implements the standard adaptive time-stepping algorithm used in
    ODE solvers like scipy.integrate.solve_ivp, but optimized for spectral methods.

    Attributes:
        rtol: Relative tolerance for error control
        atol: Absolute tolerance for error control
        safety_factor: Safety factor for step size adjustment (typically 0.9)
        min_dt_factor: Minimum allowed step size reduction factor
        max_dt_factor: Maximum allowed step size increase factor
        max_rejections: Maximum number of consecutive step rejections
    """

    rtol: float = 1e-6
    atol: float = 1e-8
    safety_factor: float = 0.9
    min_dt_factor: float = 0.1
    max_dt_factor: float = 5.0
    max_rejections: int = 10

    def compute_new_dt(
        self, dt_current: float, error: float, solution_scale: float, order: int
    ) -> Tuple[float, bool]:
        """Compute new time step and whether to accept current step.

        Args:
            dt_current: Current time step
            error: Local truncation error estimate (max norm across all components)
            solution_scale: Scale of the solution (for relative error)
            order: Power of ``dt`` in the local embedded error estimate

        Returns:
            Tuple of (new_dt, accept) where:
            - new_dt: Recommended new time step
            - accept: Whether to accept current step
        """
        # Compute tolerance
        tolerance = self.atol + self.rtol * solution_scale

        # Accept/reject decision
        accept = error <= tolerance

        if error > 0:
            # Compute optimal step size using the order of the method
            dt_optimal = dt_current * (tolerance / error) ** (1.0 / order)
            dt_optimal *= self.safety_factor

            # Limit step size changes
            dt_new = dt_current * np.clip(
                dt_optimal / dt_current, self.min_dt_factor, self.max_dt_factor
            )
        else:
            # If error is zero (unlikely but possible), increase step size
            dt_new = dt_current * self.max_dt_factor

        return dt_new, accept


class EmbeddedRungeKutta:
    """Base class for embedded Runge-Kutta methods.

    This provides the common interface and utilities for implementing
    embedded RK methods like RK23, RK45, and RK78.
    """

    def __init__(self, name: str, order_high: int, order_low: int):
        """Initialize embedded RK method.

        Args:
            name: Name of the method (e.g., "rk45")
            order_high: Order of the higher-order solution
            order_low: Order of the lower-order solution (for error estimation)
        """
        self.name = name
        self.order_high = order_high
        self.order_low = order_low

    @property
    def error_estimator_order(self) -> int:
        """Power used by the controller for the embedded local-error estimate."""
        return self.order_low + 1

    def _validate_tableau(self) -> None:
        """Fail fast if a tableau is internally inconsistent."""
        stages = len(self.c)
        if self.a.shape != (stages, stages):
            raise ValueError(
                f"{self.name} has an invalid A matrix shape {self.a.shape}; "
                f"expected {(stages, stages)}"
            )
        if self.b_high.shape != (stages,) or self.b_low.shape != (stages,):
            raise ValueError(f"{self.name} has invalid embedded weight dimensions")
        if not np.allclose(np.sum(self.a, axis=1), self.c, rtol=0.0, atol=5e-14):
            raise ValueError(f"{self.name} stage rows do not sum to c")
        if not np.isclose(np.sum(self.b_high), 1.0, rtol=0.0, atol=5e-14):
            raise ValueError(f"{self.name} high-order weights do not sum to one")
        if not np.isclose(np.sum(self.b_low), 1.0, rtol=0.0, atol=5e-14):
            raise ValueError(f"{self.name} embedded weights do not sum to one")

    def step(self, rhs_func, U: Any, dt: float) -> Tuple[Any, Any, float]:
        """Perform one embedded RK step.

        Args:
            rhs_func: Function that computes dU/dt = f(U)
            U: Current solution vector
            dt: Time step

        Returns:
            Tuple of (U_high, U_low, error) where:
            - U_high: Higher-order solution
            - U_low: Lower-order solution (for error estimation)
            - error: Local truncation error estimate
        """
        stages = len(self.c)
        k = [None] * stages
        k[0] = rhs_func(U)

        for i in range(1, stages):
            U_stage = _copy_state(U)
            for j in range(i):
                coefficient = self.a[i, j]
                if coefficient != 0.0:
                    U_stage = U_stage + dt * coefficient * k[j]
            k[i] = rhs_func(U_stage)

        U_high = _copy_state(U)
        U_low = _copy_state(U)
        for i in range(stages):
            if self.b_high[i] != 0.0:
                U_high = U_high + dt * self.b_high[i] * k[i]
            if self.b_low[i] != 0.0:
                U_low = U_low + dt * self.b_low[i] * k[i]

        return U_high, U_low, self.compute_error(U_high, U_low)

    def compute_error(self, U_high: np.ndarray, U_low: np.ndarray) -> float:
        """Compute error estimate between high and low order solutions.

        Args:
            U_high: Higher-order solution
            U_low: Lower-order solution

        Returns:
            Maximum error across all components
        """
        if _TORCH_AVAILABLE and isinstance(U_high, torch.Tensor):
            error = torch.max(torch.abs(U_high - U_low)).item()
        else:
            error = float(np.max(np.abs(U_high - U_low)))
        return error


class RK23(EmbeddedRungeKutta):
    """Bogacki-Shampine RK23 embedded method.

    This is a 3rd-order method with 2nd-order embedded solution.
    Good for problems that don't require very high precision.
    """

    def __init__(self):
        super().__init__("rk23", 3, 2)

        # Butcher tableau for Bogacki-Shampine RK23
        # c = [0, 1/2, 3/4, 1]
        # a = [[0, 0, 0, 0],
        #      [1/2, 0, 0, 0],
        #      [0, 3/4, 0, 0],
        #      [2/9, 1/3, 4/9, 0]]
        # b_high = [2/9, 1/3, 4/9, 0]  # 3rd order
        # b_low = [7/24, 1/4, 1/3, 1/8]  # 2nd order

        self.c = np.array([0.0, 0.5, 0.75, 1.0])
        self.a = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.0, 0.0],
                [0.0, 0.75, 0.0, 0.0],
                [2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0],
            ]
        )
        self.b_high = np.array([2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0])
        self.b_low = np.array([7.0 / 24.0, 1.0 / 4.0, 1.0 / 3.0, 1.0 / 8.0])
        self._validate_tableau()


class RK45(EmbeddedRungeKutta):
    """Dormand-Prince RK45 embedded method.

    This is a 5th-order method with 4th-order embedded solution.
    The most commonly used adaptive method, good balance of accuracy and efficiency.
    """

    def __init__(self):
        super().__init__("rk45", 5, 4)

        # Butcher tableau for Dormand-Prince RK45
        self.c = np.array([0.0, 1.0 / 5.0, 3.0 / 10.0, 4.0 / 5.0, 8.0 / 9.0, 1.0, 1.0])
        self.a = np.array(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [1.0 / 5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [3.0 / 40.0, 9.0 / 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [44.0 / 45.0, -56.0 / 15.0, 32.0 / 9.0, 0.0, 0.0, 0.0, 0.0],
                [
                    19372.0 / 6561.0,
                    -25360.0 / 2187.0,
                    64448.0 / 6561.0,
                    -212.0 / 729.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    9017.0 / 3168.0,
                    -355.0 / 33.0,
                    46732.0 / 5247.0,
                    49.0 / 176.0,
                    -5103.0 / 18656.0,
                    0.0,
                    0.0,
                ],
                [
                    35.0 / 384.0,
                    0.0,
                    500.0 / 1113.0,
                    125.0 / 192.0,
                    -2187.0 / 6784.0,
                    11.0 / 84.0,
                    0.0,
                ],
            ]
        )
        # 5th order weights
        self.b_high = np.array(
            [
                35.0 / 384.0,
                0.0,
                500.0 / 1113.0,
                125.0 / 192.0,
                -2187.0 / 6784.0,
                11.0 / 84.0,
                0.0,
            ]
        )
        # 4th order weights (embedded)
        self.b_low = np.array(
            [
                5179.0 / 57600.0,
                0.0,
                7571.0 / 16695.0,
                393.0 / 640.0,
                -92097.0 / 339200.0,
                187.0 / 2100.0,
                1.0 / 40.0,
            ]
        )
        self._validate_tableau()


class RK78(EmbeddedRungeKutta):
    """Classical 13-stage Fehlberg embedded 8(7) method.

    ``U_high`` is Fehlberg's eighth-order estimate and ``U_low`` is the
    embedded seventh-order estimate.  Their difference is the standard
    ``41/840 * (k1 + k11 - k12 - k13)`` local-error estimate.

    References
    ----------
    E. Fehlberg, NASA TR R-287 (1968).
    """

    def __init__(self):
        super().__init__("rk78", 8, 7)

        self.c = np.array(
            [
                0.0,
                2.0 / 27.0,
                1.0 / 9.0,
                1.0 / 6.0,
                5.0 / 12.0,
                1.0 / 2.0,
                5.0 / 6.0,
                1.0 / 6.0,
                2.0 / 3.0,
                1.0 / 3.0,
                1.0,
                0.0,
                1.0,
            ]
        )
        self.a = np.array(
            [
                [0.0] * 13,
                [
                    2.0 / 27.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0 / 36.0,
                    1.0 / 12.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0 / 24.0,
                    0.0,
                    1.0 / 8.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    5.0 / 12.0,
                    0.0,
                    -25.0 / 16.0,
                    25.0 / 16.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0 / 20.0,
                    0.0,
                    0.0,
                    1.0 / 4.0,
                    1.0 / 5.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -25.0 / 108.0,
                    0.0,
                    0.0,
                    125.0 / 108.0,
                    -65.0 / 27.0,
                    125.0 / 54.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    31.0 / 300.0,
                    0.0,
                    0.0,
                    0.0,
                    61.0 / 225.0,
                    -2.0 / 9.0,
                    13.0 / 900.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    2.0,
                    0.0,
                    0.0,
                    -53.0 / 6.0,
                    704.0 / 45.0,
                    -107.0 / 9.0,
                    67.0 / 90.0,
                    3.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -91.0 / 108.0,
                    0.0,
                    0.0,
                    23.0 / 108.0,
                    -976.0 / 135.0,
                    311.0 / 54.0,
                    -19.0 / 60.0,
                    17.0 / 6.0,
                    -1.0 / 12.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    2383.0 / 4100.0,
                    0.0,
                    0.0,
                    -341.0 / 164.0,
                    4496.0 / 1025.0,
                    -301.0 / 82.0,
                    2133.0 / 4100.0,
                    45.0 / 82.0,
                    45.0 / 164.0,
                    18.0 / 41.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    3.0 / 205.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -6.0 / 41.0,
                    -3.0 / 205.0,
                    -3.0 / 41.0,
                    3.0 / 41.0,
                    6.0 / 41.0,
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    -1777.0 / 4100.0,
                    0.0,
                    0.0,
                    -341.0 / 164.0,
                    4496.0 / 1025.0,
                    -289.0 / 82.0,
                    2193.0 / 4100.0,
                    51.0 / 82.0,
                    33.0 / 164.0,
                    12.0 / 41.0,
                    0.0,
                    1.0,
                    0.0,
                ],
            ]
        )
        # Eighth-order estimate.
        self.b_high = np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                34.0 / 105.0,
                9.0 / 35.0,
                9.0 / 35.0,
                9.0 / 280.0,
                9.0 / 280.0,
                0.0,
                41.0 / 840.0,
                41.0 / 840.0,
            ]
        )
        # Embedded seventh-order estimate.
        self.b_low = np.array(
            [
                41.0 / 840.0,
                0.0,
                0.0,
                0.0,
                0.0,
                34.0 / 105.0,
                9.0 / 35.0,
                9.0 / 35.0,
                9.0 / 280.0,
                9.0 / 280.0,
                41.0 / 840.0,
                0.0,
                0.0,
            ]
        )
        self._validate_tableau()


def create_adaptive_method(method_name: str) -> EmbeddedRungeKutta:
    """Create an adaptive Runge-Kutta method by name.

    Args:
        method_name: Name of the method ("rk23", "rk45", "rk78")

    Returns:
        Instance of the requested embedded RK method

    Raises:
        ValueError: If method_name is not recognized
    """
    method_name = method_name.lower()

    if method_name == "rk23":
        return RK23()
    elif method_name == "rk45":
        return RK45()
    elif method_name == "rk78":
        return RK78()
    else:
        raise ValueError(
            f"Unknown adaptive method: {method_name}. "
            f"Supported methods: rk23, rk45, rk78"
        )


@dataclass
class AdaptiveStepResult:
    """Result of an adaptive time step.

    Attributes:
        U_new: New solution (if step was accepted) or current solution (if rejected)
        dt_used: Time step that was actually used
        dt_next: Recommended next time step
        accepted: Whether the step was accepted
        error: Local truncation error estimate
        rejections: Number of consecutive rejections
    """

    U_new: np.ndarray
    dt_used: float
    dt_next: float
    accepted: bool
    error: float
    rejections: int


class AdaptiveTimeStepper:
    """Main class for adaptive time-stepping integration.

    This class coordinates the embedded RK method, step controller, and
    integrates with the existing spectral solver infrastructure.
    """

    def __init__(self, method: EmbeddedRungeKutta, controller: AdaptiveStepController):
        """Initialize adaptive time stepper.

        Args:
            method: Embedded Runge-Kutta method
            controller: Step size controller
        """
        self.method = method
        self.controller = controller
        self.consecutive_rejections = 0

    def step(
        self, rhs_func, U: np.ndarray, dt: float, solution_scale: float
    ) -> AdaptiveStepResult:
        """Perform one adaptive time step.

        Args:
            rhs_func: Function that computes dU/dt = f(U)
            U: Current solution vector
            dt: Proposed time step
            solution_scale: Scale of the solution (for relative error)

        Returns:
            AdaptiveStepResult containing step outcome and recommendations
        """
        # Perform embedded RK step
        U_high, U_low, error = self.method.step(rhs_func, U, dt)

        # Check if we should accept or reject the step
        dt_next, accept = self.controller.compute_new_dt(
            dt, error, solution_scale, self.method.error_estimator_order
        )

        if accept:
            # Step accepted
            self.consecutive_rejections = 0
            return AdaptiveStepResult(
                U_new=U_high,
                dt_used=dt,
                dt_next=dt_next,
                accepted=True,
                error=error,
                rejections=0,
            )
        else:
            # Step rejected
            self.consecutive_rejections += 1

            if self.consecutive_rejections >= self.controller.max_rejections:
                # Too many rejections, accept the step anyway to avoid infinite loop
                self.consecutive_rejections = 0
                return AdaptiveStepResult(
                    U_new=U_high,
                    dt_used=dt,
                    dt_next=dt_next,
                    accepted=True,
                    error=error,
                    rejections=self.controller.max_rejections,
                )
            else:
                # Reject and try again with smaller step
                return AdaptiveStepResult(
                    U_new=U,
                    dt_used=dt,
                    dt_next=dt_next,
                    accepted=False,
                    error=error,
                    rejections=self.consecutive_rejections,
                )

    def reset_rejections(self):
        """Reset the consecutive rejection counter."""
        self.consecutive_rejections = 0


def create_adaptive_stepper(
    method_name: str, rtol: float = 1e-6, atol: float = 1e-8, **controller_kwargs
) -> AdaptiveTimeStepper:
    """Create a complete adaptive time stepper with method and controller.

    Args:
        method_name: Name of the embedded RK method ("rk23", "rk45", "rk78")
        rtol: Relative tolerance
        atol: Absolute tolerance
        **controller_kwargs: Additional arguments for AdaptiveStepController

    Returns:
        Configured AdaptiveTimeStepper instance
    """
    method = create_adaptive_method(method_name)
    controller = AdaptiveStepController(rtol=rtol, atol=atol, **controller_kwargs)
    return AdaptiveTimeStepper(method, controller)
