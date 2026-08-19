"""Passive tracer particles for periodic Fourier grids.

Particles sample the retained, dealiased Fourier velocity representation and
use an explicit-midpoint update between the gas states bracketing each accepted step.
This needs two position evaluations per step, remains second-order accurate
when the gas timestep changes, and requires no restart history.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - Torch is optional
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _is_torch(value: Any) -> bool:
    return bool(
        _TORCH_AVAILABLE and torch is not None and isinstance(value, torch.Tensor)
    )


def _as_backend(value: Any, template: Any) -> Any:
    """Move a position array to the field backend, device, and dtype."""
    if _is_torch(template):
        assert torch is not None
        if isinstance(value, torch.Tensor):
            return value.to(dtype=template.dtype, device=template.device)
        return torch.as_tensor(value, dtype=template.dtype, device=template.device)
    if _is_torch(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.asarray(template).dtype)


def _wrap(value: Any, length: float) -> Any:
    if _is_torch(value):
        assert torch is not None
        wrapped = torch.remainder(value, length)
        return torch.where(wrapped >= length, torch.zeros_like(wrapped), wrapped)
    wrapped = np.mod(value, length)
    return np.where(wrapped >= length, np.zeros_like(wrapped), wrapped)


def _velocities(equations: Any, state: Any, ndim: int) -> Tuple[Any, ...]:
    primitive = equations.primitive(state)
    if len(primitive) < ndim + 1:
        raise ValueError(
            f"Expected at least {ndim + 1} primitive fields, got {len(primitive)}"
        )
    return tuple(primitive[1 : ndim + 1])


class FourierTracers1D:
    """Passive tracers for a periodic one-dimensional Fourier grid."""

    def __init__(self, x0: np.ndarray, mass: float = 1.0) -> None:
        self.x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
        self.x_unwrapped = self.x.copy()
        self.mass = float(mass)
        self.lattice_shape: Optional[Tuple[int, ...]] = None

    def _ensure_backend(self, template: Any) -> None:
        self.x = _as_backend(self.x, template)
        self.x_unwrapped = _as_backend(self.x_unwrapped, template)

    @staticmethod
    def _sample(grid: Any, velocity: Any, x: Any) -> Any:
        return grid.evaluate_fourier_at_points_1d(velocity.reshape(-1), x)

    def step(
        self,
        grid: Any,
        equations: Any,
        U: Any,
        dt: float,
        U_new: Optional[Any] = None,
    ) -> None:
        """Advance one accepted gas step with two-evaluation explicit midpoint."""
        if getattr(grid, "_basis_name", None) != "fourier":
            raise ValueError("FourierTracers1D requires a Fourier grid")
        if dt < 0.0 or not np.isfinite(dt):
            raise ValueError("Tracer timestep must be finite and non-negative")
        u0 = _velocities(equations, U, 1)[0]
        self._ensure_backend(u0)
        v0 = self._sample(grid, u0, self.x_unwrapped)
        midpoint = self.x_unwrapped + 0.5 * dt * v0
        end_state = U if U_new is None else U_new
        midpoint_state = 0.5 * (U + end_state)
        u_midpoint = _velocities(equations, midpoint_state, 1)[0]
        v_midpoint = self._sample(grid, u_midpoint, midpoint)
        self.x_unwrapped = self.x_unwrapped + dt * v_midpoint
        self.x = _wrap(self.x_unwrapped, grid.Lx)


class FourierTracers2D:
    """Passive tracers for a periodic two-dimensional Fourier grid."""

    def __init__(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        mass: float = 1.0,
        lattice_shape: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
        self.y = np.asarray(y0, dtype=np.float64).reshape(-1).copy()
        if self.x.shape != self.y.shape:
            raise ValueError("x0 and y0 must have the same size")
        self.x_unwrapped = self.x.copy()
        self.y_unwrapped = self.y.copy()
        self.mass = float(mass)
        self.lattice_shape = lattice_shape

    def _ensure_backend(self, template: Any) -> None:
        for name in ("x", "y", "x_unwrapped", "y_unwrapped"):
            setattr(self, name, _as_backend(getattr(self, name), template))

    @staticmethod
    def _sample(
        grid: Any, velocities: Tuple[Any, Any], x: Any, y: Any
    ) -> Tuple[Any, Any]:
        batched = getattr(grid, "evaluate_fourier_at_points_batched_2d", None)
        if batched is not None:
            return batched(*velocities, x, y)
        return tuple(
            grid.evaluate_fourier_at_points(field, x, y)
            for field in velocities
        )  # type: ignore[return-value]

    def step(
        self,
        grid: Any,
        equations: Any,
        U: Any,
        dt: float,
        U_new: Optional[Any] = None,
    ) -> None:
        """Advance one accepted gas step with two-evaluation explicit midpoint."""
        if (
            getattr(grid, "basis_x", None) != "fourier"
            or getattr(grid, "basis_y", None) != "fourier"
        ):
            raise ValueError("FourierTracers2D requires Fourier basis in x and y")
        if dt < 0.0 or not np.isfinite(dt):
            raise ValueError("Tracer timestep must be finite and non-negative")
        velocity0 = _velocities(equations, U, 2)
        ux0 = velocity0[0]
        self._ensure_backend(ux0)
        vx0, vy0 = self._sample(
            grid, velocity0, self.x_unwrapped, self.y_unwrapped
        )
        x_midpoint = self.x_unwrapped + 0.5 * dt * vx0
        y_midpoint = self.y_unwrapped + 0.5 * dt * vy0
        end_state = U if U_new is None else U_new
        midpoint_state = 0.5 * (U + end_state)
        velocity_midpoint = _velocities(equations, midpoint_state, 2)
        vx_midpoint, vy_midpoint = self._sample(
            grid, velocity_midpoint, x_midpoint, y_midpoint
        )
        self.x_unwrapped = self.x_unwrapped + dt * vx_midpoint
        self.y_unwrapped = self.y_unwrapped + dt * vy_midpoint
        self.x = _wrap(self.x_unwrapped, grid.Lx)
        self.y = _wrap(self.y_unwrapped, grid.Ly)


class FourierTracers3D:
    """Passive tracers for a periodic three-dimensional Fourier grid."""

    def __init__(
        self,
        x0: np.ndarray,
        y0: np.ndarray,
        z0: np.ndarray,
        mass: float = 1.0,
        lattice_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        self.x = np.asarray(x0, dtype=np.float64).reshape(-1).copy()
        self.y = np.asarray(y0, dtype=np.float64).reshape(-1).copy()
        self.z = np.asarray(z0, dtype=np.float64).reshape(-1).copy()
        if self.x.shape != self.y.shape or self.y.shape != self.z.shape:
            raise ValueError("x0, y0, z0 must have the same size")
        self.x_unwrapped = self.x.copy()
        self.y_unwrapped = self.y.copy()
        self.z_unwrapped = self.z.copy()
        self.mass = float(mass)
        self.lattice_shape = lattice_shape

    def _ensure_backend(self, template: Any) -> None:
        for name in (
            "x",
            "y",
            "z",
            "x_unwrapped",
            "y_unwrapped",
            "z_unwrapped",
        ):
            setattr(self, name, _as_backend(getattr(self, name), template))

    @staticmethod
    def _sample(
        grid: Any,
        velocities: Tuple[Any, Any, Any],
        x: Any,
        y: Any,
        z: Any,
    ) -> Tuple[Any, Any, Any]:
        batched = getattr(grid, "evaluate_fourier_at_points_batched_3d", None)
        if batched is not None:
            return batched(*velocities, x, y, z)
        return tuple(
            grid.evaluate_fourier_at_points(field, x, y, z)
            for field in velocities
        )  # type: ignore[return-value]

    def step(
        self,
        grid: Any,
        equations: Any,
        U: Any,
        dt: float,
        U_new: Optional[Any] = None,
    ) -> None:
        """Advance one accepted gas step with two-evaluation explicit midpoint."""
        if dt < 0.0 or not np.isfinite(dt):
            raise ValueError("Tracer timestep must be finite and non-negative")
        velocity0 = _velocities(equations, U, 3)
        self._ensure_backend(velocity0[0])
        vx0, vy0, vz0 = self._sample(
            grid,
            velocity0,
            self.x_unwrapped,
            self.y_unwrapped,
            self.z_unwrapped,
        )
        x_midpoint = self.x_unwrapped + 0.5 * dt * vx0
        y_midpoint = self.y_unwrapped + 0.5 * dt * vy0
        z_midpoint = self.z_unwrapped + 0.5 * dt * vz0
        end_state = U if U_new is None else U_new
        midpoint_state = 0.5 * (U + end_state)
        velocity_midpoint = _velocities(equations, midpoint_state, 3)
        vx_midpoint, vy_midpoint, vz_midpoint = self._sample(
            grid, velocity_midpoint, x_midpoint, y_midpoint, z_midpoint
        )
        self.x_unwrapped = self.x_unwrapped + dt * vx_midpoint
        self.y_unwrapped = self.y_unwrapped + dt * vy_midpoint
        self.z_unwrapped = self.z_unwrapped + dt * vz_midpoint
        self.x = _wrap(self.x_unwrapped, grid.Lx)
        self.y = _wrap(self.y_unwrapped, grid.Ly)
        self.z = _wrap(self.z_unwrapped, grid.Lz)
