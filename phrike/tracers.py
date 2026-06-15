"""Fourier-basis tracer particles with leap-frog advection.

Tracers interpolate velocity from the spectral (Fourier) representation at their
positions and advance using leap-frog (Euler on first step). Supported only when
the grid is Fourier in all periodic directions (1D/2D) or for Grid3D (always Fourier).
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _to_numpy(a: Any) -> np.ndarray:
    """Convert array to NumPy (e.g. for saving or NumPy-only grid eval)."""
    if _TORCH_AVAILABLE and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _is_torch(u: Any) -> bool:
    """True if u is a torch tensor (for choosing tracer step path)."""
    return _TORCH_AVAILABLE and isinstance(u, torch.Tensor)


class FourierTracers1D:
    """Tracer particles for 1D Fourier (periodic) grids. Leap-frog advection."""

    def __init__(self, x0: np.ndarray, mass: float = 1.0) -> None:
        self.x = np.asarray(x0, dtype=np.float64).flatten().copy()
        self._x_prev = np.empty_like(self.x)
        self.mass = float(mass)
        self._first_step = True

    def _ensure_torch(self, device: Any, dtype: Any) -> None:
        if not _TORCH_AVAILABLE or isinstance(self.x, torch.Tensor):
            return
        self.x = torch.from_numpy(np.asarray(self.x)).to(device=device, dtype=dtype)
        self._x_prev = torch.empty_like(self.x, device=device, dtype=dtype)
        self._x_prev.copy_(self.x)

    def step(self, grid: Any, equations: Any, U: Any, dt: float) -> None:
        if getattr(grid, "_basis_name", None) != "fourier":
            raise ValueError("FourierTracers1D requires grid with Fourier basis")
        u = equations.primitive(U)[1]
        if _is_torch(u):
            device, dtype = u.device, u.dtype
            self._ensure_torch(device, dtype)
            v = grid.evaluate_fourier_at_points_1d(u.flatten(), self.x)
            if self._first_step:
                self._x_prev.copy_(self.x)
                self.x = self.x + dt * v
                self._first_step = False
            else:
                x_new = self._x_prev + 2.0 * dt * v
                self._x_prev.copy_(self.x)
                self.x = x_new
            self.x = torch.remainder(self.x, grid.Lx)
        else:
            u_np = _to_numpy(u).flatten()
            v = grid.evaluate_fourier_at_points_1d(u_np, _to_numpy(self.x))
            if self._first_step:
                self._x_prev[:] = self.x
                self.x = self.x + dt * v
                self._first_step = False
            else:
                x_new = self._x_prev + 2.0 * dt * v
                self._x_prev[:] = self.x
                self.x = x_new
            self.x = np.mod(self.x, grid.Lx)


class FourierTracers2D:
    """Tracer particles for 2D Fourier (periodic) grids. Leap-frog advection."""

    def __init__(self, x0: np.ndarray, y0: np.ndarray, mass: float = 1.0) -> None:
        self.x = np.asarray(x0, dtype=np.float64).flatten().copy()
        self.y = np.asarray(y0, dtype=np.float64).flatten().copy()
        if self.x.shape != self.y.shape:
            raise ValueError("x0 and y0 must have the same size")
        self._x_prev = np.empty_like(self.x)
        self._y_prev = np.empty_like(self.y)
        self.mass = float(mass)
        self._first_step = True

    def _ensure_torch(self, device: Any, dtype: Any) -> None:
        if not _TORCH_AVAILABLE or isinstance(self.x, torch.Tensor):
            return
        self.x = torch.from_numpy(np.asarray(self.x)).to(device=device, dtype=dtype)
        self.y = torch.from_numpy(np.asarray(self.y)).to(device=device, dtype=dtype)
        self._x_prev = torch.empty_like(self.x, device=device, dtype=dtype)
        self._y_prev = torch.empty_like(self.y, device=device, dtype=dtype)
        self._x_prev.copy_(self.x)
        self._y_prev.copy_(self.y)

    def step(self, grid: Any, equations: Any, U: Any, dt: float) -> None:
        if getattr(grid, "basis_x", None) != "fourier" or getattr(grid, "basis_y", None) != "fourier":
            raise ValueError("FourierTracers2D requires Fourier basis in both x and y")
        _, ux, uy, _ = equations.primitive(U)
        if _is_torch(ux):
            device, dtype = ux.device, ux.dtype
            self._ensure_torch(device, dtype)
            vx = grid.evaluate_fourier_at_points(ux, self.x, self.y)
            vy = grid.evaluate_fourier_at_points(uy, self.x, self.y)
            if self._first_step:
                self._x_prev.copy_(self.x)
                self._y_prev.copy_(self.y)
                self.x = self.x + dt * vx
                self.y = self.y + dt * vy
                self._first_step = False
            else:
                x_new = self._x_prev + 2.0 * dt * vx
                y_new = self._y_prev + 2.0 * dt * vy
                self._x_prev.copy_(self.x)
                self._y_prev.copy_(self.y)
                self.x = x_new
                self.y = y_new
            self.x = torch.remainder(self.x, grid.Lx)
            self.y = torch.remainder(self.y, grid.Ly)
        else:
            ux_np = _to_numpy(ux)
            uy_np = _to_numpy(uy)
            vx = grid.evaluate_fourier_at_points(ux_np, self.x, self.y)
            vy = grid.evaluate_fourier_at_points(uy_np, self.x, self.y)
            if self._first_step:
                self._x_prev[:] = self.x
                self._y_prev[:] = self.y
                self.x = self.x + dt * vx
                self.y = self.y + dt * vy
                self._first_step = False
            else:
                x_new = self._x_prev + 2.0 * dt * vx
                y_new = self._y_prev + 2.0 * dt * vy
                self._x_prev[:] = self.x
                self._y_prev[:] = self.y
                self.x = x_new
                self.y = y_new
            self.x = np.mod(self.x, grid.Lx)
            self.y = np.mod(self.y, grid.Ly)


class FourierTracers3D:
    """Tracer particles for 3D Fourier (periodic) grids. Leap-frog advection."""

    def __init__(
        self, x0: np.ndarray, y0: np.ndarray, z0: np.ndarray, mass: float = 1.0
    ) -> None:
        self.x = np.asarray(x0, dtype=np.float64).flatten().copy()
        self.y = np.asarray(y0, dtype=np.float64).flatten().copy()
        self.z = np.asarray(z0, dtype=np.float64).flatten().copy()
        if self.x.shape != self.y.shape or self.y.shape != self.z.shape:
            raise ValueError("x0, y0, z0 must have the same size")
        self._x_prev = np.empty_like(self.x)
        self._y_prev = np.empty_like(self.y)
        self._z_prev = np.empty_like(self.z)
        self.mass = float(mass)
        self._first_step = True

    def _ensure_torch(self, device: Any, dtype: Any) -> None:
        if not _TORCH_AVAILABLE or isinstance(self.x, torch.Tensor):
            return
        self.x = torch.from_numpy(np.asarray(self.x)).to(device=device, dtype=dtype)
        self.y = torch.from_numpy(np.asarray(self.y)).to(device=device, dtype=dtype)
        self.z = torch.from_numpy(np.asarray(self.z)).to(device=device, dtype=dtype)
        self._x_prev = torch.empty_like(self.x, device=device, dtype=dtype)
        self._y_prev = torch.empty_like(self.y, device=device, dtype=dtype)
        self._z_prev = torch.empty_like(self.z, device=device, dtype=dtype)
        self._x_prev.copy_(self.x)
        self._y_prev.copy_(self.y)
        self._z_prev.copy_(self.z)

    def step(self, grid: Any, equations: Any, U: Any, dt: float) -> None:
        _, ux, uy, uz, _ = equations.primitive(U)
        if _is_torch(ux):
            device, dtype = ux.device, ux.dtype
            self._ensure_torch(device, dtype)
        batched = getattr(grid, "evaluate_fourier_at_points_batched_3d", None)
        if batched is not None:
            vx, vy, vz = batched(ux, uy, uz, self.x, self.y, self.z)
        else:
            vx = grid.evaluate_fourier_at_points(ux, self.x, self.y, self.z)
            vy = grid.evaluate_fourier_at_points(uy, self.x, self.y, self.z)
            vz = grid.evaluate_fourier_at_points(uz, self.x, self.y, self.z)
        if _is_torch(ux):
            if self._first_step:
                self._x_prev.copy_(self.x)
                self._y_prev.copy_(self.y)
                self._z_prev.copy_(self.z)
                self.x = self.x + dt * vx
                self.y = self.y + dt * vy
                self.z = self.z + dt * vz
                self._first_step = False
            else:
                x_new = self._x_prev + 2.0 * dt * vx
                y_new = self._y_prev + 2.0 * dt * vy
                z_new = self._z_prev + 2.0 * dt * vz
                self._x_prev.copy_(self.x)
                self._y_prev.copy_(self.y)
                self._z_prev.copy_(self.z)
                self.x = x_new
                self.y = y_new
                self.z = z_new
            self.x = torch.remainder(self.x, grid.Lx)
            self.y = torch.remainder(self.y, grid.Ly)
            self.z = torch.remainder(self.z, grid.Lz)
        else:
            if self._first_step:
                self._x_prev[:] = self.x
                self._y_prev[:] = self.y
                self._z_prev[:] = self.z
                self.x = self.x + dt * vx
                self.y = self.y + dt * vy
                self.z = self.z + dt * vz
                self._first_step = False
            else:
                x_new = self._x_prev + 2.0 * dt * vx
                y_new = self._y_prev + 2.0 * dt * vy
                z_new = self._z_prev + 2.0 * dt * vz
                self._x_prev[:] = self.x
                self._y_prev[:] = self.y
                self._z_prev[:] = self.z
                self.x = x_new
                self.y = y_new
                self.z = z_new
            self.x = np.mod(self.x, grid.Lx)
            self.y = np.mod(self.y, grid.Ly)
            self.z = np.mod(self.z, grid.Lz)
