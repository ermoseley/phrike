from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    def njit(*args, **kwargs):  # type: ignore
        def inner(func):
            return func
        return inner
    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)


# --- Numba-accelerated kernels (1D) ---

@njit(cache=True, fastmath=True)
def _primitive_kernel(U: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = U[0]
    mom = U[1]
    E = U[2]
    u = mom / rho
    kinetic = 0.5 * rho * u * u
    p = (gamma - 1.0) * (E - kinetic)
    a = np.sqrt(gamma * p / rho)
    return rho, u, p, a


@njit(cache=True, fastmath=True)
def _flux_kernel(U: np.ndarray, gamma: float) -> np.ndarray:
    rho, u, p, _ = _primitive_kernel(U, gamma)
    mom = rho * u
    F = np.empty_like(U)
    F[0] = mom
    F[1] = mom * u + p
    F[2] = (U[2] + p) * u
    return F


@njit(cache=True, fastmath=True)
def _max_wave_speed_kernel(U: np.ndarray, gamma: float) -> float:
    # U shape (3, N)
    rho, u, p, a = _primitive_kernel(U, gamma)
    c = np.abs(u) + a
    max_c = 0.0
    for i in range(c.shape[0]):
        if c[i] > max_c:
            max_c = c[i]
    return max_c


# --- 2D Euler kernels ---

@njit(cache=True, fastmath=True)
def _primitive2d_kernel(U: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = U[0]
    momx = U[1]
    momy = U[2]
    E = U[3]
    ux = momx / rho
    uy = momy / rho
    kinetic = 0.5 * rho * (ux * ux + uy * uy)
    p = (gamma - 1.0) * (E - kinetic)
    a = np.sqrt(gamma * p / rho)
    return rho, ux, uy, p


@njit(cache=True, fastmath=True)
def _flux2d_kernel(U: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
    rho, ux, uy, p = _primitive2d_kernel(U, gamma)
    momx = rho * ux
    momy = rho * uy
    Fx = np.empty((4, U.shape[-2], U.shape[-1]), dtype=U.dtype)
    Fy = np.empty_like(Fx)
    Fx[0] = momx
    Fx[1] = momx * ux + p
    Fx[2] = momx * uy
    Fx[3] = (U[3] + p) * ux
    Fy[0] = momy
    Fy[1] = momy * ux
    Fy[2] = momy * uy + p
    Fy[3] = (U[3] + p) * uy
    return Fx, Fy


@njit(cache=True, fastmath=True)
def _max_wave_speed2d_kernel(U: np.ndarray, gamma: float) -> float:
    rho, ux, uy, p = _primitive2d_kernel(U, gamma)
    a = np.sqrt(gamma * p / rho)
    c = np.abs(ux) + a
    max_c = 0.0
    for j in range(c.shape[0]):
        for i in range(c.shape[1]):
            if c[j, i] > max_c:
                max_c = c[j, i]
    return max_c


@njit(cache=True, fastmath=True)
def _primitive3d_kernel(U: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = U[0]
    momx = U[1]
    momy = U[2]
    momz = U[3]
    E = U[4]
    ux = momx / rho
    uy = momy / rho
    uz = momz / rho
    kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
    p = (gamma - 1.0) * (E - kinetic)
    return rho, ux, uy, uz, p


@njit(cache=True, fastmath=True)
def _flux3d_kernel(U: np.ndarray, gamma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rho, ux, uy, uz, p = _primitive3d_kernel(U, gamma)
    momx = rho * ux
    momy = rho * uy
    momz = rho * uz
    Fx = np.empty((5, U.shape[-3], U.shape[-2], U.shape[-1]), dtype=U.dtype)
    Fy = np.empty_like(Fx)
    Fz = np.empty_like(Fx)
    # x-fluxes
    Fx[0] = momx
    Fx[1] = momx * ux + p
    Fx[2] = momx * uy
    Fx[3] = momx * uz
    Fx[4] = (U[4] + p) * ux
    # y-fluxes
    Fy[0] = momy
    Fy[1] = momy * ux
    Fy[2] = momy * uy + p
    Fy[3] = momy * uz
    Fy[4] = (U[4] + p) * uy
    # z-fluxes
    Fz[0] = momz
    Fz[1] = momz * ux
    Fz[2] = momz * uy
    Fz[3] = momz * uz + p
    Fz[4] = (U[4] + p) * uz
    return Fx, Fy, Fz


@njit(cache=True, fastmath=True)
def _max_wave_speed3d_kernel(U: np.ndarray, gamma: float) -> float:
    rho, ux, uy, uz, p = _primitive3d_kernel(U, gamma)
    a = np.sqrt(gamma * p / rho)
    # conservative bound using max component speed
    comp = np.maximum(np.maximum(np.abs(ux), np.abs(uy)), np.abs(uz)) + a
    max_c = 0.0
    for k in range(comp.shape[0]):
        for j in range(comp.shape[1]):
            for i in range(comp.shape[2]):
                if comp[k, j, i] > max_c:
                    max_c = comp[k, j, i]
    return max_c


Array = np.ndarray


@dataclass
class EulerEquations1D:
    """Compressible Euler equations in 1D, conservative form.

    State vector U = [rho, mom, E], where:
      - rho: density
      - mom: momentum density (rho * u)
      - E: total energy density = p/(gamma-1) + 0.5*rho*u^2
    """

    gamma: float = 1.4

    def primitive(self, U: Array) -> Tuple[Array, Array, Array, Array]:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            rho = U[0]
            mom = U[1]
            E = U[2]
            u = mom / rho
            kinetic = 0.5 * rho * u * u
            p = (self.gamma - 1.0) * (E - kinetic)
            a = torch.sqrt(self.gamma * p / rho)
            return rho, u, p, a
        return _primitive_kernel(U, self.gamma)

    def conservative(self, rho: Array, u: Array, p: Array) -> Array:
        mom = rho * u
        if _TORCH_AVAILABLE and isinstance(rho, (torch.Tensor,)):
            E = p / (self.gamma - 1.0) + 0.5 * rho * u * u
            return torch.stack([rho, mom, E], dim=0)
        else:
            E = p / (self.gamma - 1.0) + 0.5 * rho * u * u
            return np.array([rho, mom, E])

    def flux(self, U: Array) -> Array:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            rho, u, p, _ = self.primitive(U)
            mom = rho * u
            F1 = mom
            F2 = mom * u + p
            F3 = (U[2] + p) * u
            return torch.stack([F1, F2, F3], dim=0)
        return _flux_kernel(U, self.gamma)

    def max_wave_speed(self, U: Array) -> float:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            _, u, _, a = self.primitive(U)
            return float(torch.max(torch.abs(u) + a).item())
        return float(_max_wave_speed_kernel(U, self.gamma))

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        rho, u, p, _ = self.primitive(U)
        if _TORCH_AVAILABLE and isinstance(rho, (torch.Tensor,)):
            total_mass = float(torch.sum(rho).item())
            total_momentum = float(torch.sum(rho * u).item())
            total_energy = float(torch.sum(p / (self.gamma - 1.0) + 0.5 * rho * u * u).item())
        else:
            total_mass = float(rho.sum())
            total_momentum = float((rho * u).sum())
            total_energy = float((p / (self.gamma - 1.0) + 0.5 * rho * u * u).sum())
        return {
            "mass": total_mass,
            "momentum": total_momentum,
            "energy": total_energy,
        }


@dataclass
class EulerEquations2D:
    gamma: float = 1.4

    def primitive(self, U: Array) -> Tuple[Array, Array, Array, Array]:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            rho = U[0]
            momx = U[1]
            momy = U[2]
            E = U[3]
            ux = momx / rho
            uy = momy / rho
            kinetic = 0.5 * rho * (ux * ux + uy * uy)
            p = (self.gamma - 1.0) * (E - kinetic)
            return rho, ux, uy, p
        return _primitive2d_kernel(U, self.gamma)

    def conservative(self, rho: Array, ux: Array, uy: Array, p: Array) -> Array:
        momx = rho * ux
        momy = rho * uy
        E = p / (self.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy)
        if _TORCH_AVAILABLE and isinstance(rho, (torch.Tensor,)):
            return torch.stack([rho, momx, momy, E], dim=0)
        return np.array([rho, momx, momy, E])

    def flux(self, U: Array) -> Tuple[Array, Array]:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            rho, ux, uy, p = self.primitive(U)
            momx = rho * ux
            momy = rho * uy
            Fx0 = momx
            Fx1 = momx * ux + p
            Fx2 = momx * uy
            Fx3 = (U[3] + p) * ux
            Fy0 = momy
            Fy1 = momy * ux
            Fy2 = momy * uy + p
            Fy3 = (U[3] + p) * uy
            Fx = torch.stack([Fx0, Fx1, Fx2, Fx3], dim=0)
            Fy = torch.stack([Fy0, Fy1, Fy2, Fy3], dim=0)
            return Fx, Fy
        return _flux2d_kernel(U, self.gamma)

    def max_wave_speed(self, U: Array) -> float:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            rho, ux, uy, p = self.primitive(U)
            a = torch.sqrt(self.gamma * p / rho)
            return float(torch.max(torch.abs(ux) + a).item())
        return float(_max_wave_speed2d_kernel(U, self.gamma))

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        rho, u, p, _ = self.primitive(U)
        total_mass = float(rho.sum())
        total_momentum = float((rho * u).sum())
        total_energy = float((p / (self.gamma - 1.0) + 0.5 * rho * u * u).sum())
        return {
            "mass": total_mass,
            "momentum": total_momentum,
            "energy": total_energy,
        }


@dataclass
class EulerEquations3D:
    gamma: float = 1.4

    def primitive(self, U: Array) -> Tuple[Array, Array, Array, Array, Array]:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            rho = U[0]
            momx = U[1]
            momy = U[2]
            momz = U[3]
            E = U[4]
            ux = momx / rho
            uy = momy / rho
            uz = momz / rho
            kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            p = (self.gamma - 1.0) * (E - kinetic)
            return rho, ux, uy, uz, p
        return _primitive3d_kernel(U, self.gamma)

    def conservative(self, rho: Array, ux: Array, uy: Array, uz: Array, p: Array) -> Array:
        momx = rho * ux
        momy = rho * uy
        momz = rho * uz
        E = p / (self.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
        if _TORCH_AVAILABLE and isinstance(rho, (torch.Tensor,)):
            return torch.stack([rho, momx, momy, momz, E], dim=0)
        return np.array([rho, momx, momy, momz, E])

    def flux(self, U: Array) -> Tuple[Array, Array, Array]:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            rho, ux, uy, uz, p = self.primitive(U)
            momx = rho * ux
            momy = rho * uy
            momz = rho * uz
            Fx0 = momx
            Fx1 = momx * ux + p
            Fx2 = momx * uy
            Fx3 = momx * uz
            Fx4 = (U[4] + p) * ux
            Fy0 = momy
            Fy1 = momy * ux
            Fy2 = momy * uy + p
            Fy3 = momy * uz
            Fy4 = (U[4] + p) * uy
            Fz0 = momz
            Fz1 = momz * ux
            Fz2 = momz * uy
            Fz3 = momz * uz + p
            Fz4 = (U[4] + p) * uz
            Fx = torch.stack([Fx0, Fx1, Fx2, Fx3, Fx4], dim=0)
            Fy = torch.stack([Fy0, Fy1, Fy2, Fy3, Fy4], dim=0)
            Fz = torch.stack([Fz0, Fz1, Fz2, Fz3, Fz4], dim=0)
            return Fx, Fy, Fz
        return _flux3d_kernel(U, self.gamma)

    def max_wave_speed(self, U: Array) -> float:
        if _TORCH_AVAILABLE and isinstance(U, (torch.Tensor,)):
            rho, ux, uy, uz, p = self.primitive(U)
            a = torch.sqrt(self.gamma * p / rho)
            comp = torch.maximum(torch.maximum(torch.abs(ux), torch.abs(uy)), torch.abs(uz)) + a
            return float(torch.max(comp).item())
        return float(_max_wave_speed3d_kernel(U, self.gamma))

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        rho, ux, uy, uz, p = self.primitive(U)
        total_mass = float(rho.sum())
        momx = float((rho * ux).sum())
        momy = float((rho * uy).sum())
        momz = float((rho * uz).sum())
        total_energy = float((p / (self.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy + uz * uz)).sum())
        return {
            "mass": total_mass,
            "momentum_x": momx,
            "momentum_y": momy,
            "momentum_z": momz,
            "energy": total_energy,
        }

