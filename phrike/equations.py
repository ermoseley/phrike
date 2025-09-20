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
def _primitive_kernel(
    U: np.ndarray, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert conservative variables to primitive variables for 1D Euler equations.

    Args:
        U: Conservative variables array of shape (3, N) where:
           U[0] = rho (density)
           U[1] = rho * u (momentum density)
           U[2] = E (total energy density)
        gamma: Adiabatic index

    Returns:
        Tuple of (rho, u, p, a) where:
        - rho: density array
        - u: velocity array
        - p: pressure array
        - a: sound speed array
    """
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
    """Compute flux vector for 1D Euler equations.

    Args:
        U: Conservative variables array of shape (3, N)
        gamma: Adiabatic index

    Returns:
        Flux array of shape (3, N) where:
        - F[0] = rho * u (mass flux)
        - F[1] = rho * u^2 + p (momentum flux)
        - F[2] = (E + p) * u (energy flux)
    """
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
def _primitive2d_kernel(
    U: np.ndarray, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rho = U[0]
    momx = U[1]
    momy = U[2]
    E = U[3]
    ux = momx / rho
    uy = momy / rho
    kinetic = 0.5 * rho * (ux * ux + uy * uy)
    p = (gamma - 1.0) * (E - kinetic)
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
def _primitive3d_kernel(
    U: np.ndarray, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
def _flux3d_kernel(
    U: np.ndarray, gamma: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    This class implements the 1D compressible Euler equations:

    .. math::
        \\frac{\\partial \\rho}{\\partial t} + \\frac{\\partial (\\rho u)}{\\partial x} = 0
        \\frac{\\partial (\\rho u)}{\\partial t} + \\frac{\\partial (\\rho u^2 + p)}{\\partial x} = 0
        \\frac{\\partial E}{\\partial t} + \\frac{\\partial ((E + p)u)}{\\partial x} = 0

    where :math:`E = \\frac{p}{\\gamma-1} + \\frac{1}{2}\\rho u^2` is the total energy density.

    State vector U = [rho, mom, E], where:
      - rho: density
      - mom: momentum density (rho * u)
      - E: total energy density = p/(gamma-1) + 0.5*rho*u^2

    Args:
        gamma: Adiabatic index (default: 1.4)
    """

    gamma: float = 1.4

    def primitive(self, U: Array) -> Tuple[Array, Array, Array, Array]:
        """Convert conservative variables to primitive variables.

        Args:
            U: Conservative variables array of shape (3, N)

        Returns:
            Tuple of (rho, u, p, a) where:
            - rho: density array
            - u: velocity array
            - p: pressure array
            - a: sound speed array
        """
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
        """Convert primitive variables to conservative variables.

        Args:
            rho: Density array
            u: Velocity array
            p: Pressure array

        Returns:
            Conservative variables array of shape (3, N) where:
            - U[0] = rho (density)
            - U[1] = rho * u (momentum density)
            - U[2] = E (total energy density)
        """
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
            total_energy = float(
                torch.sum(p / (self.gamma - 1.0) + 0.5 * rho * u * u).item()
            )
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


# -------------------- 2D split-form and boundary helper utilities --------------------

def _ensure_numpy(a: Array) -> np.ndarray:
    """Convert input to numpy array without copying if already numpy."""
    if _TORCH_AVAILABLE and isinstance(a, (torch.Tensor,)):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def build_wall_state_y(U_row: Array, gamma: float) -> Array:
    """Build bottom-wall boundary state for a y-normal face (v=0, recompute E).

    Args:
        U_row: Conservative variables on the interior boundary row, shape (4, Nx)
        gamma: Adiabatic index
    Returns:
        U_bc: Wall state with v=0 and consistent energy, shape (4, Nx)
    """
    U = _ensure_numpy(U_row)
    rho = U[0]
    momx = U[1]
    ux = momx / rho
    # Estimate pressure from interior conservative vars
    E = U[3]
    p = (gamma - 1.0) * (E - 0.5 * rho * ux * ux)
    uy = np.zeros_like(ux)
    momy = rho * uy
    Eb = p / (gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy)
    return np.stack([rho, rho * ux, momy, Eb], axis=0)


def flux_x_from_U(U: Array, gamma: float) -> Array:
    """Return physical x-flux Fx(U) for 2D Euler. U shape (4, Ny, Nx) or (4, N)."""
    U_np = _ensure_numpy(U)
    rho = U_np[0]
    momx = U_np[1]
    momy = U_np[2]
    E = U_np[3]
    ux = momx / rho
    uy = momy / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * (ux * ux + uy * uy))
    Fx0 = momx
    Fx1 = momx * ux + p
    Fx2 = momx * uy
    Fx3 = (E + p) * ux
    return np.stack([Fx0, Fx1, Fx2, Fx3], axis=0)


def flux_y_from_U(U: Array, gamma: float) -> Array:
    """Return physical y-flux Fy(U) for 2D Euler. U shape (4, Ny, Nx) or (4, N)."""
    U_np = _ensure_numpy(U)
    rho = U_np[0]
    momx = U_np[1]
    momy = U_np[2]
    E = U_np[3]
    ux = momx / rho
    uy = momy / rho
    p = (gamma - 1.0) * (E - 0.5 * rho * (ux * ux + uy * uy))
    Fy0 = momy
    Fy1 = momy * ux
    Fy2 = momy * uy + p
    Fy3 = (E + p) * uy
    return np.stack([Fy0, Fy1, Fy2, Fy3], axis=0)


def rusanov_flux_x(U_L: Array, U_R: Array, gamma: float) -> Array:
    """Rusanov (local Lax-Friedrichs) numerical flux for x-normal interfaces.

    Args:
        U_L, U_R: Left/right conservative states, shape (4, N)
        gamma: Adiabatic index
    Returns:
        F*: Numerical flux, shape (4, N)
    """
    UL = _ensure_numpy(U_L)
    UR = _ensure_numpy(U_R)
    FxL = flux_x_from_U(UL, gamma)
    FxR = flux_x_from_U(UR, gamma)
    rhoL = UL[0]; uxL = UL[1] / rhoL; pL = (gamma - 1.0) * (UL[3] - 0.5 * rhoL * (uxL * uxL + (UL[2] / rhoL) ** 2))
    rhoR = UR[0]; uxR = UR[1] / rhoR; pR = (gamma - 1.0) * (UR[3] - 0.5 * rhoR * (uxR * uxR + (UR[2] / rhoR) ** 2))
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    alpha = np.maximum(np.abs(uxL) + aL, np.abs(uxR) + aR)
    return 0.5 * (FxL + FxR) - 0.5 * alpha[None, :] * (UR - UL)


def rusanov_flux_y(U_D: Array, U_U: Array, gamma: float) -> Array:
    """Rusanov (local Lax-Friedrichs) numerical flux for y-normal interfaces.

    Args:
        U_D: Downwind/interior state (at j=0 row), shape (4, N)
        U_U: Upwind/boundary/exterior state, shape (4, N)
        gamma: Adiabatic index
    Returns:
        F*: Numerical flux, shape (4, N)
    """
    UD = _ensure_numpy(U_D)
    UU = _ensure_numpy(U_U)
    FyD = flux_y_from_U(UD, gamma)
    FyU = flux_y_from_U(UU, gamma)
    rhoD = UD[0]; uyD = UD[2] / rhoD; pD = (gamma - 1.0) * (UD[3] - 0.5 * rhoD * ((UD[1] / rhoD) ** 2 + uyD * uyD))
    rhoU = UU[0]; uyU = UU[2] / rhoU; pU = (gamma - 1.0) * (UU[3] - 0.5 * rhoU * ((UU[1] / rhoU) ** 2 + uyU * uyU))
    aD = np.sqrt(gamma * pD / rhoD)
    aU = np.sqrt(gamma * pU / rhoU)
    alpha = np.maximum(np.abs(uyD) + aD, np.abs(uyU) + aU)
    return 0.5 * (FyD + FyU) - 0.5 * alpha[None, :] * (UU - UD)


def skew_symmetric_flux_divergence_2d(grid, eqs: EulerEquations2D, U: Array) -> Array:
    """Compute skew-symmetric split-form flux divergence for 2D Euler.

    Args:
        grid: Grid2D providing dx1, dy1
        eqs: EulerEquations2D with gamma
        U: Conservative state, shape (4, Ny, Nx)
    Returns:
        rhs_div: Flux divergence contribution (no sources/BCs), shape (4, Ny, Nx)
    """
    U_np = _ensure_numpy(U)
    rho, ux, uy, p = eqs.primitive(U_np)
    E = U_np[3]

    # Derivatives of primitives and products
    dx = grid.dx1; dy = grid.dy1
    drx = dx(rho); dry = dy(rho)
    dux = dx(ux); duy = dy(ux)
    dvy = dy(uy); dvx = dx(uy)
    dpx = dx(p); dpy = dy(p)

    rhou = U_np[1]
    rhov = U_np[2]
    drhou_dx = dx(rhou); drhov_dy = dy(rhov)

    # Mass
    div_mass = 0.5 * (drhou_dx + ux * drx + rho * dux) + 0.5 * (drhov_dy + uy * dry + rho * dvy)

    # Momentum x: ∂x(ρu^2 + p) + ∂y(ρuv)
    rho_u2 = rho * ux * ux
    dx_rho_u2 = dx(rho_u2)
    term_x_mx = 0.5 * (dx_rho_u2 + (ux * ux) * drx + 2.0 * rho * ux * dux) + dpx
    rho_uv = rho * ux * uy
    dy_rho_uv = dy(rho_uv)
    term_y_mx = 0.5 * (dy_rho_uv + (ux * uy) * dry + rho * (uy * duy + ux * dvy))
    div_mx = term_x_mx + term_y_mx

    # Momentum y: ∂x(ρuv) + ∂y(ρv^2 + p)
    dx_rho_uv = dx(rho_uv)
    term_x_my = 0.5 * (dx_rho_uv + (ux * uy) * drx + rho * (uy * dux + ux * dvx))
    rho_v2 = rho * uy * uy
    dy_rho_v2 = dy(rho_v2)
    term_y_my = 0.5 * (dy_rho_v2 + (uy * uy) * dry + 2.0 * rho * uy * dvy) + dpy
    div_my = term_x_my + term_y_my

    # Energy: ∂x((E+p)u) + ∂y((E+p)v)
    H = E + p
    dHdx = dx(H); dHdy = dy(H)
    term_x_E = 0.5 * (dx(H * ux) + H * dux + ux * dHdx)
    term_y_E = 0.5 * (dy(H * uy) + H * dvy + uy * dHdy)
    div_E = term_x_E + term_y_E

    rhs_div = np.empty_like(U_np)
    rhs_div[0] = -div_mass
    rhs_div[1] = -div_mx
    rhs_div[2] = -div_my
    rhs_div[3] = -div_E
    return rhs_div


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

    def conservative(
        self, rho: Array, ux: Array, uy: Array, uz: Array, p: Array
    ) -> Array:
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
            comp = (
                torch.maximum(
                    torch.maximum(torch.abs(ux), torch.abs(uy)), torch.abs(uz)
                )
                + a
            )
            return float(torch.max(comp).item())
        return float(_max_wave_speed3d_kernel(U, self.gamma))

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        rho, ux, uy, uz, p = self.primitive(U)
        total_mass = float(rho.sum())
        momx = float((rho * ux).sum())
        momy = float((rho * uy).sum())
        momz = float((rho * uz).sum())
        total_energy = float(
            (p / (self.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy + uz * uz)).sum()
        )
        return {
            "mass": total_mass,
            "momentum_x": momx,
            "momentum_y": momy,
            "momentum_z": momz,
            "energy": total_energy,
        }
