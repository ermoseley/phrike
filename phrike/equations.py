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


@njit(cache=True, fastmath=False)
def _max_wave_speed_kernel(U: np.ndarray, gamma: float) -> float:
    # U shape (3, N). fastmath stays off: it makes np.max skip NaN, so one
    # bad cell used to report the max of the rest (or 0 if every cell was NaN).
    rho, u, p, a = _primitive_kernel(U, gamma)
    c = np.abs(u) + a
    return np.max(c)


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


@njit(cache=True, fastmath=False)
def _max_wave_speed2d_kernel(U: np.ndarray, gamma: float) -> float:
    # fastmath stays off so a NaN cell is not dropped. See _max_wave_speed_kernel.
    rho, ux, uy, p = _primitive2d_kernel(U, gamma)
    a = np.sqrt(gamma * p / rho)
    c = np.maximum(np.abs(ux), np.abs(uy)) + a
    return np.max(c)


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


@njit(cache=True, fastmath=False)
def _max_wave_speed3d_kernel(U: np.ndarray, gamma: float) -> float:
    # fastmath stays off so a NaN cell is not dropped. See _max_wave_speed_kernel.
    rho, ux, uy, uz, p = _primitive3d_kernel(U, gamma)
    a = np.sqrt(gamma * p / rho)
    # conservative bound using max component speed
    comp = np.maximum(np.maximum(np.abs(ux), np.abs(uy)), np.abs(uz)) + a
    return np.max(comp)


Array = np.ndarray


def _is_torch(x: Array) -> bool:
    return _TORCH_AVAILABLE and isinstance(x, torch.Tensor)


def _grid_sum(q: Array) -> float:
    """Unweighted sum of a cell-centered field, as a Python float.

    Conserved-quantity diagnostics are these grid sums, not volume integrals.
    The equation object does not know the mesh spacing, so it cannot multiply
    by a cell volume or by SBP quadrature weights. On a uniform mesh the
    integral is this sum times the cell volume (``dx``, ``dx*dy`` or
    ``dx*dy*dz``). ``Grid2D.conserved_quantities`` already returns a
    quadrature integral on mixed Fourier/SBP grids. Relative errors
    ``(sum - sum_0) / sum_0`` are unchanged by a uniform cell-volume factor.
    """
    if _is_torch(q):
        return float(torch.sum(q).item())
    return float(np.sum(q))


def _max_wave_speed_value(speed: Array) -> float:
    """Maximum wave speed, propagating NaN and Inf like NumPy and Torch.

    A NaN anywhere in ``speed`` makes the result NaN. Inf with no NaN returns
    Inf. Callers must not replace that with a finite fallback.
    """
    if _is_torch(speed):
        return float(torch.max(speed).item())
    return float(np.max(speed))


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
        if _is_torch(U):
            _, u, _, a = self.primitive(U)
            return _max_wave_speed_value(torch.abs(u) + a)
        return float(_max_wave_speed_kernel(U, self.gamma))

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        """Unweighted grid sums of mass, momentum and energy, not integrals.

        On a uniform mesh multiply by the cell volume to integrate. Relative
        conservation errors are invariant under that factor. See ``_grid_sum``.
        """
        rho, u, p, _ = self.primitive(U)
        return {
            "mass": _grid_sum(rho),
            "momentum": _grid_sum(rho * u),
            "energy": _grid_sum(p / (self.gamma - 1.0) + 0.5 * rho * u * u),
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
        if _is_torch(U):
            rho, ux, uy, p = self.primitive(U)
            a = torch.sqrt(self.gamma * p / rho)
            speed = torch.maximum(torch.abs(ux), torch.abs(uy)) + a
            return _max_wave_speed_value(speed)
        return float(_max_wave_speed2d_kernel(U, self.gamma))

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        """Unweighted grid sums of mass, momentum and energy, not integrals.

        ``momentum`` is an alias of ``momentum_x`` for callers that still look
        up that key. Prefer ``momentum_x`` and ``momentum_y``. On a uniform
        mesh multiply by the cell volume to integrate; relative conservation
        errors are invariant under that factor. See ``_grid_sum``.
        """
        rho, ux, uy, p = self.primitive(U)
        momx = _grid_sum(rho * ux)
        momy = _grid_sum(rho * uy)
        kinetic = 0.5 * rho * (ux * ux + uy * uy)
        return {
            "mass": _grid_sum(rho),
            "momentum_x": momx,
            "momentum_y": momy,
            "momentum": momx,
            "energy": _grid_sum(p / (self.gamma - 1.0) + kinetic),
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
        if _is_torch(U):
            rho, ux, uy, uz, p = self.primitive(U)
            a = torch.sqrt(self.gamma * p / rho)
            comp = (
                torch.maximum(
                    torch.maximum(torch.abs(ux), torch.abs(uy)), torch.abs(uz)
                )
                + a
            )
            return _max_wave_speed_value(comp)
        return float(_max_wave_speed3d_kernel(U, self.gamma))

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        """Unweighted grid sums, not volume integrals. See ``_grid_sum``."""
        rho, ux, uy, uz, p = self.primitive(U)
        kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
        return {
            "mass": _grid_sum(rho),
            "momentum_x": _grid_sum(rho * ux),
            "momentum_y": _grid_sum(rho * uy),
            "momentum_z": _grid_sum(rho * uz),
            "energy": _grid_sum(p / (self.gamma - 1.0) + kinetic),
        }


# ===================== Ideal MHD (mu0 = 1) =====================
#
# State vector (8 components), chosen so the first 5 match the Euler classes:
#   U = [ rho, rho*ux, rho*uy, rho*uz, E, Bx, By, Bz ]
# with total energy and total (thermal+magnetic) pressure
#   E  = p/(gamma-1) + 0.5*rho*|u|^2 + 0.5*|B|^2
#   p* = p + 0.5*|B|^2
# Thermal pressure recovery subtracts the magnetic energy:
#   p  = (gamma-1) * ( E - 0.5*rho*|u|^2 - 0.5*|B|^2 )
#
# The magnetic-flux rows form the antisymmetric (uB - Bu) tensor, i.e. the
# induction equation  dB/dt + div(uB - Bu) = 0  <=>  dB/dt = curl(u x B).


@dataclass
class MHDEquations1D:
    """Ideal compressible MHD in 1D (variation along x only), conservative form.

    State U has shape (8, N): [rho, rho*ux, rho*uy, rho*uz, E, Bx, By, Bz].
    Only the x-flux is needed; the Bx flux row is identically zero so Bx stays
    constant (the 1D solenoidal constraint d(Bx)/dx = 0).
    """

    gamma: float = 5.0 / 3.0

    def primitive(self, U: Array):
        rho = U[0]
        ux = U[1] / rho
        uy = U[2] / rho
        uz = U[3] / rho
        E = U[4]
        Bx = U[5]
        By = U[6]
        Bz = U[7]
        kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
        magnetic = 0.5 * (Bx * Bx + By * By + Bz * Bz)
        p = (self.gamma - 1.0) * (E - kinetic - magnetic)
        return rho, ux, uy, uz, p, Bx, By, Bz

    def conservative(self, rho, ux, uy, uz, p, Bx, By, Bz) -> Array:
        momx = rho * ux
        momy = rho * uy
        momz = rho * uz
        E = (
            p / (self.gamma - 1.0)
            + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            + 0.5 * (Bx * Bx + By * By + Bz * Bz)
        )
        if _is_torch(rho):
            return torch.stack([rho, momx, momy, momz, E, Bx, By, Bz], dim=0)
        return np.array([rho, momx, momy, momz, E, Bx, By, Bz])

    def flux(self, U: Array) -> Array:
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        E = U[4]
        uB = ux * Bx + uy * By + uz * Bz
        pstar = p + 0.5 * (Bx * Bx + By * By + Bz * Bz)
        mx = rho * ux
        if _is_torch(U):
            Fx = torch.stack(
                [
                    mx,
                    mx * ux + pstar - Bx * Bx,
                    mx * uy - Bx * By,
                    mx * uz - Bx * Bz,
                    (E + pstar) * ux - Bx * uB,
                    torch.zeros_like(rho),
                    ux * By - uy * Bx,
                    ux * Bz - uz * Bx,
                ],
                dim=0,
            )
            return Fx
        Fx = np.stack(
            [
                mx,
                mx * ux + pstar - Bx * Bx,
                mx * uy - Bx * By,
                mx * uz - Bx * Bz,
                (E + pstar) * ux - Bx * uB,
                np.zeros_like(rho),
                ux * By - uy * Bx,
                ux * Bz - uz * Bx,
            ],
            axis=0,
        )
        return Fx

    def max_wave_speed(self, U: Array) -> float:
        """Max signal speed in x: |ux| + c_f,x (exact fast magnetosonic in x)."""
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        if _is_torch(U):
            a2 = self.gamma * p / rho
            b2 = (Bx * Bx + By * By + Bz * Bz) / rho
            cax2 = (Bx * Bx) / rho
            disc = torch.clamp((a2 + b2) * (a2 + b2) - 4.0 * a2 * cax2, min=0.0)
            cf = torch.sqrt(0.5 * (a2 + b2 + torch.sqrt(disc)))
            return _max_wave_speed_value(torch.abs(ux) + cf)
        a2 = self.gamma * p / rho
        b2 = (Bx * Bx + By * By + Bz * Bz) / rho
        cax2 = (Bx * Bx) / rho
        disc = np.maximum((a2 + b2) ** 2 - 4.0 * a2 * cax2, 0.0)
        cf = np.sqrt(0.5 * (a2 + b2 + np.sqrt(disc)))
        return _max_wave_speed_value(np.abs(ux) + cf)

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        """Unweighted grid sums, not volume integrals. See ``_grid_sum``."""
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        b2 = Bx * Bx + By * By + Bz * Bz
        total_energy = (
            p / (self.gamma - 1.0)
            + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            + 0.5 * b2
        )
        cross_hel = ux * Bx + uy * By + uz * Bz
        return {
            "mass": _grid_sum(rho),
            "momentum_x": _grid_sum(rho * ux),
            "momentum_y": _grid_sum(rho * uy),
            "momentum_z": _grid_sum(rho * uz),
            "energy": _grid_sum(total_energy),
            "magnetic_energy": _grid_sum(0.5 * b2),
            "cross_helicity": _grid_sum(cross_hel),
        }


@dataclass
class MHDEquations2D:
    """Ideal compressible MHD in 2D (a.k.a. 2.5D: fields depend on x,y only,
    but all three vector components are retained), conservative form.

    State U has shape (8, Ny, Nx): [rho, rho*ux, rho*uy, rho*uz, E, Bx, By, Bz].
    flux() returns (Fx, Fy) (the x- and y-flux tensors); Bz is advected via the
    in-plane induction terms. The solenoidal constraint div(B)=dxBx+dyBy=0 is
    maintained by Helmholtz projection in the solver (Grid2D.project_solenoidal).
    """

    gamma: float = 5.0 / 3.0

    def primitive(self, U: Array):
        rho = U[0]
        ux = U[1] / rho
        uy = U[2] / rho
        uz = U[3] / rho
        E = U[4]
        Bx = U[5]
        By = U[6]
        Bz = U[7]
        kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
        magnetic = 0.5 * (Bx * Bx + By * By + Bz * Bz)
        p = (self.gamma - 1.0) * (E - kinetic - magnetic)
        return rho, ux, uy, uz, p, Bx, By, Bz

    def conservative(self, rho, ux, uy, uz, p, Bx, By, Bz) -> Array:
        momx = rho * ux
        momy = rho * uy
        momz = rho * uz
        E = (
            p / (self.gamma - 1.0)
            + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            + 0.5 * (Bx * Bx + By * By + Bz * Bz)
        )
        if _is_torch(rho):
            return torch.stack([rho, momx, momy, momz, E, Bx, By, Bz], dim=0)
        return np.array([rho, momx, momy, momz, E, Bx, By, Bz])

    def flux(self, U: Array):
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        E = U[4]
        uB = ux * Bx + uy * By + uz * Bz
        pstar = p + 0.5 * (Bx * Bx + By * By + Bz * Bz)
        mx = rho * ux
        my = rho * uy
        if _is_torch(U):
            zero = torch.zeros_like(rho)
            Fx = torch.stack(
                [
                    mx,
                    mx * ux + pstar - Bx * Bx,
                    mx * uy - Bx * By,
                    mx * uz - Bx * Bz,
                    (E + pstar) * ux - Bx * uB,
                    zero,
                    ux * By - uy * Bx,
                    ux * Bz - uz * Bx,
                ],
                dim=0,
            )
            Fy = torch.stack(
                [
                    my,
                    my * ux - By * Bx,
                    my * uy + pstar - By * By,
                    my * uz - By * Bz,
                    (E + pstar) * uy - By * uB,
                    uy * Bx - ux * By,
                    zero,
                    uy * Bz - uz * By,
                ],
                dim=0,
            )
            return Fx, Fy
        zero = np.zeros_like(rho)
        Fx = np.stack(
            [
                mx,
                mx * ux + pstar - Bx * Bx,
                mx * uy - Bx * By,
                mx * uz - Bx * Bz,
                (E + pstar) * ux - Bx * uB,
                zero,
                ux * By - uy * Bx,
                ux * Bz - uz * Bx,
            ],
            axis=0,
        )
        Fy = np.stack(
            [
                my,
                my * ux - By * Bx,
                my * uy + pstar - By * By,
                my * uz - By * Bz,
                (E + pstar) * uy - By * uB,
                uy * Bx - ux * By,
                zero,
                uy * Bz - uz * By,
            ],
            axis=0,
        )
        return Fx, Fy

    def max_wave_speed(self, U: Array) -> float:
        """Conservative isotropic bound: max(|ux|,|uy|) + sqrt(a^2 + b^2)."""
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        if _is_torch(U):
            a2 = self.gamma * p / rho
            b2 = (Bx * Bx + By * By + Bz * Bz) / rho
            fast = torch.sqrt(torch.clamp(a2 + b2, min=0.0))
            comp = torch.maximum(torch.abs(ux), torch.abs(uy)) + fast
            return _max_wave_speed_value(comp)
        a2 = self.gamma * p / rho
        b2 = (Bx * Bx + By * By + Bz * Bz) / rho
        fast = np.sqrt(np.maximum(a2 + b2, 0.0))
        comp = np.maximum(np.abs(ux), np.abs(uy)) + fast
        return _max_wave_speed_value(comp)

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        """Unweighted grid sums, not volume integrals. See ``_grid_sum``."""
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        b2 = Bx * Bx + By * By + Bz * Bz
        total_energy = (
            p / (self.gamma - 1.0)
            + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            + 0.5 * b2
        )
        cross_hel = ux * Bx + uy * By + uz * Bz
        return {
            "mass": _grid_sum(rho),
            "momentum_x": _grid_sum(rho * ux),
            "momentum_y": _grid_sum(rho * uy),
            "momentum_z": _grid_sum(rho * uz),
            "energy": _grid_sum(total_energy),
            "magnetic_energy": _grid_sum(0.5 * b2),
            "cross_helicity": _grid_sum(cross_hel),
        }


@dataclass
class MHDEquations3D:
    """Ideal compressible MHD in 3D, conservative form.

    State U has shape (8, Nz, Ny, Nx): [rho, rho*ux, rho*uy, rho*uz, E, Bx, By, Bz].
    flux() returns the 8-component tensors (Fx, Fy, Fz). The solenoidal
    constraint div(B)=0 is maintained by Helmholtz projection in the solver
    (see Grid3D.project_solenoidal), not by the flux itself.
    """

    gamma: float = 5.0 / 3.0

    def primitive(self, U: Array):
        rho = U[0]
        ux = U[1] / rho
        uy = U[2] / rho
        uz = U[3] / rho
        E = U[4]
        Bx = U[5]
        By = U[6]
        Bz = U[7]
        kinetic = 0.5 * rho * (ux * ux + uy * uy + uz * uz)
        magnetic = 0.5 * (Bx * Bx + By * By + Bz * Bz)
        p = (self.gamma - 1.0) * (E - kinetic - magnetic)
        return rho, ux, uy, uz, p, Bx, By, Bz

    def conservative(self, rho, ux, uy, uz, p, Bx, By, Bz) -> Array:
        momx = rho * ux
        momy = rho * uy
        momz = rho * uz
        E = (
            p / (self.gamma - 1.0)
            + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            + 0.5 * (Bx * Bx + By * By + Bz * Bz)
        )
        if _is_torch(rho):
            return torch.stack([rho, momx, momy, momz, E, Bx, By, Bz], dim=0)
        return np.array([rho, momx, momy, momz, E, Bx, By, Bz])

    def flux(self, U: Array):
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        E = U[4]
        uB = ux * Bx + uy * By + uz * Bz
        pstar = p + 0.5 * (Bx * Bx + By * By + Bz * Bz)
        mx = rho * ux
        my = rho * uy
        mz = rho * uz
        if _is_torch(U):
            zero = torch.zeros_like(rho)
            Fx = torch.stack(
                [
                    mx,
                    mx * ux + pstar - Bx * Bx,
                    mx * uy - Bx * By,
                    mx * uz - Bx * Bz,
                    (E + pstar) * ux - Bx * uB,
                    zero,
                    ux * By - uy * Bx,
                    ux * Bz - uz * Bx,
                ],
                dim=0,
            )
            Fy = torch.stack(
                [
                    my,
                    my * ux - By * Bx,
                    my * uy + pstar - By * By,
                    my * uz - By * Bz,
                    (E + pstar) * uy - By * uB,
                    uy * Bx - ux * By,
                    zero,
                    uy * Bz - uz * By,
                ],
                dim=0,
            )
            Fz = torch.stack(
                [
                    mz,
                    mz * ux - Bz * Bx,
                    mz * uy - Bz * By,
                    mz * uz + pstar - Bz * Bz,
                    (E + pstar) * uz - Bz * uB,
                    uz * Bx - ux * Bz,
                    uz * By - uy * Bz,
                    zero,
                ],
                dim=0,
            )
            return Fx, Fy, Fz
        zero = np.zeros_like(rho)
        Fx = np.stack(
            [
                mx,
                mx * ux + pstar - Bx * Bx,
                mx * uy - Bx * By,
                mx * uz - Bx * Bz,
                (E + pstar) * ux - Bx * uB,
                zero,
                ux * By - uy * Bx,
                ux * Bz - uz * Bx,
            ],
            axis=0,
        )
        Fy = np.stack(
            [
                my,
                my * ux - By * Bx,
                my * uy + pstar - By * By,
                my * uz - By * Bz,
                (E + pstar) * uy - By * uB,
                uy * Bx - ux * By,
                zero,
                uy * Bz - uz * By,
            ],
            axis=0,
        )
        Fz = np.stack(
            [
                mz,
                mz * ux - Bz * Bx,
                mz * uy - Bz * By,
                mz * uz + pstar - Bz * Bz,
                (E + pstar) * uz - Bz * uB,
                uz * Bx - ux * Bz,
                uz * By - uy * Bz,
                zero,
            ],
            axis=0,
        )
        return Fx, Fy, Fz

    def max_wave_speed(self, U: Array) -> float:
        """Conservative isotropic bound: max(|ux|,|uy|,|uz|) + sqrt(a^2 + b^2).

        Uses c_f <= sqrt(a^2 + b^2) (a = sound speed, b = total Alfven speed).
        """
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        if _is_torch(U):
            a2 = self.gamma * p / rho
            b2 = (Bx * Bx + By * By + Bz * Bz) / rho
            fast = torch.sqrt(torch.clamp(a2 + b2, min=0.0))
            comp = (
                torch.maximum(
                    torch.maximum(torch.abs(ux), torch.abs(uy)), torch.abs(uz)
                )
                + fast
            )
            return _max_wave_speed_value(comp)
        a2 = self.gamma * p / rho
        b2 = (Bx * Bx + By * By + Bz * Bz) / rho
        fast = np.sqrt(np.maximum(a2 + b2, 0.0))
        comp = np.maximum(np.maximum(np.abs(ux), np.abs(uy)), np.abs(uz)) + fast
        return _max_wave_speed_value(comp)

    def conserved_quantities(self, U: Array) -> Dict[str, float]:
        """Unweighted grid sums, not volume integrals. See ``_grid_sum``."""
        rho, ux, uy, uz, p, Bx, By, Bz = self.primitive(U)
        b2 = Bx * Bx + By * By + Bz * Bz
        total_energy = (
            p / (self.gamma - 1.0)
            + 0.5 * rho * (ux * ux + uy * uy + uz * uz)
            + 0.5 * b2
        )
        cross_hel = ux * Bx + uy * By + uz * Bz
        return {
            "mass": _grid_sum(rho),
            "momentum_x": _grid_sum(rho * ux),
            "momentum_y": _grid_sum(rho * uy),
            "momentum_z": _grid_sum(rho * uz),
            "energy": _grid_sum(total_energy),
            "magnetic_energy": _grid_sum(0.5 * b2),
            "cross_helicity": _grid_sum(cross_hel),
        }
