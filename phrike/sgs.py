"""Active SGS energy coefficients and periodic scalar transport."""

from __future__ import annotations

import numpy as np

from .dust_moments import _divide, _grid, _minmod, diffuse


def sgs_coefficients(e_s: np.ndarray, rho_g: np.ndarray, nu: np.ndarray,
                     Delta: float, model: str = "mt25", *, omega=0.0) -> dict:
    """Evaluate frozen SGS forcing; D retains the parallel coefficient API.

    ``mt25_magnetized`` uses the infinite-inertial-range MT25 tensor.
    Other models remain isotropic, independently of omega.
    """
    e_s, rho_g, nu, Delta = np.broadcast_arrays(
        *[np.asarray(a, dtype=np.float64) for a in (e_s, rho_g, nu, Delta)])
    if np.any(e_s < 0) or np.any(rho_g <= 0) or np.any(nu < 0) or np.any(Delta <= 0):
        raise ValueError("require e_s, nu >= 0 and rho_g, Delta > 0")
    sigma = np.sqrt(2*e_s/(3*rho_g))
    tau = np.divide(Delta, sigma, out=np.full_like(sigma, np.inf), where=sigma > 0)
    eps = sigma**3/Delta
    x = np.zeros_like(sigma)
    np.multiply(nu, tau, out=x, where=nu > 0)
    omega = np.broadcast_to(np.asarray(omega, dtype=np.float64), sigma.shape)
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite")
    y = np.zeros_like(sigma)
    np.multiply(np.abs(omega), tau, out=y, where=omega != 0)
    if model in ("mt25", "mt25_magnetized"):
        D = eps*np.arctan(x)
    elif model == "white":
        D = eps.copy()
    elif model == "finite_tau":
        D = eps/(1 + x)
    elif model == "none":
        D = np.zeros_like(eps)
    else:
        raise ValueError(f"unknown SGS model: {model}")
    D_perp = D
    if model == "mt25_magnetized":
        # atan2 gives the continuous x -> 0 limits, including the half
        # resonance at y=1. Zero-energy cells have x,y=infinity but D=0.
        xx, yy = np.where(sigma > 0, x, 0.0), np.where(sigma > 0, y, 0.0)
        # Complementary angles avoid subtracting two pi/2 terms when x is
        # tiny and y<1. At exact resonance use the x -> 0+ half plateau.
        lower = np.where((xx == 0) & (yy == 1), np.pi/2, np.arctan2(xx, 1 - yy))
        D_perp = 0.5*eps*(lower + np.arctan2(xx, 1 + yy))
        D_perp = np.where(y == 0, D, D_perp)
    return dict(sigma=sigma, tau=tau, eps=eps, x=x, y=y, D=D,
                D_par=D, D_perp=D_perp)


def drain_production(K_drop: np.ndarray, dxs: tuple, Delta: float,
                     rho_eint: np.ndarray, c_ell: float = 1.0,
                     f_cap: float = 0.5) -> np.ndarray:
    """Smooth and renormalise removed kinetic energy, then cap locally."""
    K_drop = np.asarray(K_drop, dtype=np.float64)
    dxs = _grid(dxs, K_drop.ndim, 0.0)
    if c_ell < 0 or Delta <= 0 or f_cap < 0 or np.any(np.asarray(rho_eint) < 0):
        raise ValueError("production scales and available heat must be non-negative")
    k2 = np.zeros_like(K_drop)
    for d, dx in enumerate(dxs):
        shape = [1]*K_drop.ndim
        shape[-1-d] = K_drop.shape[-1-d]
        k = 2*np.pi*np.fft.fftfreq(K_drop.shape[-1-d], d=dx)
        k2 += k.reshape(shape)**2
    smooth = np.fft.ifftn(np.fft.fftn(K_drop)*np.exp(-0.5*k2*(c_ell*Delta)**2)).real
    positive = np.maximum(smooth, 0)
    target = max(float(np.sum(K_drop)), 0.0)
    positive *= _divide(target, np.sum(positive))
    return np.minimum(positive, f_cap*np.asarray(rho_eint, dtype=np.float64))


def advect_scalar(e: np.ndarray, u: np.ndarray, dt: float, dxs: tuple,
                  cfl: float = 0.8) -> tuple[np.ndarray, int]:
    """Conservative positive MUSCL advection of a scalar density."""
    e = np.array(e, dtype=np.float64, copy=True)
    u = np.asarray(u, dtype=np.float64)
    dxs = _grid(dxs, e.ndim, dt)
    if np.any(e < 0) or not np.all(np.isfinite(e)) or not 0 < cfl <= 1:
        raise ValueError("require non-negative finite density and cfl in (0, 1]")
    rate = sum(float(np.max(np.abs(u[d])))/dx for d, dx in enumerate(dxs))
    nsub = max(1, int(np.ceil(dt*rate/(0.5*cfl)))) if dt > 0 else 0
    velocities = [0.5*(u[d] + np.roll(u[d], -1, axis=-1-d))
                  for d in range(e.ndim)]

    def rhs(state: np.ndarray) -> np.ndarray:
        result = np.zeros_like(state)
        for d, dx in enumerate(dxs):
            axis = -1-d
            slope = _minmod(state - np.roll(state, 1, axis),
                             np.roll(state, -1, axis) - state)
            left = state + 0.5*slope
            right = np.roll(state - 0.5*slope, -1, axis)
            speed = velocities[d]
            flux = speed*np.where(speed >= 0, left, right)
            result -= (flux - np.roll(flux, 1, axis))/dx
        return result

    for _ in range(nsub):
        h = dt/nsub
        e1 = e + h*rhs(e)
        e = 0.5*e + 0.5*(e1 + h*rhs(e1))
    return e, nsub


def compression_factor(divu: np.ndarray, dt: float) -> np.ndarray:
    """Return the exact frozen-divergence SGS compression multiplier."""
    return np.exp(-(2/3)*np.asarray(divu, dtype=np.float64)*dt)


def diffuse_scalar(e: np.ndarray, kappa: np.ndarray, dt: float,
                   dxs: tuple) -> tuple[np.ndarray, int]:
    """Conservatively diffuse a scalar with arithmetic face diffusivities."""
    result, nsub = diffuse(np.asarray(e, dtype=np.float64)[None], kappa, dt, dxs)
    return result[0], nsub
