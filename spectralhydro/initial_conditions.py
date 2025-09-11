from __future__ import annotations

from typing import Dict

import numpy as np

from .equations import EulerEquations1D, EulerEquations2D


def sod_shock_tube(
    x: np.ndarray,
    x0: float,
    left: Dict[str, float],
    right: Dict[str, float],
    gamma: float,
) -> np.ndarray:
    """Return conservative state U for the Sod shock tube initial condition.

    left/right dictionaries should include keys: rho, u, p
    """
    rho = np.where(x < x0, left["rho"], right["rho"])  # type: ignore[index]
    u = np.where(x < x0, left["u"], right["u"])  # type: ignore[index]
    p = np.where(x < x0, left["p"], right["p"])  # type: ignore[index]
    eqs = EulerEquations1D(gamma=gamma)
    return eqs.conservative(rho, u, p)


def sinusoidal_density(
    x: np.ndarray,
    rho0: float = 1.0,
    u0: float = 0.0,
    p0: float = 1.0,
    amplitude: float = 1e-3,
    k: int = 1,
    Lx: float = 1.0,
    gamma: float = 1.4,
) -> np.ndarray:
    rho = rho0 * (1.0 + amplitude * np.sin(2.0 * np.pi * k * x / Lx))
    u = u0 * np.ones_like(x)
    p = p0 * np.ones_like(x)
    eqs = EulerEquations1D(gamma=gamma)
    return eqs.conservative(rho, u, p)


def kelvin_helmholtz_2d(
    X: np.ndarray,
    Y: np.ndarray,
    rho_outer: float = 1.0,
    rho_inner: float = 1.0,
    u0: float = 1.0,
    shear_thickness: float = 0.02,
    pressure_outer: float = 1.0,
    pressure_inner: float = 1.0,
    perturb_eps: float = 0.01,
    perturb_sigma: float = 0.02,
    gamma: float = 1.4,
) -> np.ndarray:
    """2D Kelvin-Helmholtz initial condition matching RAMSES khi.py profile.

    Creates two shear layers at y=0.25 and y=0.75 with:
    - Velocity: outer regions at -0.5*u0, middle region at +0.5*u0
    - Density/pressure: can differ between outer and inner regions
    - Perturbation: Gaussian-modulated sinusoidal vy at both interfaces
    """
    # Normalize coordinates to [0,1) like RAMSES script
    Lx = X.max() - X.min()
    Ly = Y.max() - Y.min()
    yn = Y / Ly  # normalized y coordinates
    
    # Velocity profile: two tanh transitions at y=0.25 and y=0.75
    U1 = -0.5 * u0  # outer regions
    U2 = +0.5 * u0  # middle region
    a = float(shear_thickness)
    tanh_low = np.tanh((yn - 0.25) / a)
    tanh_high = np.tanh((yn - 0.75) / a)
    ux = U1 + 0.5 * (U2 - U1) * (tanh_low - tanh_high)
    
    # Density profile: same tanh structure
    rho_outer_val = float(rho_outer)
    rho_inner_val = float(rho_inner)
    rho = rho_outer_val + 0.5 * (rho_inner_val - rho_outer_val) * (tanh_low - tanh_high)
    
    # Pressure profile: same tanh structure
    p0 = float(pressure_outer)
    p_inner = float(pressure_inner)
    p = p0 + 0.5 * (p_inner - p0) * (tanh_low - tanh_high)
    
    # vy perturbation: Gaussian-modulated sinusoidal at both interfaces
    sig = float(perturb_sigma)
    gauss = (np.exp(-((yn - 0.25) ** 2) / (2.0 * sig * sig)) + 
             np.exp(-((yn - 0.75) ** 2) / (2.0 * sig * sig)))
    sinus = np.sin(2.0 * np.pi * X / Lx)
    uy = perturb_eps * sinus * gauss
    
    eqs = EulerEquations2D(gamma=gamma)
    return eqs.conservative(rho, ux, uy, p)

