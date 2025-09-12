from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple, Optional

import numpy as np

from .equations import EulerEquations1D


WaveDir = Literal["right", "left"]


@dataclass
class AcousticParams:
    rho0: float = 1.0
    p0: float = 1.0
    u0: float = 0.0
    amplitude: float = 1e-6  # fractional perturbation of rho unless absolute=True
    mode_m: int = 1
    Lx: float = 1.0
    direction: WaveDir = "right"
    fractional: bool = True  # if True, amplitude is fraction of rho0
    gamma: float = 1.4


def _fields_from_phase(
    phase: np.ndarray, params: AcousticParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.sqrt(params.gamma * params.p0 / params.rho0)

    if params.fractional:
        drho_amp = params.amplitude * params.rho0
    else:
        drho_amp = params.amplitude

    s = np.sin(phase)
    rho = params.rho0 + drho_amp * s
    u = params.u0 + c * (drho_amp / params.rho0) * s
    p = params.p0 + (c * c) * (drho_amp) * s
    return rho, u, p


def acoustic_ic(x: np.ndarray, params: AcousticParams) -> np.ndarray:
    k = 2.0 * np.pi * params.mode_m / params.Lx
    phase = k * x
    rho, u, p = _fields_from_phase(phase, params)
    eqs = EulerEquations1D(gamma=params.gamma)
    return eqs.conservative(rho, u, p)


def acoustic_exact(
    x: np.ndarray, t: float, params: AcousticParams
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = np.sqrt(params.gamma * params.p0 / params.rho0)
    k = 2.0 * np.pi * params.mode_m / params.Lx
    shift = -c * t if params.direction == "right" else c * t
    phase = k * (x + shift)
    return _fields_from_phase(phase, params)


def two_gaussian_acoustic_ic(
    x: np.ndarray,
    params: AcousticParams,
    center_left: float,
    center_right: float,
    sigma: float,
) -> np.ndarray:
    """Sum of two Gaussian perturbations that propagate oppositely in linear acoustics.

    We perturb density (and thus u and p consistently) using the linear relations.
    The left packet will propagate right; the right packet left, producing a collision.
    """
    c = np.sqrt(params.gamma * params.p0 / params.rho0)
    if params.fractional:
        drho_amp = params.amplitude * params.rho0
    else:
        drho_amp = params.amplitude

    def gaussian(xc: float) -> np.ndarray:
        dx = np.minimum(np.abs(x - xc), params.Lx - np.abs(x - xc))  # periodic wrap
        return np.exp(-0.5 * (dx / sigma) ** 2)

    gL = gaussian(center_left)
    gR = gaussian(center_right)
    drho = drho_amp * (gL + gR)
    # For counter-propagating, set velocity as sum of right-going and left-going relations
    u = params.u0 + c * (drho_amp / params.rho0) * (gL - gR)
    rho = params.rho0 + drho
    p = params.p0 + (c * c) * drho

    eqs = EulerEquations1D(gamma=params.gamma)
    return eqs.conservative(rho, u, p)


def two_modulated_gaussian_acoustic_ic(
    x: np.ndarray,
    params: AcousticParams,
    center_left: float,
    center_right: float,
    sigma: float,
    carrier_m: int,
    phase_left: float = 0.0,
    phase_right: float = 0.0,
    amp_left_factor: float = 1.0,
    amp_right_factor: float = 1.0,
    same_direction: Optional[WaveDir] = None,
) -> np.ndarray:
    """Two counter-propagating Gaussian envelopes with high-frequency sinusoidal carriers.

    drho(x) = A * [ g_L(x) * cos(k_c (x - x_L) + phi_L) + g_R(x) * cos(k_c (x - x_R) + phi_R) ]
    u(x)    = u0 + c * (drho / rho0) but with opposite sign for the right packet to represent left-going wave.
    p(x)    = p0 + c^2 * drho
    """
    c = np.sqrt(params.gamma * params.p0 / params.rho0)
    k_c = 2.0 * np.pi * carrier_m / params.Lx
    if params.fractional:
        amp = params.amplitude * params.rho0
    else:
        amp = params.amplitude

    def wrap_dist(xc: float) -> np.ndarray:
        dx = np.minimum(np.abs(x - xc), params.Lx - np.abs(x - xc))
        return dx

    gL = np.exp(-0.5 * (wrap_dist(center_left) / sigma) ** 2)
    gR = np.exp(-0.5 * (wrap_dist(center_right) / sigma) ** 2)
    carrierL = np.cos(k_c * (x - center_left) + phase_left)
    carrierR = np.cos(k_c * (x - center_right) + phase_right)

    drho = amp * (amp_left_factor * gL * carrierL + amp_right_factor * gR * carrierR)
    if same_direction == "right":
        u = params.u0 + c * (drho / params.rho0)
    elif same_direction == "left":
        u = params.u0 - c * (drho / params.rho0)
    else:
        # Counter-propagating default
        u = params.u0 + c * (amp / params.rho0) * (
            amp_left_factor * gL * carrierL - amp_right_factor * gR * carrierR
        )
    rho = params.rho0 + drho
    p = params.p0 + (c * c) * drho

    eqs = EulerEquations1D(gamma=params.gamma)
    return eqs.conservative(rho, u, p)
