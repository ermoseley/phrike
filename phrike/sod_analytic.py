from __future__ import annotations

"""Analytic solution for the Sod shock tube (1D Riemann problem).

This implements a standard solver for the piecewise self-similar solution.
The implementation is adapted from classical formulations (e.g. Toro's book).
"""

from typing import Dict, Tuple

import numpy as np


def _pressure_function(p: float, rho: float, pK: float, gamma: float) -> float:
    if p > pK:
        # Shock
        A = 2.0 / ((gamma + 1.0) * rho)
        B = (gamma - 1.0) / (gamma + 1.0) * pK
        return (p - pK) * np.sqrt(A / (p + B))
    else:
        # Rarefaction
        aK = np.sqrt(gamma * pK / rho)
        return (2.0 * aK / (gamma - 1.0)) * (np.power(p / pK, (gamma - 1.0) / (2.0 * gamma)) - 1.0)


def _solve_star_state(left: Dict[str, float], right: Dict[str, float], gamma: float) -> Tuple[float, float]:
    rhoL, uL, pL = left["rho"], left["u"], left["p"]
    rhoR, uR, pR = right["rho"], right["u"], right["p"]

    # Initial guess using PVRS Riemann solver
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)
    pPV = 0.5 * (pL + pR) - 0.125 * (uR - uL) * (rhoL + rhoR) * (aL + aR)
    p0 = max(1e-8, pPV)

    # Newton iterations to find p*
    p = p0
    for _ in range(30):
        fL = _pressure_function(p, rhoL, pL, gamma)
        fR = _pressure_function(p, rhoR, pR, gamma)
        gL = (p > pL)
        gR = (p > pR)
        if gL:
            A = 2.0 / ((gamma + 1.0) * rhoL)
            B = (gamma - 1.0) / (gamma + 1.0) * pL
            dfL = np.sqrt(A / (p + B)) * (1.0 - 0.5 * (p - pL) / (p + B))
        else:
            aL = np.sqrt(gamma * pL / rhoL)
            dfL = (aL / (gamma * p)) * np.power(p / pL, -(gamma + 1.0) / (2.0 * gamma))
        if gR:
            A = 2.0 / ((gamma + 1.0) * rhoR)
            B = (gamma - 1.0) / (gamma + 1.0) * pR
            dfR = np.sqrt(A / (p + B)) * (1.0 - 0.5 * (p - pR) / (p + B))
        else:
            aR = np.sqrt(gamma * pR / rhoR)
            dfR = (aR / (gamma * p)) * np.power(p / pR, -(gamma + 1.0) / (2.0 * gamma))

        f = fL + fR + (uR - uL)
        df = dfL + dfR
        p_new = p - f / df
        if abs(p_new - p) < 1e-10:
            p = p_new
            break
        p = max(1e-12, p_new)

    u_star = 0.5 * (uL + uR + fR - fL)
    return p, u_star


def sod_sample(x: np.ndarray, t: float, x0: float, left: Dict[str, float], right: Dict[str, float], gamma: float) -> Dict[str, np.ndarray]:
    # Shift coords
    xi = (x - x0) / max(t, 1e-12)
    rhoL, uL, pL = left["rho"], left["u"], left["p"]
    rhoR, uR, pR = right["rho"], right["u"], right["p"]
    aL = np.sqrt(gamma * pL / rhoL)
    aR = np.sqrt(gamma * pR / rhoR)

    p_star, u_star = _solve_star_state(left, right, gamma)

    # Left state
    if p_star > pL:  # left shock
        SL = uL - aL * np.sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / pL - 1.0) + 1.0)
        SML = u_star
        rho_star_L = rhoL * ((p_star / pL + (gamma - 1.0) / (gamma + 1.0)) / ((gamma - 1.0) / (gamma + 1.0) * p_star / pL + 1.0))
        pLfan = None
    else:  # left rarefaction
        SHL = uL - aL
        a_star_L = aL * (p_star / pL) ** ((gamma - 1.0) / (2.0 * gamma))
        STL = u_star - a_star_L
        SL = SHL
        SML = STL
        pLfan = (SHL, STL)

    # Right state
    if p_star > pR:  # right shock
        SR = uR + aR * np.sqrt((gamma + 1.0) / (2.0 * gamma) * (p_star / pR - 1.0) + 1.0)
        SMR = u_star
        rho_star_R = rhoR * ((p_star / pR + (gamma - 1.0) / (gamma + 1.0)) / ((gamma - 1.0) / (gamma + 1.0) * p_star / pR + 1.0))
        pRfan = None
    else:  # right rarefaction
        SHR = uR + aR
        a_star_R = aR * (p_star / pR) ** ((gamma - 1.0) / (2.0 * gamma))
        STR = u_star + a_star_R
        SR = SHR
        SMR = STR
        pRfan = (SMR, SR)

    rho = np.empty_like(x)
    u = np.empty_like(x)
    p = np.empty_like(x)

    for i, s in enumerate(xi):
        if s < u_star:  # left of contact
            if p_star > pL:  # left shock
                if s < SL:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                elif s < SML:
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
                else:
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
            else:  # left rarefaction
                SHL, STL = pLfan  # type: ignore[misc]
                if s < SHL:
                    rho[i], u[i], p[i] = rhoL, uL, pL
                elif s < STL:
                    u_loc = (2.0 / (gamma + 1.0)) * (aL + 0.5 * (gamma - 1.0) * uL + s)
                    a_loc = (2.0 / (gamma + 1.0)) * (aL + 0.5 * (gamma - 1.0) * (uL - s))
                    rho[i] = rhoL * (a_loc / aL) ** (2.0 / (gamma - 1.0))
                    u[i] = u_loc
                    p[i] = pL * (a_loc / aL) ** (2.0 * gamma / (gamma - 1.0))
                else:
                    rho_star_L = rhoL * (p_star / pL) ** (1.0 / gamma)
                    rho[i], u[i], p[i] = rho_star_L, u_star, p_star
        else:  # right of contact
            if p_star > pR:  # right shock
                if s > SR:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                elif s > SMR:
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
                else:
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star
            else:  # right rarefaction
                SMR, SRR = pRfan  # type: ignore[misc]
                if s > SRR:
                    rho[i], u[i], p[i] = rhoR, uR, pR
                elif s > SMR:
                    u_loc = (2.0 / (gamma + 1.0)) * (-aR + 0.5 * (gamma - 1.0) * uR + s)
                    a_loc = (2.0 / (gamma + 1.0)) * (-aR + 0.5 * (gamma - 1.0) * (s - uR))
                    rho[i] = rhoR * (a_loc / aR) ** (2.0 / (gamma - 1.0))
                    u[i] = u_loc
                    p[i] = pR * (a_loc / aR) ** (2.0 * gamma / (gamma - 1.0))
                else:
                    rho_star_R = rhoR * (p_star / pR) ** (1.0 / gamma)
                    rho[i], u[i], p[i] = rho_star_R, u_star, p_star

    return {"rho": rho, "u": u, "p": p}


