"""Exact local gas, dust and SGS reservoir exchanges."""

from __future__ import annotations

import numpy as np

from .dust_moments import Closure, _divide, ou_gaussian, ou_hyqmom, phi, regularize
from .sgs import sgs_coefficients


_LEDGER = ("Q_mean", "Q_rand", "L_dust", "L_decay", "Q_projection")
# Largest reservoir loss rate times substep: the midpoint-rate exponential
# is accurate to O((rate h)^3) per substep, and the SGS decay is not stiff
# in the AP sense (it is resolved by sub-cycling the local ODE instead).
_MAX_RATE_STEP = 0.25
_MAX_SUBSTEPS = 4096


def local_coupling_step(gas_U: np.ndarray, dust_U: np.ndarray, e_s: np.ndarray,
                        h: float, *, closure: Closure, gamma: float,
                        nu: np.ndarray, Delta: float, ndim: int,
                        d_model: str = "mt25", c_eps: float = 1.0,
                        sgs_decay: bool = True, backreaction: bool = True) -> tuple:
    """Apply the SGS budget, two-body drag, OU dispersion and gas heat over h.

    Drag and dispersion maps are exact for any h; the nonlinear reservoir
    decay is resolved by sub-cycling when its rate times h exceeds
    _MAX_RATE_STEP.  Round-off cone violations of the input or output are
    projected (exact maps preserve the cone) and their tiny energy is given
    to the gas, reported as Q_projection.
    """
    gas = np.array(gas_U, dtype=np.float64, copy=True)
    dust = np.array(dust_U, dtype=np.float64, copy=True)
    es = np.array(e_s, dtype=np.float64, copy=True)
    nu = np.broadcast_to(np.asarray(nu, dtype=np.float64), es.shape)
    if ndim not in (1, 2, 3) or gas.shape != (ndim + 2, *es.shape):
        raise ValueError("gas layout does not match ndim and SGS shape")
    if closure.ndim != ndim or dust.shape != (closure.nfields, *es.shape):
        raise ValueError("dust layout does not match closure and ndim")
    if h < 0 or not np.isfinite(h) or c_eps < 0:
        raise ValueError("h and c_eps must be non-negative")
    if not np.all(np.isfinite(dust)) or np.any(dust[0] < 0) or np.any(es < 0):
        raise ValueError("dust must be finite with non-negative density, and e_s >= 0")
    ledger = {key: np.zeros_like(es) for key in _LEDGER}
    if h == 0:
        return gas, dust, es, ledger
    rho_d = dust[0]
    rate = np.zeros_like(es)
    if d_model != "none":
        D = sgs_coefficients(es, gas[0], nu, Delta, model=d_model)["D"]
        rate += _divide(3*rho_d*D, es)
    if sgs_decay:
        rate += c_eps*np.sqrt(es/gas[0])/Delta
    nsub = int(min(_MAX_SUBSTEPS, max(1, np.ceil(float(np.max(rate))*h/_MAX_RATE_STEP))))
    for _ in range(nsub):
        gas, dust, es, part = _coupling_substep(
            gas, dust, es, h/nsub, closure=closure, nu=nu, Delta=Delta, ndim=ndim,
            d_model=d_model, c_eps=c_eps, sgs_decay=sgs_decay, backreaction=backreaction)
        for key in _LEDGER:
            ledger[key] += part[key]
    return gas, dust, es, ledger


def _coupling_substep(gas, dust, es, h, *, closure, nu, Delta, ndim, d_model, c_eps,
                      sgs_decay, backreaction):
    zeros = np.zeros_like(es)
    dust, reg = regularize(dust, closure)
    gas[-1] -= reg["dE"]
    Q_projection = -reg["dE"]
    p = closure.primitives(dust)
    rho_g, rho_d = gas[0], p["rho"]
    def rates(e):
        D = sgs_coefficients(e, rho_g, nu, Delta, model=d_model)["D"]
        decay = c_eps*np.sqrt(e/rho_g)/Delta if sgs_decay else zeros
        return D, _divide(3*rho_d*D, e), decay

    # Rates at the midpoint of a frozen-rate predictor make the reservoir
    # update second order; the exponential keeps e_s positive for any h.
    D, rd, re = rates(es)
    D, rd, re = rates(es*np.exp(-0.5*(rd + re)*h))
    rate = rd + re
    # The reservoir loses exactly es(1 - e^{-rate h}), split in proportion to
    # the rates; the dust receives precisely its share as D_eff.
    weight = phi(rate, h)
    L_dust, L_decay = es*rd*weight, es*re*weight
    es_new = es*np.exp(-rate*h)
    D_eff = _divide(L_dust, 3*rho_d*h)

    ug = np.zeros_like(p["u"])
    ug[:ndim] = gas[1:1+ndim]/rho_g
    ud = p["u"]
    ud_new = ud.copy()
    gas_momentum = gas[1:1+ndim].copy()
    delta = ud[:ndim] - ug[:ndim]
    if backreaction:
        ratio = rho_d/rho_g
        drag_rate = nu*(1 + ratio)
        lost_fraction = -np.expm1(-drag_rate*h)
        impulse = (rho_d/(1 + ratio))*lost_fraction*delta
        gas[1:1+ndim] += impulse
        ud_new[:ndim] -= (lost_fraction/(1 + ratio))*delta
        Q_mean = (0.5*(rho_d/(1 + ratio))*np.sum(delta**2, axis=0)
                  * (-np.expm1(-2*drag_rate*h)))
    else:
        ud_new[:ndim] -= (-np.expm1(-nu*h))*delta
        Q_mean = 0.5*rho_d*np.sum(
            (ud[:ndim] - ud_new[:ndim])*(ud[:ndim] + ud_new[:ndim]), axis=0)
    A = np.exp(-nu*h)
    ud_new[ndim:] = A*ud[ndim:]
    Q_mean += (0.5*rho_d*np.sum(ud[ndim:]**2, axis=0)*(-np.expm1(-2*nu*h)))
    # Forcing linear in time across the step (ETD2): mean D_eff (so the energy
    # the dust receives is exactly L_dust), slope from the reservoir's own decay,
    # limited so D stays non-negative.  Under stiff drag the dispersion then
    # tracks D(t_end)/nu with the physical lag, not the step-averaged forcing.
    D_end = sgs_coefficients(es_new, rho_g, nu, Delta, model=d_model)["D"]
    D_start = sgs_coefficients(es, rho_g, nu, Delta, model=d_model)["D"]
    slope = np.clip((D_end - D_start)/h, -2*D_eff/h, 2*D_eff/h)
    D_a = D_eff - 0.5*slope*h
    Sigma = 2*(D_a*phi(2*nu, h) + slope*psi(2*nu, h))
    if closure.name == "gaussian":
        C_new = ou_gaussian(p["C"], A, Sigma)
        dust_new = closure.from_primitives(rho_d, ud_new, C_new)
        random_old = 0.5*rho_d*sum(p["C"][i, i] for i in range(3))
    else:
        v, q, r, sperp2 = ou_hyqmom(p["v"], p["q"], p["r"], p["sperp2"], A, Sigma)
        dust_new = closure.from_primitives(rho_d, ud_new, v, q, r, sperp2)
        random_old = 0.5*rho_d*(p["v"] + 2*p["sperp2"])
    dust_new = np.where((rho_d == 0)[None], dust, dust_new)
    # Evaluate heat without subtracting nearly equal random energies.
    Q_rand = random_old*(-np.expm1(-2*nu*h)) + 3*rho_d*D_eff*h - 1.5*rho_d*Sigma
    kinetic_change = 0.5*np.sum((gas[1:1+ndim] - gas_momentum)
                               * (gas[1:1+ndim] + gas_momentum), axis=0)/rho_g
    gas[-1] += kinetic_change + Q_mean + Q_rand + L_decay
    dust_new, reg = regularize(dust_new, closure)
    gas[-1] -= reg["dE"]
    Q_projection = Q_projection - reg["dE"]
    return gas, dust_new, es_new, dict(Q_mean=Q_mean, Q_rand=Q_rand, L_dust=L_dust,
                                       L_decay=L_decay, Q_projection=Q_projection)


def psi(rate: np.ndarray, h: float) -> np.ndarray:
    """Evaluate int_0^h exp(-rate (h - t)) t dt = (h - phi)/rate, continuous at zero."""
    rate = np.asarray(rate, dtype=np.float64)
    z = rate*h
    small = np.abs(z) < 1e-2
    zs = np.where(small, z, 0.0)
    series = h*h*(0.5 + zs*(-1/6 + zs*(1/24 + zs*(-1/120 + zs/720))))
    return np.where(small, series, _divide(h - phi(rate, h), rate))


def epstein_nu(gas_U: np.ndarray, gamma: float, grain_rho_a: float,
               ndim: int) -> np.ndarray:
    """Return Epstein stopping frequency from the gas thermal pressure."""
    gas = np.asarray(gas_U, dtype=np.float64)
    rho = gas[0]
    pressure = (gamma - 1)*(gas[-1] - 0.5*np.sum(gas[1:1+ndim]**2, axis=0)/rho)
    return rho*np.sqrt(8*pressure/(np.pi*rho))/grain_rho_a
