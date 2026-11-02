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
                        d_model: str | None = None, c_eps: float = 1.0,
                        sgs_decay: bool = True, backreaction: bool = True,
                        charge_to_mass: float = 0.0) -> tuple:
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
    mhd = gas.shape == (8, *es.shape)
    if ndim not in (1, 2, 3) or (not mhd and gas.shape != (ndim + 2, *es.shape)):
        raise ValueError("gas layout does not match ndim and SGS shape")
    if closure.ndim != ndim or dust.shape != (closure.nfields, *es.shape):
        raise ValueError("dust layout does not match closure and ndim")
    if h < 0 or not np.isfinite(h) or c_eps < 0:
        raise ValueError("h and c_eps must be non-negative")
    if not np.all(np.isfinite(dust)) or np.any(dust[0] < 0) or np.any(es < 0):
        raise ValueError("dust must be finite with non-negative density, and e_s >= 0")
    if not np.isfinite(charge_to_mass):
        raise ValueError("charge_to_mass must be finite")
    if d_model is None:
        d_model = "mt25_magnetized" if mhd and charge_to_mass != 0 else "mt25"
    omega = np.zeros_like(es)
    bhat = np.zeros((3, *es.shape))
    if mhd:
        Bmag = np.sqrt(np.sum(gas[5:8]**2, axis=0))
        bhat = _divide(gas[5:8], Bmag)
        omega = charge_to_mass*Bmag
        if closure.name == "hyqmom" and np.any(np.abs(gas[6:8]) > 1e-12*Bmag):
            raise ValueError("field-aligned HyQMOM requires B parallel to x")
    ledger = {key: np.zeros_like(es) for key in _LEDGER}
    if h == 0:
        return gas, dust, es, ledger
    rho_d = dust[0]
    rate = np.zeros_like(es)
    if d_model != "none":
        coeff = sgs_coefficients(es, gas[0], nu, Delta, model=d_model, omega=omega)
        if mhd:
            rate += _divide(rho_d*(coeff["D_par"] + 2*coeff["D_perp"]), es)
        else:
            rate += _divide(3*rho_d*coeff["D"], es)
    if sgs_decay:
        rate += c_eps*np.sqrt(es/gas[0])/Delta
    nsub = int(min(_MAX_SUBSTEPS, max(1, np.ceil(float(np.max(rate))*h/_MAX_RATE_STEP))))
    for _ in range(nsub):
        coupling = _mhd_coupling_substep if mhd else _coupling_substep
        magnetic = dict(omega=omega, bhat=bhat) if mhd else {}
        gas, dust, es, part = coupling(
            gas, dust, es, h/nsub, closure=closure, nu=nu, Delta=Delta, ndim=ndim,
            d_model=d_model, c_eps=c_eps, sgs_decay=sgs_decay, backreaction=backreaction,
            **magnetic)
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


def rotation_matrix(bhat: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Rodrigues map R with d(v)/d(theta) = v cross bhat.

    Axes are component-first: bhat (3, ...), R (3, 3, ...).
    A zero axis gives the identity (the zero-field limit).
    """
    bhat = np.asarray(bhat, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    theta = np.where(np.any(bhat != 0, axis=0), theta, 0.0)
    cosine, sine = np.cos(theta), np.sin(theta)
    R = (1 - cosine)*bhat[:, None]*bhat[None, :]
    for i in range(3):
        R[i, i] += cosine
    R[0, 1] += sine*bhat[2]
    R[1, 0] -= sine*bhat[2]
    R[0, 2] -= sine*bhat[1]
    R[2, 0] += sine*bhat[1]
    R[1, 2] += sine*bhat[0]
    R[2, 1] -= sine*bhat[0]
    return R


def _mhd_coupling_substep(gas, dust, es, h, *, closure, nu, Delta, ndim, d_model,
                          c_eps, sgs_decay, backreaction, omega, bhat):
    zeros = np.zeros_like(es)
    dust, reg = regularize(dust, closure)
    gas[4] -= reg["dE"]
    Q_projection = -reg["dE"]
    p = closure.primitives(dust)
    rho_g, rho_d = gas[0], p["rho"]

    def coefficients(e):
        return sgs_coefficients(e, rho_g, nu, Delta, model=d_model, omega=omega)

    def rates(e):
        c = coefficients(e)
        work = c["D_par"] + 2*c["D_perp"]
        decay = c_eps*np.sqrt(e/rho_g)/Delta if sgs_decay else zeros
        return c, _divide(rho_d*work, e), decay

    coeff, rd, re = rates(es)
    coeff, rd, re = rates(es*np.exp(-0.5*(rd + re)*h))
    rate = rd + re
    weight = phi(rate, h)
    L_dust, L_decay = es*rd*weight, es*re*weight
    es_new = es*np.exp(-rate*h)
    # Split the exact reservoir work in the same ratio as its midpoint
    # directional forcing. The one parallel and two perpendicular components
    # therefore receive exactly L_dust, including reservoir sub-cycling.
    work = coeff["D_par"] + 2*coeff["D_perp"]
    amplitude = _divide(L_dust, rho_d*h)
    Dpar_eff = amplitude*_divide(coeff["D_par"], work)
    Dperp_eff = amplitude*_divide(coeff["D_perp"], work)

    ug, ud = gas[1:4]/rho_g, p["u"]
    delta = ud - ug
    ratio = rho_d/rho_g if backreaction else zeros
    factor = 1 + ratio
    drag_rate = factor*nu
    lost = -np.expm1(-drag_rate*h)
    theta = factor*omega*h
    parallel = bhat*np.sum(bhat*delta, axis=0)
    perp = delta - parallel
    cross = np.cross(delta, bhat, axisa=0, axisb=0, axisc=0)
    # Evaluate Delta' - Delta without cancellation at small drag/gyro rates.
    change = (-lost*delta + np.exp(-drag_rate*h)
              * (-2*np.sin(0.5*theta)**2*perp + np.sin(theta)*cross))
    ud_new = ud + change/factor
    gas_momentum = gas[1:4].copy()
    if backreaction:
        gas[1:4] -= (rho_d/factor)*change
        Q_mean = (0.5*(rho_d/factor)*np.sum(delta**2, axis=0)
                  * (-np.expm1(-2*drag_rate*h)))
    else:
        # With prescribed gas velocity, exchange its work with gas E while
        # retaining fixed gas momentum, as in the neutral source operator.
        Q_mean = 0.5*rho_d*np.sum((ud - ud_new)*(ud + ud_new), axis=0)

    start, end = coefficients(es), coefficients(es_new)

    def forcing(key, effective):
        slope = np.clip((end[key] - start[key])/h, -2*effective/h, 2*effective/h)
        initial = effective - 0.5*slope*h
        return 2*(initial*phi(2*nu, h) + slope*psi(2*nu, h))

    Sigma_par = forcing("D_par", Dpar_eff)
    Sigma_perp = forcing("D_perp", Dperp_eff)
    A = np.exp(-nu*h)
    if closure.name == "gaussian":
        R = rotation_matrix(bhat, omega*h)
        C_new = A*A*np.einsum("ik...,kl...,jl...->ij...", R, p["C"], R)
        C_new += (Sigma_par - Sigma_perp)*bhat[:, None]*bhat[None, :]
        for i in range(3):
            C_new[i, i] += Sigma_perp
        dust_new = closure.from_primitives(rho_d, ud_new, C_new)
        random_old = 0.5*rho_d*sum(p["C"][i, i] for i in range(3))
    else:
        v, q, r, _ = ou_hyqmom(p["v"], p["q"], p["r"], p["sperp2"], A, Sigma_par)
        sperp2 = A*A*p["sperp2"] + Sigma_perp
        dust_new = closure.from_primitives(rho_d, ud_new, v, q, r, sperp2)
        random_old = 0.5*rho_d*(p["v"] + 2*p["sperp2"])
    dust_new = np.where((rho_d == 0)[None], dust, dust_new)
    Q_rand = (random_old*(-np.expm1(-2*nu*h)) + L_dust
              - 0.5*rho_d*(Sigma_par + 2*Sigma_perp))
    kinetic_change = 0.5*np.sum((gas[1:4] - gas_momentum)
                               * (gas[1:4] + gas_momentum), axis=0)/rho_g
    gas[4] += kinetic_change + Q_mean + Q_rand + L_decay
    dust_new, reg = regularize(dust_new, closure)
    gas[4] -= reg["dE"]
    Q_projection -= reg["dE"]
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
    if gas.shape[0] == 8:
        pressure = (gamma - 1)*(gas[4] - 0.5*np.sum(gas[1:4]**2, axis=0)/rho
                                - 0.5*np.sum(gas[5:8]**2, axis=0))
    else:
        pressure = (gamma - 1)*(gas[-1] - 0.5*np.sum(gas[1:1+ndim]**2, axis=0)/rho)
    return rho*np.sqrt(8*pressure/(np.pi*rho))/grain_rho_a
