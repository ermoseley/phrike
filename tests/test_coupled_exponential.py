"""Independent small-matrix references for the local exponential study tools."""

import math

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm

from phrike.coupled_exponential import (
    covariance_phi, homogeneous_source, hyqmom_phi, phi, slip_phi,
)


def matrix_phi(k, A):
    n = len(A)
    if k == 0:
        return expm(A)
    augmented = np.zeros(((k + 1)*n, (k + 1)*n), dtype=A.dtype)
    augmented[:n, :n] = A
    for i in range(k):
        augmented[i*n:(i+1)*n, (i+1)*n:(i+2)*n] = np.eye(n)
    return expm(augmented)[:n, k*n:]


def cross_matrix(b):
    x, y, z = b
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0.]])


@pytest.mark.parametrize("k", range(4))
def test_phi_series_and_stiff_complex(k):
    z = np.array([0, 1e-16, -1e-8 + 2e-8j, -.9 + .1j, -1., -30.,
                  -1000., 1000j, -3 + 29j])
    ref = np.array([matrix_phi(k, np.array([[value]]))[0, 0] for value in z])
    assert_allclose(phi(k, z), ref, rtol=2e-13, atol=3e-16)
    assert phi(k, 0) == 1/math.factorial(k)
    with pytest.raises(ValueError):
        phi(4, z)


@pytest.mark.parametrize("k", range(4))
def test_rotated_vectorized_blocks(k):
    rng = np.random.default_rng(72)
    n = 7
    b = rng.normal(size=(3, n))
    b /= np.linalg.norm(b, axis=0)
    nu = np.array([0., 1e-12, 1e-3, .1, 1., 10., 1000.])
    omega = np.array([0., 1e-9, 1., 2*np.pi, 30., 1000., 100.])
    w = rng.normal(size=(3, n))
    M = rng.normal(size=(3, 3, n))
    C = np.einsum("ik...,jk...->ij...", M, M)
    alpha = 2.7
    vp = slip_phi(k, w, .7, nu, omega, b, alpha)
    cp = covariance_phi(k, C, .7, nu, omega, b)
    for i in range(n):
        A = -nu[i]*np.eye(3) - omega[i]*cross_matrix(b[:, i])
        assert_allclose(vp[:, i], matrix_phi(k, .7*alpha*A)@w[:, i],
                        atol=3e-13, rtol=3e-12)
        lifted = np.kron(A, np.eye(3)) + np.kron(np.eye(3), A)
        ref = (matrix_phi(k, .7*lifted)@C[:, :, i].ravel()).reshape(3, 3)
        assert_allclose(cp[:, :, i], ref, atol=3e-12, rtol=3e-12)


@pytest.mark.parametrize("k", range(4))
def test_hyqmom_triangular_phi_and_repeated_limits(k):
    rates = np.array([0., 1e-13, 1e-4, .2, 1., 10., 1000.])
    state = np.array([.7, .13, 2.8, .4, 1.])[:, None]*np.ones((1, len(rates)))
    result = hyqmom_phi(k, state, .4, rates, .2, .3)
    for i, nu in enumerate(rates):
        H = np.diag([-2*nu, -3*nu, -4*nu, -2*nu, 0.])
        H[0, 4], H[2, 0], H[3, 4] = .4, 2.4, .6
        assert_allclose(result[:, i], matrix_phi(k, .4*H)@state[:, i],
                        atol=5e-13, rtol=2e-12)


def reference_source(w, C, h, nu, omega, b, alpha, d0, d1, D0, D1):
    """Lift the affine ODE and its quadratic heat into independent expm systems."""
    A = -nu*np.eye(3) - omega*cross_matrix(b)
    G = np.zeros((5, 5))
    G[:3, :3], G[:3, 3], G[:3, 4], G[4, 3] = alpha*A, d0, d1, 1.
    y0 = np.r_[w, 1., 0.]
    end = expm(h*G)@y0
    integ = np.zeros((8, 8))
    integ[:5, :5] = G
    integ[5:, :3] = np.eye(3)
    iw = (expm(h*integ)@np.r_[y0, np.zeros(3)])[5:]
    lifted = np.kron(G, np.eye(5)) + np.kron(np.eye(5), G)
    heat = np.zeros((26, 26))
    heat[:25, :25] = lifted
    heat[-1, [0, 6, 12]] = 1
    integral_square = (expm(h*heat)@np.r_[np.outer(y0, y0).ravel(), 0.])[-1]
    cov = np.zeros((12, 12))
    cov[:9, :9] = np.kron(A, np.eye(3)) + np.kron(np.eye(3), A)
    cov[:9, 9], cov[:9, 10], cov[10, 9] = 2*D0.ravel(), 2*D1.ravel(), 1.
    cov[11, [0, 4, 8]] = 1
    cend = expm(h*cov)@np.r_[C.ravel(), 1., 0., 0.]
    return end[:3], iw, integral_square, cend[:9].reshape(3, 3), cend[-1]


@pytest.mark.parametrize("nu,omega", [(0., 0.), (0., np.pi), (1e-12, 1e3),
                                        (.01, .02), (.3, .8), (1., 10.), (1e3, 1e3)])
def test_affine_source_state_heat_work_against_lifted_exponential(nu, omega):
    b = np.array([2., -1., 2.])/3
    ug, ud = np.array([.2, .3, -.1]), np.array([.7, -.5, .2])
    C = np.array([[.7, .1, -.2], [.1, .4, .02], [-.2, .02, .9]])
    D0 = np.diag([.3, .4, .2])
    D1 = np.array([[-.1, .01, 0], [.01, .2, 0], [0, 0, .1]])
    d0, d1 = np.array([.4, -.2, .3]), np.array([-.1, .3, .2])
    a0, a1 = np.array([.1, -.2, .1]), np.array([.2, .3, -.1])
    rg, rd, h = 2., 3., .7
    alpha = 1 + rd/rg
    out = homogeneous_source(ug, ud, h, rho_g=rg, rho_d=rd, nu=nu, omega=omega,
                             bhat=b, C=C, D0=D0, D1=D1, d0=d0, d1=d1, a0=a0, a1=a1)
    w, iw, square, cend, tr_integral = reference_source(
        ud - ug, C, h, nu, omega, b, alpha, d0, d1, D0, D1)
    assert_allclose(out["w"], w, rtol=2e-11, atol=5e-13)
    assert_allclose(out["integrated_slip"], iw, rtol=2e-11, atol=5e-13)
    assert_allclose(out["C"], cend, rtol=2e-11, atol=5e-13)
    assert_allclose(out["Q_mean"], rd*nu*square, rtol=2e-10, atol=2e-12)
    assert_allclose(out["Q_random"], rd*nu*tr_integral, rtol=2e-11, atol=2e-12)
    old = .5*rg*np.sum(ug**2) + .5*rd*(np.sum(ud**2) + np.trace(C))
    new = .5*rg*np.sum(out["ug"]**2) + .5*rd*(np.sum(out["ud"]**2) + np.trace(out["C"]))
    assert_allclose(new - old, out["mechanical_work"] + out["SGS_work"]
                    - out["Q_mean"] - out["Q_random"], rtol=2e-10, atol=3e-12)
    assert out["Q_mean"] >= 0 and out["Q_random"] >= 0


def test_half_gyroperiod_work_and_full_period_transport():
    zero, b = np.zeros(3), np.array([0., 0., 1.])
    d, rg, rd, omega = np.array([1., 0., 0.]), 1., 2., 7.
    alpha = 1 + rd/rg
    h = np.pi/(alpha*omega)
    out = homogeneous_source(zero, zero, h, rho_g=rg, rho_d=rd, nu=0,
                             omega=omega, bhat=b, C=np.zeros((3, 3)), d0=d)
    assert out["Q_mean"] == 0 and out["Q_random"] == 0
    assert_allclose(out["w"], -2*np.cross(b, d)/(alpha*omega), atol=1e-16)
    assert_allclose(out["mechanical_work"], 2*(rd/alpha)/(alpha*omega)**2, rtol=1e-14)
    out = homogeneous_source(zero, d, 2*h, rho_g=rg, rho_d=rd, nu=0,
                             omega=omega, bhat=b, C=np.zeros((3, 3)))
    assert_allclose(out["integrated_slip"], zero, atol=2e-16)


def test_hyqmom_linear_forcing_and_source_parity():
    from phrike.dust_moments import GaussianClosure
    from phrike.dust_sources import local_coupling_step

    b = np.array([1., 0., 0.])[:, None]
    nu, omega = np.array([0., .7, 1000.]), np.array([0., 3., 20.])
    ug, ud = np.zeros((3, 3)), np.array([.7, -.4, .2])[:, None]*np.ones((1, 3))
    initial = np.array([.8, .12, 2.7, .4])[:, None]*np.ones((1, 3))
    D0, D1 = np.diag([.3, .2, .2])[:, :, None], np.diag([-.1, .3, .3])[:, :, None]
    out = homogeneous_source(ug, ud, .5, rho_g=1., rho_d=2., nu=nu, omega=omega,
                             bhat=b, hyqmom=initial, D0=D0, D1=D1)
    C = np.zeros((3, 3, 3))
    C[0, 0], C[1, 1], C[2, 2] = initial[0], initial[3], initial[3]
    gaussian = homogeneous_source(ug, ud, .5, rho_g=1., rho_d=2., nu=nu, omega=omega,
                                  bhat=b, C=C, D0=D0, D1=D1)
    assert_allclose(out["hyqmom"][0], gaussian["C"][0, 0])
    assert_allclose(out["hyqmom"][3], gaussian["C"][1, 1])
    assert_allclose(out["Q_random"], gaussian["Q_random"])
    s = np.exp(-nu*.5)
    sigma = gaussian["C"][0, 0] - s*s*initial[0]
    assert_allclose(out["hyqmom"][1], s**3*initial[1])
    assert_allclose(out["hyqmom"][2], s**4*initial[2] + 6*s*s*sigma*initial[0] + 3*sigma*sigma)
    closure = GaussianClosure(ndim=1)
    gas = np.zeros((8, 3))
    gas[0], gas[4], gas[5] = 1., 4., 1.
    dust = closure.from_primitives(np.full(3, 2.), ud, C)
    oldgas, olddust, _, ledger = local_coupling_step(
        gas, dust, np.zeros(3), .5, closure=closure, gamma=5/3, nu=nu,
        Delta=1., ndim=1, d_model="none", sgs_decay=False, charge_to_mass=3.)
    exact = homogeneous_source(ug, ud, .5, rho_g=1., rho_d=2., nu=nu, omega=3., bhat=b, C=C)
    assert_allclose(exact["ug"], oldgas[1:4]/oldgas[0], atol=3e-16)
    assert_allclose(exact["ud"], closure.primitives(olddust)["u"], atol=3e-16)
    assert_allclose(exact["C"], closure.primitives(olddust)["C"], atol=5e-16)
    assert_allclose(exact["Q_mean"], ledger["Q_mean"], atol=5e-15)
    assert_allclose(exact["Q_random"], ledger["Q_rand"], atol=5e-15)


def test_constant_components_broadcast_over_cells_and_invalid_diffusion():
    w, b, C = np.array([.3, -.2, .7]), np.array([0., 0., 1.]), np.eye(3)
    rates = np.array([0., .1, 1., 10., 1000.])
    for k in range(4):
        vectors = slip_phi(k, w, .2, rates, 3., b)
        tensors = covariance_phi(k, C, .2, rates, 3., b)
        for i, rate in enumerate(rates):
            assert_allclose(vectors[:, i], slip_phi(k, w, .2, rate, 3., b))
            assert_allclose(tensors[:, :, i], covariance_phi(k, C, .2, rate, 3., b))
    result = homogeneous_source(w, -.5*w, .2, rho_g=2., rho_d=rates, nu=rates,
                                bhat=b, C=C, d0=w, D0=.1*C)
    for i, rate in enumerate(rates):
        scalar = homogeneous_source(w, -.5*w, .2, rho_g=2., rho_d=rate, nu=rate,
                                    bhat=b, C=C, d0=w, D0=.1*C)
        for key in result:
            assert_allclose(result[key][..., i], scalar[key])
    with pytest.raises(ValueError, match="positive semidefinite"):
        homogeneous_source(w, w, 1., rho_g=1., rho_d=1., nu=1., C=C,
                           D0=C, D1=-2*C)
    assert_allclose(slip_phi(0, w, .1, 2., 50., np.zeros(3)), np.exp(-.2)*w)
