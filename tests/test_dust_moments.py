import numpy as np
import pytest
from numpy.polynomial.hermite import hermgauss
from scipy.integrate import solve_ivp

from phrike.dust_moments import (
    TOL, GaussianClosure, HyqmomClosure, diffuse, max_stable_dt,
    ou_gaussian, ou_hyqmom, phi, transport,
)


def state(closure, shape, seed=42, cold=False):
    rng = np.random.default_rng(seed)
    rho = rng.uniform(0.2, 1.5, shape)
    u = rng.normal(0, 0.4, (3, *shape))
    if closure.name == "gaussian":
        B = rng.normal(0, 0.3, (3, 3, *shape))
        C = np.einsum("ik...,jk...->ij...", B, B)
        if cold:
            C[:] = 0
        return closure.from_primitives(rho, u, C)
    v = np.zeros(shape) if cold else rng.uniform(0.1, 0.3, shape)
    q = 0.3*v**1.5
    r = 3*v*v
    return closure.from_primitives(rho, u, v, q, r, v)


@pytest.mark.parametrize("closure", [GaussianClosure(1), GaussianClosure(2),
                                     GaussianClosure(3), HyqmomClosure()])
def test_primitives_realizability_and_energy(closure):
    for cold in (False, True):
        U = state(closure, (16,), cold=cold)
        U[:, :2] = 0
        assert closure.realizable(U).all()
        p = closure.primitives(U)
        np.testing.assert_allclose(closure.from_primitives(**p), U, atol=3e-15)
        np.testing.assert_allclose(closure.random_energy(U) + closure.mean_kinetic_energy(U),
                                   closure.kinetic_energy(U), atol=3e-15)
        assert np.isfinite(closure.flux(U, 0)).all()
        assert np.isfinite(closure.signal_speed(U, 0)).all()
    bad = state(closure, (16,))
    bad[0, 0] = -1
    bad[2 if closure.name == "hyqmom" else 4, 1] = -1
    bad[:, 2] = 0
    bad[1, 2] = 1e-3
    if closure.name == "gaussian":
        bad[7, 3] = 10
    else:
        bad[4, 3] = -1
        bad[7, 4] = -1
    assert not closure.realizable(bad)[:4].any()
    if closure.name == "hyqmom":
        assert not closure.realizable(bad)[4]
        cold_bad = closure.from_primitives(np.ones(1), np.zeros((3, 1)), 0, 0, 1, 0)
        assert not closure.realizable(cold_bad).any()


def test_gaussian_flux_quadrature():
    rng = np.random.default_rng(10)
    B = rng.normal(size=(3, 3))
    C = B @ B.T
    u = rng.normal(size=3)
    closure = GaussianClosure(3)
    U = closure.from_primitives(np.array(1.3), u, C)
    nodes, weights = hermgauss(4)
    indices = np.indices((4, 4, 4)).reshape(3, -1)
    z = np.sqrt(2)*nodes[indices]
    w = np.prod(weights[indices]/np.sqrt(np.pi), axis=0)
    velocities = u[:, None] + np.linalg.cholesky(C) @ z
    raw = np.array([np.ones(64), *velocities,
                    velocities[0]**2, velocities[1]**2, velocities[2]**2,
                    velocities[0]*velocities[1], velocities[0]*velocities[2],
                    velocities[1]*velocities[2]])
    for d in range(3):
        exact = 1.3*np.sum(raw*velocities[d]*w, axis=1)
        np.testing.assert_allclose(closure.flux(U, d), exact, rtol=3e-14, atol=1e-14)


def test_hyqmom_fifth_moment():
    closure = HyqmomClosure()
    rho, m, v = np.ones(8), np.linspace(-1, 1, 8), np.linspace(0.1, 1, 8)
    u = np.zeros((3, 8))
    u[0] = m
    U = closure.from_primitives(rho, u, v, 0, 3*v*v, 0)
    np.testing.assert_allclose(closure.flux(U, 0)[4], m**5 + 10*m**3*v + 15*m*v*v,
                               atol=2e-14)
    weight = np.linspace(0.1, 0.9, 8)
    velocities = np.array([-0.7, 1.3])[:, None]
    weights = np.array([weight, 1-weight])
    U[:5] = np.array([np.sum(weights*velocities**k, axis=0) for k in range(5)])
    assert closure.realizable(U).all()
    np.testing.assert_allclose(closure.flux(U, 0)[4], np.sum(weights*velocities**5, axis=0),
                               rtol=5e-14, atol=1e-14)
    p = closure.primitives(U)
    s3 = p["q"]/p["v"]**1.5
    c5 = (closure.flux(U, 0)[4] - p["u"][0]**5 - 10*p["u"][0]**3*p["v"]
          - 10*p["u"][0]**2*p["q"] - 5*p["u"][0]*p["r"])
    np.testing.assert_allclose(c5/p["v"]**2.5, s3*(s3*s3+2), atol=3e-14)


def assert_transport_bookkeeping(closure, U, out, stats):
    """Mass and momentum are conserved exactly; dust kinetic energy changes only
    by the reported sub-cell/round-off projection energy (returned to the gas
    by the coupled solver)."""
    axes = tuple(range(1, U.ndim))
    np.testing.assert_allclose(out[:4].sum(axis=axes) if closure.name == "gaussian"
                               else out[[0, 1, 5, 6]].sum(axis=axes),
                               U[:4].sum(axis=axes) if closure.name == "gaussian"
                               else U[[0, 1, 5, 6]].sum(axis=axes), rtol=1e-12, atol=1e-13)
    gained = closure.kinetic_energy(out).sum() - closure.kinetic_energy(U).sum()
    np.testing.assert_allclose(gained, stats["regularization_dE"].sum(),
                               atol=1e-12*max(1.0, closure.kinetic_energy(U).sum()))


@pytest.mark.parametrize("closure,shape", [(GaussianClosure(1), (32,)),
    (HyqmomClosure(), (32,)), (GaussianClosure(2), (7, 9)),
    (GaussianClosure(3), (4, 5, 6))])
def test_rough_transport(closure, shape):
    U = state(closure, shape)
    cold = state(closure, shape, cold=True)
    U[..., :3] = cold[..., :3]
    U[..., 3:5] = 0
    dxs = tuple(1/n for n in reversed(shape))
    out, stats = transport(U, closure, 0.02, dxs)
    assert closure.realizable(out).all()
    assert_transport_bookkeeping(closure, U, out, stats)
    assert stats["substeps"] >= 1
    assert stats["max_doublings"] <= 60
    assert 0 <= stats["min_slope_scale"] <= 1


@pytest.mark.parametrize("closure", [GaussianClosure(1), HyqmomClosure()])
def test_200_substeps_conserve_all_fields(closure):
    U = state(closure, (12,))
    initial = U.sum(axis=1)
    steps = 0
    for _ in range(200):
        U, stats = transport(U, closure, 0.002, (1.0,))
        steps += stats["substeps"]
    assert steps >= 200
    assert closure.realizable(U).all()
    np.testing.assert_allclose(U.sum(axis=1), initial, rtol=1e-12, atol=1e-13)


@pytest.mark.parametrize("closure,shape", [(GaussianClosure(1), (32,)),
    (HyqmomClosure(), (32,)), (GaussianClosure(2), (5, 32)),
    (GaussianClosure(3), (3, 4, 32))])
def test_uniform_velocity_cold_and_colliding_streams(closure, shape):
    n = 32
    rho = np.broadcast_to(1 + 0.2*np.sin(2*np.pi*np.arange(n)/n), shape).copy()
    u = np.zeros((3, *shape))
    u[0] = 0.7
    dxs = (1/n,)*closure.ndim
    def make():
        if closure.name == "gaussian":
            return closure.from_primitives(rho, u, np.zeros((3, 3, *shape)))
        return closure.from_primitives(rho, u, 0, 0, 0, 0)
    U = make()
    out, _ = transport(U, closure, 0.1, dxs)
    p = closure.primitives(out)
    variance = p["C"][0, 0] if closure.name == "gaussian" else p["v"]
    assert np.max(np.abs(variance)) < 8*TOL
    u[0, ..., n//2:] = -0.7
    out, _ = transport(make(), closure, 0.05, dxs)
    assert closure.realizable(out).all()
    assert np.max(closure.random_energy(out)) > 1e-3


def streaming_state(closure, n, time):
    x = (np.arange(n) + 0.5)/n
    z, w = hermgauss(12)
    vx = 0.8 + np.sqrt(2)*0.04*z
    weight = w[:, None]/np.sqrt(np.pi)*(1 + 0.2*np.sin(2*np.pi*(x - vx[:, None]*time)))
    moments = np.array([np.sum(weight*vx[:, None]**k, axis=0) for k in range(5)])
    if closure.name == "hyqmom":
        return np.array([*moments, 0.2*moments[0], -0.1*moments[0], 0.21*moments[0]])
    return np.array([moments[0], moments[1], 0.2*moments[0], -0.1*moments[0],
                     moments[2], 0.12*moments[0], 0.09*moments[0],
                     0.2*moments[1], -0.1*moments[1], -0.02*moments[0]])


@pytest.mark.parametrize("closure", [GaussianClosure(1), HyqmomClosure()])
def test_free_streaming_second_order(closure):
    errors = []
    for n in (32, 64, 128):
        U = streaming_state(closure, n, 0)
        out, _ = transport(U, closure, 0.1, (1/n,))
        exact = streaming_state(closure, n, 0.1)
        errors.append(np.mean(np.abs(out - exact)))
    order = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.min(order) >= 1.7, (errors, order)


@pytest.mark.parametrize("closure,shape", [(GaussianClosure(1), (24,)),
    (HyqmomClosure(), (24,)), (GaussianClosure(2), (6, 8)),
    (GaussianClosure(3), (4, 5, 6))])
def test_diffusion(closure, shape):
    U = state(closure, shape)
    U[..., :2] = 0
    rng = np.random.default_rng(9)
    kappa = rng.uniform(0, 1, shape)
    dxs = (0.1,)*closure.ndim
    out, nsub = diffuse(U, kappa, 0.04, dxs)
    assert nsub > 1
    assert closure.realizable(out).all()
    axes = tuple(range(1, U.ndim))
    np.testing.assert_allclose(out.sum(axis=axes), U.sum(axis=axes), rtol=1e-13, atol=1e-13)
    density = U[0].copy()
    for _ in range(nsub):
        rhs = np.zeros(shape)
        for d, dx in enumerate(dxs):
            axis = -1-d
            kp = (kappa + np.roll(kappa, -1, axis))/2
            km = np.roll(kp, 1, axis)
            rhs += (kp*np.roll(density, -1, axis) + km*np.roll(density, 1, axis)
                    - (kp+km)*density)/dx**2
        density += (0.04/nsub)*rhs
    np.testing.assert_allclose(out[0], density, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("nh", [1e-4, 0.01, 1.0, 100.0, 1e4])
def test_ou_against_ode(nh):
    D, nu = 0.7, 1.3
    h = nh/nu
    C = np.array([[0.8, 0.1, -0.2], [0.1, 1.2, 0.2], [-0.2, 0.2, 0.9]])
    v, q, r, sperp2 = 0.8, 0.1, 2.0, 1.2
    y0 = np.r_[C.ravel(), v, q, r, sperp2]
    def rhs(t, y):
        dc = -2*nu*y[:9].reshape(3, 3) + 2*D*np.eye(3)
        return np.r_[dc.ravel(), -2*nu*y[9]+2*D, -3*nu*y[10],
                     -4*nu*y[11]+12*D*y[9], -2*nu*y[12]+2*D]
    sol = solve_ivp(rhs, (0, h), y0, method="Radau", rtol=2e-13, atol=2e-14)
    A, Sigma = np.exp(-nh), 2*D*phi(2*nu, h)
    exact = np.r_[ou_gaussian(C, A, Sigma).ravel(),
                  ou_hyqmom(v, q, r, sperp2, A, Sigma)]
    assert sol.success
    np.testing.assert_allclose(exact, sol.y[:, -1], rtol=1e-11, atol=3e-13)
    if nh >= 100:
        np.testing.assert_allclose(exact[[0, 4, 8, 9, 12]], D/nu, rtol=1e-12)
        np.testing.assert_allclose(exact[11], 3*(D/nu)**2, rtol=1e-12)


def test_phi_zero_and_broadcast():
    rates = np.array([0, 1e-12, 1e-3, 1, 1e8])
    result = phi(rates, 0.3)
    assert result[0] == 0.3
    np.testing.assert_allclose(result[1:], -np.expm1(-rates[1:]*0.3)/rates[1:], rtol=2e-15)
    C = np.broadcast_to(np.eye(3)[:, :, None], (3, 3, 5)).copy()
    assert ou_gaussian(C, np.ones(5), np.zeros(5)).shape == C.shape
    moments = ou_hyqmom(*(np.ones(5, dtype=np.float32) for _ in range(4)), 0.5, 0.2)
    assert all(value.dtype == np.float64 for value in moments)


def test_cfl_and_stationary_vacuum():
    closure = GaussianClosure(1)
    U = closure.from_primitives(np.ones(8), np.ones((3, 8)),
                                 np.broadcast_to(np.eye(3)[:, :, None], (3, 3, 8)))
    limit = max_stable_dt(U, closure, (0.1,))
    assert limit <= 0.5*0.8*0.1/np.max(closure.signal_speed(U, 0))
    out, stats = transport(U, closure, 3*limit, (0.1,))
    np.testing.assert_array_equal(out, U)
    assert stats["substeps"] >= 3
    U[:] = 0
    assert np.isinf(max_stable_dt(U, closure, (0.1,)))
    out, stats = transport(U, closure, 1e8, (0.1,))
    np.testing.assert_array_equal(out, U)
    assert stats["substeps"] == 1


def test_cold_outflow_into_vacuum_3d():
    closure = GaussianClosure(3)
    shape = (3, 4, 6)
    U = state(closure, shape, seed=1)
    U[..., :2] = state(closure, shape, seed=1, cold=True)[..., :2]
    U[..., 2:4] = 0
    out, stats = transport(U, closure, 0.01, (1/6, 1/4, 1/3))
    assert closure.realizable(out).all()
    assert stats["substeps"] < 20
    assert_transport_bookkeeping(closure, U, out, stats)


def test_cold_acoustic_wave_split_vacuum():
    # Near a velocity zero, a Rusanov split can have zero density but residual
    # momentum from round-off. Accepting that as vacuum stalls every retry.
    closure = GaussianClosure(1)
    phase = 2*np.pi*np.arange(64)/64
    rho = 1 + 1e-4*np.sin(phase)
    u = np.zeros((3, 64))
    u[0] = np.sqrt(1.4)*1e-4*np.sin(phase)
    U = closure.from_primitives(rho, u, np.zeros((3, 3, 64)))
    out, stats = transport(U, closure, 0.001, (1/64,))
    assert closure.realizable(out).all()
    assert stats["substeps"] == 1
    np.testing.assert_allclose(out.sum(axis=1), U.sum(axis=1), atol=1e-18)


def _blob_moments(n, t, nq=32):
    """Cell-averaged x moments of a free-streaming Gaussian-velocity packet."""
    gx, wx = np.polynomial.legendre.leggauss(4)
    x = (np.arange(n)[:, None] + 0.5 + 0.5*gx)/n
    z, w = hermgauss(nq)
    vel, weights = 0.6 + np.sqrt(2)*0.15*z, w/np.sqrt(np.pi)
    dist = (x[..., None] - vel*t - 0.35 + 0.5) % 1.0 - 0.5
    density = 0.05 + np.exp(-0.5*(dist/0.07)**2)
    return np.array([np.einsum("nqv,v,q->n", density, weights*vel**k, wx/2) for k in range(5)])


def _hyqmom_from_raw(raw):
    closure = HyqmomClosure()
    rho, m = raw[0], raw[1]/raw[0]
    v = raw[2]/rho - m*m
    q = raw[3]/rho - 3*m*raw[2]/rho + 2*m**3
    r = raw[4]/rho - 4*m*raw[3]/rho + 6*m*m*raw[2]/rho - 3*m**4
    u = np.zeros((3, rho.size))
    u[0] = m
    return closure, closure.from_primitives(rho, u, v, q, r, np.zeros_like(rho))


def test_hyqmom_smooth_packet_is_second_order_and_smooth():
    # Raw-component reconstruction made the face skewness/kurtosis noisy here
    # (error stalled, grid-scale roughness grew with N); the interior
    # primitive-space reconstruction must converge and get smoother.
    errors, roughness = [], []
    for n in (64, 128, 256):
        closure, U = _hyqmom_from_raw(_blob_moments(n, 0.0))
        out, stats = transport(U, closure, 0.3, (1/n,))
        ref = _blob_moments(n, 0.3)
        errors.append(np.mean(np.abs(out[0] - ref[0]))/np.mean(ref[0]))
        v = closure.primitives(out)["v"]
        roughness.append(np.std(np.diff(v, 2)))
        assert stats["mood_cells"] == 0
    orders = np.log2(np.array(errors[:-1])/errors[1:])
    assert np.min(orders) >= 1.7, (errors, orders)
    assert roughness[2] < 0.5*roughness[1] < 0.25*roughness[0], roughness
