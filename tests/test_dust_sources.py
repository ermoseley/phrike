import numpy as np
import pytest

from phrike.dust_moments import GaussianClosure, HyqmomClosure
from phrike.dust_sources import epstein_nu, local_coupling_step
from phrike.sgs import sgs_coefficients


def setup(closure, shape):
    rng = np.random.default_rng(6)
    gas = np.empty((closure.ndim+2, *shape))
    gas[0] = rng.uniform(0.5, 2, shape)
    gas[1:-1] = rng.normal(0, 0.5, (closure.ndim, *shape))
    gas[-1] = 5 + 0.5*np.sum(gas[1:-1]**2, axis=0)/gas[0]
    rho = rng.uniform(0.1, 1, shape)
    u = rng.normal(size=(3, *shape))
    if closure.name == "gaussian":
        C = np.zeros((3, 3, *shape))
        for i in range(3):
            C[i, i] = 0.3
        C[0, 1] = C[1, 0] = 0.1
        dust = closure.from_primitives(rho, u, C)
    else:
        dust = closure.from_primitives(rho, u, 0.3, 0.03, 0.3, 0.2)
    es = rng.uniform(0.2, 2, shape)
    return gas, dust, es


@pytest.mark.parametrize("closure,shape", [(GaussianClosure(1), (9,)),
    (HyqmomClosure(), (9,)), (GaussianClosure(2), (4, 5)),
    (GaussianClosure(3), (3, 4, 5))])
@pytest.mark.parametrize("h", [1e-8, 0.1, 1e2, 1e8])
def test_coupling_conservation_and_positivity(closure, shape, h):
    gas, dust, es = setup(closure, shape)
    dust[..., 0] = 0
    gas0, dust0, es0 = gas.copy(), dust.copy(), es.copy()
    nu = np.linspace(0, 2, np.prod(shape)).reshape(shape)
    g, d, e, ledger = local_coupling_step(gas, dust, es, h, closure=closure, gamma=1.4,
                                        nu=nu, Delta=0.2, ndim=closure.ndim)
    np.testing.assert_array_equal(gas, gas0)
    np.testing.assert_array_equal(dust, dust0)
    np.testing.assert_array_equal(es, es0)
    initial = gas[-1] + closure.kinetic_energy(dust) + es
    final = g[-1] + closure.kinetic_energy(d) + e
    # Sub-cycling the reservoir decay (up to 4096 substeps) accumulates round-off.
    np.testing.assert_allclose(final, initial, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(final.sum(), initial.sum(), rtol=1e-12)
    p0, p1 = closure.primitives(dust), closure.primitives(d)
    for i in range(closure.ndim):
        np.testing.assert_allclose(g[i+1] + p1["rho"]*p1["u"][i],
                                   gas[i+1] + p0["rho"]*p0["u"][i],
                                   rtol=2e-14, atol=8e-16)
    assert closure.realizable(d).all()
    assert (e >= 0).all()
    assert (ledger["Q_mean"] >= -1e-15).all()
    assert (ledger["Q_rand"] >= -1e-15).all()
    np.testing.assert_allclose(ledger["L_dust"] + ledger["L_decay"], es-e, atol=5e-16)
    scale = np.max(closure.kinetic_energy(dust)) + np.max(es)
    np.testing.assert_allclose(ledger["L_dust"], ledger["Q_rand"] + ledger["Q_projection"]
        + closure.random_energy(d) - closure.random_energy(dust), atol=1e-12*scale)
    np.testing.assert_array_equal(d[..., 0], dust[..., 0])
    assert (ledger["L_dust"][..., 0] == 0).all()
    if h == 1e8:
        active = (nu > 0) & (dust[0] > 0)
        np.testing.assert_allclose(p1["u"][:closure.ndim, active],
                                   (g[1:-1]/g[0])[:, active], atol=2e-15)
        np.testing.assert_allclose(p1["u"][closure.ndim:, active], 0, atol=2e-15)


@pytest.mark.parametrize("closure", [GaussianClosure(1), HyqmomClosure()])
def test_effective_forcing_small_step_and_terminal_map(closure):
    gas, dust, es = setup(closure, (8,))
    nu, Delta, h = 0.7, 0.3, 1e-9
    D = sgs_coefficients(es, gas[0], nu, Delta)["D"]
    g, d, e, ledger = local_coupling_step(gas, dust, es, h, closure=closure, gamma=1.4,
                                        nu=nu, Delta=Delta, ndim=1)
    np.testing.assert_allclose(ledger["L_dust"]/(3*dust[0]*h), D, rtol=2e-8)
    # With negligible dust loading the reservoir does not drain, so the
    # forcing is constant and the terminal map must give exactly D/nu.
    h = 1e4
    light = dust*1e-12  # residual drain 3 rho_d D h / e_s ~ 1e-8
    g, d, e, ledger = local_coupling_step(gas, light, es, h, closure=closure, gamma=1.4,
                                        nu=nu, Delta=Delta, ndim=1, sgs_decay=False)
    D = sgs_coefficients(es, gas[0], nu, Delta)["D"]
    p = closure.primitives(d)
    if closure.name == "gaussian":
        for i in range(3):
            np.testing.assert_allclose(p["C"][i, i], D/nu, rtol=1e-6)
    else:
        np.testing.assert_allclose(p["v"], D/nu, rtol=1e-6)
        np.testing.assert_allclose(p["sperp2"], D/nu, rtol=1e-6)
        np.testing.assert_allclose(p["q"], 0, atol=3e-16)
        np.testing.assert_allclose(p["r"], 3*p["v"]**2, rtol=1e-7, atol=1e-15)


@pytest.mark.parametrize("backreaction", [True, False])
@pytest.mark.parametrize("model", ["mt25", "white", "finite_tau", "none"])
def test_zero_frequency_zero_reservoir_and_no_decay(backreaction, model):
    closure = GaussianClosure(2)
    gas, dust, es = setup(closure, (3, 4))
    es[:, 0] = 0
    for nu in (0.0, 1.0):
        g, d, e, ledger = local_coupling_step(gas, dust, es, 3, closure=closure, gamma=1.4,
            nu=nu, Delta=0.3, ndim=2, d_model=model, sgs_decay=False,
            backreaction=backreaction)
        np.testing.assert_allclose(g[-1]+closure.kinetic_energy(d)+e,
                                   gas[-1]+closure.kinetic_energy(dust)+es, rtol=3e-15)
        assert closure.realizable(d).all()
        assert (e >= 0).all()
        if not backreaction:
            np.testing.assert_array_equal(g[1:-1], gas[1:-1])
        if nu == 0 and model in ("mt25", "none"):
            np.testing.assert_allclose(d, dust, atol=2e-15)
            np.testing.assert_array_equal(e, es)


def test_epstein_and_zero_step():
    for ndim in (1, 2, 3):
        closure = GaussianClosure(ndim)
        gas, dust, es = setup(closure, (4,)*ndim)
        rho = gas[0]
        expected = rho*np.sqrt(8*0.4*5/(np.pi*rho))/0.2
        np.testing.assert_allclose(epstein_nu(gas, 1.4, 0.2, ndim), expected, rtol=1e-15)
        g, d, e, ledger = local_coupling_step(gas, dust, es, 0, closure=closure,
            gamma=1.4, nu=0, Delta=0.2, ndim=ndim)
        np.testing.assert_array_equal(g, gas)
        np.testing.assert_array_equal(d, dust)
        np.testing.assert_array_equal(e, es)
        assert all(not a.any() for a in ledger.values())
