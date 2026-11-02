import numpy as np
import pytest

from phrike.sgs import (
    advect_scalar, compression_factor, diffuse_scalar, drain_production, sgs_coefficients,
)


def test_coefficients_and_limits():
    e, rho, Delta = np.array([0.0, 1.5, 1.5, 1.5]), np.ones(4), 0.7
    nu = np.array([0, 0, 1e-9, 1e10])
    with np.errstate(all="raise"):
        c = sgs_coefficients(e, rho, nu, Delta)
    assert np.isinf(c["tau"][0])
    assert c["D"][0] == c["D"][1] == 0
    assert np.isfinite(c["D"]).all()
    np.testing.assert_allclose(c["D"][2], nu[2]*c["sigma"][2]**2, rtol=1e-14)
    np.testing.assert_allclose(c["D"][3], np.pi/2*c["eps"][3], rtol=1e-9)
    for model in ("mt25", "white", "finite_tau", "none"):
        with np.errstate(all="raise"):
            zero = sgs_coefficients(np.zeros(4), rho, np.ones(4), Delta, model)
        assert (zero["D"] == 0).all()
        actual = sgs_coefficients(e, rho, nu, Delta, model)
        if model == "white":
            np.testing.assert_array_equal(actual["D"], actual["eps"])
        elif model == "finite_tau":
            np.testing.assert_array_equal(actual["D"], actual["eps"]/(1+actual["x"]))
        elif model == "none":
            assert (actual["D"] == 0).all()
    with pytest.raises(ValueError):
        sgs_coefficients(e, rho, nu, Delta, "unknown")


@pytest.mark.parametrize("shape", [(32,), (9, 12), (5, 6, 8)])
def test_production(shape):
    rng = np.random.default_rng(7)
    drop = rng.normal(0.4, 0.8, shape)
    dxs = tuple(1/n for n in reversed(shape))
    P = drain_production(drop, dxs, 0.1, np.full(shape, 100.0))
    assert (P >= 0).all()
    np.testing.assert_allclose(P.sum(), drop.sum(), rtol=3e-15)
    available = rng.uniform(0, 0.1, shape)
    capped = drain_production(drop, dxs, 0.1, available, f_cap=0.3)
    np.testing.assert_allclose(capped, np.minimum(P, 0.3*available))
    assert (capped <= 0.3*available).all()
    assert not drain_production(-np.ones(shape), dxs, 0.1, available).any()


def test_gaussian_smoothing_axis_convention():
    shape = (8, 12)
    y, x = np.indices(shape)
    drop = 2 + 0.3*np.cos(2*np.pi*x/12) + 0.2*np.cos(2*np.pi*y/8)
    dxs, ell = (0.13, 0.21), 0.4
    expected = (2 + 0.3*np.cos(2*np.pi*x/12)*np.exp(-0.5*(2*np.pi/(12*dxs[0])*ell)**2)
                + 0.2*np.cos(2*np.pi*y/8)*np.exp(-0.5*(2*np.pi/(8*dxs[1])*ell)**2))
    actual = drain_production(drop, dxs, ell, np.full(shape, 100.0))
    np.testing.assert_allclose(actual, expected, atol=2e-15)


@pytest.mark.parametrize("shape", [(64,), (11, 14), (5, 6, 8)])
def test_scalar_transport_and_diffusion(shape):
    rng = np.random.default_rng(11)
    e = rng.uniform(0, 1, shape)
    e[..., :3] = 0
    u = rng.normal(size=(len(shape), *shape))
    dxs = tuple(1/n for n in reversed(shape))
    out, nsub = advect_scalar(e, u, 0.1, dxs)
    assert nsub > 1
    assert np.min(out) >= 0
    np.testing.assert_allclose(out.sum(), e.sum(), rtol=3e-15)
    diffused, nsub = diffuse_scalar(e, rng.uniform(0, 0.1, shape), 0.1, dxs)
    assert np.min(diffused) >= 0
    np.testing.assert_allclose(diffused.sum(), e.sum(), rtol=3e-15)
    assert nsub > 1


def test_compression_and_zero_time():
    div = np.array([-3.0, 0, 2.0])
    np.testing.assert_allclose(compression_factor(div, 0.2), np.exp(-2/3*div*0.2))
    e = np.array([0.0, 1, 2])
    out, nsub = advect_scalar(e, np.ones((1, 3)), 0, (0.1,))
    np.testing.assert_array_equal(out, e)
    assert nsub == 0
