"""Analytic path integrals retain resonances and the oscillating-mean variance."""
import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.integrate import quad
from scipy.linalg import expm

from phrike.coupled_exponential import covariance_phi, slip_phi
from phrike.exponential_path import ExponentialPath, covariance_path, slip_path


@pytest.mark.parametrize('rate', [0, -1e-10, -.2 + .1j, -2 + 1000j, 1000j])
def test_polynomial_exponential_integrals(rate):
    coefficients = np.array([1., -.3, .2])
    path = ExponentialPath([(rate, coefficients)])
    reference = sum(c*quad(lambda t: (t**p*np.exp(rate*t)).real, 0, 1,
                           epsabs=2e-13, limit=1000)[0]
                    for p, c in enumerate(coefficients))
    assert_allclose(path.integral(), reference, atol=2e-13)


@pytest.mark.parametrize('source,target', [(0, 0), (1j, 1j), (0, -1e-12),
                                          (1000j, 1000j), (-2 + 1000j, -2 - 1000j)])
def test_convolution_resonance(source, target):
    path = ExponentialPath([(source, np.array([.7, -.2, .3]))]).convolve(target)
    # A triangular autonomous system generates the polynomial forced mode.
    operator = np.zeros((4, 4), complex)
    operator[0, 0] = target
    operator[0, 1:] = [.7, -.2, .3]
    operator[1:, 1:] = np.eye(3)*source
    operator[2, 1] = 1
    operator[3, 2] = 2
    for t in [.3, 1.]:
        expected = (expm(t*operator) @ [0, 1, 0, 0])[0]
        assert_allclose(path.evaluate(t), expected.real, atol=4e-13)


@pytest.mark.parametrize('nu,omega', [(0., 0.), (1e-12, 1e-12), (.4, 1.), (.4, 1000.)])
def test_frozen_slip_and_covariance(nu, omega):
    w = np.array([.3, -.5, .7])
    C = np.array([[.5, .1, .02], [.1, .3, .04], [.02, .04, .4]])
    b = np.array([.2, .3, .4]); b /= np.linalg.norm(b)
    force = np.array([[.1, -.2, .3], [.2, .1, -.1], [-.1, .05, .02]])
    diffusion = np.stack([np.eye(3)*.1, np.eye(3)*.02, np.eye(3)*.01])
    wp = slip_path(w, .7, nu, omega, b, forcing=force, alpha=1.3)
    cp = covariance_path(C, .7, nu, omega, b, forcing=diffusion)
    for t in [.3, 1.]:
        expected_w = slip_phi(0, w, .7*t, nu, omega, b, 1.3)
        expected_c = covariance_phi(0, C, .7*t, nu, omega, b)
        for p in range(3):
            # Integral polynomial forcing = p! t^(p+1) phi_(p+1)(t L).
            import math
            expected_w += .7*math.factorial(p)*t**(p+1)*slip_phi(p+1, force[p], .7*t, nu, omega, b, 1.3)
            expected_c += .7*math.factorial(p)*t**(p+1)*covariance_phi(p+1, diffusion[p], .7*t, nu, omega, b)
        assert_allclose(wp.evaluate(t), expected_w, atol=3e-13)
        assert_allclose(cp.evaluate(t), expected_c, atol=3e-13)


def test_mean_variance_survives_raw_average():
    # One full, unresolved rotation has zero averaged mean and finite pressure.
    w = slip_path(np.array([1., 0., 0.]), 1., 0., 2000*np.pi,
                  np.array([0., 0., 1.]))
    assert_allclose(w.integral(), 0, atol=1e-15)
    outer = np.array([[w.component(i).multiply(w.component(j)).integral()
                       for j in range(3)] for i in range(3)])
    assert_allclose(outer, np.diag([.5, .5, 0]), atol=1e-14)


def test_arbitrary_forcing_tensor_reshape_and_stack():
    C0 = np.diag([.2, .3, .4])
    b = np.array([0., 0., 1.])
    prior = covariance_path(C0, 1., .1, 1000., b)
    transformed = prior.reshape((9,)).linear_map(np.eye(9)*.2).reshape((3, 3))
    forced = covariance_path(C0, 1., .1, 1000., b, forcing=transformed)
    # Resonant scalar multiplier of the same homogeneous generator.
    assert_allclose(forced.evaluate(), 1.2*prior.evaluate(), atol=1e-13)
    stacked = ExponentialPath.stack([prior.component(0, 0), prior.component(1, 1)])
    assert_allclose(stacked.integral(), prior.integral()[[0, 1], [0, 1]], atol=1e-14)


def test_component_first_cells_and_envelopes():
    w0 = np.array([[.2, .3], [.4, -.1], [.1, .2]])
    nu = np.array([1e-10, .3]); omega = np.array([0., 1000.])
    b = np.array([[0., 0.], [0., 0.], [1., 1.]])
    path = slip_path(w0, .5, nu, omega, b)
    assert_allclose(path.evaluate(), slip_phi(0, w0, .5, nu, omega, b), atol=1e-14)
    assert_allclose(path.integral(), slip_phi(1, w0, .5, nu, omega, b), atol=1e-14)
    envelope = np.array([[1., 2.], [.1, .2], [.03, -.02]])
    weighted = path.weighted_integral(envelope)
    assert np.all(np.isfinite(weighted))


@pytest.mark.parametrize('power,rate', [(8, 1000j), (40, 1000j), (40, -.9),
                                      (8, -1000.), (40, -.1 + 30j)])
def test_high_polynomial_moments(power, rate):
    coefficients = np.zeros(power + 1); coefficients[-1] = 1
    path = ExponentialPath([(rate, coefficients)])
    expected = quad(lambda t: (t**power*np.exp(rate*t)).real, 0, 1,
                    epsabs=1e-14, limit=1000)[0]
    assert_allclose(path.integral(), expected, atol=2e-14)


def test_scalar_polynomial_convolution_with_cell_rates():
    target = np.array([0., -1e-8, -.3, -1000., 1000j])
    path = ExponentialPath.polynomial([1.]).convolve(target)
    from phrike.coupled_exponential import phi
    assert_allclose(path.evaluate(), phi(1, target).real, atol=2e-14)


@pytest.mark.parametrize('power,gap', [(8, .5), (8, 1.), (18, .6), (40, 8.), (40, 20.)])
def test_high_power_convolution_avoids_cancelling_poles(power, gap):
    coefficients = np.zeros(power + 1); coefficients[-1] = 1
    path = ExponentialPath.polynomial(coefficients).convolve(-gap)
    expected = quad(lambda s: np.exp(-gap*(1-s))*s**power, 0, 1,
                    epsabs=1e-14)[0]
    assert_allclose(path.evaluate(), expected, atol=3e-14)
