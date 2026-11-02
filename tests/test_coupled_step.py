"""Conservative acceptance, analytic feedthrough and transactional recovery."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.linalg import expm

from phrike.coupled_step import _Trial, _Rejected
from phrike.dusty_mhd import (DustyMHDSolver, SGSMHDEquations1D,
                              SGSMHDEquations2D, SGSMHDEquations3D)
from phrike.exponential_path import slip_path
from phrike.grid import Grid1D, Grid2D, Grid3D


def box(ndim=1, *, nu=1., omega=2., dispersion=.1):
    opts = dict(backend='numpy', precision='double', fft_workers=1)
    g = (Grid1D(8, 1., **opts) if ndim == 1 else Grid2D(8, 8, 1., 1., **opts)
         if ndim == 2 else Grid3D(6, 6, 6, 1., 1., 1., **opts))
    eq = (SGSMHDEquations1D, SGSMHDEquations2D, SGSMHDEquations3D)[ndim-1](5/3)
    s = DustyMHDSolver(g, eq, cfl=.3,
        dust_config=dict(coupling_integrator='exponential', charge_to_mass=omega,
                         stopping=dict(model='constant', t_s=1/nu if nu else np.inf)),
        sgs_config=dict(enabled=False))
    one, zero = np.ones(s.shape), np.zeros(s.shape)
    s.U_gas = eq.conservative(one, zero, zero, zero, one, zero, zero, one)
    C = np.eye(3).reshape((3, 3, *([1]*ndim)))*dispersion**2*one
    s.U_dust = s.closure.from_primitives(one, np.zeros((3, *s.shape)), C)
    return s


@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_source_free_raw_conservative_acceptance(ndim):
    s = box(ndim, nu=0., omega=0.)
    x = np.arange(s.shape[-1])/s.shape[-1]
    wave = np.sin(2*np.pi*x)+np.zeros(s.shape)
    p = s.closure.primitives(s.U_dust)
    p['u'][0] = .03*wave
    s.U_dust = s.closure.from_primitives(p['rho']*(1+.1*wave), p['u'], p['C'])
    initial = np.concatenate((s.U_gas, s.U_dust)).sum(axis=tuple(range(1, ndim+1)))
    for _ in range(3):
        s.step(.001)
    final = np.concatenate((s.U_gas, s.U_dust)).sum(axis=tuple(range(1, ndim+1)))
    assert_allclose(final, initial, atol=3e-13, rtol=3e-15)
    assert s.transport_stats['exponential_fallback_substeps'] == 0


def test_averaged_raw_second_moment_retains_mean_rotation():
    s = box(nu=0., omega=np.pi, dispersion=0.)
    s.U_gas[1] = -.5
    p = s.closure.primitives(s.U_dust); p['u'][0] = .5
    s.U_dust = s.closure.from_primitives(p['rho'], p['u'], p['C'])
    trial = _Trial(s, 1., np.zeros(s.shape, bool))
    paths = trial.paths(np.zeros((1, 3, *s.shape)), np.zeros((1, 3, 3, *s.shape)), [trial.coeff0])
    _, dust, _ = trial.averaged_state(trial.x0, paths, 1.)
    assert_allclose(dust[1:4], 0., atol=2e-16)
    assert_allclose(dust[4:6], .125, atol=2e-16)


def test_averaged_path_linear_tableau_equals_candidate_C():
    # Independent dense upper-triangular K oracle. Fast forcing is the
    # ETD-RK3 interpolant; the slow block is B times its analytic integral.
    h, nu, omega, alpha = .2, .4, 1000., 1.7
    L = -alpha*np.array([[nu, -omega, 0], [omega, nu, 0], [0, 0, nu]])
    B = np.array([[.2, -.3, .1], [-.1, .4, .2]])
    K = np.zeros((5, 5)); K[:2, 2:], K[2:, 2:] = B, L
    lift = np.zeros((20, 20)); lift[:5, :5] = h*K
    for j in range(3):
        lift[j*5:(j+1)*5, (j+1)*5:(j+2)*5] = np.eye(5)
    e = expm(lift)
    ph = [e[:5, j*5:(j+1)*5] for j in range(4)]
    w0 = np.array([.3, -.1, .2])
    n0, na, nb = np.array([.1, .2, -.2]), np.array([.3, -.1, .4]), np.array([-.2, .2, .1])
    forcing = np.stack((n0, 4*na-3*n0-nb, 2*(n0-2*na+nb)))
    path = slip_path(w0, h, nu, omega, np.array([0., 0., 1.]), forcing, alpha)
    ns = [np.r_[np.zeros(2), n] for n in (n0, na, nb)]
    expected = ph[0]@np.r_[np.zeros(2), w0]+h*((ph[1]-3*ph[2]+4*ph[3])@ns[0]
        +(4*ph[2]-8*ph[3])@ns[1]+(-ph[2]+4*ph[3])@ns[2])
    assert_allclose(np.r_[h*B@path.integral(), path.evaluate()], expected, atol=2e-14)


@pytest.mark.parametrize('omega', [0., 30.])
def test_epstein_tanh_sech_solution_and_heating(omega):
    errors = []
    for h in (.00625, .003125):
        s = box(nu=1., omega=omega, dispersion=0.)
        s.dust_config['stopping'] = dict(model='epstein', grain_rho_a=np.sqrt(8/np.pi))
        s.U_gas[1] = -1.; s.U_gas[4] += .5
        p = s.closure.primitives(s.U_dust); p['u'][0] = 1.
        s.U_dust = s.closure.from_primitives(p['rho'], p['u'], p['C'])
        energy = s.diagnostics()['E_total']
        for _ in range(round(.05/h)):
            s.step(h)
        q = np.arctanh(np.sqrt(3/5))+2*np.sqrt(5/3)*.05
        exact = np.sqrt(10.)/np.cosh(q)*np.array([np.cos(2*omega*.05), -np.sin(2*omega*.05), 0.])
        slip = s.closure.primitives(s.U_dust)['u']-s.U_gas[1:4]/s.U_gas[0]
        temperature = (s.equations.gamma-1)*s._internal_energy(s.U_gas)/s.U_gas[0]
        errors.append(max(np.max(abs(slip-exact[:, None])), np.max(abs(temperature-(5/3)*np.tanh(q)**2))))
        assert abs(s.diagnostics()['E_total']-energy) < 3e-14
        assert s.ledger['drag_heat_mean'] >= 0
        assert s.transport_stats['exponential_fallback_substeps'] == 0
    assert errors[0]/errors[1] > 3.7


def test_rejection_rolls_back_ledgers_before_resolved_strang(monkeypatch):
    s, reference = box(), box()
    h = .001
    reference._strang_step(h)
    def reject(trial):
        trial.s.ledger['dust_work'] += 99
        trial.s.U_gas[4] += 99
        raise _Rejected('injected acceptance failure', np.ones(trial.s.shape, bool))
    monkeypatch.setattr(_Trial, 'run', reject)
    s.step(h)
    for name in ('U_gas', 'U_dust', 'e_s'):
        assert_array_equal(getattr(s, name), getattr(reference, name))
    assert s.ledger == reference.ledger
    assert s.transport_stats['exponential_rejections'] == 2
    assert s.transport_stats['exponential_fallback_substeps'] == 1
    assert s.t == h


def test_failed_fallback_restores_entire_step(monkeypatch):
    s = box()
    initial = [s.U_gas.copy(), s.U_dust.copy(), s.e_s.copy(), dict(s.ledger), dict(s.transport_stats)]
    def reject(trial):
        raise _Rejected('injected failure', np.ones(trial.s.shape, bool))
    def stop(h):
        s.ledger['dust_work'] = 99
        raise RuntimeError('injected fallback failure')
    monkeypatch.setattr(_Trial, 'run', reject)
    monkeypatch.setattr(s, '_strang_step', stop)
    with pytest.raises(RuntimeError, match='injected fallback'):
        s.step(.001)
    for actual, expected in zip((s.U_gas, s.U_dust, s.e_s), initial):
        assert_array_equal(actual, expected)
    assert s.ledger == initial[3] and s.transport_stats == initial[4]
    assert s.t == s.gas_solver.t == 0
