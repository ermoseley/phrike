"""Real nonlinear charged-dust AP qualification; failures remain in metrics.json.

Run: PYTHONPATH=. python scripts/dust_validation/charged_ap.py [--family V1 ...]
V1 compares Fourier amplitudes with study candidate C and the continuum matrix
exponential. Spatial truncation and nonlinear amplitude errors are retained.
V2 uses the actual Epstein law, with grain_rho_a=sqrt(8/pi), so nu=sqrt(T).
V3 distinguishes the real coupled PDE from the prescribed-gas betatron and
curvature harnesses authorized by the architect. Those harnesses have no public
configuration option. GC comparisons are leading-order, finite-frequency tests.
V4 checks each raw conservative component and exercises difficult admissibility.
No successful recovery is counted as an unrepaired AP step.
"""
from __future__ import annotations

import argparse
import traceback

from common import Report, mhd_solver, np, plt
from scipy.linalg import expm
from ap_integrator_study import fourier_system, constant_result
from phrike.coupled_exponential import homogeneous_source
from phrike.problems.problem_list.dusty_box import dust_state


STIFFNESS = (.01, 1., 100., 1000.)
GC_STIFFNESS = (.1, 1., 10., 100., 1000.)
COUNTERS = ('exponential_rejections', 'exponential_mood_cells',
            'exponential_fallback_substeps', 'exponential_clamp_failures')
FALLBACK_BUDGET = 128


class ValidationBudget(RuntimeError):
    """A failed AP trial requested too much resolved recovery for this campaign."""


def make(integrator='exponential', *, nu=0., omega=0., **kwargs):
    config = dict(kwargs.pop('dust_config', {}), coupling_integrator=integrator)
    return mhd_solver(ts=1/nu if nu else np.inf, charge=omega,
                      dust_config=config, **kwargs)


def stats(s):
    return {**s.transport_stats, **{k: s.diagnostics().get(k, 0) for k in COUNTERS}}


def integrals(s):
    gas = np.array([s._integral(x) for x in s.U_gas])
    dust = np.array([s._integral(x) for x in s.U_dust])
    momentum = gas[1:4]+dust[1:4]
    energy = gas[4]+.5*sum(dust[4:7])+s._integral(s.e_s)
    return gas, dust, momentum, energy


def advance(s, h, steps):
    original, calls = s._strang_step, []
    def bounded(dt):
        requested = int(round(h/dt))
        if requested > FALLBACK_BUDGET or len(calls) >= FALLBACK_BUDGET:
            raise ValidationBudget(f'exponential rejection requested {requested} resolved Strang '
                                   f'substeps; validation recovery budget={FALLBACK_BUDGET}')
        calls.append(1)
        return original(dt)
    if s.coupling_integrator == 'exponential':
        s._strang_step = bounded
    try:
        for _ in range(steps):
            s.step(h)
    finally:
        s._strang_step = original
    return s


def amplitude(field, x):
    return 2*np.mean(field*np.exp(-2j*np.pi*x), axis=-1)


def prescribe_gas(s):
    """Private tracer harness also holds gas during resolved Strang recovery."""
    s._prescribed_gas = True
    s._gas_step = lambda dt: np.zeros(s.shape)
    local = s._local_step
    def held_local(dt):
        gas = s.U_gas.copy()
        local(dt)
        s.U_gas = gas
    s._local_step = held_local


def linear_run(kind, nu_h, omega_h, integrator, refinement, *, n=64, uniform=True):
    h0, steps, amp, eps = .002, 4*refinement, 1e-7, 1.
    h = h0/refinement
    nu, omega = nu_h/(h if uniform else h0), omega_h/(h if uniform else h0)
    F, S, T, P, inv, nslow, fields, initial = fourier_system(kind, eps, nu, omega, 2*np.pi)
    initial = initial['slip']
    s = make(integrator, n=n, eps=eps, nu=nu, omega=omega, field=(1., 0., 0.))
    x, one = s.grid.x, np.ones(s.shape)
    wave = np.exp(2j*np.pi*x)
    if kind == 'acoustic':
        rg, rd, ug, ud = np.real(amp*initial[:, None]*wave)
        s.U_gas = s.equations.conservative(1+rg, ug, 0*one, 0*one,
                         1/s.equations.gamma+rg, one, 0*one, 0*one)
        s.U_dust = dust_state(s.closure, eps*(1+rd), np.array([ud, 0*one, 0*one]), 1e-4)
    else:
        # Helicity uy+i*uz has the study's -i*omega slip eigenvalue for B=xhat.
        g, d, b = amp*initial[:, None]*wave
        s.U_gas = s.equations.conservative(one, 0*one, g.real, g.imag,
                                          one, one, b.real, b.imag)
        s.U_dust = dust_state(s.closure, eps*one, np.array([0*one, d.real, d.imag]), 1e-4)
    advance(s, h, steps)
    p, ug = s.closure.primitives(s.U_dust), s.U_gas[1:4]/s.U_gas[0]
    if kind == 'acoustic':
        fields_out = np.array([s.U_gas[0]-1, p['rho']/eps-1, ug[0], p['u'][0]])
        numerical = amplitude(fields_out, x)/amp
    else:
        # A circular complex field already contains one Fourier helicity.
        numerical = np.array([np.mean((ug[1]+1j*ug[2])/wave),
                              np.mean((p['u'][1]+1j*p['u'][2])/wave),
                              np.mean((s.U_gas[6]+1j*s.U_gas[7])/wave)])/amp
    exact = expm(steps*h*F) @ initial
    candidate = inv @ constant_result(S, T, nslow, P @ initial, h, steps, 'C')
    scale = np.linalg.norm(initial)
    return dict(h=h, n=n, error_exact=float(np.linalg.norm(numerical-exact)/scale),
                error_candidate_C=float(np.linalg.norm(numerical-candidate)/scale),
                candidate_C_exact_error=float(np.linalg.norm(candidate-exact)/scale),
                numerical=numerical, exact=exact, candidate_C=candidate, counters=stats(s))


def transported_covariance():
    """Source-free fast-coordinate transport must restrict to candidate C/RK3."""
    from phrike.dust_moments import _transport_rhs
    s = make(n=32, eps=1.)
    x, one, zero, h = s.grid.x, np.ones(s.shape), np.zeros(s.shape), .002
    s.U_gas = s.equations.conservative(one, one, zero, zero, one, one, zero, zero)
    C = np.einsum('ij,...->ij...', np.diag([.01, .1, .01]), one)
    C[1, 1] += 1e-6*np.cos(2*np.pi*x)
    s.U_dust = s.closure.from_primitives(one, np.array([one, zero, zero]), C)
    a0 = amplitude(s.U_dust[5], x)
    rhs = _transport_rhs(s.U_dust, s.closure, s.dxs, np.zeros(s.shape, bool))[0]
    rate = amplitude(rhs[5], x)/a0
    advance(s, h, 1)
    ratio = amplitude(s.U_dust[5], x)/a0
    z = h*rate
    candidate = 1+z+z*z/2+z**3/6
    return dict(h=h, rate=[rate.real, rate.imag], ratio=[ratio.real, ratio.imag],
                candidate_C=[candidate.real, candidate.imag],
                candidate_C_error=float(abs(ratio-candidate)), exact_error=float(abs(ratio-np.exp(z))),
                counters=stats(s))


def v1(r):
    row = transported_covariance()
    r.runs.append(dict(case='V1 source-free advected Cyy strict candidate C identity', **row,
                       reproducer='transported_covariance()'))
    r.check('V1 transported Cyy strict candidate C identity', row['candidate_C_error'], 2e-12,
            reason='With zero source, candidate C is the semidiscrete RK3 polynomial; no spatial-oracle mismatch.')
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for kind in ('acoustic', 'alfven'):
        pairs = [(a, 0.) for a in STIFFNESS] if kind == 'acoustic' else [
            (a, b) for a in STIFFNESS for b in STIFFNESS]
        for nu_h, omega_h in pairs:
            for method in ('exponential', 'strang'):
                name = f'V1 {kind} nu*h={nu_h:g} omega*h={omega_h:g} {method}'
                rows = []
                for q in (1, 2, 4):
                    try:
                        rows.append(linear_run(kind, nu_h, omega_h, method, q))
                    except Exception as exc:
                        rows.append(dict(h=.002/q, n=64, failure=repr(exc)))
                        r.check(name+f' refinement={q} completed', 0, 1, op='==', reason=repr(exc))
                errors = [x.get('error_exact') for x in rows]
                orders = [float(np.log2(a/b)) if a is not None and b is not None and min(a, b)>0
                          else None for a, b in zip(errors[:-1], errors[1:])]
                r.runs.append(dict(case=name, h=[x['h'] for x in rows],
                    errors_exact=errors, uniform_orders=orders,
                    error_over_h2=[e/x['h']**2 if e is not None else None for e, x in zip(errors, rows)],
                    errors_candidate_C=[x.get('error_candidate_C') for x in rows],
                    candidate_C_exact_errors=[x.get('candidate_C_exact_error') for x in rows],
                    counters=[x.get('counters') for x in rows], failures=[x.get('failure') for x in rows],
                    reproducer=f"linear_run({kind!r}, {nu_h!r}, {omega_h!r}, {method!r}, refinement)"))
                if method == 'exponential' and errors[-1] is not None:
                    r.check(name+' candidate C amplitude error', rows[-1]['error_candidate_C'], 2e-3,
                            reason='Includes nonlinear and fixed-grid WENO spatial error; not roundoff identity.')
                    r.check(name+' exact amplitude error', errors[-1], 2e-3)
                    r.check(name+' unresolved fast fallback',
                            sum(x.get('counters', {}).get('exponential_fallback_substeps', 0) for x in rows), 0)
                # Orders are always reported. A spatial floor is not relabeled as temporal convergence.
                if (omega_h == 0 or nu_h == omega_h) and all(e is not None for e in errors):
                    axes[0 if kind == 'acoustic' else 1].loglog(
                        [x['h'] for x in rows], errors, 'o-' if method == 'exponential' else 'x:',
                        label=f'{method}, nu*h={nu_h:g}, omega*h={omega_h:g}')
            r.finish(complete=False)
    for ax, title in zip(axes, ('acoustic', 'charged Alfven')):
        ax.set(xlabel='h', ylabel='Fourier amplitude error / initial norm', title=title)
        ax.grid(alpha=.2)
        ax.legend(fontsize=6)
    r.figure(fig, 'linear_modes.png')
    r.notes.append('V1 scales physical nu and omega as 1/h under uniform h refinement, '
                   'holding nu*h and omega*h fixed as in the study uniform_refinement; '
                   'h0=.002, T=.008, N=64. Every nu*h0/omega*h0 combination is retained. '
                   'Continuum candidate C and exact references include no spatial truncation. '
                   'The unfiltered production nonlinear solver is used without source-only shortcuts. '
                   'Initial isotropic dispersion=1e-4 keeps the smooth linear case inside the moment cone; '
                   'the cold testbed oracle consequently omits O(1e-8) dust-pressure coefficients. '
                   'Cold zero-dispersion states are exercised in V4 instead.')


def epstein_run(h, omega=0., integrator='exponential', end=.2):
    s = make(integrator, n=4, eps=1., omega=omega, field=(0., 0., 1.),
             dust_config={'stopping': {'model': 'epstein', 'grain_rho_a': np.sqrt(8/np.pi)}})
    one = np.ones(s.shape)
    s.equations.gamma = 5/3
    s.U_gas = s.equations.conservative(one, -one, 0*one, 0*one, one, 0*one, 0*one, one)
    s.U_dust = dust_state(s.closure, one, np.array([one, 0*one, 0*one]))
    E0 = integrals(s)[-1]
    steps = round(end/h)
    advance(s, end/steps, steps)
    ug, ud = s.U_gas[1:4]/s.U_gas[0], s.closure.primitives(s.U_dust)['u']
    c, Tinf, alpha = 1/6, 1+4/6, 2.
    a = np.arctanh(np.sqrt(1/Tinf))+alpha*np.sqrt(Tinf)*end
    exact_T, exact_w = Tinf*np.tanh(a)**2, np.sqrt(Tinf/c)/np.cosh(a)
    exact = exact_w*np.array([np.cos(alpha*omega*end), -np.sin(alpha*omega*end), 0.])
    temperature = (s.equations.gamma-1)*s._internal_energy(s.U_gas)/s.U_gas[0]
    heat = s.ledger['drag_heat_mean']+s.ledger['drag_heat_random']
    return dict(h=end/steps, slip_error=float(np.max(np.abs((ud-ug)-exact[:, None]))),
                temperature_error=float(np.max(abs(temperature-exact_T))),
                heat_error=abs(heat-1.5*(exact_T-1)), physical_drag_heat=heat,
                temperature=float(np.mean(temperature)), exact_temperature=exact_T,
                energy_error=abs(integrals(s)[-1]-E0), counters=stats(s), ledger=dict(s.ledger))


def v2(r):
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for omega in (0., 30.):
        hs = (.05, .025, .0125, .00625) if omega == 0 else (
            .05, .025, .0125, .00625, .003125, .0015625)
        rows = [epstein_run(h, omega) for h in hs]
        for field in ('slip_error', 'temperature_error', 'heat_error'):
            errors = np.array([x[field] for x in rows])
            order = np.log2(errors[-2]/max(errors[-1], 1e-300))
            name = f'V2 Epstein omega={omega:g} {field}'
            r.check(name+' finest order', order, 1.8, op='>=')
            ax.loglog([x['h'] for x in rows], errors, 'o-', label=name.removeprefix('V2 '))
        r.check(f'V2 omega={omega:g} energy', max(x['energy_error'] for x in rows), 2e-12)
        r.check(f'V2 omega={omega:g} minimum physical drag heat',
                min(x['physical_drag_heat'] for x in rows), 0., op='>=')
        r.check(f'V2 omega={omega:g} finest-pair fast fallback',
                sum(x['counters'].get('exponential_fallback_substeps', 0) for x in rows[-2:]), 0)
        r.runs.append(dict(case=f'V2 Epstein omega={omega:g}', results=rows,
                           reproducer=f'epstein_run(h, {omega!r})'))
    ax.set(xlabel='h', ylabel='absolute error', title='Nonlinear drag and exact heating')
    ax.legend(fontsize=7)
    ax.grid(alpha=.2)
    r.figure(fig, 'epstein.png')


def exb_run(omega_h, method='exponential', *, h=.001, n=64):
    omega, eps, end = omega_h/h, 1e-9, .008
    s = make(method, n=n, eps=eps, omega=omega, field=(0., 0., 1.))
    one, x, ug0 = np.ones(s.shape), s.grid.x, np.array([.1, -.03, 0.])
    s.U_gas = s.equations.conservative(one, *(u*one for u in ug0), one, 0*one, 0*one, one)
    s.U_dust = dust_state(s.closure, eps*(1+.1*np.cos(2*np.pi*x)), np.zeros((3, *s.shape)))
    before = amplitude(s.U_dust[0], x)
    advance(s, h, round(end/h))
    # Weak-loading tracer reference includes the initial gyroradius displacement.
    path = homogeneous_source(ug0, np.zeros(3), end, rho_g=1., rho_d=0., nu=0.,
                              omega=omega, bhat=np.array([0., 0., 1.]), C=np.zeros((3, 3)))
    displacement = ug0*end+path['integrated_slip']
    ratio = amplitude(s.U_dust[0], x)/before
    reference = np.exp(-2j*np.pi*displacement[0])
    return dict(error=float(abs(ratio-reference)), density_amplitude_ratio=float(abs(ratio)),
                phase=float(np.angle(ratio)), exact_phase=float(np.angle(reference)), counters=stats(s),
                observable=np.array([ratio.real, ratio.imag]))


def force_run(omega_h, method='exponential', *, nu_h=.3, h=.001):
    omega, nu, eps, end = omega_h/h, nu_h/h, .1, .008
    s = make(method, n=8, eps=eps, nu=nu, omega=omega, field=(0., 0., 1.))
    force = np.array([.2, -.1, 0.])
    rhs = s.gas_solver._compute_rhs
    def forced(U):
        out = rhs(U)
        out[1:4] += U[0]*force[:, None]
        out[4] += np.sum(U[1:4]*force[:, None], axis=0)
        return out
    s.gas_solver._compute_rhs = forced
    advance(s, h, round(end/h))
    ref = homogeneous_source(np.zeros(3), np.zeros(3), end, rho_g=1., rho_d=eps,
        nu=nu, omega=omega, bhat=np.array([0., 0., 1.]), C=np.zeros((3, 3)),
        d0=-force, a0=force/(1+eps))
    p = s.closure.primitives(s.U_dust)
    slip = np.mean(p['u']-s.U_gas[1:4]/s.U_gas[0], axis=-1)
    terminal = (-nu*force+omega*np.cross(np.array([0., 0., 1.]), force))/(
                (1+eps)*(nu*nu+omega*omega))
    return dict(error=float(np.linalg.norm(slip-ref['w'])), terminal_error=float(np.linalg.norm(slip-terminal)),
                slip=slip, exact_slip=ref['w'], terminal_slip=terminal, counters=stats(s), observable=slip)


def diamagnetic_run(omega_h, method='exponential', *, h=.001, n=32):
    omega, eps, end, c = omega_h/h, 1e-7, .004, .01
    s = make(method, ndim=2, n=n, eps=eps, omega=omega, field=(0., 0., 1.))
    x, y = np.meshgrid(s.grid.x, s.grid.y)
    rho = eps*(1+.1*np.cos(2*np.pi*x))
    u = np.zeros((3, *s.shape))
    # Initially zero mean: the gyro-averaged pressure response develops from rest.
    C = np.einsum('ij,...->ij...', np.diag([c, c, .002]), np.ones(s.shape))
    s.U_dust = s.closure.from_primitives(rho, u, C)
    initial = s.U_dust.copy()
    advance(s, h, round(end/h))
    p = s.closure.primitives(s.U_dust)
    drift = -2*np.pi*.1*c*np.sin(2*np.pi*x)/(omega*(1+.1*np.cos(2*np.pi*x)))
    # Compare the exact finite-time uniform-B pressure response, then leading GC.
    phase = omega*end
    finite = drift*(1-np.cos(phase))
    return dict(density_change=float(np.max(abs(s.U_dust[0]-rho))/eps),
                momentum_drift_error=float(np.sqrt(np.mean((p['u'][1]-finite)**2))),
                leading_gc_drift_rms=float(np.sqrt(np.mean(drift**2))),
                dust_mass_error=abs(s._integral(s.U_dust[0]-initial[0])), counters=stats(s),
                observable=p['u'])


def betatron_run(omega_h, method='exponential', *, h=.001, n=32):
    omega, end, a, c = omega_h/h, .008, .05, .01
    s = make(method, ndim=2, n=n, eps=1e-8, omega=omega, field=(0., 0., 1.),
             dust_config={'backreaction': False})
    prescribe_gas(s)
    x, y = np.meshgrid(s.grid.x, s.grid.y)
    one, zero = np.ones(s.shape), np.zeros(s.shape)
    u = np.array([-a*np.sin(2*np.pi*x), zero, zero])
    C = np.einsum('ij,...->ij...', np.diag([c, c, .002]), one)
    s.U_dust = s.closure.from_primitives(1e-8*one, u.copy(), C)
    for j in range(round(end/h)):
        # Exact periodic induction/density solution of the prescribed compressive flow.
        q = np.exp(2*np.pi*a*(j+.5)*h)
        B = q/(np.cos(np.pi*x)**2+q*q*np.sin(np.pi*x)**2)
        s.U_gas = s.equations.conservative(B, *u, one, zero, zero, B)
        advance(s, h, 1)
    q = np.exp(2*np.pi*a*end)
    B = q/(np.cos(np.pi*x)**2+q*q*np.sin(np.pi*x)**2)
    p = s.closure.primitives(s.U_dust)
    observed = .5*(p['C'][0, 0]+p['C'][1, 1])/c
    return dict(error=float(np.sqrt(np.mean((observed-B)**2))),
                observed_min=float(observed.min()), observed_max=float(observed.max()),
                reference_min=float(B.min()), reference_max=float(B.max()), counters=stats(s),
                observable=observed,
                evidence='prescribed periodic compressive gas and analytically imposed ideal-induction B')


def curvature_run(omega_h, method='exponential', *, h=.001, n=24):
    omega, end, vpar = omega_h/h, .004, .02
    s = make(method, ndim=2, n=n, eps=1e-8, omega=omega,
             dust_config={'backreaction': False})
    prescribe_gas(s)
    x, y = np.meshgrid(s.grid.x, s.grid.y)
    one, zero = np.ones(s.shape), np.zeros(s.shape)
    B = np.array([-np.cos(2*np.pi*x)*np.sin(2*np.pi*y),
                   np.sin(2*np.pi*x)*np.cos(2*np.pi*y), .2*one])
    mag = np.sqrt(np.sum(B*B, axis=0))
    b = B/mag
    curvature = b[0]*s.grid.dx1(b)+b[1]*s.grid.dy1(b)
    drift = vpar*vpar*np.cross(b, curvature, axisa=0, axisb=0, axisc=0)/(omega*mag)
    u = vpar*b+drift
    s.U_gas = s.equations.conservative(one, zero, zero, zero, one, *B)
    s.U_dust = dust_state(s.closure, 1e-8*one, u, 1e-5)
    advance(s, h, round(end/h))
    ud = s.closure.primitives(s.U_dust)['u']
    perpendicular = ud-b*np.sum(b*ud, axis=0)
    return dict(error=float(np.sqrt(np.mean((perpendicular-drift)**2))),
                leading_drift_rms=float(np.sqrt(np.mean(drift*drift))), counters=stats(s),
                observable=perpendicular,
                evidence='prescribed periodic divergence-free cellular field; prepared streaming dust')


def v3(r):
    runs = (('ExB', exb_run, 3e-3), ('PedersenHall', force_run, 2e-5),
            ('polarization', lambda z, method='exponential': force_run(z, method, nu_h=0.), 2e-5),
            ('diamagnetic', diamagnetic_run, None), ('betatron', betatron_run, 2e-3),
            ('curvature', curvature_run, None))
    for name, run, tolerance in runs:
        for strength in GC_STIFFNESS:
            try:
                row = run(strength)
            except Exception as exc:
                r.check(f'V3 {name} omega*h={strength:g} completed', 0, 1, op='==', reason=repr(exc))
                r.runs.append(dict(case=f'V3 {name}', omega_h=strength, failure=repr(exc),
                                   traceback=traceback.format_exc()))
                continue
            observable = row.pop('observable')
            reproducer = (f'force_run({strength!r}, nu_h=0.)' if name == 'polarization' else
                          f'force_run({strength!r})' if name == 'PedersenHall' else
                          f'{name.lower()}_run({strength!r})')
            r.runs.append(dict(case=f'V3 {name}', omega_h=strength, **row,
                              reproducer=reproducer))
            r.check(f'V3 {name} omega*h={strength:g} AP recovery cost',
                    row['counters'].get('exponential_fallback_substeps', 0), 0)
            if tolerance is not None:
                r.check(f'V3 {name} omega*h={strength:g}', row['error'], tolerance)
            if name == 'diamagnetic':
                r.check(f'V3 diamagnetic mass omega*h={strength:g}', row['dust_mass_error'], 2e-20)
                r.check(f'V3 diamagnetic density omega*h={strength:g}', row['density_change'], 3e-3,
                        reason='Includes finite-gyroradius transient and kinetic upwind dissipation.')
                r.check(f'V3 diamagnetic drift omega*h={strength:g}',
                        row['momentum_drift_error']/max(row['leading_gc_drift_rms'], 1e-30), .1,
                        reason='Finite-time pressure response error relative to leading GC drift amplitude.')
            if name == 'curvature':
                r.check(f'V3 curvature leading drift omega*h={strength:g}',
                        row['error']/max(row['leading_drift_rms'], 1e-30), .25,
                        reason='Short-time prepared streaming; coarse spatial error remains included.')
            if strength == .1:
                reference = run(strength, 'strang')
                reference_observable = reference.pop('observable')
                r.runs.append(dict(case=f'V3 {name} resolved Strang comparison', omega_h=strength,
                    difference_linf=float(np.max(abs(observable-reference_observable))),
                    difference_relative_l2=float(np.linalg.norm(observable-reference_observable)/
                                                 max(np.linalg.norm(reference_observable), 1e-30)),
                    exponential=row, strang=reference))
            r.finish(complete=False)
    r.notes.append('V3 ExB measures density-mode displacement including the initial gyroradius. '
        'PedersenHall/polarization impose constant gas acceleration through a test-only RHS wrapper, '
        'including its conservative energy work. Diamagnetic uses the real coupled 2D PDE. '
        'Betatron and curvature hold prescribed gas/field states during each step using the private '
        '_prescribed_gas attribute; these are separate tracer-harness evidence. Betatron analytically '
        'updates B between steps with the periodic ideal induction solution. Curvature and diamagnetic '
        'leading drift amplitudes and finite-frequency errors are descriptive; no absent proof is a pass.')


def conservation_run(ndim=1, sources=False, case='smooth'):
    s = make(n=16 if ndim < 3 else 8, ndim=ndim, eps=.1, nu=200. if sources else 0.,
             omega=300. if sources else 0., sgs_on=(case == 'reservoir'))
    coords = np.meshgrid(*[getattr(s.grid, a) for a in ('x', 'y', 'z')[:ndim]], indexing='xy')
    wave = np.sin(2*np.pi*coords[0])
    one, zero = np.ones(s.shape), np.zeros(s.shape)
    pressure = (1e-7 if case == 'low_thermal' else 1.)*one
    u = np.array([.03*wave, .02*one, -.01*one])
    s.U_gas = s.equations.conservative(1+.03*wave, *u, pressure, one, zero, zero)
    C = np.einsum('ij,...->ij...', np.diag([.01, .02, .03]), one)
    if case == 'cold': C[:] = 0.
    if case == 'near_singular': C[2, 2] = 1e-14
    s.U_dust = s.closure.from_primitives(.1*(1+.02*wave), u+.01, C)
    if case == 'reservoir': s.e_s[:] = 1e-8
    before = integrals(s)
    advance(s, .0002, 3)
    after = integrals(s)
    return dict(gas_errors=after[0]-before[0], raw_dust_errors=after[1]-before[1],
                momentum_error=float(np.max(abs(after[2]-before[2]))), energy_error=abs(after[3]-before[3]),
                div_b=float(s.gas_solver.max_div_b(s.U_gas)),
                minimum_thermal=float(np.min(s._internal_energy(s.U_gas))),
                minimum_reservoir=float(np.min(s.e_s)),
                unrealizable=int(np.count_nonzero(~s.closure.realizable(s.U_dust))),
                counters=stats(s), ledger=dict(s.ledger))


def rollback_run():
    """Poison both trial attempts; compare recovery with untouched Strang."""
    from phrike.coupled_step import _Trial, _Rejected
    import copy
    s = make(n=8, eps=.1, nu=2., omega=3., sgs_on=True)
    s.e_s[:] = .02
    s.U_dust = dust_state(s.closure, .1*np.ones(s.shape),
                          np.full((3, *s.shape), .03), .01)
    reference = copy.deepcopy(s)
    reference.coupling_integrator = 'strang'
    original, calls, h = _Trial.run, [], .01
    def reject(self):
        calls.append(1)
        self.s.U_gas[4] += 7.
        self.s.e_s += .01
        self.s.t += .3
        for key in self.s.ledger:
            self.s.ledger[key] += 123.
        raise _Rejected('injected validation rejection', np.ones(self.s.shape, bool))
    _Trial.run = reject
    try:
        s.step(h)
    finally:
        _Trial.run = original
    nsub = s.transport_stats.get('exponential_fallback_substeps', 0)
    for _ in range(nsub):
        reference._strang_step(h/nsub)
    error = max(float(np.max(abs(getattr(s, key)-getattr(reference, key))))
                for key in ('U_gas', 'U_dust', 'e_s'))
    ledger_error = max(abs(s.ledger.get(k, 0)-reference.ledger.get(k, 0))
                       for k in set(s.ledger)|set(reference.ledger))
    return dict(trial_calls=len(calls), state_error=error, ledger_error=ledger_error,
                time_error=abs(s.t-reference.t), counters=stats(s))


def v4(r):
    for ndim in (1, 2, 3):
        for sources in (False, True):
            row = conservation_run(ndim, sources)
            name = f'V4 {ndim}D sources={sources}'
            r.runs.append(dict(case=name, **row))
            if not sources:
                for j, err in enumerate(row['raw_dust_errors']):
                    r.check(f'{name} dust raw component {j}', abs(err), 2e-13)
                for j, err in enumerate(row['gas_errors']):
                    r.check(f'{name} gas component {j}', abs(err), 2e-13)
            for key in ('momentum_error', 'energy_error', 'div_b'):
                r.check(name+' '+key, row[key], 2e-12)
    for case in ('cold', 'near_singular', 'low_thermal', 'reservoir'):
        row = conservation_run(1, True, case)
        r.runs.append(dict(case='V4 admissibility '+case, **row,
                           reproducer=f'conservation_run(1, True, {case!r})'))
        r.check('V4 '+case+' thermal', row['minimum_thermal'], 0., op='>')
        r.check('V4 '+case+' reservoir', row['minimum_reservoir'], 0., op='>=')
        r.check('V4 '+case+' realizability', row['unrealizable'], 0)
        r.check('V4 '+case+' energy', row['energy_error'], 2e-12)
    row = rollback_run()
    r.runs.append(dict(case='V4 forced rejection and ledger rollback', **row,
                       reproducer='rollback_run()'))
    r.check('V4 rollback two trial attempts', row['trial_calls'], 2, op='==')
    r.check('V4 rollback fallback exercised', row['counters']['exponential_fallback_substeps'], 0, op='>')
    for key in ('state_error', 'ledger_error', 'time_error'):
        r.check('V4 exact rollback '+key, row[key], 0., op='==')
    r.notes.append('V4 records actual rejection, MOOD, fallback, and clamp counters. '
        'The forced-rejection exercise poisons trial state, time, and every ledger before raising; '
        'the accepted fallback is compared with an untouched resolved-Strang reference.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', nargs='+', choices=('V1', 'V2', 'V3', 'V4'),
                        default=['V1', 'V2', 'V3', 'V4'])
    args = parser.parse_args()
    r = Report('charged_ap')
    r.notes.append('Float64, NumPy backend; integrator=exponential unless labeled Strang. '
                   'Each exception is retained with a failed completion check. '
                   f'Test-only recovery budget={FALLBACK_BUDGET} resolved Strang substeps per run; '
                   'budget exhaustion is a failure with no invented numerical solution. The production '
                   'fallback policy is unchanged.')
    for name in args.family:
        try:
            globals()[name.lower()](r)
        except Exception as exc:
            r.check(name+' completed', 0, 1, op='==', reason=repr(exc))
            r.runs.append(dict(case=name, exception=repr(exc), traceback=traceback.format_exc(),
                reproducer=f'PYTHONPATH=. python scripts/dust_validation/charged_ap.py --family {name}'))
            traceback.print_exc()
        r.finish(complete=False)
    r.notes.append('Requested families: '+', '.join(args.family))
    return r.finish()


if __name__ == '__main__':
    raise SystemExit(main())
