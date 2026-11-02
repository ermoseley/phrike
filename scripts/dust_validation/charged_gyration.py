"""Exact two-body gyration and MT25 magnetized OU growth in homogeneous boxes.

Source-only steps are the complete uniform-box problem and allow arbitrary nu*h
and Omega*h without imposing a spatial CFL limit. MT25 uses the production
contract section 3, boxed infinite-inertial-range coefficients. A held e_s=0.06
reservoir (production none, decay off, loading 1e-9) isolates per-direction OU
heating; its small physical depletion is retained in the error budget.

PYTHONPATH=. python scripts/dust_validation/charged_gyration.py
"""
from common import Report, mhd_solver, np, plt
from phrike.problems.problem_list.dusty_box import dust_state

E_S = 0.06
B_OBLIQUE = np.array([1., 2., -3.])/np.sqrt(14.)


def coefficients(x, y):
    return np.arctan(x), .5*(np.pi - np.arctan2(1-y, x) - np.arctan2(1+y, x))


def variances(s, b):
    p = s.closure.primitives(s.U_dust)
    if s.closure.name == 'hyqmom':
        return np.array([np.mean(p['v']), np.mean(p['sperp2'])])
    C = np.mean(p['C'], axis=tuple(range(2, s.ndim+2)))
    parallel = b @ C @ b
    return np.array([parallel, (np.trace(C)-parallel)/2])


def ou_run(x, y, closure, ndim):
    b = np.array([1., 0., 0.]) if closure == 'hyqmom' else B_OBLIQUE
    s = mhd_solver(ndim, 4, closure=closure, eps=1e-9, sgs_on=True, field=b,
                   sgs_config={'pressure': False})
    sigma = np.sqrt(2*E_S/3)
    eD, tau = sigma**3/s.Delta, s.Delta/sigma
    nu, omega = x/tau, y/tau
    s.dust_config['charge_to_mass'] = omega
    s.dust_config['stopping']['t_s'] = 1/nu
    s.e_s[:] = E_S
    D = eD*np.array(coefficients(x, y))
    s.U_dust = dust_state(s.closure, 1e-9*np.ones(s.shape),
                          np.zeros((3, *s.shape)), 0.05*np.sqrt(D[0]/nu))
    dt, steps = 3/nu/100, 100
    history = [variances(s, b)]
    for _ in range(steps):
        s._local_step(dt)
        history.append(variances(s, b))
    times = np.arange(steps+1)*dt
    history = np.array(history)
    decay = np.exp(-2*nu*times)[:, None]
    exact = history[0]*decay + D/nu*(1-decay)
    rates = nu*(history[1]-history[0]*decay[1])/(1-decay[1])/eD
    error = np.max(np.abs(history-exact)/(D/nu), axis=0)
    return times*nu, history/(D/nu), exact/(D/nu), rates, error, s


def main():
    r = Report('charged_gyration')
    slip0, ug0 = np.array([.7, -.2, .4]), np.array([-.1, .15, -.05])
    for closure, ndim in (('gaussian', 1), ('gaussian', 2), ('gaussian', 3), ('hyqmom', 1)):
        b = np.array([1., 0., 0.]) if closure == 'hyqmom' else B_OBLIQUE
        for eps in (.01, 1.):
            for backreaction in (False, True):
                worst = dict(slip=0., energy=0., momentum=0.)
                for nu in (0., 1e-4, .01, 1., 100., 1e4):
                    for omega in (-1e4, -1., 1e-4, 1., 1e4):
                        s = mhd_solver(ndim, 4, closure=closure, eps=eps,
                                       ts=np.inf if nu == 0 else 1/nu, charge=omega,
                                       field=b, dust_config={'backreaction': backreaction})
                        one = np.ones(s.shape)
                        s.U_gas = s.equations.conservative(one, *(v*one for v in ug0), one,
                                                          *(v*one for v in b))
                        u = np.broadcast_to((ug0+slip0).reshape((3,)+(1,)*ndim), (3, *s.shape))
                        s.U_dust = dust_state(s.closure, eps*one, u, .03)
                        e0 = s.U_gas[4] + s.closure.kinetic_energy(s.U_dust)
                        factor = 1+eps if backreaction else 1.
                        par = b*np.dot(b, slip0)
                        perp = slip0-par
                        theta = factor*omega
                        exact = np.exp(-factor*nu)*(par + np.cos(theta)*perp
                                                   + np.sin(theta)*np.cross(perp, b))
                        s._local_step(1.)
                        ud = s.closure.primitives(s.U_dust)['u']
                        ug = s.U_gas[1:4]/s.U_gas[0]
                        worst['slip'] = max(worst['slip'], float(np.max(np.abs(
                            ud-ug-exact.reshape((3,)+(1,)*ndim)))))
                        if backreaction:
                            energy = s.U_gas[4]+s.closure.kinetic_energy(s.U_dust)
                            momentum = ug+eps*ud
                            p0 = ug0+eps*(ug0+slip0)
                            worst['energy'] = max(worst['energy'], float(np.max(np.abs(energy/e0-1))))
                            worst['momentum'] = max(worst['momentum'], float(np.max(np.abs(
                                momentum-p0.reshape((3,)+(1,)*ndim)))))
                name = f'{closure}/{ndim}D eps={eps:g} backreaction={backreaction}'
                for key in ('slip', 'energy', 'momentum') if backreaction else ('slip',):
                    r.check(name+'/'+key, worst[key], 1e-12)
                r.runs.append(dict(case=name, worst_errors=worst,
                                   nu_h=[0., 1e-4, .01, 1., 100., 1e4],
                                   omega_h=[-1e4, -1., 1e-4, 1., 1e4]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for closure, ndim in (('gaussian', 1), ('gaussian', 2), ('gaussian', 3), ('hyqmom', 1)):
        for x in (.1, 10.):
            rates = []
            for y in (0., .5, 2., 20.):
                t, history, exact, rate, error, s = ou_run(x, y, closure, ndim)
                name = f'{closure}/{ndim}D x={x:g} y={y:g}'
                for j, component in enumerate(('parallel', 'perpendicular')):
                    r.check(name+'/'+component+' OU error', error[j], 1e-7)
                    r.check(name+'/'+component+' initial growth error',
                            abs(rate[j]/coefficients(x, y)[j]-1), 1e-9)
                rates.append(rate)
                r.check(name+'/unrealizable_cells', np.count_nonzero(~s.closure.realizable(s.U_dust)), 0)
                r.runs.append(dict(case=name, growth_over_eps_D=rate, OU_error=error))
                if closure == 'gaussian' and ndim == 1 and x == .1 and y == 2.:
                    for j, component in enumerate(('parallel', 'perpendicular')):
                        axes[2].plot(t, history[:, j], label=component)
                        axes[2].plot(t, exact[:, j], 'k:', lw=1)
            rates = np.array(rates)
            if ndim == 1:
                for j in range(2):
                    axes[j].semilogx([.01, .5, 2., 20.], rates[:, j], 'o', mfc='none',
                                     label=f'{closure}, x={x:g}')
    ys = np.logspace(-2, 1.4, 300)
    for x in (.1, 10.):
        for j in range(2):
            axes[j].semilogx(ys, [coefficients(x, y)[j] for y in ys], '-', lw=1)
    for j, component in enumerate(('parallel', 'perpendicular')):
        axes[j].set(xlabel=r'$y=|\Omega|\tau_\Delta$ (zero at 0.01)',
                    ylabel=r'$D_i/\varepsilon_\Delta$', title=component+' growth')
        axes[j].legend(fontsize=7)
    axes[2].set(xlabel=r'$\nu t$', ylabel=r'$\sigma_i^2/(D_i/\nu)$', title='OU saturation (dotted: exact)')
    axes[2].legend(fontsize=8)
    r.figure(fig, 'charged_gyration.png')
    r.notes.append('Gaussian boxes use oblique B=(1,2,-3)/sqrt(14); HyQMOM uses B=xhat. '
                   'Gyrofrequency and drag sweeps are source-only homogeneous problems. '
                   'Fixed-gas runs test slip; conservation checks require backreaction. '
                   'OU window nu*t=3 reaches 99.75% of the stationary variance; '
                   'loading 1e-9 bounds physical reservoir depletion within the 1e-7 comparison.')
    return r.finish()


if __name__ == '__main__':
    raise SystemExit(main())
