"""Exact homogeneous two-body drag: resolved momentum and heat bookkeeping."""
from common import Report, solver, np, plt
from phrike.problems.problem_list.dusty_box import dust_state


def main():
    report = Report('drag_decay')
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), constrained_layout=True)
    # Local sources are the entire homogeneous problem; bypassing the gas CFL
    # permits the requested arbitrary drag steps without a spatial CFL violation.
    report.notes.append('Uniform source-only boxes: transport is identically zero; '
                        'repaired_cells=roundoff_projections=0 because transport is not invoked. '
                        'HyQMOM is implemented only in 1D; Gaussian also covers 2D and 3D.')
    report.notes.append('The cold Gaussian eps=.01, dt*nu=.01 source sequence rejects '
                        'its own state near a zero crossing of the dust mean speed. '
                        'Affected runs stop immediately; see repro_cold_drag_source.py. '
                        'Passing exact-drag checks apply only to the accepted steps.')
    for closure, ndim in [('gaussian', 1), ('gaussian', 2), ('gaussian', 3), ('hyqmom', 1)]:
        for eps in (.01, 1., 10.):
            for dtnu in (.01, 1., 10., 100.):
                s = solver(ndim=ndim, n=4, closure=closure, eps=eps, ts=1.)
                rho = np.ones(s.shape)
                ug0, ud0 = -.3, .7
                gas_velocity = [ug0*rho] + [np.zeros(s.shape) for _ in range(ndim-1)]
                s.U_gas = s.equations.conservative(rho, *gas_velocity, rho)
                ud = np.zeros((3, *s.shape)); ud[0] = ud0
                s.U_dust = dust_state(s.closure, eps*rho, ud, sigma0=0.)
                momentum0 = ug0 + eps*ud0
                kin0 = .5*(ug0**2 + eps*ud0**2)
                energy0 = s.U_gas[-1] + s.closure.kinetic_energy(s.U_dust)
                internal0 = s.U_gas[-1] - s._kinetic(s.U_gas)
                nsteps = max(1, int(np.ceil(5/((1+eps)*dtnu))))
                times, rel, momentum, heat_error, energy, velocities = [], [], [], [], [], []
                label = f'{closure}/{ndim}D eps={eps:g} dt_nu={dtnu:g}'
                exception = None
                for j in range(nsteps):
                    try:
                        s._local_step(dtnu)
                    except ValueError as exc:
                        # Do not repair a rejected cold state in the validator:
                        # retain the failed run and stop it at the first rejection.
                        exception = str(exc)
                        break
                    t = (j+1)*dtnu
                    p = s.closure.primitives(s.U_dust)
                    ug = s.U_gas[1]/rho
                    exact = (ud0-ug0)*np.exp(-(1+eps)*t)
                    # Absolute error normalised by initial relative velocity
                    # remains meaningful after the exponential underflows.
                    rel.append(float(np.max(np.abs(p['u'][0]-ug-exact)))/(ud0-ug0))
                    velocities.append(float(np.mean(p['u'][0]-ug))/(ud0-ug0))
                    momentum.append(float(np.max(np.abs(ug+eps*p['u'][0]-momentum0))) /
                                    (abs(ug0)+eps*abs(ud0)))
                    lost = kin0 - .5*(ug**2+eps*p['u'][0]**2)
                    heat = s.U_gas[-1]-s._kinetic(s.U_gas)-internal0
                    heat_error.append(float(np.max(np.abs(heat-lost)))/kin0)
                    energy.append(float(np.max(np.abs(s.U_gas[-1] +
                                  s.closure.kinetic_energy(s.U_dust)-energy0)/energy0)))
                    times.append(t)
                report.check(f'{label} completed_steps', len(times), nsteps, op='>=',
                             reason='A realizable exact drag solution must remain accepted.')
                # These are exact algebraic updates, so 1e-12 allows accumulated
                # float64 roundoff over <=500 source steps, not truncation error.
                for name, values in [('velocity', rel), ('momentum', momentum),
                                     ('heat', heat_error), ('energy', energy)]:
                    report.check(f'{label} {name}', max(values, default=float('nan')), 1e-12,
                                 reason='Exact two-body solution; float64 roundoff only.')
                report.runs.append(dict(label=label, closure=closure, ndim=ndim, eps=eps,
                                        dt_nu=dtnu, steps=len(times), requested_steps=nsteps,
                                        exception=exception, repaired_cells=0,
                                        roundoff_projections=0, time=times,
                                        max_velocity_error=max(rel,default=0),
                                        max_heat_error=max(heat_error,default=0)))
                if closure == 'gaussian' and ndim == 1 and dtnu == .01:
                    theta = (1+eps)*np.asarray(times)
                    axes[0].semilogy(theta, velocities, label=f'eps={eps:g}')
                    axes[0].semilogy(theta, np.exp(-theta), 'k--', lw=.6)
                    axes[1].plot(theta, momentum, label=f'eps={eps:g}')
                    axes[2].plot(theta, heat_error, label=f'eps={eps:g}')
    for ax, title, ylabel in zip(axes, ['Relative velocity', 'Momentum conservation', 'Heat balance'],
                                ['$|u_d-u_g|/|u_{d,0}-u_{g,0}|$',
                                 '$|P-P_0|/(|P_{g,0}|+|P_{d,0}|)$',
                                 r'$|Q-\Delta K|/K_0$']):
        ax.set(xlabel='$\\nu(1+\\epsilon)t$', ylabel=ylabel, title=title)
        ax.legend(fontsize=7)
    report.figure(fig, 'drag_decay.png')
    return report.finish()


if __name__ == '__main__':
    raise SystemExit(main())
