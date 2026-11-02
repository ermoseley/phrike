"""Homogeneous SGS/OU budgets against an independent continuous box ODE."""
from phrike import dust_sources
from common import Report, solver, np, plt
from scipy.integrate import solve_ivp, quad
from phrike.problems.problem_list.dusty_box import dust_state
from phrike.dust_moments import ou_hyqmom, ou_gaussian


def reference(nu, delta, times, eps=.1):
    """Radau evolves es, per-axis variance, relative mean speed, gas heat."""
    def rhs(t, y):
        es, variance, relative, _ = y
        es = max(es, 0.)
        sigma = np.sqrt(2*es/3)
        D = sigma**3/delta*np.arctan(nu*delta/sigma) if sigma else 0.
        decay = es**1.5/delta
        return [-3*eps*D-decay, 2*D-2*nu*variance,
                -nu*(1+eps)*relative,
                3*eps*nu*variance+decay+eps*nu*relative**2]
    ref = solve_ivp(rhs, (0, times[-1]), [1.5, .04, 1., 0.],
                    method='Radau', t_eval=times, rtol=2e-11, atol=1e-13)
    if not ref.success:
        raise RuntimeError(ref.message)
    return ref.y


def main():
    report = Report('ou_box')
    report.notes.extend([
        'Uniform source-only boxes; no spatial transport is invoked. Each source duration '
        'is dt, not two half steps. Gaussian supports 1D/2D/3D; HyQMOM only 1D.',
        'ODE errors are max-in-time absolute errors divided by the maximum absolute '
        'reference value of that field. A 1% science-accuracy target applies at both step '
        'sizes; coarse reservoir updates need not meet it despite exact frozen-coefficient OU.',
        'The source uses ETD2 time-linear forcing. Its large-step map is compared '
        'with independent scalar quadrature of that forcing; constant-forcing AP '
        'terminal balance is tested separately. Instantaneous D(es), step-mean D_eff, '
        'and the ETD2 endpoint forcing differ when the finite SGS reservoir drains.',
        'Source budget freeze at x=.01, dt*nu=10 is reproduced in '
        'scripts/dust_validation/repro_ou_budget_freeze.py. That affected run stops '
        'after its first source step; the ODE discrepancy remains a failure.'
    ])
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), constrained_layout=True)
    refs = {}
    for closure, ndim in [('gaussian', 1), ('gaussian', 2), ('gaussian', 3), ('hyqmom', 1)]:
        for x in (.01, 1., 100.):
            for dtnu in (.01, 10.):
                s = solver(ndim=ndim, n=4, closure=closure, eps=.1, sgs_on=True,
                           sgs_config=dict(decay=True, c_eps=1.))
                nu = x/s.Delta  # es0=3/2 -> sigma_Delta=1, tau_Delta=Delta.
                s.dust_config['stopping']['t_s'] = 1/nu
                rho = np.ones(s.shape)
                s.U_gas = s.equations.conservative(rho, -.3*rho,
                              *[np.zeros(s.shape) for _ in range(ndim-1)], rho)
                u = np.zeros((3, *s.shape)); u[0] = .7
                s.U_dust = dust_state(s.closure, .1*rho, u, sigma0=.2)
                s.e_s[:] = 1.5
                internal0 = s.U_gas[-1]-s._kinetic(s.U_gas)
                E0 = s.U_gas[-1]+s.e_s+s.closure.kinetic_energy(s.U_dust)
                nsteps = int(round(10/dtnu))
                dt = dtnu/nu
                times = np.arange(nsteps+1)*dt
                key = (x, dtnu)
                if key not in refs:
                    refs[key] = reference(nu, s.Delta, times)
                ref = refs[key]
                values = [[1.5, .04, 1., 0.]]
                energy = [0.]
                ap_error, etd2_checked = 0., False
                for j in range(nsteps):
                    previous = s.closure.primitives(s.U_dust)
                    old_v = (previous['C'][0, 0] if closure == 'gaussian' else previous['v']).copy()
                    old_es = float(np.mean(s.e_s))
                    old_L = s.ledger['dust_work']
                    s._local_step(dt)
                    p = s.closure.primitives(s.U_dust)
                    variance = p['C'][0, 0] if closure == 'gaussian' else p['v']
                    heat = s.U_gas[-1]-s._kinetic(s.U_gas)-internal0
                    relative = p['u'][0]-s.U_gas[1]/rho
                    values.append([float(np.mean(s.e_s)), float(np.mean(variance)),
                                   float(np.mean(relative)), float(np.mean(heat))])
                    energy.append(float(np.max(np.abs(s.U_gas[-1]+s.e_s+
                                  s.closure.kinetic_energy(s.U_dust)-E0)/E0)))
                    # The one-step ETD2 formula describes the source only when it
                    # is not sub-cycled: reservoir loss rate * dt <= _MAX_RATE_STEP.
                    sig0 = np.sqrt(2*old_es/3)
                    rate0 = ((3*.1*sig0**3/s.Delta*np.arctan(nu*s.Delta/sig0))/old_es
                             + np.sqrt(old_es)/s.Delta) if old_es > 0 else 0.
                    single = rate0*dt <= dust_sources._MAX_RATE_STEP
                    if dtnu == 10 and single:
                        etd2_checked = True
                        # Check the documented time-linear forcing against an
                        # independent integral rather than reusing phi/psi.
                        D_eff = (s.ledger['dust_work']-old_L)/(3*.1*dt)
                        def independent_D(es):
                            sig = np.sqrt(2*es/3)
                            return sig**3/s.Delta*np.arctan(nu*s.Delta/sig) if sig else 0.
                        slope = np.clip((independent_D(float(np.mean(s.e_s)))-
                                        independent_D(old_es))/dt, -2*D_eff/dt, 2*D_eff/dt)
                        Da = D_eff-.5*slope*dt
                        sigma = quad(lambda z: 2*(Da+slope*z)*np.exp(-2*nu*(dt-z)),
                                     0,dt,epsabs=1e-13,epsrel=1e-12)[0]
                        target = old_v*np.exp(-2*nu*dt)+sigma
                        ap_error = max(ap_error, float(np.max(np.abs(variance-target)/
                                       np.maximum(old_v, D_eff/nu))))
                values = np.asarray(values).T
                errors = np.max(np.abs(values-ref), axis=1)/np.maximum(np.max(np.abs(ref), axis=1), 1e-14)
                label = f'{closure}/{ndim}D x={x:g} dt_nu={dtnu:g}'
                # 1% is an explicit scientific accuracy requirement for the full
                # continuous ODE; it is deliberately not relaxed for large dt.
                for name, error in zip(('e_s', 'variance', 'drag', 'gas_heat'), errors):
                    report.check(f'{label} ODE {name}', error, .01,
                                 reason='1% continuous homogeneous-box accuracy target.')
                report.check(f'{label} energy', max(energy), 1e-12,
                             reason='Closed three-reservoir source exchange; roundoff only.')
                if dtnu == 10 and etd2_checked:
                    report.check(f'{label} ETD2 map', ap_error, 1e-12,
                                 reason='Exact linear-forcing convolution, independent quadrature reference.')
                report.runs.append(dict(label=label, ndim=ndim, closure=closure, x=x,
                       dt_nu=dtnu, steps=nsteps, repaired_cells=0, roundoff_projections=0,
                       max_relative_ODE_error=dict(zip(('e_s','variance','drag','gas_heat'), errors)),
                       max_energy_drift=max(energy), ETD2_map_error=ap_error if dtnu==10 else None,
                       final_state=values[:,-1], final_reference=ref[:,-1]))
                if closure == 'gaussian' and ndim == 1:
                    col = (.01, 1., 100.).index(x)
                    theta = times*nu
                    style = '-' if dtnu == .01 else 'o'
                    axes[0,col].plot(theta, values[0]/1.5, style, label=f'dt nu={dtnu:g}')
                    axes[1,col].plot(theta, values[1], style, label=f'dt nu={dtnu:g}')
                    axes[2,col].plot(theta, values[3], style, label=f'dt nu={dtnu:g}')
                    if dtnu == .01:
                        axes[0,col].plot(theta, ref[0]/1.5, '--', label='Radau')
                        axes[1,col].plot(theta, ref[1], '--', label='Radau')
                        axes[2,col].plot(theta, ref[3], '--', label='Radau')
                    axes[0,col].set_title(f'x=nu tau_Delta={x:g}')
    for j in range(3):
        axes[0,j].set(xlabel='$\\nu t$', ylabel='$e_s/e_{s,0}$')
        axes[1,j].set(xlabel='$\\nu t$', ylabel='$C_{xx}$ [velocity$^2$]')
        axes[2,j].set(xlabel='$\\nu t$', ylabel='$Q_g$ [energy density]')
        for row in range(3): axes[row,j].legend(fontsize=7)
    report.figure(fig, 'ou_box.png')
    gaussianisation(report)
    return report.finish()


def gaussianisation(report):
    # Constant D isolates the exact OU moment map from the separate SGS-budget
    # discretisation. Initial S4=5 exceeds the Hankel bound 1+S3^2=2.
    nu, D, v0, q0, r0 = 1., .4, 1., 1., 5.
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), constrained_layout=True)
    for dt in (.01, 10.):
        t = np.arange(int(round(20/dt))+1)*dt
        v, q, r, sp = np.array([v0]), np.array([q0]), np.array([r0]), np.array([v0])
        actual = [[v0, q0, r0]]
        for _ in t[1:]:
            A = np.exp(-nu*dt)
            Sigma = D/nu*(-np.expm1(-2*nu*dt))
            v, q, r, sp = ou_hyqmom(v, q, r, sp, A, Sigma)
            actual.append([v[0], q[0], r[0]])
        actual = np.asarray(actual).T
        A = np.exp(-nu*t); Sigma = D/nu*(-np.expm1(-2*nu*t))
        exact = np.array([A*A*v0+Sigma, A**3*q0,
                          A**4*r0+6*A*A*Sigma*v0+3*Sigma**2])
        S3 = actual[1]/actual[0]**1.5; S4 = actual[2]/actual[0]**2
        exS3 = exact[1]/exact[0]**1.5; exS4 = exact[2]/exact[0]**2
        # OU is analytic here. Absolute standardized-moment error avoids an
        # undefined relative error when S3 exponentially tends to zero.
        for name, a, b in [('v',actual[0],exact[0]), ('q',actual[1],exact[1]),
                           ('r',actual[2],exact[2]), ('S3',S3,exS3), ('S4',S4,exS4)]:
            report.check(f'gaussianisation dt_nu={dt:g} {name}',
                         np.max(np.abs(a-b))/max(np.max(np.abs(b)),1.), 1e-12,
                         reason='Exact constant-coefficient OU map, float64 roundoff only.')
        report.check(f'gaussianisation dt_nu={dt:g} terminal_S4', abs(S4[-1]-3), 1e-12,
                     reason='At nu*t=20 the non-Gaussian fourth cumulant decays as exp(-80).')
        report.runs.append(dict(label=f'gaussianisation dt_nu={dt:g}', closure='hyqmom', ndim=1,
                                repaired_cells=0, roundoff_projections=0, steps=len(t)-1,
                                final_S3=S3[-1], final_S4=S4[-1]))
        axes[0].plot(t,S3,'-' if dt<1 else 'o',label=f'dt nu={dt:g}')
        axes[1].plot(t,S4,'-' if dt<1 else 'o',label=f'dt nu={dt:g}')
        if dt < 1:
            axes[0].plot(t,exS3,'k--',lw=.7,label='exact OU')
            axes[1].plot(t,exS4,'k--',lw=.7,label='exact OU')
    # With a fixed reservoir/constant forcing the AP balance has a precise
    # exp(-20) residual after one step, independent of closure or spatial grid.
    A = np.exp(-10.); Sigma = D/nu*(-np.expm1(-20.))
    for closure in ('gaussian','hyqmom'):
        if closure == 'gaussian':
            value = ou_gaussian(np.eye(3)[:,:,None]*v0, np.array([A]),
                                np.array([Sigma]))[0,0,0]
        else:
            value = ou_hyqmom(np.array([v0]),np.array([q0]),np.array([r0]),
                             np.array([v0]),np.array([A]),np.array([Sigma]))[0][0]
        report.check(f'{closure} constant_D AP dt_nu=10',
                     abs(value-D/nu)/max(v0,D/nu), 3e-9,
                     reason='Frozen-forcing terminal balance: residual <=exp(-20)=2.06e-9.')
        report.runs.append(dict(label=f'{closure} constant_D AP dt_nu=10',
                                closure=closure, ndim=1, repaired_cells=0,
                                roundoff_projections=0, steps=1, variance=value, terminal=D/nu))
    for ax, ylabel in zip(axes,('$S_3=q/v^{3/2}$','$S_4=r/v^2$')):
        ax.set(xlabel='$\\nu t$',ylabel=ylabel); ax.legend(fontsize=8)
    axes[1].axhline(3,color='k',ls='--',lw=.7)
    report.figure(fig, 'gaussianisation.png')


if __name__ == '__main__':
    raise SystemExit(main())
