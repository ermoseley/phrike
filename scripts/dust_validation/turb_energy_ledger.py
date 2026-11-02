"""Short, fully coupled turbulence ledger qualification at the requested grids.

PYTHONPATH=. python scripts/dust_validation/turb_energy_ledger.py
The window is 0.08 initial grid-eddy times, not a long-time decay experiment.
"""
from common import Report, solver, np, plt
from phrike.sgs import sgs_coefficients
from phrike.problems.problem_list.dusty_box import dust_state
from phrike.problems.problem_list.dusty_turb2d import turbulent_velocity_2d
from phrike.problems.problem_list.turb3d import turbulent_velocity_3d


def initial(ndim, n, eps, regime):
    sigma_s = 0.1
    tau = 1 / (n * sigma_s)
    ts = tau * dict(short=0.05, intermediate=1., long=20.)[regime]
    s = solver(ndim, n, eps=eps, ts=ts, sgs_on=True, filtering=True, cfl=0.6,
               dust_config={'eddy_diffusion': {'enabled': True, 'c_kappa': 1.}},
               sgs_config={'production': 'filter_drain', 'decay': True,
                           'diffusion': {'enabled': True, 'c_kappa': 1.}})
    if ndim == 2:
        u = turbulent_velocity_2d(*s.grid.xy_mesh(), vrms=0.2, kmin=1, kmax=5, seed=42)
        s.U_gas = s.equations.conservative(np.ones(s.shape), *u, np.ones(s.shape))
    else:
        s.U_gas = turbulent_velocity_3d(*s.grid.xyz_mesh(), vrms=0.2, kmin=1,
                                      kmax=5, alpha=0., seed=42, gamma=1.4)
    s.e_s[:] = 1.5*sigma_s**2
    s.U_dust = dust_state(s.closure, eps*np.ones(s.shape), np.zeros((3, *s.shape)), 0.05)
    return s, tau, ts


def measured(s):
    d = s.diagnostics()
    d['E_resolved'] = s._integral(s._kinetic(s.U_gas))
    coeff = sgs_coefficients(s.e_s, s.U_gas[0], s._nu(), s.Delta)
    d['D_mass'] = s._integral(s.U_dust[0]*coeff['D'])/d['mass_dust']
    p = s.closure.primitives(s.U_dust)
    d['min_covariance_eigenvalue'] = float(np.min(np.linalg.eigvalsh(
        np.moveaxis(p['C'], (0, 1), (-2, -1)))))
    d['unrealizable_cells'] = int(np.count_nonzero(~s.closure.realizable(s.U_dust)))
    return d


def main():
    r = Report('turb_energy_ledger')
    r.notes += ['12 runs: 128² and 32³; all coupling, pressure, production, decay, '
                'backreaction, dust and SGS diffusion enabled. Initial vrms=0.2, '
                'sigma_SGS=0.1, dust sigma=0.05; dust starts at rest.',
                'Window 0.08 tau_Delta: exact ledger and initial loading-response tests; '
                'insufficient for long-time turbulent decay or stationary statistics.',
                'The scalar OU prediction omits conversion of spatially varying mean '
                'dust kinetic energy into cell dispersion by transport. A 10% tolerance '
                'is a short-window accuracy target, not an exact identity. '
                'The measured transport_random_energy_gain = E_random-E_random(0) '
                '-dust_work+drag_heat_random isolates that omitted conversion.']
    outcomes = {}
    for ndim, n in ((2, 128), (3, 32)):
        for regime in ('short', 'intermediate', 'long'):
            for eps in (0.01, 1.):
                name = f'{ndim}d_{regime}_eps{eps:g}'
                s, tau, ts = initial(ndim, n, eps, regime)
                hist = [measured(s)]
                prediction = [hist[0]['dust_var_mean']]
                # Three samples in <0.1 grid-eddy times keep the full-size suite
                # laptop-sized while resolving the variation of mass-weighted D.
                for target in np.linspace(0, 0.08*tau, 4)[1:]:
                    while s.t < target - 1e-14:
                        # Dust, SGS advection and diffusion subcycle themselves;
                        # the outer step need only obey the gas stability bound.
                        s._set_pressure()
                        s.step(min(s.gas_solver._stability_limit(s.U_gas), target-s.t))
                        if s.transport_stats['repaired_cells']:
                            break
                    now = measured(s)
                    h = now['time'] - hist[-1]['time']
                    a = np.exp(-2*h/ts)
                    D = 0.5*(hist[-1]['D_mass']+now['D_mass'])
                    prediction.append(a*prediction[-1]+D*ts*(1-a))
                    hist.append(now)
                    if now['repaired_cells']:
                        break
                h = {key: np.array([row[key] for row in hist]) for key in hist[0]}
                h['transport_random_energy_gain'] = (h['E_dust_random']-h['E_dust_random'][0]
                                                     -h['dust_work']+h['drag_heat_random'])
                e0 = h['E_total'][0]
                drift = float(np.max(np.abs(h['E_total']/e0-1)))
                ledger = (h['E_sgs'][0] + h['production'] - h['sgs_decay']
                          - h['dust_work'] + h['compression_to_sgs'])
                ledger_error = float(np.max(np.abs(ledger-h['E_sgs']))/h['E_sgs'][0])
                ou_error = float(np.max(np.abs(h['dust_var_mean']-prediction)
                                        /np.maximum(h['dust_var_mean'], prediction)))
                # Periodic fluxes conserve raw sums. Coupling exchanges reservoirs
                # pointwise. These tolerances allow accumulated float64 reduction
                # error and do not allow physical energy loss.
                r.check(name+'/energy_drift', drift, 1e-10)
                r.check(name+'/sgs_ledger_error', ledger_error, 1e-10)
                r.check(name+'/repaired_cells', h['repaired_cells'][-1], 0)
                # Projection counts are reported, not hidden as repairs. Interior
                # covariance starts positive, so zero is expected in this experiment.
                r.check(name+'/roundoff_projections', h['roundoff_projections'][-1], 0)
                r.check(name+'/unrealizable_cells', h['unrealizable_cells'].max(), 0)
                # The scalar OU prediction omits variance created when transport
                # mixes a spatially varying mean dust velocity.  Short-t_s dust
                # follows the gas, so that mixing dominates its tiny D/nu and the
                # comparison is not a test of the model there: record it only.
                transport_fraction = float(abs(h['transport_random_energy_gain'][-1])
                                           / max(abs(h['E_dust_random'][-1]), 1e-300))
                r.runs.append(dict(name=name+'/ou_diagnostic', ou_error=ou_error,
                                   transport_random_energy_fraction=transport_fraction))
                if regime != 'short':
                    r.check(name+'/dispersion_OU_relative_error', ou_error, 0.1,
                            reason='Transport mixing is subdominant when t_s is not short.')
                outcomes[ndim, regime, eps] = h
                r.runs.append(dict(name=name, ndim=ndim, n=n, eps=eps, ts=ts,
                                   tau_Delta=tau, history=h, sigma2_prediction=prediction,
                                   repaired_cells=int(h['repaired_cells'][-1]),
                                   roundoff_projections=int(h['roundoff_projections'][-1])))
                fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
                t = h['time']/tau
                for key in ('E_resolved', 'E_sgs', 'E_dust_mean', 'E_dust_random'):
                    ax[0, 0].plot(t, h[key], label=key)
                ax[0, 0].plot(t, h['E_gas']-h['E_gas'][0], '--', label='gas total minus initial')
                ax[0, 0].set_ylabel('Energy / (rho0 L^d velocity_unit²)')
                ax[0, 0].legend(fontsize=6)
                for key in ('production', 'sgs_decay', 'dust_work', 'compression_to_sgs',
                            'drag_heat_mean', 'drag_heat_random', 'filter_kinetic_drop'):
                    ax[0, 1].plot(t, h[key], label=key)
                ax[0, 1].set_ylabel('Cumulative energy transfer / energy_unit')
                ax[0, 1].legend(fontsize=6)
                ax[1, 0].plot(t, h['E_total']/e0-1)
                ax[1, 0].set_ylabel('(E_total-E_initial) / E_initial')
                ax[1, 1].plot(t, h['dust_var_mean'], label='moment field')
                ax[1, 1].plot(t, prediction, '--', label='mass-weighted OU prediction')
                ax[1, 1].set_ylabel('sigma_d² / velocity_unit²')
                ax[1, 1].legend(fontsize=6)
                for a in ax.flat:
                    a.set_xlabel('t / initial tau_Delta')
                fig.suptitle(name)
                r.figure(fig, name+'.png')
            light, heavy = (outcomes[ndim, regime, eps] for eps in (0.01, 1.))
            # Same gas initial state and finite dust slip: increasing dust mass
            # should extract more resolved and SGS energy over this initial window.
            for field in ('E_resolved', 'E_sgs'):
                ratio = heavy[field][-1]/light[field][-1]
                r.check(f'{ndim}d_{regime}/heavy_to_light_{field}', ratio, 1.)
    return r.finish()


if __name__ == '__main__':
    raise SystemExit(main())
