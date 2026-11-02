"""128-square charged Orszag-Tang, t=0.5: conservation and loading response.

Standard transonic OT gas (gamma=5/3), initially co-moving Gaussian dust,
sigma_d=0.05, charge_to_mass=10, t_s=0.1, e_s=0.01. Dust and SGS diffusion,
backreaction, SGS pressure, decay and kinetic-filter production are enabled.
Gas and dust use CFL=0.8; each dust stage checks its kinetic CFL bound.
An order-two spectral filter (cutoff rate 500, onset zero) regularizes the gas
shocks. This is a conservation/realizability qualification of that explicitly
filtered problem, not a shock-resolution or ideal-MHD accuracy claim. Magnetic
filter loss stays in gas total energy as heat; kinetic loss feeds SGS.

PYTHONPATH=. python scripts/dust_validation/dusty_orszag_tang.py
"""
from common import Report, mhd_solver, np, plt
from concurrent.futures import ProcessPoolExecutor, as_completed
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d
from phrike.problems.problem_list.dusty_box import dust_state

FILTER = dict(enabled=True, p=2, rate=500., onset_fraction=0.)


def initial(eps, n=128):
    s = mhd_solver(2, n, eps=eps, ts=.1, charge=10., sgs_on=True,
                   filtering=FILTER, cfl=.8,
                   dust_config={'eddy_diffusion': {'enabled': True, 'c_kappa': 1.}},
                   sgs_config={'production': 'filter_drain', 'decay': True,
                               'diffusion': {'enabled': True, 'c_kappa': 1.}})
    s.equations.gamma = 5/3
    s.U_gas = orszag_tang_2d(*s.grid.xy_mesh(), gamma=5/3)
    s.U_dust = dust_state(s.closure, eps*s.U_gas[0], s.U_gas[1:4]/s.U_gas[0], .05)
    s.e_s[:] = .01
    return s


def measured(s):
    d = s.diagnostics()
    d['E_resolved'] = s._integral(s._kinetic(s.U_gas))
    d['unrealizable_cells'] = int(np.count_nonzero(~s.closure.realizable(s.U_dust)))
    return d


def run(eps):
    s = initial(eps)
    history = [measured(s)]
    while s.t < .5-1e-14:
        s._set_pressure()
        s.step(min(s.gas_solver._stability_limit(s.U_gas), .5-s.t))
        history.append(measured(s))
    h = {key: np.array([row[key] for row in history]) for key in history[0]}
    return eps, h, s.U_gas[0]


def main():
    r = Report('dusty_orszag_tang')
    r.worker_count = 2
    r.finish(complete=False)
    outcomes = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    with ProcessPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, eps) for eps in (.01, 1.)]
        for future in as_completed(futures):
            eps, h, final_density = future.result()
            ie = (.01, 1.).index(eps)
            drift = float(np.max(np.abs(h['E_total']/h['E_total'][0]-1)))
            predicted = h['E_sgs'][0]+h['production']-h['sgs_decay']-h['dust_work']+h['compression_to_sgs']
            ledger = float(np.max(np.abs(h['E_sgs']-predicted))/h['E_total'][0])
            name = f'128x128 eps={eps:g}'
            r.check(name+'/reached t=0.5', h['time'][-1], .5, op='==')
            r.check(name+'/total energy drift', drift, 1e-11)
            r.check(name+'/SGS ledger closure', ledger, 1e-11)
            r.check(name+'/repaired_cells', h['repaired_cells'][-1], 0, op='==')
            r.check(name+'/gas positivity cells', h['positivity_cells'][-1], 0, op='==')
            r.check(name+'/unrealizable_cells', h['unrealizable_cells'].max(), 0, op='==')
            # Spectral projection error grows with k_max; 1e-12 remains at the
            # float64 reduction scale for 128-square fields of magnitude ~0.1.
            r.check(name+'/div B maximum', h['div_b_max'].max(), 1e-12)
            r.runs.append(dict(case=name, n=128, eps=eps, charge_to_mass=10.,
                               stopping_time=.1, gas_cfl=.8, dust_cfl=.8,
                               spectral_dissipation=FILTER, history=h,
                               energy_drift=drift, SGS_ledger_error=ledger))
            r.finish(complete=False)
            outcomes[eps] = h
            for key in ('E_resolved', 'E_sgs', 'E_dust_mean', 'E_dust_random', 'E_magnetic'):
                axes[ie, 0].plot(h['time'], h[key], label=key)
            axes[ie, 0].set(xlabel='t', ylabel='volume-integrated energy', title=f'eps={eps:g}')
            axes[ie, 0].legend(fontsize=7)
            axes[ie, 1].plot(h['time'], h['E_total']/h['E_total'][0]-1, label='total energy')
            axes[ie, 1].plot(h['time'], (h['E_sgs']-predicted)/h['E_total'][0], label='SGS ledger')
            axes[ie, 1].set(xlabel='t', ylabel='error / initial total energy')
            axes[ie, 1].legend(fontsize=7)
            image = axes[ie, 2].imshow(final_density, origin='lower', extent=(0, 1, 0, 1))
            axes[ie, 2].set(title=f"gas density t={h['time'][-1]:g}", xlabel='x', ylabel='y')
            fig.colorbar(image, ax=axes[ie, 2])
    r.figure(fig, 'dusty_orszag_tang.png')
    # This nonlinear comparison is reported, not assigned an assumed sign:
    # dust inertia can alter cascade production, compression and dissipation.
    r.runs.append(dict(case='loading comparison at t=0.5',
                       heavy_to_light={key: float(outcomes[1.][key][-1]/outcomes[.01][key][-1])
                                       for key in ('E_resolved', 'E_sgs', 'E_magnetic')},
                       dust_work_per_mass={str(eps): float(h['dust_work'][-1]/h['mass_dust'][-1])
                                           for eps, h in outcomes.items()}))
    r.notes.append('The spectral filter is explicitly p=2, cutoff rate=500, onset=0. '
                   'Loading response is descriptive: no universal monotonicity is assumed '
                   'for nonlinear OT. Full histories retain filter/SGS/drag energy ledgers, '
                   'roundoff projections and positivity counters.')
    return r.finish()


if __name__ == '__main__':
    raise SystemExit(main())
