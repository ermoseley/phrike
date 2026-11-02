"""Short, sharp-filter charged MHD turbulence and reduced Orszag--Tang.

Default: 128², t=0.005, both coupling integrators, float64. The decaying
random-phase, solenoidal velocity has rms Mach 0.3 and integer modes 2 <= |k|
<= 8; the mean-plus-fluctuating magnetic field has volume-averaged beta 1.
Orszag--Tang velocities and fields are both scaled by 0.5 at fixed rho and p.
Gaussian dust (epsilon=0.1, sigma=0.05, t_s=0.1, q/m=10) starts co-moving;
SGS pressure, filter production, decay, transport and eddy diffusion are on.

The order-eight filter acts only above half the spectral cutoff. These short
runs qualify admissibility, conservation and ledger closure before shocks;
they do not establish a developed cascade or long-time OT robustness. Each
case stops if its wall-time budget is exhausted and records the last state.

PYTHONPATH=. python scripts/dust_validation/dusty_orszag_tang.py
PYTHONPATH=. python scripts/dust_validation/dusty_orszag_tang.py --n 32 --time .005
"""
import argparse
import resource
import sys
import time

from common import Report, mhd_solver, np, plt
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d
from phrike.problems.problem_list.dusty_box import dust_state

FILTER = dict(enabled=True, p=8, rate=500., onset_fraction=.5)
SEED = 4217


def solenoidal(X, Y, rng):
    """Random-phase real Fourier field with no duplicated wavevectors."""
    vector = np.zeros((3, *X.shape))
    for kx in range(-8, 9):
        for ky in range(0, 9):
            k = np.hypot(kx, ky)
            if not 2 <= k <= 8 or (ky == 0 and kx <= 0):
                continue
            wave = np.cos(2*np.pi*(kx*X+ky*Y) + rng.uniform(0., 2*np.pi))/k
            vector[0] -= ky*wave/k
            vector[1] += kx*wave/k
    return vector/np.sqrt(np.mean(np.sum(vector**2, axis=0)))


def initial(case, integrator, n):
    s = mhd_solver(2, n, eps=.1, ts=.1, charge=10., sgs_on=True,
                   filtering=FILTER, cfl=.5,
                   dust_config={'coupling_integrator': integrator,
                                'eddy_diffusion': {'enabled': True, 'c_kappa': 1.}},
                   sgs_config={'production': 'filter_drain', 'decay': True,
                               'diffusion': {'enabled': True, 'c_kappa': 1.}})
    s.equations.gamma = 5/3
    X, Y = s.grid.xy_mesh()
    if case == 'turbulence':
        rng = np.random.default_rng(SEED)
        rho, p = np.ones(s.shape), np.ones(s.shape)
        u = .3*np.sqrt(s.equations.gamma)*solenoidal(X, Y, rng)
        B = .25*solenoidal(X, Y, rng)
        B[2] = 1.
        B *= np.sqrt(2/np.mean(np.sum(B**2, axis=0)))
        s.U_gas = s.equations.conservative(rho, *u, p, *B)
    else:
        U = orszag_tang_2d(X, Y, gamma=s.equations.gamma)
        rho, ux, uy, uz, p, bx, by, bz = s.equations.primitive(U)
        s.U_gas = s.equations.conservative(rho, .5*ux, .5*uy, .5*uz,
                                          p, .5*bx, .5*by, .5*bz)
    s.U_dust = dust_state(s.closure, .1*s.U_gas[0], s.U_gas[1:4]/s.U_gas[0], .05)
    s.e_s[:] = .005
    return s


def measured(s):
    d = s.diagnostics()
    d['E_resolved'] = s._integral(s._kinetic(s.U_gas))
    d['E_thermal'] = s._integral(s._internal_energy(s.U_gas))
    d['unrealizable_cells'] = int(np.count_nonzero(~s.closure.realizable(s.U_dust)))
    d['thermal_min'] = float(np.min(s._internal_energy(s.U_gas)))
    d['gas_density_min'] = float(np.min(s.U_gas[0]))
    d['dust_density_min'] = float(np.min(s.U_dust[0]))
    d['reservoir_min'] = float(np.min(s.e_s))
    return d


def run(case, integrator, n, t_end, budget):
    start = time.perf_counter()
    s = initial(case, integrator, n)
    prim = s.equations.primitive(s.U_gas)
    mach = np.sqrt(np.mean(np.sum((s.U_gas[1:4]/s.U_gas[0])**2, axis=0)))
    mach /= np.sqrt(s.equations.gamma*np.mean(prim[4])/np.mean(prim[0]))
    beta = 2*np.mean(prim[4])/np.mean(np.sum(s.U_gas[5:8]**2, axis=0))
    history, failure, steps = [measured(s)], None, 0
    while s.t < t_end-1e-14:
        if time.perf_counter()-start >= budget:
            failure = f'wall-time budget {budget:g} s exhausted before requested final time'
            break
        # Do not publish a failed trial as an accepted physical state.
        gas, dust, reservoir = s.U_gas.copy(), s.U_dust.copy(), s.e_s.copy()
        try:
            s.step(min(s.compute_dt(), t_end-s.t))
            state = measured(s)
            if (state['unrealizable_cells'] or state['thermal_min'] <= 0
                    or state['gas_density_min'] <= 0 or state['dust_density_min'] <= 0
                    or state['reservoir_min'] < 0):
                raise ValueError('accepted solver state violates admissibility')
        except Exception as exc:
            failure = f'{type(exc).__name__}: {exc}'
            s.U_gas, s.U_dust, s.e_s = gas, dust, reservoir
            break
        history.append(state)
        steps += 1
    h = {key: np.array([row[key] for row in history]) for key in history[0]}
    p = s.closure.primitives(s.U_dust)
    maps = (s.U_gas[0], s.U_dust[0], np.sqrt(np.maximum(0., np.trace(p['C'], axis1=0, axis2=1)/3)))
    units = 1024**2 if sys.platform == 'darwin' else 1024
    return dict(case=case, integrator=integrator, n=n, requested_time=t_end,
                dtype=str(s.U_gas.dtype), seed=SEED, spectral_dissipation=FILTER,
                dust_to_gas=.1, stopping_time=.1, charge_to_mass=10.,
                gas_cfl=.5, dust_cfl=.8, initial_dust_dispersion=.05,
                initial_reservoir_energy=.005,
                runtime_seconds=time.perf_counter()-start,
                process_peak_memory_upper_bound_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/units,
                initial_rms_mach=mach, initial_mean_beta=beta, steps=steps,
                failure=failure, history=h), maps


def plot(r, result, maps):
    h = result['history']
    name = f"{result['case']}_{result['integrator']}"
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for ax, data, title in zip(axes, maps, ('Gas density', 'Dust density', 'Dust dispersion')):
        image = ax.imshow(data, origin='lower', extent=(0, 1, 0, 1),
                          interpolation='nearest', cmap='viridis')
        ax.set(title=title, xlabel='x', ylabel='y', aspect='equal')
        fig.colorbar(image, ax=ax, shrink=.8)
    fig.suptitle(f"{result['case']}, {result['integrator']}, {result['n']}², t={h['time'][-1]:.4g}")
    r.figure(fig, name+'_maps.png')
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for key in ('E_resolved', 'E_thermal', 'E_sgs', 'E_dust_mean', 'E_dust_random', 'E_magnetic'):
        axes[0].plot(h['time'], h[key], label=key.removeprefix('E_'))
    for key in ('production', 'sgs_decay', 'dust_work', 'drag_heat_mean', 'drag_heat_random', 'compression_to_sgs'):
        axes[1].plot(h['time'], h[key], label=key)
    predicted = h['E_sgs'][0]+h['production']-h['sgs_decay']-h['dust_work']+h['compression_to_sgs']
    axes[2].plot(h['time'], h['E_total']/h['E_total'][0]-1, label='total energy')
    axes[2].plot(h['time'], (h['E_sgs']-predicted)/h['E_total'][0], label='SGS ledger')
    for ax, title, ylabel in zip(axes, ('Energy partition', 'Cumulative exchanges', 'Conservation'),
                                ('Integrated energy', 'Integrated energy', 'Error / initial total energy')):
        ax.set(title=title, xlabel='t', ylabel=ylabel)
        ax.legend(fontsize=7)
    r.figure(fig, name+'_ledger.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=128)
    parser.add_argument('--time', type=float, default=.005)
    parser.add_argument('--budget', type=float, default=170., help='wall-time limit in seconds per case')
    parser.add_argument('--integrators', nargs='+', choices=('strang', 'exponential'), default=['strang', 'exponential'])
    parser.add_argument('--cases', nargs='+', choices=('turbulence', 'reduced_ot'), default=['turbulence', 'reduced_ot'])
    args = parser.parse_args()
    if args.n < 32 or args.time <= 0 or args.budget <= 0:
        parser.error('require n >= 32, time > 0 and budget > 0 (modes reach k=8)')
    r = Report('dusty_orszag_tang')
    r.notes.append('Short-time qualification only. p=8 filter, rate=500, onset=0.5; no broad filter. '
                   'Sequential cases; reported per-case process high-water memory is a conservative upper bound. '
                   'No expected sign is imposed on the nonlinear loading or integrator comparison.')
    r.finish(complete=False)
    for case in args.cases:
        for integrator in args.integrators:
            result, maps = run(case, integrator, args.n, args.time, args.budget)
            h, name = result['history'], f'{case}/{integrator}'
            predicted = h['E_sgs'][0]+h['production']-h['sgs_decay']-h['dust_work']+h['compression_to_sgs']
            result['energy_drift'] = float(np.max(np.abs(h['E_total']/h['E_total'][0]-1)))
            result['SGS_ledger_error'] = float(np.max(np.abs(h['E_sgs']-predicted))/h['E_total'][0])
            r.runs.append(result)
            r.check(name+'/reached final time', h['time'][-1], args.time, op='==')
            r.check(name+'/runtime seconds', result['runtime_seconds'], 180., op='<')
            r.check(name+'/memory upper bound MiB', result['process_peak_memory_upper_bound_MiB'], 4096., op='<')
            r.check(name+'/total energy drift', result['energy_drift'], 1e-10)
            r.check(name+'/SGS ledger closure', result['SGS_ledger_error'], 1e-10)
            for key in ('mass_gas', 'mass_dust', 'momentum_x', 'momentum_y', 'momentum_z'):
                error = np.max(np.abs(h[key]-h[key][0]))/max(1., abs(h[key][0]))
                r.check(name+'/'+key+' drift', error, 1e-10)
            for key in ('repaired_cells', 'positivity_cells', 'unrealizable_cells'):
                r.check(name+'/'+key, np.max(h[key]), 0, op='==')
            r.check(name+'/div B maximum', np.max(h['div_b_max']), 1e-11)
            if result['failure']:
                r.notes.append(f"{name} stopped: {result['failure']}. Reproduce with --n {args.n} "
                               f'--time {args.time} --cases {case} --integrators {integrator}.')
            plot(r, result, maps)
            r.finish(complete=False)
    return r.finish()


if __name__ == '__main__':
    raise SystemExit(main())
