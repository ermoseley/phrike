"""Periodic kinetic free streaming: Gaussian packet and crossing cold beams.

Run as PYTHONPATH=. python scripts/dust_validation/beams_free_streaming.py.
The source-free limit nu=0 is used: t_s=1e12 would change the answer by <1e-12.
References integrate f(x-vt,v), independently of the closure flux quadrature.
"""
import time

from common import Report, np, plt
from phrike.dust_moments import GaussianClosure, HyqmomClosure, transport


def packet(x, centre, width):
    return sum(np.exp(-0.5*((x-centre+j)/width)**2) for j in (-2, -1, 0, 1, 2))


def reference(case, n, t, nq=48):
    """Cell-averaged x moments 0..4 from independent space/velocity quadrature."""
    gx, wx = np.polynomial.legendre.leggauss(6)
    x = (np.arange(n)[:, None]+0.5+0.5*gx)/n
    if case == 'gaussian_blob':
        z, w = np.polynomial.hermite.hermgauss(nq)
        vel = 0.6+np.sqrt(2)*0.15*z
        weights = w/np.sqrt(np.pi)
        density = 0.05+packet(x[..., None]-vel*t, 0.35, 0.07)
    else:
        # Each stream's floor is 1e-6 of its peak: avoid the documented
        # near-vacuum Gaussian-closure pathology without hiding bimodality.
        vel, weights = np.array([1., -1.]), np.ones(2)
        density = np.stack([1e-6+packet(x-t, 0.28, 0.055),
                            1e-6+packet(x+t, 0.72, 0.055)], axis=-1)
    return np.array([np.einsum('nqv,v,q->n', density, weights*vel**k, wx/2)
                     for k in range(5)])


def initial(closure, raw):
    rho, m = raw[0], raw[1]/raw[0]
    variance = raw[2]/rho-m*m
    u = np.zeros((3, rho.size))
    u[0] = m
    if closure.name == 'gaussian':
        C = np.zeros((3, 3, rho.size))
        C[0, 0] = variance
        return closure.from_primitives(rho, u, C)
    q = raw[3]/rho-3*m*raw[2]/rho+2*m**3
    r = raw[4]/rho-4*m*raw[3]/rho+6*m*m*raw[2]/rho-3*m**4
    return closure.from_primitives(rho, u, variance, q, r, np.zeros_like(rho))


def properties(closure, U):
    p = closure.primitives(U)
    variance = p['C'][0, 0] if closure.name == 'gaussian' else p['v']
    result = dict(rho=p['rho'], m=p['u'][0], v=variance)
    if closure.name == 'hyqmom':
        live = variance > 1e-8
        denom = np.where(live, variance, 1.)
        result.update(S3=np.where(live, p['q']/denom**1.5, 0.),
                      S4=np.where(live, p['r']/denom**2, 0.))
    return result


def exact_properties(raw):
    rho, m = raw[0], raw[1]/raw[0]
    v = raw[2]/rho-m*m
    q = raw[3]/rho-3*m*raw[2]/rho+2*m**3
    r = raw[4]/rho-4*m*raw[3]/rho+6*m*m*raw[2]/rho-3*m**4
    live = v > 1e-8
    vv = np.where(live, v, 1.)
    return dict(rho=rho, m=m, v=v, S3=np.where(live, q/vv**1.5, 0.),
                S4=np.where(live, r/vv**2, 0.))


def margins(closure, U):
    p = closure.primitives(U)
    if closure.name == 'gaussian':
        eigen = np.linalg.eigvalsh(np.moveaxis(p['C'], (0, 1), (-2, -1)))
        return dict(min_variance=float(np.min(eigen)), min_hankel_scaled=0.)
    v, q, r = p['v'], p['q'], p['r']
    live = v > 1e-8
    K = r-v*v-np.divide(q*q, v, out=np.zeros_like(v), where=live)
    # K/v^2 is ill conditioned for cold fast streams. Also retain a margin
    # normalised by the cancelled raw-moment scales, used by the code itself.
    raw_scale = np.abs(U[4]/U[0])+4*np.abs(p['u'][0]*U[3]/U[0])+6*p['u'][0]**2*np.abs(U[2]/U[0])+3*p['u'][0]**4
    return dict(min_variance=float(np.min(v)),
                min_K_over_v2=float(np.min((K/np.where(live, v*v, 1.))[live])) if np.any(live) else 0.,
                min_hankel_scaled=float(np.min(K/np.maximum(raw_scale, 1e-30))))


def main():
    report = Report('beams_free_streaming')
    report.notes += [
        'Pure transport, nu=0; the t_s=1e12 source correction is below 1e-12 over these durations.',
        'Space quadrature: 6-point Gauss-Legendre cell averages. Gaussian velocities: 48-point Gauss-Hermite; cold beams: exact two-atom measure.',
        'rho error is relative volume L1. m,v errors are rho-reference weighted L1, normalised by characteristic velocity=1 and velocity squared=1. S3,S4 errors are relative rho-weighted L1 with denominator max(1,<abs(reference)>).',
        'All conserved raw fields checked, including zero transverse fields. Maximum drift uses initial integral of abs(field), floored by total mass; this remains defined for zero momentum.',
        'Cold K/v^2 is reported but its roundoff allowance is cancellation-scaled; closure.realizable is also tested at every checkpoint. No assumed Gaussian/HyQMOM accuracy ranking.',
    ]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), constrained_layout=True)
    convergence, profiles = {}, {}
    for row, (case, final_time) in enumerate([('gaussian_blob', 0.30), ('crossing_streams', 0.22)]):
        for closure in (GaussianClosure(1), HyqmomClosure()):
            errors = []
            for n in (64, 128, 256, 512):
                start = time.perf_counter()
                label = f'{case}/{closure.name}/N{n}'
                raw = reference(case, n, 0.)
                U = initial(closure, raw)
                total0 = U.sum(axis=1)/n
                normalizer = np.maximum(np.abs(U).sum(axis=1)/n, total0[0])
                stats = dict(substeps=0, halvings=0, first_order_cells=0,
                             repaired_cells=0, roundoff_projections=0,
                             min_slope_scale=1., regularization_dE=0.)
                worst_drift = np.zeros(closure.nfields)
                worst_cone, largest_variance = 0, 0.
                min_margin = dict(min_variance=np.inf, min_hankel_scaled=np.inf)
                failed = None
                advanced_time, bound_violation = 0., False
                for _ in range(4):
                    try:
                        U, update = transport(U, closure, final_time/4, (1/n,))
                    except (RuntimeError, ValueError, FloatingPointError) as exc:
                        failed = str(exc)
                        break
                    advanced_time += final_time/4
                    for key in stats:
                        if key == 'min_slope_scale':
                            stats[key] = min(stats[key], update[key])
                        else:
                            stats[key] += float(np.sum(update[key]))
                    worst_drift = np.maximum(worst_drift, np.abs(U.sum(axis=1)/n-total0)/normalizer)
                    worst_cone = max(worst_cone, int(np.count_nonzero(~closure.realizable(U))))
                    for key, value in margins(closure, U).items():
                        min_margin[key] = min(min_margin.get(key, np.inf), value)
                    largest_variance = max(largest_variance, float(np.max(properties(closure, U)['v'])))
                    if update['repaired_cells']:
                        failed = 'nonzero repaired_cells: affected run stopped'
                        break
                    if closure.name == 'hyqmom' and case == 'crossing_streams' and largest_variance > 1+1e-10:
                        bound_violation = True
                        break  # Preserve this failure and stop at its first checkpoint.
                report.check(label+'/completed', int(failed is None and advanced_time >= final_time-1e-15), 1, op='==',
                             reason='A transport failure cannot be accepted as a closure comparison.')
                report.check(label+'/repaired_cells', stats['repaired_cells'], 0, op='==',
                             reason='A physical crossing or smooth packet must not require repairs outside tolerance.')
                report.check(label+'/cone_violations', worst_cone, 0, op='==',
                             reason='The positive kinetic flux is required to preserve the moment cone.')
                # Cold fast beams cancel raw O(1) terms. Use an absolute
                # velocity-unit variance tolerance, and the cancelled fourth
                # moment scale for K, rather than an ill-conditioned K/v^2.
                report.check(label+'/minimum_variance', min_margin['min_variance'], -1e-10, op='>=',
                             reason='Allow the documented 1e-10 cold raw-moment cancellation tolerance.')
                report.check(label+'/minimum_scaled_hankel', min_margin['min_hankel_scaled'], -1e-12, op='>=',
                             reason='Fourth-order cone slack scaled by the raw terms permits float64 cancellation.')
                # Periodic flux differences telescope. 2e-12 allows accumulated
                # float64 summation; it does not allow physical moment loss.
                for field, drift in zip(closure.field_names, worst_drift):
                    report.check(label+'/conservation_'+field, drift, 2e-12,
                                 reason='Periodic raw-moment conservation to accumulated float64 roundoff.')
                if closure.name == 'hyqmom' and case == 'crossing_streams':
                    # A probability measure supported in [-1,1] has variance<=1
                    # (Popoviciu bound). Permit only floating-point cancellation.
                    # v = M2/M0 - m^2 from raw float64 moments carries cancellation
                    # error ~eps*s_v per operation; over ~1e3 substeps it random-walks
                    # to ~1e-10..1e-9 of s_v (~1 here), the same floor below which the
                    # closure treats variances as cold (docs §6).
                    report.check(label+'/kinetic_variance_excess', largest_variance-1., 1e-9,
                                 reason='Velocities are +/-1, so variance-1 must stay within the raw-moment round-off floor (~1e-9).')
                ref = exact_properties(reference(case, n, advanced_time))
                p = properties(closure, U)
                norm_mass = ref['rho'].sum()
                err = {}
                if failed is None:
                    for key in p:
                        if key == 'rho':
                            err[key] = float(np.abs(p[key]-ref[key]).sum()/norm_mass)
                        else:
                            scale = max(1., float(np.sum(ref['rho']*np.abs(ref[key]))/norm_mass)) if key in ('S3', 'S4') else 1.
                            err[key] = float(np.sum(ref['rho']*np.abs(p[key]-ref[key]))/norm_mass/scale)
                entry = dict(case=case, closure=closure.name, n=n, t=advanced_time,
                             stats=stats, min_margins=min_margin, max_variance=largest_variance,
                             conservation_by_field=dict(zip(closure.field_names, worst_drift)),
                             L1_errors=err, runtime_seconds=time.perf_counter()-start, failure=failed,
                             stopped_on_kinetic_bound=bound_violation)
                report.runs.append(entry)
                print(f'INFO {label}: repaired_cells={stats["repaired_cells"]:.0f}, roundoff_projections={stats["roundoff_projections"]:.0f}, L1={err}', flush=True)
                errors.append(err)
                if n == 512:
                    profiles[(case, closure.name)] = p
            convergence[(case, closure.name)] = errors
            axes[row, 0].loglog([64,128,256,512], [e.get('rho', np.nan) for e in errors], 'o-', label=closure.name)
        ref = exact_properties(reference(case, 512, final_time))
        x = (np.arange(512)+.5)/512
        for col, quantity in ((1, 'rho'), (2, 'v')):
            axes[row, col].plot(x, ref[quantity], 'k--', lw=1.3, label='kinetic reference')
            for name in ('gaussian', 'hyqmom'):
                axes[row, col].plot(x, profiles[(case, name)][quantity], label=name, lw=1)
            axes[row, col].set_xlabel('x / box length')
            axes[row, col].set_ylabel('rho_d / stream peak' if quantity == 'rho' else 'variance / velocity unit²')
        axes[row, 0].set_xlabel('cells N')
        axes[row, 0].set_xticks([64, 128, 256, 512], ['64', '128', '256', '512'])
        axes[row, 0].minorticks_off()
        axes[row, 0].set_ylabel('relative L1 density error')
        for ax in axes[row]:
            ax.set_title(case.replace('_', ' '))
            ax.legend(fontsize=7)
    report.figure(fig, 'kinetic_comparison.png')
    # A reference convergence check separates velocity quadrature error from
    # closure and FV errors. This threshold is far below expected O(dx^2) error.
    r48 = reference('gaussian_blob', 512, .30, 48)
    r80 = reference('gaussian_blob', 512, .30, 80)
    report.check('kinetic_reference_quadrature', np.max(np.abs(r48-r80))/np.max(np.abs(r80)), 1e-10,
                 reason='Independent quadrature must be negligible relative to finite-volume truncation.')
    report.notes.append('Density observed orders (successive resolutions) and closure error floors are reported, not imposed as transport accuracy requirements for a non-Gaussian kinetic reference.')
    for (case, name), errs in convergence.items():
        vals = [e.get('rho', np.nan) for e in errs]
        report.notes.append(f'{case}/{name}: density observed orders '+str([float(np.log2(a/b)) for a,b in zip(vals[:-1],vals[1:])]))
    return report.finish()


if __name__ == '__main__':
    raise SystemExit(main())
