"""Decision study, independent of the production solver.

A is Strang with exact frozen source halves and SSP-RK3 transport. For explicit
coefficient time dependence, the halves sample t_n and t_n+h (clock in T).
B is the consult-2 ETD-RK3 tableau, with the COMPLETE residual F(t,X)-S_n X.
F1 independently qualifies homogeneous_source, including its work ledger, and
checks every candidate on an autonomous source lift with the same exact flow.
The study tableaus operate on dense affine generators, not production states.

C: a partitioned exponential predictor, derived here for affine semidiscrete
systems. Write X=(s,f), where s contains densities, barycentre velocity and B,
and f is slip (or central moments). With constant coefficients,
    s' = A s + B f,       f' = L f + C s.
Instead of assigning s to an entirely zero exponential block, use
    K = [[0,B],[0,L]],    N = F - K X.
The action of exp(t K) is (s+B*t*phi_1(t L)f, exp(t L)f). For k>=1,
phi_k(h K) has diagonal blocks I/k!, phi_k(h L), and upper-right block
B*h*phi_{k+1}(h L). No inverse of L is needed at zero or at resonance.
For fast forcing d0+t*d1, the corresponding slow increment is exactly
B*(h*phi_1*w0+h^2*phi_2*d0+h^3*phi_3*d1). The midpoint and endpoint predictors
of the consult-2 tableau fit constant and linear residual forcing exactly;
the final three-node residual interpolant also includes a quadratic correction.
Thus C has at least classical second order (third for these smooth affine
systems), exact constant-L fast-to-slow response with linear forcing, and the
same RK3 gas-wave stability polynomial as B. Coefficient changes enter N in full.
The fast-to-slow operator B is applied as an operator, so its spatial derivative
can commute past the local phi functions only for constant coefficients. Here
we use tiny dense Fourier blocks to test that construction, not a global PDE
matrix exponential. This is a qualification of the constant-coefficient linear
predictor, NOT a proof or implementation of nonlinear conservative AP fluxes.

In particular K does NOT include slow feedback A or C. C is distinct from the
exact reference exp(h F); retaining fast-to-slow feedthrough is the only change
from B. A row for constant differential-force work can be included in B too,
which makes the half-gyroperiod work counterexample exact. Time-varying work is
obtained from the analytic local source path, not endpoint power samples.

Cprime (derived interaction-integrated variant): the measured collisionless
feedback error in C motivates three Picard waveform iterations,
    X_0(tau)=exp(tau S_n)X_n,
    X_j'=S_n X_j+R(tau)X_{j-1}, X_j(0)=X_n, j=1,2,3,
where R=F-S_n is the COMPLETE residual. Fit F(t) linearly between the step
endpoints; for constant affine systems this is exact. Evaluate the resulting
integrals analytically by a strictly triangular lift on tau^p X_j, p<=3-j.
Its generator has only S_n on the diagonal, R0/R1 between iteration levels,
and p times the lower polynomial power. Thus it never exponentiates the full
coupled F. On a constant split, X_3 sums the Dyson series through three T
insertions. If exp(t S) is bounded uniformly in the physical energy norm and
T is bounded independently of stiffness, the omitted Volterra terms are
bounded by a constant times (h||T||)^4; composition gives uniform order three
on these constant linear systems. In the Fourier tests S is zero plus scalar
slip relaxation/rotation, so this hypothesis holds in barycentre/slip norms
at each fixed loading. For smooth changing coefficients, linear interpolation
of F gives at least order two, without a uniform-geometry claim. For T=0 and
constant S the map is exact; for S=0 and constant T it is the RK3 polynomial.
For x'=w,w'=Lw+d0+t*d1 the nilpotent coupling chain terminates and the slip
integral and constant-force work are exact. This is a study-only affine
waveform construction, not an attributed implementation of a literature scheme.
Nonlinear fluxes cannot be integrated by this finite lift in general.

Run: PYTHONPATH=. python scripts/dust_validation/ap_integrator_study.py
"""
from __future__ import annotations

import hashlib
from pathlib import Path

from common import Report, np, plt
from scipy.linalg import expm
from scipy.special import erfi
from phrike.coupled_exponential import homogeneous_source

CANDIDATES = ('A', 'B', 'C', 'Cprime')
COLORS = dict(A='tab:orange', B='tab:blue', C='tab:green', Cprime='tab:red')
EPSILONS = (.01, 1., 10.)


def matrix_phis(z):
    """Independent dense oracle for tiny study operators, including singular ones."""
    n = len(z)
    lift = np.zeros((4*n, 4*n), dtype=np.complex128)
    lift[:n, :n] = z
    for j in range(3):
        lift[j*n:(j+1)*n, (j+1)*n:(j+2)*n] = np.eye(n)
    e = expm(lift)
    return tuple(e[:n, j*n:(j+1)*n] for j in range(4))


def rk3_step(y, t, h, rhs):
    a = y+h*rhs(t, y)
    b = .75*y+.25*(a+h*rhs(t+h, a))
    return y/3+2/3*(b+h*rhs(t+h/2, b))


def step(y, t, h, source, transport, nslow, candidate):
    """Generic affine split, augmented with constants/clocks when needed."""
    if candidate == 'A':
        a = expm(.5*h*source(t)) @ y
        a = rk3_step(a, t, h, lambda time, x: transport(time) @ x)
        return expm(.5*h*source(t+h)) @ a
    if candidate == 'Cprime':
        S = np.asarray(source(t), dtype=complex)
        R0 = source(t)+transport(t)-S
        R1 = (source(t+h)+transport(t+h)-S-R0)/h
        # Derivative of tau^p X_j: S*state + p*tau^(p-1)X_j
        # + R0*tau^p X_(j-1) + R1*tau^(p+1)X_(j-1).
        keys = ([(j, 0) for j in range(4)] if not np.any(R1)
                else [(j, p) for j in range(4) for p in range(4-j)])
        n = len(S)
        index = {key: slice(i*n, (i+1)*n) for i, key in enumerate(keys)}
        G = np.zeros((len(keys)*n, len(keys)*n), complex)
        initial = np.zeros((len(keys)*n, *np.shape(y)[1:]), complex)
        for (j, p), ix in index.items():
            G[ix, ix] = S
            if p:
                G[ix, index[j, p-1]] = p*np.eye(n)
            else:
                initial[ix] = y
            if j:
                G[ix, index[j-1, p]] = R0
                if np.any(R1):
                    G[ix, index[j-1, p+1]] = R1
        return (expm(h*G) @ initial)[index[3, 0]]
    frozen = np.array(source(t), dtype=np.complex128, copy=True)
    if candidate == 'C':
        frozen[:nslow, nslow:] += transport(t)[:nslow, nslow:]
    e, p1, p2, p3 = matrix_phis(h*frozen)
    eh, ph, _, _ = matrix_phis(.5*h*frozen)

    def residual(time, x):
        return (source(time)+transport(time)-frozen) @ x

    n0 = residual(t, y)
    a = eh @ y+.5*h*(ph @ n0)
    na = residual(t+h/2, a)
    b = e @ y+h*(p1 @ (2*na-n0))
    nb = residual(t+h, b)
    return e @ y+h*((p1-3*p2+4*p3) @ n0
                     +(4*p2-8*p3) @ na+(-p2+4*p3) @ nb)


def constant_result(S, T, nslow, initial, h, steps, candidate):
    propagator = step(np.eye(len(S)), 0., h, lambda t: S, lambda t: T,
                      nslow, candidate)
    return np.linalg.matrix_power(propagator, steps) @ initial


def field_metrics(y, reference, initial, fields):
    scale = max(float(np.linalg.norm(initial)), 1e-30)
    return dict(error=float(np.linalg.norm(y-reference)/scale),
                fields={name: dict(absolute_error=float(abs(a-b)),
                                  scaled_error=float(abs(a-b)/scale),
                                  amplitude=float(abs(a)), reference_amplitude=float(abs(b)),
                                  phase_error=float(np.angle(a*np.conj(b)))
                                  if abs(a)>1e-14*scale and abs(b)>1e-14*scale else None)
                        for name, a, b in zip(fields, y, reference)})


def fourier_system(kind, eps, nu, omega, k=1.):
    """Consult-2 matrices, transformed by exact barycentre/slip coordinates."""
    if kind == 'acoustic':
        F = np.array([[0, 0, -1j*k, 0], [0, 0, 0, -1j*k],
                      [-1j*k, 0, -eps*nu, eps*nu], [0, 0, nu, -nu]], complex)
        P = np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                      [0, 0, 1/(1+eps), eps/(1+eps)], [0, 0, -1, 1]])
        fields, nslow = ('r_g', 'r_d', 'u_g', 'u_d'), 3
        unprepared = np.array([.2, -.3, -.4, .8], complex)
    else:
        ac = -nu-1j*omega
        F = np.array([[eps*ac, -eps*ac, 1j*k], [-ac, ac, 0], [1j*k, 0, 0]])
        P = np.array([[1/(1+eps), eps/(1+eps), 0], [0, 0, 1], [-1, 1, 0]])
        fields, nslow = ('u_g', 'u_d', 'delta_B'), 2
        unprepared = np.array([-.4, .8, .3j], complex)
    inv = np.linalg.inv(P)
    M = P @ F @ inv
    S = np.zeros_like(M)
    S[-1, -1] = -(1+eps)*(nu+1j*(omega if kind == 'alfven' else 0.))
    T = M-S
    values, vectors = np.linalg.eig(F)
    # The positive-frequency, smallest-|lambda| nonstationary wave branch.
    modes = np.flatnonzero(values.imag > 1e-8)
    index = modes[np.argmin(abs(values[modes]))] if len(modes) else np.argmax(values.imag)
    prepared = vectors[:, index]
    prepared /= np.linalg.norm(prepared)
    return F, S, T, P, inv, nslow, fields, dict(slow_mode=prepared, slip=unprepared)


def homogeneous_reference(ug, ud, C, h, rho_g, rho_d, nu, omega, d0, d1,
                          a0, a1, D0, D1, *, system=False):
    """Exact lifted linear/quadratic matrix exponential, independent of phi code."""
    alpha = 1+rho_d/rho_g
    rr = rho_d/alpha
    # z=(w[3],U[3],t,1), plus z tensor z, covariance, integral w, and ledgers.
    j = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
    A = -nu*np.eye(3)-omega*j
    m = np.zeros((8, 8))
    m[:3, :3], m[:3, 6], m[:3, 7] = alpha*A, d1, d0
    m[3:6, 6], m[3:6, 7], m[6, 7] = a1, a0, 1
    g = np.zeros((88, 88))
    g[:8, :8] = m
    g[8:72, 8:72] = np.kron(m, np.eye(8))+np.kron(np.eye(8), m)
    g[72:81, 72:81] = np.kron(A, np.eye(3))+np.kron(np.eye(3), A)
    g[72:81, 6], g[72:81, 7] = 2*D1.ravel(), 2*D0.ravel()
    g[81:84, :3] = np.eye(3)
    for i in range(3):
        g[84, 8+8*i+i] = rho_d*nu
        g[85, 72+3*i+i] = rho_d*nu
        g[86, 8+8*i+7] += rr*d0[i]
        g[86, 8+8*i+6] += rr*d1[i]
        g[86, 8+8*(i+3)+7] += (rho_g+rho_d)*a0[i]
        g[86, 8+8*(i+3)+6] += (rho_g+rho_d)*a1[i]
    g[87, 6], g[87, 7] = rho_d*np.trace(D1), rho_d*np.trace(D0)
    z = np.r_[ud-ug, (rho_g*ug+rho_d*ud)/(rho_g+rho_d), 0., 1.]
    init = np.r_[z, np.outer(z, z).ravel(), C.ravel(), np.zeros(7)]
    ans = expm(h*g) @ init
    if system:
        return g, init, ans
    return dict(w=ans[:3], ug=ans[3:6]-rho_d/(rho_g+rho_d)*ans[:3],
                ud=ans[3:6]+rho_g/(rho_g+rho_d)*ans[:3],
                C=ans[72:81].reshape(3, 3), integrated_slip=ans[81:84],
                Q_mean=ans[84], Q_random=ans[85], mechanical_work=ans[86], SGS_work=ans[87])


def hyqmom_reference(state, h, nu, Dpar0, Dpar1, Dperp0, Dperp1, *, system=False):
    # (v,q,r,sperp2,t,1,t*v,t^2): closes exactly for linear forcing.
    M = np.zeros((8, 8))
    M[0, 0], M[0, 4], M[0, 5] = -2*nu, 2*Dpar1, 2*Dpar0
    M[1, 1] = -3*nu
    M[2, 2], M[2, 0], M[2, 6] = -4*nu, 12*Dpar0, 12*Dpar1
    M[3, 3], M[3, 4], M[3, 5] = -2*nu, 2*Dperp1, 2*Dperp0
    M[4, 5] = 1
    M[6, 0], M[6, 6], M[6, 4], M[6, 7] = 1, -2*nu, 2*Dpar0, 2*Dpar1
    M[7, 4] = 2
    initial = np.r_[state, 0., 1., 0., 0.]
    if system:
        return M, initial
    return (expm(h*M) @ initial)[:4]


def family1(report):
    errors = {key: 0. for key in ('w', 'ug', 'ud', 'C', 'integrated_slip', 'Q_mean', 'Q_random',
                                 'mechanical_work', 'SGS_work', 'hyqmom')}
    ug, ud = np.array([.2, -.3, .1]), np.array([.9, .2, -.4])
    C = np.array([[.7, .1, -.08], [.1, .5, .12], [-.08, .12, .9]])
    d0, d1 = np.array([.4, -.1, .2]), np.array([-.2, .3, .1])
    a0, a1 = np.array([.1, .2, -.1]), np.array([.03, -.04, .02])
    D0, D1 = np.diag([.2, .2, .3]), np.diag([.04, .04, -.1])
    h = .7
    for eps in EPSILONS:
        for nu, omega in ((0., 0.), (0., 4.), (.8, 0.), (.8, 4.), (1e3, 1e3)):
            ref = homogeneous_reference(ug, ud, C, h, 1., eps, nu, omega,
                                        d0, d1, a0, a1, D0, D1)
            args = dict(rho_g=1., rho_d=eps, nu=nu, omega=omega,
                        bhat=np.array([0., 0., 1.]), d0=d0, d1=d1, a0=a0, a1=a1,
                        D0=D0, D1=D1)
            result = homogeneous_source(ug, ud, h, C=C, **args)
            local = {}
            for key, exact in ref.items():
                err = float(np.max(abs(result[key]-exact)))/max(1., float(np.max(abs(exact))))
                errors[key] = max(errors[key], err)
                local[key] = err
            hs = np.array([.9, .25, 2.7, .5])
            hy = homogeneous_source(ug, ud, h, hyqmom=hs, **args)
            hr = hyqmom_reference(hs, h, nu, D0[2, 2], D1[2, 2], D0[0, 0], D1[0, 0])
            local['hyqmom'] = float(np.max(abs(hy['hyqmom']-hr)))
            errors['hyqmom'] = max(errors['hyqmom'], local['hyqmom'])
            hyref = homogeneous_reference(ug, ud, np.diag([hs[3], hs[3], hs[0]]),
                                          h, 1., eps, nu, omega,
                                          d0, d1, a0, a1, D0, D1)
            for key in ('Q_mean', 'Q_random', 'mechanical_work', 'SGS_work'):
                err = float(abs(hy[key]-hyref[key]))/max(1., abs(hyref[key]))
                local['hyqmom_'+key] = err
                errors[key] = max(errors[key], err)
            report.runs.append(dict(family='F1', eps=eps, nu=nu, omega=omega,
                                    source_primitive=True, errors=local,
                                    Q_mean=float(result['Q_mean']), Q_random=float(result['Q_random']),
                                    mechanical_work=float(result['mechanical_work']),
                                    SGS_work=float(result['SGS_work'])))
    for key, err in errors.items():
        report.check('F1 exact local '+key, err, 2e-11)
    # All three actual tableaus are exact on an autonomous affine source lift.
    # The clock and quadratic products belong to S, not endpoint work sampling.
    G, initial, reference = homogeneous_reference(ug, ud, C, h, 1., 1., .8, 4.,
                                                  d0, d1, a0, a1, D0, D1, system=True)
    for candidate in CANDIDATES:
        result = step(initial, 0., h, lambda t: G, lambda t: np.zeros_like(G), 0, candidate)
        err = float(np.max(abs(result-reference)))
        report.check('F1 '+candidate+' homogeneous affine tableau and ledgers', err, 2e-12)
    H, initial = hyqmom_reference(hs, h, .8, .3, -.1, .2, .04, system=True)
    for candidate in CANDIDATES:
        result = step(initial, 0., h, lambda t: H, lambda t: np.zeros_like(H), 0, candidate)
        report.check('F1 '+candidate+' HyQMOM affine tableau',
                     np.max(abs(result-expm(h*H) @ initial)), 2e-12)
    # Zero mean and perpendicular force, Omega*h=pi, finite loading.
    eps, h, Omega = 1., 1., np.pi
    force = np.array([1., 0., 0.])
    ans = homogeneous_source(np.zeros(3), np.zeros(3), h, rho_g=1., rho_d=eps,
                             nu=0., omega=Omega/2, bhat=np.array([0., 0., 1.]),
                             C=np.zeros((3, 3)), d0=force)
    exact_work = 2*(eps/2)/Omega**2
    report.check('F1 half gyro exact mechanical work', abs(float(ans['mechanical_work'])-exact_work), 2e-14)
    report.check('F1 half gyro physical heat', abs(float(ans['Q_mean']))+abs(float(ans['Q_random'])), 0., op='==')
    report.runs.append(dict(family='F1_half_gyro', exact_work=exact_work,
                            work=float(ans['mechanical_work']), heat=float(ans['Q_mean'])))
    return errors


def family2(report):
    def zero(t):
        return np.zeros((3, 3), complex)
    jz = np.array([[0., -1, 0], [1, 0, 0], [0, 0, 0]])
    K = np.array([[0., 0., .8], [0., 0., 0.], [-.8, 0., 0.]])
    L0 = -.3*np.eye(3)-1.7*jz
    w0 = np.array([1., .4, -.2])
    def forced_scalar(a, b, d0, d1, w=1.):
        # Integrating factor for w'=-(a+b*t)w+d0+d1*t, t_end=1.
        root = np.sqrt(complex(b)/2)
        I0 = np.sqrt(np.pi)/(2*root)*np.exp(-a*a/(2*b))*(
            erfi(root*(1+a/b))-erfi(root*a/b))
        I1 = (np.exp(a+b/2)-1-a*I0)/b
        return np.exp(-a-b/2)*(w+d0*I0+d1*I1)

    def affine_source(matrix):
        out = np.zeros((4, 4))
        out[:3, :3] = matrix
        return out

    forcing0, forcing1 = np.array([.2, -.1, .3]), np.array([-.1, .2, .05])
    lift = np.zeros((12, 12))
    lift[:3, :3] = L0-K
    lift[:3, 3:6], lift[:3, 9:12] = np.eye(3), np.eye(3)
    for i in (3, 6, 9):
        lift[i:i+3, i:i+3] = -K
    lift[9:12, 6:9] = np.eye(3)
    rotating_forced_ref = expm(K) @ (expm(lift) @ np.r_[w0, forcing0, forcing1, np.zeros(3)])[:3]
    def forcing_transport(t):
        return np.column_stack((np.zeros((4, 3)), np.r_[forcing0+t*forcing1, 0.]))
    # Linear OU forcing is a separate affine case with a constant-component row.
    cases = dict(
        linear_drag=(lambda t: -(.7+.4*t)*np.eye(3), np.exp(-.9)*w0, w0, zero),
        linear_frequency=(lambda t: -(.6+1.2*t)*jz, expm(-1.2*jz) @ w0, w0, zero),
        rotating_axis=(lambda t: expm(t*K) @ L0 @ expm(-t*K),
                       expm(K) @ expm(L0-K) @ w0, w0, zero),
        linear_drag_forced=(lambda t: np.diag([-(.7+.4*t), 0.]),
                            np.array([forced_scalar(.7, .4, .2, -.1), 1.]), np.ones(2),
                            lambda t: np.array([[0., .2-.1*t], [0., 0.]])),
        linear_frequency_forced=(lambda t: np.diag([-1j*(.6+1.2*t), 0.]),
                                 np.array([forced_scalar(.6j, 1.2j, .2, -.1), 1.]), np.ones(2),
                                 lambda t: np.array([[0., .2-.1*t], [0., 0.]])),
        rotating_axis_forced=(lambda t: affine_source(expm(t*K) @ L0 @ expm(-t*K)),
                              np.r_[rotating_forced_ref, 1.], np.r_[w0, 1.], forcing_transport),
        linear_OU=(lambda t: np.array([[0., 0., 0.], [.4, -1.4, .6], [1., 0., 0.]]),
                   expm(np.array([[0., 0., 0.], [.4, -1.4, .6], [1., 0., 0.]]))
                   @ np.array([1., .8, 0.]), np.array([1., .8, 0.]), zero))
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), constrained_layout=True)
    summaries = {}
    for ax, (name, (source, reference, initial, transport)) in zip(axes.flat, cases.items()):
        for candidate in CANDIDATES:
            errors, hs = [], []
            for n in (8, 16, 32, 64):
                h, y = 1/n, initial.copy()
                for i in range(n):
                    y = step(y, i*h, h, source, transport, 0, candidate)
                err = float(np.linalg.norm(y-reference))
                errors.append(err)
                hs.append(h)
            order = float(np.log2(errors[-2]/errors[-1])) if errors[-1]>2e-13 else None
            report.runs.append(dict(family='F2', case=name, candidate=candidate,
                                    h=hs, errors=errors, finest_order=order))
            summaries[name+'/'+candidate] = order
            if order is not None:
                report.check('F2 '+name+'/'+candidate+' smooth order', order, 1.85, op='>=')
            ax.loglog(hs, np.maximum(errors, 1e-16), 'o-', color=COLORS[candidate], label=candidate)
        ax.set(title=name, xlabel='h', ylabel='absolute error')
        ax.set_xticks([hs[-1], hs[0]], labels=[f'{hs[-1]:g}', f'{hs[0]:g}'])
        ax.minorticks_off()
        ax.grid(alpha=.2)
    axes.flat[0].legend()
    axes.flat[-1].set_visible(False)
    report.figure(fig, 'changing_coefficients.png')
    return summaries


def family3(report):
    # x,w,clock,one; clock lies in slow block. Linear forcing is in T.
    records = []
    for regime in ('drag', 'integer_gyro', 'half_integer_gyro', 'pedersen_hall'):
        strengths = (np.r_[np.logspace(-2, 3, 16), 2*np.pi, 20*np.pi] if regime in ('drag', 'pedersen_hall')
                     else 2*np.pi*(np.array([1, 2, 5, 10, 25, 50, 100, 150])
                                   +(.5 if regime=='half_integer_gyro' else 0)))
        for strength in strengths:
            h, steps = .1, 10
            if regime == 'drag':
                rate = -strength/h
            elif regime == 'pedersen_hall':
                rate = -(1+1j)*strength/h
            else:
                rate = -1j*strength/h
            S, T = np.zeros((4, 4), complex), np.zeros((4, 4), complex)
            S[3, 3] = rate
            T[0, 3], T[1, 2] = 1., 1.
            T[3, 2], T[3, 1] = .3, -.2  # d(t)=.3-.2t
            initial = np.array([0., 0., 1., 1.], complex)
            reference = expm((S+T)*h*steps) @ initial
            for c in CANDIDATES:
                y = constant_result(S, T, 3, initial, h, steps, c)
                record = dict(family='F3', regime=regime, strength=float(abs(rate)*h),
                              nu_h=float(-rate.real*h), omega_h=float(-rate.imag*h),
                              candidate=c, x_error=float(abs(y[0]-reference[0])),
                              w_error=float(abs(y[3]-reference[3])))
                report.runs.append(record)
                records.append(record)
    fig, axes = plt.subplots(2, 4, figsize=(13, 6), constrained_layout=True)
    for col, regime in enumerate(('drag', 'integer_gyro', 'half_integer_gyro', 'pedersen_hall')):
        for c in CANDIDATES:
            rows = sorted([r for r in records if r['regime']==regime and r['candidate']==c], key=lambda r:r['strength'])
            for row, field in enumerate(('x_error', 'w_error')):
                axes[row, col].loglog([r['strength'] for r in rows],
                                      np.maximum([r[field] for r in rows], 1e-16),
                                      '.-', color=COLORS[c], label=c)
                axes[row, col].set(xlabel='rate * h', ylabel=field, title=regime)
                axes[row, col].grid(alpha=.2)
    axes[0, 0].legend()
    report.figure(fig, 'fast_to_slow.png')
    exact_errors = {c: max(max(r['x_error'], r['w_error']) for r in records if r['candidate']==c)
                    for c in ('C', 'Cprime')}
    for c, cerror in exact_errors.items():
        report.check('F3 '+c+' exact integrated response', cerror, 2e-11)
    # Work is a slow row, with no external acceleration of the barycentre.
    # z=(W,one,w), Omega*h=pi; physical power = rho_r*d*w_x.
    Omega, rho_r, d = np.pi, .5, 1.
    # Complex w packs the two transverse coordinates; real(W) is physical work.
    S, T = np.zeros((3, 3), complex), np.zeros((3, 3), complex)
    S[2, 2], T[0, 2], T[2, 1] = -1j*Omega, rho_r*d, d
    init = np.array([0., 1., 0.], complex)
    reference = expm(S+T) @ init
    work = {}
    for c in CANDIDATES:
        ans = constant_result(S, T, 2, init, 1., 1, c)
        work[c] = dict(work=float(ans[0].real), exact_work=float(reference[0].real),
                       work_error=float(abs((ans[0]-reference[0]).real)),
                       apparent_heat=float(ans[0].real-.5*rho_r*abs(ans[2])**2))
        report.runs.append(dict(family='F3_work', candidate=c, **work[c]))
    for c in ('C', 'Cprime'):
        report.check('F3 '+c+' half gyro coupled work', work[c]['work_error'], 2e-14)
    terminal_error = 0.
    b, force = np.array([0., 0., 1.]), np.array([1., .3, 0.])
    for eps in EPSILONS:
        for omega in (-700., 700.):
            nu = 1e3
            expected = (nu*force-omega*np.cross(b, force))/((1+eps)*(nu**2+omega**2))
            ans = homogeneous_source(np.zeros(3), np.zeros(3), 1., rho_g=1., rho_d=eps,
                                     nu=nu, omega=omega, bhat=b, C=np.zeros((3, 3)), d0=force)
            terminal_error = max(terminal_error, float(np.max(abs(ans['w']-expected))))
    report.check('F3 finite-loading terminal Pedersen-Hall response/sign', terminal_error, 2e-15)
    return dict(max_C_error=exact_errors['C'], max_Cprime_error=exact_errors['Cprime'], coupled_work=work)


def family4(report):
    sweep, refine = [], []
    strengths = np.logspace(-2, 3, 11)
    h, n, time = 1/32, 32, 1.
    for kind in ('acoustic', 'alfven'):
        for eps in EPSILONS:
            for nh in strengths:
                for oh in (np.unique(np.r_[strengths, np.pi/(1+eps), 2*np.pi/(1+eps),
                                            4*np.pi/(1+eps), 20*np.pi/(1+eps)])
                           if kind=='alfven' else (0.,)):
                    F, S, T, P, inv, ns, fields, states = fourier_system(kind, eps, nh/h, oh/h)
                    reference_map = expm(time*F)
                    maps = {c: inv @ constant_result(S, T, ns, P, h, n, c) for c in CANDIDATES}
                    for state, initial in states.items():
                        reference = reference_map @ initial
                        for c in CANDIDATES:
                            metrics = field_metrics(maps[c] @ initial, reference, initial, fields)
                            row = dict(family='F4_sweep', kind=kind, eps=eps, nu_h=float(nh),
                                       omega_h=float(oh), h=h, time=time, steps=n, state=state,
                                       candidate=c, **metrics)
                            sweep.append(row)
                            report.runs.append(row)
    # Fixed physical coefficients and final time; only h changes. Powers apply
    # the same macro-step recurrence without a costly Python loop of 100k steps.
    nu, time = 2e4, .4
    for kind in ('acoustic', 'alfven'):
        for eps in EPSILONS:
            omega = nu if kind=='alfven' else 0.
            F, S, T, P, inv, ns, fields, states = fourier_system(kind, eps, nu, omega)
            reference_map = expm(time*F)
            for n in (8*2**j for j in range(15)):
                h = time/n
                maps = {c: inv @ constant_result(S, T, ns, P, h, n, c) for c in CANDIDATES}
                for state, initial in states.items():
                    reference = reference_map @ initial
                    for c in CANDIDATES:
                        row = dict(family='F4_refine', kind=kind, eps=eps, nu=nu, omega=omega,
                                   nu_h=nu*h, omega_h=omega*h, h=h, time=time, steps=n,
                                   state=state, candidate=c,
                                   **field_metrics(maps[c] @ initial, reference, initial, fields))
                        refine.append(row)
                        report.runs.append(row)
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    fig2, axes2 = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    summary = []
    for row_index, kind in enumerate(('acoustic', 'alfven')):
        for col, eps in enumerate(EPSILONS):
            ax, ax2 = axes[row_index, col], axes2[row_index, col]
            for c in CANDIDATES:
                # Worst over the independent omega sweep and both initial states.
                es = [max(r['error'] for r in sweep if r['kind']==kind and r['eps']==eps
                          and r['candidate']==c and r['nu_h']==nh) for nh in strengths]
                ax.loglog(strengths, es, 'o-', color=COLORS[c], label=c)
                for state in ('slow_mode', 'slip'):
                    rs = [r for r in refine if r['kind']==kind and r['eps']==eps
                          and r['candidate']==c and r['state']==state]
                    orders = [float(np.log2(a['error']/b['error'])) if min(a['error'],b['error'])>5e-11
                              else None for a, b in zip(rs[:-1], rs[1:])]
                    report.runs.append(dict(family='F4_orders', kind=kind, eps=eps, candidate=c,
                                            state=state, h=[r['h'] for r in rs[1:]], orders=orders))
                    # Coarse stiff and intermediate h intervals are more informative
                    # than asymptotic slopes after the fast scale becomes resolved.
                    summary.append(dict(kind=kind, eps=eps, candidate=c, state=state,
                                        stiff_pair_order=orders[0], transition_order=orders[7],
                                        finest_order=orders[-1], max_sweep_error=max(es),
                                        coarse_error=rs[0]['error'], finest_error=rs[-1]['error']))
                    ax2.loglog([r['h'] for r in rs], [r['error'] for r in rs],
                               '-' if state=='slow_mode' else '--', color=COLORS[c],
                               label=c+'/'+state)
            ax.set(title=f'{kind}, eps={eps:g}', xlabel='nu * h', ylabel='worst relative state error')
            ax2.set(title=f'{kind}, eps={eps:g}', xlabel='h (nu fixed at 20000)', ylabel='relative state error')
            ax.grid(alpha=.2)
            ax2.grid(alpha=.2)
    axes[0, 0].legend()
    axes2[0, 0].legend(fontsize=7)
    report.figure(fig, 'fourier_stiffness.png')
    report.figure(fig2, 'fourier_refinement.png')
    # Independent frequency axis, so high omega failures cannot be hidden in a nu maximum.
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), constrained_layout=True)
    for ax, eps in zip(axes, EPSILONS):
        for c in CANDIDATES:
            es = [max(r['error'] for r in sweep if r['kind']=='alfven' and r['eps']==eps
                      and r['candidate']==c and r['omega_h']==oh) for oh in strengths]
            ax.loglog(strengths, es, 'o-', color=COLORS[c], label=c)
        ax.set(title=f'Alfven, eps={eps:g}', xlabel='omega * h', ylabel='worst over nu and states')
        ax.grid(alpha=.2)
    axes[0].legend()
    report.figure(fig, 'alfven_frequency.png')
    return summary


def uniform_refinement(report):
    """Error envelopes while stiffness scales as 1/h; includes exact resonances."""
    rows = []
    strengths = (.01, .1, 1., 10., 100., 1000.)
    for kind in ('acoustic', 'alfven'):
        for eps in EPSILONS:
            for n in (8, 16, 32, 64):
                h, time = .4/n, .4
                cases = [(nh, oh) for nh in ((0., *strengths) if kind=='alfven' else strengths)
                         for oh in ((0., *strengths, np.pi/(1+eps), 2*np.pi/(1+eps))
                                    if kind=='alfven' else (0.,))]
                envelopes = {(c, state): 0. for c in CANDIDATES for state in ('slow_mode', 'slip')}
                for nh, oh in cases:
                    F, S, T, P, inv, ns, fields, states = fourier_system(kind, eps, nh/h, oh/h)
                    exact = expm(time*F)
                    for c in CANDIDATES:
                        numerical = inv @ constant_result(S, T, ns, P, h, n, c)
                        for state, initial in states.items():
                            error = float(np.linalg.norm((numerical-exact) @ initial)/np.linalg.norm(initial))
                            envelopes[c, state] = max(envelopes[c, state], error)
                for (c, state), error in envelopes.items():
                    row = dict(family='F4_uniform', kind=kind, eps=eps, candidate=c,
                               state=state, h=h, steps=n, error=error, error_over_h2=error/h**2)
                    rows.append(row)
                    report.runs.append(row)
    summary = []
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for i, kind in enumerate(('acoustic', 'alfven')):
        for j, eps in enumerate(EPSILONS):
            ax = axes[i, j]
            for c in CANDIDATES:
                for state in ('slow_mode', 'slip'):
                    rs = [r for r in rows if r['kind']==kind and r['eps']==eps
                          and r['candidate']==c and r['state']==state]
                    orders = np.log2(np.array([r['error'] for r in rs[:-1]])/
                                     np.array([r['error'] for r in rs[1:]]))
                    summary.append(dict(kind=kind, eps=eps, candidate=c, state=state,
                                        orders=orders.tolist(), max_error_over_h2=max(r['error_over_h2'] for r in rs)))
                    if c == 'Cprime':
                        report.check(f'F4 uniform {kind}/eps={eps:g}/{state}/Cprime order',
                                     min(orders), 1.85, op='>=')
                    ax.loglog([r['h'] for r in rs], [r['error'] for r in rs],
                              '-' if state=='slow_mode' else '--', color=COLORS[c], label=c+'/'+state)
            ax.set(title=f'{kind}, eps={eps:g}', xlabel='h (nu*h, omega*h fixed)',
                   ylabel='maximum state error over stiffness')
            ax.set_xticks([rs[-1]['h'], rs[0]['h']],
                          labels=[f'{rs[-1]["h"]:g}', f'{rs[0]["h"]:g}'])
            ax.minorticks_off()
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=7)
    report.figure(fig, 'uniform_refinement.png')
    return summary


def gas_stability(report):
    y = np.linspace(0, 2., 201)
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    zero = np.zeros((1, 1), complex)
    for c in CANDIDATES:
        amplitudes = np.array([abs(constant_result(zero, np.array([[1j*v]]), 1,
                                                  np.ones(1), 1., 1, c)[0]) for v in y])
        exact = abs(1+1j*y-y*y/2-1j*y**3/6)
        report.check('gas '+c+' RK3 polynomial', np.max(abs(amplitudes-exact)), 3e-14)
        report.check('gas '+c+' stable for |kh|<=1.7', np.max(amplitudes[y<=1.7]), 1+2e-14)
        ax.plot(y, amplitudes, color=COLORS[c], label=c)
    ax.axhline(1., color='k', lw=.7)
    ax.axvline(np.sqrt(3), color='k', ls=':', label='sqrt(3)')
    ax.set(xlabel='imaginary wave eigenvalue * h', ylabel='one-step amplitude', ylim=(.9, 1.25))
    ax.legend()
    report.figure(fig, 'gas_stability.png')


def write_report(report, local, changing, fast, summary, uniform):
    peak = {c: max(r['max_sweep_error'] for r in summary if r['candidate']==c) for c in CANDIDATES}
    lines = ['# Coupled exponential integrator decision study', '',
             'Recommendation: use Cprime (C\u2032) as the next coupled-integrator prototype: analytically',
             'integrate transport interactions along',
             'the exponential source path. C fixes displacement but retains collisionless feedback error.',
             'Retain the current production solver until the nonlinear flux and energy gates below pass.',
             'This study does not certify any',
             'candidate as a uniformly accurate, conservative and realizable nonlinear PDE integrator.', '',
             '## Construction and scope', '',
             'A = Strang / SSP-RK3 transport; B = consult-2 three-stage ETD with the complete residual;',
             'C = the same tableau with triangular fast-to-slow exponential blocks. The script docstring',
             'derives C and its exact integrated-slip formula. Cprime uses three analytic Picard waveform',
             'iterations, derived in the same docstring; it does not exponentiate the full coupled matrix.',
             'Reference D is an independent dense matrix',
             'exponential (including polynomial/quadratic lifts for forcing and heat), never C itself.',
             'F1 independently validates the exact homogeneous source primitive, then checks every candidate',
             'on autonomous affine source lifts. Candidate updates use dense study generators, not a call to',
             'the production source routine or a full nonlinear solver. F3 additionally tests',
             'the coupled tableau with transport and a work row; these are distinct checks.', '',
             'Ambiguities resolved explicitly for architect review:',
             '- “Second order” is interpreted as at least second order. B and C are classically third order',
             '  on smooth affine systems; stiff order is measured separately.',
             '- C integrates exponential fast-to-slow coupling analytically, including linear-in-time forcing.',
             '  Its final residual fit has an additional quadratic term from the requested three-stage tableau.',
             '- C is a constant-coefficient semidiscrete predictor. Extending its B operator through spatially',
             '  varying local phi blocks requires a conservative face construction; that is outside this testbed.',
             '- Covariance cross terms are tested in the Gaussian state. The existing four-central-variable',
             '  HyQMOM local block is gyrotropic and cannot represent covariance cross terms; it is tested with',
             '  nonzero skewness and kurtosis. No unimplemented 12-moment closure is silently assumed.',
             '- No general nonlinear error theorem follows from a finite stiffness interval or gas-only stability.',
             '- For constant linear splits, Cprime has a uniform Volterra truncation bound when the source',
             '  semigroup and transport operator are bounded; this does not cover nonlinear variable geometry.',
             '- Scite personal returned its monthly quota limit. Cprime is derived here, with no literature',
             '  attribution or unverified claim of being an existing published scheme.', '',
             '## Measured evidence', '',
             f'Maximum normalized F1 local-map error: {max(local.values()):.6g}.',
             f'Maximum F3 C displacement/velocity error: {fast["max_C_error"]:.6g}.',
             f'Maximum F3 Cprime displacement/velocity error: {fast["max_Cprime_error"]:.6g}.',
             'F3 at resonant integer periods tests repeated steps at a fixed physical final time.',
             'The half-period work test has zero physical drag heat; apparent heat below uses the',
             'coupled tableau work minus the change of relative kinetic energy.', '',
             '| Candidate | Half-period work error | Apparent heat | Worst F4 sweep state error |',
             '|---|---:|---:|---:|']
    for c in CANDIDATES:
        w = fast['coupled_work'][c]
        lines.append(f'| {c} | {w["work_error"]:.6g} | {w["apparent_heat"]:.6g} | {peak[c]:.6g} |')
    lines += ['', 'F4 uses k=c_g=B0=rho_g=1, eps=0.01,1,10; h=1/32 and final time 1.',
              'nu*h and omega*h independently span 0.01 through 1000 (11 logarithmic values each).',
              'Additional Alfven frequencies give relative gyro angles pi, 2pi, 4pi and 20pi.',
              'Acoustic states store both normalized density perturbations and both velocities;',
              'Alfven states store complex gas/dust velocity and delta_B. Both a unit-norm prepared',
              'slow eigenmode and a slip-bearing state are tested. metrics.json contains individual',
              'field amplitude, phase, absolute and normalized errors, and every refinement-pair slope.',
              'Phase is null where either amplitude is below 1e-14 of the initial state norm.', '',
              'Fixed-stiffness refinement holds nu=20000, omega=nu for Alfven (zero for acoustic),',
              'final time=0.4, and uses 8 through 131072 macrosteps (nu*h=1000 down to 0.061).',
              'The table shows eps=1; all loadings and every refinement-pair order are in metrics.json.',
              'Errors below 5e-11 are marked as a numerical floor in F4 slope estimates.', '',
              '| System | State | Candidate | Stiff pair order | Transition order | Finest order |',
              '|---|---|---|---:|---:|---:|']
    def fmt(x):
        return 'numerical floor' if x is None else f'{x:.3f}'
    for r in summary:
        if r['eps'] == 1.:
            lines.append(f'| {r["kind"]} | {r["state"]} | {r["candidate"]} | '
                         f'{fmt(r["stiff_pair_order"])} | {fmt(r["transition_order"])} | {fmt(r["finest_order"])} |')
    lines += ['', 'F2 smooth, changing-coefficient finest-pair orders:', '',
              '| Case | A | B | C | Cprime |', '|---|---:|---:|---:|---:|']
    for name in dict.fromkeys(key.split('/')[0] for key in changing):
        lines.append('| '+name+' | '+' | '.join(fmt(changing[name+'/'+c]) for c in CANDIDATES)+' |')
    lines += ['', 'Uniform-refinement envelopes additionally hold nu*h and omega*h fixed while h shrinks.',
              'They use final time 0.4; n=8,16,32,64; stiffness in {0.01,0.1,1,10,100,1000},',
              'plus zero drag and zero frequency and relative gyro angles pi,2pi for Alfven.',
              'The finest-order range below covers eps=0.01,1,10 for each state. This is measured',
              'uniform-order evidence on linear systems, not a proof for every stiffness or nonlinear PDE.', '',
              '| System | State | Candidate | Finest envelope order range | Max error / h^2 |',
              '|---|---|---|---:|---:|']
    for kind in ('acoustic', 'alfven'):
        for state in ('slow_mode', 'slip'):
            for c in CANDIDATES:
                rs = [r for r in uniform if r['kind']==kind and r['state']==state and r['candidate']==c]
                orders = [r['orders'][-1] for r in rs]
                lines.append(f'| {kind} | {state} | {c} | {min(orders):.3f} to {max(orders):.3f} | '
                             f'{max(r["max_error_over_h2"] for r in rs):.6g} |')
    lines += ['', '## Remaining gates', '',
              '- Exact local work and nonnegative physical drag heat are demonstrated for frozen coefficients',
              '  with linear forcing. They do not establish a shared nonlinear conservative total-energy flux.',
              '- Fixed-frequency smooth convergence is separate from uniform error for changing, noncommuting',
              '  magnetic geometry. F2 uses moderate rates; stiff rotating-field accuracy is not qualified.',
              '- C does not yet prove a uniform second-order bound, positivity, realizability, or cold/vacuum',
              '  handling for nonlinear Gaussian/HyQMOM PDE transport. No fallback or subcycling is hidden here.',
              '- The uniform sweep rejects A and B on unprepared fast states. C removes the leading displacement',
              '  error but still has first-order collisionless feedback error; Cprime resolves that defect in',
              '  the affine systems tested here. Its analytic transport integrals need a new face-flux realization',
              '  before replacing the production integrator.',
              '- SGS forcing here is prescribed linear-in-time D, with an exact work ledger. Evolving density',
              '  and a nonlinear depleting SGS reservoir require further manufactured coupled tests.',
              '- F5 (real 2D dust transport and diamagnetic cancellation) is deferred. This linear testbed',
              '  cannot establish compatible discrete magnetization currents or nonlinear face realizability.',
              '- Ideal MHD current partition only; nonideal electric work and joint magnetic-moment closure',
              '  remain separate model questions from the integrator.', '',
              '## Artifacts and reproduction', '',
              '`PYTHONPATH=. python scripts/dust_validation/ap_integrator_study.py`', '',
              'Expected candidate defects are recorded as measurements, not failed implementation checks.',
              'Checks require exact primitives, at least second smooth order, C/Cprime counterexample exactness,',
              'and the RK3 gas stability polynomial. See metrics.json for runtime, memory, commit, hashes',
              'and all check thresholds. Existing solver behavior is unchanged.', '']
    for name in report.figures:
        lines.append(f'- [{name}]({name})')
    lines += ['', '![Uniform refinement](uniform_refinement.png)', '']
    (report.out/'report.md').write_text('\n'.join(lines))


def main():
    report = Report('ap_integrator_study')
    # Report's shared hashes cover production files. Include the experiment itself.
    report.notes.append('Float64/complex128, NumPy backend; no source-scale subcycling, filters, '
                        'limiters, or positivity repairs. Dense matrices are tiny reference/testbed blocks.')
    for name in ('phrike/coupled_exponential.py', 'scripts/dust_validation/ap_integrator_study.py'):
        path = Path(name)
        report.notes.append(name+' sha256='+hashlib.sha256(path.read_bytes()).hexdigest())
    local = family1(report)
    changing = family2(report)
    fast = family3(report)
    summary = family4(report)
    uniform = uniform_refinement(report)
    gas_stability(report)
    write_report(report, local, changing, fast, summary, uniform)
    return report.finish()


if __name__ == '__main__':
    raise SystemExit(main())
