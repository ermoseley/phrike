"""Hall-term viability measurements. Run ONLY on a cluster; no output files.

1D: NumPy CPU. 2D: NumPy CPU or torch CUDA. The `mri` suite is always 1D NumPy.
All numerical work, including the small analytic eigenproblems, happens at run time.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phrike.equations import MHDEquations1D, MHDEquations2D
from phrike.grid import Grid1D, Grid2D
from phrike.nonideal import Hall
from phrike.problems.problem_list.hall_mri1d import hall_mri_fastest, hall_mri_matrix, hall_mri_mode
from phrike.solver import SpectralSolverMHD1D, SpectralSolverMHD2D


def host(a):
    return a.detach().cpu().numpy() if hasattr(a, 'detach') else np.asarray(a)


def array(a, g):
    if g._use_torch:
        import torch
        return torch.as_tensor(a, dtype=g.x.dtype, device=g.x.device)
    return np.asarray(a)


def sync(g):
    if g._use_torch and g.x.device.type == 'cuda':
        import torch
        torch.cuda.synchronize(g.x.device)


def make_grid(args, L=2*np.pi):
    kw = dict(dealias=True, backend=args.backend, torch_device=args.device if args.backend == 'torch' else None,
              precision='double', fft_workers=1)
    if args.dim == 1:
        return Grid1D(N=args.n, Lx=L, **kw)
    return Grid2D(Nx=args.n, Ny=args.n, Lx=L, Ly=L, **kw)


def coords(g):
    if hasattr(g, 'Ny'):
        return np.meshgrid(host(g.x), host(g.y), indexing='xy')
    return host(g.x), 0*host(g.x)


def build(g, cfl=0.15, cls=None, **blocks):
    """Solver for a grid; each given block (hall, ambipolar, shearing_sheet) is enabled."""
    g.mhd_config = {name: dict(cfg, enabled=True) for name, cfg in blocks.items()}
    two = hasattr(g, 'Ny')
    cls = cls or (SpectralSolverMHD2D if two else SpectralSolverMHD1D)
    return cls(grid=g, equations=(MHDEquations2D if two else MHDEquations1D)(), cfl=cfl)


def state(g, eq, B, rho=None, u=None, p=None):
    shape = B.shape[1:]
    rho = np.ones(shape) if rho is None else rho
    u = np.zeros_like(B) if u is None else u
    p = np.ones(shape) if p is None else p
    return eq.conservative(*[array(a, g) for a in (rho, *u, p, *B)])


def advance(g, step, U, end, dt, hall=None, sample=None):
    """Fixed-dt loop with step(U, h) -> U; returns the state and timing/Hall statistics."""
    if hall is not None:
        hall.reset_stats()
    t, steps = 0.0, 0
    sync(g)
    start = time.perf_counter()
    if sample:
        sample(t, U)
    while t < end-1e-14:
        h = min(dt, end-t)
        U = step(U, h)
        t += h
        steps += 1
        if sample:
            sample(t, U)
    sync(g)
    info = dict(seconds=time.perf_counter()-start, steps=steps)
    if hall is not None:
        total = hall.total
        info.update(substeps=total['substeps'], cg_mean=total['cg_iterations']/max(total['cg_calls'], 1),
                    cg_max=total['cg_max_iterations'], cg_residual=total['cg_residual'])
    return U, info


class Results:
    def __init__(self):
        self.failed = 0

    def check(self, name, ok, **numbers):
        self.failed += not bool(ok)
        fields = ' '.join(f'{k}={v:.6g}' if isinstance(v, (float, np.floating))
                          else f'{k}={v}' for k, v in numbers.items())
        print(f'{"PASS" if ok else "FAIL"} {name}: {fields}', flush=True)


def linear_modes(k, B0, eta, length):
    """Independent six-variable compressible Hall-MHD Fourier symbol.

    q=(delta rho, u_x,u_y,u_z,b_1,b_2), rho0=|B0|=1, cs=0.5.
    The inertia length regularizes only the Hall subflow, so eta_eff replaces
    eta in the usual Hall-MHD polynomial; ideal induction and momentum are unchanged.
    """
    kn = np.linalg.norm(k)
    e1 = np.cross([0., 0., 1.], k/kn)
    e2 = np.cross(k/kn, e1)
    P = np.array([e1, e2]).T
    effective = eta/(1+length**2*kn**2)
    M = np.zeros((6, 6), complex)
    for j in range(6):
        q = np.eye(6)[:, j]
        rho, u, b = q[0], q[1:4], P@q[4:6]
        M[0, j] = -1j*np.dot(k, u)
        M[1:4, j] = -1j*k*0.25*rho + 1j*(np.dot(k, B0)*b-k*np.dot(B0, b))
        db = 1j*(np.dot(k, B0)*u-B0*np.dot(k, u))
        db = db + effective*np.dot(k, B0)*np.cross(k, b)
        M[4:6, j] = P.T@db
    values, vectors = np.linalg.eig(M)
    ids = sorted((i for i in range(6) if values[i].imag > 1e-8), key=lambda i: values[i].imag)
    c2 = (np.dot(k, B0)/kn)**2
    h2 = (effective*kn)**2*c2
    roots = np.sort(np.sqrt(np.sort(np.roots([1, -(1.25+c2+h2),
                                              0.25*h2+1.5*c2, -0.25*c2*c2]).real)))*kn
    return [(values[i].imag, vectors[:, i]/np.linalg.norm(vectors[:, i])) for i in ids], P, roots


class WaveMode:
    """Eigenmode of ideal + Hall MHD (rho0=|B0|=1, cs=0.5, k=1 along x or (1,1)) and its exact
    linear evolution; length > 0 gives the regularized-linear mode of the inertia method."""

    def __init__(self, args, g, branch, eta, length=0.):
        X, Y = coords(g)
        self.shape = [1]*args.dim
        self.k = np.array([1., 1. if args.dim == 2 else 0., 0.])
        self.B0 = np.array([1., 0., 0.])
        self.phase = self.k[0]*X+self.k[1]*Y
        modes, self.P, _ = linear_modes(self.k, self.B0, eta, length)
        # Lowest and highest branch: ion cyclotron / whistler for parallel k.
        self.omega, self.v = modes[0 if branch == 'ion-cyclotron' else -1]
        self.background = np.broadcast_to(self.B0.reshape(3, *self.shape), (3, *X.shape)).copy()

    def fields(self, amp, t=0.):
        q = amp*np.real(self.v.reshape(6, *self.shape)*np.exp(1j*(self.phase+self.omega*t)))
        return q, self.background+np.einsum('ij,j...->i...', self.P, q[4:6])

    def initial(self, g, eq, amp):
        q, B = self.fields(amp)
        return state(g, eq, B, rho=1+q[0], u=q[1:4], p=0.25/eq.gamma+0.25*q[0])


def waves(args, out):
    g = make_grid(args)
    eq = (MHDEquations2D if args.dim == 2 else MHDEquations1D)()
    X, Y = coords(g)
    for oblique in (False, True):
        k = np.array([1., 1. if oblique and args.dim == 2 else 0., 0.])
        B0 = np.array([1., 0., 0.])
        if oblique and args.dim == 1:
            B0 = np.array([1., 1., 0.])/np.sqrt(2)
        phase = k[0]*X+k[1]*Y
        for method, length in [('reference', 0.), ('inertia', 0.3)]:
            cfg = dict(method=method, eta_h=3., length=length)
            hall = Hall(g, cfg)
            # Pure induction has only the whistler pair, not ion cyclotron.
            kn = np.linalg.norm(k)
            e1 = np.cross([0., 0., 1.], k/kn)
            e2 = np.cross(k/kn, e1)
            amp = 1e-6
            B = B0.reshape(3, *([1]*args.dim)) + amp*(
                e1.reshape(3, *([1]*args.dim))*np.cos(phase)
                + e2.reshape(3, *([1]*args.dim))*np.sin(phase))
            U = state(g, eq, B)
            omega = 3*np.dot(k, B0)*kn/(1+length**2*kn**2)
            times, phases = [], []

            def sample(t, V):
                b = host(V[5:8])-B0.reshape(3, *([1]*args.dim))
                z = np.mean(np.einsum('i,i...->...', e1, b)*np.exp(-1j*phase))
                times.append(t)
                phases.append(np.angle(z))

            V, _ = advance(g, lambda W, h: hall.step(W, eq, h), U, 1/omega, 0.025/omega, sample=sample)
            measured = np.polyfit(times, np.unwrap(phases), 1)[0]
            out.check(f'induction {method} oblique={oblique}', abs(measured/omega-1) < 2e-4,
                      omega=measured, analytic=omega,
                      divB=np.max(np.abs(host(g.divergence(V[5], V[6])))) if args.dim == 2
                      else np.max(np.abs(host(g.dx1(V[5])))))
            modes, P, roots = linear_modes(k, B0, 3., length)
            error = np.max(np.abs(np.array([m[0] for m in modes])-roots))
            out.check(f'dispersion polynomial {method} oblique={oblique}',
                      error < 1e-10, max_error=error)
            for branch, j in [('ion-cyclotron', 0), ('whistler', -1)]:
                omega, v = modes[j]
                base = np.exp(1j*phase)
                dq = amp*np.real(v.reshape(6, *([1]*args.dim))*base)
                b = np.einsum('ij,j...->i...', P, dq[4:6])
                B = B0.reshape(3, *([1]*args.dim)) + b
                s = build(g, hall=cfg)
                U = state(g, s.equations, B, rho=1+dq[0], u=dq[1:4], p=0.25/s.equations.gamma+0.25*dq[0])
                times, phases = [], []

                def sample_mode(t, V):
                    rho, ux, uy, uz, _, bx, by, bz = [host(a) for a in s.equations.primitive(V)]
                    b = np.array([bx, by, bz])-B0.reshape(3, *([1]*args.dim))
                    q = np.concatenate((np.array([rho-1, ux, uy, uz]),
                                        np.einsum('ij,j...->i...', P.T, b)))
                    z = np.mean(np.einsum('i,i...->...', v.conj(), q)*np.exp(-1j*phase))
                    times.append(t)
                    phases.append(np.angle(z))

                dt = min(s.compute_dt(U), 0.025/omega)
                V, _ = advance(g, s.step, U, 0.8/omega, dt, hall=s.hall, sample=sample_mode)
                measured = np.polyfit(times, np.unwrap(phases), 1)[0]
                true_modes, _, _ = linear_modes(k, B0, 3., 0.)
                out.check(f'{branch} {method} oblique={oblique}', abs(measured/omega-1) < 0.01,
                          omega=measured, analytic=omega, true_hall=true_modes[j][0])


def invariants(args, out):
    g = make_grid(args)
    eq = (MHDEquations2D if args.dim == 2 else MHDEquations1D)()
    X, Y = coords(g)
    for force_free in (True, False):
        B = np.array([0*X, np.cos(X), np.sin(X)] if force_free
                     else [0*X, 1+0.1*np.cos(X), 0*X])
        for method in ('reference', 'inertia'):
            cfg = dict(method=method, eta_h=-1., length=0.3, length_mode='local')
            hall = Hall(g, cfg)
            U = state(g, eq, B, rho=1+0.2*np.cos(X+Y))
            V, _ = advance(g, lambda W, h: hall.step(W, eq, h), U, 0.1, 0.01)
            err = np.max(np.abs(host(V-U)))
            out.check(f'equilibrium {method} force_free={force_free}', err < 2e-12, drift=err,
                      electric_max=np.max(np.abs(host(hall.electric_field(U[5:8], U[0])))))
    # Nonlinear, divergence-free multimode field; Y dependence exercises the 2D curl.
    B = np.array([1+0.04*np.sin(Y), 0.1*np.cos(X), 0.1*np.sin(X)+0.03*np.cos(2*Y+X)])
    for method, local in [('reference', False), ('inertia', False), ('inertia', True)]:
        cfg = dict(method=method, eta_h=0.3, eta_h_form='proportional_to_B_over_rho',
                   length=0.3, length_mode='local' if local else 'constant',
                   cg_tol=min(args.cg_tol, 1e-11))
        hall = Hall(g, cfg)
        U = state(g, eq, B, rho=1+0.2*np.cos(X+Y))
        l2 = hall.length_squared(U, 0.01) if method == 'inertia' else 0.
        initial = hall.modified_energy(U[5:8], l2)
        V, stats = advance(g, lambda W, h: hall.step(W, eq, h), U, 0.2, 0.01, hall=hall)
        drift = abs(hall.modified_energy(V[5:8], l2)/initial-1)
        dp = np.max(np.abs(host(eq.primitive(V)[4]-eq.primitive(U)[4])))
        out.check(f'energy {method} local={local}', drift < 1e-11 and dp < 1e-12,
                  relative_drift=drift, pressure_drift=dp, **stats)
    hall = Hall(g, dict(method='inertia', eta_h=1.))
    rhs = hall.curl(U[5:8])
    diagonal, _, _ = hall.inertia_solve(rhs, 0.09)
    iterative, it, residual = hall.inertia_solve(rhs, 0*U[0]+0.09, tol=1e-11)
    err = np.max(np.abs(host(diagonal-iterative)))
    out.check('constant Fourier vs CG', err < 1e-10, max_error=err, cg_iterations=it, residual=residual)


def shear_mode(n, L, floor):
    """Wood et al. (2014), eq. 62, k_x=1, B_x=n_e=f(y).

    sigma^2 b_y = [d_y^2 - 1 - (log f)''] b_y. Fourier Galerkin
    in y is independent of the time stepper and Hall/inertia implementation.
    For the floor-free Gaussian, (log f)''=-2 and sigma=1, b_y=constant.
    """
    y = np.arange(n)*L/n-L/2
    gaussian = np.exp(-y*y)
    f = gaussian+floor
    if floor:
        q = gaussian/f
        potential = -2*q+4*y*y*q*(floor/f)
    else:
        potential = np.full(n, -2.)
    modes = np.arange(-(n//3), n//3+1)
    ky = 2*np.pi*modes/L
    vhat = np.fft.fft(potential)/n
    H = np.diag(-1-ky*ky)-vhat[(modes[:, None]-modes[None, :]) % n]
    values, vectors = np.linalg.eigh(H)
    value, vector = values[-1], vectors[:, -1]
    residual = np.linalg.norm(H@vector-value*vector)
    if value <= 0:
        raise ValueError('Configured shear profile has no growing k_x=1 mode')
    bhat = np.zeros(n, complex)
    bhat[modes % n] = vector
    by = np.fft.ifft(bhat)*n
    bhat *= np.exp(-1j*np.angle(by[n//2]))
    by = (np.fft.ifft(bhat)*n).real
    derivative = (np.fft.ifft(1j*2*np.pi*np.fft.fftfreq(n, L/n)*bhat)*n).real
    scale = np.max(np.abs(by))
    return np.sqrt(value), by/scale, derivative/scale, residual


def shear(args, out):
    L = 2*np.pi*args.shear_box_periods
    g = make_grid(args, L=L)
    eq = MHDEquations2D()
    X, Y = coords(g)
    Y = Y-L/2
    f = np.exp(-Y**2)+args.shear_floor
    target, by, dby, residual = shear_mode(args.n, L, args.shear_floor)
    coarse, _, _, _ = shear_mode(args.n//2, L, args.shear_floor)
    gaussian, _, _, _ = shear_mode(args.n, L, 0.)
    out.check('shear independent y eigenproblem', abs(target/coarse-1) < 0.01
              and residual < 1e-10 and abs(gaussian-1) < 1e-12,
              sigma=target, coarse_sigma=coarse, floor_free_sigma=gaussian,
              residual=residual, floor=args.shear_floor, box=L)
    if args.shear_seed == 'eigenmode':
        B = np.array([f-1e-7*dby[:, None]*np.sin(X),
                      1e-7*by[:, None]*np.cos(X), -1e-7*target*by[:, None]*np.cos(X)])
    else:
        B = np.array([f, 1e-7*np.cos(X), 0*X])
    growths, seconds_A = [], None
    # length None: automatic l^2 from the local |eta_H| and dt (informational, no accuracy gate).
    for method, length, local, ratio in [('reference', 0., False, None), ('inertia', 0.05, False, None),
                                         ('inertia', 0.2, False, None), ('inertia', 0.5, False, None),
                                         ('inertia', 0.2, True, None), ('inertia', None, True, 4.)]:
        auto = length is None
        cfg = dict(method=method, eta_h=1., eta_h_form='proportional_to_B_over_rho',
                   length_mode='local' if local else 'constant', cg_tol=args.cg_tol)
        if not auto:
            cfg['length'] = length
        if ratio is not None:
            cfg['length_ratio'] = ratio
        hall = Hall(g, cfg)
        U = state(g, eq, B, rho=f, p=0.55*(1+args.shear_floor)**2-0.5*f**2)
        l2 = hall.length_squared(U, args.shear_dt) if method == 'inertia' else 0.
        energy = hall.modified_energy(U[5:8], l2)
        # Derivative-weighted metric: both curl components, not just B_y energy.
        weight = np.sum(host(hall.curl(U[5:8]-array(np.array([f, 0*X, 0*X]), g)))**2, axis=0)
        effective_l = np.sqrt(np.sum(host(l2)*weight)/np.sum(weight))
        ts, logs = [], []

        def sample(t, V):
            if args.shear_fit_start-1e-12 <= t <= args.shear_fit_end+1e-12:
                ts.append(t)
                logs.append(np.log(np.sqrt(np.mean(host(V[6])**2))))

        step = lambda W, h: hall.step(W, eq, h)
        step(U, args.shear_dt)  # warm kernels without changing the initial state
        V, stats = advance(g, step, U, args.shear_end, args.shear_dt, hall=hall, sample=sample)
        growth = np.polyfit(ts, logs, 1)[0]
        middle = len(ts)//2
        first = np.polyfit(ts[:middle], logs[:middle], 1)[0]
        last = np.polyfit(ts[middle:], logs[middle:], 1)[0]
        growths.append(growth)
        if seconds_A is None:
            seconds_A = stats['seconds']
        drift = abs(hall.modified_energy(V[5:8], l2)/energy-1)
        pressure = np.max(np.abs(host(eq.primitive(V)[4]-eq.primitive(U)[4])))
        # Large lengths are comparisons, not an assertion of unchanged physics.
        ok = abs(growth/target-1) < 0.01 if method == 'reference' else np.isfinite(growth)
        if auto:
            # The automatic metric follows |B|, so the initial-l^2 energy is only approximately invariant.
            ok = ok and np.isfinite(drift) and pressure < 1e-12
        else:
            ok = ok and abs(first/last-1) < 0.01 and drift < 1e-8 and pressure < 1e-12
        if method == 'inertia' and length == 0.05:
            ok = ok and abs(growth/growths[0]-1) < 0.05
        out.check(f'density shear {method} l/a={"auto" if auto else length} local={local} ratio={ratio}', ok,
                  sigma=growth, analytic=target, floor_free=1., ratio_to_A=growth/growths[0],
                  sigma_first=first, sigma_last=last, seed=args.shear_seed,
                  effective_l=effective_l, l_min=np.sqrt(np.min(host(l2))),
                  l_max=np.sqrt(np.max(host(l2))), A_over_B=seconds_A/stats['seconds'],
                  energy_drift=drift, pressure_drift=pressure, cg_tol=args.cg_tol, **stats)


def cost(args, out):
    g = make_grid(args)
    eq = (MHDEquations2D if args.dim == 2 else MHDEquations1D)()
    X, _ = coords(g)
    B = np.array([1+0*X, 1e-5*np.cos(X), 1e-5*np.sin(X)])
    for ratio in (1., 10., 100.):
        eta = ratio*g.dx  # |B0|=rho0=v_A=1, l_H=eta/v_A.
        cfg = dict(method='reference', eta_h=eta)
        U = state(g, eq, B)
        dt = build(g, cfl=0.04).compute_dt(U)
        cfg_B = dict(cfg, method='inertia')
        l2 = Hall(g, cfg_B).length_squared(U, dt)
        cfg_B['length'] = math.sqrt(l2)  # Fixed metric for this timed comparison.
        measurements = []
        for config in (cfg, cfg_B):
            hall = Hall(g, config)
            step = lambda W, h: hall.step(W, eq, h)
            step(U, dt)  # warm FFT plans and kernels outside the timed interval
            V, stats = advance(g, step, U, 5*dt, dt, hall=hall)
            z = np.mean((host(V[6])+1j*host(V[7]))*np.exp(-1j*X))
            phase = np.angle(z/1e-5)
            true_phase = eta*5*dt/np.sqrt(1+1e-10)
            error = abs(phase/true_phase-1)
            measurements.append((stats['seconds']/(5*dt), error))
            out.check(f'cost {config["method"]} l_H/dx={ratio}', error < 0.02,
                      seconds_per_time=measurements[-1][0], relative_phase_error=error,
                      length=math.sqrt(l2) if config['method'] == 'inertia' else 0., **stats)
        print(f'MEASURE l_H/dx={ratio:g} A_over_B={measurements[0][0]/measurements[1][0]:.6g} '
              '(shared 2% resolved-mode phase-error budget)', flush=True)


def observed_orders(errors):
    return [math.log2(a/max(b, 1e-300)) for a, b in zip(errors[:-1], errors[1:])]


def listing(values):
    return ' '.join(f'{v:.3g}' for v in values)


class ForwardSplit(SpectralSolverMHD2D):
    """Hall then ambipolar both before and after the ideal step: not palindromic."""

    def _fixed_step(self, U, dt):
        g, h, eq = self.grid, 0.5*dt, self.equations
        V = self.ambipolar.step(self.hall.step(g.apply_spectral_dissipation(U, h), eq, h), eq, h)
        V = self._project_B(self._rk_substages(V, dt))
        V = self.ambipolar.step(self.hall.step(V, eq, h), eq, h)
        return g.apply_accepted_state_stabilization(V, h)


def convergence(args, out):
    """Coupled ideal+Hall Strang step, fixed dt, B-field error at a fixed final time."""
    g = make_grid(args)
    eta, length, amp = 3., 0.3, 1e-7
    dts = [args.conv_dt0/2**i for i in range(4)]
    T = 40*args.conv_dt0
    scale = lambda mode, B: np.linalg.norm(B-mode.background)

    def final_B(mode, cfg, dt, a, cls=None, **blocks):
        s = build(g, cls=cls, hall=cfg, **blocks)
        V, _ = advance(g, s.step, mode.initial(g, s.equations, a), T, dt, hall=s.hall)
        return host(V[5:8])

    for branch in ('ion-cyclotron', 'whistler'):
        true, regular = WaveMode(args, g, branch, eta), WaveMode(args, g, branch, eta, length)
        exact, exact_l = true.fields(amp, T)[1], regular.fields(amp, T)[1]
        tag = f'{branch} dim={args.dim} T={T:g} omega={true.omega:.4g}'
        # (a) reference RK4 Hall vs the true-Hall linear solution.
        errors = [np.linalg.norm(final_B(true, dict(method='reference', eta_h=eta), dt, amp)-exact)
                  / scale(true, exact) for dt in dts]
        orders = observed_orders(errors)
        out.check(f'convergence (a) reference vs true Hall {tag}', orders[-1] > 1.8,
                  finest_order=orders[-1], orders=listing(orders), errors=listing(errors))
        # (b) inertia with a fixed length vs the regularized-linear solution.
        cfg = dict(method='inertia', eta_h=eta, length=length)
        errors = [np.linalg.norm(final_B(regular, cfg, dt, amp)-exact_l)
                  / scale(regular, exact_l) for dt in dts]
        orders = observed_orders(errors)
        out.check(f'convergence (b) inertia fixed length vs regularized {tag}', orders[-1] > 1.8,
                  finest_order=orders[-1], orders=listing(orders), errors=listing(errors))
        # (c) automatic length, l^2 proportional to dt: first order to the true Hall solution.
        cfg = dict(method='inertia', eta_h=eta)
        U = true.initial(g, build(g).equations, amp)
        l2 = [Hall(g, cfg).length_squared(U, 0.5*dt) for dt in dts]
        errors = [np.linalg.norm(final_B(true, cfg, dt, amp)-exact) / scale(true, exact) for dt in dts]
        orders = observed_orders(errors)
        out.check(f'convergence (c) inertia automatic length vs true Hall {tag}',
                  0.8 <= orders[-1] <= 1.3, finest_order=orders[-1], orders=listing(orders),
                  errors=listing(errors), length2=listing(l2))
    if args.dim != 2:
        print('SKIP convergence (d): ambipolar requires --dim 2', flush=True)
        return
    # (d) Hall + ambipolar: the operators do not commute, so the order of the two half-steps
    # matters. Self-convergence of a nonlinear oblique whistler, palindromic vs forward order.
    mode = WaveMode(args, g, 'whistler', eta)
    cfg, ad = dict(method='reference', eta_h=eta), dict(eta=0.5, midpoint_tol=1e-12)
    s = build(g, hall=cfg, ambipolar=ad)
    U, h = mode.initial(g, s.equations, 0.1), args.conv_dt0
    forward = s.ambipolar.step(s.hall.step(U, s.equations, h), s.equations, h)
    backward = s.hall.step(s.ambipolar.step(U, s.equations, h), s.equations, h)
    rms = lambda B: np.sqrt(np.mean(B**2))
    commutator = rms(host(forward-backward)[5:8])/rms(host(U)[5:8]-mode.background)
    out.check('convergence (d) Hall and ambipolar do not commute for this state',
              commutator > 1e-4, relative_commutator=commutator)

    def self_orders(cls):
        B = [final_B(mode, cfg, dt, 0.1, cls=cls, ambipolar=ad) for dt in dts]
        diffs = [rms(a-b)/rms(b-mode.background) for a, b in zip(B[:-1], B[1:])]
        return observed_orders(diffs), diffs

    orders, diffs = self_orders(None)
    out.check(f'convergence (d) palindromic Hall+ambipolar self-convergence dim={args.dim}',
              orders[-1] > 1.8, finest_order=orders[-1], orders=listing(orders), diffs=listing(diffs))
    orders, diffs = self_orders(ForwardSplit)
    print(f'MEASURE (d) forward (non-palindromic) order: orders={listing(orders)} diffs={listing(diffs)}',
          flush=True)


def decomposed_step(s, U, dt):
    """The Hall + ideal solver step split into measured pieces; reproduces s.step(U, dt).

    W = mean(E) + reservoir(l^2) is what a Hall call conserves for its frozen l^2; the ideal
    (and shearing) step changes mean(E) and the reservoir; a change of l^2 between the two Hall
    calls changes the reservoir at fixed B.
    """
    g, hall, eq, half = s.grid, s.hall, s.equations, 0.5*dt
    inertia = hall.method == 'inertia'
    reservoir = lambda V, l2: hall.reservoir_energy(V[5:8], l2) if inertia else 0.
    mean_E = lambda V: float(V[4].mean())
    V = g.apply_spectral_dissipation(U, half)
    l2a = hall.length_squared(V, half) if inertia else 0.
    A = hall.step(V, eq, half)
    hall_1 = mean_E(A)+reservoir(A, l2a)-mean_E(V)-reservoir(V, l2a)
    B = s._project_B(s._rk_substages(A, dt))
    ideal = mean_E(B)-mean_E(A)
    l2b = hall.length_squared(B, half) if inertia else 0.
    metric = reservoir(B, l2b)-reservoir(B, l2a)
    C = hall.step(B, eq, half)
    hall_2 = mean_E(C)+reservoir(C, l2b)-mean_E(B)-reservoir(B, l2b)
    Un = g.apply_accepted_state_stabilization(C, half)
    if s.dim == 1:
        Un = g.apply_boundary_conditions(Un, eq)
    return Un, dict(hall=max(abs(hall_1), abs(hall_2)), ideal=ideal, metric=metric,
                    reservoir=reservoir(Un, hall.length_squared(Un, half) if inertia else 0.))


def energy(args, out):
    """Coupled ideal+Hall energy, with the Hall-subflow, ideal and metric-change parts separated."""
    g = make_grid(args)
    eta, amp, dt, T = 3., 0.1, 0.025, 2.
    mode = WaveMode(args, g, 'whistler', eta)
    # A small safety factor keeps the RK4 truncation of a Hall call at roundoff.
    for name, cfg in [('reference', dict(method='reference', eta_h=eta, safety=0.1)),
                      ('inertia length=0.3', dict(method='inertia', eta_h=eta, length=0.3, safety=0.1)),
                      ('inertia automatic', dict(method='inertia', eta_h=eta, safety=0.1))]:
        s = build(g, hall=cfg)
        U = mode.initial(g, s.equations, amp)
        E0 = float(U[4].mean())
        times, E, R = [0.], [E0], [0.]
        parts = dict(hall=0., ideal=0., metric=0.)
        first = True
        t = 0.
        while t < T-1e-14:
            V, piece = decomposed_step(s, U, dt)
            if first:
                same = np.max(np.abs(host(V-s.step(U, dt))))
                out.check(f'decomposed step reproduces the solver step {name}', same < 1e-13, max_diff=same)
                first = False
            U, t = V, t+dt
            parts['hall'] = max(parts['hall'], piece['hall']/E0)
            parts['ideal'] += piece['ideal']/E0
            parts['metric'] += piece['metric']/E0
            times.append(t)
            E.append(float(U[4].mean()))
            R.append(piece['reservoir'])
        drift = np.array(E)/E0-1
        reservoir = np.max(np.array(R))/E0
        secular = np.polyfit(times, drift, 1)[0]*T
        ok = (parts['hall'] < 1e-12 and np.max(np.abs(drift)) <= 2*reservoir+1e-9
              and abs(secular) <= reservoir+1e-9)
        out.check(f'coupled energy {name} dim={args.dim}', ok,
                  hall_subflow_max=parts['hall'], ideal_sum=parts['ideal'], metric_sum=parts['metric'],
                  max_drift_E=np.max(np.abs(drift)), secular_drift=secular, reservoir_max=reservoir)


def mri_polynomial(k, omega, shear, field, rho0, eta, length):
    """Roots of s^4 + s^2 (2 w^2 + kappa^2 + g) + [...] = 0 for h = eta sgn(B0) k^2/(1 + l^2 k^2)."""
    x = k*k*field*field/rho0
    h = eta*np.sign(field)*k*k/(1+(length*k)**2)
    kappa2, g = 2*omega*(2*omega-shear), h*(h-shear)
    return np.roots([1, 0, 2*x+kappa2+g, 0, x*x+x*(4*omega*h-2*omega*shear-h*shear)+kappa2*g])


def mri_amplitude(V, n):
    return 2*math.hypot(abs(np.fft.fft(host(V[6]))[1])/n, abs(np.fft.fft(host(V[7]))[1])/n)


def mri(args, out):
    """Hall-MRI eigenmode (1D NumPy, vertical x, radial y, azimuthal z) vs the linear problem."""
    field, rho0 = 1.0, 1.0
    cases = [('ideal Keplerian', 1.0, 1.5, 0.0)]
    for X in (4., 1., 0.25):   # X = v_A^2/(eta_H Omega)
        cases += [(f'Keplerian X={X:g} sign(eta Omega.B)={sign:+d}', 1.0, 1.5, sign*field**2/(X*1.0))
                  for sign in (1, -1)]
    for X in (0.5, 0.25):      # Hall-shear: X = v_A^2/(eta_H S), no rotation
        cases += [(f'Hall-shear X={X:g} sign(eta S.B)={sign:+d}', 0.0, 1.0, sign*field**2/(X*1.0))
                  for sign in (1, -1)]
    variants = args.mri_variants.split(',')
    for label, omega, shear_rate, eta in cases:
        k, s, v = hall_mri_fastest(omega, shear_rate, field, rho0, eta)
        roots = mri_polynomial(k, omega, shear_rate, field, rho0, eta, 0.)
        eigen = np.linalg.eigvals(hall_mri_matrix(k, omega, shear_rate, field, rho0, eta))
        match = max(min(abs(r-e) for e in eigen) for r in roots)
        out.check(f'mri linear {label}', match < 1e-8,
                  k_fastest=k, growth=s.real, frequency=s.imag, polynomial_vs_matrix=match)
        if s.real < 1e-3*max(abs(omega), abs(shear_rate)):
            print(f'MEASURE mri {label}: stable in the scanned range (max growth {s.real:.3g})', flush=True)
            continue
        L, n = 2*np.pi/k, args.mri_n
        g = Grid1D(N=n, Lx=L, dealias=True, backend='numpy', precision='double', fft_workers=1)
        eq = MHDEquations1D()
        sheet = dict(omega=omega, shear=shear_rate, vertical_axis=0, radial_axis=1)
        beta = args.mri_beta
        x = host(g.x)
        one = np.ones_like(x)

        def initial(vec, amp):
            wave = amp*abs(field)*np.real(vec[:, None]*np.exp(1j*k*x)[None, :])
            return state(g, eq, np.array([field*one, wave[2], wave[3]]), rho=rho0*one,
                         u=np.array([0*one, wave[0], wave[1]]), p=0.5*beta*field**2*one)

        dt = build(g, cfl=args.mri_cfl, shearing_sheet=sheet).compute_dt(initial(v, 0.))
        reference_seconds = None
        for variant in variants:
            if eta == 0 and variant != 'reference':
                continue
            if variant == 'reference':
                cfg, length = dict(method='reference', eta_h=eta), 0.
            elif variant == 'auto':
                cfg = dict(method='inertia', eta_h=eta)
                length = math.sqrt(Hall(g, cfg).length_squared(initial(v, 0.), 0.5*dt))
            else:
                length = float(variant)/k
                cfg = dict(method='inertia', eta_h=eta, length=length)
            s_reg, v_reg = hall_mri_mode(k, omega, shear_rate, field, rho0, eta, length)
            if s_reg.real <= 0:
                print(f'MEASURE mri {label} {variant}: regularized mode is stable', flush=True)
                continue
            end = math.ceil(args.mri_efolds/s_reg.real/dt)*dt
            solver = build(g, cfl=args.mri_cfl, hall=dict(cfg, eta_h_form='constant'), shearing_sheet=sheet)
            ts, logs = [], []

            def sample(t, V):
                ts.append(t)
                logs.append(math.log(mri_amplitude(V, n)))

            V, stats = advance(g, solver.step, initial(v_reg, args.mri_amp), end, dt, hall=solver.hall,
                               sample=sample)
            ts, logs = np.array(ts), np.array(logs)
            window = ts >= 0.4*end
            t_w, l_w = ts[window], logs[window]
            half = len(t_w)//2
            growth = np.polyfit(t_w, l_w, 1)[0]
            first, last = np.polyfit(t_w[:half], l_w[:half], 1)[0], np.polyfit(t_w[half:], l_w[half:], 1)[0]
            target = s.real if variant == 'reference' else s_reg.real
            if variant == 'reference':
                reference_seconds = stats['seconds']
            kl2 = (k*length)**2
            ok = abs(growth/target-1) < 0.01 and abs(first/last-1) < 0.01
            out.check(f'mri {label} {variant}', ok, growth=growth, linear=target, true_hall_linear=s.real,
                      growth_over_target=growth/target, half_fits=f'{first:.5g}/{last:.5g}',
                      error_vs_true_hall=growth/s.real-1, regularization_error=s_reg.real/s.real-1,
                      k_l_squared=kl2, length=length, dt=dt, speedup_vs_reference=(
                          reference_seconds/stats['seconds'] if reference_seconds else float('nan')),
                      **stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='all', choices=['waves', 'invariants', 'convergence', 'energy',
                                                           'cost', 'shear', 'mri', 'all'])
    parser.add_argument('--dim', type=int, choices=[1, 2], default=1)
    parser.add_argument('--n', type=int, default=None, help='grid points per axis (default 128)')
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='numpy')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--cg-tol', type=float, default=1e-10)
    parser.add_argument('--shear-seed', choices=['eigenmode', 'uniform'], default='eigenmode')
    parser.add_argument('--shear-floor', type=float, default=0.01)
    parser.add_argument('--shear-box-periods', type=int, default=3)
    parser.add_argument('--shear-end', type=float, default=10.)
    parser.add_argument('--shear-fit-start', type=float, default=6.)
    parser.add_argument('--shear-fit-end', type=float, default=10.)
    parser.add_argument('--shear-dt', type=float, default=0.02)
    parser.add_argument('--conv-dt0', type=float, default=0.05, help='coarsest dt of the convergence suite (T = 40 dt0)')
    parser.add_argument('--mri-n', type=int, default=32, help='grid points of the 1D mri box (one wavelength)')
    parser.add_argument('--mri-beta', type=float, default=400., help='plasma beta 2p/B0^2 of the mri box')
    parser.add_argument('--mri-amp', type=float, default=1e-7, help='mri seed amplitude |b|/|B0|')
    parser.add_argument('--mri-efolds', type=float, default=10., help='mri run length in regularized e-foldings')
    parser.add_argument('--mri-cfl', type=float, default=0.3)
    parser.add_argument('--mri-variants', default='reference,auto,0.05,0.2,0.5',
                        help='comma list: reference, auto, or fixed k*l values for inertia')
    args = parser.parse_args()
    if args.n is None:
        args.n = 128
    if args.n < 24:
        parser.error('--n must be at least 24')
    if not (np.isfinite(args.cg_tol) and args.cg_tol > 0):
        parser.error('--cg-tol must be finite and positive')
    if not (np.isfinite(args.shear_floor) and args.shear_floor > 0 and args.shear_box_periods >= 2
            and all(np.isfinite(t) for t in (args.shear_dt, args.shear_fit_start,
                                           args.shear_fit_end, args.shear_end)) and args.shear_dt > 0
            and 0 <= args.shear_fit_start < args.shear_fit_end <= args.shear_end
            and args.shear_fit_end-args.shear_fit_start >= 8*args.shear_dt):
        parser.error('Invalid shear floor, box, timestep or fit window')
    if args.dim == 1 and args.backend != 'numpy':
        parser.error('Use NumPy for the 1D suite; torch CUDA is exercised in 2D.')
    if args.suite == 'shear' and args.dim != 2:
        parser.error('Density shear requires --dim 2')
    if not (np.isfinite(args.conv_dt0) and args.conv_dt0 > 0):
        parser.error('--conv-dt0 must be finite and positive')
    if args.backend == 'torch' and args.device != 'cuda':
        parser.error('The torch comparison targets --device cuda.')
    if not (args.mri_n >= 24 and args.mri_beta > 0 and args.mri_amp > 0 and args.mri_efolds >= 5):
        parser.error('Invalid mri grid, beta, amplitude or e-folding count')
    for variant in args.mri_variants.split(','):
        if variant not in ('reference', 'auto') and not float(variant) > 0:
            parser.error('--mri-variants entries must be reference, auto or a positive k*l')
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True,
                                   stderr=subprocess.DEVNULL).strip():
            revision += '+dirty'
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown (no git)'
    print(f'Hall viability REVISION={revision} backend={args.backend} '
          f'device={args.device if args.backend == "torch" else "cpu"} '
          f'dim={args.dim} N={args.n} precision=double numpy={np.__version__}', flush=True)
    for name in ('phrike/nonideal.py', 'phrike/solver.py', 'phrike/grid.py', 'phrike/equations.py',
                 'phrike/problems/problem_list/hall_mri1d.py', 'scripts/hall_viability.py'):
        print(f'SOURCE {name} sha256={hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}', flush=True)
    if args.backend == 'torch':
        import torch
        print(f'torch={torch.__version__} GPU={torch.cuda.get_device_name()}', flush=True)
    result = Results()
    for name, run in [('waves', waves), ('invariants', invariants), ('shear', shear), ('cost', cost),
                      ('convergence', convergence), ('energy', energy), ('mri', mri)]:
        if name == 'shear' and args.dim == 1:
            if args.suite == 'all':
                print('SKIP shear: requires --dim 2', flush=True)
            continue
        if args.suite in (name, 'all'):
            run(args, result)
    print(f'{"PASS" if not result.failed else "FAIL"} summary: {result.failed} failed checks', flush=True)
    return int(bool(result.failed))


if __name__ == '__main__':
    raise SystemExit(main())
