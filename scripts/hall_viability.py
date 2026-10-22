"""Hall-term viability measurements. Run ONLY on a cluster; no output files except --ot-cache and --stab-cache.

1D: NumPy CPU. 2D: NumPy CPU or torch CUDA. The `mri` suite is always 1D NumPy and the `adsi`
suite always 2D NumPy (shearing-sheet ambipolar diffusion). Only `waves` also runs in 3D. `ot`,
`stability` and `precision` are 2D torch CUDA suites; they, `adsi` and `h2h` (head-to-head of Hall
time-integration strategies: 1D NumPy waves and mri, 2D shear on the chosen backend) are not part
of `--suite all`.
All numerical work, including the small analytic eigenproblems, happens at run time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from phrike.equations import MHDEquations1D, MHDEquations2D, MHDEquations3D
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.io import load_config
from phrike.nonideal import Hall
from phrike.problems.problem_list.adsi_shear import (
    adsi_incompressible_roots, adsi_shear_box, adsi_shear_fastest, adsi_shear_fields, adsi_shear_left,
    adsi_shear_matrix, adsi_shear_mode, adsi_shear_vector)
from phrike.problems.problem_list.hall_mri1d import hall_mri_fastest, hall_mri_matrix, hall_mri_mode
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d
from phrike.solver import SpectralSolverMHD1D, SpectralSolverMHD2D, SpectralSolverMHD3D


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


def make_grid(args, L=2*np.pi, precision='double'):
    kw = dict(dealias=True, backend=args.backend, torch_device=args.device if args.backend == 'torch' else None,
              precision=precision, fft_workers=1)
    if args.dim == 1:
        return Grid1D(N=args.n, Lx=L, **kw)
    if args.dim == 3:
        return Grid3D(Nx=args.n, Ny=args.n, Nz=args.n, Lx=L, Ly=L, Lz=L, **kw)
    return Grid2D(Nx=args.n, Ny=args.n, Lx=L, Ly=L, **kw)


def coords(g):
    """(X, Y) for 1D and 2D grids (Y = 0 in 1D), (X, Y, Z) of shape (Nz, Ny, Nx) in 3D."""
    if hasattr(g, 'Nz'):
        Z, Y, X = np.meshgrid(host(g.z), host(g.y), host(g.x), indexing='ij')
        return X, Y, Z
    if hasattr(g, 'Ny'):
        return np.meshgrid(host(g.x), host(g.y), indexing='xy')
    return host(g.x), 0*host(g.x)


def equations(g):
    return (MHDEquations3D if hasattr(g, 'Nz') else MHDEquations2D if hasattr(g, 'Ny') else MHDEquations1D)()


def div_b(g, V):
    d = (g.divergence(V[5], V[6], V[7]) if hasattr(g, 'Nz') else g.divergence(V[5], V[6])
         if hasattr(g, 'Ny') else g.dx1(V[5]))
    return float(np.max(np.abs(host(d))))


def build(g, cfl=0.15, cls=None, **blocks):
    """Solver for a grid; each given block (hall, ambipolar, shearing_sheet) is enabled."""
    g.mhd_config = {name: dict(cfg, enabled=True) for name, cfg in blocks.items()}
    cls = cls or (SpectralSolverMHD3D if hasattr(g, 'Nz') else SpectralSolverMHD2D if hasattr(g, 'Ny')
                  else SpectralSolverMHD1D)
    return cls(grid=g, equations=equations(g), cfl=cfl)


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


def transverse_basis(k):
    """Orthonormal (e1, e2) with e1 x e2 = k/|k|; e1 is horizontal, so cross(z, k) is renormalized."""
    n = k/np.linalg.norm(k)
    e1 = np.cross([0., 0., 1.], n)
    e1 = e1/np.linalg.norm(e1)
    return e1, np.cross(n, e1)


def linear_modes(k, B0, eta, length):
    """Independent six-variable compressible Hall-MHD Fourier symbol.

    q=(delta rho, u_x,u_y,u_z,b_1,b_2), rho0=|B0|=1, cs=0.5.
    The inertia length regularizes only the Hall subflow, so eta_eff replaces
    eta in the usual Hall-MHD polynomial; ideal induction and momentum are unchanged.
    """
    kn = np.linalg.norm(k)
    P = np.array(transverse_basis(k)).T
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
    eq = equations(g)
    X, Y, *rest = coords(g)
    Z = rest[0] if rest else 0*X
    for oblique in (False, True):
        # Oblique: k = (1, 1) in 2D and (1, 1, 1) in 3D, with B0 along x.
        k = np.array([1., 1. if oblique and args.dim >= 2 else 0., 1. if oblique and args.dim == 3 else 0.])
        B0 = np.array([1., 0., 0.])
        if oblique and args.dim == 1:
            B0 = np.array([1., 1., 0.])/np.sqrt(2)
        phase = k[0]*X+k[1]*Y+k[2]*Z
        for method, length in [('reference', 0.), ('inertia', 0.3)]:
            cfg = dict(method=method, eta_h=3., length=length)
            hall = Hall(g, cfg)
            # Pure induction has only the whistler pair, not ion cyclotron.
            kn = np.linalg.norm(k)
            e1, e2 = transverse_basis(k)
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
                      divB=div_b(g, V))
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
    print('NOTE cost: Hall-only steps under a shared phase-error ceiling, not equal-error coupled '
          'performance', flush=True)
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
    calls changes the reservoir at fixed B. ``stabilization`` is the change of mean(E) in the
    two accepted-state steps (spectral dissipation, dealias projection), which filter the total
    energy field E rather than internal energy; mean(E) is the k = 0 mode and is untouched.
    """
    g, hall, eq, half = s.grid, s.hall, s.equations, 0.5*dt
    inertia = hall.method == 'inertia'
    reservoir = lambda V, l2: hall.reservoir_energy(V[5:8], l2) if inertia else 0.
    mean_E = lambda V: float(V[4].mean())
    V = g.apply_spectral_dissipation(U, half)
    stabilization = mean_E(V)-mean_E(U)
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
    stabilization += mean_E(Un)-mean_E(C)
    if s.dim == 1:
        Un = g.apply_boundary_conditions(Un, eq)
    return Un, dict(hall=max(abs(hall_1), abs(hall_2)), ideal=ideal, metric=metric,
                    stabilization=stabilization,
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
        parts = dict(hall=0., ideal=0., metric=0., stabilization=0.)
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
            parts['stabilization'] += piece['stabilization']/E0
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
                  stabilization_sum=parts['stabilization'],
                  max_drift_E=np.max(np.abs(drift)), secular_drift=secular, reservoir_max=reservoir)
    energy_sweep(args, out)


def energy_sweep(args, out):
    """Coupled physical-energy drift of a multimode state vs fixed l, per unit perturbation energy.

    A Hall call at fixed l conserves mean(E) + reservoir, so each call moves 0.5 l^2 |curl B|^2 in
    or out of mean(E): the coupled drift is O(l^2) while l = 0 (reference) conserves it at the
    level of the ideal step. The accepted-state steps filter the total energy field E, which leaves
    the k = 0 mode mean(E) unchanged but moves energy between thermal and other parts.
    """
    g = make_grid(args)
    eq = equations(g)
    X, Y = coords(g)[:2]
    eta, dt, T, k_ref = 3., 0.025, 2., 1.
    B = np.array([1+0.04*np.sin(Y), 0.1*np.cos(X), 0.1*np.sin(X)+0.03*np.cos(2*Y+X)])
    u = np.array([0.03*np.sin(Y), 0.03*np.cos(X+Y), 0.02*np.sin(2*X)])
    rho = 1+0.2*np.cos(X+Y)
    axes = tuple(range(1, B.ndim))
    perturbation = (0.5*np.mean(((B-B.mean(axis=axes, keepdims=True))**2).sum(axis=0))
                    + 0.5*np.mean(rho*(u**2).sum(axis=0)))
    print('NOTE energy sweep: drift is max |mean(E) - mean(E0)| / perturbation energy; the accepted-state '
          'stabilization filters the total energy field E, so its mean(E) change is roundoff by construction',
          flush=True)
    rows = []
    for name, length in [('reference', None)] + [(f'k*l={kl:g}', kl/k_ref) for kl in (0.4, 0.2, 0.1, 0.05)]:
        cfg = (dict(method='reference', eta_h=eta, safety=0.1) if length is None
               else dict(method='inertia', eta_h=eta, length=length, safety=0.1))
        s = build(g, hall=cfg)
        U = state(g, eq, B, rho=rho, u=u, p=0.3*np.ones_like(X))
        E0 = float(U[4].mean())
        parts = dict(hall=0., ideal=0., metric=0., stabilization=0.)
        reservoir = peak = 0.
        t = 0.
        while t < T-1e-14:
            U, piece = decomposed_step(s, U, dt)
            t += dt
            peak = max(peak, abs(float(U[4].mean())-E0))
            parts['hall'] = max(parts['hall'], piece['hall'])
            for key in ('ideal', 'metric', 'stabilization'):
                parts[key] += piece[key]
            reservoir = max(reservoir, piece['reservoir'])
        rows.append((name, length, peak/perturbation))
        out.check(f'energy sweep {name} dim={args.dim}', np.isfinite(peak),
                  physical_drift=peak/perturbation, hall_subflow_max=parts['hall']/perturbation,
                  ideal_sum=parts['ideal']/perturbation, metric_sum=parts['metric']/perturbation,
                  stabilization_sum=parts['stabilization']/perturbation,
                  reservoir_max=reservoir/perturbation, perturbation_energy=perturbation)
    drifts = np.array([r[2] for r in rows[1:]])
    lengths = np.array([r[1] for r in rows[1:]])
    orders = observed_orders(drifts)  # l halves along the list, so these are drift orders in l
    fit = np.polyfit(np.log(lengths), np.log(np.maximum(drifts, 1e-300)), 1)[0]
    out.check(f'energy sweep drift scales as l^2 dim={args.dim}', fit > 1.7,
              fitted_order=fit, halving_orders=listing(orders), drifts=listing(drifts))
    out.check(f'energy sweep reference drift is below the smallest-l drift dim={args.dim}',
              rows[0][2] < 0.1*drifts[-1], reference_drift=rows[0][2], smallest_l_drift=drifts[-1])


def mri_polynomial(k, omega, shear, field, rho0, eta, length):
    """Roots of s^4 + s^2 (2 w^2 + kappa^2 + g) + [...] = 0 for h = eta sgn(B0) k^2/(1 + l^2 k^2)."""
    x = k*k*field*field/rho0
    h = eta*np.sign(field)*k*k/(1+(length*k)**2)
    kappa2, g = 2*omega*(2*omega-shear), h*(h-shear)
    return np.roots([1, 0, 2*x+kappa2+g, 0, x*x+x*(4*omega*h-2*omega*shear-h*shear)+kappa2*g])


def mri_amplitude(V, n):
    return 2*math.hypot(abs(np.fft.fft(host(V[6]))[1])/n, abs(np.fft.fft(host(V[7]))[1])/n)


def mri_vector(V, eq, n):
    """Complex box-mode amplitudes c of (u_r, u_phi, b_r, b_phi), field = Re(c exp(i k x)); radial y, azimuthal z."""
    _, _, uy, uz, _, _, by, bz = [host(a).astype(np.float64) for a in eq.primitive(V)]
    return np.array([2*np.fft.fft(a)[1]/n for a in (uy, uz, by, bz)])


def mri_state(g, eq, k, vec, amp, field, rho0, beta):
    """Uniform vertical field plus the real wave Re(amp |B0| vec exp(i k x)) in (u_r, u_phi, b_r, b_phi)."""
    x = host(g.x).astype(np.float64)
    one = np.ones_like(x)
    wave = amp*abs(field)*np.real(vec[:, None]*np.exp(1j*k*x)[None, :])
    return state(g, eq, np.array([field*one, wave[2], wave[3]]), rho=rho0*one,
                 u=np.array([0*one, wave[0], wave[1]]), p=0.5*beta*field**2*one)


def mri_run(args, g, eq, k, params, cfg, vec, amp, end, dt, tick=None):
    """Evolve a seed under shearing sheet + Hall block cfg; returns times, complex amplitudes, stats."""
    omega, shear_rate, field, rho0 = params
    sheet = dict(omega=omega, shear=shear_rate, vertical_axis=0, radial_axis=1)
    solver = build(g, cfl=args.mri_cfl, hall=dict(cfg, eta_h_form='constant'), shearing_sheet=sheet)
    ts, cs = [], []

    def sample(t, V):
        ts.append(t)
        cs.append(mri_vector(V, eq, g.N))
        if tick:
            tick(t, V)

    _, stats = advance(g, solver.step, mri_state(g, eq, k, vec, amp, field, rho0, args.mri_beta),
                       end, dt, hall=solver.hall, sample=sample)
    return np.array(ts), np.array(cs), stats


def mri_duration(s, rate, efolds, dt, cap=math.inf):
    """Run length, a multiple of dt: efolds of growth, else four oscillation periods."""
    if s.real > 1e-3*rate:
        length = min(efolds/s.real, cap)
    else:
        length = 4*2*np.pi/max(abs(s), 0.1*rate)
    return math.ceil(length/dt)*dt


def complex_rate(ts, a):
    """Least-squares growth and frequency of a(t) ~ exp((growth + i frequency) t)."""
    return (np.polyfit(ts, np.log(np.abs(a)), 1)[0], np.polyfit(ts, np.unwrap(np.angle(a)), 1)[0])


def mri_polarization(c, v):
    """Norm of the part of the unit vector c/|c| orthogonal to the eigenvector direction v."""
    w, u = c/np.linalg.norm(c), v/np.linalg.norm(v)
    return float(np.linalg.norm(w-u*np.vdot(u, w)))


def mri_left(k, omega, shear_rate, field, rho0, eta, length):
    """Left eigenvector of the leading eigenvalue (same ordering as hall_mri_mode)."""
    values, vectors = np.linalg.eig(hall_mri_matrix(k, omega, shear_rate, field, rho0, eta, length))
    return np.linalg.inv(vectors)[int(np.argmax(values.real))]


def mri_marginal(args, out, label, omega, shear_rate, field, rho0, eta, k_fast):
    """Growth at k values around the neutral wavenumbers, true Hall, vs the eigenproblem."""
    rate = max(abs(omega), abs(shear_rate))
    scale = max(abs(omega), abs(shear_rate), 1e-12)/(abs(field)/math.sqrt(rho0))
    ks = np.linspace(4*scale/801, 4*scale, 801)
    growth = lambda q: hall_mri_mode(q, omega, shear_rate, field, rho0, eta)[0].real
    curve = np.array([growth(q) for q in ks])
    threshold = 1e-3*rate
    neutral = []
    for i in np.nonzero((curve[:-1] > threshold) != (curve[1:] > threshold))[0]:
        lo, hi = ks[i], ks[i+1]
        for _ in range(40):
            mid = 0.5*(lo+hi)
            lo, hi = (mid, hi) if (growth(mid) > threshold) == (curve[i] > threshold) else (lo, mid)
        neutral.append(0.5*(lo+hi))
    tests = [0.5*k_fast, k_fast] + [f*kc for kc in neutral[:2] for f in (0.9, 1.1)]
    if not neutral:
        tests.append(2*k_fast)
    params = (omega, shear_rate, field, rho0)
    for q in sorted(tests):
        s, v = hall_mri_mode(q, omega, shear_rate, field, rho0, eta)
        g = Grid1D(N=args.mri_n, Lx=2*np.pi/q, dealias=True, backend='numpy', precision='double', fft_workers=1)
        eq = MHDEquations1D()
        sheet = dict(omega=omega, shear=shear_rate, vertical_axis=0, radial_axis=1)
        dt = build(g, cfl=args.mri_cfl, shearing_sheet=sheet).compute_dt(mri_state(g, eq, q, v, 0., field, rho0, args.mri_beta))
        end = mri_duration(s, rate, args.mri_efolds, dt, cap=300.)
        ts, cs, stats = mri_run(args, g, eq, q, params, dict(method='reference', eta_h=eta), v, args.mri_amp, end, dt)
        measured, frequency = complex_rate(ts, cs @ mri_left(q, omega, shear_rate, field, rho0, eta, 0.))
        reference = max(abs(s), 0.1*rate)
        ok = abs(measured-s.real) < 1e-2*reference and abs(frequency-s.imag) < args.mri_tol*reference
        out.check(f'mri marginal curve {label} k={q:.4g}', ok, growth=measured, linear=s.real,
                  frequency=frequency, linear_frequency=s.imag, neutral_k=listing(neutral),
                  growth_error_over_scale=(measured-s.real)/reference, **stats)


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
    generic_variants = [v for v in variants if v in ('reference', '0.2')] or variants[:1]
    rng = np.random.default_rng(20240611)
    tol = args.mri_tol
    for label, omega, shear_rate, eta in cases:
        k, s, v = hall_mri_fastest(omega, shear_rate, field, rho0, eta)
        roots = mri_polynomial(k, omega, shear_rate, field, rho0, eta, 0.)
        eigen = np.linalg.eigvals(hall_mri_matrix(k, omega, shear_rate, field, rho0, eta))
        match = max(min(abs(r-e) for e in eigen) for r in roots)
        out.check(f'mri linear {label}', match < 1e-8,
                  k_fastest=k, growth=s.real, frequency=s.imag, polynomial_vs_matrix=match)
        rate = max(abs(omega), abs(shear_rate))
        stable = s.real < 1e-3*rate
        if stable and args.mri_quick:
            print(f'MEASURE mri {label}: stable in the scanned range (max growth {s.real:.3g})', flush=True)
            continue
        L, n = 2*np.pi/k, args.mri_n
        g = Grid1D(N=n, Lx=L, dealias=True, backend='numpy', precision='double', fft_workers=1)
        eq = MHDEquations1D()
        sheet = dict(omega=omega, shear=shear_rate, vertical_axis=0, radial_axis=1)
        params = (omega, shear_rate, field, rho0)
        beta = args.mri_beta
        dt = build(g, cfl=args.mri_cfl, shearing_sheet=sheet).compute_dt(
            mri_state(g, eq, k, v, 0., field, rho0, beta))
        reference_seconds = None
        for variant in variants:
            if eta == 0 and variant != 'reference':
                continue
            if variant == 'reference':
                cfg, length = dict(method='reference', eta_h=eta), 0.
            elif variant == 'auto':
                cfg = dict(method='inertia', eta_h=eta)
                length = math.sqrt(Hall(g, cfg).length_squared(mri_state(g, eq, k, v, 0., field, rho0, beta), 0.5*dt))
            else:
                length = float(variant)/k
                cfg = dict(method='inertia', eta_h=eta, length=length)
            s_reg, v_reg = hall_mri_mode(k, omega, shear_rate, field, rho0, eta, length)
            grows = s_reg.real > 1e-3*rate
            if not grows and args.mri_quick:
                print(f'MEASURE mri {label} {variant}: regularized mode is stable', flush=True)
                continue
            end = mri_duration(s_reg, rate, args.mri_efolds, dt)
            ts, cs, stats = mri_run(args, g, eq, k, params, cfg, v_reg, args.mri_amp, end, dt)
            ref = int(np.argmax(np.abs(v_reg)))
            big = np.abs(v_reg) > 0.05*np.abs(v_reg).max()
            window = ts >= 0.4*end
            t_w = ts[window]
            polarization = max(mri_polarization(c, v_reg) for c in cs[window])
            relative = (cs[-1]/cs[-1][ref])/(v_reg/v_reg[ref])
            phase_error, ratio_error = (np.max(np.abs(np.angle(relative[big]))),
                                        np.max(np.abs(np.abs(relative[big])-1)))
            frequency = np.polyfit(t_w, np.unwrap(np.angle(cs[window, ref])), 1)[0]
            kl2 = (k*length)**2
            if grows:
                logs = np.log(np.hypot(np.abs(cs[:, 2]), np.abs(cs[:, 3])))
                l_w = logs[window]
                half = len(t_w)//2
                growth = np.polyfit(t_w, l_w, 1)[0]
                first, last = np.polyfit(t_w[:half], l_w[:half], 1)[0], np.polyfit(t_w[half:], l_w[half:], 1)[0]
                target = s.real if variant == 'reference' else s_reg.real
                if variant == 'reference':
                    reference_seconds = stats['seconds']
                ok = (abs(growth/target-1) < 0.01 and abs(first/last-1) < 0.01 and polarization < tol
                      and abs(frequency-s_reg.imag) < tol*abs(s_reg))
                out.check(f'mri {label} {variant}', ok, growth=growth, linear=target, true_hall_linear=s.real,
                          growth_over_target=growth/target, half_fits=f'{first:.5g}/{last:.5g}',
                          error_vs_true_hall=growth/s.real-1, regularization_error=s_reg.real/s.real-1,
                          polarization_error=polarization, phase_error=phase_error, ratio_error=ratio_error,
                          frequency=frequency, linear_frequency=s_reg.imag,
                          k_l_squared=kl2, length=length, dt=dt, speedup_vs_reference=(
                              reference_seconds/stats['seconds'] if reference_seconds else float('nan')),
                          **stats)
            else:
                # Stable oscillation: the amplitude stays bounded and rotates at the eigenfrequency.
                norms = np.linalg.norm(cs, axis=1)
                bounded = norms.max()/norms[0]-1
                frequency = complex_rate(ts, cs[:, ref])[1]
                ok = (bounded < 1e-3 and polarization < tol
                      and abs(frequency-s_reg.imag) < tol*max(abs(s_reg), 0.1*rate))
                out.check(f'mri stable {label} {variant}', ok, amplitude_growth=bounded,
                          frequency=frequency, linear_frequency=s_reg.imag, linear_growth=s_reg.real,
                          polarization_error=polarization, phase_error=phase_error,
                          ratio_error=ratio_error, k_l_squared=kl2, length=length, dt=dt, **stats)
            if args.mri_quick or variant not in generic_variants:
                continue
            # Generic seed: random complex combination of the four components. The projection
            # onto the left eigenvector of the leading mode evolves as exp(s t) exactly, so the
            # whole run (early times too) fixes its growth and frequency.
            seed = rng.standard_normal(4)+1j*rng.standard_normal(4)
            ts, cs, stats = mri_run(args, g, eq, k, params, cfg, seed, args.mri_amp, end, dt)
            projection = cs @ mri_left(k, omega, shear_rate, field, rho0, eta, length)
            measured, frequency = complex_rate(ts, projection)
            reference = max(abs(s_reg), 0.1*rate)
            ok = abs(measured-s_reg.real) < 1e-2*reference and abs(frequency-s_reg.imag) < tol*reference
            out.check(f'mri generic seed {label} {variant}', ok, growth=measured, linear=s_reg.real,
                      frequency=frequency, linear_frequency=s_reg.imag,
                      growth_error_over_scale=(measured-s_reg.real)/reference, length=length, **stats)
        if not args.mri_quick:
            mri_marginal(args, out, label, omega, shear_rate, field, rho0, eta, k)


def adsi_params(args, omega, shear, bphi, am):
    """Background of an ADSI case: B0 = (1, 0, bphi), rho0 = 1 and eta = 1/(rho0 Am |S|) (zero for Am = inf)."""
    return dict(omega=omega, shear=shear, bv=1., bphi=bphi, rho0=1., p0=0.5*args.adsi_beta*(1+bphi**2),
                gamma=5./3., eta=0. if math.isinf(am) else 1./(am*abs(shear)))


def adsi_grid(args, kv, kr):
    Lx, Ly, mv, mr = adsi_shear_box(kv, kr)
    return Grid2D(Nx=args.adsi_n, Ny=args.adsi_n, Lx=Lx, Ly=Ly, dealias=True, backend='numpy',
                  precision='double', fft_workers=1), mv, mr


def adsi_state(g, par, kv, kr, vec, amp):
    """Uniform state plus the real wave amp |B0| Re(fields exp(i (kv x + kr y))) of a mode vector."""
    X, Y = coords(g)
    wave = amp*math.hypot(par['bv'], par['bphi'])*np.real(
        adsi_shear_fields(vec, kv, kr)[:, None, None]*np.exp(1j*(kv*X+kr*Y))[None])
    one = np.ones_like(X)
    return state(g, MHDEquations2D(), np.array([par['bv']*one+wave[4], wave[5], par['bphi']*one+wave[6]]),
                 rho=par['rho0']*one+wave[0], u=wave[1:4], p=par['p0']*one+wave[7])


def adsi_vector(V, kv, kr, mv, mr):
    """Complex amplitudes of the seven mode variables at box mode (mv, mr)."""
    rho, ux, uy, uz, p, bx, by, bz = [host(a).astype(np.float64) for a in MHDEquations2D().primitive(V)]
    return adsi_shear_vector(np.array([rho, ux, uy, uz, bx, by, bz, p]), kv, kr, mv, mr)


def adsi_solver(args, par, g):
    blocks = dict(shearing_sheet=dict(omega=par['omega'], shear=par['shear'], vertical_axis=0, radial_axis=1))
    if par['eta'] > 0:
        blocks['ambipolar'] = dict(eta=par['eta'], method='midpoint_resolvent', midpoint_tol=1e-11)
    return build(g, cfl=args.adsi_cfl, **blocks)


def adsi_dt(args, par, kv, kr, vec):
    g, _, _ = adsi_grid(args, kv, kr)
    return adsi_solver(args, par, g).compute_dt(adsi_state(g, par, kv, kr, vec, 0.))


def adsi_run(args, par, kv, kr, vec, end, dt):
    """Evolve a seed under shearing sheet + ambipolar diffusion; returns times, amplitudes, stats."""
    g, mv, mr = adsi_grid(args, kv, kr)
    ts, cs = [], []

    def sample(t, V):
        ts.append(t)
        cs.append(adsi_vector(V, kv, kr, mv, mr))

    _, stats = advance(g, adsi_solver(args, par, g).step, adsi_state(g, par, kv, kr, vec, args.adsi_amp),
                       end, dt, sample=sample)
    return np.array(ts), np.array(cs), stats


def adsi_marginal(args, out, label, par, rate, k, psi_fast):
    """Growth near the neutral angles at fixed |k| and at half the fastest |k|, vs the eigenproblem."""
    angles = np.radians(np.linspace(-85., 85., 681))
    growth = lambda a: np.linalg.eigvals(adsi_shear_matrix(k*math.cos(a), k*math.sin(a), **par)).real.max()
    curve = np.array([growth(a) for a in angles])
    threshold = 1e-3*rate
    neutral = []
    for i in np.nonzero((curve[:-1] > threshold) != (curve[1:] > threshold))[0]:
        lo, hi = angles[i], angles[i+1]
        for _ in range(40):
            mid = 0.5*(lo+hi)
            lo, hi = (mid, hi) if (growth(mid) > threshold) == (curve[i] > threshold) else (lo, mid)
        neutral.append(0.5*(lo+hi))
    tests = [(0.5*k, psi_fast)] + [(k, a+d) for a in neutral[:2] for d in (-0.05, 0.05)]
    for q, a in tests:
        kv, kr = q*math.cos(a), q*math.sin(a)
        s, v = adsi_shear_mode(kv, kr, **par)
        dt = adsi_dt(args, par, kv, kr, v)
        end = mri_duration(s, rate, args.adsi_efolds, dt, cap=args.adsi_tmax)
        left = adsi_shear_left(kv, kr, **par)
        ts_coarse, cs_coarse, _ = adsi_run(args, par, kv, kr, v, end, dt)
        coarse, _ = complex_rate(ts_coarse, cs_coarse @ left)
        ts, cs, stats = adsi_run(args, par, kv, kr, v, end, 0.5*dt)
        measured, frequency = complex_rate(ts, cs @ left)
        # Second-order splitting: judge the Richardson-extrapolated growth, as in the main adsi gate.
        extrapolated = measured+(measured-coarse)/3.
        reference = max(abs(s), 0.1*rate)
        ok = abs(extrapolated-s.real) < 1e-2*reference and abs(frequency-s.imag) < args.adsi_tol*reference
        out.check(f'adsi marginal {label} k={q:.4g} psi={math.degrees(a):.3f}deg', ok, growth=measured,
                  extrapolated_growth=extrapolated,
                  linear=s.real, frequency=frequency, linear_frequency=s.imag,
                  neutral_psi_deg=listing(np.degrees(neutral)),
                  growth_error_over_scale=(measured-s.real)/reference, **stats)


def adsi(args, out):
    """Shearing-sheet ADSI (2D NumPy, vertical x, radial y, azimuthal z) vs the compressible eigenproblem."""
    tol, rng = args.adsi_tol, np.random.default_rng(20241007)
    s, _ = adsi_shear_mode(math.sqrt(15.)/4., 0., 1., 1.5, 1., 0., p0=100.)
    out.check('adsi linear ideal MRI limit', abs(s.real-0.75) < 1e-10 and abs(s.imag) < 1e-10,
              growth=s.real, frequency=s.imag)
    for omega, shear, bphi, eta, kv, kr in [(0., 1., 1., 2., 0.7, -1.4), (0., 1., 0.3, 3., 0.5, -3.),
                                            (1., 1.5, 1., 1., 0.6, 0.3), (1., 1.5, 1., 1., 0.6, -0.3)]:
        roots = adsi_incompressible_roots(kv, kr, omega, shear, 1., bphi, eta=eta)
        eigen = np.linalg.eigvals(adsi_shear_matrix(kv, kr, omega, shear, 1., bphi, p0=1e6, eta=eta))
        match = max(min(abs(r-e) for e in eigen) for r in roots)
        out.check(f'adsi linear incompressible quartic Omega={omega:g} tan={bphi:g} k=({kv:g},{kr:g})',
                  match < 1e-4, root_mismatch=match, growth=max(roots.real))
    worst = max(np.linalg.eigvals(adsi_shear_matrix(kv, 0., 0., 1., 1., 0.7, p0=5., eta=eta)).real.max()
                for kv in (0.3, 1., 3.) for eta in (0., 0.5, 5.))
    out.check('adsi linear pure shear with k along the vertical axis is stable', worst < 1e-9, max_growth=worst)
    cases = [(f'Omega=0 S=1 tan={t:g} sign(S Bphi Bz)={sign:+d} Am={am:g}', 0., 1., sign*t, am, sign > 0 and t == 1. and am < 2)
             for sign in (1, -1) for t in (1., 0.3) for am in (math.inf, 1., 0.3)]
    cases += [(f'Keplerian tan=1 Am={am:g}', 1., 1.5, 1., am, am == 1.) for am in (math.inf, 1., 0.3)]
    for label, omega, shear, bphi, am, marginal in cases:
        par = adsi_params(args, omega, shear, bphi, am)
        rate, k0 = max(abs(omega), abs(shear)), max(abs(omega), abs(shear))/math.hypot(1., bphi)
        k, psi, s, v = adsi_shear_fastest(**par)
        edge = not (0.1*k0*1.02 < k < 6*k0/1.02 and abs(psi) < math.radians(84.9))
        stable = s.real < 1e-3*rate
        if stable:
            # No growth anywhere in the scan: a representative mode on the favorable side of k_r.
            k, psi = 2*k0, -math.copysign(math.radians(40.), shear*bphi)
            s, v = adsi_shear_mode(k*math.cos(psi), k*math.sin(psi), **par)
        kv, kr = k*math.cos(psi), k*math.sin(psi)
        out.check(f'adsi linear {label}', np.isfinite(s) and s.real <= 0.5*abs(shear)*(1+1e-9),
                  k=k, psi_deg=math.degrees(psi), growth=s.real, frequency=s.imag, growth_bound=0.5*abs(shear),
                  scan_edge=edge, stable=stable)
        if stable and args.adsi_quick:
            print(f'MEASURE adsi {label}: stable in the scanned range (max growth {s.real:.3g})', flush=True)
            continue
        dt = adsi_dt(args, par, kv, kr, v)
        scale = max(abs(s), 0.1*rate)
        left = adsi_shear_left(kv, kr, **par)
        end = mri_duration(s, rate, args.adsi_efolds, dt, cap=args.adsi_tmax)
        if stable and s.real < -1e-9:
            end = min(end, math.ceil(8./abs(s.real)/dt)*dt)
        fits = {}
        for division in (1, 2):
            ts, cs, stats = adsi_run(args, par, kv, kr, v, end, dt/division)
            fits[division] = (*complex_rate(ts, cs @ left), ts, cs, stats)
        growth, frequency, ts, cs, stats = fits[2]
        # Strang + midpoint AD is second order in dt: gate the Richardson-extrapolated growth,
        # so stiff AD cases are judged on convergence to the eigenvalue, not on the dt/2 error.
        extrapolated = growth+(growth-fits[1][0])/3.
        errors = [fits[d][0]-s.real for d in (1, 2)]
        window = ts >= 0.4*end
        t_w, half = ts[window], window.sum()//2
        polarization = max(mri_polarization(c, v) for c in cs[window])
        compressive = [0, 1, 6]
        pol_compressive = (max(mri_polarization(c[compressive], v[compressive]) for c in cs[window])
                           if np.linalg.norm(v[compressive]) > 1e-3*np.linalg.norm(v) else 0.)
        ref = int(np.argmax(np.abs(v)))
        big = np.abs(v) > 0.05*np.abs(v).max()
        relative = (cs[-1]/cs[-1][ref])/(v/v[ref])
        numbers = dict(growth=growth, linear=s.real, frequency=frequency, linear_frequency=s.imag,
                       growth_error_over_scale=(growth-s.real)/scale,
                       extrapolated_error_over_scale=(extrapolated-s.real)/scale, polarization_error=polarization,
                       compressive_polarization_error=pol_compressive,
                       phase_error=np.max(np.abs(np.angle(relative[big]))),
                       ratio_error=np.max(np.abs(np.abs(relative[big])-1)), k=k, psi_deg=math.degrees(psi),
                       dt=dt/2, **stats)
        ok = (abs(extrapolated-s.real) < tol*scale and abs(frequency-s.imag) < tol*scale
              and polarization < tol and pol_compressive < 3*tol)
        if not stable:
            logs = np.log(np.hypot(np.abs(cs[:, 4]), np.abs(cs[:, 5])))[window]
            first, last = np.polyfit(t_w[:half], logs[:half], 1)[0], np.polyfit(t_w[half:], logs[half:], 1)[0]
            ok = ok and abs(first-last) < tol*scale
            out.check(f'adsi {label}', ok, half_fits=f'{first:.6g}/{last:.6g}', **numbers)
        else:
            norms = np.linalg.norm(cs, axis=1)
            ok = ok and norms.max()/norms[0]-1 < tol
            name = 'adsi ideal control stays neutral' if am == math.inf else 'adsi stable'
            out.check(f'{name} {label}', ok and (am != math.inf or abs(s.real) < 1e-8*rate),
                      amplitude_growth=norms.max()/norms[0]-1, **numbers)
        # Strang splitting is second order in dt: the error at dt must be about four times that at dt/2.
        order = math.log2(abs(errors[0]/errors[1])) if min(abs(e) for e in errors) > 1e-12*scale else float('nan')
        out.check(f'adsi timestep convergence {label}',
                  abs(extrapolated-s.real) < tol*scale and (abs(errors[0]) < 0.02*tol*scale or 1.5 < order < 2.6),
                  growth_error_dt=errors[0]/scale, growth_error_half_dt=errors[1]/scale, observed_order=order,
                  extrapolated_error=(errors[1]-(errors[0]-errors[1])/3.)/scale)
        if stable or args.adsi_quick:
            continue
        # Generic seed: the projection onto the left eigenvector of the leading mode evolves as exp(s t).
        seed = rng.standard_normal(7)+1j*rng.standard_normal(7)
        ts, cs, stats = adsi_run(args, par, kv, kr, seed, end, dt/2)
        measured, frequency = complex_rate(ts, cs @ left)
        # Same dt as the eigenmode run: the projection must reproduce its growth, whatever the dt error.
        out.check(f'adsi generic seed {label}', abs(measured-growth) < tol*scale and abs(frequency-s.imag) < tol*scale,
                  growth=measured, eigenmode_growth_same_dt=growth, linear=s.real, frequency=frequency,
                  linear_frequency=s.imag,
                  growth_error_over_scale=(measured-s.real)/scale, **stats)
        if marginal:
            adsi_marginal(args, out, label, par, rate, k, psi)


def truncate(a, n):
    """Spectrally truncate fields on an m x m grid (last two axes) to n x n, keeping the mean."""
    m = a.shape[-1]
    if m == n:
        return a
    idx = np.r_[0:n//2, m-n//2:m]
    F = np.fft.fft2(a)[..., idx][..., idx, :]
    return np.fft.ifft2(F).real*(n/m)**2


def relative_l2(a, b):
    return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def magnetic_spectrum(B):
    """Shell-summed magnetic energy 0.5 sum |B_hat|^2 per integer shell |k| = round(sqrt(kx^2 + ky^2))."""
    n = B.shape[-1]
    k = np.fft.fftfreq(n, 1./n)
    shell = np.rint(np.hypot(*np.meshgrid(k, k, indexing='xy'))).astype(int)
    power = 0.5*(np.abs(np.fft.fft2(B)/n**2)**2).sum(axis=0)
    return np.bincount(shell.ravel(), weights=power.ravel())


def current_magnitude(g, B):
    """Vector magnitude |curl B| (all three components) on the grid, as a host array."""
    J = host(Hall(g, dict(eta_h=1.)).curl(array(B, g)))
    return np.sqrt((J*J).sum(axis=0))


def ot_setup(args, n, hall):
    """Grid, solver and initial state of main's orszag_tang2d problem at n^2 with a Hall block.

    Grid, filter, MHD options, scheme and CFL come from configs/orszag_tang2d.yaml; the precision
    is double and the initial state is main's orszag_tang_2d with the projections of run().
    """
    cfg = load_config(str(ROOT/'configs'/'orszag_tang2d.yaml'))
    integration, gcfg = cfg['integration'], cfg['grid']
    filters = {name: integration.get(name, {'enabled': False})
               for name in ('spectral_dissipation', 'modal_filter', 'sbp_dissipation')}
    g = Grid2D(Nx=n, Ny=n, Lx=float(gcfg.get('Lx', 1.)), Ly=float(gcfg.get('Ly', 1.)),
               dealias=bool(gcfg.get('dealias', True)), filter_params=filters, fft_workers=1,
               backend=args.backend, torch_device=args.device if args.backend == 'torch' else None,
               precision='double', basis_x='fourier', basis_y='fourier')
    g.mhd_config = dict(cfg['physics']['mhd'], hall=dict(hall, enabled=True))
    gamma = float(cfg['physics']['gamma'])
    s = SpectralSolverMHD2D(grid=g, equations=MHDEquations2D(gamma=gamma),
                            scheme=str(integration.get('scheme', 'rk4')), cfl=float(integration['cfl']))
    ic = cfg.get('initial_conditions') or {}
    U = orszag_tang_2d(*g.xy_mesh(), gamma=gamma, Lx=g.Lx, Ly=g.Ly,
                       rho0=ic.get('rho0'), p0=ic.get('p0'), B0=ic.get('B0'))
    return g, s, g.project_dealiased(s._project_B(U))


REVISION = 'unknown'  # set by main; stored with every cached result


class TooExpensive(Exception):
    """A reference Hall run needs more substeps per ideal step than --max-hall-substeps-per-step."""


def hms(seconds):
    if not math.isfinite(seconds):
        return 'n/a'
    hours, rest = divmod(int(seconds), 3600)
    return f'{hours}h{rest//60:02d}m{rest % 60:02d}s'


class Probe:
    """Live cost of one run with a Hall block, for progress lines and the substep cap.

    Wraps hall.electric_field (called once per RK4 stage, so four times per Hall substep) and
    hall.inertia_solve (CG iterations), which lets a progress line appear even while a single ideal
    step is still inside its Hall substeps. cap is the largest number of Hall substeps one ideal
    step (both Hall half-steps) may take before TooExpensive is raised.
    """

    def __init__(self, label, hall, end=None, every=60., cap=None, total_steps=None, note=''):
        self.label, self.end, self.every, self.cap, self.total_steps = label, end, every, cap, total_steps
        self.t = self.dt = 0.
        self.steps = self.evals = self.cg = self.step_evals = 0
        self.max_step_substeps = 0.
        self.start = self.last = time.perf_counter()
        field, solve = hall.electric_field, hall.inertia_solve

        def counted_field(B, rho):
            self.evals += 1
            if self.cap is not None and (self.evals-self.step_evals)/4 > self.cap:
                raise TooExpensive(self.label)
            self.tick()
            return field(B, rho)

        def counted_solve(*a, **kw):
            result = solve(*a, **kw)
            self.cg += result[1]
            return result

        hall.electric_field, hall.inertia_solve = counted_field, counted_solve
        print(f'START {label}: end={end if end is not None else total_steps} cap={cap} {note}'.rstrip(), flush=True)

    def begin_step(self, dt):
        self.step_evals, self.dt = self.evals, dt

    def end_step(self, dt, advance=True):
        """Close an ideal step; the warm-up step passes advance=False so t and steps exclude it."""
        self.max_step_substeps = max(self.max_step_substeps, (self.evals-self.step_evals)/4)
        if advance:
            self.t += dt
            self.steps += 1
        self.tick()

    def tick(self):
        now = time.perf_counter()
        if now-self.last >= self.every:
            self.last = now
            self.report()

    def report(self, tag='PROGRESS'):
        wall = time.perf_counter()-self.start
        done = self.steps/self.total_steps if self.total_steps else self.t/self.end
        eta = wall*(1-done)/done if done > 0 else math.nan
        of = f'/{self.end:g}' if self.end is not None else f'/{self.total_steps}'
        now = f't={self.t:.5g}' if self.end is not None else f'step={self.steps}'
        print(f'{tag} {self.label}: {now}{of} steps={self.steps} substeps={self.evals//4} cg_iterations={self.cg} '
              f'step_substeps={(self.evals-self.step_evals)//4} wall={hms(wall)} eta={hms(eta)}', flush=True)

    def projection(self):
        """Projected total cost from the substeps per ideal step seen so far (a lower bound as B grows)."""
        wall = time.perf_counter()-self.start
        per_step = max((self.evals-self.step_evals)/4, self.max_step_substeps)
        if self.total_steps:
            steps_total = float(self.total_steps)
        elif self.steps and self.t > 0:
            steps_total = self.end*self.steps/self.t
        else:
            steps_total = self.end/self.dt if self.dt else math.nan
        substeps = self.evals/4
        total = per_step*steps_total
        return dict(per_step_substeps=per_step, steps_total=steps_total, substeps_so_far=substeps,
                    projected_substeps=total, wall_so_far_s=wall,
                    projected_hours=wall/max(substeps, 1.)*total/3600.)


def too_expensive_line(what, cap, p):
    return (f'MEASURE {what} TOO_EXPENSIVE (stopped, not a failure): cap={cap} '
            f'per_step_substeps>={p["per_step_substeps"]:.6g} projected_steps={p["steps_total"]:.6g} '
            f'projected_substeps={p["projected_substeps"]:.6g} projected_wall_hours={p["projected_hours"]:.6g} '
            f'substeps_so_far={p["substeps_so_far"]:.6g} wall_so_far={hms(p["wall_so_far_s"])}')


def save_json(path, record):
    """Atomic write, so a preempted job never leaves a partial cache file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(record, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))
    os.replace(tmp, path)


def ot_evolve(g, s, U, marks, probe):
    """Ideal-CFL loop to each mark. Seconds include compute_dt and the CUDA sync, and exclude the
    warm-up step and the host snapshots. Returns per-mark (host state, max|div B|), steps, seconds."""
    dt = s.compute_dt(U)
    probe.begin_step(dt)
    s.step(U, dt)
    probe.end_step(dt, advance=False)
    if s.hall is not None:
        s.hall.reset_stats()
    t, steps, seconds, snaps = 0., 0, 0., []
    for mark in marks:
        sync(g)
        start = time.perf_counter()
        while t < mark-1e-12:
            dt = min(s.compute_dt(U), mark-t)
            probe.begin_step(dt)
            U = s.step(U, dt)
            probe.end_step(dt)
            t += dt
            steps += 1
        sync(g)
        seconds += time.perf_counter()-start
        snaps.append((host(U).copy(), s.max_div_b(U)))
    return snaps, U, steps, seconds


OT_VARIANTS = ('ref_N', 'ref_2N', 'inertia_const', 'inertia_local4', 'inertia_fixed')


def ot_hall(args):
    """Hall block of the ot suite and the Alfven speed of the mean state."""
    b0, rho0 = 1/np.sqrt(4*np.pi), 25/(36*np.pi)  # main's orszag_tang_2d amplitude and density
    v_a = b0/np.sqrt(rho0)
    hall = dict(eta_h=args.ot_di*v_a, eta_h_form=args.ot_form, b0=b0, rho0=rho0, cg_tol=args.cg_tol)
    if args.ot_eta_max is not None:
        hall['eta_max'] = args.ot_eta_max
    return hall, v_a


def ot_table(args):
    """variant -> (label, grid points, Hall overrides); labels are the check names of the original suite."""
    n = args.n
    k_ref = 2*np.pi*4  # energy-containing band of the box L = 1
    return {'ref_N': ('reference', n, dict(method='reference')),
            'ref_2N': ('reference 2N', 2*n, dict(method='reference')),
            'inertia_const': ('inertia constant automatic l', n, dict(method='inertia', length_mode='constant')),
            'inertia_local4': ('inertia local automatic l ratio=4', n,
                               dict(method='inertia', length_mode='local', length_ratio=4.)),
            'inertia_fixed': ('inertia fixed l k_ref*l=0.2', n,
                              dict(method='inertia', length_mode='constant', length=0.2/k_ref))}


def ot_tag(mark):
    return f'{round(mark*100):03d}'


def ot_base(args, key, size):
    """Cache path stem (no extension). Keyed by variant, N and the parameters that change the numbers;
    not by revision (the revision is stored inside) and not by --ot-tend (the marks are stored inside)."""
    form = '' if args.ot_form == 'proportional_to_B_over_rho' else f'_{args.ot_form}'
    cap = '' if args.ot_eta_max is None else f'_etamax{args.ot_eta_max:g}'
    return str(Path(args.ot_cache)/f'ot_{key}_N{size}_di{args.ot_di:g}{form}{cap}_cg{args.cg_tol:g}')


def ot_save(args, record):
    """<base>.npz holds B and rho per mark; <base>.json holds the statistics and is written last."""
    if not args.ot_cache:
        return
    base = ot_base(args, record['key'], record['size'])
    meta = {k: v for k, v in record.items() if k != 'results'}
    meta.update(revision=REVISION, ot_di=args.ot_di, cg_tol=args.cg_tol, ot_eta_max=args.ot_eta_max)
    if record['status'] == 'ok':
        Path(base).parent.mkdir(parents=True, exist_ok=True)
        arrays, meta['scalars'] = {}, []
        for tag, r in zip(record['tags'], record['results']):
            arrays[f'B_{tag}'], arrays[f'rho_{tag}'] = r['B'], r['rho']
            meta['scalars'].append({k: v for k, v in r.items() if k not in ('B', 'rho', 'spectrum')})
        np.savez(base+'.tmp.npz', **arrays)
        os.replace(base+'.tmp.npz', base+'.npz')
    save_json(base+'.json', meta)
    print(f'CACHE ot {record["key"]} N={record["size"]} status={record["status"]}: {base}.json', flush=True)


def ot_load(args, key, size):
    """Cached record of a variant, or None if absent."""
    if not args.ot_cache:
        return None
    base = ot_base(args, key, size)
    if not Path(base+'.json').exists():
        return None
    record = json.loads(Path(base+'.json').read_text())
    if record['status'] == 'ok':
        if not Path(base+'.npz').exists():
            return None
        with np.load(base+'.npz') as z:
            record['results'] = [dict(scalars, B=z[f'B_{tag}'], rho=z[f'rho_{tag}'],
                                      spectrum=magnetic_spectrum(z[f'B_{tag}']))
                                 for tag, scalars in zip(record['tags'], record['scalars'])]
    return record


def ot_run(args, out, key, label, size, cfg, hall, marks):
    """One variant: run, check, return its record (status ok, too_expensive or error)."""
    g, s, U0 = ot_setup(args, size, dict(hall, **cfg))
    initial = host(U0).copy()
    reference = cfg['method'] == 'reference'
    cap = args.max_hall_substeps_per_step if reference else None
    probe = Probe(f'ot {key} N={size}', s.hall, end=marks[-1], every=args.progress_every, cap=cap)
    record = dict(key=key, label=label, size=size, marks=list(marks), tags=[ot_tag(m) for m in marks])
    try:
        snaps, U, steps, seconds = ot_evolve(g, s, U0, marks, probe)
    except RuntimeError as error:
        out.check(f'ot {label} N={size}', False, error=str(error)[:120])
        return dict(record, status='error', error=str(error)[:120])
    except TooExpensive:
        projection = probe.projection()
        print(too_expensive_line(f'ot {label} N={size}', cap, projection), flush=True)
        return dict(record, status='too_expensive', cap=cap, projection=projection)
    probe.report('DONE')
    total = s.hall.total
    gamma = s.equations.gamma
    l_eff = float('nan')
    if cfg['method'] == 'inertia':
        B_end = snaps[-1][0][5:8]
        l2 = s.hall.length_squared(U, 0.5*s.compute_dt(U))
        J2 = current_magnitude(g, B_end)**2
        l_eff = (math.sqrt(l2) if isinstance(l2, float)
                 else math.sqrt(float(np.sum(host(l2)*J2)/np.sum(J2))))
    info = dict(N=size, seconds=seconds, steps=steps, substeps=total['substeps'],
                cg_mean=total['cg_iterations']/max(total['cg_calls'], 1), cg_max=total['cg_max_iterations'],
                cg_worst_residual=total['cg_residual'],
                l_min=math.sqrt(total['length2_min']) if math.isfinite(total['length2_min']) else 0.,
                l_max=math.sqrt(total['length2_max']), l_current_weighted=l_eff)
    results = []
    for state_h, divb in snaps:
        rho, m, E, B = state_h[0], state_h[1:4], state_h[4], state_h[5:8]
        p = (gamma-1)*(E-0.5*(m*m).sum(axis=0)/rho-0.5*(B*B).sum(axis=0))
        results.append(dict(
            B=B, rho=rho, J=float(current_magnitude(g, B).max()), spectrum=magnetic_spectrum(B),
            divb=divb, rho_min=float(rho.min()), p_min=float(p.min()),
            dE_total=float(E.mean()/initial[4].mean()-1),
            dE_mag=float(0.5*(B*B).sum(axis=0).mean()/(0.5*(initial[5:8]**2).sum(axis=0).mean())-1)))
    finite = all(np.isfinite(r['B']).all() for r in results)
    last = record['tags'][-1]
    out.check(f'ot {label} N={size}', finite and all(r['rho_min'] > 0 and r['p_min'] > 0 for r in results),
              max_divB=max(r['divb'] for r in results), min_rho=results[-1]['rho_min'],
              min_p=results[-1]['p_min'], **{f'dE_total_{last}': results[-1]['dE_total'],
                                              f'dE_mag_{last}': results[-1]['dE_mag']}, **info)
    return dict(record, status='ok', info=info, results=results)


def ot_compare(args, out, records, table, hall):
    """Gate and speedups from the records in memory, else from --ot-cache."""
    n = args.n
    recs = {}
    for key, (label, size, cfg) in table.items():
        recs[key] = records.get(key) or ot_load(args, key, size)
    revisions = sorted({str(r.get('revision', REVISION)) for r in recs.values() if r})
    print(f'NOTE ot compare: revisions of the records: {revisions}', flush=True)
    missing = [k for k in ('ref_N', 'ref_2N') if recs[k] is None or recs[k]['status'] not in ('ok', 'too_expensive')]
    expensive = [k for k in ('ref_N', 'ref_2N') if k not in missing and recs[k]['status'] == 'too_expensive']
    if missing:
        out.check('ot reference runs complete', False, unavailable=','.join(missing))
        return
    if expensive:
        print(f'SKIP ot accuracy gates: reference {",".join(expensive)} was too expensive (see its MEASURE line)',
              flush=True)
        return
    ref, fine = ({r['tags'][i]: r['results'][i] for i in range(len(r['tags']))} for r in (recs['ref_N'], recs['ref_2N']))
    tags = [t for t in recs['ref_N']['tags'] if t in fine]
    if '025' not in tags:
        out.check('ot compare has t=0.25 snapshots of both references', False)
        return
    kmax = n//6
    g = ot_setup(args, n, dict(hall, method='reference'))[0]

    # Resolution-error yardstick: reference at 2N truncated to N against reference at N.
    yardstick = {}
    for tag in tags:
        B = truncate(fine[tag]['B'], n)
        yardstick[tag] = relative_l2(B, ref[tag]['B'])
        es, er = magnetic_spectrum(B), ref[tag]['spectrum']
        out.check(f'ot resolution error reference 2N->N vs N t={tag}', True, dB=yardstick[tag],
                  drho=relative_l2(truncate(fine[tag]['rho'], n), ref[tag]['rho']),
                  Jmax_ratio_2N_over_N=float(current_magnitude(g, B).max())/ref[tag]['J'],
                  spectrum_max=float(np.max(np.abs(es[1:kmax+1]-er[1:kmax+1])/er[1:kmax+1])))
    seconds_ref = recs['ref_N']['info']['seconds']
    for key in ('inertia_const', 'inertia_local4', 'inertia_fixed'):
        label, size, cfg = table[key]
        rec = recs[key]
        if rec is None or rec['status'] != 'ok' or '025' not in rec['tags']:
            print(f'SKIP ot accuracy gate {label}: no complete record', flush=True)
            continue
        info = rec['info']
        by_tag = dict(zip(rec['tags'], rec['results']))
        numbers = {}
        ratios = {}
        for tag in (t for t in tags if t in by_tag):
            B = by_tag[tag]['B']
            numbers[f'dB_{tag}'] = relative_l2(B, ref[tag]['B'])
            numbers[f'drho_{tag}'] = relative_l2(by_tag[tag]['rho'], ref[tag]['rho'])
            numbers[f'Jmax_ratio_{tag}'] = by_tag[tag]['J']/ref[tag]['J']
            es, er = by_tag[tag]['spectrum'], ref[tag]['spectrum']
            numbers[f'spectrum_max_{tag}'] = float(np.max(np.abs(es[1:kmax+1]-er[1:kmax+1])/er[1:kmax+1]))
            ratios[tag] = numbers[f'dB_{tag}']/yardstick[tag]
        out.check(f'ot accuracy gate {label}: regularization error <= resolution error at t=0.25',
                  ratios['025'] <= 1, ratio_025=ratios['025'],
                  **{f'ratio_{t}_informational': r for t, r in ratios.items() if t != '025'},
                  resolution_error_025=yardstick['025'], **numbers)
        print(f'MEASURE ot speedup {label}: A_over_B={seconds_ref/info["seconds"]:.6g} '
              f'(reference-N seconds / run seconds) accuracy_ratio_025={ratios["025"]:.6g} '
              f'substeps_reference={recs["ref_N"]["info"]["substeps"]} substeps_run={info["substeps"]}',
              flush=True)


def ot(args, out):
    """Hall Orszag-Tang: reference at N and 2N against inertia runs, with an accuracy gate.

    Each selected variant is run and cached (--ot-cache); the comparison reads the records of this
    process and, for the others, the cache.
    """
    hall, v_a = ot_hall(args)
    table = ot_table(args)
    marks = (0.25,) if args.ot_tend <= 0.25+1e-12 else (0.25, args.ot_tend)
    print(f'NOTE ot: eta_h={hall["eta_h"]:.6g} gives d_i = eta_H/v_A = {args.ot_di:g} at the mean state '
          f'(v_A={v_a:.4g}); fixed l={table["inertia_fixed"][2]["length"]:.4g}; times {marks}', flush=True)
    if args.ot_run and not args.ot_cache and len(args.ot_run) < len(OT_VARIANTS):
        print('NOTE ot: no --ot-cache, so these results are not saved for a later compare', flush=True)
    records = {}
    for key in args.ot_run:
        label, size, cfg = table[key]
        records[key] = ot_run(args, out, key, label, size, cfg, hall, marks)
        ot_save(args, records[key])
    if args.ot_compare:
        ot_compare(args, out, records, table, hall)


def stability_fields(n, seed):
    """Seeded random fields (box 2 pi) with amplitude spectrum peaked at k = 4: a solenoidal
    (b_x, b_y) from a potential plus a scalar b_z, the same for u, and a smooth scalar f, max|f| = 1."""
    rng = np.random.default_rng(seed)
    k = np.fft.fftfreq(n, 1./n)
    kx, ky = np.meshgrid(k, k, indexing='xy')
    kk = np.hypot(kx, ky)
    amp = np.exp(-0.5*((kk-4.)/1.5)**2)*(kk > 0)*(kk <= n/6)
    scalar = lambda: np.fft.ifft2((rng.standard_normal((n, n))+1j*rng.standard_normal((n, n)))*amp).real

    def vector():
        A = np.fft.fft2(scalar())
        return np.array([np.fft.ifft2(1j*ky*A).real, -np.fft.ifft2(1j*kx*A).real, scalar()])

    dB, du, f = vector(), vector(), scalar()
    rms = lambda a: np.sqrt(np.mean((a*a).sum(axis=0)))
    return dB/rms(dB), du/rms(du), f/np.max(np.abs(f))


def stability_run(g, cfg, B, u, rho, steps, cfl, label='stability', every=60., cap=None):
    """steps ideal-CFL solver steps from the given state; diagnostics after every step. With a cap
    (Hall substeps per ideal step) a run that exceeds it is stopped and returns too_expensive=projection."""
    s = build(g, cfl=cfl, hall=cfg)
    eq = s.equations
    probe = Probe(label, s.hall, every=every, cap=cap, total_steps=steps)
    U = state(g, eq, B, rho=rho, u=u, p=0.5*np.ones_like(rho))
    emag = lambda V: 0.5*(V[5:8]**2).sum(axis=0).mean()
    V = host(U)
    initial = dict(emag=emag(V), etot=V[4].mean())
    history = dict(emag=[initial['emag']], etot=[initial['etot']], bmax=[], rho_min=[], rho_max=[],
                   p_min=[], divb=[])
    result = dict(finite=True, error='', steps=0)
    try:
        for _ in range(steps):
            dt = s.compute_dt(U)
            probe.begin_step(dt)
            U = s.step(U, dt)
            probe.end_step(dt)
            V = host(U)
            result['steps'] += 1
            if not np.isfinite(V).all():
                result['finite'] = False
                break
            rho_, m, E, Bf = V[0], V[1:4], V[4], V[5:8]
            history['emag'].append(emag(V))
            history['etot'].append(E.mean())
            history['bmax'].append(np.sqrt((Bf*Bf).sum(axis=0)).max())
            history['rho_min'].append(rho_.min())
            history['rho_max'].append(rho_.max())
            history['p_min'].append(((eq.gamma-1)*(E-0.5*(m*m).sum(axis=0)/rho_-0.5*(Bf*Bf).sum(axis=0))).min())
            history['divb'].append(s.max_div_b(U))
    except RuntimeError as error:
        result['error'] = str(error)[:120]
    except TooExpensive:
        return dict(result, too_expensive=probe.projection(), cap=cap)
    probe.report('DONE')
    total = s.hall.total
    h = {key: np.array(value) for key, value in history.items()}
    result.update(
        emag_peak=float(h['emag'].max()), emag_over_initial_max=float(h['emag'].max()/initial['emag']),
        emag_over_initial_min=float(h['emag'].min()/initial['emag']),
        etot_ratio_min=float(h['etot'].min()/initial['etot']), etot_ratio_max=float(h['etot'].max()/initial['etot']),
        bmax=float(h['bmax'].max()) if len(h['bmax']) else float('nan'),
        rho_min=float(h['rho_min'].min()) if len(h['bmax']) else float('nan'),
        rho_max=float(h['rho_max'].max()) if len(h['bmax']) else float('nan'),
        p_min=float(h['p_min'].min()) if len(h['bmax']) else float('nan'),
        divB_max=float(h['divb'].max()) if len(h['bmax']) else float('nan'),
        substeps=total['substeps'], cg_mean=total['cg_iterations']/max(total['cg_calls'], 1),
        cg_max=total['cg_max_iterations'], cg_worst_residual=total['cg_residual'],
        l_min=math.sqrt(total['length2_min']) if math.isfinite(total['length2_min']) else 0.,
        l_max=math.sqrt(total['length2_max']), rho_floor=s.rho_min, p_floor=s.p_min)
    return result


STAB_FORMS = ('proportional_to_B_over_rho', 'constant')


def stability_cases():
    """All 8 reference then 16 inertia cases in the original order, with an index and label words.

    Words (selected by --stab-cases): ref|inertia, dB=, contrast=, form=prop|const and, for inertia,
    mode= and safety=. The half-fraction design sets safety 0.5 or 1.0 by the parity of the four level indices.
    """
    cases = []
    for amp in (0.1, 1.0):
        for contrast in (1., 100.):
            for form in STAB_FORMS:
                cases.append(dict(kind='ref', amp=amp, contrast=contrast, form=form, mode=None, safety=None))
    for amp in (0.1, 1.0):
        for contrast in (1., 100.):
            for form in STAB_FORMS:
                for mode in ('constant', 'local'):
                    bits = (amp > 0.5)+(contrast > 1.)+(form == 'constant')+(mode == 'local')
                    cases.append(dict(kind='inertia', amp=amp, contrast=contrast, form=form, mode=mode,
                                      safety=0.5 if bits % 2 == 0 else 1.0))
    for index, c in enumerate(cases):
        c['index'] = index
        c['words'] = [c['kind'], f'dB={c["amp"]:g}', f'contrast={c["contrast"]:g}',
                      'form='+('const' if c['form'] == 'constant' else 'prop')]
        c['text'] = (f'reference dB={c["amp"]:g} contrast={c["contrast"]:g} form={c["form"]}' if c['kind'] == 'ref'
                     else f'inertia dB={c["amp"]:g} contrast={c["contrast"]:g} form={c["form"]} '
                          f'mode={c["mode"]} safety={c["safety"]:g}')
        if c['kind'] == 'inertia':
            c['words'] += [f'mode={c["mode"]}', f'safety={c["safety"]:g}']
        c['label'] = ' '.join(c['words'])
    return cases


def stability_select(args, cases):
    """--stab-cases: comma list of an index, an a-b index range, or space-separated label words that
    must all be present (e.g. "ref dB=1 contrast=100"); --stab-ref-only / --stab-inertia-only filter the kind."""
    keep = set(c['index'] for c in cases)
    if args.stab_cases:
        keep = set()
        for token in (t.strip() for t in args.stab_cases.split(',') if t.strip()):
            m = re.fullmatch(r'(\d+)(?:-(\d+))?', token)
            if m:
                lo, hi = int(m[1]), int(m[2] or m[1])
                picked = [c['index'] for c in cases if lo <= c['index'] <= hi]
            else:
                picked = [c['index'] for c in cases if all(w in c['words'] for w in token.split())]
            if not picked:
                raise ValueError(f'--stab-cases entry {token!r} matches no case')
            keep.update(picked)
    selected = [c for c in cases if c['index'] in keep and not (args.stab_ref_only and c['kind'] != 'ref')
                and not (args.stab_inertia_only and c['kind'] != 'inertia')]
    if not selected:
        raise ValueError('the stability case filters select no case')
    return selected


def stability_path(args, case):
    return Path(args.stab_cache)/(f'stability_{"_".join(case["words"])}_N{args.n}_lh{args.stab_lh_dx:g}'
                                  f'_steps{args.stab_steps}_cfl{args.stab_cfl:g}_cg{args.cg_tol:g}.json')


def stability_load(args, case, records):
    """Record of a case from this process, else from --stab-cache, else None."""
    if case['index'] in records:
        return records[case['index']]
    if args.stab_cache and stability_path(args, case).exists():
        return json.loads(stability_path(args, case).read_text())
    return None


def stability_reference(args, case, records):
    """(state, peak magnetic energy) of the matched reference: state is ok, too_expensive or missing;
    the peak is None when the reference blew up or raised (the energy gate then passes, as before)."""
    ref = next(c for c in stability_cases() if c['kind'] == 'ref' and (c['amp'], c['contrast'], c['form'])
               == (case['amp'], case['contrast'], case['form']))
    record = stability_load(args, ref, records)
    if record is None:
        return 'missing', None
    if record['status'] == 'too_expensive':
        return 'too_expensive', None
    r = record['result']
    return 'ok', (r['emag_peak'] if r['finite'] and not r['error'] else None)


def stability_measure(case, record):
    if record['status'] == 'too_expensive':
        print(too_expensive_line(f'stability {case["text"]}', record['cap'], record['too_expensive']), flush=True)
    else:
        print(f'MEASURE stability {case["text"]}: ' + ' '.join(
            f'{k}={v:.6g}' if isinstance(v, float) else f'{k}={v}' for k, v in record['result'].items()), flush=True)


def stability_gate(args, out, case, record, records):
    r = record['result']
    state_, reference = stability_reference(args, case, records)
    basic = (r['finite'] and not r['error'] and r['divB_max'] < 1e-10
             and r['rho_min'] > 10*r['rho_floor'] and r['p_min'] > 10*r['p_floor']
             and r['cg_worst_residual'] <= max(args.cg_tol, 10*np.finfo(float).eps))
    energy_ok = reference is None or r['emag_peak'] <= 1.1*reference
    energy = 'skipped' if state_ != 'ok' else 'pass' if energy_ok else 'fail'
    out.check(f'stability {case["text"]}', basic and energy != 'fail',
              energy_gate=energy,
              emag_peak_over_reference=(r['emag_peak']/reference if reference else float('nan')), **r)
    if energy == 'skipped':
        print(f'SKIP stability energy gate {case["text"]}: matched reference is {state_} '
              f'(run the reference case, then --stab-report)', flush=True)


def stability(args, out):
    """Nonlinear robustness of the inertia method at large l_H/dx against the reference method.

    Each selected case is run and cached (--stab-cache); the inertia energy gate reads the matched
    reference from this process or the cache. --stab-report gates the cached results without running.
    """
    cases = stability_cases()
    selected = stability_select(args, cases)
    eta = args.stab_lh_dx*2*np.pi/args.n  # b0 = rho0 = 1 and v_A = 1 at the mean state, so l_H = eta_h
    print(f'NOTE stability: l_H/dx={args.stab_lh_dx:g} (eta_h={eta:.4g}), N={args.n}, {args.stab_steps} ideal-CFL '
          f'steps at cfl={args.stab_cfl:g}; B0=1, rms dB/B0 in (0.1, 1), u_rms = 0.5 dB, p=0.5', flush=True)
    print('NOTE stability cases: ' + '; '.join(f'{c["index"]}={c["label"]}' for c in selected), flush=True)
    records = {}
    if not args.stab_report:
        g = make_grid(args)
        dB, du, f = stability_fields(args.n, seed=20240611)
        base = lambda form: dict(eta_h=eta, eta_h_form=form, cg_tol=args.cg_tol)
        fields = lambda amp, contrast: (np.array([1., 0., 0.])[:, None, None]+amp*dB, 0.5*amp*du,
                                        np.exp(0.5*math.log(contrast)*f))
        for case in selected:
            reference = case['kind'] == 'ref'
            cfg = dict(base(case['form']), method='reference' if reference else 'inertia')
            if not reference:
                cfg.update(length_mode=case['mode'], safety=case['safety'])
                if case['mode'] == 'local':
                    cfg['length_ratio'] = 4.
            B, u, rho = fields(case['amp'], case['contrast'])
            r = stability_run(g, cfg, B, u, rho, args.stab_steps, args.stab_cfl, label=f'stability {case["label"]}',
                              every=args.progress_every, cap=args.max_hall_substeps_per_step if reference else None)
            record = dict(index=case['index'], label=case['label'], status='ok', revision=REVISION)
            if 'too_expensive' in r:
                record.update(status='too_expensive', cap=r['cap'], too_expensive=r['too_expensive'])
            else:
                record['result'] = r
            records[case['index']] = record
            if args.stab_cache:
                save_json(stability_path(args, case), record)
                print(f'CACHE stability {case["label"]} status={record["status"]}: {stability_path(args, case)}',
                      flush=True)
            if reference:
                stability_measure(case, record)
            else:
                stability_gate(args, out, case, record, records)
        return
    for case in selected:
        record = stability_load(args, case, records)
        if record is None:
            print(f'SKIP stability {case["text"]}: no cached result', flush=True)
            continue
        records[case['index']] = record
        if case['kind'] == 'ref':
            stability_measure(case, record)
        else:
            stability_gate(args, out, case, record, records)
    print(f'NOTE stability report: revisions of the records: '
          f'{sorted({str(r.get("revision")) for r in records.values()})}', flush=True)


def induction_phase_error(g, eq, X, method, length, amp, eta=3.):
    """Circular Hall induction wave on grid g: measured phase rate / exact - 1, and the final state."""
    hall = Hall(g, dict(method=method, eta_h=eta, length=length))
    omega = eta/math.sqrt(1+amp**2)/(1+length**2)
    U = state(g, eq, np.array([1+0*X, amp*np.cos(X), amp*np.sin(X)]))
    ts, phases = [], []

    def sample(t, V):
        b = host(V[5:8]).astype(np.float64)
        ts.append(t)
        phases.append(np.angle(np.mean((b[1]+1j*b[2])*np.exp(-1j*X))))

    V, _ = advance(g, lambda W, h: hall.step(W, eq, h), U, 1/omega, 0.025/omega, sample=sample)
    return np.polyfit(ts, np.unwrap(phases), 1)[0]/omega-1, V


def precision(args, out):
    """float32 against float64 on torch CUDA: induction waves, equilibria, energy and Hall-MRI growth."""
    g64, g32 = make_grid(args), make_grid(args, precision='single')
    eq = MHDEquations2D()
    X, Y = coords(g64)  # float64 coordinates; state() casts to the grid dtype
    is32 = lambda a: '32' in str(a.dtype)
    eps32 = float(np.finfo(np.float32).eps)
    out.check('precision grids', is32(g32.x) and not is32(g64.x), grid_dtype=str(g32.x.dtype),
              cg_tol_effective=max(args.cg_tol, 100*eps32))
    ic = orszag_tang_2d(*g32.xy_mesh())
    out.check('precision main orszag_tang_2d initial state is float32 on a single grid', is32(ic),
              dtype=str(ic.dtype))
    B = np.array([1+0.04*np.sin(Y), 0.1*np.cos(X), 0.1*np.sin(X)+0.03*np.cos(2*Y+X)])
    rho = 1+0.2*np.cos(X+Y)
    U32 = state(g32, eq, B, rho=rho)
    out.check('precision state() on a single grid is float32', is32(U32), dtype=str(U32.dtype))
    # Circular Hall induction wave: amplitude 1e-2 so that the perturbation exceeds float32 roundoff.
    for method, length in (('reference', 0.), ('inertia', 0.3)):
        try:
            e32, V = induction_phase_error(g32, eq, X, method, length, 1e-2)
            e64, _ = induction_phase_error(g64, eq, X, method, length, 1e-2)
            out.check(f'precision induction {method}', abs(e32) < 1e-3 and is32(V),
                      phase_rate_error_float32=e32, phase_rate_error_float64=e64, dtype=str(V.dtype))
        except RuntimeError as error:
            out.check(f'precision induction {method}', False, error=str(error)[:120])
    # Equilibria stay put: force-free field and non-force-free uniform-current-free field.
    for force_free in (True, False):
        Beq = np.array([0*X, np.cos(X), np.sin(X)] if force_free else [0*X, 1+0.1*np.cos(X), 0*X])
        for method in ('reference', 'inertia'):
            drifts = []
            for g in (g32, g64):
                hall = Hall(g, dict(method=method, eta_h=-1., length=0.3, length_mode='local', cg_tol=args.cg_tol))
                U = state(g, eq, Beq, rho=rho)
                try:
                    V, _ = advance(g, lambda W, h: hall.step(W, eq, h), U, 0.1, 0.01)
                    drifts.append(float(np.max(np.abs(host(V-U)))))
                except RuntimeError as error:
                    drifts.append(float('nan'))
                    print(f'  precision equilibrium error: {str(error)[:120]}', flush=True)
            out.check(f'precision equilibrium {method} force_free={force_free}', drifts[0] < 1e-5,
                      drift_float32=drifts[0], drift_float64=drifts[1])
    # Modified-energy conservation of one nonlinear Hall subflow, and the unchanged pressure.
    for method, local in (('reference', False), ('inertia', False), ('inertia', True)):
        numbers = []
        for g in (g32, g64):
            hall = Hall(g, dict(method=method, eta_h=0.3, eta_h_form='proportional_to_B_over_rho', length=0.3,
                                length_mode='local' if local else 'constant', cg_tol=args.cg_tol))
            U = state(g, eq, B, rho=rho)
            l2 = hall.length_squared(U, 0.01) if method == 'inertia' else 0.
            initial = hall.modified_energy(U[5:8], l2)
            try:
                V, stats = advance(g, lambda W, h: hall.step(W, eq, h), U, 0.2, 0.01, hall=hall)
                numbers.append((abs(hall.modified_energy(V[5:8], l2)/initial-1),
                                float(np.max(np.abs(host(eq.primitive(V)[4]-eq.primitive(U)[4])))), stats, V))
            except RuntimeError as error:
                numbers.append((float('nan'), float('nan'), dict(error=str(error)[:120]), None))
        (drift32, dp32, stats, V), (drift64, dp64, _, _) = numbers
        out.check(f'precision energy {method} local={local}', drift32 < 1e-5 and dp32 < 1e-5 and V is not None
                  and is32(V), relative_drift_float32=drift32, relative_drift_float64=drift64,
                  pressure_drift_float32=dp32, pressure_drift_float64=dp64, **stats)
    # Coupled ideal + Hall solver steps: dtype is preserved and the float32 trajectory tracks float64.
    cfg = dict(method='inertia', eta_h=0.3, eta_h_form='proportional_to_B_over_rho', length=0.3,
               length_mode='local', cg_tol=args.cg_tol)
    states = []
    for g in (g32, g64):
        s = build(g, cfl=0.25, hall=cfg)
        U = state(g, s.equations, B, rho=rho)
        dt = 0.5*s.compute_dt(state(g64, s.equations, B, rho=rho))
        try:
            for _ in range(10):
                U = s.step(U, dt)
            states.append((host(U).astype(np.float64), is32(U), s.max_div_b(U), s.hall.total['cg_residual']))
        except RuntimeError as error:
            states.append((None, False, float('nan'), float('nan')))
            print(f'  precision coupled error: {str(error)[:120]}', flush=True)
    (V32, kept, divb32, res32), (V64, _, divb64, res64) = states
    difference = float(np.max(np.abs(V32-V64))) if V32 is not None and V64 is not None else float('nan')
    out.check('precision coupled step float32 vs float64', kept and difference < 1e-4,
              max_difference=difference, divB_float32=divb32, divB_float64=divb64,
              cg_worst_residual_float32=res32, cg_worst_residual_float64=res64)
    precision_mri(args, out)


def precision_mri(args, out):
    """Short Hall-MRI eigenmode growth (1D torch CUDA), float32 against float64."""
    field, rho0, omega, shear_rate = 1.0, 1.0, 1.0, 1.5
    eta = field**2  # X = 1, sign(eta Omega.B) = +1
    k, s, v = hall_mri_fastest(omega, shear_rate, field, rho0, eta)
    rate, params = max(abs(omega), abs(shear_rate)), (omega, shear_rate, field, rho0)
    kw = dict(N=args.mri_n, Lx=2*np.pi/k, dealias=True, backend='torch', torch_device=args.device,
              fft_workers=1)
    try:
        grids = {name: Grid1D(precision=name, **kw) for name in ('double', 'single')}
        eq = MHDEquations1D()
        sheet = dict(omega=omega, shear=shear_rate, vertical_axis=0, radial_axis=1)
        dt = build(grids['double'], cfl=args.mri_cfl, shearing_sheet=sheet).compute_dt(
            mri_state(grids['double'], eq, k, v, 0., field, rho0, args.mri_beta))
        for variant, cfg, length in (('reference', dict(method='reference', eta_h=eta), 0.),
                                     ('k*l=0.2', dict(method='inertia', eta_h=eta, length=0.2/k), 0.2/k)):
            s_reg, v_reg = hall_mri_mode(k, omega, shear_rate, field, rho0, eta, length)
            end = mri_duration(s_reg, rate, 3., dt)  # three e-folds: the 1e-3 seed stays linear
            growth = {}
            for name, g in grids.items():
                ts, cs, stats = mri_run(args, g, eq, k, params, cfg, v_reg, 1e-3, end, dt)
                window = ts >= 0.4*end
                growth[name] = np.polyfit(ts[window], np.log(np.hypot(np.abs(cs[window, 2]),
                                                                       np.abs(cs[window, 3]))), 1)[0]
            ok = abs(growth['single']/growth['double']-1) < 1e-3
            out.check(f'precision mri {variant}', ok, growth_float32=growth['single'],
                      growth_float64=growth['double'], linear=s_reg.real,
                      float32_over_float64=growth['single']/growth['double'],
                      float32_error_vs_linear=growth['single']/s_reg.real-1, **stats)
    except Exception as error:  # report any torch 1D failure as a check instead of aborting the suite
        out.check('precision mri', False, error=f'{type(error).__name__}: {str(error)[:160]}')


H2H_TESTS = ('waves', 'mri', 'shear')
H2H_METHODS = ('reference', 'hyper', 'inertia-auto', 'inertia-fixed', 'xmhd-auto', 'xmhd-fixed')
H2H_THRESHOLDS = (0.01, 0.001)


def float_list(text):
    return [float(v) for v in text.split(',') if v]


def h2h_selected(tokens, method):
    return any(t == 'all' or method == t or method.startswith(t+'-') for t in tokens)


def h2h_points(args, k):
    """(method, knob label, Hall block) for every accuracy-knob value; a fixed k*l becomes l = k*l/k."""
    xl = args.h2h_xmhd_kl
    points = [('reference', f'safety={s:g}', dict(method='reference', safety=s)) for s in args.h2h_ref_safety]
    points += [('hyper', f'C={c:g}', dict(method='reference', hyper=c, safety=0.5)) for c in args.h2h_hyper]
    points += [('inertia-auto', f'safety={s:g}', dict(method='inertia', safety=s)) for s in args.h2h_safety]
    points += [('inertia-fixed', f'kl={q:g}', dict(method='inertia', length=q/k, safety=0.5)) for q in args.h2h_kl]
    points += [('xmhd-auto', f'c={c:g}', dict(method='xmhd', c_factor=c, safety=0.5)) for c in args.h2h_cfactor]
    points += [('xmhd-fixed', f'c={c:g},kl={xl:g}', dict(method='xmhd', c_factor=c, length=xl/k, safety=0.5))
               for c in args.h2h_cfactor]
    return [p for p in points if h2h_selected(args.h2h_method_tokens, p[0])]


def h2h_ticker(label, end, every):
    """Sample callback that prints a PROGRESS line every `every` wall seconds."""
    start = last = time.perf_counter()

    def tick(t, V):
        nonlocal last
        now = time.perf_counter()
        if now-last >= every:
            last = now
            done = t/end
            print(f'PROGRESS {label}: t={t:.5g}/{end:g} wall={hms(now-start)} '
                  f'eta={hms((now-start)*(1-done)/done if done > 0 else math.nan)}', flush=True)
    return tick


def h2h_waves_cases(args):
    """Linear whistler and ion-cyclotron eigenmodes at l_H/dx in --h2h-lh-dx (1D NumPy, coupled ideal+Hall).

    Error: relative L2 error of B - B0 against the exact true-Hall linear solution at T = --h2h-periods
    whistler periods, at the ideal CFL step.
    """
    wargs = argparse.Namespace(**dict(vars(args), dim=1, backend='numpy'))
    g = make_grid(wargs)
    amp, cases = 1e-6, []
    for ratio in args.h2h_lh_dx:
        eta = ratio*g.dx  # |B0| = rho0 = 1, so v_A = 1 and l_H = eta
        T = args.h2h_periods*2*np.pi/abs(WaveMode(wargs, g, 'whistler', eta).omega)
        for branch in ('whistler', 'ion-cyclotron'):
            mode = WaveMode(wargs, g, branch, eta)
            exact = mode.fields(amp, T)[1]
            scale = np.linalg.norm(exact-mode.background)
            ideal = build(g, cfl=args.h2h_cfl)
            dt = ideal.compute_dt(mode.initial(g, ideal.equations, amp))

            def run(cfg, tick, mode=mode, exact=exact, scale=scale, dt=dt, T=T, eta=eta):
                s = build(g, cfl=args.h2h_cfl, hall=dict(cfg, eta_h=eta))
                V, info = advance(g, s.step, mode.initial(g, s.equations, amp), T, dt, hall=s.hall, sample=tick)
                return float(np.linalg.norm(host(V[5:8])-exact)/scale), info

            cases.append(dict(label=f'{branch} l_H/dx={ratio:g}', k=1., run=run, end=T,
                              notes=f'N={g.N} eta={eta:.4g} T={T:.4g} dt={dt:.4g} omega={mode.omega:.4g} '
                                    f'periods={args.h2h_periods:g}'))
    return cases


def h2h_mri_cases(args):
    """Hall-dominated MRI eigenmodes (1D shearing sheet, one fastest-growing wavelength on N points).

    N is chosen so that l_H/dx = |eta_H|/(v_A dx) >= --h2h-mri-lh-dx. Error: |growth - s|/s for the
    fitted growth rate of |b| against the true-Hall eigenvalue s. Cases with no growing mode are skipped.
    """
    field, rho0 = 1.0, 1.0
    margs = argparse.Namespace(**dict(vars(args), mri_beta=args.h2h_mri_beta))
    cases = []
    for label, omega, shear_rate, eta in [('Keplerian X=1 sign=-1', 1.0, 1.5, -field**2/1.0),
                                          ('Hall-shear X=0.5 sign=-1', 0.0, 1.0, -field**2/0.5),
                                          ('Hall-shear X=0.5 sign=+1', 0.0, 1.0, field**2/0.5)]:
        k, s, v = hall_mri_fastest(omega, shear_rate, field, rho0, eta)
        rate = max(abs(omega), abs(shear_rate))
        if s.real < 1e-3*rate:
            print(f'MEASURE h2h mri {label}: stable in the scanned range (max growth {s.real:.3g}), skipped',
                  flush=True)
            continue
        n = max(args.mri_n, 8*math.ceil(args.h2h_mri_lh_dx*2*np.pi/(k*abs(eta))/8))
        g = Grid1D(N=n, Lx=2*np.pi/k, dealias=True, backend='numpy', precision='double', fft_workers=1)
        eq = MHDEquations1D()
        sheet = dict(omega=omega, shear=shear_rate, vertical_axis=0, radial_axis=1)
        dt = build(g, cfl=args.mri_cfl, shearing_sheet=sheet).compute_dt(
            mri_state(g, eq, k, v, 0., field, rho0, margs.mri_beta))
        end = mri_duration(s, rate, args.h2h_efolds, dt)
        stiffness = abs(eta)*Hall(g, dict(eta_h=eta)).kmax2(np.zeros((3, n)))*0.5*dt

        def run(cfg, tick, g=g, eq=eq, k=k, s=s, v=v, end=end, dt=dt, eta=eta, omega=omega, shear_rate=shear_rate):
            ts, cs, stats = mri_run(margs, g, eq, k, (omega, shear_rate, field, rho0), dict(cfg, eta_h=eta), v,
                                    args.mri_amp, end, dt, tick=tick)
            window = ts >= 0.4*end
            logs = np.log(np.hypot(np.abs(cs[:, 2]), np.abs(cs[:, 3])))
            growth = np.polyfit(ts[window], logs[window], 1)[0]
            return abs(growth-s.real)/s.real, dict(stats, growth=growth, linear=s.real)

        cases.append(dict(label=label, k=k, run=run, end=end,
                          notes=f'N={n} k={k:.4g} eta={eta:.4g} l_H/dx={abs(eta)/g.dx:.4g} growth={s.real:.4g} '
                                f'dt={dt:.4g} end={end:.4g} beta={margs.mri_beta:g} '
                                f'hall_rate_x_half_dt={stiffness:.4g}'))
    return cases


def h2h_shear_cases(args):
    """Density-shear instability under Hall steps of --shear-dt (2D): growth against the floored eigenvalue."""
    L = 2*np.pi*args.shear_box_periods
    g = make_grid(args, L=L)
    eq = MHDEquations2D()
    X, Y = coords(g)
    Y = Y-L/2
    f = np.exp(-Y**2)+args.shear_floor
    target, by, dby, _ = shear_mode(args.n, L, args.shear_floor)
    if args.shear_seed == 'eigenmode':
        B = np.array([f-1e-7*dby[:, None]*np.sin(X), 1e-7*by[:, None]*np.cos(X),
                      -1e-7*target*by[:, None]*np.cos(X)])
    else:
        B = np.array([f, 1e-7*np.cos(X), 0*X])

    def run(cfg, tick):
        hall = Hall(g, dict(cfg, eta_h=1., eta_h_form='proportional_to_B_over_rho', cg_tol=args.cg_tol))
        U = state(g, eq, B, rho=f, p=0.55*(1+args.shear_floor)**2-0.5*f**2)
        step = lambda W, h: hall.step(W, eq, h)
        step(U, args.shear_dt)  # warm kernels without changing the initial state
        hall.reset_fields()
        ts, logs = [], []

        def sample(t, V):
            if args.shear_fit_start-1e-12 <= t <= args.shear_fit_end+1e-12:
                ts.append(t)
                logs.append(np.log(np.sqrt(np.mean(host(V[6])**2))))
            tick(t, V)

        V, info = advance(g, step, U, args.shear_end, args.shear_dt, hall=hall, sample=sample)
        growth = np.polyfit(ts, logs, 1)[0]
        return abs(growth/target-1), dict(info, growth=growth, linear=target)

    return [dict(label=f'density shear N={args.n}', k=1., run=run, end=args.shear_end,
                 notes=f'N={args.n} box={L:.4g} dx={g.dx:.4g} sigma={target:.6g} floor={args.shear_floor:g} '
                       f'dt={args.shear_dt:g} seed={args.shear_seed} backend={args.backend}')]


def h2h_pareto(points):
    """Per (test, case, method): the cheapest point reaching each error threshold, plus the fastest method."""
    cheapest, summary = {}, []
    for p in points:
        cheapest.setdefault((p['test'], p['case']), {}).setdefault(p['method'], []).append(p)
    cell = lambda best, ref: ('not reached' if best is None else f'{best["seconds"]:.4g}s [{best["knob"]}]'
                              + (f' x{ref["seconds"]/best["seconds"]:.3g}' if ref and ref is not best else ''))
    for (test, case), methods in cheapest.items():
        best = {}
        for method in H2H_METHODS:
            for thr in H2H_THRESHOLDS:
                good = [p for p in methods.get(method, []) if p['error'] is not None and p['error'] <= thr]
                best[method, thr] = min(good, key=lambda p: p['seconds']) if good else None
        for method in H2H_METHODS:
            if method not in methods:
                continue
            errors = [p['error'] for p in methods[method] if p['error'] is not None]
            cells = [cell(best[method, thr], best['reference', thr]) for thr in H2H_THRESHOLDS]
            print(f'MEASURE h2h pareto {test}/{case} {method}: min wall at error<=1%: {cells[0]}; '
                  f'at error<=0.1%: {cells[1]}; best_error={min(errors) if errors else math.nan:.4g}', flush=True)
            summary += [dict(test=test, case=case, method=method, threshold=thr,
                             seconds=None if best[method, thr] is None else best[method, thr]['seconds'],
                             knob=None if best[method, thr] is None else best[method, thr]['knob'])
                        for thr in H2H_THRESHOLDS]
        for thr in H2H_THRESHOLDS:
            reached = [(best[m, thr]['seconds'], m) for m in H2H_METHODS if best[m, thr] is not None]
            print(f'MEASURE h2h pareto {test}/{case} FASTEST at error<={thr:.1%}: '
                  f'{min(reached)[1] if reached else "none reached"}', flush=True)
    return summary


def h2h(args, out):
    """Head-to-head of Hall integration strategies: error vs exact against wall time, per test and case."""
    builders = dict(waves=h2h_waves_cases, mri=h2h_mri_cases, shear=h2h_shear_cases)
    records = {}
    for path in args.h2h_in:
        for p in json.loads(Path(path).read_text())['points']:
            records[p['test'], p['case'], p['method'], p['knob']] = p
    print(f'NOTE h2h: tests={",".join(args.h2h_tests)} methods={",".join(args.h2h_method_tokens)} '
          f'loaded_points={len(records)}', flush=True)

    def write():
        if args.h2h_out:
            save_json(args.h2h_out, dict(revision=REVISION, args={k: v for k, v in vars(args).items()
                                                                   if k != 'h2h_method_tokens'},
                                         points=list(records.values())))

    for test in args.h2h_tests:
        cases = [c for c in builders[test](args)
                 if not args.h2h_cases or any(t.lower() in c['label'].lower() for t in args.h2h_cases)]
        for case in cases:
            print(f'NOTE h2h {test}/{case["label"]}: {case["notes"]}', flush=True)
            plan = h2h_points(args, case['k'])
            for i, (method, knob, cfg) in enumerate(plan, 1):
                label = f'h2h {test}/{case["label"]} {method} {knob}'
                print(f'START {label}: point={i}/{len(plan)} hall={cfg}', flush=True)
                start = time.perf_counter()
                error, info, status = math.nan, {}, 'ok'
                try:
                    error, info = case['run'](cfg, h2h_ticker(label, case['end'], args.progress_every))
                except (RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
                    status = f'error: {str(exc)[:80]}'
                if status == 'ok' and not math.isfinite(error):
                    status = 'nonfinite'
                record = dict(test=test, case=case['label'], method=method, knob=knob, hall=cfg, status=status,
                              error=float(error) if math.isfinite(error) else None,
                              seconds=float(info['seconds']) if info else None,
                              substeps=info.get('substeps'), steps=info.get('steps'),
                              growth=info.get('growth'), linear=info.get('linear'))
                records[test, case['label'], method, knob] = record
                write()
                print(f'DONE {label}: wall={hms(time.perf_counter()-start)}', flush=True)
                print(f'MEASURE {label}: error={error:.4g} seconds={record["seconds"] or math.nan:.4g} '
                      f'substeps={record["substeps"]} steps={record["steps"]} status={status}', flush=True)
        h2h_pareto([p for p in records.values() if p['test'] == test])
    write()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', default='all', choices=['waves', 'invariants', 'convergence', 'energy',
                                                           'cost', 'shear', 'mri', 'adsi', 'ot',
                                                           'stability', 'precision', 'h2h', 'all'],
                        help='`all` runs the first seven; adsi (2D NumPy), ot, stability and precision '
                             '(2D, torch CUDA) and h2h (see --h2h-*) must be named')
    parser.add_argument('--dim', type=int, choices=[1, 2, 3], default=1, help='3 is only for the waves suite')
    parser.add_argument('--n', type=int, default=None,
                        help='grid points per axis (default 128; 256 for ot; 16 for --dim 3)')
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
    parser.add_argument('--mri-quick', action='store_true',
                        help='mri: only the unstable-eigenmode runs (no stable-case evolution, generic seeds '
                             'or marginal-curve scan)')
    parser.add_argument('--mri-tol', type=float, default=1e-3,
                        help='mri: tolerance of the polarization and frequency (relative to |s|) gates')
    parser.add_argument('--adsi-n', type=int, default=16, help='grid points per axis of the adsi box (one wavelength)')
    parser.add_argument('--adsi-beta', type=float, default=10., help='plasma beta 2p/|B0|^2 of the adsi box')
    parser.add_argument('--adsi-amp', type=float, default=1e-7, help='adsi seed amplitude |b|/|B0|')
    parser.add_argument('--adsi-efolds', type=float, default=8., help='adsi run length in e-foldings')
    parser.add_argument('--adsi-tmax', type=float, default=200., help='adsi cap on the run length')
    parser.add_argument('--adsi-cfl', type=float, default=0.3)
    parser.add_argument('--adsi-tol', type=float, default=1e-3,
                        help='adsi: relative tolerance of the growth, frequency and polarization gates, '
                             'applied at half the CFL step (the Strang splitting is second order)')
    parser.add_argument('--adsi-quick', action='store_true',
                        help='adsi: only the unstable-eigenmode runs with their timestep check (no stable '
                             'evolution, generic seeds or marginal points)')
    parser.add_argument('--ot-di', type=float, default=0.05, help='ot: d_i = eta_H/v_A at the mean state')
    parser.add_argument('--ot-form', choices=['proportional_to_B_over_rho', 'constant'],
                        default='proportional_to_B_over_rho',
                        help='ot: eta_H form; constant avoids eta_H ~ 1/rho at the near-vacuum shocks of OT')
    parser.add_argument('--ot-eta-max', type=float, default=None,
                        help='ot: cap |eta_H(x)| at this value (a physical approximation); default no cap')
    parser.add_argument('--stab-lh-dx', type=float, default=20., help='stability: l_H/dx at the mean state')
    parser.add_argument('--stab-steps', type=int, default=200, help='stability: ideal-CFL steps per case')
    parser.add_argument('--stab-cfl', type=float, default=0.25)
    parser.add_argument('--mri-variants', default='reference,auto,0.05,0.2,0.5',
                        help='comma list: reference, auto, or fixed k*l values for inertia')
    parser.add_argument('--ot-variants', default='all',
                        help='ot: comma list of ref_N, ref_2N, inertia_const, inertia_local4, inertia_fixed to run '
                             '(each is cached), plus `compare` for the gates and speedups from this process and '
                             '--ot-cache; `all` (default) is the five variants and compare')
    parser.add_argument('--ot-compare-only', action='store_true',
                        help='ot: run nothing, gate and compare the variants cached in --ot-cache')
    parser.add_argument('--ot-cache', default=None, help='ot: directory of the per-variant npz and json results')
    parser.add_argument('--ot-tend', type=float, default=0.5,
                        help='ot: end time (snapshots at 0.25 and this time; 0.25 runs the gate alone)')
    parser.add_argument('--stab-cases', default=None,
                        help='stability: comma list of case indices, index ranges a-b, or space-separated label '
                             'words that must all match (e.g. "ref dB=1 contrast=100"); default all. The case '
                             'indices and labels are printed in the NOTE line')
    parser.add_argument('--stab-ref-only', action='store_true', help='stability: only the reference cases')
    parser.add_argument('--stab-inertia-only', action='store_true', help='stability: only the inertia cases')
    parser.add_argument('--stab-cache', default=None, help='stability: directory of the per-case json results')
    parser.add_argument('--stab-report', action='store_true',
                        help='stability: run nothing, gate the cases cached in --stab-cache')
    parser.add_argument('--max-hall-substeps-per-step', type=float, default=None,
                        help='ot and stability: a reference run needing more Hall substeps than this per ideal '
                             'step is stopped, reported as a cost MEASURE and recorded as too expensive')
    parser.add_argument('--progress-every', type=float, default=60.,
                        help='ot, stability and h2h: wall seconds between progress lines')
    parser.add_argument('--h2h-tests', default=None,
                        help='h2h: comma list of waves, mri, shear (default waves,mri, plus shear with --dim 2); '
                             'waves and mri are 1D NumPy whatever --dim says, shear uses --dim 2 and --backend')
    parser.add_argument('--h2h-methods', default='all',
                        help='h2h: comma list of reference, hyper, inertia-auto, inertia-fixed, xmhd-auto, '
                             'xmhd-fixed, the families inertia and xmhd, all, or none (summary only)')
    parser.add_argument('--h2h-cases', default=None, help='h2h: comma list of case-label substrings to keep')
    parser.add_argument('--h2h-out', default=None, help='h2h: JSON of every point, rewritten after each point')
    parser.add_argument('--h2h-in', default='', help='h2h: comma list of earlier --h2h-out files added to the summary')
    parser.add_argument('--h2h-lh-dx', type=float_list, default=[10., 100.], help='h2h waves: l_H/dx values')
    parser.add_argument('--h2h-periods', type=float, default=3., help='h2h waves: whistler periods to run')
    parser.add_argument('--h2h-cfl', type=float, default=0.3, help='h2h waves: ideal CFL number of the outer step')
    parser.add_argument('--h2h-mri-lh-dx', type=float, default=20., help='h2h mri: minimum l_H/dx, which sets N')
    parser.add_argument('--h2h-mri-beta', type=float, default=2.,
                        help='h2h mri: plasma beta; low so that the ideal step does not hide the Hall stiffness')
    parser.add_argument('--h2h-efolds', type=float, default=6., help='h2h mri: run length in e-foldings')
    parser.add_argument('--h2h-ref-safety', type=float_list, default=[0.5, 1.], help='h2h reference: safety values')
    parser.add_argument('--h2h-hyper', type=float_list, default=[0.02, 0.05], help='h2h hyper: C values')
    parser.add_argument('--h2h-safety', type=float_list, default=[0.25, 0.5, 1., 2.],
                        help='h2h inertia-auto: safety values (l^2 ~ 1/safety)')
    parser.add_argument('--h2h-kl', type=float_list, default=[0.05, 0.1, 0.2], help='h2h inertia-fixed: k*l values')
    parser.add_argument('--h2h-cfactor', type=float_list, default=[1., 2., 4.], help='h2h xmhd: c_factor values')
    parser.add_argument('--h2h-xmhd-kl', type=float, default=0.1, help='h2h xmhd-fixed: k*l')
    args = parser.parse_args()
    if args.n is None:
        args.n = 256 if args.suite == 'ot' else 16 if args.dim == 3 else 128
    if args.n < (16 if args.dim == 3 else 24):
        parser.error('--n must be at least 24 (16 in 3D)')
    if args.dim == 3 and args.suite not in ('waves', 'all'):
        parser.error('--dim 3 is only implemented for the waves suite')
    if args.suite in ('ot', 'stability', 'precision') and args.dim != 2:
        parser.error('ot, stability and precision require --dim 2')
    if args.suite == 'precision' and args.backend != 'torch':
        parser.error('precision compares float32 and float64 on torch: use --backend torch --device cuda')
    if not (args.ot_di > 0 and args.stab_lh_dx > 0 and args.stab_steps >= 1 and args.stab_cfl > 0
            and args.mri_tol > 0):
        parser.error('Invalid ot/stability/mri-tol parameters')
    if not (np.isfinite(args.cg_tol) and args.cg_tol > 0):
        parser.error('--cg-tol must be finite and positive')
    if args.ot_eta_max is not None and not (np.isfinite(args.ot_eta_max) and args.ot_eta_max > 0):
        parser.error('--ot-eta-max must be finite and positive')
    if not (np.isfinite(args.shear_floor) and args.shear_floor > 0 and args.shear_box_periods >= 2
            and all(np.isfinite(t) for t in (args.shear_dt, args.shear_fit_start,
                                           args.shear_fit_end, args.shear_end)) and args.shear_dt > 0
            and 0 <= args.shear_fit_start < args.shear_fit_end <= args.shear_end
            and args.shear_fit_end-args.shear_fit_start >= 8*args.shear_dt):
        parser.error('Invalid shear floor, box, timestep or fit window')
    if args.dim == 1 and args.backend != 'numpy':
        parser.error('Use NumPy for the 1D suite; torch CUDA is exercised in 2D.')
    if args.dim == 3 and args.backend != 'numpy':
        parser.error('The 3D waves check runs on NumPy.')
    if args.suite == 'shear' and args.dim != 2:
        parser.error('Density shear requires --dim 2')
    args.h2h_tests = ([t for t in args.h2h_tests.split(',') if t] if args.h2h_tests
                      else ['waves', 'mri'] + (['shear'] if args.dim == 2 else []))
    args.h2h_method_tokens = [t for t in args.h2h_methods.split(',') if t]
    args.h2h_cases = [t for t in (args.h2h_cases or '').split(',') if t]
    args.h2h_in = [t for t in args.h2h_in.split(',') if t]
    if args.suite == 'h2h':
        if any(t not in H2H_TESTS for t in args.h2h_tests) or not args.h2h_tests:
            parser.error(f'--h2h-tests entries must be among {", ".join(H2H_TESTS)}')
        if 'shear' in args.h2h_tests and args.dim != 2:
            parser.error('h2h shear requires --dim 2')
        if not args.h2h_method_tokens or any(
                t not in H2H_METHODS+('inertia', 'xmhd', 'all', 'none') for t in args.h2h_method_tokens):
            parser.error(f'--h2h-methods entries must be among {", ".join(H2H_METHODS)}, inertia, xmhd, all, none')
        lists = (args.h2h_lh_dx, args.h2h_ref_safety, args.h2h_hyper, args.h2h_safety, args.h2h_kl,
                 args.h2h_cfactor)
        if not (all(all(np.isfinite(v) and v > 0 for v in values) for values in lists)
                and all(0 < s <= 2 for s in args.h2h_ref_safety+args.h2h_safety)
                and args.h2h_periods > 0 and args.h2h_cfl > 0 and args.h2h_mri_lh_dx > 0
                and args.h2h_mri_beta > 0 and args.h2h_efolds >= 5 and args.h2h_xmhd_kl > 0):
            parser.error('Invalid h2h knob lists (positive, safety at most 2), periods, cfl, mri parameters')
    if not (np.isfinite(args.conv_dt0) and args.conv_dt0 > 0):
        parser.error('--conv-dt0 must be finite and positive')
    if args.backend == 'torch' and args.device != 'cuda':
        parser.error('The torch comparison targets --device cuda.')
    if not (args.mri_n >= 24 and args.mri_beta > 0 and args.mri_amp > 0 and args.mri_efolds >= 5):
        parser.error('Invalid mri grid, beta, amplitude or e-folding count')
    if not (args.adsi_n >= 12 and args.adsi_beta > 0 and args.adsi_amp > 0 and args.adsi_efolds >= 4
            and args.adsi_tmax > 0 and args.adsi_cfl > 0 and args.adsi_tol > 0):
        parser.error('Invalid adsi grid, beta, amplitude, length, cfl or tolerance')
    for variant in args.mri_variants.split(','):
        if variant not in ('reference', 'auto') and not float(variant) > 0:
            parser.error('--mri-variants entries must be reference, auto or a positive k*l')
    if not (np.isfinite(args.ot_tend) and args.ot_tend >= 0.25 and args.progress_every > 0
            and (args.max_hall_substeps_per_step is None or args.max_hall_substeps_per_step > 0)):
        parser.error('--ot-tend must be at least 0.25; --progress-every and --max-hall-substeps-per-step positive')
    tokens = [t for t in args.ot_variants.split(',') if t]
    if args.ot_compare_only:
        tokens = ['compare']
    if not tokens or any(t not in OT_VARIANTS+('all', 'compare') for t in tokens):
        parser.error(f'--ot-variants entries must be among {", ".join(OT_VARIANTS)}, compare, all')
    args.ot_compare = 'all' in tokens or 'compare' in tokens
    args.ot_run = list(OT_VARIANTS) if 'all' in tokens else [v for v in OT_VARIANTS if v in tokens]
    if args.suite == 'ot' and not args.ot_cache and args.ot_compare and len(args.ot_run) < len(OT_VARIANTS):
        parser.error('ot compare of a partial run needs --ot-cache')
    if args.stab_ref_only and args.stab_inertia_only:
        parser.error('--stab-ref-only and --stab-inertia-only are exclusive')
    if args.stab_report and not args.stab_cache:
        parser.error('--stab-report needs --stab-cache')
    if args.suite == 'stability':
        try:
            stability_select(args, stability_cases())
        except ValueError as error:
            parser.error(str(error))
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        if subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True,
                                   stderr=subprocess.DEVNULL).strip():
            revision += '+dirty'
    except (OSError, subprocess.CalledProcessError):
        revision = 'unknown (no git)'
    global REVISION
    REVISION = revision
    print(f'Hall viability REVISION={revision} backend={args.backend} '
          f'device={args.device if args.backend == "torch" else "cpu"} '
          f'dim={args.dim} N={args.n} precision={"double and single" if args.suite == "precision" else "double"} '
          f'numpy={np.__version__}', flush=True)
    for name in ('phrike/nonideal.py', 'phrike/solver.py', 'phrike/grid.py', 'phrike/equations.py',
                 'phrike/problems/problem_list/hall_mri1d.py', 'phrike/problems/problem_list/adsi_shear.py',
                 'phrike/problems/problem_list/orszag_tang2d.py',
                 'configs/orszag_tang2d.yaml', 'scripts/hall_viability.py'):
        print(f'SOURCE {name} sha256={hashlib.sha256((ROOT/name).read_bytes()).hexdigest()}', flush=True)
    if args.backend == 'torch':
        import torch
        print(f'torch={torch.__version__} GPU={torch.cuda.get_device_name()}', flush=True)
    result = Results()
    for name, run in [('waves', waves), ('invariants', invariants), ('shear', shear), ('cost', cost),
                      ('convergence', convergence), ('energy', energy), ('mri', mri),
                      ('adsi', adsi), ('ot', ot), ('stability', stability), ('precision', precision),
                      ('h2h', h2h)]:
        if name in ('adsi', 'ot', 'stability', 'precision', 'h2h'):
            if args.suite == name:
                run(args, result)
            continue
        if args.dim == 3 and name != 'waves':
            continue
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
