"""Exponential averaged-state flux coupling for periodic Gaussian dust/MHD.

Predictions use barycentric conservative slow coordinates and analytic slip
and covariance paths. Acceptance is in raw variables, with paired source
impulses. Nonlinear numerical fluxes are evaluated after averaging raw M2;
this approximation is generally second order, not a waveform-flux quadrature.
Coefficient envelopes retain their products with exponential modes. Their
integration cost is independent of the number of gyroperiods in a step.
"""
from __future__ import annotations

import copy
import numpy as np

from . import dust_moments as dm, dust_sources, sgs
from .exponential_path import ExponentialPath as Path, slip_path, covariance_path


class _Rejected(FloatingPointError):
    def __init__(self, message, mask):
        super().__init__(message)
        self.mask = mask


def _poly(values):
    """Envelope in theta=t/h, through zero, half and one (or zero/half)."""
    if len(values) == 1:
        return np.asarray(values)
    a, b = values[:2]
    if len(values) == 2:
        return np.stack((a, 2*(b-a)))
    c = values[2]
    return np.stack((a, 4*b-3*a-c, 2*(a-2*b+c)))


def _outer(a, b):
    return Path.stack([Path.stack([a.component(i)*b.component(j) for j in range(3)])
                       for i in range(3)])


def _trace(a):
    return sum((a.component(i, i) for i in range(3)), Path.constant(0.))


def _matrix(nu, omega):
    """A v = -nu v + v cross omega, component-first."""
    out = np.zeros((3, 3, *np.shape(nu)))
    for i in range(3):
        out[i, i] = -nu
    out[0, 1], out[1, 0] = omega[2], -omega[2]
    out[0, 2], out[2, 0] = -omega[1], omega[1]
    out[1, 2], out[2, 1] = omega[0], -omega[0]
    return out


def _cov_matrix(A):
    eye = np.eye(3).reshape((3, 3, *([1]*(A.ndim-2))))
    return (np.einsum('ik...,jl...->ijkl...', A, eye)
            + np.einsum('ik...,jl...->ijkl...', eye, A)).reshape((9, 9, *A.shape[2:]))


def _apply_envelope(path, matrices):
    result = path*0
    for p, matrix in enumerate(matrices):
        factor = np.zeros((p+1,)); factor[-1] = 1
        result = result + path.linear_map(matrix)*Path.polynomial(factor)
    return result


class _Trial:
    def __init__(self, solver, h, first):
        self.s, self.h, self.first = solver, h, first
        self.g0, self.d0, self.e0 = solver.U_gas, solver.U_dust, solver.e_s
        self.prescribed = bool(getattr(solver, '_prescribed_gas', False))
        self.x0, self.w0, self.C0 = self.coordinates(self.g0, self.d0, self.e0)
        self.coeff0 = self.coefficients(self.g0, self.d0)
        self.nu0, self.alpha0, self.om0 = self.coeff0
        mag = np.sqrt(np.sum(self.om0**2, axis=0))
        self.omega = mag
        self.b = dm._divide(self.om0, mag)
        self.A0 = _matrix(self.nu0, self.om0)
        self.L0 = self.alpha0*self.A0
        self.H0 = _cov_matrix(self.A0)
        self.D, self.decay = self.reservoir()
        self.ledger = {}

    def coordinates(self, g, d, e):
        x = np.concatenate((g.copy(), d[:1], e[None]))
        x[1:4] += d[1:4]
        x[4] += self.s.closure.kinetic_energy(d) + e
        p = self.s.closure.primitives(d)
        return x, p['u']-g[1:4]/g[0], p['C']

    def state(self, x, w, C):
        rd, es = x[8], x[9]
        total = x[0]+rd
        ud = x[1:4]/total + (x[0]/total)*w
        if self.prescribed:
            ud = self.g0[1:4]/self.g0[0]+w
        d = self.s.closure.from_primitives(rd, ud, C)
        g = x[:8].copy()
        g[1:4] -= d[1:4]
        g[4] -= self.s.closure.kinetic_energy(d)+es
        if self.prescribed:
            g = self.g0.copy()
        self.check(g, d, es)
        return g, d, es

    def check(self, g, d, e):
        s = self.s
        bad = (~np.all(np.isfinite(g), axis=0) | ~np.isfinite(e)
               | (g[0] < s.gas_solver.rho_min) | ((d[0] <= 0) & s.dust_enabled) | (e < 0)
               | ((s.equations.gamma-1)*s._internal_energy(g) < s.gas_solver.p_min)
               | ~s.closure.realizable(d))
        if np.any(bad):
            raise _Rejected('inadmissible exponential stage or acceptance', bad)

    def coefficients(self, g, d):
        old = self.s.U_gas
        try:
            self.s.U_gas = g
            nu = self.s._nu() if self.s.dust_enabled else np.zeros_like(g[0])
        finally:
            self.s.U_gas = old
        alpha = 1 + (d[0]/g[0] if not self.prescribed else 0)
        return nu, np.broadcast_to(alpha, g[0].shape), float(
            self.s.dust_config.get('charge_to_mass', 0.))*g[5:8]

    def reservoir(self):
        """Existing midpoint-rate reservoir budget and ETD2 directional forcing."""
        s, es, h = self.s, self.e0, self.h
        zero = np.zeros_like(es)
        if not s.sgs_enabled:
            return Path.constant(np.zeros((3, 3, *es.shape)), 2), zero
        model = s.sgs_config.get('d_model', s._default_d_model())
        def coefficients(e):
            return sgs.sgs_coefficients(e, self.g0[0], self.nu0, s.Delta,
                                       model, omega=self.omega)
        def rates(e):
            c = coefficients(e)
            rd = dm._divide(self.d0[0]*(c['D_par']+2*c['D_perp']), e)
            re = (float(s.sgs_config.get('c_eps', 1.))*np.sqrt(e/self.g0[0])/s.Delta
                  if s.sgs_config.get('decay', True) else zero)
            return c, rd, re
        _, rd, re = rates(es)
        mid, rd, re = rates(es*np.exp(-.5*h*(rd+re)))
        weight = dm.phi(rd+re, h)
        dust, decay = es*rd*weight, es*re*weight
        end = es*np.exp(-(rd+re)*h)
        amplitude = dm._divide(dust, self.d0[0]*h)
        work = mid['D_par']+2*mid['D_perp']
        start, final = coefficients(es), coefficients(end)
        tensors = [np.zeros((3, 3, *es.shape)) for _ in range(2)]
        for key, parallel in (('D_par', True), ('D_perp', False)):
            effective = amplitude*dm._divide(mid[key], work)
            slope = np.clip(final[key]-start[key], -2*effective, 2*effective)
            a = effective-.5*slope
            projector = self.b[:, None]*self.b[None, :]
            if not parallel:
                projector = np.eye(3).reshape((3, 3, *([1]*es.ndim)))-projector
            # The zero-field frame uses z, just as the local phi blocks do.
            if not np.any(self.b):
                projector = np.zeros_like(projector)
                for i in range(3):
                    projector[i, i] = float((i == 2) == parallel)
            tensors[0] += projector*a
            tensors[1] += projector*slope
        return Path.polynomial(np.stack(tensors), 2), decay/h

    def transport(self, state):
        g, d, es = state
        self.s.equations.p_sgs = ((2/3)*es if self.s.sgs_enabled and
                                 self.s.sgs_config.get('pressure', True) else None)
        rg = self.s.gas_solver._compute_rhs(g)
        if self.prescribed:
            rg = np.zeros_like(g)
        rd = dm._transport_rhs(d, self.s.closure, self.s.dxs, self.first)[0]
        if not self.s.dust_enabled:
            rd[:] = 0
        p = self.s.closure.primitives(d)
        ug, ud, C = g[1:4]/g[0], p['u'], p['C']
        density = np.where(d[0] > 0, d[0], 1.)
        dud = (rd[1:4]-ud*rd[0])/density
        dug = (rg[1:4]-ug*rg[0])/g[0]
        R2 = np.empty_like(C)
        for k, (i, j) in enumerate(dm._PAIRS):
            R2[i, j] = R2[j, i] = rd[4+k]
        dc = ((R2-(C+ud[:, None]*ud[None, :])*rd[0])/density
              - dud[:, None]*ud[None, :] - ud[:, None]*dud[None, :])
        sx = np.concatenate((rg.copy(), rd[:1], np.zeros_like(es)[None]))
        sx[1:4] += rd[1:4]
        sx[4] += .5*np.sum(rd[4:7], axis=0)
        return sx, dud-dug, dc, rg, rd

    def paths(self, tw, tc, coeffs, previous=None):
        fw, fc = Path.polynomial(tw, 1), Path.polynomial(tc, 2)+2*self.D
        if previous is not None:
            nu = _poly([c[0] for c in coeffs])
            alpha = _poly([c[1] for c in coeffs])
            omega = _poly([c[2] for c in coeffs])
            # Fit physical coefficients, never samples of an oscillating force.
            Am = [_matrix(nu[k] if k < len(nu) else np.zeros_like(nu[0]),
                           omega[k] if k < len(omega) else np.zeros_like(omega[0]))
                  for k in range(max(len(nu), len(omega)))]
            Lm = np.zeros((len(alpha)+len(Am)-1, *Am[0].shape))
            for i, a in enumerate(alpha):
                for j, A in enumerate(Am):
                    Lm[i+j] += a*A
            Lm[0] -= self.L0
            Hm = np.stack([_cov_matrix(A) for A in Am]); Hm[0] -= self.H0
            fw = fw + _apply_envelope(previous[0], Lm)
            fc = fc + _apply_envelope(previous[1].reshape((9,)), Hm).reshape((3, 3))
        return (slip_path(self.w0, self.h, self.nu0, self.omega, self.b, fw, self.alpha0),
                covariance_path(self.C0, self.h, self.nu0, self.omega, self.b, fc))

    def averaged_state(self, x, paths, theta, states=None):
        w, C = paths
        if states is not None:
            # Use the same slow envelopes as raw-source acceptance. This
            # retains density/mean correlations as well as slip variance.
            rho = Path.polynomial(_poly([q[1][0] for q in states]))
            fraction = Path.polynomial(_poly([q[0][0]/(q[0][0]+q[1][0])
                if not self.prescribed else np.ones_like(q[0][0]) for q in states]))
            velocity = Path.polynomial(_poly([(q[0][1:4]+q[1][1:4])/(q[0][0]+q[1][0])
                if not self.prescribed else q[0][1:4]/q[0][0] for q in states]), 1)
            ud = velocity+fraction*w
            dust = np.empty_like(self.d0)
            dust[0] = rho.integral(theta)/theta
            dust[1:4] = (rho*ud).integral(theta)/theta
            for k, (i, j) in enumerate(dm._PAIRS):
                dust[4+k] = (rho*(ud.component(i)*ud.component(j)+C.component(i, j))).integral(theta)/theta
            gas = x[:8].copy()
            gas[1:4] -= dust[1:4]
            gas[4] -= self.s.closure.kinetic_energy(dust)+x[9]
            if self.prescribed:
                gas = self.g0.copy()
            self.check(gas, dust, x[9])
            return gas, dust, x[9]
        wm, cm = w.integral(theta)/theta, C.integral(theta)/theta
        # Reclose only after averaging raw M2, including the oscillating mean.
        fraction = x[0]/(x[0]+x[8]) if not self.prescribed else np.ones_like(x[0])
        for i in range(3):
            for j in range(i, 3):
                ww = (w.component(i)*w.component(j)).integral(theta)/theta
                cm[i, j] += fraction**2*(ww-wm[i]*wm[j])
                cm[j, i] = cm[i, j]
        return self.state(x, wm, cm)

    def slow_stage(self, sample, paths, theta):
        state = self.averaged_state(sample, paths, theta)
        rate = self.transport(state)[0]
        x = self.x0 + theta*self.h*rate
        x[9] -= self.h*(Path.polynomial(_poly([self.d0[0]]))*_trace(self.D)
                         ).integral(theta) + theta*self.h*self.decay
        return x

    def source_integral(self, paths, states, coeffs, frozen):
        """Raw Q_n on final path, or complete raw source residual on prior path."""
        w, C = paths
        rho = Path.polynomial(_poly([q[1][0] for q in states]))
        gamma = Path.polynomial(_poly([q[0][0]/(q[0][0]+q[1][0])
                                      if not self.prescribed else np.ones_like(q[0][0])
                                      for q in states]))
        velocity = Path.polynomial(_poly([(q[0][1:4]+q[1][1:4])/(q[0][0]+q[1][0])
                                         if not self.prescribed else q[0][1:4]/q[0][0]
                                         for q in states]), 1)
        ud = velocity+gamma*w
        if frozen:
            dw = w.linear_map(self.L0)
            dc = C.reshape((9,)).linear_map(self.H0).reshape((3, 3))+2*self.D
        else:
            nu, al, om = (_poly([c[i] for c in coeffs]) for i in range(3))
            As = [_matrix(a, b) for a, b in zip(nu, om)]
            Ls = np.zeros((len(al)+len(As)-1, *As[0].shape))
            for i, a in enumerate(al):
                for j, A in enumerate(As):
                    Ls[i+j] += a*A
            Ls[0] -= self.L0
            Hs = np.stack([_cov_matrix(A) for A in As]); Hs[0] -= self.H0
            dw = _apply_envelope(w, Ls)
            dc = _apply_envelope(C.reshape((9,)), Hs).reshape((3, 3))
        du = gamma*dw
        momentum = self.h*(rho*du).integral()
        raw = np.zeros_like(self.d0)
        raw[1:4] = momentum
        for k, (i, j) in enumerate(dm._PAIRS):
            raw[4+k] = self.h*(rho*(ud.component(i)*du.component(j)
                + du.component(i)*ud.component(j)+dc.component(i, j))).integral()
        return raw

    def run(self):
        state0 = (self.g0, self.d0, self.e0)
        t0 = self.transport(state0)
        pa = self.paths(t0[1][None], t0[2][None], [self.coeff0])
        xa = self.slow_stage(self.x0, pa, .5)
        a = self.state(xa, pa[0].evaluate(.5), pa[1].evaluate(.5))
        ca, ta = self.coefficients(a[0], a[1]), self.transport(a)
        pb = self.paths((2*ta[1]-t0[1])[None], (2*ta[2]-t0[2])[None],
                        [self.coeff0, ca], pa)
        xb = self.slow_stage(2*xa-self.x0, pb, 1.)
        b = self.state(xb, pb[0].evaluate(), pb[1].evaluate())
        cb, tb = self.coefficients(b[0], b[1]), self.transport(b)
        pf = self.paths(_poly([t0[1], ta[1], tb[1]]), _poly([t0[2], ta[2], tb[2]]),
                        [self.coeff0, ca, cb], pb)
        # The ETD-RK3 slow block has Simpson weights on its residual samples;
        # its fast-to-slow block is the analytic integral of this final path.
        sample = (self.x0+4*xa+xb)/6
        final_prediction = self.slow_stage(sample, pf, 1.)
        self.state(final_prediction, pf[0].evaluate(), pf[1].evaluate())
        # D1: the complete raw residual, including transport, belongs to the
        # previous path; only the frozen-source integral uses the final path.
        states, coefficients = [state0, a, b], [self.coeff0, ca, cb]
        avg = self.averaged_state(sample, pb, 1., states)
        _, _, _, rg, rd = self.transport(avg)
        impulse = self.source_integral(pf, states, coefficients, True)
        impulse += self.source_integral(pb, states, coefficients, False)
        d = self.d0+self.h*rd+impulse
        g = self.g0+self.h*rg
        g[1:4] -= impulse[1:4]
        rho = Path.polynomial(_poly([q[1][0] for q in states]))
        withdrawal = self.h*(rho*_trace(self.D)).integral()
        es = self.e0-withdrawal-self.h*self.decay
        g[4] -= .5*np.sum(impulse[4:7], axis=0)-withdrawal-self.h*self.decay
        if self.prescribed:
            g = self.g0.copy()
        _project(self.s, g)
        self.check(g, d, es)
        nu = Path.polynomial(_poly([c[0] for c in coefficients]))
        mean = self.h*(rho*nu*sum((pf[0].component(i)*pf[0].component(i)
                                 for i in range(3)), Path.constant(0.))).integral()
        random = self.h*(rho*nu*_trace(pf[1])).integral()
        for value in (mean, random, withdrawal):
            if np.any(value < -1e-12*np.maximum(1., np.abs(value))):
                raise _Rejected('negative analytic source ledger', value < 0)
        self.ledger = dict(drag_heat_mean=self.s._integral(mean),
                           drag_heat_random=self.s._integral(random),
                           dust_work=self.s._integral(withdrawal),
                           sgs_decay=self.s._integral(self.h*self.decay))
        # Work is an independent path integral, never inferred from a kinetic
        # energy difference. Here it is the resolved transport acceleration's
        # mechanical work on barycentric and relative motion.
        reduced = Path.polynomial(_poly([q[1][0]/c[1] for q, c in zip(states, coefficients)]))
        mass = Path.polynomial(_poly([q[0][0]+q[1][0] for q in states]))
        bary = Path.polynomial(_poly([(q[0][1:4]+q[1][1:4])/(q[0][0]+q[1][0])
                                     for q in states]), 1)
        acceleration = Path.polynomial(_poly([
            (t[0][1:4]-v*(t[0][0]+t[0][8]))/(q[0][0]+q[1][0])
            for t, q, v in zip((t0, ta, tb), states,
                [(q[0][1:4]+q[1][1:4])/(q[0][0]+q[1][0]) for q in states])]), 1)
        forcing = Path.polynomial(_poly([t[1] for t in (t0, ta, tb)]), 1)
        work = sum((reduced*pf[0].component(i)*forcing.component(i)
                    + mass*bary.component(i)*acceleration.component(i) for i in range(3)),
                   Path.constant(0.))
        self.ledger['mechanical_work'] = self.s._integral(self.h*work.integral())
        self.s.U_gas, self.s.U_dust, self.s.e_s = g, d, es
        for key, value in self.ledger.items():
            self.s.ledger[key] = self.s.ledger.get(key, 0.)+value


def _project(s, g):
    """Project B at fixed gas/mixture total energy; never rebuild pressure."""
    if s.gas_solver.divergence_clean == 'none':
        return
    if s.ndim == 1:
        g[5] = s.grid.project_solenoidal_x(g[5])
    else:
        g[5:5+s.ndim] = np.stack(s.grid.project_solenoidal(*g[5:5+s.ndim]))


def _filter(s, h, final=False):
    old = s.U_gas
    new = (s.grid.apply_accepted_state_stabilization(old, h) if final else
           s.grid.apply_spectral_dissipation(old, h))
    drop = s._kinetic(old)-s._kinetic(new)
    s.U_gas = new
    _project(s, new)
    bad = ((new[0] < s.gas_solver.rho_min) |
           ((s.equations.gamma-1)*s._internal_energy(new) < s.gas_solver.p_min))
    if np.any(bad):
        raise _Rejected('gas filter would activate a clamp', bad)
    s.ledger['filter_kinetic_drop'] += s._integral(drop)
    if s.sgs_enabled and s.sgs_config.get('production', 'filter_drain') == 'filter_drain':
        production = sgs.drain_production(drop, s.dxs, s.Delta, s._internal_energy(new),
            c_ell=s.sgs_config.get('c_ell', 1.), f_cap=s.sgs_config.get('f_cap', .5))
        s.U_gas[4] -= production
        s.e_s += production
        s.ledger['production'] += s._integral(production)


def _post_transport(s, h, velocity):
    if s.sgs_enabled:
        s.e_s, count = sgs.advect_scalar(s.e_s, velocity[:s.ndim], h, s.dxs, cfl=s.cfl)
        s.transport_stats['sgs_substeps'] += count
        divu = sum(getattr(s.grid, name)(velocity[d]) for d, name in
                   enumerate(('dx1', 'dy1', 'dz1')[:s.ndim]))
        change = np.minimum(s.e_s*np.expm1(-(2/3)*h*divu), .5*s._internal_energy(s.U_gas))
        s.e_s += change; s.U_gas[4] -= change
        s.ledger['compression_to_sgs'] += s._integral(change)
        diffusion = s.sgs_config.get('diffusion', {})
        if diffusion.get('enabled', False):
            kappa = diffusion.get('kappa', float(diffusion.get('c_kappa', 1.))*
                                  np.sqrt(2*s.e_s/(3*s.U_gas[0]))*s.Delta)
            s.e_s, count = sgs.diffuse_scalar(s.e_s, kappa, h, s.dxs)
            s.transport_stats['diffusion_substeps'] += count
    diffusion = s.dust_config.get('eddy_diffusion', {})
    if s.dust_enabled and s.sgs_enabled and diffusion.get('enabled', False):
        c = sgs.sgs_coefficients(s.e_s, s.U_gas[0], s._nu(), s.Delta,
                                 s.sgs_config.get('d_model', s._default_d_model()),
                                 omega=float(s.dust_config.get('charge_to_mass', 0.))*
                                 np.sqrt(np.sum(s.U_gas[5:8]**2, axis=0)))
        fraction = np.divide(c['x'], 1+c['x'], out=np.ones_like(c['x']), where=np.isfinite(c['x']))
        kappa = float(diffusion.get('c_kappa', 1.))*c['sigma']*s.Delta*fraction
        s.U_dust, count = dm.diffuse(s.U_dust, kappa, h, s.dxs)
        s.transport_stats['diffusion_substeps'] += count


def _snapshot(s):
    return (s.U_gas.copy(), s.U_dust.copy(), s.e_s.copy(), s.t,
            s.ledger.copy(), s.transport_stats.copy(), copy.copy(vars(s.positivity)),
            s.equations.p_sgs)


def _restore(s, snapshot):
    g, d, e, s.t, ledger, stats, positivity, pressure = snapshot
    s.U_gas, s.U_dust, s.e_s = g.copy(), d.copy(), e.copy()
    s.ledger, s.transport_stats = ledger.copy(), stats.copy()
    vars(s.positivity).update(positivity)
    s.equations.p_sgs = pressure
    s.gas_solver.t, s.gas_solver.U = s.t, s.U_gas


def exponential_step(s, h):
    """Transactional ETD-RK3 attempt, MOOD retry, then resolved Strang recovery."""
    if not np.isfinite(h) or h <= 0:
        raise ValueError('dt must be finite and positive')
    s._validate_state()
    if s.gas_solver.resistivity != 0 or s.gas_solver.hall or s.gas_solver.ambipolar:
        raise ValueError('exponential coupling supports ideal MHD only')
    if not s.dust_config.get('backreaction', True) and not getattr(s, '_prescribed_gas', False):
        raise ValueError('exponential backreaction=False requires the private prescribed-gas test harness')
    saved = _snapshot(s)
    first = np.zeros(s.shape, bool)
    rejections, mood, clamp = 0, 0, 0
    for attempt in range(2):
        _restore(s, saved)
        try:
            _filter(s, .5*h)
            before = s.U_gas[1:4]/s.U_gas[0]
            trial = _Trial(s, h, first)
            trial.check(s.U_gas, s.U_dust, s.e_s)
            trial.run()
            _filter(s, .5*h, final=True)
            _post_transport(s, h, .5*(before+s.U_gas[1:4]/s.U_gas[0]))
            trial.check(s.U_gas, s.U_dust, s.e_s)
            s.t += h
            break
        except _Rejected as error:
            rejections += 1
            first |= dm._dilate(error.mask)
            mood = int(np.count_nonzero(first))
        except (FloatingPointError, np.linalg.LinAlgError):
            rejections += 1
            first[:] = True
            mood = first.size
        except Exception:
            _restore(s, saved)
            raise
    else:
        _restore(s, saved)
        # Resolve the loaded slip rate too. A recovery is never counted as AP
        # evidence; the source-scale work is explicit in diagnostics.
        rate = float(np.max((1+s.U_dust[0]/s.U_gas[0])*
                            np.maximum(s._nu(), abs(float(s.dust_config.get('charge_to_mass', 0.)))*
                                       np.sqrt(np.sum(s.U_gas[5:8]**2, axis=0)))))
        nsub = max(1, int(np.ceil(10*h*rate)))
        try:
            for _ in range(20):
                if nsub > 131072:
                    raise RuntimeError('resolved Strang fallback exceeds 131072 substeps')
                _restore(s, saved)
                try:
                    for j in range(nsub):
                        events = s.positivity.events
                        local_rate = float(np.max((1+s.U_dust[0]/s.U_gas[0])*
                            np.maximum(s._nu(), abs(float(s.dust_config.get('charge_to_mass', 0.)))*
                                       np.sqrt(np.sum(s.U_gas[5:8]**2, axis=0)))))
                        if h*local_rate/nsub > .1*(1+1e-12):
                            raise _Rejected('fallback no longer resolves source rate', np.ones(s.shape, bool))
                        s._strang_step(h/nsub)
                        end_rate = float(np.max((1+s.U_dust[0]/s.U_gas[0])*
                            np.maximum(s._nu(), abs(float(s.dust_config.get('charge_to_mass', 0.)))*
                                       np.sqrt(np.sum(s.U_gas[5:8]**2, axis=0)))))
                        if h*end_rate/nsub > .1*(1+1e-12):
                            raise _Rejected('fallback end rate exceeds resolution bound', np.ones(s.shape, bool))
                        if s.positivity.events != events:
                            clamp += 1
                            raise _Rejected('Strang fallback activated gas clamp', np.ones(s.shape, bool))
                        s._validate_state()
                    break
                except (_Rejected, ValueError, FloatingPointError):
                    nsub *= 2
            else:
                raise RuntimeError('resolved Strang fallback failed admissibility')
        except Exception:
            _restore(s, saved)
            raise
        s.transport_stats['exponential_fallback_substeps'] = (
            s.transport_stats.get('exponential_fallback_substeps', 0)+nsub)
    s.transport_stats['exponential_rejections'] = s.transport_stats.get('exponential_rejections', 0)+rejections
    s.transport_stats['exponential_mood_cells'] = s.transport_stats.get('exponential_mood_cells', 0)+mood
    s.transport_stats['exponential_clamp_failures'] = s.transport_stats.get('exponential_clamp_failures', 0)+clamp
    s.gas_solver.t, s.gas_solver.U = s.t, s.U_gas
    s._set_pressure()
    return s.U_gas
