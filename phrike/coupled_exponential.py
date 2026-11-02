"""Local exponential building blocks for the coupled-integrator decision study.

All physical arrays are float64 and component-first.  The slip generator is
L = -(1 + rho_d/rho_g) (nu I + omega [b cross]); central moments instead use
A = -nu I - omega [b cross].  This distinction is essential at finite loading.
These routines do not change any production solver or choose a transport method.

``homogeneous_source`` keeps densities and the field axis fixed.  Its external
accelerations are d(t) = f_d - f_g = d0 + t*d1 and the barycentric acceleration
a(t) = (f_g + epsilon*f_d)/(1 + epsilon) = a0 + t*a1.
Diffusion is D(t) = D0 + t*D1, with C' = A C + C A.T + 2 D.
Heat is integrated from nu*rho_d*(|w|**2 + trace(C)), independently of work.
The mean-square integral uses exponential-polynomial antiderivatives, with a
convergent series near zero; it never resolves gyroperiods by quadrature.
"""

from __future__ import annotations

import math

import numpy as np


def _phi(k, z):
    z = np.asarray(z, dtype=np.complex128)
    if k == 0:
        return np.exp(z)
    small = np.abs(z) < 1.0
    zs = np.where(small, z, 0)
    term = np.full_like(z, 1/math.factorial(k))
    series = term.copy()
    for j in range(1, 28):
        term = term*zs/(j + k)
        series += term
    safe = np.where(small, 1, z)
    value = np.expm1(z)/safe
    for j in range(1, k):
        value = (value - 1/math.factorial(j))/safe
    return np.where(small, series, value)


def phi(k, z):
    """Cancellation-safe complex phi_k(z), k=0,...,3; phi_k(0)=1/k!."""
    if k not in (0, 1, 2, 3):
        raise ValueError("phi supports k=0,1,2,3")
    return _phi(k, z)


def _frame(bhat, shape):
    b = np.asarray(bhat, dtype=np.float64)
    b = np.broadcast_to(b.reshape((3, *([1]*(len(shape) - b.ndim + 1)),
                                    *b.shape[1:])), (3, *shape)).copy()
    norm = np.sqrt(np.sum(b*b, axis=0))
    active = norm > 0
    if np.any(active & (np.abs(norm - 1) > 1e-12)):
        raise ValueError("bhat must be a unit vector or zero")
    b[2] = np.where(active, b[2], 1)
    axis = np.argmin(np.abs(b), axis=0)
    seed = np.stack([axis == i for i in range(3)]).astype(np.float64)
    e1 = np.cross(seed, b, axisa=0, axisb=0, axisc=0)
    e1 /= np.sqrt(np.sum(e1*e1, axis=0))
    e2 = np.cross(b, e1, axisa=0, axisb=0, axisc=0)
    return np.stack((e1, e2, b)), active


def slip_phi(k, w, h, nu, omega, bhat, alpha=1):
    """Apply phi_k(h L) to a (3,...) slip without a matrix exponential."""
    if k not in (0, 1, 2, 3):
        raise ValueError("slip_phi supports k=0,1,2,3")
    return _slip_phi(k, w, h, nu, omega, bhat, alpha)


def _slip_phi(k, w, h, nu, omega, bhat, alpha=1):
    w = np.asarray(w, dtype=np.float64)
    shape = np.broadcast_shapes(w.shape[1:], np.shape(nu), np.shape(omega),
                               np.shape(alpha), np.shape(h), np.shape(bhat)[1:])
    frame, active = _frame(bhat, shape)
    wc = np.einsum("ai...,i...->a...", frame, w)
    z = -np.asarray(alpha)*np.asarray(h)*(np.asarray(nu) + 1j*np.asarray(omega)*active)
    transverse = _phi(k, z)*(wc[0] + 1j*wc[1])
    result = np.stack((transverse.real, transverse.imag,
                       (_phi(k, z.real)*wc[2]).real))
    return np.einsum("ai...,a...->i...", frame, result)


def covariance_phi(k, C, h, nu, omega, bhat):
    """Apply phi_k(h*(A C+C A.T)) using the four corrected b-frame blocks."""
    if k not in (0, 1, 2, 3):
        raise ValueError("covariance_phi supports k=0,1,2,3")
    C = np.asarray(C, dtype=np.float64)
    shape = np.broadcast_shapes(C.shape[2:], np.shape(nu), np.shape(omega),
                               np.shape(h), np.shape(bhat)[1:])
    frame, active = _frame(bhat, shape)
    c = np.einsum("ai...,ij...,bj...->ab...", frame, C, frame)
    z = -2*np.asarray(nu)*h
    f0 = phi(k, z).real
    cross = phi(k, z - 1j*np.asarray(omega)*h*active)*(c[0, 2] + 1j*c[1, 2])
    spin2 = phi(k, z - 2j*np.asarray(omega)*h*active)*(c[0, 0] - c[1, 1] + 2j*c[0, 1])
    iso = 0.5*f0*(c[0, 0] + c[1, 1])
    out = np.empty_like(c)
    out[0, 0], out[1, 1], out[2, 2] = iso + 0.5*spin2.real, iso - 0.5*spin2.real, f0*c[2, 2]
    out[0, 1] = out[1, 0] = 0.5*spin2.imag
    out[0, 2] = out[2, 0] = cross.real
    out[1, 2] = out[2, 1] = cross.imag
    return np.einsum("ai...,ab...,bj...->ij...", frame, out, frame)


def _dd(k, *nodes):
    """Divided differences at the HyQMOM nodes, including their common zero."""
    nodes = np.broadcast_arrays(*nodes)
    degree = len(nodes) - 1
    small = np.maximum.reduce([np.abs(x) for x in nodes]) < 1
    # Complete homogeneous polynomials: sum h_j(nodes)/(j+degree+k)!.
    polys = [np.ones_like(nodes[0])] + [np.zeros_like(nodes[0]) for _ in range(27)]
    for node in nodes:
        x = np.where(small, node, 0)
        for j in range(1, 28):
            polys[j] += x*polys[j-1]
    series = sum(poly/float(math.factorial(j + degree + k))
                 for j, poly in enumerate(polys))
    if degree == 1:
        numerator = phi(k, nodes[1]) - phi(k, nodes[0])
    else:
        numerator = _dd(k, *nodes[1:]) - _dd(k, *nodes[:-1])
    denominator = np.where(small, 1, nodes[-1] - nodes[0])
    return np.where(small, series, numerator/denominator).real


def hyqmom_phi(k, state, h, nu, Dpar, Dperp):
    """Apply phi_k(h H) to (v,q,r,sperp2,one), including affine couplings.

    ``one`` is an ordinary input component; for an affine state it is 1.
    No diagonal-only approximation is made to the fourth-moment generator.
    """
    if k not in (0, 1, 2, 3):
        raise ValueError("hyqmom_phi supports k=0,1,2,3")
    v, q, r, s, one = np.asarray(state, dtype=np.float64)
    z = -np.asarray(nu)*h
    dp, dt = np.asarray(Dpar)*h, np.asarray(Dperp)*h
    f2, f3, f4 = (phi(k, j*z).real for j in (2, 3, 4))
    zero = np.zeros_like(z)
    out = np.broadcast_arrays(f2*v + 2*dp*_dd(k, 2*z, zero)*one,
                              f3*q,
                              f4*r + 12*dp*_dd(k, 4*z, 2*z)*v
                              + 24*dp*dp*_dd(k, 4*z, 2*z, zero)*one,
                              f2*s + 2*dt*_dd(k, 2*z, zero)*one,
                              phi(k, zero).real*one)
    return np.stack(out)


def _mean_square(z, w, p, q):
    """Integral_0^1 |e^(zs)w+s phi1(zs)p+s² phi2(zs)q|² ds."""
    z, w, p, q = np.broadcast_arrays(z, w, p, q)
    small = np.abs(z) < 1
    zs = np.where(small, z, 0)
    coeff = [w, zs*w + p]
    coeff.append((zs*coeff[-1] + q)/2)
    for j in range(3, 26):
        coeff.append(zs*coeff[-1]/j)
    series = np.zeros_like(z.real)
    for j, a in enumerate(coeff):
        series += np.abs(a)**2/(2*j + 1)
        for m in range(j):
            series += 2*(a*np.conj(coeff[m])).real/(j + m + 1)
    safe = np.where(small, 1, z)
    a = w + p/safe + q/safe**2
    b, c = -p/safe - q/safe**2, -q/safe
    value = (np.abs(a)**2*_phi(1, 2*z.real).real + np.abs(b)**2
             + (b*np.conj(c)).real + np.abs(c)**2/3
             + 2*(a*np.conj(b)*_phi(1, z)
                  + a*np.conj(c)*(_phi(1, z) - _phi(2, z))).real)
    scale = (np.abs(a)**2*np.abs(_phi(1, 2*z.real)) + np.abs(b)**2
             + np.abs(b*c) + np.abs(c)**2/3
             + 2*np.abs(a)*(np.abs(b*_phi(1, z))
                            + np.abs(c*(_phi(1, z) - _phi(2, z)))))
    return _nonnegative(np.where(small, series, value),
                        np.where(small, sum(np.abs(x) for x in coeff)**2, scale))


def _nonnegative(value, scale):
    """Remove only bounded roundoff; a significant negative integral is an error."""
    if np.any(value < -512*np.finfo(np.float64).eps*scale):
        raise FloatingPointError("negative analytic heat integral exceeds roundoff")
    return np.maximum(value, 0)


def homogeneous_source(ug, ud, h, *, rho_g, rho_d, nu, omega=0, bhat=None,
                       C=None, hyqmom=None, d0=None, d1=None, a0=None, a1=None,
                       D0=None, D1=None):
    """Exact fixed-coefficient homogeneous flow with affine time forcing.

    Supply exactly one of ``C`` (3,3,...) or ``hyqmom`` (4,...), containing
    (v,q,r,sperp2). D0,D1 are symmetric (3,3,...) diffusivity tensors; HyQMOM
    requires them gyrotropic about bhat. Their endpoint values must be PSD.
    Returned work/heat are energy densities. SGS_work is energy delivered to
    random dust motion, before drag dissipation; mechanical_work is external
    work on both species, with no electromagnetic heat. This is a prescribed
    diffusion testbed, not a nonlinear SGS-reservoir integrator.
    """
    if (C is None) == (hyqmom is None):
        raise ValueError("supply exactly one of C and hyqmom")
    if h < 0 or not np.isfinite(h):
        raise ValueError("h must be finite and non-negative")
    ug, ud = np.asarray(ug, dtype=np.float64), np.asarray(ud, dtype=np.float64)
    rho_g, rho_d, nu = np.broadcast_arrays(*[np.asarray(x, dtype=np.float64)
                                           for x in (rho_g, rho_d, nu)])
    if np.any(rho_g <= 0) or np.any(rho_d < 0) or np.any(nu < 0):
        raise ValueError("require rho_g>0 and rho_d,nu>=0")
    shapes = [ug.shape[1:], ud.shape[1:], rho_g.shape, np.shape(omega)]
    shapes += [np.shape(x)[1:] for x in (bhat, hyqmom, d0, d1, a0, a1) if x is not None]
    shapes += [np.shape(x)[2:] for x in (C, D0, D1) if x is not None]
    shape = np.broadcast_shapes(*shapes)

    def field(value, rank):
        value = np.asarray(value, dtype=np.float64)
        leading = value.shape[:rank]
        return np.broadcast_to(value.reshape((*leading, *([1]*(len(shape) - value.ndim + rank)),
                                              *value.shape[rank:])), (*leading, *shape))

    ug, ud = field(ug, 1), field(ud, 1)
    alpha = 1 + rho_d/rho_g
    mass, reduced = rho_g + rho_d, rho_d/alpha
    w0, u0 = ud - ug, (rho_g*ug + rho_d*ud)/mass
    if bhat is None:
        bhat = np.broadcast_to(np.array([0., 0., 1.]).reshape(3, *([1]*len(shape))), (3, *shape))
    else:
        bhat = field(bhat, 1)
    zero = np.zeros((3, *shape))
    d0, d1, a0, a1 = [zero if x is None else field(x, 1)
                       for x in (d0, d1, a0, a1)]
    Dzero = np.zeros((3, 3, *shape))
    D0, D1 = [Dzero if x is None else field(x, 2) for x in (D0, D1)]
    for diffusion in (D0, D0 + h*D1):
        if not np.all(np.isfinite(diffusion)) or not np.allclose(
                diffusion, np.swapaxes(diffusion, 0, 1), rtol=1e-12, atol=1e-14):
            raise ValueError("diffusivity must be finite and symmetric")
        eigenvalues = np.linalg.eigvalsh(np.moveaxis(diffusion, (0, 1), (-2, -1)))
        scale = np.max(np.abs(diffusion), axis=(0, 1))
        if np.any(eigenvalues[..., 0] < -1e-13*scale):
            raise ValueError("diffusivity must be positive semidefinite at both endpoints")
    def P(k, x):
        return _slip_phi(k, x, h, nu, omega, bhat, alpha)
    w = P(0, w0) + h*P(1, d0) + h*h*P(2, d1)
    iw = h*P(1, w0) + h*h*P(2, d0) + h**3*P(3, d1)
    itw = (h*h*(P(1, w0) - P(2, w0))
           + h**3*(P(2, d0) - P(3, d0))
           + h**4*(P(3, d1) - P(4, d1)))
    u = u0 + h*a0 + 0.5*h*h*a1
    iu = h*u0 + 0.5*h*h*a0 + h**3*a1/6
    itu = 0.5*h*h*u0 + h**3*a0/3 + h**4*a1/8
    frame, active = _frame(bhat, shape)
    wc, dc, ec = [np.einsum("ai...,i...->a...", frame, x) for x in (w0, d0, d1)]
    z = -alpha*h*(nu + 1j*np.asarray(omega)*active)
    square = (_mean_square(z, wc[0] + 1j*wc[1], h*(dc[0] + 1j*dc[1]),
                           h*h*(ec[0] + 1j*ec[1]))
              + _mean_square(z.real, wc[2], h*dc[2], h*h*ec[2]))
    Q_mean = rho_d*nu*h*square
    def trace(x):
        return sum(x[i, i] for i in range(3))
    work = (reduced*np.sum(d0*iw + d1*itw, axis=0)
            + mass*np.sum(a0*iu + a1*itu, axis=0))
    result = dict(ug=u - (rho_d/mass)*w, ud=u + (rho_g/mass)*w, w=w,
                  integrated_slip=iw, Q_mean=Q_mean, mechanical_work=work,
                  SGS_work=rho_d*(h*trace(D0) + 0.5*h*h*trace(D1)))
    if C is not None:
        C = field(C, 2)
        result["C"] = (covariance_phi(0, C, h, nu, omega, bhat)
                       + 2*h*covariance_phi(1, D0, h, nu, omega, bhat)
                       + 2*h*h*covariance_phi(2, D1, h, nu, omega, bhat))
        tr0 = trace(C)
    else:
        v, q, r, sperp = field(hyqmom, 1)
        b = frame[2]
        dp0, dp1 = [np.einsum("i...,ij...,j...->...", b, x, b) for x in (D0, D1)]
        dt0, dt1 = 0.5*(trace(D0) - dp0), 0.5*(trace(D1) - dp1)
        for diffusion, dp, dt in ((D0, dp0, dt0), (D1, dp1, dt1)):
            gyro = (dp - dt)*b[:, None]*b[None, :] + np.einsum("ij,...->ij...", np.eye(3), dt)
            if not np.allclose(diffusion, gyro, rtol=1e-12, atol=1e-14):
                raise ValueError("HyQMOM diffusion must be gyrotropic about bhat")
        s = np.exp(-nu*h)
        sigma = 2*h*phi(1, -2*nu*h).real*dp0 + 2*h*h*phi(2, -2*nu*h).real*dp1
        sigmap = 2*h*phi(1, -2*nu*h).real*dt0 + 2*h*h*phi(2, -2*nu*h).real*dt1
        result["hyqmom"] = np.stack(np.broadcast_arrays(s*s*v + sigma, s**3*q,
                                    s**4*r + 6*s*s*sigma*v + 3*sigma*sigma,
                                    s*s*sperp + sigmap))
        tr0 = v + 2*sperp
    zc = -2*nu*h
    random_parts = (h*phi(1, zc).real*tr0, 2*h*h*phi(2, zc).real*trace(D0),
                    2*h**3*phi(3, zc).real*trace(D1))
    result["Q_random"] = rho_d*nu*_nonnegative(sum(random_parts),
                                              sum(np.abs(x) for x in random_parts))
    return result
