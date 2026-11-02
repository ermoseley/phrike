"""Analytic exponential-polynomial paths on the normalized interval 0 <= theta <= 1.

A term is ``sum_p coefficient[p] theta**p exp(rate theta)``.  Physical
components precede cell axes.  Products retain the oscillating mean's variance;
no time sampling or quadrature is used by evaluation, integration or convolution.
"""
from __future__ import annotations

import numpy as np

from .coupled_exponential import _frame


def _moments(z, degree):
    """Integral_0^1 t**p exp(z*t) dt, including large imaginary z."""
    z = np.asarray(z, dtype=np.complex128)
    large = np.abs(z) > degree + 1
    safe = np.where(large, z, 1)
    forward = [np.expm1(z)/safe]
    exponential = np.exp(z)
    for p in range(1, degree + 1):
        forward.append((exponential - p*forward[-1])/safe)
    # Backward recurrence is stable where the forward recurrence is not.
    zz = np.where(large, 0, z)
    top = max(degree + 32, int(2*np.max(np.abs(zz), initial=0)) + 32)
    term = np.ones_like(z)/(top + 1)
    value = term.copy()
    for k in range(1, 96):
        term *= -zz/(top + k + 1)
        value += term
    expzz = np.exp(zz)
    value *= expzz
    backward = [None]*(degree + 1)
    for p in range(top, 0, -1):
        if p <= degree:
            backward[p] = value
        value = (expzz - zz*value)/p
    backward[0] = value
    return [np.where(large, f, b) for f, b in zip(forward, backward)]


class ExponentialPath:
    """Finite per-cell exponential-polynomial modes, with real public values.

    ``terms`` is an iterable of ``(rate, coefficient)`` pairs.  The first axis
    of each coefficient contains ascending polynomial powers.  ``rank`` counts
    physical component axes (0 for a scalar field, 1 for a vector, 2 for a tensor).
    """
    def __init__(self, terms, rank=0):
        self.rank = rank
        self.terms = []
        for rate, coefficient in terms:
            rate = np.asarray(rate, dtype=np.complex128)
            coefficient = np.asarray(coefficient, dtype=np.complex128)
            while len(coefficient) > 1 and not np.any(coefficient[-1]):
                coefficient = coefficient[:-1]
            if not np.any(coefficient):
                continue
            for i, (oldrate, old) in enumerate(self.terms):
                if oldrate.shape == rate.shape and np.array_equal(oldrate, rate):
                    size = max(len(old), len(coefficient))
                    shape = np.broadcast_shapes(old.shape[1:], coefficient.shape[1:])
                    merged = np.zeros((size, *shape), dtype=np.complex128)
                    merged[:len(old)] += old
                    merged[:len(coefficient)] += coefficient
                    self.terms[i] = (oldrate, merged)
                    break
            else:
                self.terms.append((rate, coefficient))
        if not self.terms:
            # Preserve the component/cell shape even for an identically zero path.
            self.terms = [(rate, np.zeros_like(coefficient[:1]))]

    @classmethod
    def polynomial(cls, coefficients, rank=0):
        return cls([(0., coefficients)], rank)

    @classmethod
    def constant(cls, value, rank=0):
        return cls.polynomial(np.asarray(value)[None], rank)

    def _value(self, theta, integral):
        result = 0
        for rate, coefficients in self.terms:
            if integral:
                moments = _moments(rate*theta, len(coefficients) - 1)
                value = sum(c*theta**(p + 1)*moments[p]
                            for p, c in enumerate(coefficients))
            else:
                value = np.zeros_like(coefficients[0])
                for c in coefficients[::-1]:
                    value = value*theta + c
                value = value*np.exp(rate*theta)
            result = result + value
        return result

    def evaluate(self, theta=1.):
        return self._value(theta, False).real

    eval = evaluate

    def integral(self, theta=1.):
        return self._value(theta, True).real

    def weighted_integral(self, coefficients, theta=1.):
        return self.multiply(self.polynomial(coefficients)).integral(theta)

    def component(self, *indices):
        return ExponentialPath([(r, c[(slice(None), *indices)])
                                for r, c in self.terms], self.rank - len(indices))

    def reshape(self, component_shape):
        """Reshape physical component axes, retaining polynomial and cell axes."""
        component_shape = tuple(component_shape)
        return ExponentialPath([(r, c.reshape((len(c), *component_shape,
                                               *c.shape[1 + self.rank:])))
                                for r, c in self.terms], len(component_shape))

    @classmethod
    def stack(cls, paths):
        """Stack equal-rank paths on a new leading physical component axis."""
        return _stack(paths, paths[0].rank + 1)

    def linear_map(self, matrix):
        """Map a vector with a component-first matrix, preserving its modes."""
        if self.rank != 1:
            raise ValueError('linear_map requires a vector path')
        return ExponentialPath([(r, np.einsum('ij...,pj...->pi...', matrix, c))
                                for r, c in self.terms], 1)

    def __add__(self, other):
        if not isinstance(other, ExponentialPath):
            other = self.constant(other, self.rank)
        if self.rank != other.rank:
            raise ValueError('path addition requires equal component ranks')
        return ExponentialPath([*self.terms, *other.terms], self.rank)

    __radd__ = __add__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def __mul__(self, other):
        if isinstance(other, ExponentialPath):
            return self.multiply(other)
        return ExponentialPath([(r, c*other) for r, c in self.terms], self.rank)

    __rmul__ = __mul__

    def multiply(self, other):
        """Pointwise product; select components first for tensor contractions."""
        if self.rank and other.rank and self.rank != other.rank:
            raise ValueError('select components before multiplying tensor paths')
        terms = []
        for a, ac in self.terms:
            for b, bc in other.terms:
                shape = np.broadcast_shapes(ac.shape[1:], bc.shape[1:])
                out = np.zeros((len(ac) + len(bc) - 1, *shape), complex)
                for i, x in enumerate(ac):
                    for j, y in enumerate(bc):
                        out[i + j] += x*y
                terms.append((a + b, out))
        return ExponentialPath(terms, max(self.rank, other.rank))

    def convolve(self, rate):
        """Return integral_0^theta exp(rate*(theta-s)) self(s) ds.

        Coincident rates use their exact polynomial limit.  Nearby rates and
        high powers use a convergent beta series instead of subtracting poles.
        """
        terms = []
        rate = np.asarray(rate, dtype=np.complex128)
        for source, coefficients in self.terms:
            cells = np.broadcast_shapes(np.shape(source), np.shape(rate),
                                        coefficients.shape[1 + self.rank:])
            leading = coefficients.shape[:1 + self.rank]
            padding = len(cells) - len(coefficients.shape[1 + self.rank:])
            coefficients = np.broadcast_to(
                coefficients.reshape((*leading, *((1,)*padding),
                                      *coefficients.shape[1 + self.rank:])),
                (*leading, *cells))
            delta = source - rate
            source_coeff = np.zeros_like(coefficients)
            target = np.zeros_like(coefficients[:1])
            near = []
            def add_near(power, value):
                while len(near) <= power:
                    near.append(np.zeros_like(coefficients[0]))
                near[power] += value
            for p, coefficient in enumerate(coefficients):
                if not np.any(coefficient):
                    continue
                close = np.abs(delta) < (p + 1)/2
                safe = np.where(close, 1, delta)
                if np.any(~close):
                    q = coefficient*np.where(close, 0, 1)/safe
                    for j in range(p, -1, -1):
                        source_coeff[j] += q
                        if j:
                            q *= -j/safe
                    target[0] -= q
                if np.any(close):
                    # Source-anchored beta series has ratio <= 1/2 and avoids
                    # large cancelling pole coefficients at high powers p.
                    factor = np.where(close, 1., 0.)/(p + 1)
                    add_near(p + 1, coefficient*factor)
                    for j in range(1, 65):
                        factor = factor*(-delta)/(p + j + 1)
                        add_near(p + j + 1, coefficient*factor)
                        if np.all(np.abs(factor) < 1e-22):
                            break
            terms.extend(((source, source_coeff), (rate, target)))
            if near:
                terms.append((source, np.stack(near)))
        return ExponentialPath(terms, self.rank)


def _stack(paths, rank):
    """Stack paths, merging rates before allocating full component arrays."""
    result = []
    for i, path in enumerate(paths):
        for rate, c in path.terms:
            for j, (oldrate, old) in enumerate(result):
                if oldrate.shape == rate.shape and np.array_equal(oldrate, rate):
                    if len(old) < len(c):
                        out = np.zeros((len(c), *old.shape[1:]), complex)
                        out[:len(old)] = old
                        result[j] = (oldrate, out)
                    else:
                        out = old
                    break
            else:
                out = np.zeros((len(c), len(paths), *c.shape[1:]), complex)
                result.append((rate, out))
            out[:len(c), i] += c
    return ExponentialPath(result, rank)


def _forcing(forcing, rank):
    return forcing if isinstance(forcing, ExponentialPath) else ExponentialPath.polynomial(forcing, rank)


def slip_path(w0, h, nu, omega, bhat, forcing=None, alpha=1):
    """Frozen slip path, forced by physical-time derivative polynomial/path."""
    w0 = np.asarray(w0, dtype=float)
    shape = np.broadcast_shapes(w0.shape[1:], np.shape(nu), np.shape(omega), np.shape(alpha), np.shape(bhat)[1:])
    frame, active = _frame(bhat, shape)
    wc = np.einsum('ai...,i...->a...', frame, w0)
    z = -h*np.asarray(alpha)*(np.asarray(nu) + 1j*np.asarray(omega)*active)
    transverse = ExponentialPath([(z, (wc[0] + 1j*wc[1])[None])])
    parallel = ExponentialPath([(z.real, wc[2][None])])
    if forcing is not None:
        fc = _forcing(forcing, 1).linear_map(frame)
        transverse = transverse + h*(fc.component(0) + 1j*fc.component(1)).convolve(z)
        parallel = parallel + h*fc.component(2).convolve(z.real)
    # Split a complex transverse path into real Cartesian components by adding
    # its conjugate; this makes later products retain the true mean variance.
    real = ExponentialPath([(r, c*.5) for r, c in transverse.terms]
                           + [(r.conjugate(), c.conjugate()*.5) for r, c in transverse.terms])
    imag = ExponentialPath([(r, c/(2j)) for r, c in transverse.terms]
                           + [(r.conjugate(), -c.conjugate()/(2j)) for r, c in transverse.terms])
    return _stack([real, imag, parallel], 1).linear_map(np.swapaxes(frame, 0, 1))


def covariance_path(C0, h, nu, omega, bhat, forcing=None):
    """Frozen covariance path, with arbitrary symmetric derivative forcing."""
    C0 = np.asarray(C0, dtype=float)
    shape = np.broadcast_shapes(C0.shape[2:], np.shape(nu), np.shape(omega), np.shape(bhat)[1:])
    frame, active = _frame(bhat, shape)
    c0 = np.einsum('ai...,ij...,bj...->ab...', frame, C0, frame)
    z = -2*h*np.asarray(nu)
    values = [c0[0, 0] + c0[1, 1], c0[2, 2],
              c0[0, 2] + 1j*c0[1, 2], c0[0, 0] - c0[1, 1] + 2j*c0[0, 1]]
    rates = [z, z, z - 1j*h*np.asarray(omega)*active, z - 2j*h*np.asarray(omega)*active]
    modes = [ExponentialPath([(r, v[None])]) for r, v in zip(rates, values)]
    if forcing is not None:
        fp = _forcing(forcing, 2)
        ft = ExponentialPath([(r, np.einsum('ai...,pij...,bj...->pab...', frame, co, frame)) for r, co in fp.terms], 2)
        fv = [ft.component(0, 0) + ft.component(1, 1), ft.component(2, 2),
              ft.component(0, 2) + 1j*ft.component(1, 2),
              ft.component(0, 0) - ft.component(1, 1) + 2j*ft.component(0, 1)]
        modes = [m + h*f.convolve(r) for m, f, r in zip(modes, fv, rates)]
    def real(p):
        return ExponentialPath([(r, c*.5) for r, c in p.terms] + [(r.conjugate(), c.conjugate()*.5) for r, c in p.terms])
    def imag(p):
        return ExponentialPath([(r, c/(2j)) for r, c in p.terms] + [(r.conjugate(), -c.conjugate()/(2j)) for r, c in p.terms])
    iso, par, cross, spin = modes
    rows = [[.5*(iso + real(spin)), .5*imag(spin), real(cross)],
            [.5*imag(spin), .5*(iso - real(spin)), imag(cross)],
            [real(cross), imag(cross), par]]
    # Merge scalar b-frame blocks before embedding them in full tensors.
    terms = []
    for i in range(3):
        for j in range(3):
            for rate, co in rows[i][j].terms:
                for k, (oldrate, old) in enumerate(terms):
                    if oldrate.shape == rate.shape and np.array_equal(oldrate, rate):
                        if len(old) < len(co):
                            out = np.zeros((len(co), *old.shape[1:]), complex)
                            out[:len(old)] = old
                            terms[k] = (oldrate, out)
                        else:
                            out = old
                        break
                else:
                    out = np.zeros((len(co), 3, 3, *co.shape[1:]), complex)
                    terms.append((rate, out))
                out[:len(co), i, j] += co
    return ExponentialPath([(rate, np.einsum('ai...,pab...,bj...->pij...', frame, co, frame))
                            for rate, co in terms], 2)
