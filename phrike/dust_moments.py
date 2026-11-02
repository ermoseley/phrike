"""Realizable raw dust moments and periodic finite-volume transport.

Transport uses kinetic flux-vector splitting over a quadrature of each closure
(Desjardins, Fox & Villedieu 2008; Vikas et al. 2011): the Gaussian closure is
represented by its six symmetric sigma points, the 1D HyQMOM closure by the
three-node Gauss quadrature of its moments.  Both reproduce the closure flux
exactly, and the update is a positive combination of quadrature nodes, so it is
realizable under CFL 1/2 without the Rusanov split-state test, whose
subtraction U - F/a amplifies round-off without bound in cold and light cells.
"""

from __future__ import annotations

import numpy as np

TOL = 64 * np.finfo(np.float64).eps
# Raw float64 moments cannot hold v = M2/M0 - m^2 at exactly zero: transport
# round-off random-walks it.  Realizability is tested to _CTOL relative to the
# cancelled terms, and variances below _COLD of them are numerically cold.
_CTOL = 1e-12
_COLD = 1e-10
# Reconstructed faces must lie strictly inside the cone.
_STRICT = 1e-3
# Cells lighter than _LIGHT times their neighbourhood maximum are reconstructed
# at first order: slopes there are differences of much larger neighbours.
_LIGHT = 1e-6
# HyQMOM cells this far inside the cone (a Gaussian has H4 = 2) reconstruct in
# primitive coordinates; closer to the boundary they reconstruct conserved moments.
_H4_INTERIOR = 0.5
_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float64), where=b != 0)


def _vacuum(U: np.ndarray) -> np.ndarray:
    scale = np.maximum(1.0, np.max(np.abs(U), axis=0))
    return np.all(np.abs(U) <= TOL * scale, axis=0)


def local_density_reference(rho: np.ndarray) -> np.ndarray:
    """Maximum density over each cell's nearest neighbours (periodic)."""
    ref = np.array(rho, dtype=np.float64, copy=True)
    for axis in range(rho.ndim):
        ref = np.maximum(ref, np.maximum(np.roll(ref, 1, axis), np.roll(ref, -1, axis)))
    return ref


class GaussianClosure:
    """Ten raw moments with a three-velocity Gaussian closure."""

    name = "gaussian"
    nfields = 10
    field_names = ("rho", "rho_ux", "rho_uy", "rho_uz", "Mxx", "Myy", "Mzz",
                   "Mxy", "Mxz", "Myz")

    def __init__(self, ndim: int):
        if ndim not in (1, 2, 3):
            raise ValueError("ndim must be 1, 2 or 3")
        self.ndim = ndim

    def primitives(self, U: np.ndarray) -> dict:
        U = np.asarray(U, dtype=np.float64)
        rho = U[0]
        u = _divide(U[1:4], rho)
        C = np.empty((3, 3, *rho.shape), dtype=np.float64)
        for k, (i, j) in enumerate(_PAIRS):
            C[i, j] = C[j, i] = _divide(U[4 + k], rho) - u[i] * u[j]
        return {"rho": rho, "u": u, "C": C}

    def from_primitives(self, rho: np.ndarray, u: np.ndarray, C: np.ndarray) -> np.ndarray:
        rho = np.asarray(rho, dtype=np.float64)
        u, C = np.asarray(u, dtype=np.float64), np.asarray(C, dtype=np.float64)
        U = np.empty((10, *rho.shape), dtype=np.float64)
        U[0], U[1:4] = rho, rho * u
        for k, (i, j) in enumerate(_PAIRS):
            U[4 + k] = rho * (u[i] * u[j] + C[i, j])
        return U

    def flux(self, U: np.ndarray, d: int) -> np.ndarray:
        p = self.primitives(U)
        rho, u, C = p["rho"], p["u"], p["C"]
        F = np.empty_like(U, dtype=np.float64)
        F[0] = U[1 + d]
        F[1:4] = rho * (u[d] * u + C[d])
        for k, (i, j) in enumerate(_PAIRS):
            F[4 + k] = rho * (u[d] * u[i] * u[j] + u[d] * C[i, j]
                              + u[i] * C[d, j] + u[j] * C[d, i])
        return np.where(rho == 0, 0.0, F)

    def nodes(self, U: np.ndarray, d: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Sigma points u ± 3^{1/2} L_k with weights 1/6: weights (6, ...), velocities (6, 3, ...).

        L is the semi-definite Cholesky factor of C pivoted so that axis d comes
        first.  Then only the first pair moves along d, and the d-velocities
        form the three-point Gauss-Hermite rule u_d ± (3 C_dd)^{1/2} (1/6 each),
        u_d (2/3).  Any square root reproduces the Gaussian moments through
        third order, so the kinetic flux equals the closure flux exactly.
        """
        p = self.primitives(U)
        u, C = p["u"], p["C"]
        a, b, c = d, (d + 1) % 3, (d + 2) % 3
        l11 = np.sqrt(np.maximum(C[a, a], 0.0))
        l21, l31 = _divide(C[b, a], l11), _divide(C[c, a], l11)
        l22 = np.sqrt(np.maximum(C[b, b] - l21*l21, 0.0))
        l32 = _divide(C[c, b] - l31*l21, l22)
        l33 = np.sqrt(np.maximum(C[c, c] - l31*l31 - l32*l32, 0.0))
        zero = np.zeros_like(l11)
        L = np.empty((3, 3, *l11.shape))  # L[component, column]
        L[a] = np.stack([l11, zero, zero])
        L[b] = np.stack([l21, l22, zero])
        L[c] = np.stack([l31, l32, l33])
        offsets = np.sqrt(3.0) * L
        vel = np.concatenate([u[:, None] + offsets, u[:, None] - offsets], axis=1)
        weights = np.full((6, *l11.shape), 1.0 / 6.0)
        return weights, np.moveaxis(vel, 1, 0)

    def split_flux(self, U: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kinetic fluxes carried by right- and left-moving nodes, and the node speed."""
        rho = np.asarray(U, dtype=np.float64)[0]
        w, vel = self.nodes(U, d)
        vd = vel[:, d]
        mass = rho * w * vd  # (6, ...)
        phi = np.empty((10, *vel.shape[:1], *rho.shape))
        phi[0] = 1.0
        phi[1:4] = np.moveaxis(vel, 1, 0)
        for k, (i, j) in enumerate(_PAIRS):
            phi[4 + k] = vel[:, i] * vel[:, j]
        contrib = mass * phi
        plus = np.sum(np.where(vd > 0, contrib, 0.0), axis=1)
        minus = np.sum(np.where(vd < 0, contrib, 0.0), axis=1)
        speed = np.where(rho > 0, np.max(np.abs(vd), axis=0), 0.0)
        return plus, minus, speed

    def signal_speed(self, U: np.ndarray, d: int) -> np.ndarray:
        p = self.primitives(U)
        return np.where(p["rho"] > 0, np.abs(p["u"][d])
                        + np.sqrt(3 * np.maximum(p["C"][d, d], 0.0)), 0.0)

    def velocity_range(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Componentwise bound u_i ± (3 C_ii)^{1/2} on the node velocities, shape (3, ...)."""
        p = self.primitives(U)
        half = np.sqrt(3 * np.maximum(np.einsum("ii...->i...", p["C"]), 0.0))
        return p["u"] - half, p["u"] + half

    def realizable(self, U: np.ndarray, strict: bool = False) -> np.ndarray:
        p = self.primitives(U)
        C, u = p["C"], p["u"]
        tol = _CTOL * (_STRICT if strict else 1.0)
        scale = np.sum(u * u, axis=0) + np.sum(np.abs(C), axis=(0, 1))
        # Eigenvalue test: minors of a near-singular C carry round-off ~ε that
        # no additive tolerance on the determinant can absorb.
        finite = np.all(np.isfinite(C), axis=(0, 1))
        lam_min = np.linalg.eigvalsh(np.moveaxis(np.where(finite, C, 0.0), (0, 1), (-2, -1)))[..., 0]
        valid = finite & (lam_min >= -tol * scale)
        return (np.all(np.isfinite(U), axis=0) & (U[0] >= 0)
                & np.where(U[0] == 0, _vacuum(U), valid))

    def needs_projection(self, U: np.ndarray) -> np.ndarray:
        return ~self.realizable(U, strict=True)

    def repaired(self, U: np.ndarray) -> np.ndarray:
        """Clip negative covariance eigenvalues (round-off repair)."""
        p = self.primitives(U)
        lam, vec = np.linalg.eigh(np.moveaxis(p["C"], (0, 1), (-2, -1)))
        C = np.einsum("...ij,...j,...kj->...ik", vec, np.maximum(lam, 0.0), vec)
        return self.from_primitives(p["rho"], p["u"], np.moveaxis(C, (-2, -1), (0, 1)))

    def reconstruct(self, U: np.ndarray, axis: int, first: np.ndarray) -> tuple:
        """Minmod faces in (rho, u, C), the covariance slope scaled until faces are PSD.

        Conserved raw-moment slopes cannot give a cold cell a velocity slope:
        its M2 = rho u u has no room for sub-cell spread, so a realizable
        conserved reconstruction flattens velocity there and cold dust becomes
        first order, with an O(1) error at velocity stagnation points.  In
        primitive variables faces are realizable by construction (rho >= 0, C
        PSD); the cell average is no longer the mean of its faces, so the
        update's realizability is guaranteed a posteriori (MOOD).
        """
        p = self.primitives(U)
        rho, u, C = p["rho"], p["u"], p["C"]
        flat = first | (rho < _LIGHT * local_density_reference(rho)) | (rho <= 0)

        def limited(a):
            return np.where(flat, 0.0, _minmod(a - np.roll(a, 1, axis), np.roll(a, -1, axis) - a))

        s_rho, s_u = limited(rho), limited(u)
        s_C = limited(C)
        theta = np.ones_like(rho)
        for _ in range(31):
            ok = np.ones_like(rho, dtype=bool)
            for sign in (1.0, -1.0):
                face = C + 0.5*sign*theta*s_C
                lam = np.linalg.eigvalsh(np.moveaxis(face, (0, 1), (-2, -1)))[..., 0]
                scale = np.sum(np.abs(C), axis=(0, 1)) + np.sum(u*u, axis=0)
                ok &= lam >= -_CTOL * _STRICT * scale
            if np.all(ok):
                break
            theta = np.where(ok, theta, 0.5*theta)
        theta = np.where(ok, theta, 0.0)

        def build(sign: float) -> np.ndarray:
            r_ = rho + 0.5*sign*s_rho
            face = self.from_primitives(r_, u + 0.5*sign*s_u, C + 0.5*sign*theta*s_C)
            return np.where(r_ > 0, face, 0.0)

        # Warm or rough cells keep the conserved Zhang-Shu faces, whose update
        # is realizable by convexity and conserves every moment exactly.
        use = subcell_eligible(U, self)
        # Conserved faces are discarded in primitive cells; do not limit them.
        cons_plus, cons_minus, cons_scale = _reconstruct_conserved(U, self, axis, first | use)
        pick = lambda a, b: np.where(use, a, b)
        return (pick(build(1.0), cons_plus), pick(build(-1.0), cons_minus),
                pick(np.where(flat, 0.0, 1.0), cons_scale))

    def subcell_project(self, U: np.ndarray, budget: np.ndarray) -> tuple:
        """Clip covariance eigenvalues whose negativity is below the sub-cell budget."""
        p = self.primitives(U)
        lam = np.linalg.eigvalsh(np.moveaxis(p["C"], (0, 1), (-2, -1)))[..., 0]
        ok = (-lam <= budget) | (p["rho"] <= 0)
        out = U.copy()
        fix = ok & ~self.realizable(U)
        if np.any(fix):
            out[:, fix] = self.repaired(U[:, fix])
        return out, ok

    def kinetic_energy(self, U: np.ndarray) -> np.ndarray:
        return 0.5 * np.sum(U[4:7], axis=0)

    def mean_kinetic_energy(self, U: np.ndarray) -> np.ndarray:
        return 0.5 * _divide(np.sum(U[1:4] ** 2, axis=0), U[0])

    def random_energy(self, U: np.ndarray) -> np.ndarray:
        return self.kinetic_energy(U) - self.mean_kinetic_energy(U)


class HyqmomClosure:
    """Fourth-order 1D raw hierarchy with isotropic perpendicular dispersion."""

    name = "hyqmom"
    ndim = 1
    nfields = 8
    field_names = ("M0", "M1", "M2", "M3", "M4", "rho_uy", "rho_uz", "Pperp")

    def __init__(self, ndim: int = 1):
        if ndim != 1:
            raise ValueError("HyQMOM requires ndim=1")

    def _central(self, U: np.ndarray) -> tuple:
        rho = U[0]
        u = _divide(U[[1, 5, 6]], rho)
        m = u[0]
        s2, s3, s4 = _divide(U[2:5], rho)
        v = s2 - m*m
        q = s3 - 3*m*s2 + 2*m**3
        r = s4 - 4*m*s3 + 6*m*m*s2 - 3*m**4
        sperp2 = 0.5 * (_divide(U[7], rho) - u[1]**2 - u[2]**2)
        # Magnitudes of the cancelled terms bound the round-off in v, q, r.
        am = np.abs(m)
        scales = (np.abs(s2) + m*m,
                  np.abs(s3) + 3*am*np.abs(s2) + 2*am**3,
                  np.abs(s4) + 4*am*np.abs(s3) + 6*m*m*np.abs(s2) + 3*m**4,
                  0.5*(np.abs(_divide(U[7], rho)) + u[1]**2 + u[2]**2))
        return rho, u, v, q, r, sperp2, scales

    def primitives(self, U: np.ndarray) -> dict:
        rho, u, v, q, r, sperp2, scales = self._central(np.asarray(U, dtype=np.float64))
        cold = v <= _COLD * scales[0]
        return {"rho": rho, "u": u, "v": v, "q": np.where(cold, 0, q),
                "r": np.where(cold, 0, r), "sperp2": sperp2, "cold": cold}

    def from_primitives(self, rho: np.ndarray, u: np.ndarray, v: np.ndarray,
                        q: np.ndarray, r: np.ndarray, sperp2: np.ndarray,
                        cold: np.ndarray | None = None) -> np.ndarray:
        rho, u = np.asarray(rho, dtype=np.float64), np.asarray(u, dtype=np.float64)
        m = u[0]
        U = np.empty((8, *rho.shape), dtype=np.float64)
        U[0], U[1] = rho, rho*m
        U[2] = rho * (m*m + v)
        U[3] = rho * (m**3 + 3*m*v + q)
        U[4] = rho * (m**4 + 6*m*m*v + 4*m*q + r)
        U[5:7] = rho * u[1:3]
        U[7] = rho * (u[1]**2 + u[2]**2 + 2*sperp2)
        return U

    def flux(self, U: np.ndarray, d: int) -> np.ndarray:
        p = self.primitives(U)
        m, v, q, r = p["u"][0], p["v"], p["q"], p["r"]
        inv_v = _divide(np.ones_like(v), np.where(p["cold"], 0, v))
        c5 = 0.5*q*(5*r*inv_v - 3*q*q*inv_v**2 - v)
        F = np.empty_like(U, dtype=np.float64)
        F[:4] = U[1:5]
        F[4] = p["rho"] * (m**5 + 10*m**3*v + 10*m*m*q + 5*m*r + c5)
        # Preserve raw-moment proportionality in the round-off cold limit.
        F[4] = np.where(p["cold"], m*U[4], F[4])
        F[5:8] = m * U[5:8]
        return np.where(p["rho"] == 0, 0.0, F)

    def nodes(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Three-node Gauss quadrature of the HyQMOM moments: weights and x-velocities.

        The n = 2 closure S5 = S3(5S4 - 3S3^2 - 1)/2 is exactly the third
        recurrence coefficient a2 = (a0 + a1)/2, so the Jacobi matrix of the
        standardised moments is [[0, 1, 0], [1, S3, H4^{1/2}], [0, H4^{1/2}, S3/2]].
        """
        p = self.primitives(U)
        rho, m, v, q, r, cold = p["rho"], p["u"][0], p["v"], p["q"], p["r"], p["cold"]
        vv = np.where(cold, 1.0, np.maximum(v, 0.0))
        s3 = np.where(cold, 0.0, q / vv**1.5)
        h4 = np.where(cold, 0.0, np.maximum(r / (vv*vv) - s3*s3 - 1, 0.0))
        J = np.zeros((*rho.shape, 3, 3))
        J[..., 0, 1] = J[..., 1, 0] = 1.0
        J[..., 1, 1] = s3
        J[..., 1, 2] = J[..., 2, 1] = np.sqrt(h4)
        J[..., 2, 2] = 0.5 * s3
        xi, vec = np.linalg.eigh(J)
        w = np.moveaxis(vec[..., 0, :]**2, -1, 0)
        ux = m + np.sqrt(np.where(cold, 0.0, vv)) * np.moveaxis(xi, -1, 0)
        w = np.where(cold, np.array([1.0, 0.0, 0.0]).reshape(3, *[1]*rho.ndim), w)
        return w, ux

    def split_flux(self, U: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Kinetic fluxes carried by right- and left-moving nodes, and the node speed."""
        U = np.asarray(U, dtype=np.float64)
        rho = U[0]
        w, ux = self.nodes(U)
        mass = rho * w * ux  # (3, ...)
        perp = _divide(U[5:8], rho)  # independent of v_x
        phi = np.stack([np.ones_like(ux), ux, ux**2, ux**3, ux**4,
                        *np.broadcast_to(perp[:, None], (3, *ux.shape))])
        contrib = mass * phi
        plus = np.sum(np.where(ux > 0, contrib, 0.0), axis=1)
        minus = np.sum(np.where(ux < 0, contrib, 0.0), axis=1)
        speed = np.where(rho > 0, np.max(np.where(w > 0, np.abs(ux), 0.0), axis=0), 0.0)
        return plus, minus, speed

    def signal_speed(self, U: np.ndarray, d: int) -> np.ndarray:
        return self.split_flux(U, d)[2]

    def velocity_range(self, U: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Min and max of the x node velocities (weighted nodes only), shape (1, ...)."""
        w, ux = self.nodes(U)
        live = w > 0
        return (np.min(np.where(live, ux, np.inf), axis=0)[None],
                np.max(np.where(live, ux, -np.inf), axis=0)[None])

    def reconstruct(self, U: np.ndarray, axis: int, first: np.ndarray) -> tuple:
        """Hybrid faces: primitive space in the cone interior, conserved near its boundary.

        Interior cells (not cold, H4 >= _H4_INTERIOR, with both neighbours)
        use minmod in (rho, m, v, S3, H4, u_y, u_z, sigma_perp^2), where
        realizability is the box v, H4, sigma_perp^2 >= 0 and minmod keeps faces
        inside their neighbours' range.  Raw components would not do there: q
        and r are small differences of O(m^3) and O(m^4) raw terms, and limiting
        raw moments separately makes the face skewness and kurtosis noisy.
        Cold cells reconstruct density and velocity in primitive space and
        stay cold.  Two-point mixtures near the cone boundary keep the conserved
        Zhang-Shu reconstruction, whose faces are convex combinations.  The
        update's realizability is then guaranteed a posteriori (MOOD).
        """
        cons_plus, cons_minus, cons_scale = _reconstruct_conserved(U, self, axis, first)
        prim_plus, prim_minus, prim_scale = self._reconstruct_primitive(U, axis, first)
        p = self.primitives(U)
        vv = np.where(p["cold"], 1.0, np.maximum(p["v"], 0.0))
        s3 = np.where(p["cold"], 0.0, p["q"] / vv**1.5)
        h4 = np.where(p["cold"], -1.0, p["r"] / (vv*vv) - s3*s3 - 1)
        interior = (h4 >= _H4_INTERIOR) & (p["rho"] > 0)
        interior &= np.roll(interior, 1, axis) & np.roll(interior, -1, axis)
        # Cold-like cells (variance below the unresolved sub-cell velocity
        # variance) also use primitive faces, with density and velocity slopes.
        interior |= (p["cold"] | subcell_eligible(U, self)) & (p["rho"] > 0)
        pick = lambda a, b: np.where(interior, a, b)
        return pick(prim_plus, cons_plus), pick(prim_minus, cons_minus), pick(prim_scale, cons_scale)

    @staticmethod
    def _coldlike(p: dict, axis: int) -> np.ndarray:
        m = p["u"][0]
        budget = np.maximum((m - np.roll(m, 1, axis))**2, (np.roll(m, -1, axis) - m)**2)
        return p["cold"] | (p["v"] <= budget)

    def _reconstruct_primitive(self, U: np.ndarray, axis: int, first: np.ndarray) -> tuple:
        p = self.primitives(U)
        rho, u = p["rho"], p["u"]
        cold = p["cold"] | subcell_eligible(U, self)
        v = np.where(cold, 0.0, np.maximum(p["v"], 0.0))
        vv = np.where(cold, 1.0, v)
        s3 = np.where(cold, 0.0, p["q"] / vv**1.5)
        h4 = np.where(cold, 0.0, np.maximum(p["r"] / (vv*vv) - s3*s3 - 1, 0.0))
        prim = np.stack([rho, u[0], v, s3, h4, u[1], u[2], np.maximum(p["sperp2"], 0.0)])
        slope = _minmod(prim - np.roll(prim, 1, axis), np.roll(prim, -1, axis) - prim)
        light = rho < _LIGHT * local_density_reference(rho)
        flat = first | light
        slope = np.where(flat, 0.0, slope)
        # A cold cell keeps density and velocity slopes but stays cold: its
        # faces are Diracs at the face velocity.  (Conserved-moment slopes
        # cannot give a cold cell a velocity slope without leaving the cone.)
        slope[[2, 3, 4, 7]] = np.where(cold, 0.0, slope[[2, 3, 4, 7]])

        def build(P: np.ndarray) -> np.ndarray:
            r_, m, v_, s3_, h4_, uy, uz, sp = P
            face = self.from_primitives(r_, np.stack([m, uy, uz]), v_, s3_*v_**1.5,
                                        (h4_ + s3_*s3_ + 1)*v_*v_, sp)
            return np.where(r_ > 0, face, 0.0)

        return build(prim + 0.5*slope), build(prim - 0.5*slope), np.where(flat, 0.0, 1.0)

    def realizable(self, U: np.ndarray, strict: bool = False) -> np.ndarray:
        rho, u, v, q, r, sperp2, (sv, sq, sr, sp) = self._central(
            np.asarray(U, dtype=np.float64))
        f = _STRICT if strict else 1.0
        cold = v <= _COLD * sv
        vv = np.maximum(v, _COLD * sv)
        K = r - v*v - _divide(q*q, vv)
        # Linearised round-off of K given the cancellation scales of v, q, r.
        tol_K = f * _CTOL * (sr + 2*np.abs(v)*sv + _divide(2*np.abs(q)*sq, vv)
                             + _divide(q*q*sv, vv*vv))
        cold_ok = (np.abs(q) <= _COLD * sq) & (np.abs(r) <= _COLD * sr)
        valid = ((v >= -f * _COLD * sv)
                 & np.where(cold, cold_ok, K >= -tol_K)
                 & (sperp2 >= -f * _CTOL * sp))
        return (np.all(np.isfinite(U), axis=0) & (rho >= 0)
                & np.where(rho == 0, _vacuum(U), valid))

    def needs_projection(self, U: np.ndarray) -> np.ndarray:
        """Strict-cone failures, and numerically cold cells, whose v, q, r are
        round-off that would otherwise accumulate across substeps."""
        rho, u, v, q, r, sperp2, (sv, sq, sr, sp) = self._central(np.asarray(U, dtype=np.float64))
        cold = (v <= _COLD * sv) & ((v != 0) | (q != 0) | (r != 0))
        return cold | ~self.realizable(U, strict=True)

    def repaired(self, U: np.ndarray) -> np.ndarray:
        """Nearest cone state (round-off repair); numerically cold cells become cold."""
        rho, u, v, q, r, sperp2, (sv, sq, sr, sp) = self._central(U)
        cold = v <= _COLD * sv
        v = np.where(cold, 0.0, np.maximum(v, 0.0))
        q = np.where(cold, 0.0, q)
        r = np.where(cold, 0.0, np.maximum(r, v*v + _divide(q*q, v)))
        return self.from_primitives(rho, u, v, q, r, np.maximum(sperp2, 0.0))

    def subcell_project(self, U: np.ndarray, budget: np.ndarray) -> tuple:
        """Make cells cold whose variance is below the sub-cell budget."""
        rho, u, v, q, r, sperp2, scales = self._central(np.asarray(U, dtype=np.float64))
        ok = (v <= budget) | (rho <= 0)
        out = U.copy()
        fix = ok & (rho > 0) & ~self.realizable(U)
        if np.any(fix):
            z = np.zeros(int(np.count_nonzero(fix)))
            out[:, fix] = self.from_primitives(rho[fix], u[:, fix], z, z, z,
                                               np.maximum(sperp2[fix], 0.0))
        return out, ok

    def kinetic_energy(self, U: np.ndarray) -> np.ndarray:
        return 0.5 * (U[2] + U[7])

    def mean_kinetic_energy(self, U: np.ndarray) -> np.ndarray:
        return 0.5 * _divide(np.sum(U[[1, 5, 6]]**2, axis=0), U[0])

    def random_energy(self, U: np.ndarray) -> np.ndarray:
        return self.kinetic_energy(U) - self.mean_kinetic_energy(U)


Closure = GaussianClosure | HyqmomClosure


def regularize(U: np.ndarray, closure: Closure) -> tuple[np.ndarray, dict]:
    """Project states outside the exact cone back onto it; mass and momentum are kept.

    States that pass only within the round-off tolerance cannot absorb any
    outward increment, so every substep starts on or inside the cone.
    ``repaired_cells`` counts states that failed the tolerance test itself;
    ``roundoff_projections`` the rest: states that passed it but not the strict
    test, and HyQMOM cells re-projected to exactly cold (v < _COLD of its scale).
    Returns the per-cell change of dust kinetic energy density, which the
    caller must return to the gas to conserve total energy.
    """
    U = np.array(U, dtype=np.float64, copy=True)
    live = U[0] > 0
    failed = live & ~closure.realizable(U)
    project = live & closure.needs_projection(U)
    E0 = closure.kinetic_energy(U)
    if np.any(project):
        U[:, project] = closure.repaired(U[:, project])
    return U, dict(repaired_cells=int(np.count_nonzero(failed)),
                   roundoff_projections=int(np.count_nonzero(project & ~failed)),
                   dE=closure.kinetic_energy(U) - E0)


def _minmod(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.where(np.signbit(a) == np.signbit(b),
                    np.copysign(np.minimum(np.abs(a), np.abs(b)), a), 0.0)


def _neighbourhood(a: np.ndarray, axis: int, reduce) -> np.ndarray:
    return reduce(reduce(a, np.roll(a, 1, axis)), np.roll(a, -1, axis))


def _reconstruct(U: np.ndarray, closure: Closure, axis: int,
                 first: np.ndarray | None = None) -> tuple:
    """Limited linear faces of the conserved moments.

    The slope is scaled by powers of 1/2 until both faces are strictly
    realizable and their node velocities lie within the range spanned by the
    neighbouring cells' nodes plus half that span (a relaxed kinetic maximum
    principle), so a light extrapolated face cannot carry spuriously fast nodes.  Scaling toward the
    cell keeps the cell average equal to the mean of its faces.
    """
    if first is None:
        first = np.zeros(U.shape[1:], dtype=bool)
    if hasattr(closure, "reconstruct"):
        return closure.reconstruct(U, axis, first)
    return _reconstruct_conserved(U, closure, axis, first)


def _reconstruct_conserved(U: np.ndarray, closure: Closure, axis: int,
                           first: np.ndarray) -> tuple:
    slope = _minmod(U - np.roll(U, 1, axis), np.roll(U, -1, axis) - U)
    light = (U[0] < _LIGHT * local_density_reference(U[0])) | first
    vmin, vmax = closure.velocity_range(U)
    vmin = np.where(U[0] > 0, vmin, np.inf)
    vmax = np.where(U[0] > 0, vmax, -np.inf)
    lo, hi = _neighbourhood(vmin, axis, np.minimum), _neighbourhood(vmax, axis, np.maximum)
    # Half the neighbourhood span of slack keeps smooth extrema second order;
    # the bound only excludes gross outliers from extrapolated light faces.
    finite = np.isfinite(lo) & np.isfinite(hi)
    span = np.where(finite, 0.5*(hi - lo) + TOL*(np.abs(hi) + np.abs(lo)), 0.0)
    scale = np.where(light, 0.0, 1.0)

    def bounded(face: np.ndarray, live: np.ndarray) -> np.ndarray:
        fmin, fmax = closure.velocity_range(face)
        ok = (fmin >= (lo - span)[:, live]) & (fmax <= (hi + span)[:, live])
        return np.all(ok | (face[0] <= 0), axis=0)

    valid = np.ones(U.shape[1:], dtype=bool)
    for _ in range(31):
        live = scale != 0
        if not np.any(live):
            break
        minus = (U - 0.5*slope*scale)[:, live]
        plus = (U + 0.5*slope*scale)[:, live]
        # Zero-slope cells are already final; test only the active faces.
        valid[:] = True
        valid[live] = (closure.realizable(minus, strict=True) & closure.realizable(plus, strict=True)
                       & ((minus[0] > 0) | np.all(minus == 0, axis=0))
                       & ((plus[0] > 0) | np.all(plus == 0, axis=0))
                       & bounded(minus, live) & bounded(plus, live))
        if np.all(valid):
            break
        scale = np.where(valid, scale, scale * 0.5)
    scale = np.where(valid, scale, 0.0)
    return U + 0.5*slope*scale, U - 0.5*slope*scale, scale


def _face_flux(U: np.ndarray, closure: Closure, d: int,
               first: np.ndarray | None = None) -> tuple:
    left, right, scale = _reconstruct(U, closure, -1-d, first)
    plus, _, speed_left = closure.split_flux(left, d)
    _, minus, speed_right = closure.split_flux(right, d)
    flux = plus + np.roll(minus, -1, axis=-1-d)
    speed = float(max(np.max(speed_left), np.max(speed_right)))
    return flux, speed, 0, scale


def _transport_rhs(U: np.ndarray, closure: Closure, dxs: tuple,
                   first: np.ndarray | None = None) -> tuple:
    rhs = np.zeros_like(U)
    rate, doubling, slope_min = 0.0, 0, 1.0
    first_order = np.zeros(U.shape[1:], dtype=bool)
    for d, dx in enumerate(dxs):
        flux, speed, ndouble, scale = _face_flux(U, closure, d, first)
        rhs -= (flux - np.roll(flux, 1, axis=-1-d)) / dx
        rate += speed / dx
        doubling = max(doubling, ndouble)
        slope_min = min(slope_min, float(np.min(scale)))
        first_order |= scale == 0
    return rhs, rate, doubling, slope_min, int(np.count_nonzero(first_order))


_MOOD_PASSES = 8


def _accept(state: np.ndarray, closure: Closure, budget: np.ndarray,
            eligible: np.ndarray) -> tuple:
    """Project sub-cell-level cone violations in eligible cells.

    Returns (state, count, dE, unresolved mask).
    """
    bad = ~closure.realizable(state)
    if not np.any(bad):
        return state, 0, np.zeros(state.shape[1:]), bad
    projected, ok = closure.subcell_project(state, budget)
    ok &= eligible
    projected = np.where(ok[None], projected, state)
    fixed = bad & ok
    dE = np.where(fixed, closure.kinetic_energy(projected) - closure.kinetic_energy(state), 0.0)
    return projected, int(np.count_nonzero(fixed)), dE, bad & ~ok


def _subcell_budget(U: np.ndarray, closure: Closure) -> np.ndarray:
    """Unresolved sub-cell velocity variance (1/12) sum_d min(|Δ₋u|², |Δ₊u|²).

    The exact cell average of a smooth cold flow is not cold: a linear velocity
    profile across a cell carries variance |Δu|²/12.  A cone violation below
    it is unresolved by the grid.  The minimum over the two neighbours makes
    the budget vanish at discontinuities and extrema.
    """
    u = closure.primitives(U)["u"]
    budget = np.zeros(U.shape[1:])
    for d in range(closure.ndim):
        axis = -1 - d
        left = np.sum((u - np.roll(u, 1, axis))**2, axis=0)
        right = np.sum((np.roll(u, -1, axis) - u)**2, axis=0)
        budget += np.minimum(left, right)/12.0
    return budget


def subcell_eligible(U: np.ndarray, closure: Closure) -> np.ndarray:
    """Cold-like cells in smooth flow: total variance below the sub-cell budget
    times a safety factor, and one-sided velocity differences within a factor 2
    along every resolved axis.  Only these use primitive (rho, u) faces and
    may have sub-cell-level cone violations projected."""
    p = closure.primitives(U)
    u = p["u"]
    budget = _subcell_budget(U, closure)
    var = (np.einsum("ii...->...", p["C"]) if "C" in p
           else np.maximum(p["v"], 0.0) + 2*np.maximum(p["sperp2"], 0.0))
    ok = (var <= 12.0*budget) & (p["rho"] > 0)
    for d in range(closure.ndim):
        axis = -1 - d
        left = np.sum((u - np.roll(u, 1, axis))**2, axis=0)
        right = np.sum((np.roll(u, -1, axis) - u)**2, axis=0)
        ok &= np.minimum(left, right) >= 0.25*np.maximum(left, right)
    return ok


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for axis in range(mask.ndim):
        out |= np.roll(mask, 1, axis) | np.roll(mask, -1, axis)
    return out


def _grid(dxs: tuple, ndim: int, dt: float) -> tuple:
    dxs = tuple(float(dx) for dx in dxs)
    if len(dxs) != ndim or any(dx <= 0 or not np.isfinite(dx) for dx in dxs):
        raise ValueError("dxs must contain one positive spacing per resolved direction")
    if dt < 0 or not np.isfinite(dt):
        raise ValueError("time interval must be finite and non-negative")
    return dxs


def max_stable_dt(U: np.ndarray, closure: Closure, dxs: tuple, cfl: float = 0.8) -> float:
    """Estimate the kinetic CFL bound sum_d (dt/dx_d) max|v_d| <= cfl/2 from cell nodes.

    Reconstructed faces may carry slightly faster nodes; ``transport`` checks
    the exact face bound and sub-cycles, so this is the coupled step's hint.
    """
    dxs = _grid(dxs, closure.ndim, 0.0)
    if not 0 < cfl <= 1:
        raise ValueError("cfl must lie in (0, 1]")
    if not np.all(closure.realizable(U)):
        raise ValueError("initial dust state is not realizable")
    U = np.asarray(U, dtype=np.float64)
    rate = sum(float(np.max(closure.signal_speed(U, d)))/dx for d, dx in enumerate(dxs))
    return 0.5*cfl/rate if rate > 0 else np.inf


def transport(U: np.ndarray, closure: Closure, dt: float, dxs: tuple,
              cfl: float = 0.8) -> tuple[np.ndarray, dict]:
    """Advance raw moments with realizable MUSCL, kinetic fluxes and SSP-RK2.

    Diagnostics include attempted stages; first_order_cells counts cell/stage
    occurrences with zero slope scale in at least one direction.
    """
    dxs = _grid(dxs, closure.ndim, dt)
    if not 0 < cfl <= 1:
        raise ValueError("cfl must lie in (0, 1]")
    U = np.array(U, dtype=np.float64, copy=True)
    if not np.all(closure.realizable(U)):
        raise ValueError("initial dust state is not realizable")
    stats = dict(substeps=0, halvings=0, max_doublings=0, min_slope_scale=1.0,
                 first_order_cells=0, repaired_cells=0, roundoff_projections=0,
                 mood_cells=0, subcell_projections=0,
                 regularization_dE=np.zeros(U.shape[1:]))
    remaining = dt
    while remaining > 0:
        U, reg = regularize(U, closure)
        stats["repaired_cells"] += reg["repaired_cells"]
        stats["roundoff_projections"] += reg["roundoff_projections"]
        stats["regularization_dE"] += reg["dE"]
        first = np.zeros(U.shape[1:], dtype=bool)
        budget = _subcell_budget(U, closure)
        eligible = subcell_eligible(U, closure)
        rhs, rate, nd, sm, nf = _transport_rhs(U, closure, dxs, first)
        stats["min_slope_scale"] = min(stats["min_slope_scale"], sm)
        stats["first_order_cells"] += nf
        h = min(remaining, 0.5*cfl/rate if rate > 0 else remaining)
        for attempt in range(31 + _MOOD_PASSES):
            # A posteriori (MOOD) limiting: cells whose update leaves the cone
            # are recomputed at first order together with their neighbours;
            # the all-first-order kinetic update is realizable under the CFL.
            U1, n1, dE1, bad = _accept(U + h*rhs, closure, budget, eligible)
            if not np.any(bad):
                rhs1, rate1, nd1, sm1, nf1 = _transport_rhs(U1, closure, dxs, first)
                stats["min_slope_scale"] = min(stats["min_slope_scale"], sm1)
                stats["first_order_cells"] += nf1
                candidate, n2, dE2, bad = _accept(0.5*U + 0.5*(U1 + h*rhs1), closure, budget, eligible)
                if h*rate1 <= 0.5*(1 + TOL) and not np.any(bad):
                    stats["subcell_projections"] += n1 + n2
                    # Stage-1 projection energy enters the final state with weight 1/2.
                    stats["regularization_dE"] += 0.5*dE1 + dE2
                    break
            if np.any(bad & ~first):
                first |= _dilate(bad)
                stats["mood_cells"] += int(np.count_nonzero(bad))
                rhs, rate, nd, sm, nf = _transport_rhs(U, closure, dxs, first)
                h = min(h, 0.5*cfl/rate if rate > 0 else h)
                continue
            if attempt == 30 + _MOOD_PASSES:
                raise RuntimeError("dust substep failed after 30 consecutive halvings")
            h *= 0.5
            stats["halvings"] += 1
        U = candidate
        stats["substeps"] += 1
        remaining = max(0.0, remaining - h)
    return U, stats


def diffuse(U: np.ndarray, kappa: np.ndarray, dt: float,
            dxs: tuple) -> tuple[np.ndarray, int]:
    """Diffuse all fields by the same scalar coefficient, conservatively."""
    U = np.array(U, dtype=np.float64, copy=True)
    dxs = _grid(dxs, U.ndim - 1, dt)
    kappa = np.broadcast_to(np.asarray(kappa, dtype=np.float64), U.shape[1:])
    if np.any(kappa < 0) or not np.all(np.isfinite(kappa)):
        raise ValueError("diffusivity must be finite and non-negative")
    faces = [0.5*(kappa + np.roll(kappa, -1, axis=-1-d))/dx**2
             for d, dx in enumerate(dxs)]
    rate = np.max(sum(face + np.roll(face, 1, axis=-1-d)
                      for d, face in enumerate(faces)))
    nsub = max(1, int(np.ceil(dt*rate/0.9))) if dt > 0 else 0
    for _ in range(nsub):
        rhs = np.zeros_like(U)
        for d, face in enumerate(faces):
            flux = face * (np.roll(U, -1, axis=-1-d) - U)
            rhs += flux - np.roll(flux, 1, axis=-1-d)
        U += (dt/nsub) * rhs
    return U, nsub


def phi(rate: np.ndarray, h: float) -> np.ndarray:
    """Evaluate (1-exp(-rate*h))/rate continuously at zero rate."""
    rate = np.asarray(rate, dtype=np.float64)
    z = rate*h
    small = np.abs(z) < 1e-2
    zs = np.where(small, z, 0.0)
    series = h*(1 + zs*(-0.5 + zs*(1/6 + zs*(-1/24 + zs*(1/120 - zs/720)))))
    return np.where(small, series, _divide(-np.expm1(-z), rate))


def ou_gaussian(C: np.ndarray, A: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """Exact isotropic OU map of the covariance tensor."""
    C = np.asarray(C, dtype=np.float64)
    result = C*np.asarray(A, dtype=np.float64)**2
    for i in range(3):
        result[i, i] += Sigma
    return result


def ou_hyqmom(v: np.ndarray, q: np.ndarray, r: np.ndarray, sperp2: np.ndarray,
              A: np.ndarray, Sigma: np.ndarray) -> tuple:
    """Exact OU map of central moments through fourth order."""
    v, q, r, sperp2 = (np.asarray(value, dtype=np.float64) for value in (v, q, r, sperp2))
    A, Sigma = np.asarray(A, dtype=np.float64), np.asarray(Sigma, dtype=np.float64)
    return (A*A*v + Sigma, A**3*q, A**4*r + 6*A*A*Sigma*v + 3*Sigma*Sigma,
            A*A*sperp2 + Sigma)
