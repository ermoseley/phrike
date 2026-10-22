"""Non-ideal MHD operators: midpoint-resolvent ambipolar diffusion in 2D and Hall induction."""

import math
import warnings

import numpy as np

try:
    import torch
except ImportError:  # NumPy remains usable without PyTorch.
    torch = None


def _tensor(a):
    return torch is not None and isinstance(a, torch.Tensor)


def _copy(a):
    return a.clone() if _tensor(a) else a.copy()


def _stack(parts):
    return torch.stack(parts) if _tensor(parts[0]) else np.stack(parts)


def _sum(a):
    return a.sum()


def _number(a):
    return float(a.item()) if _tensor(a) else float(a)


def _host(a):
    return a.detach().cpu().numpy() if _tensor(a) else np.asarray(a)


def _sqrt(a):
    return torch.sqrt(a) if _tensor(a) else np.sqrt(a)


def _abs(a):
    return torch.abs(a) if _tensor(a) else np.abs(a)


def _eps(a):
    return float(torch.finfo(a.dtype).eps if _tensor(a) else np.finfo(a.dtype).eps)


def _floor(a, value):
    return torch.clamp(a, min=value) if _tensor(a) else np.maximum(a, value)


def _cap(a, value):
    return torch.clamp(a, max=value) if _tensor(a) else np.minimum(a, value)


def _clip(a, bound):
    return torch.clamp(a, -bound, bound) if _tensor(a) else np.clip(a, -bound, bound)


def _max(a):
    return float(a.max().item()) if _tensor(a) else float(np.max(a))


def _min(a):
    return float(a.min().item()) if _tensor(a) else float(np.min(a))


def _cross(a, b):
    return _stack([a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]])


def _fft(V, dim):
    axes = tuple(range(-dim, 0))
    return torch.fft.fftn(V, dim=axes) if _tensor(V) else np.fft.fftn(V, axes=axes)


def _ifft(F, dim, like):
    axes = tuple(range(-dim, 0))
    if _tensor(F):
        return torch.fft.ifftn(F, dim=axes).real
    return np.fft.ifftn(F, axes=axes).real.astype(like.dtype, copy=False)


class MidpointAmbipolar:
    """Second-order coefficient prediction and an H-weighted SPD resolvent.

    Open-y grids use zero tangential *diffusive* electric field. This is a
    natural flux closure, separate from the ideal-MHD outflow condition. The
    boundary tensor is R K R, so endpoint constitutive accuracy can be lower
    than the interior stencil accuracy when the normal field is nonzero.

    The electric field E = eta (|B|^2 J - B (B.J)) is a diffusion of J_perp with
    effective diffusivity eta_eff = eta |B|^2. ``eta_max`` caps it by the factor
    c(x) = min(1, eta_max/(eta |B_ref|^2)) on the frozen reference field, which
    keeps the tensor eta c (|B|^2 I - B B^T) symmetric positive semidefinite.
    """

    def __init__(self, grid, config):
        self.grid = grid
        self.config = dict(config)
        self.eta = float(config.get("eta", config.get("alpha", 1.0)))
        if self.eta < 0 or not np.isfinite(self.eta):
            raise ValueError("Ambipolar eta must be finite and nonnegative")
        if config.get("method", "midpoint_resolvent") != "midpoint_resolvent":
            raise ValueError("Ambipolar method must be midpoint_resolvent")
        self.preconditioner = config.get("midpoint_preconditioner",
                                         config.get("preconditioner", "fourier_tensor"))
        if self.preconditioner not in ("fourier_tensor", "none", "identity"):
            raise ValueError("Unknown ambipolar preconditioner")
        self.eta_max = None if config.get("eta_max") is None else float(config["eta_max"])
        if self.eta_max is not None and not (np.isfinite(self.eta_max) and self.eta_max > 0.0):
            raise ValueError("Ambipolar eta_max must be finite and positive")
        self.last_iterations = 0
        self.last_residual = 0.0
        self.total_iterations = 0
        self.total_solves = 0
        self._weights = None
        self._wavevectors = None

    def _weight(self, B):
        if self._weights is None:
            w = (np.asarray(self.grid.wy) if getattr(self.grid, "_mixed_open_y", False)
                 else np.ones(B.shape[-2]))
            w = w.reshape(1, -1, 1)
            self._weights = (torch.as_tensor(w, dtype=B.dtype, device=B.device)
                             if _tensor(B) else w.astype(B.dtype))
        return self._weights

    def _mask(self, V):
        if not getattr(self.grid, "_mixed_open_y", False):
            return V
        result = _copy(V)
        result[0, 0] = result[0, -1] = 0
        result[2, 0] = result[2, -1] = 0
        return result

    def _curl(self, B):
        # Exact constants should not create roundoff currents in SBP closures.
        B = B - B[:, :1, :1]
        dx, dy = self.grid.dx1(B), self.grid.dy1(B)
        return _stack([dy[2], -dx[2], dx[1]-dy[0]])

    def _coupling(self, b2):
        """c(x) = min(1, eta_max/(eta b2)) of the frozen reference, so that eta c b2 <= eta_max."""
        ratio = self.eta_max/self.eta if self.eta > 0 else math.inf
        return _cap(ratio/_floor(b2, 1e-30), 1.0)

    def _electric(self, B, reference):
        B = self.grid.project_dealiased(B)
        J = self._mask(self._curl(B))
        b2 = (reference * reference).sum(axis=0)
        dot = (reference * J).sum(axis=0)
        E = self.eta * (b2 * J - reference * dot)
        if self.eta_max is not None:
            E = E * self._coupling(b2)
        return self.grid.project_dealiased(self._mask(E))

    def _operator(self, B, reference):
        return -self._curl(self._electric(B, reference))

    def _precondition(self, reference, h):
        if self.preconditioner != "fourier_tensor":
            return lambda x: x
        w = self._weight(reference)
        mean = (reference * w).sum(axis=(-2, -1)) / (w.sum() * reference.shape[-1])
        ny, nx = reference.shape[-2:]
        if self._wavevectors is None:
            kx = 2*np.pi*np.fft.fftfreq(nx, d=float(self.grid.dx))
            ky = 2*np.pi*np.fft.fftfreq(ny, d=float(self.grid.dy))
            if getattr(self.grid, "_mixed_open_y", False):
                # Match the SBP interior symbol, including its near-Nyquist null.
                theta = ky*float(self.grid.dy)
                ky = (8*np.sin(theta)-np.sin(2*theta))/(6*float(self.grid.dy))
            if nx % 2 == 0:
                kx[nx//2] = 0
            if ny % 2 == 0:
                ky[ny//2] = 0
            kx, ky = np.broadcast_arrays(kx[None, :], ky[:, None])
            k = np.stack([kx, ky, np.zeros_like(kx)])
            self._wavevectors = (torch.as_tensor(k.copy(), dtype=reference.dtype,
                                                device=reference.device)
                                 if _tensor(reference) else k.astype(reference.dtype))
        k = self._wavevectors
        b = mean[:, None, None]
        a = _stack([k[1]*b[2], -k[0]*b[2], k[0]*b[1]-k[1]*b[0]])
        k2 = (k*k).sum(axis=0)
        a2 = (a*a).sum(axis=0)
        # Avoid divisions at null modes without removing those modes.
        if _tensor(reference):
            ik2 = torch.where(k2 > 0, 1/torch.clamp(k2, min=1e-30), 0)
            ia2 = torch.where(a2 > 0, 1/torch.clamp(a2, min=1e-30), 0)
        else:
            ik2 = np.divide(1, k2, out=np.zeros_like(k2), where=k2 > 0)
            ia2 = np.divide(1, a2, out=np.zeros_like(a2), where=a2 > 0)
        eta = self.eta
        if self.eta_max is not None:
            # The |B|^2-weighted mean of the cap factor scales the mean-field tensor eta |b|^2.
            b2 = (reference*reference).sum(axis=0)
            total = _number(_sum(w*b2))
            if total > 0:
                eta = eta*_number(_sum(w*self._coupling(b2)*b2))/total
        small = h*eta*((k*b).sum(axis=0))**2
        large = h*eta*(mean*mean).sum()*k2
        fs = 1/(1+small+0.5*small*small)
        fl = 1/(1+large+0.5*large*large)

        def apply(V):
            if _tensor(V):
                Vhat = torch.fft.fft2(V, dim=(-2, -1))
            else:
                Vhat = np.fft.fft2(V, axes=(-2, -1))
            longitudinal = k*((k*Vhat).sum(axis=0)*ik2)
            along_a = a*((a*Vhat).sum(axis=0)*ia2)
            result = longitudinal + fs*along_a + fl*(Vhat-longitudinal-along_a)
            if _tensor(V):
                return torch.fft.ifft2(result, dim=(-2, -1)).real
            return np.fft.ifft2(result, axes=(-2, -1)).real.astype(V.dtype)
        return apply

    def _resolve(self, B0, reference, h):
        L = lambda B: self._operator(B, reference)
        LB = L(B0)
        forcing = h*LB - 0.5*h*h*L(LB)
        sqrtw = self._weight(B0)**0.5
        b = sqrtw*forcing
        normb = _number(_sum(b*b))**0.5
        if normb == 0:
            return _copy(B0)
        tol = float(self.config.get("midpoint_tol", 2e-5 if B0.dtype in
                    (np.dtype("float32"), getattr(torch, "float32", None)) else 1e-10))
        maxiter = int(self.config.get("midpoint_maxiter", 200))
        if tol <= 0 or maxiter < 1:
            raise ValueError("Ambipolar CG tolerance and iteration limit must be positive")

        def A(x):
            B = x/sqrtw
            LB = L(B)
            return sqrtw*(B-h*LB+0.5*h*h*L(LB))

        precondition = self._precondition(reference, h)
        x = b*0
        r = _copy(b)
        z = precondition(r)
        p = _copy(z)
        rz = _number(_sum(r*z))
        residual = 1.0
        iterations = 0
        for iterations in range(1, maxiter+1):
            Ap = A(p)
            curvature = _number(_sum(p*Ap))
            if curvature <= 0 or not np.isfinite(curvature) or rz <= 0:
                raise RuntimeError("Ambipolar CG lost positive curvature")
            alpha = rz/curvature
            x = x+alpha*p
            r = r-alpha*Ap
            residual = _number(_sum(r*r))**0.5/normb
            if residual <= tol:
                r = b-A(x)
                residual = _number(_sum(r*r))**0.5/normb
                if residual <= tol:
                    break
                # Reliable residual replacement restarts CG after roundoff drift.
                z = precondition(r)
                p = _copy(z)
                rz = _number(_sum(r*z))
                continue
            z = precondition(r)
            new_rz = _number(_sum(r*z))
            p = z+(new_rz/rz)*p
            rz = new_rz
        self.last_iterations += iterations
        self.total_iterations += iterations
        self.total_solves += 1
        self.last_residual = max(self.last_residual, residual)
        if not np.isfinite(residual) or residual > tol:
            raise RuntimeError(f"Ambipolar CG did not converge: residual={residual:.3e}, "
                               f"iterations={iterations}")
        return B0+x/sqrtw

    def step(self, U, equations, dt):
        """Advance B and conservative Poynting flux; preserve other fields."""
        if U.ndim != 3 or U.shape[0] != 8:
            raise ValueError("Ambipolar diffusion requires a 2D eight-field MHD state")
        h = float(dt)
        if h < 0 or not np.isfinite(h):
            raise ValueError("Ambipolar timestep must be finite and nonnegative")
        self.last_iterations = 0
        self.last_residual = 0.0
        if h == 0 or self.eta == 0:
            return _copy(U)
        B0 = U[5:8]
        half = self._resolve(B0, B0, 0.5*h)
        Bn = self._resolve(B0, half, h)
        Bstar = Bn-0.5*h*self._operator(Bn, half)
        E = self._electric(Bstar, half)
        Bmid = 0.5*(B0+Bn)
        # dE_total/dt = -div(E_AD x B), with the same discrete AD electric field.
        flux_x = E[1]*Bmid[2]-E[2]*Bmid[1]
        flux_y = E[2]*Bmid[0]-E[0]*Bmid[2]
        result = _copy(U)
        result[4] = U[4]-h*(self.grid.dx1(flux_x)+self.grid.dy1(flux_y))
        result[5:8] = Bn
        return result


class Hall:
    """Hall induction dB/dt = -curl(eta_H J x B/|B|) on periodic Fourier grids.

    ``reference`` integrates the nonlinear equation with RK4 substeps limited by
    the Hall rate max|eta_H| kmax^2 of the masked curl. An optional ``eta_max`` caps
    |eta_H(x)| (sign kept) in the field, the rate and the automatic l^2. ``inertia`` solves
    (I + curl l^2 curl) dB/dt = -curl E_H with l frozen over the call: solenoidal
    modes see eta_H/(1 + l^2 k^2) and curl-free parts are untouched. A uniform l
    uses an exact Fourier inverse; l(x) uses preconditioned CG, with a tolerance
    floored at 10 eps (float64) or 100 eps (float32) and raises if it does not converge. Density, momentum and internal energy are held fixed, so
    total energy changes only by the change of 0.5|B|^2.

    ``hyper`` C (reference only) adds E_hyp = -C |eta_H(x)| dx^2 laplacian(J) (Iwasaki and Tomida
    2025), dB/dt = -nu k^4 B per mode with nu = C |eta_H| dx^2 for the smallest spacing dx. Its work
    E_hyp.J is deposited in the total energy, i.e. as internal energy, so the call conserves total
    energy; its rate C |eta_H| dx^2 kmax^4 is added to the substep rate.

    ``xmhd`` is a reduced-speed extended-MHD Hall subflow with persistent fields E and J (velocity
    frozen):  dB/dt = -curl E,  (1/c^2) dE/dt = curl B - J,  l^2 dJ/dt = E - eta_H (J x b),
    b = B/|B|. Its steady state is J = curl B, E = eta_H J x b, and it conserves
    0.5|B|^2 + |E|^2/(2 c^2) + 0.5 l^2 |J|^2 (``xmhd_energy``). Linear whistler, k along B:
    l^2 w^3 + eta w^2 - c^2 (1 + k^2 l^2) w - c^2 k^2 eta = 0; w -> -eta k^2/(1 + k^2 l^2) as c -> inf.
    The speed is ``c_tilde`` or ``c_factor`` dx/dt, and l is ``length`` or the automatic ``inertia``
    value (a call whose automatic l^2 is at most 1/kmax^2 uses the reference substeps instead).
    The unsplit RK4 substep rate bounds the eigenfrequencies of the system, which are at most
    c sqrt(kmax^2 + 1/l^2) + max|eta_H|/l^2 (light waves and plasma oscillation, plus the local
    cyclotron rotation, which is integrated explicitly so that equilibria stay exact).
    E and J live on the object between calls, start at their steady values for the first B (and
    after ``reset_fields``) and are not saved in checkpoints: a restart re-initialises them.
    """

    def __init__(self, grid, config):
        self.grid = grid
        self.config = dict(config)
        cfg = self.config
        if (getattr(grid, "_mixed_open_y", False)
                or any(getattr(grid, name, "fourier") != "fourier"
                       for name in ("_basis_name", "basis_x", "basis_y"))):
            raise ValueError("Hall requires a periodic Fourier grid")
        self.dim = (3 if getattr(grid, "Nz", None) is not None
                    else 2 if getattr(grid, "Ny", None) is not None else 1)
        self.eta = float(cfg.get("eta_h", 0.0))
        self.b0 = float(cfg.get("b0", 1.0))
        self.rho0 = float(cfg.get("rho0", 1.0))
        if not np.isfinite(self.eta):
            raise ValueError("Hall eta_h must be finite")
        if not (np.isfinite(self.b0) and np.isfinite(self.rho0) and self.b0 > 0 and self.rho0 > 0):
            raise ValueError("Hall b0 and rho0 must be positive and finite")
        self.eta_max = None if cfg.get("eta_max") is None else float(cfg["eta_max"])
        if self.eta_max is not None and not (np.isfinite(self.eta_max) and self.eta_max > 0.0):
            raise ValueError("Hall eta_max must be finite and positive")
        self.form = str(cfg.get("eta_h_form", "constant")).lower()
        if self.form not in ("constant", "proportional_to_b_over_rho"):
            raise ValueError("Hall eta_h_form must be constant or proportional_to_B_over_rho")
        self.method = str(cfg.get("method", "reference")).lower()
        if self.method not in ("reference", "inertia", "xmhd"):
            raise ValueError("Hall method must be reference, inertia or xmhd")
        # RK4 is stable for imaginary rates up to 2 sqrt(2); 2 is kept for the damped regularized modes.
        self.safety = float(cfg.get("safety", 0.5))
        if not 0.0 < self.safety <= 2.0:
            raise ValueError("Hall safety must lie in (0, 2]")
        self.length_mode = str(cfg.get("length_mode", "constant")).lower()
        if self.length_mode not in ("constant", "local"):
            raise ValueError("Hall length_mode must be constant or local")
        self.length = None if cfg.get("length") is None else float(cfg["length"])
        if self.length is not None and not (np.isfinite(self.length) and self.length >= 0.0):
            raise ValueError("Hall length must be finite and nonnegative")
        self.length_ratio = None if cfg.get("length_ratio") is None else float(cfg["length_ratio"])
        if self.length_ratio is not None and not (np.isfinite(self.length_ratio)
                                                  and self.length_ratio >= 1.0):
            raise ValueError("Hall length_ratio must be finite and at least 1")
        self.cg_tol = None if cfg.get("cg_tol") is None else float(cfg["cg_tol"])
        self.cg_maxiter = int(cfg.get("cg_maxiter", 200))
        if (self.cg_tol is not None and not self.cg_tol > 0) or self.cg_maxiter < 1:
            raise ValueError("Hall CG tolerance and iteration limit must be positive")
        self.hyper = float(cfg.get("hyper", 0.0))
        if not (np.isfinite(self.hyper) and self.hyper >= 0.0):
            raise ValueError("Hall hyper must be finite and nonnegative")
        if self.hyper > 0.0 and self.method != "reference":
            raise ValueError("Hall hyper requires method reference")
        self.c_tilde = None if cfg.get("c_tilde") is None else float(cfg["c_tilde"])
        self.c_factor = float(cfg.get("c_factor", 2.0))
        if not ((self.c_tilde is None or (np.isfinite(self.c_tilde) and self.c_tilde > 0.0))
                and np.isfinite(self.c_factor) and self.c_factor > 0.0):
            raise ValueError("Hall c_tilde and c_factor must be finite and positive")
        self.E = self.J = None
        self.last_c = self.last_length2 = None
        self._work = 0.0
        if self.method == "xmhd":
            if self.length_mode != "constant":
                raise ValueError("Hall xmhd requires length_mode constant")
            warnings.warn("Hall xmhd keeps E and J between calls and does not save them in checkpoints: "
                          "a restart re-initialises them to the steady values for B",
                          RuntimeWarning, stacklevel=2)
        self._k = self._k_key = self._kmax2 = None
        self.last_stats = None
        self.reset_stats()

    def reset_stats(self):
        self.total = dict(calls=0, substeps=0, cg_calls=0, cg_iterations=0,
                          cg_max_iterations=0, cg_residual=0.0,
                          length2_min=math.inf, length2_max=0.0)

    def _record(self, stats):
        """Keep the last call's statistics and fold them into the running totals."""
        calls = max(stats["cg_calls"], 1)
        stats["cg_mean_iterations"] = stats["cg_iterations"]/calls
        self.last_stats = stats
        t = self.total
        t["calls"] += 1
        for key in ("substeps", "cg_calls", "cg_iterations"):
            t[key] += stats[key]
        t["cg_max_iterations"] = max(t["cg_max_iterations"], stats["cg_max_iterations"])
        t["cg_residual"] = max(t["cg_residual"], stats["cg_residual"])
        t["length2_min"] = min(t["length2_min"], stats["length2_min"])
        t["length2_max"] = max(t["length2_max"], stats["length2_max"])

    def _wavevectors(self, ref):
        """Symbols of the masked curl: dealias mask applied, Nyquist set to zero."""
        key = (_tensor(ref), str(ref.dtype), str(getattr(ref, "device", "")))
        if self._k_key != key:
            g = self.grid
            names = ("k",) if self.dim == 1 else ("kx", "ky", "kz")[:self.dim]
            mask = _host(g.dealias_mask)
            k = []
            for axis, name in enumerate(names):
                a = _host(getattr(g, name)).copy()
                if len(a) % 2 == 0:
                    a[len(a)//2] = 0.0
                shape = [1]*self.dim
                shape[self.dim-1-axis] = -1
                k.append(a.reshape(shape)*mask)
            k += [np.zeros_like(k[0])]*(3-self.dim)
            self._kmax2 = float(max(np.max(k[0]**2+k[1]**2+k[2]**2), 1e-30))
            if _tensor(ref):
                self._k = [torch.as_tensor(a, dtype=ref.dtype, device=ref.device) for a in k]
            else:
                self._k = [a.astype(ref.dtype) for a in k]
            self._k_key = key
        return self._k

    def kmax2(self, ref):
        self._wavevectors(ref)
        return self._kmax2

    def curl(self, V):
        kx, ky, kz = self._wavevectors(V)
        F = _fft(V, self.dim)
        C = _stack([ky*F[2]-kz*F[1], kz*F[0]-kx*F[2], kx*F[1]-ky*F[0]])
        return _ifft(1j*C, self.dim, V)

    def eta_field(self, B, rho):
        """eta_H(x): constant, or eta_h (|B|/b0)(rho0/rho); eta_max caps |eta_H(x)| keeping the sign."""
        if self.form == "constant":
            eta = self.eta+0*rho
        else:
            eta = self.eta*_sqrt((B*B).sum(axis=0))/self.b0*self.rho0/_floor(rho, 1e-30)
        return eta if self.eta_max is None else _clip(eta, self.eta_max)

    def hall_coupling(self, B, rho):
        """eta_H/|B|; |B| cancels analytically in the proportional form unless eta_max is set."""
        if self.eta_max is not None:
            return self.eta_field(B, rho)/_floor(_sqrt((B*B).sum(axis=0)), 1e-30)
        if self.form == "constant":
            return self.eta/_floor(_sqrt((B*B).sum(axis=0)), 1e-30)
        return self.eta*self.rho0/self.b0/_floor(rho, 1e-30)

    def spacing(self):
        """Smallest grid spacing."""
        return min(float(getattr(self.grid, name)) for name in ("dx", "dy", "dz")[:self.dim])

    def hyper_field(self, B, rho, J):
        """E_hyp = -C |eta_H(x)| dx^2 laplacian(J) with the masked symbols, so that E_hyp.J integrates to >= 0."""
        kx, ky, kz = self._wavevectors(J)
        nu = self.hyper*self.spacing()**2*_abs(self.eta_field(B, rho))
        return nu*_ifft((kx*kx+ky*ky+kz*kz)*_fft(J, self.dim), self.dim, J)

    def electric_field(self, B, rho):
        """E_H = eta_H J x B/|B| (plus E_hyp, whose work E_hyp.J is left in ``_work``)."""
        J = self.curl(B)
        E = self.hall_coupling(B, rho)*_cross(J, B)
        if self.hyper > 0.0:
            hyper = self.hyper_field(B, rho, J)
            self._work = (hyper*J).sum(axis=0)
            E = E+hyper
        return E

    def steady_fields(self, B, rho):
        """Steady values (E, J) = (E_H, curl B) of the xmhd subflow for the field B."""
        return self.electric_field(B, rho), self.curl(B)

    def reset_fields(self):
        """Forget E and J; the next xmhd call sets them to their steady values for its B."""
        self.E = self.J = None

    def length_squared(self, U, dt):
        """Inertia metric l^2 frozen for one call; an explicit length wins over the automatic one.

        Automatic: l^2 = max(0, max|eta_H| dt/safety - 1/kmax^2), which makes one RK4
        step stable and is zero when the Hall rate is already resolved. With
        length_mode local, |eta_H(x)| replaces the maximum (or l^2 ~ rho0/rho for an
        explicit length) and length_ratio raises l^2 so that max(l)/min(l) <= ratio.
        """
        B, rho = U[5:8], U[0]
        local = self.length_mode == "local"
        if self.length is not None:
            if not local:
                return self.length**2
            length2 = self.length**2*self.rho0/_floor(rho, 1e-30)
        else:
            eta = _abs(self.eta_field(B, rho))
            length2 = (eta if local else _max(eta))*dt/self.safety-1.0/self.kmax2(B)
            if not local:
                return max(0.0, length2)
            length2 = _floor(length2, 0.0)
        if self.length_ratio is not None:
            length2 = _floor(length2, _max(length2)/self.length_ratio**2)
        return length2

    def _inverse_constant(self, V, length2):
        """(I + l^2 curl curl)^-1 for uniform l: (v + l^2 k (k.v))/(1 + l^2 k^2)."""
        kx, ky, kz = self._wavevectors(V)
        F = _fft(V, self.dim)
        dot = kx*F[0]+ky*F[1]+kz*F[2]
        G = _stack([F[0]+length2*kx*dot, F[1]+length2*ky*dot, F[2]+length2*kz*dot])
        return _ifft(G/(1.0+length2*(kx*kx+ky*ky+kz*kz)), self.dim, V)

    def inertia_solve(self, rhs, length2, tol=None, maxiter=None):
        """Solve (I + curl l^2 curl) x = rhs; returns (x, iterations, relative residual)."""
        if isinstance(length2, float):
            if not np.isfinite(length2) or length2 < 0.0:
                raise ValueError("Hall length squared must be finite and nonnegative")
            return self._inverse_constant(rhs, length2), 0, 0.0
        if not np.isfinite(_max(length2)) or _min(length2) < 0.0:
            raise ValueError("Hall length squared must be finite and nonnegative")
        if tol is None:
            tol = self.cg_tol or (2e-5 if str(rhs.dtype).endswith("float32") else 1e-10)
        # A smaller relative residual is below roundoff. Two curls and an FFT pair per matvec
        # stagnate float32 at ~55 eps (6.6e-6 measured), so 100 eps is reachable; float64 keeps 10 eps.
        eps = _eps(rhs)
        tol = max(tol, (100 if eps > 1e-10 else 10)*eps)
        maxiter = self.cg_maxiter if maxiter is None else int(maxiter)
        mu = _number(length2.mean())
        precondition = lambda v: self._inverse_constant(v, mu)
        matvec = lambda v: v+self.curl(length2*self.curl(v))
        norm2 = _number(_sum(rhs*rhs))
        # Corrections lie in range(curl), so curl-free parts of rhs are preserved.
        x = _copy(rhs)
        if norm2 == 0.0:
            return x, 0, 0.0
        r = -self.curl(length2*self.curl(x))
        residual = (_number(_sum(r*r))/norm2)**0.5
        if residual <= tol:
            return x, 0, residual
        z = precondition(r)
        p, rz = _copy(z), _number(_sum(r*z))
        for iterations in range(1, maxiter+1):
            Ap = matvec(p)
            curvature = _number(_sum(p*Ap))
            if curvature <= 0 or not np.isfinite(curvature):
                raise RuntimeError("Hall PCG lost positive curvature")
            alpha = rz/curvature
            x, r = x+alpha*p, r-alpha*Ap
            residual = (_number(_sum(r*r))/norm2)**0.5
            if residual <= tol:
                r = rhs-matvec(x)
                residual = (_number(_sum(r*r))/norm2)**0.5
                if residual <= tol:
                    return x, iterations, residual
                # Reliable residual replacement restarts CG after roundoff drift.
                z = precondition(r)
                p, rz = _copy(z), _number(_sum(r*z))
                continue
            z = precondition(r)
            new_rz = _number(_sum(r*z))
            p, rz = z+(new_rz/rz)*p, new_rz
        raise RuntimeError(f"Hall PCG failed: iterations={maxiter}, residual={residual:.3e}")

    def reservoir_energy(self, B, length2):
        """Volume mean 0.5 l^2 |curl B|^2, the inertia reservoir for a fixed l(x)."""
        J = self.curl(B)
        return _number((length2*0.5*(J*J).sum(axis=0)).mean())

    def modified_energy(self, B, length2=0.0):
        """Volume mean 0.5(|B|^2 + l^2 |curl B|^2), conserved by the Hall subflow at fixed l(x)."""
        return _number((0.5*(B*B).sum(axis=0)).mean())+self.reservoir_energy(B, length2)

    def xmhd_energy(self, B, c_tilde=None, length2=None):
        """Volume mean 0.5|B|^2 + |E|^2/(2 c^2) + 0.5 l^2 |J|^2 of B and the stored E, J; conserved by xmhd.

        c and l^2 default to the values of the last xmhd call.
        """
        c = self.last_c if c_tilde is None else c_tilde
        length2 = self.last_length2 if length2 is None else length2
        mean = lambda a: _number((0.5*(a*a).sum(axis=0)).mean())
        return mean(B)+mean(self.E)/(c*c)+length2*mean(self.J)

    def _peak(self, B, rho):
        """max|eta_H|, the sign-free Hall coefficient of the rate."""
        if self.form == "constant":
            return abs(self.eta) if self.eta_max is None else min(abs(self.eta), self.eta_max)
        return _max(_abs(self.eta_field(B, rho)))

    def _finish(self, U, B, stats, heat=None):
        """Result state: B replaced, total energy changed by the magnetic energy (and the hyper heat)."""
        if not np.isfinite(_max(_abs(B))):
            raise RuntimeError("Nonfinite Hall magnetic field")
        B0 = U[5:8]
        result = _copy(U)
        result[5:8] = B
        result[4] = U[4]+0.5*((B*B).sum(axis=0)-(B0*B0).sum(axis=0))
        if heat is not None:
            result[4] = result[4]+heat
        self._record(stats)
        return result

    def _xmhd(self, U, h, stats):
        """RK4 substeps of (B, E, J); returns B, or None when the reference substeps should be used."""
        B0, rho = U[5:8], U[0]
        length2 = self.length_squared(U, h)
        kmax2 = self.kmax2(B0)
        # l = 0 is the reference flow; an automatic l below the grid scale would cost more than it.
        if length2 == 0.0 or (self.length is None and length2 <= 1.0/kmax2):
            self.reset_fields()
            return None
        stats["initialised"] = self.E is None or self.E.shape != B0.shape
        if stats["initialised"]:
            self.E, self.J = self.steady_fields(B0, rho)
        c = self.c_tilde if self.c_tilde is not None else self.c_factor*self.spacing()/h
        self.last_c, self.last_length2 = c, length2
        stats.update(length2_min=length2, length2_max=length2, c_tilde=c)

        def rhs(X):
            B, E, J = X
            return _stack([-self.curl(E), c*c*(self.curl(B)-J),
                           (E-self.hall_coupling(B, rho)*_cross(J, B))/length2])

        X, elapsed = _stack([B0, self.E, self.J]), 0.0
        while True:
            # Skew in the energy norm: |w| <= c sqrt(kmax^2 + 1/l^2) (light, plasma) + max|eta_H|/l^2 (cyclotron).
            rate = c*math.sqrt(kmax2+1.0/length2)+self._peak(X[0], rho)/length2
            if not np.isfinite(rate):
                raise RuntimeError("Nonfinite Hall rate")
            remaining = h-elapsed
            n = max(1, math.ceil(remaining*rate/self.safety-1e-9))
            hs = remaining/n
            if elapsed+hs == elapsed:
                raise RuntimeError("Hall substep underflow")
            k1 = rhs(X)
            k2 = rhs(X+0.5*hs*k1)
            k3 = rhs(X+0.5*hs*k2)
            k4 = rhs(X+hs*k3)
            X = X+(hs/6.0)*(k1+2*k2+2*k3+k4)
            elapsed += hs
            stats["substeps"] += 1
            if n == 1:
                break
        self.E, self.J = X[1], X[2]
        return X[0]

    def step(self, U, equations, dt):
        """Advance B by the Hall subflow; density, momentum and internal energy are preserved (except hyper heat)."""
        h = float(dt)
        if not np.isfinite(h) or h < 0.0:
            raise ValueError("Hall timestep must be finite and nonnegative")
        if U.ndim != self.dim+1 or U.shape[0] != 8:
            raise ValueError("Hall requires an eight-field MHD state matching the grid")
        stats = dict(substeps=0, cg_calls=0, cg_iterations=0, cg_max_iterations=0,
                     cg_residual=0.0, length2_min=0.0, length2_max=0.0)
        if h == 0 or self.eta == 0:
            self._record(stats)
            return _copy(U)
        if self.method == "xmhd":
            B = self._xmhd(U, h, stats)
            if B is not None:
                return self._finish(U, B, stats)
        B0, rho = U[5:8], U[0]
        inertia = self.method == "inertia"
        length2 = self.length_squared(U, h) if inertia else 0.0
        iterative = inertia and not isinstance(length2, float)
        if inertia:
            stats["length2_min"] = length2 if not iterative else _min(length2)
            stats["length2_max"] = length2 if not iterative else _max(length2)
        kmax2 = self.kmax2(B0)
        # The hyper rate nu_max kmax^4 = C |eta_H| dx^2 kmax^4 adds to the Hall rate |eta_H| kmax^2.
        hyper_factor = 1.0+self.hyper*self.spacing()**2*kmax2
        works = []

        def rhs(B):
            value = -self.curl(self.electric_field(B, rho))
            if self.hyper > 0.0:
                works.append(self._work)
            if inertia:
                value, iterations, residual = self.inertia_solve(value, length2)
                if iterative:
                    stats["cg_calls"] += 1
                    stats["cg_iterations"] += iterations
                    stats["cg_max_iterations"] = max(stats["cg_max_iterations"], iterations)
                    stats["cg_residual"] = max(stats["cg_residual"], residual)
            return value

        B, elapsed, heat = _copy(B0), 0.0, 0.0
        while True:
            # Reevaluate nonlinear eta_H each substep; the smallest l bounds the rate.
            rate = self._peak(B, rho)*kmax2*hyper_factor
            if inertia:
                rate /= 1.0+stats["length2_min"]*kmax2
            if not np.isfinite(rate):
                raise RuntimeError("Nonfinite Hall rate")
            remaining = h-elapsed
            n = max(1, math.ceil(remaining*rate/self.safety-1e-9))
            hs = remaining/n
            if elapsed+hs == elapsed:
                raise RuntimeError("Hall substep underflow")
            k1 = rhs(B)
            k2 = rhs(B+0.5*hs*k1)
            k3 = rhs(B+0.5*hs*k2)
            k4 = rhs(B+hs*k3)
            B = B+(hs/6.0)*(k1+2*k2+2*k3+k4)
            if works:
                # Heat E_hyp.J of this substep, with the RK4 weights of the stages.
                heat = heat+(hs/6.0)*(works[0]+2*works[1]+2*works[2]+works[3])
                works.clear()
            elapsed += hs
            stats["substeps"] += 1
            if n == 1:
                break
        return self._finish(U, B, stats, heat if self.hyper > 0.0 else None)
