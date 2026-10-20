"""Frozen midpoint-resolvent ambipolar diffusion on two-dimensional grids."""

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


class MidpointAmbipolar:
    """Second-order coefficient prediction and an H-weighted SPD resolvent.

    Open-y grids use zero tangential *diffusive* electric field. This is a
    natural flux closure, separate from the ideal-MHD outflow condition. The
    boundary tensor is R K R, so endpoint constitutive accuracy can be lower
    than the interior stencil accuracy when the normal field is nonzero.
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

    def _electric(self, B, reference):
        B = self.grid.project_dealiased(B)
        J = self._mask(self._curl(B))
        b2 = (reference * reference).sum(axis=0)
        dot = (reference * J).sum(axis=0)
        E = self.eta * (b2 * J - reference * dot)
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
        small = h*self.eta*((k*b).sum(axis=0))**2
        large = h*self.eta*(mean*mean).sum()*k2
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
