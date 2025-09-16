from __future__ import annotations

from typing import Optional

import numpy as np

try:  # Optional torch backend for GPU-accelerated DCTs
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch may not be installed
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

try:  # Prefer pyfftw if available for performance
    import pyfftw.interfaces.scipy_fft as scipy_fft  # type: ignore
    from pyfftw.interfaces import cache as fftw_cache  # type: ignore

    _FFT_CACHE = True
except Exception:  # Fallback to SciPy's FFT/DCT
    from scipy import fft as scipy_fft  # type: ignore

    _FFT_CACHE = False
    fftw_cache = None  # type: ignore

from .base import Basis1D, _as_last_axis_matrix_apply


def _cheb_nodes_gl(N: int) -> np.ndarray:
    """Chebyshev–Gauss–Lobatto nodes y in [-1, 1], j=0..N-1.

    y_j = cos(pi * j / (N-1)).
    """
    if N < 2:
        return np.array([1.0])
    j = np.arange(N, dtype=float)
    return np.cos(np.pi * j / float(N - 1))


def _build_cheb_diff_matrix(N: int) -> np.ndarray:
    """Build Chebyshev first-derivative collocation matrix on y in [-1,1].

    Implements standard Trefethen formula for D on Chebyshev–Gauss–Lobatto points.
    """
    if N == 1:
        return np.zeros((1, 1), dtype=float)
    y = _cheb_nodes_gl(N)
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N))

    Y = y.reshape((N, 1))
    dY = Y - Y.T  # y_i - y_j
    D = (c.reshape((N, 1)) / c.reshape((1, N))) / (dY + np.eye(N))  # off-diagonals temp
    D = D - np.diag(np.diag(D))  # zero diagonal
    D = D + np.diag(-np.sum(D, axis=1))  # set diagonal so rows sum to zero

    return D


class ChebyshevBasis1D(Basis1D):
    """Chebyshev–Gauss–Lobatto basis on [0, Lx] using DCT-I for transforms.

    - Nodes: x_j = (1 - y_j) * Lx / 2, y_j = cos(pi j/(N-1))
    - forward/inverse: DCT-I/idct-I (orthonormal) along last axis
    - derivatives: computed in spectral space via Chebyshev coefficient
      recurrence and evaluated via IDCT (O(N log N)).
    """

    def __init__(self, N: int, Lx: float, *, bc: str = "dirichlet", fft_workers: int = 1) -> None:
        super().__init__(N=N, Lx=Lx, bc=bc)
        self.fft_workers = int(fft_workers)

        # Nodes and mapping
        self._y = _cheb_nodes_gl(self.N)
        self._x = (1.0 - self._y) * (self.Lx / 2.0)

        # Precompute scaling from dy to dx
        self._dy_dx = 2.0 / self.Lx

        if _FFT_CACHE and fftw_cache is not None:
            try:
                fftw_cache.enable()
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

    # Grid and weights
    def nodes(self) -> np.ndarray:
        return self._x

    def quadrature_weights(self) -> Optional[np.ndarray]:
        """Clenshaw–Curtis quadrature weights mapped to [0, Lx].

        Uses O(N^2) stable formula. For large N this can be replaced by
        O(N log N) FFT-based construction if needed.
        """
        N = self.N
        if N == 1:
            return np.array([self.Lx])

        w = np.zeros(N, dtype=float)
        n = N - 1
        for j in range(N):
            s = 0.0
            for k in range(1, n // 1 + 1):
                # Only odd k contribute in closed-form expression
                if k % 2 == 1:
                    s += (1.0 / (1.0 - (k * k))) * np.cos(np.pi * k * j / n)
            w[j] = (2.0 / n) * (1.0 - 2.0 * s)
        # Map from [-1,1] to [0,Lx]: dx = (Lx/2) dy
        w *= (self.Lx / 2.0)
        return w

    # --- Torch helpers for DCT-I (orthonormal) ---
    @staticmethod
    def _torch_scales(n: int, *, device, dtype):
        # D_in (pre-scale columns / input samples), D_out (post-scale rows / output coeffs)
        # These diagonals satisfy: A_ortho = diag(D_out) @ A_none @ diag(D_in)
        # where A_none is DCT-I with norm=None built via mirrored rFFT.
        if n <= 0:
            raise ValueError("Invalid size for DCT-I scales")
        dn = torch.full((n,), 0.5, device=device, dtype=dtype)
        if n >= 2:
            sqrt2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
            dn[0] = 1.0 / sqrt2
            dn[-1] = 1.0 / sqrt2
        dout = torch.full((n,), float(np.sqrt(2.0 / max(n - 1, 1))), device=device, dtype=dtype)
        if n >= 2:
            dout[0] = float(np.sqrt(1.0 / (n - 1)))
            dout[-1] = float(np.sqrt(1.0 / (n - 1)))
        return dn, dout

    @staticmethod
    def _dct1_ortho_torch(x):
        assert torch is not None
        n = x.shape[-1]
        if n == 1:
            # Orthonormal DCT-I on length-1 is identity
            return x.clone()
        dn, dout = ChebyshevBasis1D._torch_scales(n, device=x.device, dtype=x.dtype)
        x_scaled = x * dn
        # Mirror without duplicating endpoints: [x0, x1, ..., xN-1, xN-2, ..., x1]
        tail = x_scaled[..., 1:-1]
        if tail.shape[-1] > 0:
            tail = torch.flip(tail, dims=(-1,))
        y = torch.cat([x_scaled, tail], dim=-1)
        c_none = torch.fft.rfft(y, dim=-1).real
        return c_none * dout

    @staticmethod
    def _idct1_ortho_torch(c):
        assert torch is not None
        n = c.shape[-1]
        if n == 1:
            return c.clone()
        # Undo output scaling, then inverse of norm=None via mirrored irfft
        _, dout = ChebyshevBasis1D._torch_scales(n, device=c.device, dtype=c.dtype)
        c_none = c / dout
        y = torch.fft.irfft(torch.complex(c_none, torch.zeros_like(c_none)), n=max(2 * n - 2, 1), dim=-1)
        x_scaled = y[..., :n]
        dn, _ = ChebyshevBasis1D._torch_scales(n, device=c.device, dtype=c.dtype)
        # Undo input scaling
        return x_scaled / dn

    # Transforms (orthonormal DCT-I pair)
    def forward(self, f: np.ndarray) -> np.ndarray:
        # Torch path
        if _TORCH_AVAILABLE and isinstance(f, (torch.Tensor,)):
            return self._dct1_ortho_torch(f)
        # NumPy/SciPy path
        return scipy_fft.dct(f, type=1, axis=-1, norm="ortho")

    def inverse(self, F: np.ndarray) -> np.ndarray:
        if _TORCH_AVAILABLE and isinstance(F, (torch.Tensor,)):
            return self._idct1_ortho_torch(F)
        return scipy_fft.idct(F, type=1, axis=-1, norm="ortho")

    # Derivatives
    def dx(self, f: np.ndarray) -> np.ndarray:
        # Compute Chebyshev coefficients of f
        a = self.forward(f)
        N = a.shape[-1]
        if N <= 1:
            if _TORCH_AVAILABLE and isinstance(f, (torch.Tensor,)):
                return torch.zeros_like(f)
            return np.zeros_like(f)
        # Differentiate coefficients w.r.t y via recurrence (real-valued)
        if _TORCH_AVAILABLE and isinstance(a, (torch.Tensor,)):
            b = torch.zeros_like(a)
            if N >= 2:
                b[..., -2] = 2.0 * (N - 1) * a[..., -1]
                for k in range(N - 3, -1, -1):
                    b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
                b[..., 0] = 0.5 * b[..., 0]
            b = b * float(self._dy_dx)
            return self.inverse(b)
        else:
            b = np.zeros_like(a)
            if N >= 2:
                b[..., -2] = 2.0 * (N - 1) * a[..., -1]
                for k in range(N - 3, -1, -1):
                    b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
                b[..., 0] *= 0.5
            b *= self._dy_dx
            return self.inverse(b)

    def d2x(self, f: np.ndarray) -> np.ndarray:
        # Differentiate twice via spectral recurrence
        a = self.forward(f)
        N = a.shape[-1]
        if N <= 2:
            if _TORCH_AVAILABLE and isinstance(f, (torch.Tensor,)):
                return torch.zeros_like(f)
            return np.zeros_like(f)
        if _TORCH_AVAILABLE and isinstance(a, (torch.Tensor,)):
            b = torch.zeros_like(a)
            b[..., -2] = 2.0 * (N - 1) * a[..., -1]
            for k in range(N - 3, -1, -1):
                b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
            b[..., 0] = 0.5 * b[..., 0]
            c = torch.zeros_like(a)
            c[..., -2] = 2.0 * (N - 1) * b[..., -1]
            for k in range(N - 3, -1, -1):
                c[..., k] = c[..., k + 2] + 2.0 * (k + 1) * b[..., k + 1]
            c[..., 0] = 0.5 * c[..., 0]
            c = c * float(self._dy_dx ** 2)
            return self.inverse(c)
        else:
            b = np.zeros_like(a)
            b[..., -2] = 2.0 * (N - 1) * a[..., -1]
            for k in range(N - 3, -1, -1):
                b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
            b[..., 0] *= 0.5
            c = np.zeros_like(a)
            c[..., -2] = 2.0 * (N - 1) * b[..., -1]
            for k in range(N - 3, -1, -1):
                c[..., k] = c[..., k + 2] + 2.0 * (k + 1) * b[..., k + 1]
            c[..., 0] *= 0.5
            c *= (self._dy_dx ** 2)
            return self.inverse(c)


