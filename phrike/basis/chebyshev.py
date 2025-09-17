from __future__ import annotations

from typing import Optional

import numpy as np

try:  # Prefer pyfftw if available for performance
    import pyfftw.interfaces.scipy_fft as scipy_fft  # type: ignore
    from pyfftw.interfaces import cache as fftw_cache  # type: ignore

    _FFT_CACHE = True
except Exception:  # Fallback to SciPy's FFT
    from scipy import fft as scipy_fft  # type: ignore

    _FFT_CACHE = False
    fftw_cache = None  # type: ignore

from .base import Basis1D, _as_last_axis_matrix_apply

# Optional torch support (used only for input/output conversion)
try:  # pragma: no cover - torch optional
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _cheb_nodes_gl(N: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1]."""
    if N == 1:
        return np.array([1.0])
    j = np.arange(N, dtype=float)
    return np.cos(np.pi * j / (N - 1))


class ChebyshevBasis1D(Basis1D):
    """Chebyshev basis on Chebyshev-Gauss-Lobatto nodes using DCT-I.

    Nodes: Chebyshev-Gauss-Lobatto points mapped to [0, Lx].
    Derivatives: computed via spectral recurrence relations.
    """

    def __init__(self, N: int, Lx: float, *, bc: str = "dirichlet", fft_workers: int = 1) -> None:
        super().__init__(N=N, Lx=Lx, bc=bc)
        self.fft_workers = int(fft_workers)

        # Chebyshev-Gauss-Lobatto nodes on [-1, 1]
        self._y = _cheb_nodes_gl(self.N)
        # Map to [0, Lx]: x = (1-y) * Lx/2
        self._x = (1.0 - self._y) * (self.Lx / 2.0)
        # Derivative scaling: dy/dx = 2/Lx
        self._dy_dx = 2.0 / self.Lx

        # Enable cache if using FFTW interface
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
        """Clenshaw-Curtis quadrature weights for Chebyshev-Gauss-Lobatto nodes."""
        N = self.N
        if N == 1:
            return np.array([self.Lx], dtype=float)
        
        # Clenshaw-Curtis weights on [-1, 1]
        w = np.zeros(N)
        for j in range(N):
            if j == 0 or j == N - 1:
                w[j] = 1.0 / (N - 1)
            else:
                # Sum over even cosine terms
                w[j] = 2.0 / (N - 1)
                for k in range(1, (N - 1) // 2 + 1):
                    w[j] += 2.0 / (N - 1) * np.cos(2.0 * np.pi * k * j / (N - 1)) / (1.0 - 4.0 * k * k)
        
        # Scale to [0, Lx]
        return w * (self.Lx / 2.0)

    # Transforms (orthonormal DCT-I pair)
    def forward(self, f: np.ndarray) -> np.ndarray:
        return scipy_fft.dct(f, type=1, axis=-1, norm="ortho")

    def inverse(self, F: np.ndarray) -> np.ndarray:
        return scipy_fft.idct(F, type=1, axis=-1, norm="ortho")

    # Derivatives
    def dx(self, f: np.ndarray) -> np.ndarray:
        # Correct Chebyshev spectral derivative using DCT-I with norm=None
        # 1) Convert nodal values -> standard Chebyshev coefficients a_k
        #    a = dct_none(f)/(N-1); a0 and a_{N-1} are halved
        # 2) Apply standard recurrence to get derivative coeffs b_k (w.r.t y)
        # 3) Evaluate derivative on nodes via IDCT-I (norm=None) using b_k with
        #    interior modes doubled in the cosine sum, then scale dy->dx.

        # Handle optional torch input
        is_torch = False
        if _TORCH_AVAILABLE and isinstance(f, (torch.Tensor,)):
            is_torch = True
            f_np = f.detach().cpu().numpy()
        else:
            f_np = np.asarray(f)

        N = f_np.shape[-1]
        if N <= 1:
            out = np.zeros_like(f_np)
            if is_torch:  # match input backend
                return torch.from_numpy(out).to(dtype=f.dtype, device=f.device)
            return out

        # Forward DCT-I (no normalization) to standard Chebyshev coefficients
        a = scipy_fft.dct(f_np, type=1, axis=-1, norm=None)
        a = a / (N - 1)
        a[..., 0] *= 0.5
        a[..., -1] *= 0.5

        # Recurrence for derivative coefficients (w.r.t y)
        b = np.zeros_like(a)
        if N >= 2:
            b[..., -2] = 2.0 * (N - 1) * a[..., -1]
            for k in range(N - 3, -1, -1):
                b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
            b[..., 0] *= 0.5

        # Build cosine-sum vector for IDCT-I (norm=None): double interior modes
        b_sum = b.copy()
        if N > 2:
            b_sum[..., 1:-1] *= 2.0

        # Inverse DCT-I (no normalization); divide by (N-1) to evaluate nodal values
        df_dy = scipy_fft.idct(b_sum, type=1, axis=-1, norm=None) / (N - 1)
        df_dx = df_dy * self._dy_dx

        if is_torch:
            return torch.from_numpy(df_dx).to(dtype=f.dtype, device=f.device)
        return df_dx

    def d2x(self, f: np.ndarray) -> np.ndarray:
        # Handle optional torch input
        is_torch = False
        if _TORCH_AVAILABLE and isinstance(f, (torch.Tensor,)):
            is_torch = True
            f_np = f.detach().cpu().numpy()
        else:
            f_np = np.asarray(f)

        N = f_np.shape[-1]
        if N <= 2:
            out = np.zeros_like(f_np)
            if is_torch:
                return torch.from_numpy(out).to(dtype=f.dtype, device=f.device)
            return out

        # Forward to standard Chebyshev coefficients (no normalization)
        a = scipy_fft.dct(f_np, type=1, axis=-1, norm=None)
        a = a / (N - 1)
        a[..., 0] *= 0.5
        a[..., -1] *= 0.5

        # First derivative coefficients (w.r.t y)
        b = np.zeros_like(a)
        b[..., -2] = 2.0 * (N - 1) * a[..., -1]
        for k in range(N - 3, -1, -1):
            b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
        b[..., 0] *= 0.5

        # Second derivative coefficients (w.r.t y)
        c = np.zeros_like(a)
        c[..., -2] = 2.0 * (N - 1) * b[..., -1]
        for k in range(N - 3, -1, -1):
            c[..., k] = c[..., k + 2] + 2.0 * (k + 1) * b[..., k + 1]
        c[..., 0] *= 0.5

        # Build cosine-sum vector for IDCT-I (norm=None): double interior modes
        c_sum = c.copy()
        if N > 2:
            c_sum[..., 1:-1] *= 2.0

        # Inverse DCT-I (no normalization); divide by (N-1) to evaluate nodal values
        d2f_dy2 = scipy_fft.idct(c_sum, type=1, axis=-1, norm=None) / (N - 1)
        d2f_dx2 = d2f_dy2 * (self._dy_dx ** 2)

        if is_torch:
            return torch.from_numpy(d2f_dx2).to(dtype=f.dtype, device=f.device)
        return d2f_dx2