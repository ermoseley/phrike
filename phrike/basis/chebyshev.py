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
        # Compute Chebyshev coefficients of f
        a = self.forward(f)
        N = a.shape[-1]
        if N <= 1:
            return np.zeros_like(f)
        
        # Differentiate coefficients w.r.t y via recurrence
        b = np.zeros_like(a)
        if N >= 2:
            b[..., -2] = 2.0 * (N - 1) * a[..., -1]
            for k in range(N - 3, -1, -1):
                b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
            b[..., 0] *= 0.5
        
        # Map dy->dx
        b *= self._dy_dx
        return self.inverse(b)

    def d2x(self, f: np.ndarray) -> np.ndarray:
        # Differentiate twice via spectral recurrence
        a = self.forward(f)
        N = a.shape[-1]
        if N <= 2:
            return np.zeros_like(f)
        
        # First derivative coeffs (w.r.t y)
        b = np.zeros_like(a)
        if N >= 2:
            b[..., -2] = 2.0 * (N - 1) * a[..., -1]
            for k in range(N - 3, -1, -1):
                b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
            b[..., 0] *= 0.5
        
        # Second derivative coeffs: apply recurrence to b
        c = np.zeros_like(a)
        if N >= 2:
            c[..., -2] = 2.0 * (N - 1) * b[..., -1]
            for k in range(N - 3, -1, -1):
                c[..., k] = c[..., k + 2] + 2.0 * (k + 1) * b[..., k + 1]
            c[..., 0] *= 0.5
        
        # Map (dy)^2 -> (dx)^2 factor
        c *= (self._dy_dx ** 2)
        return self.inverse(c)