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

from .base import Basis1D


class FourierBasis1D(Basis1D):
    """Periodic Fourier basis on a uniform grid using FFT.

    Nodes: equispaced x in [0, Lx).
    Derivatives: computed by multiplying by i*k in spectral space.
    """

    def __init__(self, N: int, Lx: float, *, fft_workers: int = 1) -> None:
        super().__init__(N=N, Lx=Lx, bc="periodic")
        self.fft_workers = int(fft_workers)

        # Nodes and wavenumbers
        self._x = np.linspace(0.0, self.Lx, num=self.N, endpoint=False)
        dx = self.Lx / self.N
        k_cycles = np.fft.fftfreq(self.N, d=dx)
        self._k = 2.0 * np.pi * k_cycles
        self._ik = 1j * self._k

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
        # Uniform trapezoidal on periodic uniform grid: dx weights
        dx = self.Lx / self.N
        return np.full(self.N, dx, dtype=float)

    # Transforms
    def forward(self, f: np.ndarray) -> np.ndarray:
        try:
            return scipy_fft.fft(f, axis=-1, workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.fft(f, axis=-1)

    def inverse(self, F: np.ndarray) -> np.ndarray:
        try:
            return scipy_fft.ifft(F, axis=-1, workers=self.fft_workers).real  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.ifft(F, axis=-1).real

    # Derivatives
    def dx(self, f: np.ndarray) -> np.ndarray:
        F = self.forward(f)
        dF = F * self._ik
        return self.inverse(dF)

    def d2x(self, f: np.ndarray) -> np.ndarray:
        F = self.forward(f)
        d2F = F * (-(self._k**2))
        return self.inverse(d2F)


