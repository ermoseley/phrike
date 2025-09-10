from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # Prefer pyfftw if available for performance
    import pyfftw.interfaces.scipy_fft as scipy_fft  # type: ignore
    scipy_fft_cache_enabled = True
except Exception:  # Fallback to SciPy's FFT
    from scipy import fft as scipy_fft  # type: ignore
    scipy_fft_cache_enabled = False


def _build_filter_mask(N: int, dealias: bool) -> np.ndarray:
    if not dealias:
        return np.ones(N, dtype=float)
    # 2/3 rule: zero modes with |k| > N/3
    k = np.fft.fftfreq(N) * N
    cutoff = (N // 3)
    mask = (np.abs(k) <= cutoff).astype(float)
    return mask


def _build_exponential_filter(N: int, p: int, alpha: float) -> np.ndarray:
    # High-order exponential filter: sigma(k) = exp(-alpha * (|k|/kmax)^p)
    k = np.fft.fftfreq(N) * N
    kmax = np.max(np.abs(k)) if N > 0 else 1.0
    eta = np.abs(k) / max(kmax, 1.0)
    sigma = np.exp(-alpha * eta ** p)
    return sigma


@dataclass
class Grid1D:
    """Uniform periodic 1D grid and spectral operators.

    Attributes
    ----------
    N : int
        Number of spatial points.
    Lx : float
        Domain length.
    dealias : bool
        If True, apply 2/3-rule dealiasing mask in spectral space.
    filter_params : Optional[Dict[str, float]]
        Optional exponential filter configuration with keys:
        - enabled: bool
        - p: even integer (>= 2)
        - alpha: filter strength
    """

    N: int
    Lx: float
    dealias: bool = True
    filter_params: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        self.dx = self.Lx / self.N
        self.x = np.linspace(0.0, self.Lx, num=self.N, endpoint=False)
        # Physical-to-spectral wave numbers (2*pi/L scaling)
        k_index = np.fft.fftfreq(self.N, d=self.dx)  # cycles per length
        self.k = 2.0 * np.pi * k_index
        self.ik = 1j * self.k

        # Dealias mask and optional spectral filter
        self.dealias_mask = _build_filter_mask(self.N, self.dealias)

        self.filter_sigma = np.ones(self.N, dtype=float)
        if self.filter_params and bool(self.filter_params.get("enabled", False)):
            p = int(self.filter_params.get("p", 8))
            alpha = float(self.filter_params.get("alpha", 36.0))
            self.filter_sigma = _build_exponential_filter(self.N, p=p, alpha=alpha)

    # FFT wrappers
    def rfft(self, f: np.ndarray) -> np.ndarray:
        return scipy_fft.fft(f)

    def irfft(self, F: np.ndarray) -> np.ndarray:
        return scipy_fft.ifft(F).real

    # Spectral derivative
    def dx1(self, f: np.ndarray) -> np.ndarray:
        F = self.rfft(f)
        F *= self.dealias_mask
        F *= self.filter_sigma
        dF = self.ik * F
        return self.irfft(dF)

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        F = self.rfft(f)
        F *= self.dealias_mask
        F *= self.filter_sigma
        return self.irfft(F)

    def convolve(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Compute product in physical space, with dealiased spectral filtering.

        For pseudo-spectral nonlinearity, compute pointwise product then
        apply dealiasing/filter on the spectral representation.
        """
        h = f * g
        return self.apply_spectral_filter(h)


