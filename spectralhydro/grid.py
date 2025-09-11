from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:  # Prefer pyfftw if available for performance
    import pyfftw.interfaces.scipy_fft as scipy_fft  # type: ignore
    from pyfftw.interfaces import cache as fftw_cache  # type: ignore
    scipy_fft_cache_enabled = True
except Exception:  # Fallback to SciPy's FFT
    from scipy import fft as scipy_fft  # type: ignore
    scipy_fft_cache_enabled = False
    fftw_cache = None  # type: ignore


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


def _build_filter_mask_2d(Nx: int, Ny: int, dealias: bool) -> np.ndarray:
    if not dealias:
        return np.ones((Ny, Nx), dtype=float)
    kx = (np.fft.fftfreq(Nx) * Nx).astype(float)
    ky = (np.fft.fftfreq(Ny) * Ny).astype(float)
    cutoff_x = (Nx // 3)
    cutoff_y = (Ny // 3)
    mask_x = (np.abs(kx) <= cutoff_x).astype(float)
    mask_y = (np.abs(ky) <= cutoff_y).astype(float)
    return mask_y[:, None] * mask_x[None, :]


def _build_exponential_filter_2d(Nx: int, Ny: int, p: int, alpha: float) -> np.ndarray:
    kx = (np.fft.fftfreq(Nx) * Nx).astype(float)
    ky = (np.fft.fftfreq(Ny) * Ny).astype(float)
    kxmax = np.max(np.abs(kx)) if Nx > 0 else 1.0
    kymax = np.max(np.abs(ky)) if Ny > 0 else 1.0
    eta_x = np.abs(kx) / max(kxmax, 1.0)
    eta_y = np.abs(ky) / max(kymax, 1.0)
    sigma_x = np.exp(-alpha * eta_x ** p)
    sigma_y = np.exp(-alpha * eta_y ** p)
    return sigma_y[:, None] * sigma_x[None, :]


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
    fft_workers: int = 1

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

        # Enable FFTW planning cache if available (no-op with SciPy backend)
        if scipy_fft_cache_enabled and fftw_cache is not None:
            try:
                fftw_cache.enable()
                # Use a reasonable cache size to avoid repeated planning
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

    # FFT wrappers
    def rfft(self, f: np.ndarray) -> np.ndarray:
        # FFT along the last axis to support batched inputs (..., N)
        try:
            return scipy_fft.fft(f, axis=-1, workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.fft(f, axis=-1)

    def irfft(self, F: np.ndarray) -> np.ndarray:
        try:
            return scipy_fft.ifft(F, axis=-1, workers=self.fft_workers).real  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.ifft(F, axis=-1).real

    # Spectral derivative
    def dx1(self, f: np.ndarray) -> np.ndarray:
        # Supports f with shape (..., N)
        F = self.rfft(f)
        F *= self.dealias_mask
        F *= self.filter_sigma
        dF = self.ik * F
        return self.irfft(dF)

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        # Supports f with shape (..., N)
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


@dataclass
class Grid2D:
    """Uniform periodic 2D grid and spectral operators.

    The last two axes are spatial: (..., Ny, Nx)
    """

    Nx: int
    Ny: int
    Lx: float
    Ly: float
    dealias: bool = True
    filter_params: Optional[Dict[str, float]] = None
    fft_workers: int = 1

    def __post_init__(self) -> None:
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.x = np.linspace(0.0, self.Lx, num=self.Nx, endpoint=False)
        self.y = np.linspace(0.0, self.Ly, num=self.Ny, endpoint=False)

        # Wave numbers for derivatives
        kx_index = np.fft.fftfreq(self.Nx, d=self.dx)
        ky_index = np.fft.fftfreq(self.Ny, d=self.dy)
        self.kx = 2.0 * np.pi * kx_index  # shape (Nx,)
        self.ky = 2.0 * np.pi * ky_index  # shape (Ny,)
        self.ikx = 1j * self.kx[None, :]  # shape (1, Nx)
        self.iky = 1j * self.ky[:, None]  # shape (Ny, 1)

        # Dealias and filter
        self.dealias_mask = _build_filter_mask_2d(self.Nx, self.Ny, self.dealias)
        self.filter_sigma = np.ones((self.Ny, self.Nx), dtype=float)
        if self.filter_params and bool(self.filter_params.get("enabled", False)):
            p = int(self.filter_params.get("p", 8))
            alpha = float(self.filter_params.get("alpha", 36.0))
            self.filter_sigma = _build_exponential_filter_2d(self.Nx, self.Ny, p=p, alpha=alpha)

        if scipy_fft_cache_enabled and fftw_cache is not None:
            try:
                fftw_cache.enable()
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

    # 2D FFT wrappers
    def fft2(self, f: np.ndarray) -> np.ndarray:
        try:
            return scipy_fft.fft2(f, axes=(-2, -1), workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.fft2(f, axes=(-2, -1))

    def ifft2(self, F: np.ndarray) -> np.ndarray:
        try:
            return scipy_fft.ifft2(F, axes=(-2, -1), workers=self.fft_workers).real  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.ifft2(F, axes=(-2, -1)).real

    def _apply_masks(self, F: np.ndarray) -> np.ndarray:
        F *= self.dealias_mask
        F *= self.filter_sigma
        return F

    def dx1(self, f: np.ndarray) -> np.ndarray:
        # derivative along x on last spatial axis
        F = self.fft2(f)
        F = self._apply_masks(F)
        dF = F * self.ikx  # broadcast over (Ny, Nx)
        return self.ifft2(dF)

    def dy1(self, f: np.ndarray) -> np.ndarray:
        # derivative along y on second-to-last spatial axis
        F = self.fft2(f)
        F = self._apply_masks(F)
        dF = F * self.iky
        return self.ifft2(dF)

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        F = self.fft2(f)
        F = self._apply_masks(F)
        return self.ifft2(F)

    def xy_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        X, Y = np.meshgrid(self.x, self.y, indexing="xy")
        return X, Y

