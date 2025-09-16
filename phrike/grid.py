from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

try:
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch may not be installed
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore

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
    cutoff = N // 3
    mask = (np.abs(k) <= cutoff).astype(float)
    return mask


def _build_exponential_filter(N: int, p: int, alpha: float) -> np.ndarray:
    # High-order exponential filter: sigma(k) = exp(-alpha * (|k|/kmax)^p)
    k = np.fft.fftfreq(N) * N
    kmax = np.max(np.abs(k)) if N > 0 else 1.0
    eta = np.abs(k) / max(kmax, 1.0)
    sigma = np.exp(-alpha * eta**p)
    return sigma


def _build_filter_mask_2d(Nx: int, Ny: int, dealias: bool) -> np.ndarray:
    if not dealias:
        return np.ones((Ny, Nx), dtype=float)
    kx = (np.fft.fftfreq(Nx) * Nx).astype(float)
    ky = (np.fft.fftfreq(Ny) * Ny).astype(float)
    cutoff_x = Nx // 3
    cutoff_y = Ny // 3
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
    sigma_x = np.exp(-alpha * eta_x**p)
    sigma_y = np.exp(-alpha * eta_y**p)
    return sigma_y[:, None] * sigma_x[None, :]


def _build_filter_mask_3d(Nx: int, Ny: int, Nz: int, dealias: bool) -> np.ndarray:
    if not dealias:
        return np.ones((Nz, Ny, Nx), dtype=float)
    kx = (np.fft.fftfreq(Nx) * Nx).astype(float)
    ky = (np.fft.fftfreq(Ny) * Ny).astype(float)
    kz = (np.fft.fftfreq(Nz) * Nz).astype(float)
    cutoff_x = Nx // 3
    cutoff_y = Ny // 3
    cutoff_z = Nz // 3
    mask_x = (np.abs(kx) <= cutoff_x).astype(float)
    mask_y = (np.abs(ky) <= cutoff_y).astype(float)
    mask_z = (np.abs(kz) <= cutoff_z).astype(float)
    return mask_z[:, None, None] * mask_y[None, :, None] * mask_x[None, None, :]


def _build_exponential_filter_3d(
    Nx: int, Ny: int, Nz: int, p: int, alpha: float
) -> np.ndarray:
    kx = (np.fft.fftfreq(Nx) * Nx).astype(float)
    ky = (np.fft.fftfreq(Ny) * Ny).astype(float)
    kz = (np.fft.fftfreq(Nz) * Nz).astype(float)
    kxmax = np.max(np.abs(kx)) if Nx > 0 else 1.0
    kymax = np.max(np.abs(ky)) if Ny > 0 else 1.0
    kzmax = np.max(np.abs(kz)) if Nz > 0 else 1.0
    eta_x = np.abs(kx) / max(kxmax, 1.0)
    eta_y = np.abs(ky) / max(kymax, 1.0)
    eta_z = np.abs(kz) / max(kzmax, 1.0)
    sigma_x = np.exp(-alpha * eta_x**p)
    sigma_y = np.exp(-alpha * eta_y**p)
    sigma_z = np.exp(-alpha * eta_z**p)
    return sigma_z[:, None, None] * sigma_y[None, :, None] * sigma_x[None, None, :]


@dataclass
class Grid1D:
    """1D grid and spectral operators with pluggable basis.

    Default is periodic Fourier basis on a uniform grid, preserving existing
    behavior. Optionally, a Chebyshev basis with non-periodic BCs can be used.

    Attributes
    ----------
    N : int
        Number of spatial points.
    Lx : float
        Domain length.
    basis : str
        Spectral basis: "fourier" (default) or "chebyshev".
    bc : Optional[str]
        Boundary condition for non-periodic bases (e.g., "dirichlet").
    dealias : bool
        If True, apply 2/3-rule dealiasing mask in spectral space (Fourier only).
    filter_params : Optional[Dict[str, float]]
        Optional exponential filter configuration with keys:
        - enabled: bool
        - p: even integer (>= 2)
        - alpha: filter strength
    """

    N: int
    Lx: float
    basis: str = "fourier"
    bc: Optional[str] = None
    dealias: bool = True
    filter_params: Optional[Dict[str, float]] = None
    fft_workers: int = 1
    backend: str = "numpy"  # "numpy" or "torch"
    torch_device: Optional[str] = None

    def __post_init__(self) -> None:
        # Basis selection (Fourier default)
        self._basis_name = str(self.basis).lower()

        if self._basis_name == "fourier":
            # Existing periodic uniform grid path
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

            # Initialize Fourier basis instance (NumPy path used inside helpers)
            try:
                from .basis.fourier import FourierBasis1D

                self._basis = FourierBasis1D(self.N, self.Lx, fft_workers=self.fft_workers)
            except Exception:
                self._basis = None  # Fallback to direct FFT wrappers below
        elif self._basis_name == "chebyshev":
            # Chebyshev–Gauss–Lobatto nodes on [0, Lx]
            try:
                from .basis.chebyshev import ChebyshevBasis1D

                bc = self.bc or "dirichlet"
                self._basis = ChebyshevBasis1D(self.N, self.Lx, bc=bc, fft_workers=self.fft_workers)
                self.x = self._basis.nodes()
                # Use min spacing as stability proxy for CFL
                x_np = np.asarray(self.x)
                if self.N > 1:
                    self.dx = float(np.min(np.diff(x_np)))
                else:
                    self.dx = self.Lx
                # No Fourier dealias/filter masks in Chebyshev mode
                self.dealias_mask = np.ones(self.N, dtype=float)
                self.filter_sigma = np.ones(self.N, dtype=float)
            except Exception as e:
                raise RuntimeError(f"Chebyshev basis initialization failed: {e}")
        else:
            raise ValueError(f"Unknown basis: {self.basis}")

        # Enable FFTW planning cache if available (no-op with SciPy backend)
        if scipy_fft_cache_enabled and fftw_cache is not None:
            try:
                fftw_cache.enable()
                # Use a reasonable cache size to avoid repeated planning
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

        # Torch backend setup (Fourier and Chebyshev supported)
        self._use_torch = (self.backend.lower() == "torch") and _TORCH_AVAILABLE
        if self._use_torch:
            # Choose device if not provided
            if self.torch_device is None:
                dev = "cpu"
                try:
                    if (
                        torch is not None
                        and hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        dev = "mps"
                    elif torch is not None and torch.cuda.is_available():
                        dev = "cuda"
                except Exception:
                    dev = "cpu"
                self.torch_device = dev

            # Choose dtype based on device (MPS only supports float32)
            torch_dtype = torch.float32 if self.torch_device == "mps" else torch.float64
            torch_cdtype = (
                torch.complex64 if self.torch_device == "mps" else torch.complex128
            )

            # Move arrays to torch
            assert torch is not None
            self.x = torch.from_numpy(np.asarray(self.x)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            if self._basis_name == "fourier":
                k_t = torch.from_numpy(np.asarray(self.k)).to(
                    dtype=torch_dtype, device=self.torch_device
                )
                self.k = k_t
                self.ik = k_t.to(dtype=torch_cdtype) * (1j)
                self.dealias_mask = torch.from_numpy(np.asarray(self.dealias_mask)).to(
                    dtype=torch_dtype, device=self.torch_device
                )
                self.filter_sigma = torch.from_numpy(np.asarray(self.filter_sigma)).to(
                    dtype=torch_dtype, device=self.torch_device
                )

    # FFT wrappers (Fourier basis only)
    def rfft(self, f: np.ndarray) -> np.ndarray:
        # FFT along the last axis to support batched inputs (..., N)
        if getattr(self, "_use_torch", False):
            assert torch is not None
            return torch.fft.fft(f, dim=-1)
        try:
            return scipy_fft.fft(f, axis=-1, workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.fft(f, axis=-1)

    def irfft(self, F: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_torch", False):
            assert torch is not None
            return torch.fft.ifft(F, dim=-1).real
        try:
            return scipy_fft.ifft(F, axis=-1, workers=self.fft_workers).real  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.ifft(F, axis=-1).real

    # Spectral derivative
    def dx1(self, f: np.ndarray) -> np.ndarray:
        # Supports f with shape (..., N)
        if self._basis_name == "fourier":
            F = self.rfft(f)
            F *= self.dealias_mask
            F *= self.filter_sigma
            dF = self.ik * F
            return self.irfft(dF)
        else:
            # Chebyshev basis path (NumPy only)
            return self._basis.dx(f)  # type: ignore[union-attr]

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        # Supports f with shape (..., N)
        if self._basis_name == "fourier":
            F = self.rfft(f)
            F *= self.dealias_mask
            F *= self.filter_sigma
            return self.irfft(F)
        else:
            # Apply exponential modal filter if configured
            if self.filter_params and bool(self.filter_params.get("enabled", False)):
                p = int(self.filter_params.get("p", 8))
                alpha = float(self.filter_params.get("alpha", 36.0))
                return self._basis.apply_spectral_filter(f, p=p, alpha=alpha)  # type: ignore[union-attr]
            return f

    def convolve(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Compute product in physical space, with dealiased spectral filtering.

        For pseudo-spectral nonlinearity, compute pointwise product then
        apply dealiasing/filter on the spectral representation.
        """
        h = f * g
        return self.apply_spectral_filter(h)

    # --- Boundary conditions (1D) ---
    def apply_boundary_conditions(self, U: np.ndarray, equations) -> np.ndarray:
        """Apply boundary conditions to conservative variables U in-place.

        For Chebyshev basis with reflective (wall) or Dirichlet BCs, we set
        the velocity to zero at the endpoints (x=0 and x=Lx), which implies
        zero momentum at the endpoints. Density and pressure are left unchanged.
        """
        if self._basis_name != "chebyshev":
            return U
        bc_type = (self.bc or "reflective").lower()
        if U.shape[-1] != self.N:
            return U
        if U.shape[0] < 2:
            return U
        # Enforce u=0 at boundaries => momentum = 0 at boundaries
        if bc_type in ("reflective", "dirichlet", "wall"):
            # Support batched leading dims (..., 3, N)
            # Only modify the momentum component index 1
            U[..., 1, 0] = 0.0
            U[..., 1, -1] = 0.0
        return U


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
    backend: str = "numpy"  # "numpy" or "torch"
    torch_device: Optional[str] = None

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
            self.filter_sigma = _build_exponential_filter_2d(
                self.Nx, self.Ny, p=p, alpha=alpha
            )

        if scipy_fft_cache_enabled and fftw_cache is not None:
            try:
                fftw_cache.enable()
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

        # Torch backend setup
        self._use_torch = (self.backend.lower() == "torch") and _TORCH_AVAILABLE
        if self._use_torch:
            assert torch is not None
            if self.torch_device is None:
                dev = "cpu"
                try:
                    if (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        dev = "mps"
                    elif torch.cuda.is_available():
                        dev = "cuda"
                except Exception:
                    dev = "cpu"
                self.torch_device = dev

            # Choose dtype based on device (MPS only supports float32)
            torch_dtype = torch.float32 if self.torch_device == "mps" else torch.float64
            torch_cdtype = (
                torch.complex64 if self.torch_device == "mps" else torch.complex128
            )

            self.x = torch.from_numpy(np.asarray(self.x)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.y = torch.from_numpy(np.asarray(self.y)).to(
                dtype=torch_dtype, device=self.torch_device
            )

            kx_t = torch.from_numpy(np.asarray(self.kx)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            ky_t = torch.from_numpy(np.asarray(self.ky)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.kx = kx_t
            self.ky = ky_t
            self.ikx = kx_t.to(dtype=torch_cdtype)[None, :] * (1j)
            self.iky = ky_t.to(dtype=torch_cdtype)[:, None] * (1j)
            self.dealias_mask = torch.from_numpy(np.asarray(self.dealias_mask)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.filter_sigma = torch.from_numpy(np.asarray(self.filter_sigma)).to(
                dtype=torch_dtype, device=self.torch_device
            )

    # 2D FFT wrappers
    def fft2(self, f: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_torch", False):
            assert torch is not None
            return torch.fft.fft2(f, dim=(-2, -1))
        try:
            return scipy_fft.fft2(f, axes=(-2, -1), workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.fft2(f, axes=(-2, -1))

    def ifft2(self, F: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_torch", False):
            assert torch is not None
            return torch.fft.ifft2(F, dim=(-2, -1)).real
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
        if getattr(self, "_use_torch", False):
            assert torch is not None
            # Convert to numpy for meshgrid, then back to torch
            x_np = self.x.detach().cpu().numpy()
            y_np = self.y.detach().cpu().numpy()
            X_np, Y_np = np.meshgrid(x_np, y_np, indexing="xy")
            # Convert back to torch with same dtype and device
            torch_dtype = torch.float32 if self.torch_device == "mps" else torch.float64
            X = torch.from_numpy(X_np).to(dtype=torch_dtype, device=self.torch_device)
            Y = torch.from_numpy(Y_np).to(dtype=torch_dtype, device=self.torch_device)
            return X, Y
        else:
            X, Y = np.meshgrid(self.x, self.y, indexing="xy")
            return X, Y


@dataclass
class Grid3D:
    """Uniform periodic 3D grid and spectral operators.

    The last three axes are spatial: (..., Nz, Ny, Nx)
    """

    Nx: int
    Ny: int
    Nz: int
    Lx: float
    Ly: float
    Lz: float
    dealias: bool = True
    filter_params: Optional[Dict[str, float]] = None
    fft_workers: int = 1
    backend: str = "numpy"  # "numpy" or "torch"
    torch_device: Optional[str] = None

    def __post_init__(self) -> None:
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        self.dz = self.Lz / self.Nz
        self.x = np.linspace(0.0, self.Lx, num=self.Nx, endpoint=False)
        self.y = np.linspace(0.0, self.Ly, num=self.Ny, endpoint=False)
        self.z = np.linspace(0.0, self.Lz, num=self.Nz, endpoint=False)

        # Wave numbers for derivatives
        kx_index = np.fft.fftfreq(self.Nx, d=self.dx)
        ky_index = np.fft.fftfreq(self.Ny, d=self.dy)
        kz_index = np.fft.fftfreq(self.Nz, d=self.dz)
        self.kx = 2.0 * np.pi * kx_index  # shape (Nx,)
        self.ky = 2.0 * np.pi * ky_index  # shape (Ny,)
        self.kz = 2.0 * np.pi * kz_index  # shape (Nz,)
        self.ikx = 1j * self.kx[None, None, :]  # broadcast along (Nz, Ny, Nx)
        self.iky = 1j * self.ky[None, :, None]
        self.ikz = 1j * self.kz[:, None, None]

        # Dealias and filter
        self.dealias_mask = _build_filter_mask_3d(
            self.Nx, self.Ny, self.Nz, self.dealias
        )
        self.filter_sigma = np.ones((self.Nz, self.Ny, self.Nx), dtype=float)
        if self.filter_params and bool(self.filter_params.get("enabled", False)):
            p = int(self.filter_params.get("p", 8))
            alpha = float(self.filter_params.get("alpha", 36.0))
            self.filter_sigma = _build_exponential_filter_3d(
                self.Nx, self.Ny, self.Nz, p=p, alpha=alpha
            )

        if scipy_fft_cache_enabled and fftw_cache is not None:
            try:
                fftw_cache.enable()
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

        # Torch backend setup
        self._use_torch = (self.backend.lower() == "torch") and _TORCH_AVAILABLE
        if self._use_torch:
            assert torch is not None
            if self.torch_device is None:
                dev = "cpu"
                try:
                    if (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
                        dev = "mps"
                    elif torch.cuda.is_available():
                        dev = "cuda"
                except Exception:
                    dev = "cpu"
                self.torch_device = dev

            torch_dtype = torch.float32 if self.torch_device == "mps" else torch.float64
            torch_cdtype = (
                torch.complex64 if self.torch_device == "mps" else torch.complex128
            )

            self.x = torch.from_numpy(np.asarray(self.x)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.y = torch.from_numpy(np.asarray(self.y)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.z = torch.from_numpy(np.asarray(self.z)).to(
                dtype=torch_dtype, device=self.torch_device
            )

            kx_t = torch.from_numpy(np.asarray(self.kx)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            ky_t = torch.from_numpy(np.asarray(self.ky)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            kz_t = torch.from_numpy(np.asarray(self.kz)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.kx = kx_t
            self.ky = ky_t
            self.kz = kz_t
            self.ikx = kx_t.to(dtype=torch_cdtype)[None, None, :] * (1j)
            self.iky = ky_t.to(dtype=torch_cdtype)[None, :, None] * (1j)
            self.ikz = kz_t.to(dtype=torch_cdtype)[:, None, None] * (1j)
            self.dealias_mask = torch.from_numpy(np.asarray(self.dealias_mask)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.filter_sigma = torch.from_numpy(np.asarray(self.filter_sigma)).to(
                dtype=torch_dtype, device=self.torch_device
            )

    # 3D FFT wrappers
    def fftn(self, f: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_torch", False):
            assert torch is not None
            return torch.fft.fftn(f, dim=(-3, -2, -1))
        try:
            return scipy_fft.fftn(f, axes=(-3, -2, -1), workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.fftn(f, axes=(-3, -2, -1))

    def ifftn(self, F: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_torch", False):
            assert torch is not None
            return torch.fft.ifftn(F, dim=(-3, -2, -1)).real
        try:
            return scipy_fft.ifftn(F, axes=(-3, -2, -1), workers=self.fft_workers).real  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.ifftn(F, axes=(-3, -2, -1)).real

    def _apply_masks(self, F: np.ndarray) -> np.ndarray:
        F *= self.dealias_mask
        F *= self.filter_sigma
        return F

    def dx1(self, f: np.ndarray) -> np.ndarray:
        F = self.fftn(f)
        F = self._apply_masks(F)
        dF = F * self.ikx
        return self.ifftn(dF)

    def dy1(self, f: np.ndarray) -> np.ndarray:
        F = self.fftn(f)
        F = self._apply_masks(F)
        dF = F * self.iky
        return self.ifftn(dF)

    def dz1(self, f: np.ndarray) -> np.ndarray:
        F = self.fftn(f)
        F = self._apply_masks(F)
        dF = F * self.ikz
        return self.ifftn(dF)

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        F = self.fftn(f)
        F = self._apply_masks(F)
        return self.ifftn(F)

    def xyz_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if getattr(self, "_use_torch", False):
            assert torch is not None
            x_np = self.x.detach().cpu().numpy()
            y_np = self.y.detach().cpu().numpy()
            z_np = self.z.detach().cpu().numpy()
            # Return arrays with shape (Nz, Ny, Nx) to align with solver axes (-3,-2,-1)
            Z_np, Y_np, X_np = np.meshgrid(z_np, y_np, x_np, indexing="ij")
            torch_dtype = torch.float32 if self.torch_device == "mps" else torch.float64
            X = torch.from_numpy(X_np).to(dtype=torch_dtype, device=self.torch_device)
            Y = torch.from_numpy(Y_np).to(dtype=torch_dtype, device=self.torch_device)
            Z = torch.from_numpy(Z_np).to(dtype=torch_dtype, device=self.torch_device)
            return X, Y, Z
        else:
            # Return arrays with shape (Nz, Ny, Nx)
            Z, Y, X = np.meshgrid(self.z, self.y, self.x, indexing="ij")
            return X, Y, Z
