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


def _validate_torch_device(device: str, debug: bool) -> None:
    """Validate that the requested torch device is available.
    
    Args:
        device: The requested device ('cpu', 'cuda', 'mps')
        debug: If True, raise error if device is not available
        
    Raises:
        RuntimeError: If debug=True and device is not available
    """
    if not debug:
        return
        
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Debug mode: CUDA requested but not available. PyTorch was not compiled with CUDA support.")
    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError(f"Debug mode: MPS requested but not available. MPS is only available on Apple Silicon Macs.")
    elif device not in ["cpu", "cuda", "mps"]:
        raise RuntimeError(f"Debug mode: Unknown device '{device}' requested. Valid devices are: cpu, cuda, mps")


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
    precision: str = "double"  # "single" or "double"
    debug: bool = False

    def __post_init__(self) -> None:
        # Normalize/parse boundary condition configuration (string or dict)
        def _normalize_bc_value(name: str) -> str:
            n = str(name).strip().lower()
            if n in ("reflective", "dirichlet", "wall"):
                return "wall"
            if n in ("neumann", "open"):
                return "neumann"
            return n

        def _normalize_var_name(v: str) -> str:
            n = str(v).strip().lower()
            if n in ("rho", "density"):
                return "density"
            if n in ("p", "pressure"):
                return "pressure"
            if n in ("u", "velocity", "mom", "momentum", "momentum_x"):
                return "momentum"
            return n

        # bc may be a string or a dict mapping variables to bc types
        self._bc_global: str | None = None
        self._bc_map: dict[str, str] | None = None
        self._dirichlet_values_map: dict[str, tuple[float, float]] | None = None
        if isinstance(self.bc, dict):
            # Per-variable map
            self._bc_map = {}
            for k, v in self.bc.items():  # type: ignore[union-attr]
                self._bc_map[_normalize_var_name(k)] = _normalize_bc_value(v)
        elif isinstance(self.bc, str) and self.bc:
            self._bc_global = _normalize_bc_value(self.bc)
        else:
            self._bc_global = None

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
        elif self._basis_name == "legendre":
            # Legendre–Gauss–Lobatto nodes on [0, Lx]
            try:
                from .basis.legendre import LegendreLobattoBasis1D

                bc = self.bc or "dirichlet"
                self._basis = LegendreLobattoBasis1D(self.N, self.Lx, bc=bc)
                self.x = self._basis.nodes()
                x_np = np.asarray(self.x)
                if self.N > 1:
                    self.dx = float(np.min(np.diff(x_np)))
                else:
                    self.dx = self.Lx
                # No Fourier masks in non-periodic modal bases
                self.dealias_mask = np.ones(self.N, dtype=float)
                self.filter_sigma = np.ones(self.N, dtype=float)
            except Exception as e:
                raise RuntimeError(f"Legendre basis initialization failed: {e}")
        else:
            raise ValueError(f"Unknown basis: {self.basis}")

    def set_dirichlet_bc_values(self, values: dict[str, tuple[float, float]]) -> None:
        """Set per-variable Dirichlet boundary values.

        Args:
            values: Mapping from variable name to (left_value, right_value). Variable
                names use normalized keys: 'density', 'pressure', 'momentum'.
        """
        # Normalize keys
        def _n(v: str) -> str:
            vn = str(v).strip().lower()
            if vn in ("rho", "density"):
                return "density"
            if vn in ("p", "pressure"):
                return "pressure"
            if vn in ("u", "velocity", "mom", "momentum", "momentum_x"):
                return "momentum"
            return vn
        self._dirichlet_values_map = { _n(k): (float(v[0]), float(v[1])) for k, v in values.items() }

        # Enable FFTW planning cache if available (no-op with SciPy backend)
        if scipy_fft_cache_enabled and fftw_cache is not None:
            try:
                fftw_cache.enable()
                # Use a reasonable cache size to avoid repeated planning
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

        # Torch backend setup (Fourier, Chebyshev, Legendre supported)
        self._use_torch = (
            (self.backend.lower() == "torch")
            and _TORCH_AVAILABLE
            and (self._basis_name in ("fourier", "chebyshev", "legendre"))
        )
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
            
            # Debug mode: validate that the requested device is available
            if self.torch_device is not None:
                _validate_torch_device(self.torch_device, self.debug)

            # Choose dtype based on precision and device
            if self.precision == "single":
                torch_dtype = torch.float32
                torch_cdtype = torch.complex64
            elif self.precision == "double":
                torch_dtype = torch.float64
                torch_cdtype = torch.complex128
            else:
                # Fallback to device-based logic (MPS only supports float32)
                torch_dtype = torch.float32 if self.torch_device == "mps" else torch.float64
                torch_cdtype = (
                    torch.complex64 if self.torch_device == "mps" else torch.complex128
                )

            # Move arrays to torch (including CPU device)
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
        else:
            # Keep as numpy arrays for CPU backend
            if self._basis_name == "fourier":
                self.k = np.asarray(self.k)
                self.ik = self.k * (1j)
                self.dealias_mask = np.asarray(self.dealias_mask)
                self.filter_sigma = np.asarray(self.filter_sigma)

    # FFT wrappers (Fourier basis only)
    def rfft(self, f: np.ndarray) -> np.ndarray:
        # FFT along the last axis to support batched inputs (..., N)
        if getattr(self, "_use_torch", False):
            assert torch is not None
            # Convert to torch if needed
            if not isinstance(f, torch.Tensor):
                f = torch.from_numpy(np.asarray(f)).to(
                    dtype=self.x.dtype, device=self.x.device
                )
            return torch.fft.fft(f, dim=-1)
        try:
            return scipy_fft.fft(f, axis=-1, workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.fft(f, axis=-1)

    def irfft(self, F: np.ndarray) -> np.ndarray:
        if getattr(self, "_use_torch", False):
            assert torch is not None
            # Convert to torch if needed
            if not isinstance(F, torch.Tensor):
                F = torch.from_numpy(np.asarray(F)).to(
                    dtype=self.x.dtype, device=self.x.device
                )
            return torch.fft.ifft(F, dim=-1).real
        try:
            return scipy_fft.ifft(F, axis=-1, workers=self.fft_workers).real  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.ifft(F, axis=-1).real

    # DCT wrappers (Chebyshev basis only)
    def dct(self, f: np.ndarray) -> np.ndarray:
        """DCT-I forward transform for Chebyshev basis."""
        if getattr(self, "_use_torch", False):
            assert torch is not None
            # Convert to torch if needed
            if not isinstance(f, torch.Tensor):
                f = torch.from_numpy(np.asarray(f)).to(
                    dtype=self.x.dtype, device=self.x.device
                )
            return self._dct1_ortho_torch(f)
        try:
            return scipy_fft.dct(f, type=1, axis=-1, norm="ortho", workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.dct(f, type=1, axis=-1, norm="ortho")

    def idct(self, F: np.ndarray) -> np.ndarray:
        """DCT-I inverse transform for Chebyshev basis."""
        if getattr(self, "_use_torch", False):
            assert torch is not None
            # Convert to torch if needed
            if not isinstance(F, torch.Tensor):
                F = torch.from_numpy(np.asarray(F)).to(
                    dtype=self.x.dtype, device=self.x.device
                )
            return self._idct1_ortho_torch(F)
        try:
            return scipy_fft.idct(F, type=1, axis=-1, norm="ortho", workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.idct(F, type=1, axis=-1, norm="ortho")

    def _dct1_ortho_torch(self, x):
        """Torch implementation of orthonormal DCT-I."""
        assert torch is not None
        n = x.shape[-1]
        if n == 1:
            return x.clone()
        
        # DCT-I orthonormal scaling factors
        dn, dout = self._torch_dct_scales(n, device=x.device, dtype=x.dtype)
        
        # Pre-scale input
        x_scaled = x * dn
        # Mirror without duplicating endpoints: [x0, x1, ..., xN-1, xN-2, ..., x1]
        tail = x_scaled[..., 1:-1]
        if tail.shape[-1] > 0:
            tail = torch.flip(tail, dims=(-1,))
        y = torch.cat([x_scaled, tail], dim=-1)
        
        # Real FFT
        c_none = torch.fft.rfft(y, dim=-1).real
        return c_none * dout

    def _idct1_ortho_torch(self, c):
        """Torch implementation of orthonormal IDCT-I."""
        assert torch is not None
        n = c.shape[-1]
        if n == 1:
            return c.clone()
        
        # Undo output scaling
        _, dout = self._torch_dct_scales(n, device=c.device, dtype=c.dtype)
        c_none = c / dout
        
        # Inverse via mirrored irfft
        y = torch.fft.irfft(torch.complex(c_none, torch.zeros_like(c_none)), n=max(2 * n - 2, 1), dim=-1)
        x_scaled = y[..., :n]
        
        # Undo input scaling
        dn, _ = self._torch_dct_scales(n, device=c.device, dtype=c.dtype)
        return x_scaled / dn

    def _torch_dct_scales(self, n: int, *, device, dtype):
        """Compute DCT-I orthonormal scaling factors for torch."""
        assert torch is not None
        # D_in (pre-scale columns / input samples), D_out (post-scale rows / output coeffs)
        # These diagonals satisfy: A_ortho = diag(D_out) @ A_none @ diag(D_in)
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

    def _chebyshev_dx(self, f: np.ndarray) -> np.ndarray:
        """Chebyshev spectral derivative using DCT methods (mirrors Fourier pattern)."""
        # Forward transform
        a = self.dct(f)
        N = a.shape[-1]
        if N <= 1:
            if getattr(self, "_use_torch", False) and isinstance(f, (torch.Tensor,)):
                return torch.zeros_like(f)
            return np.zeros_like(f)
        
        # Differentiate coefficients w.r.t y via recurrence
        if getattr(self, "_use_torch", False) and isinstance(a, (torch.Tensor,)):
            assert torch is not None
            b = torch.zeros_like(a)
            if N >= 2:
                b[..., -2] = 2.0 * (N - 1) * a[..., -1]
                for k in range(N - 3, -1, -1):
                    b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
                b[..., 0] = 0.5 * b[..., 0]
            # Map dy->dx
            b = b * float(self._basis._dy_dx)
            return self.idct(b)
        else:
            b = np.zeros_like(a)
            if N >= 2:
                b[..., -2] = 2.0 * (N - 1) * a[..., -1]
                for k in range(N - 3, -1, -1):
                    b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
                b[..., 0] *= 0.5
            # Map dy->dx
            b *= self._basis._dy_dx
            return self.idct(b)

    def _chebyshev_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        """Chebyshev spectral filter using DCT methods (mirrors Fourier pattern)."""
        # Forward transform
        F = self.dct(f)
        N = F.shape[-1]
        
        # Build filter in spectral space
        k = np.arange(N, dtype=float)
        kmax = max(float(N - 1), 1.0)
        eta = k / kmax
        sigma_np = np.exp(-float(alpha) * eta**int(p))
        
        # Match backend of F (torch or numpy)
        try:
            import torch  # type: ignore
            is_torch = isinstance(F, torch.Tensor)
        except Exception:
            is_torch = False
            torch = None  # type: ignore
            
        if is_torch:  # type: ignore[truthy-bool]
            sigma = torch.from_numpy(sigma_np).to(dtype=F.dtype, device=F.device)  # type: ignore[attr-defined]
            shape = (1,) * (F.ndim - 1) + (N,)
            sigma = sigma.reshape(shape)
            F_filtered = F * sigma
            return self.idct(F_filtered)
        else:
            F_filtered = F * sigma_np.reshape((1,) * (F.ndim - 1) + (N,))
            return self.idct(F_filtered)

    # Spectral derivative
    def dx1(self, f: np.ndarray) -> np.ndarray:
        # Supports f with shape (..., N)
        if self._basis_name == "fourier":
            F = self.rfft(f)
            F *= self.dealias_mask
            F *= self.filter_sigma
            dF = self.ik * F
            return self.irfft(dF)
        elif self._basis_name == "chebyshev":
            # Delegate to basis implementation to avoid normalization drift
            return self._basis.dx(f)  # type: ignore[union-attr]
        else:
            # Generic basis path (e.g., Legendre): delegate to basis implementation
            return self._basis.dx(f)  # type: ignore[union-attr]

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        # Supports f with shape (..., N)
        if self._basis_name == "fourier":
            F = self.rfft(f)
            F *= self.dealias_mask
            F *= self.filter_sigma
            return self.irfft(F)
        else:
            # Non-Fourier path
            if self.filter_params and bool(self.filter_params.get("enabled", False)):
                p = int(self.filter_params.get("p", 8))
                alpha = float(self.filter_params.get("alpha", 36.0))
                # Delegate to basis filter (Chebyshev/Legendre implement modal filters)
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

        Supports global or per-variable BCs for non-periodic bases. Semantics:
        - "wall" (alias: reflective, dirichlet): velocity Dirichlet (u=0),
          scalars density/pressure use zero-gradient (Neumann).
        - "neumann" (alias: open): zero-gradient for all selected variables.
        Variable keys accepted: density|rho, momentum|velocity|u, pressure|p.
        """
        if self._basis_name not in ("chebyshev", "legendre"):
            return U
        # Determine BCs for each variable
        def _get_var_bc(var: str) -> str:
            var_n = var
            if self._bc_map is not None:
                return self._bc_map.get(var_n, self._bc_global or "wall")
            # No per-var map: interpret global
            g = self._bc_global or "wall"
            # For global "wall": momentum Dirichlet, scalars Neumann
            if g == "wall":
                if var_n == "momentum":
                    return "wall"
                else:
                    return "neumann"
            return g

        if U.shape[-1] != self.N:
            return U
        if U.shape[0] < 2:
            return U

        # Resolve per-variable BCs
        bc_rho = _get_var_bc("density")
        bc_mom = _get_var_bc("momentum")
        bc_p   = _get_var_bc("pressure")

        # Helper: apply zero-gradient (copy nearest interior) on a component idx
        def _apply_neumann(comp_idx: int) -> None:
            if self.N >= 3:
                U[..., comp_idx, 0] = U[..., comp_idx, 1]
                U[..., comp_idx, -1] = U[..., comp_idx, -2]
            elif self.N == 2:
                avg = 0.5 * (U[..., comp_idx, 0] + U[..., comp_idx, 1])
                U[..., comp_idx, 0] = avg
                U[..., comp_idx, 1] = avg

        # Helpers for Dirichlet using configured values
        def _apply_dirichlet(comp_idx: int, var_key: str) -> None:
            if self._dirichlet_values_map is not None and var_key in self._dirichlet_values_map:
                left_val, right_val = self._dirichlet_values_map[var_key]
                U[..., comp_idx, 0] = left_val
                U[..., comp_idx, -1] = right_val
            else:
                # Fallback: zero for momentum, else copy interior (approximate Neumann)
                if var_key == "momentum":
                    U[..., comp_idx, 0] = 0
                    U[..., comp_idx, -1] = 0
                else:
                    _apply_neumann(comp_idx)

        # Density
        if bc_rho in ("neumann", "open", "wall"):
            _apply_neumann(0)
        elif bc_rho == "dirichlet":
            _apply_dirichlet(0, "density")
        # Momentum / velocity
        if bc_mom in ("wall", "reflective"):
            U[..., 1, 0] = 0
            U[..., 1, -1] = 0
        elif bc_mom == "dirichlet":
            _apply_dirichlet(1, "momentum")
        elif bc_mom in ("neumann", "open"):
            _apply_neumann(1)
        # Pressure: approximate by applying to energy component
        if bc_p in ("neumann", "open", "wall"):
            _apply_neumann(2)
        elif bc_p == "dirichlet":
            _apply_dirichlet(2, "pressure")
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
    precision: str = "double"  # "single" or "double"
    debug: bool = False

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

            # Choose dtype based on precision and device
            if self.precision == "single":
                torch_dtype = torch.float32
                torch_cdtype = torch.complex64
            elif self.precision == "double":
                torch_dtype = torch.float64
                torch_cdtype = torch.complex128
            else:
                # Fallback to device-based logic (MPS only supports float32)
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
    precision: str = "double"  # "single" or "double"
    debug: bool = False

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

            # Choose dtype based on precision and device
            if self.precision == "single":
                torch_dtype = torch.float32
                torch_cdtype = torch.complex64
            elif self.precision == "double":
                torch_dtype = torch.float64
                torch_cdtype = torch.complex128
            else:
                # Fallback to device-based logic (MPS only supports float32)
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
