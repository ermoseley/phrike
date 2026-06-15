from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

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

try:
    import os as _os
    _os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    import finufft as _finufft  # type: ignore
    _FINUFFT_AVAILABLE = True
except Exception:
    _FINUFFT_AVAILABLE = False
    _finufft = None  # type: ignore

_FINUFFT_NTHREADS = 1 if _TORCH_AVAILABLE else 0  # 0 = auto; 1 avoids OMP conflicts with torch
_FINUFFT_OPTS: Dict[str, Any] = dict(isign=1, eps=1e-6, modeord=1, nthreads=_FINUFFT_NTHREADS)


def _to_np(x: Any) -> np.ndarray:
    """Convert tensor or array to numpy, handling MPS/CUDA tensors."""
    if _TORCH_AVAILABLE and torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


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

    Default is periodic Fourier basis on a uniform grid. Optionally, a
    Legendre–Gauss–Lobatto collocation basis with non-periodic BCs can be used.

    Attributes
    ----------
    N : int
        Number of spatial points.
    Lx : float
        Domain length.
    basis : str
        Spectral basis: "fourier" (default) or "legendre".
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
    problem_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Normalize/parse boundary condition configuration (string, dict, or per-boundary dict)
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

        def _normalize_boundary_name(b: str) -> str:
            n = str(b).strip().lower()
            if n in ("left", "x0", "x_min", "start"):
                return "left"
            if n in ("right", "x1", "x_max", "end"):
                return "right"
            return n

        # bc may be a string, per-variable dict, or per-boundary dict
        self._bc_global: str | None = None
        self._bc_map: dict[str, str] | None = None
        self._bc_boundary_map: dict[str, dict[str, str]] | None = None  # {boundary: {var: bc}}
        self._dirichlet_values_map: dict[str, tuple[float, float]] | None = None
        
        if isinstance(self.bc, dict):
            # Check if this is a per-boundary configuration
            if any(k.lower() in ("left", "right", "x0", "x1", "x_min", "x_max", "start", "end") 
                   for k in self.bc.keys()):
                # Per-boundary configuration: {left: {var: bc}, right: {var: bc}}
                self._bc_boundary_map = {}
                for boundary_key, var_bc_dict in self.bc.items():
                    boundary = _normalize_boundary_name(boundary_key)
                    if isinstance(var_bc_dict, dict):
                        self._bc_boundary_map[boundary] = {}
                        for var_key, bc_value in var_bc_dict.items():
                            var = _normalize_var_name(var_key)
                            self._bc_boundary_map[boundary][var] = _normalize_bc_value(bc_value)
                    else:
                        # Single BC for all variables on this boundary
                        bc_value = _normalize_bc_value(var_bc_dict)
                        self._bc_boundary_map[boundary] = {
                            "density": bc_value,
                            "momentum": bc_value,
                            "pressure": bc_value
                        }
            else:
                # Per-variable configuration: {var: bc}
                self._bc_map = {}
                for k, v in self.bc.items():
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
        elif self._basis_name == "legendre":
            # Legendre–Gauss–Lobatto nodes on [0, Lx]
            try:
                from .basis.legendre import LegendreLobattoBasis1D

                bc = self.bc or "dirichlet"
                # Check if parallel threading is enabled via runtime.num_threads
                use_parallel = False
                if self.problem_config:
                    runtime_cfg = self.problem_config.get("runtime", {})
                    num_threads_cfg = runtime_cfg.get("num_threads", None)
                    if num_threads_cfg is not None and num_threads_cfg != 1:
                        use_parallel = True
                self._basis = LegendreLobattoBasis1D(self.N, self.Lx, bc=bc, precision=self.precision, 
                                                   use_parallel=use_parallel)
                self.x = self._basis.nodes()
                x_np = np.asarray(self.x)
                if self.N > 1:
                    self.dx = float(np.min(np.diff(x_np)))
                else:
                    self.dx = self.Lx
                # No Fourier masks in non-periodic modal bases
                self.dealias_mask = np.ones(self.N, dtype=float)
                self.filter_sigma = np.ones(self.N, dtype=float)
                # Expose quadrature weights for monitoring integrations
                try:
                    self.wx = self._basis.quadrature_weights()
                except Exception:
                    self.wx = None
            except Exception as e:
                raise RuntimeError(f"Legendre basis initialization failed: {e}")
        else:
            raise ValueError(f"Unknown basis: {self.basis}")

        # Backend (torch/numpy) setup for periodic Fourier grids. Non-Fourier
        # bases keep the legacy path (backend set up in set_dirichlet_bc_values).
        if self._basis_name == "fourier":
            self._setup_fourier_backend()

    def _setup_fourier_backend(self) -> None:
        """Set up the array backend (torch device / numpy) for a Fourier grid.

        This was historically (and erroneously) only executed inside
        ``set_dirichlet_bc_values`` — which is never called for periodic Fourier
        problems — so a Fourier ``Grid1D`` silently ignored ``backend='torch'``.
        It is now invoked from ``__post_init__`` so the torch/MPS (Metal) backend
        works for periodic 1D problems (Euler and MHD).
        """
        # Enable FFTW planning cache if available (no-op with SciPy backend)
        if scipy_fft_cache_enabled and fftw_cache is not None:
            try:
                fftw_cache.enable()
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

        self._use_torch = (
            (self.backend.lower() == "torch")
            and _TORCH_AVAILABLE
            and (self._basis_name == "fourier")
        )
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

            _validate_torch_device(self.torch_device, self.debug)

            if self.precision == "single":
                torch_dtype = torch.float32
                torch_cdtype = torch.complex64
            else:
                torch_dtype = torch.float64
                torch_cdtype = torch.complex128
            if self.torch_device == "mps":
                torch_dtype = torch.float32
                torch_cdtype = torch.complex64

            self.x = torch.from_numpy(np.asarray(self.x)).to(
                dtype=torch_dtype, device=self.torch_device
            )
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
            self.k = np.asarray(self.k)
            self.ik = self.k * (1j)
            self.dealias_mask = np.asarray(self.dealias_mask)
            self.filter_sigma = np.asarray(self.filter_sigma)

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

        # Torch backend setup (Fourier, Legendre supported)
        self._use_torch = (
            (self.backend.lower() == "torch")
            and _TORCH_AVAILABLE
            and (self._basis_name in ("fourier", "legendre"))
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
                torch_dtype = torch.float64
                torch_cdtype = torch.complex128
            if self.torch_device == "mps":
                torch_dtype = torch.float32
                torch_cdtype = torch.complex64

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

    # (Chebyshev-specific DCT helpers removed)

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
                # Delegate to basis filter (e.g., Legendre implements modal filters)
                return self._basis.apply_spectral_filter(f, p=p, alpha=alpha)  # type: ignore[union-attr]
            return f

    def convolve(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Compute product in physical space, with dealiased spectral filtering.

        For pseudo-spectral nonlinearity, compute pointwise product then
        apply dealiasing/filter on the spectral representation.
        """
        h = f * g
        return self.apply_spectral_filter(h)

    # --- MHD solenoidal helpers (1D) ---
    def divergence_x(self, Bx: np.ndarray) -> np.ndarray:
        """1D spectral divergence d(Bx)/dx using the masked derivative operator."""
        return self.dx1(Bx)

    def project_solenoidal_x(self, Bx: np.ndarray) -> np.ndarray:
        """1D Helmholtz (Leray) projection of the x-magnetic field.

        In 1D the only solenoidal constraint is d(Bx)/dx = 0, whose unique
        periodic solution is Bx = const. The projection therefore keeps only the
        k=0 (mean) mode of Bx so that div(B) is zero to machine precision. The
        transverse components By, Bz are unconstrained in 1D and untouched here.
        """
        if self._basis_name != "fourier":
            raise ValueError("project_solenoidal_x requires Fourier basis")
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(Bx, torch.Tensor):
            return torch.ones_like(Bx) * torch.mean(Bx)
        Bx = np.asarray(Bx)
        return np.full_like(Bx, float(Bx.mean()))

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """1D spectral Laplacian d^2/dx^2 (for explicit resistivity/viscosity)."""
        if self._basis_name != "fourier":
            raise ValueError("laplacian requires Fourier basis")
        F = self.rfft(f)
        F = F * self.dealias_mask * self.filter_sigma
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(F, torch.Tensor):
            k2 = (self.k ** 2).to(F.dtype)
        else:
            k2 = np.asarray(self.k) ** 2
        return self.irfft(-k2 * F)

    def evaluate_fourier_at_points_1d(self, f: np.ndarray, xp: np.ndarray) -> np.ndarray:
        """Evaluate a Fourier-represented field at arbitrary points (Fourier basis only).

        f(x) = (1/N) * Re( sum_k F_k exp(i*k*x) ) with dealias and filter applied.
        Supports NumPy and PyTorch (including MPS/CUDA); keeps tensors on same device.

        Args:
            f: Field on grid, shape (..., N); last axis is spatial.
            xp: Query points, shape (P,).

        Returns:
            Values at xp, shape (P,). Same dtype/device as f (real).
        """
        if self._basis_name != "fourier":
            raise ValueError("evaluate_fourier_at_points_1d requires Fourier basis")
        is_torch_input = _TORCH_AVAILABLE and torch is not None and getattr(self, "_use_torch", False) and isinstance(f, torch.Tensor)
        if is_torch_input:
            f_np = f.detach().cpu().numpy().astype(np.float64)
            xp_np = np.asarray(xp.detach().cpu().numpy() if isinstance(xp, torch.Tensor) else xp, dtype=np.float64)
            device, dtype = f.device, f.dtype
        else:
            f_np = np.asarray(f)
            xp_np = np.asarray(xp, dtype=np.float64)
        dealias = _to_np(self.dealias_mask)
        sigma = _to_np(self.filter_sigma)
        f_flat = np.reshape(f_np, (-1, self.N))
        F = np.fft.fft(f_flat, axis=-1)
        F = np.ascontiguousarray((F * dealias * sigma)[0], dtype=np.complex128)
        if _FINUFFT_AVAILABLE:
            x_fin = np.ascontiguousarray((2.0 * np.pi / self.Lx) * xp_np)
            vals = np.real(_finufft.nufft1d2(x_fin, F, **_FINUFFT_OPTS)) / self.N
        else:
            k = np.asarray(self.k)
            E = np.exp(1j * xp_np[:, None] * k[None, :])
            vals = np.real((E * F[None, :]).sum(axis=1)) / self.N
        if is_torch_input:
            return torch.from_numpy(vals.astype(np.float32 if dtype == torch.float32 else np.float64)).to(device=device)
        return vals.astype(f_np.dtype)

    # --- Boundary conditions (1D) ---
    def apply_boundary_conditions(self, U: np.ndarray, equations) -> np.ndarray:
        """Apply boundary conditions to conservative variables U in-place.

        Supports global, per-variable, or per-boundary BCs for non-periodic bases. Semantics:
        - "wall" (alias: reflective, dirichlet): velocity Dirichlet (u=0),
          scalars density/pressure use zero-gradient (Neumann).
        - "neumann" (alias: open): zero-gradient for all selected variables.
        Variable keys accepted: density|rho, momentum|velocity|u, pressure|p.
        Boundary keys accepted: left|x0|x_min|start, right|x1|x_max|end.
        """
        if self._basis_name != "legendre":
            return U
        
        # Determine BCs for each variable and boundary
        def _get_var_bc(var: str, boundary: str = None) -> str:
            var_n = var
            
            # Check per-boundary configuration first
            if boundary and self._bc_boundary_map is not None:
                if boundary in self._bc_boundary_map:
                    return self._bc_boundary_map[boundary].get(var_n, self._bc_global or "wall")
            
            # Fall back to per-variable configuration
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

        # Resolve per-variable BCs for each boundary
        bc_rho_left = _get_var_bc("density", "left")
        bc_rho_right = _get_var_bc("density", "right")
        bc_mom_left = _get_var_bc("momentum", "left")
        bc_mom_right = _get_var_bc("momentum", "right")
        bc_p_left = _get_var_bc("pressure", "left")
        bc_p_right = _get_var_bc("pressure", "right")

        # Helper: apply zero-gradient (copy nearest interior) on a component idx
        def _apply_neumann(comp_idx: int, boundary: str) -> None:
            if boundary == "left":
                if self.N >= 2:
                    U[..., comp_idx, 0] = U[..., comp_idx, 1]
            elif boundary == "right":
                if self.N >= 2:
                    U[..., comp_idx, -1] = U[..., comp_idx, -2]

        # Helpers for Dirichlet using configured values
        def _apply_dirichlet(comp_idx: int, var_key: str, boundary: str) -> None:
            if self._dirichlet_values_map is not None and var_key in self._dirichlet_values_map:
                left_val, right_val = self._dirichlet_values_map[var_key]
                if boundary == "left":
                    U[..., comp_idx, 0] = left_val
                elif boundary == "right":
                    U[..., comp_idx, -1] = right_val
            else:
                # Fallback: zero for momentum, else copy interior (approximate Neumann)
                if var_key == "momentum":
                    if boundary == "left":
                        U[..., comp_idx, 0] = 0
                    elif boundary == "right":
                        U[..., comp_idx, -1] = 0
                else:
                    _apply_neumann(comp_idx, boundary)

        # Apply boundary conditions for each variable and boundary
        # Density
        if bc_rho_left in ("neumann", "open", "wall"):
            _apply_neumann(0, "left")
        elif bc_rho_left == "dirichlet":
            _apply_dirichlet(0, "density", "left")
            
        if bc_rho_right in ("neumann", "open", "wall"):
            _apply_neumann(0, "right")
        elif bc_rho_right == "dirichlet":
            _apply_dirichlet(0, "density", "right")
            
        # Momentum / velocity
        if bc_mom_left in ("wall", "reflective"):
            U[..., 1, 0] = 0
        elif bc_mom_left == "dirichlet":
            _apply_dirichlet(1, "momentum", "left")
        elif bc_mom_left in ("neumann", "open"):
            _apply_neumann(1, "left")
            
        if bc_mom_right in ("wall", "reflective"):
            U[..., 1, -1] = 0
        elif bc_mom_right == "dirichlet":
            _apply_dirichlet(1, "momentum", "right")
        elif bc_mom_right in ("neumann", "open"):
            _apply_neumann(1, "right")
            
        # Pressure: approximate by applying to energy component
        if bc_p_left in ("neumann", "open", "wall"):
            _apply_neumann(2, "left")
        elif bc_p_left == "dirichlet":
            _apply_dirichlet(2, "pressure", "left")
            
        if bc_p_right in ("neumann", "open", "wall"):
            _apply_neumann(2, "right")
        elif bc_p_right == "dirichlet":
            _apply_dirichlet(2, "pressure", "right")
            
        return U


def _build_legendre_diff_matrix_1d(x: np.ndarray, L: float) -> np.ndarray:
    """Build Legendre differentiation matrix for 1D.
    
    Args:
        x: Legendre-Gauss-Lobatto points
        L: Domain length
        
    Returns:
        Differentiation matrix D with shape (N, N)
    """
    N = len(x)
    D = np.zeros((N, N))
    
    # Map x from [0, L] to [-1, 1]
    xi = 2.0 * x / L - 1.0
    
    # Legendre differentiation matrix on [-1, 1]
    for i in range(N):
        for j in range(N):
            if i != j:
                # Standard Legendre differentiation formula
                D[i, j] = (1.0 / (xi[i] - xi[j])) * np.sqrt((1 - xi[j]**2) / (1 - xi[i]**2))
            else:
                # Diagonal elements
                if i == 0:
                    D[i, i] = -N * (N - 1) / 4.0
                elif i == N - 1:
                    D[i, i] = N * (N - 1) / 4.0
                else:
                    D[i, i] = 0.0
    
    # Scale by 2/L to account for mapping from [-1,1] to [0,L]
    D *= 2.0 / L
    
    return D


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
    basis_x: str = "fourier"  # "fourier" or "legendre"
    basis_y: str = "fourier"  # "fourier" or "legendre"
    bc: str = "dirichlet"     # Simple boundary condition like SOD
    # Boundary condition configuration
    bc_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.dx = self.Lx / self.Nx
        self.dy = self.Ly / self.Ny
        # Precompute minimum node spacings for CFL with nonuniform grids (Legendre)
        self.dx_min = None
        self.dy_min = None
        
        # Check if we're using a pure Legendre basis
        if self.basis_x == "legendre" and self.basis_y == "legendre":
            # Enable 2D Legendre tensor-product basis
            try:
                from .basis.legendre import LegendreLobattoBasis2D
                # Determine parallel option from problem_config
                use_parallel = False
                if hasattr(self, "problem_config") and self.problem_config:
                    runtime_cfg = self.problem_config.get("runtime", {})
                    num_threads_cfg = runtime_cfg.get("num_threads", None)
                    if num_threads_cfg is not None and num_threads_cfg != 1:
                        use_parallel = True

                self._legendre_basis = LegendreLobattoBasis2D(
                    self.Nx, self.Ny, self.Lx, self.Ly,
                    bc=self.bc, precision=self.precision, use_parallel=use_parallel
                )

                # Nodes from basis
                self.x, self.y = self._legendre_basis.nodes()
                try:
                    import numpy as _np
                    self.dx_min = float(_np.min(_np.diff(_np.asarray(self.x)))) if self.Nx > 1 else self.Lx
                    self.dy_min = float(_np.min(_np.diff(_np.asarray(self.y)))) if self.Ny > 1 else self.Ly
                except Exception:
                    self.dx_min = self.Lx / max(self.Nx, 1)
                    self.dy_min = self.Ly / max(self.Ny, 1)

                # No Fourier wave numbers in Legendre-Legendre mode
                self.kx = None
                self.ky = None
                self.ikx = None
                self.iky = None
            except Exception as e:
                raise RuntimeError(f"2D Legendre basis initialization failed: {e}")
        else:
            # Hybrid or pure Fourier basis
            self._legendre_basis = None
            
            # Set up x-direction (Fourier or Legendre)
            if self.basis_x == "fourier":
                self.x = np.linspace(0.0, self.Lx, num=self.Nx, endpoint=False)
                kx_index = np.fft.fftfreq(self.Nx, d=self.dx)
                self.kx = 2.0 * np.pi * kx_index  # shape (Nx,)
                self.ikx = 1j * self.kx[None, :]  # shape (1, Nx)
                self.Dx = None  # Will use FFT for derivatives
            elif self.basis_x == "legendre":
                # Legendre-Gauss-Lobatto points
                from scipy.special import roots_legendre
                xi, _ = roots_legendre(self.Nx)
                self.x = 0.5 * self.Lx * (xi + 1.0)  # Map [-1,1] to [0, Lx]
                # Legendre differentiation matrix
                self.Dx = _build_legendre_diff_matrix_1d(self.x, self.Lx)
                self.kx = None
                self.ikx = None
                try:
                    import numpy as _np
                    self.dx_min = float(_np.min(_np.diff(self.x))) if len(self.x) > 1 else self.dx
                except Exception:
                    self.dx_min = self.dx
            else:
                raise ValueError(f"Unknown basis_x: {self.basis_x}")
            
            # Set up y-direction (Fourier or Legendre)
            if self.basis_y == "fourier":
                self.y = np.linspace(0.0, self.Ly, num=self.Ny, endpoint=False)
                ky_index = np.fft.fftfreq(self.Ny, d=self.dy)
                self.ky = 2.0 * np.pi * ky_index  # shape (Ny,)
                self.iky = 1j * self.ky[:, None]  # shape (Ny, 1)
                self.Dy = None  # Will use FFT for derivatives
            elif self.basis_y == "legendre":
                # Legendre-Gauss-Lobatto points
                from scipy.special import roots_legendre
                xi, _ = roots_legendre(self.Ny)
                self.y = 0.5 * self.Ly * (xi + 1.0)  # Map [-1,1] to [0, Ly]
                # Legendre differentiation matrix
                self.Dy = _build_legendre_diff_matrix_1d(self.y, self.Ly)
                self.ky = None
                self.iky = None
                try:
                    import numpy as _np
                    self.dy_min = float(_np.min(_np.diff(self.y))) if len(self.y) > 1 else self.dy
                except Exception:
                    self.dy_min = self.dy
            else:
                raise ValueError(f"Unknown basis_y: {self.basis_y}")

        # Solenoidal-projection wavenumbers (Fourier x & y only; None otherwise)
        if self.kx is not None and self.ky is not None:
            kx_np = np.asarray(self.kx)
            ky_np = np.asarray(self.ky)
            self.k2 = kx_np[None, :] ** 2 + ky_np[:, None] ** 2
            self.k2_safe = np.where(self.k2 == 0.0, 1.0, self.k2)
        else:
            self.k2 = None
            self.k2_safe = None

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

        # Parse boundary condition configuration
        self._parse_boundary_conditions()

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
                torch_dtype = torch.float64
                torch_cdtype = torch.complex128
            if self.torch_device == "mps":
                torch_dtype = torch.float32
                torch_cdtype = torch.complex64

            self.x = torch.from_numpy(np.asarray(self.x)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.y = torch.from_numpy(np.asarray(self.y)).to(
                dtype=torch_dtype, device=self.torch_device
            )

            # Only convert kx, ky to torch if they exist (not None for Legendre basis)
            if self.kx is not None:
                kx_t = torch.from_numpy(np.asarray(self.kx)).to(
                    dtype=torch_dtype, device=self.torch_device
                )
                self.kx = kx_t
                self.ikx = kx_t.to(dtype=torch_cdtype)[None, :] * (1j)
            else:
                self.ikx = None
                
            if self.ky is not None:
                ky_t = torch.from_numpy(np.asarray(self.ky)).to(
                    dtype=torch_dtype, device=self.torch_device
                )
                self.ky = ky_t
                self.iky = ky_t.to(dtype=torch_cdtype)[:, None] * (1j)
            else:
                self.iky = None
            self.dealias_mask = torch.from_numpy(np.asarray(self.dealias_mask)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            self.filter_sigma = torch.from_numpy(np.asarray(self.filter_sigma)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            if self.k2_safe is not None:
                self.k2 = torch.from_numpy(np.asarray(self.k2)).to(
                    dtype=torch_dtype, device=self.torch_device
                )
                self.k2_safe = torch.from_numpy(np.asarray(self.k2_safe)).to(
                    dtype=torch_dtype, device=self.torch_device
                )

    def _parse_boundary_conditions(self) -> None:
        """Parse boundary condition configuration for 2D grid."""
        # Initialize boundary condition maps
        self._bc_boundary_map = {}
        self._dirichlet_values_map = {}
        
        if not self.bc_config:
            # Default all boundaries to global bc if provided
            # Accept strings: "dirichlet" (treated as reflective wall) or "neumann"
            if isinstance(self.bc, str) and self.bc in ("dirichlet", "neumann", "wall", "reflective"):
                default = self.bc
                if default in ("neumann",):
                    for boundary in ["left", "right", "top", "bottom"]:
                        self._bc_boundary_map[boundary] = {
                            "density": "neumann",
                            "momentum_x": "neumann",
                            "momentum_y": "neumann",
                            "pressure": "neumann",
                        }
                else:
                    # Treat "dirichlet"/"wall"/"reflective" as solid reflective walls:
                    # - zero normal momentum (Dirichlet u_n=0)
                    # - zero-gradient for tangential momentum, density, and pressure (Neumann)
                    # Left/right: normal is x
                    for boundary in ["left", "right"]:
                        self._bc_boundary_map[boundary] = {
                            "density": "neumann",
                            "momentum_x": "dirichlet",  # u_n = 0
                            "momentum_y": "neumann",    # tangential
                            "pressure": "neumann",
                        }
                        # Provide Dirichlet value for momentum_x → 0
                        self._dirichlet_values_map[f"{boundary}_momentum_x"] = 0.0
                    # Bottom/top: normal is y
                    for boundary in ["bottom", "top"]:
                        self._bc_boundary_map[boundary] = {
                            "density": "neumann",
                            "momentum_x": "neumann",    # tangential
                            "momentum_y": "dirichlet",  # u_n = 0
                            "pressure": "neumann",
                        }
                        # Provide Dirichlet value for momentum_y → 0
                        self._dirichlet_values_map[f"{boundary}_momentum_y"] = 0.0
            return
            
        # Parse boundary-specific configurations
        for boundary, config in self.bc_config.items():
            if boundary in ["left", "right", "top", "bottom"]:
                if isinstance(config, dict):
                    # Dirichlet boundary with specific values
                    self._bc_boundary_map[boundary] = {}
                    self._dirichlet_values_map[f"{boundary}_density"] = config.get("density", 1.0)
                    self._dirichlet_values_map[f"{boundary}_momentum_x"] = config.get("momentum_x", 0.0)
                    self._dirichlet_values_map[f"{boundary}_momentum_y"] = config.get("momentum_y", 0.0)
                    self._dirichlet_values_map[f"{boundary}_pressure"] = config.get("pressure", 1.0)
                    
                    # Set boundary conditions for each variable
                    self._bc_boundary_map[boundary]["density"] = "dirichlet"
                    self._bc_boundary_map[boundary]["momentum_x"] = "dirichlet"
                    self._bc_boundary_map[boundary]["momentum_y"] = "dirichlet"
                    self._bc_boundary_map[boundary]["pressure"] = "dirichlet"
                elif config == "neumann":
                    # Neumann boundary
                    self._bc_boundary_map[boundary] = {
                        "density": "neumann",
                        "momentum_x": "neumann", 
                        "momentum_y": "neumann",
                        "pressure": "neumann"
                    }
                elif config == "dirichlet":
                    # Default Dirichlet boundary
                    self._bc_boundary_map[boundary] = {
                        "density": "dirichlet",
                        "momentum_x": "dirichlet",
                        "momentum_y": "dirichlet", 
                        "pressure": "dirichlet"
                    }

    def apply_boundary_conditions(self, U: np.ndarray, equations) -> np.ndarray:
        """Apply boundary conditions to conservative variables U in-place for 2D grid.
        
        Args:
            U: Conservative variables array with shape (4, Ny, Nx)
            equations: Equations object for variable interpretation
            
        Returns:
            U with boundary conditions applied
        """
        if self._legendre_basis is None:
            # Only apply BCs for Legendre basis
            return U
            
        if U.shape[-2:] != (self.Ny, self.Nx):
            return U
        if U.shape[0] < 4:
            return U

        # Helper function to apply zero-gradient (Neumann)
        def _apply_neumann(comp_idx: int, boundary: str) -> None:
            if boundary == "left":
                if self.Nx >= 2:
                    U[comp_idx, :, 0] = U[comp_idx, :, 1]
            elif boundary == "right":
                if self.Nx >= 2:
                    U[comp_idx, :, -1] = U[comp_idx, :, -2]
            elif boundary == "bottom":
                if self.Ny >= 2:
                    U[comp_idx, 0, :] = U[comp_idx, 1, :]
            elif boundary == "top":
                if self.Ny >= 2:
                    U[comp_idx, -1, :] = U[comp_idx, -2, :]

        # Helper function to apply Dirichlet values
        def _apply_dirichlet(comp_idx: int, var_key: str, boundary: str) -> None:
            value_key = f"{boundary}_{var_key}"
            if value_key in self._dirichlet_values_map:
                value = self._dirichlet_values_map[value_key]
                if boundary == "left":
                    U[comp_idx, :, 0] = value
                elif boundary == "right":
                    U[comp_idx, :, -1] = value
                elif boundary == "bottom":
                    U[comp_idx, 0, :] = value
                elif boundary == "top":
                    U[comp_idx, -1, :] = value
            else:
                # Fallback to Neumann
                _apply_neumann(comp_idx, boundary)

        # Apply boundary conditions for each boundary
        for boundary in ["left", "right", "top", "bottom"]:
            if boundary not in self._bc_boundary_map:
                continue
                
            bc_config = self._bc_boundary_map[boundary]
            
            # Density (component 0)
            if bc_config.get("density") == "neumann":
                _apply_neumann(0, boundary)
            elif bc_config.get("density") == "dirichlet":
                _apply_dirichlet(0, "density", boundary)
                
            # Momentum x (component 1)
            if bc_config.get("momentum_x") == "neumann":
                _apply_neumann(1, boundary)
            elif bc_config.get("momentum_x") == "dirichlet":
                _apply_dirichlet(1, "momentum_x", boundary)
                
            # Momentum y (component 2)
            if bc_config.get("momentum_y") == "neumann":
                _apply_neumann(2, boundary)
            elif bc_config.get("momentum_y") == "dirichlet":
                _apply_dirichlet(2, "momentum_y", boundary)
                
            # Pressure (component 3 - energy)
            if bc_config.get("pressure") == "neumann":
                _apply_neumann(3, boundary)
            elif bc_config.get("pressure") == "dirichlet":
                _apply_dirichlet(3, "pressure", boundary)

        return U

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

    def evaluate_fourier_at_points(self, f: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """Evaluate a Fourier-represented 2D field at arbitrary points (Fourier x and y only).

        f(x,y) = (1/(Nx*Ny)) * Re( sum_{kx,ky} F exp(i*kx*x + i*ky*y) ) with dealias and filter.
        Supports NumPy and PyTorch (including MPS/CUDA); keeps tensors on same device.

        Args:
            f: Field on grid, shape (Ny, Nx).
            xp: Query x coordinates, shape (P,).
            yp: Query y coordinates, shape (P,).

        Returns:
            Values at (xp, yp), shape (P,).
        """
        if self.basis_x != "fourier" or self.basis_y != "fourier":
            raise ValueError("evaluate_fourier_at_points requires Fourier basis in both x and y")
        is_torch_input = _TORCH_AVAILABLE and torch is not None and getattr(self, "_use_torch", False) and isinstance(f, torch.Tensor)
        if is_torch_input:
            f_np = f.detach().cpu().numpy().astype(np.float64)
            xp_np = np.asarray(xp.detach().cpu().numpy() if isinstance(xp, torch.Tensor) else xp, dtype=np.float64)
            yp_np = np.asarray(yp.detach().cpu().numpy() if isinstance(yp, torch.Tensor) else yp, dtype=np.float64)
            device, dtype = f.device, f.dtype
        else:
            f_np = np.asarray(f)
            xp_np = np.asarray(xp, dtype=np.float64)
            yp_np = np.asarray(yp, dtype=np.float64)
        dealias = _to_np(self.dealias_mask)
        sigma = _to_np(self.filter_sigma)
        F = np.ascontiguousarray(np.fft.fft2(f_np, axes=(-2, -1)) * dealias * sigma, dtype=np.complex128)
        N_total = self.Nx * self.Ny
        if _FINUFFT_AVAILABLE:
            # F is (Ny, Nx): first axis=y, second=x -> nufft2d2(y, x, F)
            y_fin = np.ascontiguousarray((2.0 * np.pi / self.Ly) * yp_np)
            x_fin = np.ascontiguousarray((2.0 * np.pi / self.Lx) * xp_np)
            vals = np.real(_finufft.nufft2d2(y_fin, x_fin, F, **_FINUFFT_OPTS)) / N_total
        else:
            kx = np.asarray(self.kx).reshape(-1)
            ky = np.asarray(self.ky).reshape(-1)
            E_ky = np.exp(1j * yp_np[:, None] * ky[None, :])
            E_kx = np.exp(1j * xp_np[:, None] * kx[None, :])
            M = E_ky @ F
            vals = np.real((M * E_kx).sum(axis=1)) / N_total
        if is_torch_input:
            return torch.from_numpy(vals.astype(np.float32 if dtype == torch.float32 else np.float64)).to(device=device)
        return vals.astype(f_np.dtype)

    def dx1(self, f: np.ndarray) -> np.ndarray:
        # derivative along x on last spatial axis
        if self._legendre_basis is not None:
            # Use 2D Legendre basis directly (supports NumPy and Torch)
            return self._legendre_basis.dx(f)
        elif self.basis_x == "fourier":
            F = self.fft2(f)
            F = self._apply_masks(F)
            dF = F * self.ikx  # broadcast over (Ny, Nx)
            return self.ifft2(dF)
        elif self.basis_x == "legendre":
            # Use Legendre differentiation matrix
            # f has shape (..., Ny, Nx), we need to apply Dx to the last axis
            return np.tensordot(f, self.Dx, axes=([-1], [1]))
        else:
            raise ValueError(f"Unknown basis_x: {self.basis_x}")

    def dy1(self, f: np.ndarray) -> np.ndarray:
        # derivative along y on second-to-last spatial axis
        if self._legendre_basis is not None:
            # Use 2D Legendre basis directly (supports NumPy and Torch)
            return self._legendre_basis.dy(f)
        elif self.basis_y == "fourier":
            F = self.fft2(f)
            F = self._apply_masks(F)
            dF = F * self.iky
            return self.ifft2(dF)
        elif self.basis_y == "legendre":
            # Use Legendre differentiation matrix
            # f has shape (..., Ny, Nx), we need to apply Dy to the second-to-last axis
            return np.tensordot(f, self.Dy, axes=([-2], [1]))
        else:
            raise ValueError(f"Unknown basis_y: {self.basis_y}")

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        if self._legendre_basis is not None:
            # Use 2D Legendre basis filtering (supports NumPy and Torch)
            if self.filter_params and bool(self.filter_params.get("enabled", False)):
                p = int(self.filter_params.get("p", 8))
                alpha = float(self.filter_params.get("alpha", 36.0))
                return self._legendre_basis.apply_spectral_filter(f, p=p, alpha=alpha)
            else:
                return f
        elif self.basis_x == "fourier" and self.basis_y == "fourier":
            F = self.fft2(f)
            F = self._apply_masks(F)
            return self.ifft2(F)
        else:
            # For hybrid basis, only apply filter to Fourier directions
            if self.basis_x == "fourier" and self.basis_y == "legendre":
                # Apply FFT in x, then apply filter
                F = self.fft2(f)
                F = self._apply_masks(F)
                return self.ifft2(F)
            elif self.basis_x == "legendre" and self.basis_y == "fourier":
                # Apply FFT in y, then apply filter
                F = self.fft2(f)
                F = self._apply_masks(F)
                return self.ifft2(F)
            else:
                # Both Legendre - no spectral filtering
                return f

    # --- MHD solenoidal helpers (2D) ---
    def divergence(self, Bx: np.ndarray, By: np.ndarray) -> np.ndarray:
        """2D spectral divergence dxBx + dyBy using the masked derivative operator."""
        Fx = self._apply_masks(self.fft2(Bx))
        Fy = self._apply_masks(self.fft2(By))
        div_hat = self.ikx * Fx + self.iky * Fy
        return self.ifft2(div_hat)

    def project_solenoidal(self, Bx: np.ndarray, By: np.ndarray):
        """2D Helmholtz (Leray) projection of the in-plane field (Bx, By).

        Removes the longitudinal part so that kx*Bx_hat + ky*By_hat = 0 for every
        mode, i.e. div(B) = 0 to machine precision. Bz (out-of-plane) does not
        enter the 2D divergence and is left untouched by the caller.
        """
        if self.kx is None or self.ky is None:
            raise ValueError("project_solenoidal requires Fourier basis in x and y")
        Fx = self.fft2(Bx)
        Fy = self.fft2(By)
        # Zero the Nyquist mode(s) to preserve Hermitian symmetry under the
        # per-mode projection (see Grid3D.project_solenoidal for the rationale).
        for F in (Fx, Fy):
            if self.Nx % 2 == 0:
                F[:, self.Nx // 2] = 0
            if self.Ny % 2 == 0:
                F[self.Ny // 2, :] = 0
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(Fx, torch.Tensor):
            kx = self.kx.to(Fx.dtype)[None, :]
            ky = self.ky.to(Fx.dtype)[:, None]
            k2_safe = self.k2_safe.to(Fx.real.dtype)
        else:
            kx = np.asarray(self.kx)[None, :]
            ky = np.asarray(self.ky)[:, None]
            k2_safe = self.k2_safe
        kdotB = kx * Fx + ky * Fy
        coef = kdotB / k2_safe
        Fx = Fx - kx * coef
        Fy = Fy - ky * coef
        return self.ifft2(Fx), self.ifft2(Fy)

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """2D spectral Laplacian (for explicit resistivity/viscosity)."""
        if self.kx is None or self.ky is None:
            raise ValueError("laplacian requires Fourier basis in x and y")
        F = self._apply_masks(self.fft2(f))
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(F, torch.Tensor):
            k2 = self.k2.to(F.dtype)
        else:
            k2 = self.k2
        return self.ifft2(-k2 * F)

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

        # Squared wavenumber for the Helmholtz/Leray projection (div-free B)
        self.k2 = (
            self.kx[None, None, :] ** 2
            + self.ky[None, :, None] ** 2
            + self.kz[:, None, None] ** 2
        )
        self.k2_safe = np.where(self.k2 == 0.0, 1.0, self.k2)

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
                torch_dtype = torch.float64
                torch_cdtype = torch.complex128
            if self.torch_device == "mps":
                torch_dtype = torch.float32
                torch_cdtype = torch.complex64

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
            self.k2 = (
                kx_t[None, None, :] ** 2
                + ky_t[None, :, None] ** 2
                + kz_t[:, None, None] ** 2
            )
            self.k2_safe = torch.where(
                self.k2 == 0, torch.ones_like(self.k2), self.k2
            )
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

    def evaluate_fourier_at_points(
        self, f: np.ndarray, xp: np.ndarray, yp: np.ndarray, zp: np.ndarray
    ) -> np.ndarray:
        """Evaluate a Fourier-represented 3D field at arbitrary points.

        f(x,y,z) = (1/(Nx*Ny*Nz)) * Re( sum F exp(i*kx*x + i*ky*y + i*kz*z) ) with dealias and filter.
        Supports NumPy and PyTorch (including MPS/CUDA); keeps tensors on same device.

        Args:
            f: Field on grid, shape (Nz, Ny, Nx).
            xp, yp, zp: Query coordinates, each shape (P,).

        Returns:
            Values at (xp, yp, zp), shape (P,).
        """
        is_torch_input = _TORCH_AVAILABLE and torch is not None and getattr(self, "_use_torch", False) and isinstance(f, torch.Tensor)
        if is_torch_input:
            f_np = f.detach().cpu().numpy().astype(np.float64)
            xp_np = np.asarray(xp.detach().cpu().numpy() if isinstance(xp, torch.Tensor) else xp, dtype=np.float64)
            yp_np = np.asarray(yp.detach().cpu().numpy() if isinstance(yp, torch.Tensor) else yp, dtype=np.float64)
            zp_np = np.asarray(zp.detach().cpu().numpy() if isinstance(zp, torch.Tensor) else zp, dtype=np.float64)
            device, dtype = f.device, f.dtype
        else:
            f_np = np.asarray(f)
            xp_np = np.asarray(xp, dtype=np.float64)
            yp_np = np.asarray(yp, dtype=np.float64)
            zp_np = np.asarray(zp, dtype=np.float64)
        dealias = _to_np(self.dealias_mask)
        sigma = _to_np(self.filter_sigma)
        F = np.ascontiguousarray(np.fft.fftn(f_np, axes=(-3, -2, -1)) * dealias * sigma, dtype=np.complex128)
        N_total = self.Nx * self.Ny * self.Nz
        if _FINUFFT_AVAILABLE:
            z_fin = np.ascontiguousarray((2.0 * np.pi / self.Lz) * zp_np)
            y_fin = np.ascontiguousarray((2.0 * np.pi / self.Ly) * yp_np)
            x_fin = np.ascontiguousarray((2.0 * np.pi / self.Lx) * xp_np)
            vals = np.real(_finufft.nufft3d2(z_fin, y_fin, x_fin, F, **_FINUFFT_OPTS)) / N_total
        else:
            kx = np.asarray(self.kx).reshape(-1)
            ky = np.asarray(self.ky).reshape(-1)
            kz = np.asarray(self.kz).reshape(-1)
            E_kz = np.exp(1j * zp_np[:, None] * kz[None, :])
            E_ky = np.exp(1j * yp_np[:, None] * ky[None, :])
            E_kx = np.exp(1j * xp_np[:, None] * kx[None, :])
            T = (E_kz @ F.reshape(self.Nz, -1)).reshape(-1, self.Ny, self.Nx)
            vals = np.real((T * E_ky[:, :, None] * E_kx[:, None, :]).sum(axis=(-2, -1))) / N_total
        if is_torch_input:
            return torch.from_numpy(vals.astype(np.float32 if dtype == torch.float32 else np.float64)).to(device=device)
        return vals.astype(f_np.dtype)

    def evaluate_fourier_at_points_batched_3d(
        self,
        fx: np.ndarray,
        fy: np.ndarray,
        fz: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray,
        zp: np.ndarray,
    ) -> tuple:
        """Evaluate three Fourier 3D fields at the same points. Returns (vx, vy, vz).

        Same as calling evaluate_fourier_at_points for fx, fy, fz separately but
        reuses chunk loop and exp(E) matrices, reducing overhead (especially for
        tracer velocity interpolation).
        """
        is_torch_input = _TORCH_AVAILABLE and torch is not None and getattr(self, "_use_torch", False) and isinstance(fx, torch.Tensor)
        if is_torch_input:
            fx_np = fx.detach().cpu().numpy().astype(np.float64)
            fy_np = fy.detach().cpu().numpy().astype(np.float64)
            fz_np = fz.detach().cpu().numpy().astype(np.float64)
            xp_np = np.asarray(xp.detach().cpu().numpy() if isinstance(xp, torch.Tensor) else xp, dtype=np.float64)
            yp_np = np.asarray(yp.detach().cpu().numpy() if isinstance(yp, torch.Tensor) else yp, dtype=np.float64)
            zp_np = np.asarray(zp.detach().cpu().numpy() if isinstance(zp, torch.Tensor) else zp, dtype=np.float64)
            device, dtype = fx.device, fx.dtype
        else:
            fx_np, fy_np, fz_np = np.asarray(fx), np.asarray(fy), np.asarray(fz)
            xp_np = np.asarray(xp, dtype=np.float64)
            yp_np = np.asarray(yp, dtype=np.float64)
            zp_np = np.asarray(zp, dtype=np.float64)
        dealias = _to_np(self.dealias_mask)
        sigma = _to_np(self.filter_sigma)
        scale = 1.0 / (self.Nx * self.Ny * self.Nz)
        Fx = np.ascontiguousarray(np.fft.fftn(fx_np, axes=(-3, -2, -1)) * dealias * sigma, dtype=np.complex128)
        Fy = np.ascontiguousarray(np.fft.fftn(fy_np, axes=(-3, -2, -1)) * dealias * sigma, dtype=np.complex128)
        Fz = np.ascontiguousarray(np.fft.fftn(fz_np, axes=(-3, -2, -1)) * dealias * sigma, dtype=np.complex128)
        if _FINUFFT_AVAILABLE:
            z_fin = np.ascontiguousarray((2.0 * np.pi / self.Lz) * zp_np)
            y_fin = np.ascontiguousarray((2.0 * np.pi / self.Ly) * yp_np)
            x_fin = np.ascontiguousarray((2.0 * np.pi / self.Lx) * xp_np)
            vx = np.real(_finufft.nufft3d2(z_fin, y_fin, x_fin, Fx, **_FINUFFT_OPTS)) * scale
            vy = np.real(_finufft.nufft3d2(z_fin, y_fin, x_fin, Fy, **_FINUFFT_OPTS)) * scale
            vz = np.real(_finufft.nufft3d2(z_fin, y_fin, x_fin, Fz, **_FINUFFT_OPTS)) * scale
        else:
            kx = np.asarray(self.kx).reshape(-1)
            ky = np.asarray(self.ky).reshape(-1)
            kz = np.asarray(self.kz).reshape(-1)
            E_kz = np.exp(1j * zp_np[:, None] * kz[None, :])
            E_ky = np.exp(1j * yp_np[:, None] * ky[None, :])
            E_kx = np.exp(1j * xp_np[:, None] * kx[None, :])
            Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
            Tx = (E_kz @ Fx.reshape(Nz, -1)).reshape(-1, Ny, Nx)
            Ty = (E_kz @ Fy.reshape(Nz, -1)).reshape(-1, Ny, Nx)
            Tz = (E_kz @ Fz.reshape(Nz, -1)).reshape(-1, Ny, Nx)
            vx = np.real((Tx * E_ky[:, :, None] * E_kx[:, None, :]).sum(axis=(-2, -1))) * scale
            vy = np.real((Ty * E_ky[:, :, None] * E_kx[:, None, :]).sum(axis=(-2, -1))) * scale
            vz = np.real((Tz * E_ky[:, :, None] * E_kx[:, None, :]).sum(axis=(-2, -1))) * scale
        if is_torch_input:
            np_dtype = np.float32 if dtype == torch.float32 else np.float64
            return (
                torch.from_numpy(vx.astype(np_dtype)).to(device=device),
                torch.from_numpy(vy.astype(np_dtype)).to(device=device),
                torch.from_numpy(vz.astype(np_dtype)).to(device=device),
            )
        return (vx.astype(fx_np.dtype), vy.astype(fy_np.dtype), vz.astype(fz_np.dtype))

    # --- MHD solenoidal helpers (3D) ---
    def divergence(self, Vx: np.ndarray, Vy: np.ndarray, Vz: np.ndarray) -> np.ndarray:
        """3D spectral divergence dxVx + dyVy + dzVz (masked derivative operator).

        Used as the div(B) diagnostic. After project_solenoidal this is zero to
        machine precision because k.V_hat = 0 on every retained mode and the
        mask only zeroes high modes.
        """
        Fx = self._apply_masks(self.fftn(Vx))
        Fy = self._apply_masks(self.fftn(Vy))
        Fz = self._apply_masks(self.fftn(Vz))
        div_hat = self.ikx * Fx + self.iky * Fy + self.ikz * Fz
        return self.ifftn(div_hat)

    def project_solenoidal(self, Vx: np.ndarray, Vy: np.ndarray, Vz: np.ndarray):
        """Helmholtz (Leray) projection: remove the longitudinal part of V.

        After this, k.V_hat = 0 for all k  =>  div(V) = 0 to machine precision.
        The k=0 (mean) mode is untouched (k.V_hat = 0 there already), so a
        uniform mean field is preserved exactly.
        """
        Fx = self.fftn(Vx)
        Fy = self.fftn(Vy)
        Fz = self.fftn(Vz)
        # Zero the Nyquist mode(s): for an even-length real FFT the Nyquist mode is
        # its own alias, so the per-mode projection below would break Hermitian
        # symmetry and ifftn(...).real would silently drop a nonzero imaginary
        # part, leaving the stored field NOT divergence-free at Nyquist.
        for F in (Fx, Fy, Fz):
            if self.Nx % 2 == 0:
                F[:, :, self.Nx // 2] = 0
            if self.Ny % 2 == 0:
                F[:, self.Ny // 2, :] = 0
            if self.Nz % 2 == 0:
                F[self.Nz // 2, :, :] = 0
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(Fx, torch.Tensor):
            kx = self.kx.to(Fx.dtype)[None, None, :]
            ky = self.ky.to(Fx.dtype)[None, :, None]
            kz = self.kz.to(Fx.dtype)[:, None, None]
            k2_safe = self.k2_safe.to(Fx.real.dtype)
        else:
            kx = np.asarray(self.kx)[None, None, :]
            ky = np.asarray(self.ky)[None, :, None]
            kz = np.asarray(self.kz)[:, None, None]
            k2_safe = self.k2_safe
        kdotV = kx * Fx + ky * Fy + kz * Fz
        coef = kdotV / k2_safe
        Fx = Fx - kx * coef
        Fy = Fy - ky * coef
        Fz = Fz - kz * coef
        return self.ifftn(Fx), self.ifftn(Fy), self.ifftn(Fz)

    def curl(self, Vx: np.ndarray, Vy: np.ndarray, Vz: np.ndarray):
        """3D spectral curl (masked derivative operator). Returns (Cx, Cy, Cz)."""
        Fx = self._apply_masks(self.fftn(Vx))
        Fy = self._apply_masks(self.fftn(Vy))
        Fz = self._apply_masks(self.fftn(Vz))
        Cx = self.ifftn(self.iky * Fz - self.ikz * Fy)
        Cy = self.ifftn(self.ikz * Fx - self.ikx * Fz)
        Cz = self.ifftn(self.ikx * Fy - self.iky * Fx)
        return Cx, Cy, Cz

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """3D spectral Laplacian (used for explicit resistivity/viscosity)."""
        F = self._apply_masks(self.fftn(f))
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(F, torch.Tensor):
            k2 = self.k2.to(F.dtype)
        else:
            k2 = self.k2
        return self.ifftn(-k2 * F)

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
