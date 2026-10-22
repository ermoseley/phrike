from __future__ import annotations

import warnings
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


def _periodic_linear_1d(f: Any, xp: Any, length: float) -> Any:
    """Linearly interpolate a periodic 1-D grid without changing backends."""
    n = int(f.shape[-1])
    if _TORCH_AVAILABLE and torch is not None and isinstance(f, torch.Tensor):
        if not isinstance(xp, torch.Tensor):
            xp = torch.as_tensor(xp, dtype=f.dtype, device=f.device)
        else:
            xp = xp.to(dtype=f.dtype, device=f.device)
        coord = torch.remainder(xp, length) * (n / length)
        cell = torch.floor(coord)
        i0 = torch.remainder(cell.to(dtype=torch.long), n)
        frac = coord - cell
        i1 = torch.remainder(i0 + 1, n)
        return (1.0 - frac) * f[..., i0] + frac * f[..., i1]

    f_np = np.asarray(f)
    xp_np = np.asarray(xp, dtype=f_np.dtype)
    coord = np.mod(xp_np, length) * (n / length)
    cell = np.floor(coord)
    i0 = cell.astype(np.intp) % n
    frac = coord - cell
    i1 = (i0 + 1) % n
    return (1.0 - frac) * f_np[..., i0] + frac * f_np[..., i1]


def _periodic_linear_2d(
    f: Any, xp: Any, yp: Any, length_x: float, length_y: float
) -> Any:
    """Bilinearly interpolate a periodic ``(..., Ny, Nx)`` grid."""
    ny, nx = (int(size) for size in f.shape[-2:])
    if _TORCH_AVAILABLE and torch is not None and isinstance(f, torch.Tensor):
        if not isinstance(xp, torch.Tensor):
            xp = torch.as_tensor(xp, dtype=f.dtype, device=f.device)
        else:
            xp = xp.to(dtype=f.dtype, device=f.device)
        if not isinstance(yp, torch.Tensor):
            yp = torch.as_tensor(yp, dtype=f.dtype, device=f.device)
        else:
            yp = yp.to(dtype=f.dtype, device=f.device)
        sx = torch.remainder(xp, length_x) * (nx / length_x)
        sy = torch.remainder(yp, length_y) * (ny / length_y)
        cell_x = torch.floor(sx)
        cell_y = torch.floor(sy)
        ix0 = torch.remainder(cell_x.to(dtype=torch.long), nx)
        iy0 = torch.remainder(cell_y.to(dtype=torch.long), ny)
        wx = sx - cell_x
        wy = sy - cell_y
        ix1 = torch.remainder(ix0 + 1, nx)
        iy1 = torch.remainder(iy0 + 1, ny)
    else:
        f = np.asarray(f)
        xp = np.asarray(xp, dtype=f.dtype)
        yp = np.asarray(yp, dtype=f.dtype)
        sx = np.mod(xp, length_x) * (nx / length_x)
        sy = np.mod(yp, length_y) * (ny / length_y)
        cell_x = np.floor(sx)
        cell_y = np.floor(sy)
        ix0 = cell_x.astype(np.intp) % nx
        iy0 = cell_y.astype(np.intp) % ny
        wx = sx - cell_x
        wy = sy - cell_y
        ix1 = (ix0 + 1) % nx
        iy1 = (iy0 + 1) % ny

    return (
        (1.0 - wx) * (1.0 - wy) * f[..., iy0, ix0]
        + wx * (1.0 - wy) * f[..., iy0, ix1]
        + (1.0 - wx) * wy * f[..., iy1, ix0]
        + wx * wy * f[..., iy1, ix1]
    )


def _periodic_linear_3d(
    f: Any,
    xp: Any,
    yp: Any,
    zp: Any,
    length_x: float,
    length_y: float,
    length_z: float,
) -> Any:
    """Trilinearly interpolate a periodic ``(..., Nz, Ny, Nx)`` grid."""
    nz, ny, nx = (int(size) for size in f.shape[-3:])
    if _TORCH_AVAILABLE and torch is not None and isinstance(f, torch.Tensor):
        positions = []
        for position in (xp, yp, zp):
            if not isinstance(position, torch.Tensor):
                position = torch.as_tensor(
                    position, dtype=f.dtype, device=f.device
                )
            else:
                position = position.to(dtype=f.dtype, device=f.device)
            positions.append(position)
        xp, yp, zp = positions
        sx = torch.remainder(xp, length_x) * (nx / length_x)
        sy = torch.remainder(yp, length_y) * (ny / length_y)
        sz = torch.remainder(zp, length_z) * (nz / length_z)
        cell_x = torch.floor(sx)
        cell_y = torch.floor(sy)
        cell_z = torch.floor(sz)
        ix0 = torch.remainder(cell_x.to(dtype=torch.long), nx)
        iy0 = torch.remainder(cell_y.to(dtype=torch.long), ny)
        iz0 = torch.remainder(cell_z.to(dtype=torch.long), nz)
        wx = sx - cell_x
        wy = sy - cell_y
        wz = sz - cell_z
        ix1 = torch.remainder(ix0 + 1, nx)
        iy1 = torch.remainder(iy0 + 1, ny)
        iz1 = torch.remainder(iz0 + 1, nz)
    else:
        f = np.asarray(f)
        xp = np.asarray(xp, dtype=f.dtype)
        yp = np.asarray(yp, dtype=f.dtype)
        zp = np.asarray(zp, dtype=f.dtype)
        sx = np.mod(xp, length_x) * (nx / length_x)
        sy = np.mod(yp, length_y) * (ny / length_y)
        sz = np.mod(zp, length_z) * (nz / length_z)
        cell_x = np.floor(sx)
        cell_y = np.floor(sy)
        cell_z = np.floor(sz)
        ix0 = cell_x.astype(np.intp) % nx
        iy0 = cell_y.astype(np.intp) % ny
        iz0 = cell_z.astype(np.intp) % nz
        wx = sx - cell_x
        wy = sy - cell_y
        wz = sz - cell_z
        ix1 = (ix0 + 1) % nx
        iy1 = (iy0 + 1) % ny
        iz1 = (iz0 + 1) % nz

    c00 = (1.0 - wx) * f[..., iz0, iy0, ix0] + wx * f[..., iz0, iy0, ix1]
    c01 = (1.0 - wx) * f[..., iz0, iy1, ix0] + wx * f[..., iz0, iy1, ix1]
    c10 = (1.0 - wx) * f[..., iz1, iy0, ix0] + wx * f[..., iz1, iy0, ix1]
    c11 = (1.0 - wx) * f[..., iz1, iy1, ix0] + wx * f[..., iz1, iy1, ix1]
    c0 = (1.0 - wy) * c00 + wy * c01
    c1 = (1.0 - wy) * c10 + wy * c11
    return (1.0 - wz) * c0 + wz * c1


def _integer_wavenumbers(n: int) -> np.ndarray:
    """FFT wavenumbers in ``fftfreq`` order, as exact integers.

    ``fftfreq(n) * n`` is not an integer for every ``n``. A float comparison
    with ``(n - 1) // 3`` drops that cutoff mode for some resolutions.
    """
    return np.rint(np.fft.fftfreq(n) * n).astype(np.int64)


def _build_filter_mask(N: int, dealias: bool) -> np.ndarray:
    if not dealias:
        return np.ones(N, dtype=float)
    # Quadratic 2/3 rule on integer modes: retain |k| < N/3.
    k = _integer_wavenumbers(N)
    cutoff = (N - 1) // 3
    return (np.abs(k) <= cutoff).astype(float)


def _stabilization_params(
    params: Optional[Dict[str, Any]], key: str
) -> Optional[Dict[str, Any]]:
    """Select one stabilization block while preserving the direct grid API."""
    if not params:
        return None
    nested_keys = {"spectral_dissipation", "modal_filter"}
    if nested_keys.intersection(params):
        selected = params.get(key)
        if selected is None:
            return None
        if not isinstance(selected, dict):
            raise TypeError(f"{key} must be a mapping")
        return selected
    return params


def _spectral_dissipation_settings(
    params: Optional[Dict[str, Any]],
) -> tuple[bool, int, float, float]:
    """Return enabled, order, cutoff rate, and top-band onset."""
    params = _stabilization_params(params, "spectral_dissipation")
    if not params:
        return False, 8, 0.0, 0.0
    obsolete = {"alpha", "interval"}.intersection(params)
    if obsolete:
        names = ", ".join(sorted(obsolete))
        raise ValueError(
            f"Fourier spectral dissipation no longer accepts {names}; "
            "use rate or e_folding_time_at_cutoff"
        )
    if not bool(params.get("enabled", False)):
        return False, 8, 0.0, 0.0
    p = int(params.get("p", params.get("order", 8)))
    if p < 2 or p % 2:
        raise ValueError("Fourier spectral dissipation order must be even and >= 2")
    has_tau = "e_folding_time_at_cutoff" in params
    has_rate = "rate" in params
    if has_tau == has_rate:
        raise ValueError(
            "Enabled Fourier spectral dissipation requires exactly one of "
            "rate or e_folding_time_at_cutoff"
        )
    if has_tau:
        tau = float(params["e_folding_time_at_cutoff"])
        if not np.isfinite(tau) or tau <= 0.0:
            raise ValueError("e_folding_time_at_cutoff must be finite and positive")
        rate = 1.0 / tau
    else:
        rate = float(params["rate"])
    if not np.isfinite(rate) or rate < 0.0:
        raise ValueError("Fourier spectral dissipation rate must be finite and non-negative")
    onset = float(params.get("onset_fraction", 0.0))
    if not np.isfinite(onset) or not 0.0 <= onset < 1.0:
        raise ValueError(
            "Fourier spectral dissipation onset_fraction must be finite and "
            "lie in [0, 1)"
        )
    return True, p, rate, onset


def _modal_filter_settings(
    params: Optional[Dict[str, Any]],
) -> tuple[bool, int, float]:
    """Return the independent accepted-step modal-filter settings."""
    params = _stabilization_params(params, "modal_filter")
    if not params or not bool(params.get("enabled", False)):
        return False, 8, 36.0
    p = int(params.get("p", params.get("order", 8)))
    alpha = float(params.get("alpha", 36.0))
    if p < 2 or p % 2:
        raise ValueError("Modal filter order must be even and >= 2")
    if not np.isfinite(alpha) or alpha < 0.0:
        raise ValueError("Modal filter alpha must be finite and non-negative")
    return True, p, alpha


def _spectral_viscosity_profile(
    eta: np.ndarray, p: int, onset_fraction: float
) -> np.ndarray:
    """Return a power ramp confined to the selected top spectral band."""
    ramp = np.maximum(
        (eta - onset_fraction) / (1.0 - onset_fraction), 0.0
    )
    return ramp**p


def _build_exponential_filter_rate(
    N: int,
    p: int,
    rate: float,
    dealias: bool,
    onset_fraction: float = 0.0,
) -> np.ndarray:
    """Build the non-negative generator lambda(k) for exp(-dt*lambda)."""
    k = np.fft.fftfreq(N) * N
    kmax = int(np.max(np.abs(k)))
    scale = (N - 1) // 3 if dealias else kmax
    eta = np.abs(k) / max(scale, 1)
    return rate * _spectral_viscosity_profile(eta, p, onset_fraction)


def _mps_available() -> bool:
    return bool(
        _TORCH_AVAILABLE
        and torch is not None
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    )


def _cuda_available() -> bool:
    return bool(
        _TORCH_AVAILABLE and torch is not None and torch.cuda.is_available()
    )


def preferred_torch_device() -> str:
    """Return the preferred Torch device: Metal, then CUDA, then CPU."""
    if _mps_available():
        return "mps"
    if _cuda_available():
        return "cuda"
    return "cpu"


def _validate_torch_device(device: str, debug: bool = False) -> None:
    """Validate an explicit Torch device regardless of debug mode."""
    del debug  # Kept for compatibility with existing call sites.
    if not _TORCH_AVAILABLE or torch is None:
        raise ImportError(
            "The Torch backend requires PyTorch; install PHRIKE with "
            "`pip install -e '.[torch]'`."
        )
    if device == "cuda" and not _cuda_available():
        raise RuntimeError(
            "CUDA was requested but is not available in this PyTorch installation."
        )
    if device == "mps" and not _mps_available():
        raise RuntimeError(
            "Metal/MPS was requested but is unavailable. MPS requires PyTorch "
            "on an Apple Silicon Mac."
        )
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError(
            f"Unknown Torch device {device!r}; expected cpu, cuda, mps, or metal."
        )


def resolve_backend(
    backend: str = "auto", device: Optional[str] = None
) -> tuple[str, Optional[str]]:
    """Resolve user-facing backend aliases to an array backend and device.

    Automatic selection prefers Apple's Metal/MPS backend, then CUDA, and
    finally the NumPy CPU backend. Explicit ``backend='torch'`` falls back to
    Torch CPU when no accelerator is present.
    """
    requested_backend = str(backend).strip().lower()
    requested_device = None if device is None else str(device).strip().lower()
    if requested_device in {"", "auto"}:
        requested_device = None
    if requested_device == "metal":
        requested_device = "mps"

    backend_aliases = {
        "metal": ("torch", "mps"),
        "mps": ("torch", "mps"),
        "cuda": ("torch", "cuda"),
        "cpu": ("numpy", None),
    }
    if requested_backend in backend_aliases:
        resolved_backend, alias_device = backend_aliases[requested_backend]
        if requested_backend == "cpu" and requested_device == "cpu":
            requested_device = None
        if requested_device is not None and requested_device != alias_device:
            raise ValueError(
                f"Backend {requested_backend!r} conflicts with device "
                f"{requested_device!r}."
            )
        requested_backend = resolved_backend
        requested_device = alias_device

    if requested_backend == "auto":
        if requested_device in {"mps", "cuda"}:
            requested_backend = "torch"
        elif requested_device == "cpu":
            return "numpy", None
        elif _mps_available():
            return "torch", "mps"
        elif _cuda_available():
            return "torch", "cuda"
        else:
            return "numpy", None

    if requested_backend == "numpy":
        if requested_device not in {None, "cpu"}:
            raise ValueError(
                "The NumPy backend only supports CPU. Use backend='torch' "
                "for Metal/MPS or CUDA."
            )
        return "numpy", None

    if requested_backend != "torch":
        raise ValueError(
            f"Unknown backend {backend!r}; expected auto, metal, cuda, cpu, "
            "numpy, or torch."
        )

    resolved_device = requested_device or preferred_torch_device()
    _validate_torch_device(resolved_device)
    return "torch", resolved_device


_mps_double_warned = False
_numpy_fourier_single_warned = False


def _torch_dtypes(precision: str, device: str) -> tuple[Any, Any, str]:
    """Return real/complex Torch dtypes and the effective precision."""
    global _mps_double_warned
    normalized = str(precision).strip().lower()
    if normalized not in {"single", "double"}:
        raise ValueError(
            f"Unknown precision {precision!r}; expected 'single' or 'double'."
        )
    # PyTorch MPS does not support float64. Keep Metal as the automatic
    # accelerator, but do not drop to float32 without saying so.
    if device == "mps" and normalized == "double":
        if not _mps_double_warned:
            _mps_double_warned = True
            warnings.warn(
                "MPS does not support float64; using float32 instead of "
                "the requested double precision.",
                UserWarning,
                stacklevel=2,
            )
        normalized = "single"
    if normalized == "single":
        return torch.float32, torch.complex64, normalized
    return torch.float64, torch.complex128, normalized


def _note_numpy_fourier_single() -> None:
    """Warn once that NumPy Fourier operators stay in float64."""
    global _numpy_fourier_single_warned
    if _numpy_fourier_single_warned:
        return
    _numpy_fourier_single_warned = True
    warnings.warn(
        "NumPy Fourier operators always compute in float64; "
        "precision='single' is not applied to them.",
        UserWarning,
        stacklevel=2,
    )


def _build_filter_mask_2d(Nx: int, Ny: int, dealias: bool) -> np.ndarray:
    """Tensor-product 2/3 mask of shape (Ny, Nx).

    A mode is kept only when both ``|kx| <= (Nx - 1) // 3`` and
    ``|ky| <= (Ny - 1) // 3``. Differentiating in one direction therefore
    also removes modes that are resolved along that direction but sit
    outside the cutoff on the other axis.
    """
    if not dealias:
        return np.ones((Ny, Nx), dtype=float)
    kx = _integer_wavenumbers(Nx)
    ky = _integer_wavenumbers(Ny)
    cutoff_x = (Nx - 1) // 3
    cutoff_y = (Ny - 1) // 3
    mask_x = (np.abs(kx) <= cutoff_x).astype(float)
    mask_y = (np.abs(ky) <= cutoff_y).astype(float)
    return mask_y[:, None] * mask_x[None, :]


def _build_exponential_filter_rate_2d(
    Nx: int,
    Ny: int,
    p: int,
    rate: float,
    dealias: bool,
    onset_fraction: float = 0.0,
) -> np.ndarray:
    """Per-mode decay rate ``lambda = rate * (ramp(|ky|) + ramp(|kx|))``.

    The ramps are added. A mode at the cutoff on both axes has
    ``lambda = 2 * rate``, so its e-folding time is ``tau / 2`` when
    ``rate = 1 / tau`` and ``tau`` is ``e_folding_time_at_cutoff``.
    That setting is the e-folding time of one axis, not of the corner mode.
    """
    kx = (np.fft.fftfreq(Nx) * Nx).astype(float)
    ky = (np.fft.fftfreq(Ny) * Ny).astype(float)
    scale_x = (Nx - 1) // 3 if dealias else int(np.max(np.abs(kx)))
    scale_y = (Ny - 1) // 3 if dealias else int(np.max(np.abs(ky)))
    eta_x = np.abs(kx) / max(scale_x, 1)
    eta_y = np.abs(ky) / max(scale_y, 1)
    return rate * (
        _spectral_viscosity_profile(eta_y, p, onset_fraction)[:, None]
        + _spectral_viscosity_profile(eta_x, p, onset_fraction)[None, :]
    )


def _build_filter_mask_3d(Nx: int, Ny: int, Nz: int, dealias: bool) -> np.ndarray:
    """Tensor-product 2/3 mask of shape (Nz, Ny, Nx).

    A mode is kept only when it is inside the cutoff on every axis, so a
    derivative in one direction also removes modes resolved on that axis
    but outside the cutoff on another.
    """
    if not dealias:
        return np.ones((Nz, Ny, Nx), dtype=float)
    kx = _integer_wavenumbers(Nx)
    ky = _integer_wavenumbers(Ny)
    kz = _integer_wavenumbers(Nz)
    cutoff_x = (Nx - 1) // 3
    cutoff_y = (Ny - 1) // 3
    cutoff_z = (Nz - 1) // 3
    mask_x = (np.abs(kx) <= cutoff_x).astype(float)
    mask_y = (np.abs(ky) <= cutoff_y).astype(float)
    mask_z = (np.abs(kz) <= cutoff_z).astype(float)
    return mask_z[:, None, None] * mask_y[None, :, None] * mask_x[None, None, :]


def _build_exponential_filter_rate_3d(
    Nx: int,
    Ny: int,
    Nz: int,
    p: int,
    rate: float,
    dealias: bool,
    onset_fraction: float = 0.0,
) -> np.ndarray:
    """Per-mode decay rate, summed over the three axis ramps.

    A mode at the cutoff on every axis has ``lambda = 3 * rate`` and an
    e-folding time ``tau / 3``, where ``tau`` is
    ``e_folding_time_at_cutoff`` for a single axis.
    """
    kx = (np.fft.fftfreq(Nx) * Nx).astype(float)
    ky = (np.fft.fftfreq(Ny) * Ny).astype(float)
    kz = (np.fft.fftfreq(Nz) * Nz).astype(float)
    scale_x = (Nx - 1) // 3 if dealias else int(np.max(np.abs(kx)))
    scale_y = (Ny - 1) // 3 if dealias else int(np.max(np.abs(ky)))
    scale_z = (Nz - 1) // 3 if dealias else int(np.max(np.abs(kz)))
    eta_x = np.abs(kx) / max(scale_x, 1)
    eta_y = np.abs(ky) / max(scale_y, 1)
    eta_z = np.abs(kz) / max(scale_z, 1)
    return rate * (
        _spectral_viscosity_profile(eta_z, p, onset_fraction)[:, None, None]
        + _spectral_viscosity_profile(eta_y, p, onset_fraction)[None, :, None]
        + _spectral_viscosity_profile(eta_x, p, onset_fraction)[None, None, :]
    )


@dataclass
class Grid1D:
    """1D grid and spectral operators with pluggable basis.

    Default is periodic Fourier basis on a uniform grid. Optionally, a
    Legendre or Chebyshev–Gauss–Lobatto basis supports non-periodic BCs.

    Attributes
    ----------
    N : int
        Number of spatial points.
    Lx : float
        Domain length.
    basis : str
        Spectral basis: "fourier" (default), "legendre", or "chebyshev".
    bc : Optional[str]
        Boundary condition for non-periodic bases (e.g., "dirichlet").
    dealias : bool
        If True, use the 2/3 Fourier mask or 3/2 polynomial flux overintegration.
    filter_params : Optional[Dict[str, Any]]
        Basis-specific stabilization configuration. Fourier grids accept only
        timestep-aware ``spectral_dissipation`` settings; polynomial grids use an
        independent accepted-step ``modal_filter``.
    """

    N: int
    Lx: float
    basis: str = "fourier"
    bc: Optional[str | Dict[str, Any]] = None
    dealias: bool = True
    filter_params: Optional[Dict[str, Any]] = None
    fft_workers: int = 1
    backend: str = "auto"  # Metal, CUDA, then NumPy CPU
    torch_device: Optional[str] = None
    precision: str = "double"  # "single" or "double"
    debug: bool = False
    problem_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.backend, self.torch_device = resolve_backend(
            self.backend, self.torch_device
        )

        if int(self.N) != self.N or self.N < 2:
            raise ValueError("Grid1D requires at least two nodes")
        if not np.isfinite(self.Lx) or self.Lx <= 0.0:
            raise ValueError("Lx must be positive and finite")

        # Normalize global, per-variable, or per-endpoint boundary conditions.
        def _normalize_bc_value(name: str) -> str:
            n = str(name).strip().lower()
            if n in ("reflective", "reflecting", "wall"):
                return "wall"
            if n in ("neumann", "open", "outflow"):
                return "outflow"
            if n in ("dirichlet", "fixed"):
                return "fixed"
            if n == "periodic":
                return n
            raise ValueError(f"Unknown 1D boundary condition: {name}")

        def _normalize_var_name(v: str) -> str:
            n = str(v).strip().lower()
            if n in ("rho", "density"):
                return "density"
            if n in ("p", "pressure"):
                return "pressure"
            if n in ("u", "velocity", "mom", "momentum", "momentum_x"):
                return "momentum"
            if n not in ("momentum_y", "momentum_z", "bx", "by", "bz", "default"):
                raise ValueError(f"Unknown 1D boundary variable: {v}")
            return n

        def _normalize_boundary_name(b: str) -> str:
            n = str(b).strip().lower()
            if n in ("left", "x0", "x_min", "start"):
                return "left"
            if n in ("right", "x1", "x_max", "end"):
                return "right"
            raise ValueError(f"Unknown 1D boundary: {b}")

        # bc may be a string, per-variable dict, or per-boundary dict
        self._bc_global: str | None = None
        self._bc_map: dict[str, str] | None = None
        self._bc_boundary_map: dict[str, dict[str, str]] | None = None  # {boundary: {var: bc}}
        self._dirichlet_values_map: dict[str, tuple[float, float]] | None = None
        self._boundary_reference = None
        
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
                        self._bc_boundary_map[boundary] = {"default": bc_value}
            else:
                # Per-variable configuration: {var: bc}
                self._bc_map = {}
                for k, v in self.bc.items():
                    self._bc_map[_normalize_var_name(k)] = _normalize_bc_value(v)
        elif isinstance(self.bc, str) and self.bc:
            self._bc_global = _normalize_bc_value(self.bc)
        elif self.bc is not None:
            raise ValueError("1D bc must be a boundary name or mapping")

        # Basis selection (Fourier default)
        self._basis_name = str(self.basis).lower()
        configured_bcs = [self._bc_global] if self._bc_global else []
        if self._bc_map:
            configured_bcs.extend(self._bc_map.values())
        if self._bc_boundary_map:
            for values in self._bc_boundary_map.values():
                configured_bcs.extend(values.values())
        if self._basis_name == "fourier" and any(b != "periodic" for b in configured_bcs):
            raise ValueError("Fourier grids require periodic boundary conditions")
        if self._basis_name != "fourier" and "periodic" in configured_bcs:
            raise ValueError("Periodic boundary conditions require the Fourier basis")

        if self._basis_name == "fourier":
            # Existing periodic uniform grid path
            self.dx = self.Lx / self.N
            self.x = np.linspace(0.0, self.Lx, num=self.N, endpoint=False)
            # Physical-to-spectral wave numbers (2*pi/L scaling)
            k_index = np.fft.fftfreq(self.N, d=self.dx)  # cycles per length
            self.k = 2.0 * np.pi * k_index
            self.ik = 1j * self.k

            # Hard dealias mask and optional timestep-aware spectral dissipation.
            self.dealias_mask = _build_filter_mask(self.N, self.dealias)

            enabled, p, rate, onset = _spectral_dissipation_settings(
                self.filter_params
            )
            self.spectral_dissipation_enabled = enabled
            self.modal_filter_enabled = False
            self.modal_filter_order = 8
            self.modal_filter_alpha = 36.0
            self.filter_rate = _build_exponential_filter_rate(
                self.N,
                p=p,
                rate=rate,
                dealias=self.dealias,
                onset_fraction=onset,
            )

            # Initialize Fourier basis instance (NumPy path used inside helpers)
            try:
                from .basis.fourier import FourierBasis1D

                self._basis = FourierBasis1D(self.N, self.Lx, fft_workers=self.fft_workers)
            except Exception:
                self._basis = None  # Fallback to direct FFT wrappers below
        elif self._basis_name in ("legendre", "chebyshev"):
            # Lobatto nodes on [0, Lx], including both endpoints.
            try:
                from .basis.legendre import LegendreLobattoBasis1D

                bc = self._bc_global or "wall"
                # Check if parallel threading is enabled via runtime.num_threads
                use_parallel = False
                if self.problem_config:
                    runtime_cfg = self.problem_config.get("runtime", {})
                    num_threads_cfg = runtime_cfg.get("num_threads", None)
                    if num_threads_cfg is not None and num_threads_cfg != 1:
                        use_parallel = True
                if self._basis_name == "chebyshev":
                    from .basis.chebyshev import ChebyshevLobattoBasis1D

                    self._basis = ChebyshevLobattoBasis1D(
                        self.N, self.Lx, bc=bc, precision=self.precision
                    )
                else:
                    self._basis = LegendreLobattoBasis1D(
                        self.N, self.Lx, bc=bc, precision=self.precision,
                        use_parallel=use_parallel,
                    )
                self.x = self._basis.nodes()
                x_np = np.asarray(self.x)
                if self.N > 1:
                    self.dx = float(np.min(np.diff(x_np)))
                else:
                    self.dx = self.Lx
                # No Fourier masks in non-periodic modal bases
                self.dealias_mask = np.ones(self.N, dtype=float)
                self.spectral_dissipation_enabled = False
                self.filter_rate = np.zeros(self.N, dtype=float)
                (
                    self.modal_filter_enabled,
                    self.modal_filter_order,
                    self.modal_filter_alpha,
                ) = _modal_filter_settings(self.filter_params)
                # Expose quadrature weights for monitoring integrations
                try:
                    self.wx = self._basis.quadrature_weights()
                except Exception:
                    self.wx = None
            except Exception as e:
                raise RuntimeError(f"{self._basis_name} basis initialization failed: {e}")
        else:
            raise ValueError(f"Unknown basis: {self.basis}")

        self._setup_backend()

    def _setup_backend(self) -> None:
        """Initialize arrays on the resolved backend for every 1D basis."""
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
        )
        if self._use_torch:
            assert torch is not None
            _validate_torch_device(self.torch_device, self.debug)
            torch_dtype, torch_cdtype, self.precision = _torch_dtypes(
                self.precision, self.torch_device
            )

            self.x = torch.from_numpy(np.asarray(self.x)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            if self._basis_name == "fourier":
                k_t = torch.from_numpy(np.asarray(self.k)).to(
                    dtype=torch_dtype, device=self.torch_device
                )
                self.k = k_t
                self.ik = k_t.to(dtype=torch_cdtype) * (1j)
                self.dealias_mask = torch.from_numpy(self.dealias_mask).to(
                    dtype=torch_dtype, device=self.torch_device
                )
                self.filter_rate = torch.from_numpy(self.filter_rate).to(
                    dtype=torch_dtype, device=self.torch_device
                )
        elif self.precision == "single" and self._basis_name == "fourier":
            # Coordinates stay float64 so they match the FFT operators.
            _note_numpy_fourier_single()
        elif self.precision == "single":
            self.x = np.asarray(self.x).astype(np.float32, copy=False)

    def set_dirichlet_bc_values(self, values: dict[str, tuple[float, float]]) -> None:
        """Set fixed endpoint density, momenta, thermal pressure, and B values.

        Each value is a (left, right) pair. Unspecified fixed values are taken
        from the initial endpoint state. Momentum aliases denote rho*u.
        """
        aliases = {"rho": "density", "p": "pressure", "mom": "momentum",
                   "u": "momentum", "velocity": "momentum", "momentum_x": "momentum"}
        allowed = {"density", "pressure", "momentum", "momentum_y", "momentum_z", "bx", "by", "bz"}
        normalized = {}
        for key, pair in values.items():
            name = str(key).lower()
            name = aliases.get(name, name)
            if name not in allowed or len(pair) != 2 or not np.all(np.isfinite(pair)):
                raise ValueError(f"Invalid fixed 1D boundary value: {key}")
            if name == "bx" and pair[0] != pair[1]:
                raise ValueError("1D fixed Bx values must match on both sides")
            normalized[name] = (float(pair[0]), float(pair[1]))
        self._dirichlet_values_map = normalized

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

    def dx1(self, f: np.ndarray) -> np.ndarray:
        """Derivative along the last axis.

        Fourier spectra are multiplied by the 2/3-rule dealias mask
        (integer modes ``|k| <= (N - 1) // 3`` when ``dealias`` is true)
        and then by ``i k``. The mask is part of this operator, not only
        of a separate product step. The stepper projects the conserved
        state onto that retained band after every accepted step and
        differentiates the nonlinear flux with this method, so the mask
        belongs on that flux path. Linear callers — sensors, resistive
        and viscous Laplacians, and divergence diagnostics — drop the
        same modes. Polynomial bases have no Fourier mask.
        """
        # Supports f with shape (..., N)
        if self._basis_name == "fourier":
            F = self.rfft(f)
            F *= self.dealias_mask
            dF = self.ik * F
            return self.irfft(dF)
        else:
            # Generic basis path (e.g., Legendre): delegate to basis implementation
            return self._basis.dx(f)  # type: ignore[union-attr]

    def flux_derivative(self, U, equations, return_boundary_flux=False):
        """Differentiate nonlinear flux with 3/2 polynomial overintegration.

        At inadmissible extra nodes, use the convex, piecewise-linear state
        between adjacent original nodes. This changes only flux evaluation;
        it does not clip the evolved conservative state or its energy.

        With ``return_boundary_flux`` the endpoint values of the flux
        polynomial that is differentiated are also returned, shape ``(C, 2)``.
        They are the boundary fluxes consistent with the derivative, which
        :meth:`apply_boundary_flux` needs; with overintegration they differ
        from the flux of the endpoint states.
        """
        if self._basis_name == "fourier" or not self.dealias:
            flux = equations.flux(U)
            derivative = self.dx1(flux)
            if return_boundary_flux:
                return derivative, flux[:, [0, -1]]
            return derivative
        if not hasattr(self, "_flux_matrices"):
            n_pad = (3 * self.N + 1) // 2
            precision = self._basis.precision
            dtype = np.float32 if precision == "single" else np.float64
            padded = type(self._basis)(n_pad, self.Lx, precision=precision)
            identity, padded_identity = np.eye(self.N, dtype=dtype), np.eye(n_pad, dtype=dtype)
            modes = self._basis.inverse(identity)
            interpolation = self._basis.forward(identity) @ padded.inverse(padded_identity)[:self.N]
            mode_derivatives = self._basis.dx(modes)
            mode_derivatives[0] = 0.0
            projection = padded.forward(padded_identity)[:, :self.N]
            differentiation = projection @ mode_derivatives
            endpoints = projection @ modes[:, [0, -1]]
            x, xp = np.asarray(self._basis.nodes()), padded.nodes()
            left = np.clip(np.searchsorted(x, xp, side="right") - 1, 0, self.N - 2)
            fraction = (xp - x[left]) / (x[left + 1] - x[left])
            self._flux_matrices = (interpolation, differentiation, left, fraction, endpoints)
            self._flux_backend_cache = {}
        is_torch = _TORCH_AVAILABLE and isinstance(U, torch.Tensor)
        key = (str(U.device) if is_torch else "numpy", str(U.dtype))
        if key not in self._flux_backend_cache:
            interp, diff, left, fraction, endpoints = self._flux_matrices
            if is_torch:
                cached = (torch.as_tensor(interp, dtype=U.dtype, device=U.device),
                          torch.as_tensor(diff, dtype=U.dtype, device=U.device),
                          torch.as_tensor(left, dtype=torch.long, device=U.device),
                          torch.as_tensor(fraction, dtype=U.dtype, device=U.device),
                          torch.as_tensor(endpoints, dtype=U.dtype, device=U.device))
            else:
                cached = (interp.astype(U.dtype), diff.astype(U.dtype),
                          left, fraction.astype(U.dtype), endpoints.astype(U.dtype))
            self._flux_backend_cache[key] = cached
        interp, diff, left, fraction, endpoints = self._flux_backend_cache[key]
        padded_U = (U - U[:, :1]) @ interp + U[:, :1]
        energy = 4 if U.shape[0] == 8 else 2
        rho = padded_U[0]
        kinetic = sum(padded_U[i] ** 2 for i in range(1, energy)) / (2.0 * rho)
        magnetic = 0.5 * sum(padded_U[i] ** 2 for i in (5, 6, 7)) if energy == 4 else 0.0
        pressure = (equations.gamma - 1.0) * (padded_U[energy] - kinetic - magnetic)
        bad = (rho <= 1e-10) | (pressure <= 1e-12)
        linear_U = U[:, left] + (U[:, left + 1] - U[:, left]) * fraction
        if is_torch:
            padded_U = torch.where(bad[None, :], linear_U, padded_U)
        else:
            padded_U = np.where(bad[None, :], linear_U, padded_U)
        flux = equations.flux(padded_U)
        derivative = (flux - flux[:, :1]) @ diff
        if return_boundary_flux:
            return derivative, (flux - flux[:, :1]) @ endpoints + flux[:, :1]
        return derivative

    def project_dealiased(self, f: np.ndarray) -> np.ndarray:
        """Project a Fourier field onto the retained 2/3 modal subspace."""
        if self._basis_name == "fourier":
            F = self.rfft(f)
            F *= self.dealias_mask
            return self.irfft(F)
        return f

    def apply_spectral_dissipation(
        self, f: np.ndarray, dt: float, *, project: bool = False
    ) -> np.ndarray:
        """Apply exp(-dt*lambda(k)); optionally combine the dealias projection."""
        if dt < 0.0:
            raise ValueError("spectral dissipation dt must be non-negative")
        if self._basis_name != "fourier":
            return f
        if not self.spectral_dissipation_enabled:
            return self.project_dealiased(f) if project else f
        F = self.rfft(f)
        if project:
            F *= self.dealias_mask
        if getattr(self, "_use_torch", False):
            assert torch is not None
            F *= torch.exp(-float(dt) * self.filter_rate)
        else:
            F *= np.exp(-float(dt) * self.filter_rate)
        return self.irfft(F)

    def apply_modal_filter(self, f: np.ndarray) -> np.ndarray:
        """Apply the polynomial basis accepted-step modal filter."""
        if self._basis_name == "fourier" or not self.modal_filter_enabled:
            return f
        return self._basis.apply_spectral_filter(  # type: ignore[union-attr]
            f, p=self.modal_filter_order, alpha=self.modal_filter_alpha
        )

    def apply_accepted_state_stabilization(
        self, f: np.ndarray, dt: float
    ) -> np.ndarray:
        """Project Fourier states or apply the polynomial modal filter."""
        if self._basis_name == "fourier":
            return self.apply_spectral_dissipation(f, dt, project=True)
        return self.apply_modal_filter(f)

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        """Backward-compatible accepted-state stabilization entrypoint."""
        return self.apply_accepted_state_stabilization(f, 0.0)

    def convolve(self, f: np.ndarray, g: np.ndarray) -> np.ndarray:
        """Compute product in physical space, with dealiased spectral filtering.

        For pseudo-spectral nonlinearity, compute pointwise product then
        apply the hard dealias projection to the spectral representation.
        """
        h = f * g
        return self.project_dealiased(h)

    # --- MHD solenoidal helpers (1D) ---
    def divergence_x(self, Bx: np.ndarray) -> np.ndarray:
        """1D divergence ``d(Bx)/dx``.

        This is :meth:`dx1`, so a Fourier result includes the 2/3 dealias
        mask described there.
        """
        return self.dx1(Bx)

    def project_solenoidal_x(self, Bx: np.ndarray) -> np.ndarray:
        """1D Helmholtz (Leray) projection of the x-magnetic field.

        In 1D the only solenoidal constraint is d(Bx)/dx = 0, whose unique
        solution is Bx = const. Keep the mean (quadrature-weighted on Lobatto
        grids) so that div(B) is zero to roundoff. The
        transverse components By, Bz are unconstrained in 1D and untouched here.
        """
        if self._basis_name != "fourier":
            if getattr(self, "_use_torch", False):
                weights = torch.as_tensor(self.wx, dtype=Bx.dtype, device=Bx.device)
                return torch.ones_like(Bx) * (Bx * weights).sum() / weights.sum()
            return np.full_like(Bx, np.dot(Bx, self.wx) / np.sum(self.wx))
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(Bx, torch.Tensor):
            return torch.ones_like(Bx) * torch.mean(Bx)
        Bx = np.asarray(Bx)
        return np.full_like(Bx, float(Bx.mean()))

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """Second derivative along the last axis.

        The Fourier result uses the same 2/3 dealias mask as :meth:`dx1`,
        including when the Laplacian is a linear diffusion operator.
        Polynomial bases use the unmasked second-derivative matrix.
        """
        if self._basis_name != "fourier":
            return self._basis.d2x(f)
        F = self.rfft(f)
        F = F * self.dealias_mask
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(F, torch.Tensor):
            k2 = (self.k ** 2).to(F.dtype)
        else:
            k2 = np.asarray(self.k) ** 2
        return self.irfft(-k2 * F)

    def diffusion(self, f):
        """Second derivative used for diffusion terms, with zero boundary flux.

        Fourier grids return :meth:`laplacian`. On Chebyshev and Legendre grids
        the endpoint rows drop the boundary term of the summation-by-parts
        identity ``H D D f = B D f - D^T H D f``, i.e. the diffusive flux
        through the endpoints is zero. That is the closure that ``outflow``
        and ``wall`` endpoints need; ``fixed`` endpoints are overwritten after
        each stage, so their result does not depend on it. Interior rows equal
        :meth:`laplacian`.
        """
        result = self.laplacian(f)
        if self._basis_name == "fourier":
            return result
        gradient = self.dx1(f)
        if _TORCH_AVAILABLE and isinstance(result, torch.Tensor):
            result = result.clone()
        else:
            result = np.array(result, copy=True)
        weights = self.wx
        result[..., 0] = result[..., 0] + gradient[..., 0] / float(weights[0])
        result[..., -1] = result[..., -1] - gradient[..., -1] / float(weights[-1])
        return result

    def interpolate_periodic_linear_at_points_1d(
        self, f: np.ndarray, xp: np.ndarray
    ) -> np.ndarray:
        """Interpolate a periodic field at points on its current array backend."""
        if self._basis_name != "fourier":
            raise ValueError(
                "interpolate_periodic_linear_at_points_1d requires Fourier basis"
            )
        return _periodic_linear_1d(f, xp, self.Lx)

    def evaluate_fourier_at_points_1d(self, f: np.ndarray, xp: np.ndarray) -> np.ndarray:
        """Evaluate a Fourier field at arbitrary points.

        ``f(x) = (1/N) * Re(sum_k F_k exp(i*k*x))`` on the retained band.
        The same 2/3-rule dealias mask used by :meth:`dx1` is applied
        before the sum. The stepper projects the conserved state onto
        that band after every accepted step. A derived field such as
        velocity ``= momentum / density`` can still contain modes with
        ``|k| > (N - 1) // 3``, and those modes are discarded here.
        Supports NumPy and PyTorch (including MPS/CUDA); tensors stay
        on their device.

        Args:
            f: Field on the grid, shape ``(N,)``. Batches are rejected.
            xp: Query points, shape ``(P,)``.

        Returns:
            Values at ``xp``, shape ``(P,)``. Same dtype/device as ``f``.
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
        if np.ndim(f_np) != 1:
            raise ValueError(
                "evaluate_fourier_at_points_1d expects one field of shape "
                f"(N,); got shape {np.shape(f_np)}"
            )
        dealias = _to_np(self.dealias_mask)
        f_flat = np.reshape(f_np, (-1, self.N))
        F = np.fft.fft(f_flat, axis=-1)
        F = np.ascontiguousarray((F * dealias)[0], dtype=np.complex128)
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
    def initialize_boundary_conditions(self, U, equations) -> np.ndarray:
        """Capture fixed endpoints once at the start of a run."""
        self._boundary_reference = None
        return self.apply_boundary_conditions(U, equations)

    def _boundary_layout(self, U):
        """Conservative rows of the boundary variables and the energy row."""
        mhd = U.shape[0] == 8
        variables = {"density": 0, "momentum": 1}
        if mhd:
            variables.update(momentum_y=2, momentum_z=3, bx=5, by=6, bz=7)
        return variables, (4 if mhd else 2), mhd

    def _boundary_kinds(self, side, names):
        """Boundary kind of every variable on one side (0 left, 1 right)."""
        boundary = "left" if side == 0 else "right"
        mapping = (self._bc_boundary_map or {}).get(boundary, self._bc_map or {})
        default = mapping.get("default", self._bc_global or "wall")
        return {name: mapping.get(name, default) for name in names}

    def _boundary_fields(self, a, equations):
        """Primitive boundary fields of conservative columns ``a``."""
        variables, energy, mhd = self._boundary_layout(a)
        values = {name: a[row] for name, row in variables.items()}
        kinetic = sum(a[row] ** 2 for row in range(1, energy)) / (2.0 * a[0])
        magnetic = 0.5 * sum(a[row] ** 2 for row in (5, 6, 7)) if mhd else 0.0
        values["pressure"] = (equations.gamma - 1.0) * (a[energy] - kinetic - magnetic)
        return values

    def _fixed_boundary_value(self, name, side, reference):
        pair = (self._dirichlet_values_map or {}).get(name)
        return pair[side] if pair is not None else reference[name][side]

    def apply_boundary_conditions(self, U: np.ndarray, equations) -> np.ndarray:
        """Return a state with the fixed endpoint values imposed.

        ``fixed`` (Dirichlet) variables are set strongly to the initial
        endpoint values or ``grid.bc_values``. ``outflow`` and ``wall``
        variables are left untouched here: they are imposed weakly by
        :meth:`apply_boundary_flux` in the right-hand side. Pressure is thermal
        pressure; energy includes all kinetic and magnetic terms. The input,
        including Runge--Kutta stage arrays, is never modified.
        """
        if self._basis_name == "fourier":
            return U
        if U.ndim != 2 or U.shape[-1] != self.N or U.shape[0] not in (3, 8):
            raise ValueError("1D boundary conditions require a (3,N) hydro or (8,N) MHD state")
        if equations is None:
            raise ValueError("1D boundary conditions require equations")
        is_torch = _TORCH_AVAILABLE and isinstance(U, torch.Tensor)

        def copy(a):
            return a.clone() if is_torch else a.copy()

        if self._boundary_reference is None:
            self._boundary_reference = copy(U[:, [0, -1]])
        variables, energy, mhd = self._boundary_layout(U)
        names = (*variables, "pressure")
        reference = self._boundary_fields(self._boundary_reference, equations)
        current = self._boundary_fields(copy(U[:, [0, -1]]), equations)
        result = None
        for side, endpoint in enumerate((0, -1)):
            kinds = self._boundary_kinds(side, names)
            if not any(kind == "fixed" for kind in kinds.values()):
                continue
            if result is None:
                result = copy(U)
            values = {}
            for name in names:
                if kinds[name] == "fixed":
                    value = self._fixed_boundary_value(name, side, reference)
                    pair = (self._dirichlet_values_map or {}).get(name)
                    if mhd and name == "bx" and pair is not None:
                        tolerance = 32.0 * (torch.finfo(U.dtype).eps if is_torch else np.finfo(U.dtype).eps)
                        if abs(float(value) - float(U[5, endpoint])) > tolerance * max(1.0, abs(float(value))):
                            raise ValueError("Fixed Bx must match the longitudinal interior field")
                else:
                    value = current[name][side]
                values[name] = value
            for name, row in variables.items():
                result[row, endpoint] = values[name]
            rho = values["density"]
            if float(rho) <= 0.0 or float(values["pressure"]) <= 0.0:
                raise ValueError("Boundary density and pressure must be positive")
            kinetic = sum(result[row, endpoint] ** 2 for row in range(1, energy)) / (2.0 * rho)
            magnetic = 0.5 * sum(result[row, endpoint] ** 2 for row in (5, 6, 7)) if mhd else 0.0
            result[energy, endpoint] = values["pressure"] / (equations.gamma - 1.0) + kinetic + magnetic
        return U if result is None else result

    def apply_boundary_flux(self, rhs, U, equations, boundary_flux=None):
        """Weakly impose ``outflow`` and ``wall`` endpoints with an SBP/Rusanov flux.

        At each endpoint that is not entirely ``fixed`` the right-hand side is
        corrected by ``-(F*(u_b, g) - F(u_b)) . n / w_b``, where ``w_b`` is the
        endpoint quadrature weight and ``F*`` the Rusanov flux against an
        exterior state ``g``. For ``outflow`` the exterior is the initial
        endpoint state (a far-field closure through which waves leave); for
        ``wall`` it is the mirror state with the normal momentum reversed, so
        the mass flux through the wall is exactly zero. ``fixed`` variables
        take their fixed values in ``g`` and are also imposed strongly by
        :meth:`apply_boundary_conditions`. ``boundary_flux`` is the endpoint
        flux consistent with the volume derivative (see
        :meth:`flux_derivative`); it defaults to the flux of the endpoint
        states. Copying endpoint values instead is not a characteristic
        closure for the global polynomial derivative and lets the endpoint
        states drift.
        """
        if self._basis_name == "fourier":
            return rhs
        if self._boundary_reference is None:
            self.initialize_boundary_conditions(U, equations)
        is_torch = _TORCH_AVAILABLE and isinstance(U, torch.Tensor)
        variables, energy, mhd = self._boundary_layout(U)
        names = (*variables, "pressure")
        boundary_states = U[:, [0, -1]]
        reference = self._boundary_fields(self._boundary_reference, equations)
        current = self._boundary_fields(boundary_states, equations)
        gamma = equations.gamma
        sqrt = torch.sqrt if is_torch else np.sqrt
        absolute = torch.abs if is_torch else np.abs

        def speed(state):
            rho = state[0]
            fields = self._boundary_fields(state, equations)
            c2 = gamma * fields["pressure"] / rho
            if mhd:
                c2 = c2 + (state[5] ** 2 + state[6] ** 2 + state[7] ** 2) / rho
            c2 = torch.clamp(c2, min=0) if is_torch else np.maximum(c2, 0.0)
            return absolute(state[1] / rho) + sqrt(c2)

        weights = self.wx
        for side, endpoint, normal in ((0, 0, -1.0), (1, -1, 1.0)):
            kinds = self._boundary_kinds(side, names)
            if all(kind == "fixed" for kind in kinds.values()):
                continue
            values = {}
            for name in names:
                kind = kinds[name]
                if kind == "fixed":
                    values[name] = self._fixed_boundary_value(name, side, reference)
                elif kind == "outflow":
                    values[name] = reference[name][side]
                elif name == "momentum":
                    values[name] = -current[name][side]
                else:
                    values[name] = current[name][side]
            u_b = boundary_states[:, side:side + 1]
            g = u_b.clone() if is_torch else u_b.copy()
            for name, row in variables.items():
                g[row, 0] = values[name]
            kinetic = sum(g[row, 0] ** 2 for row in range(1, energy)) / (2.0 * g[0, 0])
            magnetic = 0.5 * sum(g[row, 0] ** 2 for row in (5, 6, 7)) if mhd else 0.0
            g[energy, 0] = values["pressure"] / (gamma - 1.0) + kinetic + magnetic
            flux_b = (equations.flux(u_b)[:, 0] if boundary_flux is None
                      else boundary_flux[:, side])
            flux_g = equations.flux(g)[:, 0]
            signal = max(float(speed(u_b)[0]), float(speed(g)[0]))
            outward = normal * flux_b
            numerical = 0.5 * (outward + normal * flux_g) - 0.5 * signal * (g[:, 0] - u_b[:, 0])
            rhs[:, endpoint] = rhs[:, endpoint] - (numerical - outward) / float(weights[endpoint])
        return rhs


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
    filter_params: Optional[Dict[str, Any]] = None
    fft_workers: int = 1
    backend: str = "auto"  # Metal, CUDA, then NumPy CPU
    torch_device: Optional[str] = None
    precision: str = "double"  # "single" or "double"
    debug: bool = False
    basis_x: str = "fourier"  # "fourier" or "legendre"
    basis_y: str = "fourier"  # "fourier", "legendre", or "sbp" (open y)
    bc: str = "dirichlet"     # Simple boundary condition like SOD
    # Boundary condition configuration
    bc_config: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.backend, self.torch_device = resolve_backend(
            self.backend, self.torch_device
        )
        self._mixed_open_y = self.basis_x == "fourier" and self.basis_y == "sbp"
        sbp_params = (self.filter_params or {}).get("sbp_dissipation", {})
        if not isinstance(sbp_params, dict):
            raise TypeError("sbp_dissipation must be a mapping")
        strength = float(sbp_params.get("strength", 0.0))
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("SBP dissipation strength must be finite and nonnegative")
        enabled = bool(sbp_params.get("enabled", False))
        if enabled and not self._mixed_open_y:
            raise ValueError("SBP dissipation requires Fourier x and SBP y")
        self.sbp_dissipation_strength = strength if enabled else 0.0
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
            # Pure Fourier, or Fourier in x with SBP in y.
            self._legendre_basis = None
            if self.basis_x == "legendre" or self.basis_y == "legendre":
                raise NotImplementedError(
                    "Mixed Fourier/Legendre grids are not implemented. "
                    "Use Legendre on both axes, Fourier on both axes, or "
                    "Fourier in x with SBP in y."
                )

            if self.basis_x == "fourier":
                self.x = np.linspace(0.0, self.Lx, num=self.Nx, endpoint=False)
                kx_index = np.fft.fftfreq(self.Nx, d=self.dx)
                self.kx = 2.0 * np.pi * kx_index  # shape (Nx,)
                self.ikx = 1j * self.kx[None, :]  # shape (1, Nx)
                self.Dx = None  # Will use FFT for derivatives
            else:
                raise ValueError(f"Unknown basis_x: {self.basis_x}")

            if self.basis_y == "fourier":
                self.y = np.linspace(0.0, self.Ly, num=self.Ny, endpoint=False)
                ky_index = np.fft.fftfreq(self.Ny, d=self.dy)
                self.ky = 2.0 * np.pi * ky_index  # shape (Ny,)
                self.iky = 1j * self.ky[:, None]  # shape (Ny, 1)
                self.Dy = None  # Will use FFT for derivatives
            elif self._mixed_open_y:
                if self.Ny < 9:
                    raise ValueError("The open SBP direction requires Ny >= 9")
                self.dy = self.Ly / (self.Ny - 1)
                self.dy_min = self.dy
                self.y = np.linspace(0.0, self.Ly, self.Ny)
                self.ky = self.iky = None
                self._setup_sbp_y()
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

        # Hard dealias projection and optional timestep-aware Fourier dissipation.
        self.dealias_mask = _build_filter_mask_2d(self.Nx, self.Ny, self.dealias)
        if self._mixed_open_y:
            self.dealias_mask = np.broadcast_to(
                _build_filter_mask(self.Nx, self.dealias), (self.Ny, self.Nx)
            ).copy()
        pure_fourier = self.basis_x == "fourier" and self.basis_y == "fourier"
        enabled, p, rate, onset = _spectral_dissipation_settings(
            self.filter_params if pure_fourier else None
        )
        self.spectral_dissipation_enabled = enabled
        self.filter_rate = _build_exponential_filter_rate_2d(
            self.Nx,
            self.Ny,
            p=p,
            rate=rate,
            dealias=self.dealias,
            onset_fraction=onset,
        )
        if self._legendre_basis is not None:
            (
                self.modal_filter_enabled,
                self.modal_filter_order,
                self.modal_filter_alpha,
            ) = _modal_filter_settings(self.filter_params)
        else:
            self.modal_filter_enabled = False
            self.modal_filter_order = 8
            self.modal_filter_alpha = 36.0

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
            _validate_torch_device(self.torch_device, self.debug)
            torch_dtype, torch_cdtype, self.precision = _torch_dtypes(
                self.precision, self.torch_device
            )

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
            self.filter_rate = torch.from_numpy(np.asarray(self.filter_rate)).to(
                dtype=torch_dtype, device=self.torch_device
            )
            if self.k2_safe is not None:
                self.k2 = torch.from_numpy(np.asarray(self.k2)).to(
                    dtype=torch_dtype, device=self.torch_device
                )
                self.k2_safe = torch.from_numpy(np.asarray(self.k2_safe)).to(
                    dtype=torch_dtype, device=self.torch_device
                )
        elif self.precision == "single" and self.basis_x == "fourier":
            _note_numpy_fourier_single()

    def _setup_sbp_y(self) -> None:
        """Fourth-order interior, second-order boundary diagonal-norm SBP."""
        rows = np.zeros((4, 6))
        rows[0, :4] = [-24/17, 59/34, -4/17, -3/34]
        rows[1, :3] = [-1/2, 0, 1/2]
        rows[2, :5] = [4/43, -59/86, 0, 59/86, -4/43]
        rows[3, :6] = [3/98, 0, -59/98, 0, 32/49, -4/49]
        self._sbp_rows = rows / self.dy
        self.wy = np.ones(self.Ny) * self.dy
        self.wy[:4] *= [17/48, 59/48, 43/48, 49/48]
        self.wy[-4:] = self.wy[:4][::-1]
        self._sbp_backend_cache = {}
        self._boundary_reference = None

    def _sbp_arrays(self, ref):
        is_torch = _TORCH_AVAILABLE and isinstance(ref, torch.Tensor)
        key = (str(ref.device) if is_torch else "numpy", str(ref.dtype))
        if key not in self._sbp_backend_cache:
            if is_torch:
                rows = torch.as_tensor(self._sbp_rows, dtype=ref.dtype, device=ref.device)
                weights = torch.as_tensor(self.wy, dtype=ref.real.dtype, device=ref.device)
            else:
                rows = self._sbp_rows.astype(ref.dtype)
                weights = self.wy.astype(ref.real.dtype)
            self._sbp_backend_cache[key] = rows, weights
        return self._sbp_backend_cache[key]

    def _sbp_dy(self, f):
        rows, _ = self._sbp_arrays(f)
        if _TORCH_AVAILABLE and isinstance(f, torch.Tensor):
            result = torch.empty_like(f)
            multiply = torch.einsum
            top = f[..., -6:, :].flip(-2)
        else:
            result = np.empty_like(f)
            multiply = np.einsum
            top = f[..., -6:, :][..., ::-1, :]
        result[..., 4:-4, :] = (
            8.0 * (f[..., 5:-3, :] - f[..., 3:-5, :])
            - (f[..., 6:-2, :] - f[..., 2:-6, :])
        ) / (12.0 * self.dy)
        result[..., :4, :] = multiply("ij,...jx->...ix", rows,
                                     f[..., :6, :] - f[..., :1, :])
        upper = -multiply("ij,...jx->...ix", rows, top - top[..., :1, :])
        result[..., -4:, :] = upper.flip(-2) if _TORCH_AVAILABLE and isinstance(upper, torch.Tensor) else upper[..., ::-1, :]
        return result

    def dy_adjoint(self, f):
        """Adjoint of the mixed-grid y derivative in its SBP quadrature."""
        if not self._mixed_open_y:
            return -self.dy1(f)
        _, weights = self._sbp_arrays(f)
        result = -self.dy1(f)
        result[..., 0, :] -= f[..., 0, :] / weights[0]
        result[..., -1, :] += f[..., -1, :] / weights[-1]
        return result

    def sbp_dissipation_rate_bound(self, speed):
        """Upper bound for the nonnegative KO6 decay rates in the SBP norm."""
        speed = float(speed)
        if not np.isfinite(speed) or speed < 0.0:
            raise ValueError("SBP dissipation speed must be finite and nonnegative")
        if self.sbp_dissipation_strength == 0.0:
            return 0.0
        return self.sbp_dissipation_strength * speed / float(np.min(self.wy))

    def sbp_dissipation(self, f, speed):
        """Conservative H-negative KO6 RHS, using unwrapped y differences.

        Q = -strength*speed/64 H^-1 R3.T R3. Its truncation contribution
        is O(dy^5) inside and O(dy^2) at the boundary closure.
        """
        rate_bound = self.sbp_dissipation_rate_bound(speed)
        result = (torch.zeros_like(f) if _TORCH_AVAILABLE and isinstance(f, torch.Tensor)
                  else np.zeros_like(f))
        if rate_bound == 0.0:
            return result
        difference = f[..., 1:, :] - f[..., :-1, :]
        difference = difference[..., 1:, :] - difference[..., :-1, :]
        difference = difference[..., 1:, :] - difference[..., :-1, :]
        result[..., :-3, :] -= difference
        result[..., 1:-2, :] += 3.0 * difference
        result[..., 2:-1, :] -= 3.0 * difference
        result[..., 3:, :] += difference
        _, weights = self._sbp_arrays(f)
        return (-self.sbp_dissipation_strength * float(speed) / 64.0) * result / weights[:, None]

    def _mixed_fft_x(self, f):
        if getattr(self, "_use_torch", False):
            return torch.fft.fft(f, dim=-1)
        return scipy_fft.fft(f, axis=-1, workers=self.fft_workers)

    def _mixed_ifft_x(self, f):
        if getattr(self, "_use_torch", False):
            return torch.fft.ifft(f, dim=-1).real
        return scipy_fft.ifft(f, axis=-1, workers=self.fft_workers).real

    def _parse_boundary_conditions(self) -> None:
        """Parse boundary conditions for a 2D grid.

        With no ``bc_config``, ``bc`` selects one treatment for every side.
        ``dirichlet``, ``wall``, and ``reflective`` are kinematic walls:
        normal momentum is zero, and density, tangential momentum, and the
        total-energy component are copied from the adjacent node. ``neumann``
        copies density, both momenta, and total energy. A Neumann entry
        named ``pressure`` copies that total-energy component. It is not a
        zero gradient of thermal pressure.

        ``bc_config`` must name ``left``, ``right``, ``bottom``, and ``top``.
        Each value is ``neumann`` or a mapping with finite ``density``,
        ``momentum_x``, ``momentum_y``, and thermal ``pressure``. Unknown
        names and incomplete mappings raise ``ValueError``.
        """
        self._bc_boundary_map = {}
        self._dirichlet_values_map = {}
        if self._mixed_open_y:
            aliases = {"outflow": "open", "neumann": "open"}
            for side in ("left", "right", "bottom", "top"):
                default = "periodic" if side in ("left", "right") else "open"
                kind = str((self.bc_config or {}).get(side, default)).lower()
                kind = aliases.get(kind, kind)
                if (side in ("left", "right") and kind != "periodic") or (
                    side in ("bottom", "top") and kind != "open"
                ):
                    raise ValueError("Fourier/SBP grids require periodic x and open y")
                self._bc_boundary_map[side] = kind
            return

        names = ("density", "momentum_x", "momentum_y", "pressure")
        sides = ("left", "right", "bottom", "top")

        def _fill_neumann(boundary: str) -> None:
            self._bc_boundary_map[boundary] = {name: "neumann" for name in names}

        def _fill_wall(boundary: str) -> None:
            normal = "momentum_x" if boundary in ("left", "right") else "momentum_y"
            spec = {name: "neumann" for name in names}
            spec[normal] = "dirichlet"
            self._bc_boundary_map[boundary] = spec
            self._dirichlet_values_map[f"{boundary}_{normal}"] = 0.0

        def _finite_value(boundary: str, key: str, value: Any) -> float:
            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Boundary value {boundary}.{key} must be a real number"
                ) from exc
            if not np.isfinite(number):
                raise ValueError(
                    f"Boundary value {boundary}.{key} must be finite"
                )
            return number

        if not self.bc_config:
            if not isinstance(self.bc, str) or not str(self.bc).strip():
                raise ValueError(
                    f"Unknown 2D boundary condition {self.bc!r}. "
                    "Expected 'dirichlet', 'wall', 'reflective', or 'neumann'."
                )
            default = str(self.bc).strip().lower()
            if default in ("dirichlet", "wall", "reflective"):
                for boundary in sides:
                    _fill_wall(boundary)
                return
            if default == "neumann":
                for boundary in sides:
                    _fill_neumann(boundary)
                return
            raise ValueError(
                f"Unknown 2D boundary condition {self.bc!r}. "
                "Expected 'dirichlet', 'wall', 'reflective', or 'neumann'."
            )

        if not isinstance(self.bc_config, dict):
            raise TypeError("bc_config must be a mapping of boundary names")
        unknown = [name for name in self.bc_config if name not in sides]
        if unknown:
            raise ValueError(f"Unknown 2D boundary {unknown[0]!r} in bc_config")
        missing = [name for name in sides if name not in self.bc_config]
        if missing:
            raise ValueError(
                "bc_config must specify left, right, bottom, and top; "
                f"missing {', '.join(missing)}"
            )
        for boundary in sides:
            config = self.bc_config[boundary]
            if isinstance(config, dict):
                missing_keys = [key for key in names if key not in config]
                if missing_keys:
                    raise ValueError(
                        f"Dirichlet boundary {boundary!r} is missing "
                        f"{', '.join(missing_keys)}. density, momentum_x, "
                        "momentum_y, and pressure are required and are not defaulted."
                    )
                extra = [key for key in config if key not in names]
                if extra:
                    raise ValueError(
                        f"Unknown Dirichlet value {extra[0]!r} for boundary {boundary!r}"
                    )
                self._bc_boundary_map[boundary] = {key: "dirichlet" for key in names}
                for key in names:
                    self._dirichlet_values_map[f"{boundary}_{key}"] = _finite_value(
                        boundary, key, config[key]
                    )
            elif isinstance(config, str) and config.strip().lower() == "neumann":
                _fill_neumann(boundary)
            elif isinstance(config, str) and config.strip().lower() == "dirichlet":
                raise ValueError(
                    f"Boundary {boundary!r} is 'dirichlet' without values. "
                    "Pass density, momentum_x, momentum_y, and pressure."
                )
            else:
                raise ValueError(
                    f"Unknown 2D boundary condition {config!r} for {boundary}. "
                    "Use 'neumann', or a mapping of density, momentum_x, "
                    "momentum_y, and pressure."
                )

    def initialize_boundary_conditions(self, U, equations):
        """Store the initial ambient exterior for the mixed-grid open faces."""
        if self._mixed_open_y:
            boundary = U[:, [0, -1], :]
            self._boundary_reference = (boundary.clone() if _TORCH_AVAILABLE
                                        and isinstance(U, torch.Tensor) else boundary.copy())
        return self.apply_boundary_conditions(U, equations)

    def apply_split_advection(self, rhs, U):
        """Conservative y-advection split with SBP kinetic-energy balance.

        The balance applies before the Fourier projection and excludes pressure,
        magnetic stresses, boundary penalties and dissipation.
        """
        if not self._mixed_open_y:
            return rhs
        momentum = slice(1, 4 if U.shape[0] == 8 else 3)
        velocity = U[momentum]/U[0]
        mass_flux = U[2]
        rhs[momentum] += 0.5*(self.dy1(mass_flux*velocity)
                              - mass_flux*self.dy1(velocity)
                              - velocity*self.dy1(mass_flux))
        return rhs

    def apply_boundary_flux(self, rhs, U, equations, Fy):
        """Strong SBP/Rusanov penalty against the initial ambient exterior.

        Open endpoints evolve with the interior. Copying neighbouring values
        instead traps the high-wavenumber branch of a centred derivative.
        MHD callers project the accepted magnetic state to enforce div(B)=0.
        """
        if not self._mixed_open_y:
            return rhs
        if self._boundary_reference is None:
            self.initialize_boundary_conditions(U, equations)
        exterior = self._boundary_reference
        exterior_flux = equations.flux(exterior)[1]
        is_torch = _TORCH_AVAILABLE and isinstance(U, torch.Tensor)

        def speed(state):
            primitive = equations.primitive(state)
            rho, uy = primitive[0], primitive[2]
            p = primitive[4] if state.shape[0] == 8 else primitive[3]
            c2 = equations.gamma*p/rho
            if state.shape[0] == 8:
                c2 = c2+(state[5:8]**2).sum(axis=0)/rho
            return (torch.abs(uy)+torch.sqrt(torch.clamp(c2, min=0)) if is_torch
                    else np.abs(uy)+np.sqrt(np.maximum(c2, 0)))

        interior = U[:, [0, -1], :]
        maximum = torch.maximum if is_torch else np.maximum
        signal = maximum(speed(interior), speed(exterior))
        for side, endpoint, normal in ((0, 0, -1), (1, -1, 1)):
            outward = normal*Fy[:, endpoint]
            numerical = (0.5*(outward+normal*exterior_flux[:, side])
                         -0.5*signal[side]*(exterior[:, side]-U[:, endpoint]))
            rhs[:, endpoint] -= (numerical-outward)/self.wy[endpoint]
        return rhs

    def apply_boundary_conditions(self, U: np.ndarray, equations) -> np.ndarray:
        """Return a 2D state with boundary conditions applied.

        The input array is not modified. Fourier and Fourier/SBP grids
        return it unchanged; open SBP faces are enforced by the boundary
        flux. On a Legendre grid the returned array is a copy.

        A Neumann ``pressure`` entry copies the total-energy component
        from the adjacent node. It does not impose a zero gradient of
        thermal pressure. A Dirichlet ``pressure`` value is thermal
        pressure and is stored as
        ``E = p / (gamma - 1) + (m_x^2 + m_y^2) / (2 rho)``.
        """
        if self._mixed_open_y:
            if U.shape[0] not in (4, 8) or U.shape[-2:] != (self.Ny, self.Nx):
                raise ValueError("Mixed 2D boundaries require a hydro or MHD state")
            if self._boundary_reference is None and equations is not None:
                self.initialize_boundary_conditions(U, equations)
            return U
        if self._legendre_basis is None:
            return U

        if U.shape[-2:] != (self.Ny, self.Nx):
            return U
        if U.shape[0] < 4:
            return U

        is_torch = _TORCH_AVAILABLE and torch is not None and isinstance(U, torch.Tensor)
        work = U.clone() if is_torch else U.copy()

        def _edge(boundary: str) -> tuple:
            if boundary == "left":
                return (slice(None), 0)
            if boundary == "right":
                return (slice(None), -1)
            if boundary == "bottom":
                return (0, slice(None))
            return (-1, slice(None))

        def _interior(boundary: str) -> tuple:
            if boundary == "left":
                return (slice(None), 1)
            if boundary == "right":
                return (slice(None), -2)
            if boundary == "bottom":
                return (1, slice(None))
            return (-2, slice(None))

        def _has_neighbor(boundary: str) -> bool:
            if boundary in ("left", "right"):
                return self.Nx >= 2
            return self.Ny >= 2

        def _write(comp: int, boundary: str, values) -> None:
            work[(comp,) + _edge(boundary)] = values

        def _read_interior(comp: int, boundary: str):
            return work[(comp,) + _interior(boundary)]

        def _gamma() -> float:
            if equations is not None and hasattr(equations, "gamma"):
                gamma = float(equations.gamma)
                if not np.isfinite(gamma) or gamma <= 1.0:
                    raise ValueError("equations.gamma must be finite and greater than 1")
                self._boundary_gamma = gamma
                return gamma
            gamma = getattr(self, "_boundary_gamma", None)
            if gamma is None:
                raise ValueError(
                    "Dirichlet pressure requires equations so thermal "
                    "pressure can be converted to total energy"
                )
            return float(gamma)

        for boundary in ("left", "right", "top", "bottom"):
            bc_config = self._bc_boundary_map.get(boundary)
            if not bc_config:
                continue
            for comp, name in ((0, "density"), (1, "momentum_x"), (2, "momentum_y")):
                kind = bc_config.get(name)
                if kind == "neumann":
                    if _has_neighbor(boundary):
                        _write(comp, boundary, _read_interior(comp, boundary))
                elif kind == "dirichlet":
                    value_key = f"{boundary}_{name}"
                    if value_key not in self._dirichlet_values_map:
                        raise ValueError(
                            f"Dirichlet boundary {boundary}.{name} has no value"
                        )
                    _write(comp, boundary, self._dirichlet_values_map[value_key])
            kind = bc_config.get("pressure")
            if kind == "neumann":
                # Copies total energy, not thermal pressure.
                if _has_neighbor(boundary):
                    _write(3, boundary, _read_interior(3, boundary))
            elif kind == "dirichlet":
                if work.shape[0] != 4:
                    raise ValueError(
                        "Dirichlet thermal pressure is implemented for a "
                        "4-component hydro state"
                    )
                value_key = f"{boundary}_pressure"
                if value_key not in self._dirichlet_values_map:
                    raise ValueError(
                        f"Dirichlet boundary {boundary}.pressure has no value"
                    )
                pressure = self._dirichlet_values_map[value_key]
                gamma = _gamma()
                rho = work[(0,) + _edge(boundary)]
                mx = work[(1,) + _edge(boundary)]
                my = work[(2,) + _edge(boundary)]
                if is_torch:
                    bad_density = bool(torch.any(rho <= 0).item())
                else:
                    bad_density = bool(np.any(np.asarray(rho) <= 0.0))
                if bad_density or not pressure > 0.0:
                    raise ValueError("Boundary density and pressure must be positive")
                kinetic = 0.5 * (mx ** 2 + my ** 2) / rho
                _write(3, boundary, pressure / (gamma - 1.0) + kinetic)
        return work

    # 2D FFT wrappers
    def fft2(self, f: np.ndarray) -> np.ndarray:
        if self._mixed_open_y:
            raise ValueError("A Fourier/SBP grid has no periodic transform in y")
        if getattr(self, "_use_torch", False):
            assert torch is not None
            return torch.fft.fft2(f, dim=(-2, -1))
        try:
            return scipy_fft.fft2(f, axes=(-2, -1), workers=self.fft_workers)  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.fft2(f, axes=(-2, -1))

    def ifft2(self, F: np.ndarray) -> np.ndarray:
        if self._mixed_open_y:
            raise ValueError("A Fourier/SBP grid has no periodic transform in y")
        if getattr(self, "_use_torch", False):
            assert torch is not None
            return torch.fft.ifft2(F, dim=(-2, -1)).real
        try:
            return scipy_fft.ifft2(F, axes=(-2, -1), workers=self.fft_workers).real  # type: ignore[call-arg]
        except TypeError:
            return scipy_fft.ifft2(F, axes=(-2, -1)).real

    def _apply_dealias_mask(self, F: np.ndarray) -> np.ndarray:
        F *= self.dealias_mask
        return F

    def interpolate_periodic_linear_at_points(
        self, f: np.ndarray, xp: np.ndarray, yp: np.ndarray
    ) -> np.ndarray:
        """Interpolate a periodic 2-D field without host/device transfers."""
        if self.basis_x != "fourier" or self.basis_y != "fourier":
            raise ValueError(
                "interpolate_periodic_linear_at_points requires Fourier basis"
            )
        return _periodic_linear_2d(f, xp, yp, self.Lx, self.Ly)

    def evaluate_fourier_at_points(self, f: np.ndarray, xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
        """Evaluate a Fourier-represented 2D field at arbitrary points (Fourier x and y only).

        f(x,y) = (1/(Nx*Ny)) * Re( sum_{kx,ky} F exp(i*kx*x + i*ky*y) )
        on the retained 2/3-dealiased modes. The same tensor-product mask
        used by :meth:`dx1` is applied first. The stepper projects the
        conserved state onto that band after every accepted step. A derived
        field such as velocity can still carry modes outside the box, and
        those modes are discarded here.
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
        F = np.ascontiguousarray(
            np.fft.fft2(f_np, axes=(-2, -1)) * dealias,
            dtype=np.complex128,
        )
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

    def evaluate_fourier_at_points_batched_2d(
        self,
        fx: np.ndarray,
        fy: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray,
    ) -> tuple:
        """Evaluate two Fourier fields at one shared set of 2-D points.

        Both fields are restricted to the retained 2/3-rule band before
        evaluation, as in :meth:`evaluate_fourier_at_points`.
        """
        if self.basis_x != "fourier" or self.basis_y != "fourier":
            raise ValueError(
                "evaluate_fourier_at_points_batched_2d requires Fourier basis"
            )
        is_torch_input = (
            _TORCH_AVAILABLE
            and torch is not None
            and getattr(self, "_use_torch", False)
            and isinstance(fx, torch.Tensor)
        )
        if is_torch_input:
            fx_np = fx.detach().cpu().numpy().astype(np.float64)
            fy_np = fy.detach().cpu().numpy().astype(np.float64)
            xp_np = np.asarray(
                xp.detach().cpu().numpy() if isinstance(xp, torch.Tensor) else xp,
                dtype=np.float64,
            )
            yp_np = np.asarray(
                yp.detach().cpu().numpy() if isinstance(yp, torch.Tensor) else yp,
                dtype=np.float64,
            )
            device, dtype = fx.device, fx.dtype
        else:
            fx_np = np.asarray(fx)
            fy_np = np.asarray(fy)
            xp_np = np.asarray(xp, dtype=np.float64)
            yp_np = np.asarray(yp, dtype=np.float64)
        dealias = _to_np(self.dealias_mask)
        scale = 1.0 / (self.Nx * self.Ny)
        Fx = np.ascontiguousarray(
            np.fft.fft2(fx_np, axes=(-2, -1)) * dealias,
            dtype=np.complex128,
        )
        Fy = np.ascontiguousarray(
            np.fft.fft2(fy_np, axes=(-2, -1)) * dealias,
            dtype=np.complex128,
        )
        if _FINUFFT_AVAILABLE:
            y_fin = np.ascontiguousarray((2.0 * np.pi / self.Ly) * yp_np)
            x_fin = np.ascontiguousarray((2.0 * np.pi / self.Lx) * xp_np)
            plan_key = (self.Ny, self.Nx, x_fin.size)
            cached_plan = getattr(self, "_finufft_batched_2d_plan", None)
            if cached_plan is None or cached_plan[0] != plan_key:
                plan = _finufft.Plan(
                    2,
                    (self.Ny, self.Nx),
                    n_trans=2,
                    dtype="complex128",
                    **_FINUFFT_OPTS,
                )
                self._finufft_batched_2d_plan = (plan_key, plan)
            else:
                plan = cached_plan[1]
            plan.setpts(y_fin, x_fin)
            values = np.real(
                plan.execute(np.ascontiguousarray(np.stack((Fx, Fy))))
            ) * scale
            vx, vy = values
        else:
            kx = _to_np(self.kx).reshape(-1)
            ky = _to_np(self.ky).reshape(-1)
            E_ky = np.exp(1j * yp_np[:, None] * ky[None, :])
            E_kx = np.exp(1j * xp_np[:, None] * kx[None, :])
            vx = np.real(((E_ky @ Fx) * E_kx).sum(axis=1)) * scale
            vy = np.real(((E_ky @ Fy) * E_kx).sum(axis=1)) * scale
        if is_torch_input:
            np_dtype = np.float32 if dtype == torch.float32 else np.float64
            return (
                torch.from_numpy(vx.astype(np_dtype)).to(device=device),
                torch.from_numpy(vy.astype(np_dtype)).to(device=device),
            )
        return vx.astype(fx_np.dtype), vy.astype(fy_np.dtype)

    def dx1(self, f: np.ndarray) -> np.ndarray:
        """Derivative along x, the last axis.

        On a periodic grid the spectrum is multiplied by the tensor-product
        2/3 mask before differentiation. A mode is removed when
        ``|kx| > (Nx - 1) // 3`` or ``|ky| > (Ny - 1) // 3``, so this
        derivative also drops modes that are resolved in x but lie outside
        the y cutoff. The mask is inside the operator. The stepper projects
        the state onto that box after every accepted step and uses this
        method for nonlinear fluxes. A Fourier/SBP grid masks only the
        periodic x wavenumbers. A pure Legendre grid has no Fourier mask.
        """
        if self._mixed_open_y:
            F = self._mixed_fft_x(f)
            return self._mixed_ifft_x(F * self.dealias_mask * self.ikx)
        if self._legendre_basis is not None:
            return self._legendre_basis.dx(f)
        if self.basis_x == "fourier":
            F = self.fft2(f)
            F = self._apply_dealias_mask(F)
            dF = F * self.ikx  # broadcast over (Ny, Nx)
            return self.ifft2(dF)
        raise ValueError(f"Unknown basis_x: {self.basis_x}")

    def dy1(self, f: np.ndarray) -> np.ndarray:
        """Derivative along y, the second-to-last axis.

        The periodic path uses the same tensor-product 2/3 mask as
        :meth:`dx1`, so a y derivative also drops modes outside the x
        cutoff. A Fourier/SBP grid uses the unmasked SBP operator in y.
        A pure Legendre grid has no Fourier mask.
        """
        if self._mixed_open_y:
            return self._sbp_dy(f)
        if self._legendre_basis is not None:
            return self._legendre_basis.dy(f)
        if self.basis_y == "fourier":
            F = self.fft2(f)
            F = self._apply_dealias_mask(F)
            dF = F * self.iky
            return self.ifft2(dF)
        raise ValueError(f"Unknown basis_y: {self.basis_y}")

    def project_dealiased(self, f: np.ndarray) -> np.ndarray:
        """Project Fourier directions onto the retained 2/3 modal subspace."""
        if self._mixed_open_y:
            if not self.dealias:
                return f
            return self._mixed_ifft_x(self._mixed_fft_x(f) * self.dealias_mask)
        if self._legendre_basis is not None:
            return f
        if self.basis_x == "fourier" and self.basis_y == "fourier":
            F = self.fft2(f)
            F = self._apply_dealias_mask(F)
            return self.ifft2(F)
        return f

    def apply_spectral_dissipation(
        self, f: np.ndarray, dt: float, *, project: bool = False
    ) -> np.ndarray:
        """Apply ``exp(-dt * lambda)`` and, if requested, the dealias mask.

        ``lambda`` is ``rate`` times the sum of the x and y power-law ramps.
        A mode at the cutoff on both axes has ``lambda = 2 * rate`` and an
        e-folding time ``tau / 2``, where ``tau`` is
        ``e_folding_time_at_cutoff``. That name is the single-axis time.
        """
        if dt < 0.0:
            raise ValueError("spectral dissipation dt must be non-negative")
        if self._mixed_open_y:
            return self.project_dealiased(f) if project else f
        if self._legendre_basis is not None:
            return f
        pure_fourier = self.basis_x == "fourier" and self.basis_y == "fourier"
        if not pure_fourier or not self.spectral_dissipation_enabled:
            return self.project_dealiased(f) if project else f
        F = self.fft2(f)
        if project:
            F = self._apply_dealias_mask(F)
        if getattr(self, "_use_torch", False):
            assert torch is not None
            F *= torch.exp(-float(dt) * self.filter_rate)
        else:
            F *= np.exp(-float(dt) * self.filter_rate)
        return self.ifft2(F)

    def apply_modal_filter(self, f: np.ndarray) -> np.ndarray:
        """Apply the independent Legendre accepted-step modal filter."""
        if self._legendre_basis is None or not self.modal_filter_enabled:
            return f
        return self._legendre_basis.apply_spectral_filter(
            f, p=self.modal_filter_order, alpha=self.modal_filter_alpha
        )

    def apply_accepted_state_stabilization(
        self, f: np.ndarray, dt: float
    ) -> np.ndarray:
        """Project Fourier states or apply the separate Legendre modal filter."""
        if self._mixed_open_y:
            return self.project_dealiased(f)
        pure_fourier = self.basis_x == "fourier" and self.basis_y == "fourier"
        if pure_fourier:
            return self.apply_spectral_dissipation(f, dt, project=True)
        return self.apply_modal_filter(f)

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        """Backward-compatible accepted-state stabilization entrypoint."""
        return self.apply_accepted_state_stabilization(f, 0.0)

    # --- MHD solenoidal helpers (2D) ---
    def conserved_quantities(self, U, equations):
        """Use physical SBP quadrature for mixed-grid conservation diagnostics."""
        if not self._mixed_open_y:
            return equations.conserved_quantities(U)
        _, weights = self._sbp_arrays(U)
        w = self.dx*weights[:, None]
        mhd = U.shape[0] == 8
        keys = (["mass", "momentum_x", "momentum_y", "momentum_z", "energy"]
                if mhd else ["mass", "momentum_x", "momentum_y", "energy"])
        totals = (U[:len(keys)]*w).sum(axis=(-2, -1))
        result = {key: float(value) for key, value in zip(keys, totals)}
        if mhd:
            result["magnetic_energy"] = float((0.5*(U[5:8]**2).sum(axis=0)*w).sum())
            result["cross_helicity"] = float(((U[1:4]*U[5:8]).sum(axis=0)/U[0]*w).sum())
        return result

    def divergence(self, Bx: np.ndarray, By: np.ndarray) -> np.ndarray:
        """Divergence of an in-plane vector.

        The periodic implementation multiplies both spectra by the
        tensor-product 2/3 mask, the same box used by :meth:`dx1` and
        :meth:`dy1`. A Fourier/SBP grid adds those derivative methods, so
        only the periodic x factor of the mask is applied.
        """
        if self._mixed_open_y:
            return self.dx1(Bx) + self.dy1(By)
        Fx = self._apply_dealias_mask(self.fft2(Bx))
        Fy = self._apply_dealias_mask(self.fft2(By))
        div_hat = self.ikx * Fx + self.iky * Fy
        return self.ifft2(div_hat)

    def project_solenoidal(self, Bx: np.ndarray, By: np.ndarray):
        """2D Helmholtz (Leray) projection of the in-plane field (Bx, By).

        Removes the longitudinal part so that kx*Bx_hat + ky*By_hat = 0 for every
        mode, i.e. div(B) = 0 to machine precision. Bz (out-of-plane) does not
        enter the 2D divergence and is left untouched by the caller.
        """
        if self._mixed_open_y:
            return self._project_mixed_solenoidal(Bx, By)
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

    def _project_mixed_solenoidal(self, Bx, By):
        """Weighted projection onto Dx Bx + Dy By = 0, with free normal B."""
        if not hasattr(self, "_mixed_poisson"):
            from scipy.linalg import eigh

            D = self._sbp_dy(np.eye(self.Ny))
            root = np.sqrt(self.wy)
            weighted_D = root[:, None] * D / root[None, :]
            values, vectors = eigh(weighted_D @ weighted_D.T)
            values[values < 1e-10 * values[-1]] = 0.0
            self._mixed_poisson = values, vectors, root
            self._mixed_poisson_cache = {}
        is_torch = _TORCH_AVAILABLE and isinstance(Bx, torch.Tensor)
        key = (str(Bx.device) if is_torch else "numpy", str(Bx.dtype))
        if key not in self._mixed_poisson_cache:
            arrays = self._mixed_poisson
            if is_torch:
                arrays = tuple(torch.as_tensor(a, dtype=Bx.dtype, device=Bx.device) for a in arrays)
            else:
                arrays = tuple(a.astype(Bx.dtype) for a in arrays)
            self._mixed_poisson_cache[key] = arrays
        values, vectors, root = self._mixed_poisson_cache[key]
        Fx, Fy = self._mixed_fft_x(Bx), self._mixed_fft_x(By)
        ik = self.ikx * self.dealias_mask
        if self.Nx % 2 == 0:
            ik = ik.clone() if is_torch else ik.copy()
            ik[..., self.Nx // 2] = 0.0
        q = root[:, None] * (ik * Fx + self.dy1(Fy))
        denominator = values[:, None] + abs(ik)**2
        if is_torch:
            inverse = torch.where(denominator > 0, 1.0 / denominator.clamp_min(1e-30), 0.0)
            def solve(part):
                return vectors @ ((vectors.T @ part) * inverse)
            phi = torch.complex(solve(q.real), solve(q.imag)) / root[:, None]
        else:
            inverse = np.divide(1.0, denominator, out=np.zeros_like(denominator), where=denominator > 0)
            phi = (vectors @ ((vectors.T @ q) * inverse)) / root[:, None]
        return (self._mixed_ifft_x(Fx + ik * phi),
                self._mixed_ifft_x(Fy - self.dy_adjoint(phi)))

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """Scalar Laplacian.

        The periodic result multiplies the spectrum by the tensor-product
        2/3 mask (see :meth:`dx1`) before applying ``-(kx^2 + ky^2)``.
        A Fourier/SBP grid uses ``dx1(dx1(f)) - dy_adjoint(dy1(f))``,
        which masks only x. The mask is applied for linear diffusion as
        well as for any other caller.
        """
        if self._mixed_open_y:
            return self.dx1(self.dx1(f)) - self.dy_adjoint(self.dy1(f))
        if self.kx is None or self.ky is None:
            raise ValueError("laplacian requires Fourier basis in x and y")
        F = self._apply_dealias_mask(self.fft2(f))
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
            X = torch.from_numpy(X_np).to(
                dtype=self.x.dtype, device=self.torch_device
            )
            Y = torch.from_numpy(Y_np).to(
                dtype=self.y.dtype, device=self.torch_device
            )
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
    filter_params: Optional[Dict[str, Any]] = None
    fft_workers: int = 1
    backend: str = "auto"  # Metal, CUDA, then NumPy CPU
    torch_device: Optional[str] = None
    precision: str = "double"  # "single" or "double"
    debug: bool = False

    def __post_init__(self) -> None:
        self.backend, self.torch_device = resolve_backend(
            self.backend, self.torch_device
        )
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

        # Hard dealias projection and optional timestep-aware Fourier dissipation.
        self.dealias_mask = _build_filter_mask_3d(
            self.Nx, self.Ny, self.Nz, self.dealias
        )
        enabled, p, rate, onset = _spectral_dissipation_settings(
            self.filter_params
        )
        self.spectral_dissipation_enabled = enabled
        self.filter_rate = _build_exponential_filter_rate_3d(
            self.Nx,
            self.Ny,
            self.Nz,
            p=p,
            rate=rate,
            dealias=self.dealias,
            onset_fraction=onset,
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
            _validate_torch_device(self.torch_device, self.debug)
            torch_dtype, torch_cdtype, self.precision = _torch_dtypes(
                self.precision, self.torch_device
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
            self.filter_rate = torch.from_numpy(np.asarray(self.filter_rate)).to(
                dtype=torch_dtype, device=self.torch_device
            )
        elif self.precision == "single":
            _note_numpy_fourier_single()

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

    def _apply_dealias_mask(self, F: np.ndarray) -> np.ndarray:
        F *= self.dealias_mask
        return F

    def interpolate_periodic_linear_at_points(
        self,
        f: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray,
        zp: np.ndarray,
    ) -> np.ndarray:
        """Interpolate a periodic 3-D field without host/device transfers."""
        return _periodic_linear_3d(
            f, xp, yp, zp, self.Lx, self.Ly, self.Lz
        )

    def interpolate_periodic_linear_at_points_batched_3d(
        self,
        fx: np.ndarray,
        fy: np.ndarray,
        fz: np.ndarray,
        xp: np.ndarray,
        yp: np.ndarray,
        zp: np.ndarray,
    ) -> tuple:
        """Interpolate three periodic fields at one shared particle set."""
        if _TORCH_AVAILABLE and torch is not None and isinstance(fx, torch.Tensor):
            fields = torch.stack((fx, fy, fz), dim=0)
        else:
            fields = np.stack((fx, fy, fz), axis=0)
        values = _periodic_linear_3d(
            fields, xp, yp, zp, self.Lx, self.Ly, self.Lz
        )
        return values[0], values[1], values[2]

    def evaluate_fourier_at_points(
        self, f: np.ndarray, xp: np.ndarray, yp: np.ndarray, zp: np.ndarray
    ) -> np.ndarray:
        """Evaluate a Fourier-represented 3D field at arbitrary points.

        f(x,y,z) = (1/(Nx*Ny*Nz)) * Re( sum F exp(i*kx*x + i*ky*y + i*kz*z) )
        on the retained 2/3-dealiased modes. The same tensor-product mask
        used by :meth:`dx1` is applied first. The stepper projects the
        conserved state onto that band after every accepted step. A derived
        field such as velocity can still carry modes outside the box, and
        those modes are discarded here.
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
        F = np.ascontiguousarray(
            np.fft.fftn(f_np, axes=(-3, -2, -1)) * dealias,
            dtype=np.complex128,
        )
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
        tracer velocity interpolation). Each field is restricted to the
        retained 2/3-rule band before evaluation.
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
        scale = 1.0 / (self.Nx * self.Ny * self.Nz)
        Fx = np.ascontiguousarray(
            np.fft.fftn(fx_np, axes=(-3, -2, -1)) * dealias,
            dtype=np.complex128,
        )
        Fy = np.ascontiguousarray(
            np.fft.fftn(fy_np, axes=(-3, -2, -1)) * dealias,
            dtype=np.complex128,
        )
        Fz = np.ascontiguousarray(
            np.fft.fftn(fz_np, axes=(-3, -2, -1)) * dealias,
            dtype=np.complex128,
        )
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
        """3D divergence ``dx Vx + dy Vy + dz Vz``.

        Each spectrum is multiplied by the tensor-product 2/3 mask before
        the derivatives. A mode is removed if it exceeds the cutoff on any
        axis, including an axis other than the one being differentiated.
        The mask is inside this operator. The stepper projects the conserved
        state onto that box after every accepted step; linear diagnostics
        that call this method drop the same modes. After
        ``project_solenoidal`` the result is zero to machine precision on
        the retained modes because ``k·V_hat = 0`` there.
        """
        Fx = self._apply_dealias_mask(self.fftn(Vx))
        Fy = self._apply_dealias_mask(self.fftn(Vy))
        Fz = self._apply_dealias_mask(self.fftn(Vz))
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
        """3D curl. Returns ``(Cx, Cy, Cz)``.

        Each spectrum is multiplied by the tensor-product 2/3 mask before
        the derivatives, the same box described on :meth:`divergence`. The
        mask is applied for every caller, not only for nonlinear products.
        """
        Fx = self._apply_dealias_mask(self.fftn(Vx))
        Fy = self._apply_dealias_mask(self.fftn(Vy))
        Fz = self._apply_dealias_mask(self.fftn(Vz))
        Cx = self.ifftn(self.iky * Fz - self.ikz * Fy)
        Cy = self.ifftn(self.ikz * Fx - self.ikx * Fz)
        Cz = self.ifftn(self.ikx * Fy - self.iky * Fx)
        return Cx, Cy, Cz

    def laplacian(self, f: np.ndarray) -> np.ndarray:
        """Scalar Laplacian used for explicit resistivity and viscosity.

        The spectrum is multiplied by the tensor-product 2/3 mask (see
        :meth:`dx1`) before ``-(kx^2 + ky^2 + kz^2)`` is applied. Linear
        diffusion therefore drops every mode outside that box.
        """
        F = self._apply_dealias_mask(self.fftn(f))
        if getattr(self, "_use_torch", False) and torch is not None and isinstance(F, torch.Tensor):
            k2 = self.k2.to(F.dtype)
        else:
            k2 = self.k2
        return self.ifftn(-k2 * F)

    def dx1(self, f: np.ndarray) -> np.ndarray:
        """Derivative along x, the last axis.

        The spectrum is multiplied by the tensor-product 2/3 mask before
        differentiation. A mode is removed when it exceeds
        ``(N - 1) // 3`` on x, y, or z, so this derivative also drops modes
        that are resolved in x. The mask is inside the operator. The
        stepper projects the state onto that box after every accepted step
        and uses this method for nonlinear fluxes. Linear callers drop the
        same modes.
        """
        F = self.fftn(f)
        F = self._apply_dealias_mask(F)
        dF = F * self.ikx
        return self.ifftn(dF)

    def dy1(self, f: np.ndarray) -> np.ndarray:
        """Derivative along y. Uses the same tensor-product 2/3 mask as :meth:`dx1`."""
        F = self.fftn(f)
        F = self._apply_dealias_mask(F)
        dF = F * self.iky
        return self.ifftn(dF)

    def dz1(self, f: np.ndarray) -> np.ndarray:
        """Derivative along z. Uses the same tensor-product 2/3 mask as :meth:`dx1`."""
        F = self.fftn(f)
        F = self._apply_dealias_mask(F)
        dF = F * self.ikz
        return self.ifftn(dF)

    def project_dealiased(self, f: np.ndarray) -> np.ndarray:
        """Project a Fourier field onto the retained 2/3 modal subspace."""
        F = self.fftn(f)
        F = self._apply_dealias_mask(F)
        return self.ifftn(F)

    def apply_spectral_dissipation(
        self, f: np.ndarray, dt: float, *, project: bool = False
    ) -> np.ndarray:
        """Apply ``exp(-dt * lambda)`` and, if requested, the dealias mask.

        ``lambda`` is ``rate`` times the sum of the x, y, and z power-law
        ramps. A mode at the cutoff on all three axes has
        ``lambda = 3 * rate`` and an e-folding time ``tau / 3``, where
        ``tau`` is ``e_folding_time_at_cutoff``. That name is the
        single-axis time.
        """
        if dt < 0.0:
            raise ValueError("spectral dissipation dt must be non-negative")
        if not self.spectral_dissipation_enabled:
            return self.project_dealiased(f) if project else f
        F = self.fftn(f)
        if project:
            F = self._apply_dealias_mask(F)
        if getattr(self, "_use_torch", False):
            assert torch is not None
            F *= torch.exp(-float(dt) * self.filter_rate)
        else:
            F *= np.exp(-float(dt) * self.filter_rate)
        return self.ifftn(F)

    def apply_accepted_state_stabilization(
        self, f: np.ndarray, dt: float
    ) -> np.ndarray:
        """Complete the Fourier split step and enforce hard dealiasing."""
        return self.apply_spectral_dissipation(f, dt, project=True)

    def apply_spectral_filter(self, f: np.ndarray) -> np.ndarray:
        """Backward-compatible accepted-state projection entrypoint."""
        return self.apply_accepted_state_stabilization(f, 0.0)

    def xyz_mesh(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if getattr(self, "_use_torch", False):
            assert torch is not None
            x_np = self.x.detach().cpu().numpy()
            y_np = self.y.detach().cpu().numpy()
            z_np = self.z.detach().cpu().numpy()
            # Return arrays with shape (Nz, Ny, Nx) to align with solver axes (-3,-2,-1)
            Z_np, Y_np, X_np = np.meshgrid(z_np, y_np, x_np, indexing="ij")
            X = torch.from_numpy(X_np).to(dtype=self.x.dtype, device=self.torch_device)
            Y = torch.from_numpy(Y_np).to(dtype=self.y.dtype, device=self.torch_device)
            Z = torch.from_numpy(Z_np).to(dtype=self.z.dtype, device=self.torch_device)
            return X, Y, Z
        else:
            # Return arrays with shape (Nz, Ny, Nx)
            Z, Y, X = np.meshgrid(self.z, self.y, self.x, indexing="ij")
            return X, Y, Z
