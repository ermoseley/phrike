from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import numpy as np


class Basis1D(ABC):
    """Abstract 1D spectral basis interface.

    Implementations provide transform pairs and differentiation operators on
    the last axis (spatial axis). Inputs may be batched with arbitrary leading
    dimensions (..., N). Concrete bases accept NumPy arrays and may accept
    Torch tensors.
    """

    def __init__(self, N: int, Lx: float, bc: str = "periodic") -> None:
        self.N = int(N)
        self.Lx = float(Lx)
        self.bc = str(bc).lower()

    # --- Grid nodes and quadrature ---
    @abstractmethod
    def nodes(self) -> np.ndarray:
        """Return physical nodes x of shape (N)."""

    @abstractmethod
    def quadrature_weights(self) -> Optional[np.ndarray]:
        """Return quadrature weights for integration on nodes, or None.

        Weights are for approximating ∫ f(x) dx ≈ sum_j w_j f(x_j).
        """

    # --- Transform pairs ---
    @abstractmethod
    def forward(self, f: np.ndarray) -> np.ndarray:
        """Physical → spectral transform along the last axis.

        Args:
            f: array of shape (..., N)
        Returns:
            Spectral coefficients array with same shape (..., N) unless
            the basis uses complex spectra; Fourier impl may return complex.
        """

    @abstractmethod
    def inverse(self, F: np.ndarray) -> np.ndarray:
        """Spectral → physical inverse transform along last axis."""

    # --- Derivatives ---
    @abstractmethod
    def dx(self, f: np.ndarray) -> np.ndarray:
        """Compute ∂f/∂x at nodes, preserving leading dims."""

    def d2x(self, f: np.ndarray) -> np.ndarray:
        """Compute ∂²f/∂x². Default via two applications of dx."""
        return self.dx(self.dx(f))

    def _spectral_filter_eta(self) -> np.ndarray:
        """Modal coordinate in ``[0, 1]`` for the exponential filter.

        The default uses polynomial degree ``k = 0 .. N-1``, with the
        highest degree at 1. Fourier overrides this with ``|k| / k_Nyquist``.
        """
        degree = np.arange(self.N, dtype=float)
        degree_max = max(float(self.N - 1), 1.0)
        return degree / degree_max

    # --- Optional spectral filtering ---
    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        """Apply a high-order exponential filter in spectral space.

        The default multiplies modal coefficient ``k`` by
        ``sigma(k) = exp(-alpha * (k / (N - 1))^p)``, where ``k`` is the
        polynomial degree. :class:`FourierBasis1D` uses ``|wavenumber| /
        k_Nyquist`` instead of the degree index.
        """
        F = self.forward(f)
        N = self.N
        eta = self._spectral_filter_eta()
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
            return self.inverse(F_filtered)
        else:
            F_filtered = F * sigma_np.reshape((1,) * (F.ndim - 1) + (N,))
            return self.inverse(F_filtered)


def _as_last_axis_matrix_apply(A: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Apply square matrix A (N,N) along the last axis of f (..., N).

    Returns array with same shape as f. Uses tensordot for clarity.
    """
    N = A.shape[0]
    assert A.shape == (N, N)
    assert f.shape[-1] == N
    # (..., N) -> (..., N), sum over last axis of f and first axis of A^T
    # We use A^T so result[..., i] = sum_j f[..., j] * A[j, i]
    return np.tensordot(f, A, axes=([-1], [0]))


class Basis2D(ABC):
    """Abstract 2D spectral basis interface.

    The last two axes are spatial: (..., Ny, Nx).
    Implementations should provide per-axis nodes, transforms, and derivatives.
    """

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float, bc: str = "dirichlet") -> None:
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.bc = str(bc).lower()

    @abstractmethod
    def nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (x_nodes, y_nodes)."""

    def quadrature_weights(self) -> Optional[np.ndarray]:
        """Return 2D quadrature weights (Ny, Nx), or None."""
        return None

    @abstractmethod
    def forward(self, f: np.ndarray) -> np.ndarray:
        """Physical → spectral along both axes."""

    @abstractmethod
    def inverse(self, F: np.ndarray) -> np.ndarray:
        """Spectral → physical along both axes."""

    @abstractmethod
    def dx(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂x at nodes."""

    @abstractmethod
    def dy(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂y at nodes."""

    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        """Optional modal/nodal filter; default no-op."""
        return f


class Basis3D(ABC):
    """Abstract 3D spectral basis interface.

    The last three axes are spatial: (..., Nz, Ny, Nx).
    """

    def __init__(self, Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float, bc: str = "dirichlet") -> None:
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.bc = str(bc).lower()

    @abstractmethod
    def nodes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x_nodes, y_nodes, z_nodes)."""

    def quadrature_weights(self) -> Optional[np.ndarray]:
        """Return 3D quadrature weights (Nz, Ny, Nx), or None."""
        return None

    @abstractmethod
    def forward(self, f: np.ndarray) -> np.ndarray:
        """Physical → spectral along all axes."""

    @abstractmethod
    def inverse(self, F: np.ndarray) -> np.ndarray:
        """Spectral → physical along all axes."""

    @abstractmethod
    def dx(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂x at nodes."""

    @abstractmethod
    def dy(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂y at nodes."""

    @abstractmethod
    def dz(self, f: np.ndarray) -> np.ndarray:
        """∂f/∂z at nodes."""

    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        return f


