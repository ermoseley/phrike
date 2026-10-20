from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .base import Basis1D

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch optional
    torch = None  # type: ignore


class ChebyshevLobattoBasis1D(Basis1D):
    """Chebyshev–Lobatto collocation on [0, Lx], including both endpoints.

    Modal coefficients multiply T_k(y), with y = 1 - 2*x/Lx. Transforms
    interpolate all N modes, including the highest mode. Quadrature uses
    physical Clenshaw–Curtis weights for the unweighted integral over x.
    Operations act on the last axis for NumPy arrays or Torch tensors.
    """

    def __init__(self, N: int, Lx: float, *, bc: str = "dirichlet",
                 precision: str = "double") -> None:
        super().__init__(N=N, Lx=Lx, bc=bc)
        if self.N < 2 or self.N != N:
            raise ValueError("Chebyshev–Lobatto requires an integer N >= 2")
        if not np.isfinite(self.Lx) or self.Lx <= 0.0:
            raise ValueError("Lx must be finite and positive")
        if precision not in ("single", "double"):
            raise ValueError("precision must be 'single' or 'double'")
        self.precision = precision
        dtype = np.float32 if precision == "single" else np.float64
        n = self.N - 1
        theta = np.pi * np.arange(self.N) / n
        y = np.cos(theta)
        self._y = y.astype(dtype)
        self._x = ((1.0 - y) * (self.Lx / 2.0)).astype(dtype)

        V = np.cos(theta[:, None] * np.arange(self.N))
        endpoint_scale = np.ones(self.N)
        endpoint_scale[[0, -1]] = 0.5
        forward = (2.0 / n) * endpoint_scale[:, None] * V * endpoint_scale
        moments = np.zeros(self.N)
        k_even = np.arange(0, self.N, 2)
        moments[k_even] = 2.0 / (1.0 - k_even**2)
        self._w_x = (forward @ moments * (self.Lx / 2.0)).astype(dtype)

        barycentric = (-1.0)**np.arange(self.N) * endpoint_scale
        differences = y[:, None] - y[None, :]
        np.fill_diagonal(differences, 1.0)
        D = (barycentric[None, :] / barycentric[:, None]) / differences
        np.fill_diagonal(D, 0.0)
        np.fill_diagonal(D, -D.sum(axis=1))
        D *= -2.0 / self.Lx
        self._matrices = {
            "forward": forward.astype(dtype),
            "inverse": V.T.astype(dtype),
            "dx": D.T.astype(dtype),
            "d2x": (D @ D).T.astype(dtype),
        }
        self._torch_cache: Dict[Tuple[str, str], dict] = {}

    def nodes(self) -> np.ndarray:
        return self._x

    def quadrature_weights(self) -> np.ndarray:
        return self._w_x.copy()

    def _get_torch_matrix(self, name: str, ref: "torch.Tensor") -> "torch.Tensor":
        key = (str(ref.device), str(ref.dtype))
        cache = self._torch_cache.setdefault(key, {})
        if name not in cache:
            cache[name] = torch.from_numpy(self._matrices[name]).to(
                dtype=ref.dtype, device=ref.device
            )
        return cache[name]

    def _apply(self, name: str, f: np.ndarray) -> np.ndarray:
        if torch is not None and isinstance(f, torch.Tensor):
            if f.ndim == 0 or f.shape[-1] != self.N:
                raise ValueError("Input last axis must have length N")
            return torch.matmul(f, self._get_torch_matrix(name, f))
        f = np.asarray(f)
        if f.ndim == 0 or f.shape[-1] != self.N:
            raise ValueError("Input last axis must have length N")
        matrix = self._matrices[name]
        if np.issubdtype(f.dtype, np.inexact):
            matrix = matrix.astype(f.dtype, copy=False)
        return f @ matrix

    def forward(self, f: np.ndarray) -> np.ndarray:
        return self._apply("forward", f)

    def inverse(self, F: np.ndarray) -> np.ndarray:
        return self._apply("inverse", F)

    def dx(self, f: np.ndarray) -> np.ndarray:
        return self._apply("dx", f)

    def d2x(self, f: np.ndarray) -> np.ndarray:
        return self._apply("d2x", f)

    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8,
                              alpha: float = 36.0) -> np.ndarray:
        F = self.forward(f)
        sigma = np.exp(-float(alpha) * (np.arange(self.N) / (self.N - 1))**int(p))
        if torch is not None and isinstance(F, torch.Tensor):
            sigma = torch.from_numpy(sigma).to(dtype=F.dtype, device=F.device)
        else:
            sigma = sigma.astype(F.dtype)
        return self.inverse(F * sigma)
