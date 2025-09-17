from __future__ import annotations

from typing import Optional, Dict, Tuple

import numpy as np
from numpy.polynomial import legendre as npleg

from .base import Basis1D, _as_last_axis_matrix_apply

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch optional
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _legendre_gauss_lobatto_nodes_weights(N: int) -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre–Gauss–Lobatto nodes y in [-1, 1] and quadrature weights w.

    For N >= 2, the nodes are the endpoints ±1 and the (N-2) roots of
    d/dy P_{N-1}(y). The weights are

        w_j = 2 / [N(N-1) (P_{N-1}(y_j))^2].

    Args:
        N: number of nodes (>= 2). If N == 1, returns y=[0], w=[2].

    Returns:
        (y, w): nodes and weights arrays of shape (N,).
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if N == 1:
        # Degenerate case: single node at center with full weight 2
        return np.array([0.0], dtype=float), np.array([2.0], dtype=float)

    # Endpoints
    y0 = -1.0
    yN = 1.0

    if N == 2:
        y = np.array([y0, yN], dtype=float)
        # P_{N-1} = P_1, P_1(±1) = ±1 → w0 = wN = 1
        # However, formula gives 2/[2*1*(±1)^2] = 1 each; sum weights = 2
        w = np.array([1.0, 1.0], dtype=float)
        return y, w

    # Interior nodes: roots of derivative of P_{N-1}
    PNm1 = npleg.Legendre.basis(N - 1)
    roots = PNm1.deriv().roots()
    y = np.empty(N, dtype=float)
    y[0] = y0
    y[-1] = yN
    y[1:-1] = np.sort(roots)

    # Weights formula: w_j = 2 / [N(N-1) (P_{N-1}(y_j))^2]
    c = np.zeros(N, dtype=float)
    c[-1] = 1.0  # coefficients for P_{N-1}
    PN_vals = npleg.legval(y, c)
    w = 2.0 / (N * (N - 1) * (PN_vals**2))
    return y, w


def _stable_legendre_diff_matrix(y: np.ndarray) -> np.ndarray:
    """Compute stable first-derivative matrix on LGL nodes via Vandermonde solve.

    Builds Legendre Vandermonde V and its derivative dV and solves D_y such that
    D_y @ V = dV, i.e., D_y = dV @ V^{-1}. This avoids barycentric products.
    """
    N = y.shape[0]
    # Vandermonde (Legendre) up to degree N-1: shape (N, N)
    V = npleg.legvander(y, N - 1)
    # Build derivative Vandermonde: for each k, evaluate d/dy P_k at nodes
    dV = np.empty_like(V)
    for k in range(N):
        coeffs = np.zeros(N)
        coeffs[k] = 1.0
        dcoeffs = npleg.legder(coeffs)
        dV[:, k] = npleg.legval(y, dcoeffs)
    # Solve V^T X^T = dV^T for X^T to avoid explicit inverse
    # i.e., for each row of dV^T solve a system; np.linalg.solve handles all columns
    Dy = np.linalg.solve(V.T, dV.T).T
    return Dy


class LegendreLobattoBasis1D(Basis1D):
    """Legendre–Gauss–Lobatto (LGL) collocation basis.

    - Nodes: LGL points mapped from [-1, 1] to [0, Lx]
    - Quadrature: LGL weights (exact for degree ≤ 2N-3), scaled by Lx/2
    - Derivatives: barycentric differentiation matrix (O(N^2) apply)
    - Transforms: modal Legendre coefficients via LGL projection

    This basis does not use FFTs. Operations are optimized around:
    - diagonal mass matrix implied by LGL quadrature (weights)
    - batched tensordot evaluations for transforms and derivatives
    """

    def __init__(self, N: int, Lx: float, *, bc: str = "dirichlet") -> None:
        super().__init__(N=N, Lx=Lx, bc=bc)

        # Nodes and quadrature on reference domain [-1, 1]
        y, w_ref = _legendre_gauss_lobatto_nodes_weights(self.N)
        # Reorder to increasing x in [0, Lx]: want y from 1 -> -1
        order = np.argsort(-y)  # sort by descending y so x asc
        y = y[order]
        w_ref = w_ref[order]
        self._y = y
        # Map to [0, Lx]: x = (1 - y) * Lx/2 yields x asc when y desc
        self._x = (1.0 - y) * (self.Lx / 2.0)
        # Scale weights: dx = (Lx/2) dy
        self._w_x = w_ref * (self.Lx / 2.0)
        # Store reference weights for dy integrals
        self._w_ref = w_ref

        # Precompute differentiation matrix on x-domain using stable solve
        D_y = _stable_legendre_diff_matrix(self._y)
        # Mapping x = (1 - y) * Lx/2 => dy/dx = -2/Lx
        self._D_x = (-2.0 / self.Lx) * D_y

        # Precompute Vandermonde (Legendre) on reference nodes
        # V[j, k] = P_k(y_j), k = 0..N-1
        self._V = npleg.legvander(self._y, self.N - 1)
        # Modal scaling: a_k = (2k+1)/2 ∫_{-1}^1 f(y) P_k(y) dy (use w_ref)
        k = np.arange(self.N, dtype=float)
        self._proj_scale = (2.0 * k + 1.0) / 2.0  # shape (N,)

        # Caches
        self._nodal_filter_cache_np: Dict[Tuple[int, float], np.ndarray] = {}
        # Torch device/dtype keyed cache of matrices
        self._torch_cache: Dict[Tuple[str, str], Dict[str, "torch.Tensor"]] = {}  # type: ignore[name-defined]

    # Grid and weights
    def nodes(self) -> np.ndarray:
        return self._x

    def quadrature_weights(self) -> Optional[np.ndarray]:
        return self._w_x.copy()

    # Transforms
    def forward(self, f: np.ndarray) -> np.ndarray:
        """Compute modal Legendre coefficients a_k from nodal values f(..., N).

        Uses LGL projection: a_k ≈ (2k+1)/2 * Σ_j w_x[j] f_j P_k(y_j).
        """
        if _TORCH_AVAILABLE and isinstance(f, torch.Tensor):  # type: ignore[arg-type]
            mats = self._get_torch_mats(f)
            fw = f * mats["w_ref"].reshape((1,) * (f.ndim - 1) + (self.N,))
            a = torch.matmul(fw, mats["V"])  # (..., N)
            a = a * mats["proj"].reshape((1,) * (a.ndim - 1) + (self.N,))
            return a
        f_arr = np.asarray(f)
        w_ref = self._w_ref
        V = self._V
        fw = f_arr * w_ref.reshape((1,) * (f_arr.ndim - 1) + (self.N,))
        a = np.tensordot(fw, V, axes=([-1], [0]))
        a = a * self._proj_scale.reshape((1,) * (a.ndim - 1) + (self.N,))
        return a

    def inverse(self, F: np.ndarray) -> np.ndarray:
        """Evaluate nodal values from modal coefficients using Vandermonde.

        f_j ≈ Σ_k a_k P_k(y_j) = (V @ a)_j, applied along last axis.
        """
        if _TORCH_AVAILABLE and isinstance(F, torch.Tensor):  # type: ignore[arg-type]
            mats = self._get_torch_mats(F)
            return torch.matmul(F, mats["V"].T)
        F_arr = np.asarray(F)
        V = self._V
        return np.tensordot(F_arr, V.T, axes=([-1], [0]))

    # Derivatives
    def dx(self, f: np.ndarray) -> np.ndarray:
        # Apply D_x along last axis
        if _TORCH_AVAILABLE and isinstance(f, torch.Tensor):  # type: ignore[arg-type]
            mats = self._get_torch_mats(f)
            return torch.matmul(f, mats["D_x"].T)
        return _as_last_axis_matrix_apply(self._D_x.T, f)

    def d2x(self, f: np.ndarray) -> np.ndarray:
        # Apply D_x twice; cache D2_x on device to avoid recomputation
        if _TORCH_AVAILABLE and isinstance(f, torch.Tensor):  # type: ignore[arg-type]
            mats = self._get_torch_mats(f)
            if "D2_x" not in mats:
                mats["D2_x"] = torch.matmul(mats["D_x"], mats["D_x"])  # (N,N)
            return torch.matmul(f, mats["D2_x"].T)
        return _as_last_axis_matrix_apply((self._D_x @ self._D_x).T, f)

    # --- Spectral filtering: precomputed nodal filter ---
    def _sigma(self, p: int, alpha: float) -> np.ndarray:
        k = np.arange(self.N, dtype=float)
        kmax = max(float(self.N - 1), 1.0)
        eta = k / kmax
        return np.exp(-float(alpha) * eta ** int(p))

    def _build_nodal_filter_np(self, p: int, alpha: float) -> np.ndarray:
        key = (int(p), float(alpha))
        if key in self._nodal_filter_cache_np:
            return self._nodal_filter_cache_np[key]
        # Combine modal sigma with projection scaling
        sigma_proj = self._sigma(p, alpha) * self._proj_scale  # (N,)
        M1 = self._V * sigma_proj.reshape((1, self.N))
        M2 = M1 @ self._V.T
        Fn = M2 * self._w_ref.reshape((1, self.N))
        self._nodal_filter_cache_np[key] = Fn
        return Fn

    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        # f_new = f @ Fn^T along last axis
        if _TORCH_AVAILABLE and isinstance(f, torch.Tensor):  # type: ignore[arg-type]
            mats = self._get_torch_mats(f)
            # Per-device/dtype cache
            dev_key = (f.device.type + (str(f.device.index) if f.device.index is not None else ""), str(f.dtype))
            cache = self._torch_cache.setdefault(dev_key, {})
            kname = f"Fn_{p}_{alpha}"
            if kname not in cache:
                sigma = torch.from_numpy(self._sigma(p, alpha)).to(dtype=f.dtype, device=f.device)
                sigma_proj = sigma * mats["proj"]
                M1 = mats["V"] * sigma_proj.reshape((1, self.N))
                M2 = torch.matmul(M1, mats["V"].T)
                Fn = M2 * mats["w_ref"].reshape((1, self.N))
                cache[kname] = Fn
            Fn = cache[kname]
            return torch.matmul(f, Fn.T)
        Fn_np = self._build_nodal_filter_np(p, alpha)
        return _as_last_axis_matrix_apply(Fn_np.T, f)

    # --- Torch helpers ---
    def _get_torch_mats(self, ref: "torch.Tensor") -> Dict[str, "torch.Tensor"]:  # type: ignore[name-defined]
        assert _TORCH_AVAILABLE and torch is not None
        dev_key = (ref.device.type + (str(ref.device.index) if ref.device.index is not None else ""), str(ref.dtype))
        cache = self._torch_cache.get(dev_key)
        if cache is None:
            cache = {}
            self._torch_cache[dev_key] = cache
        if "V" not in cache:
            cache["V"] = torch.from_numpy(self._V).to(dtype=ref.dtype, device=ref.device)
        if "D_x" not in cache:
            cache["D_x"] = torch.from_numpy(self._D_x).to(dtype=ref.dtype, device=ref.device)
        if "w_ref" not in cache:
            cache["w_ref"] = torch.from_numpy(self._w_ref).to(dtype=ref.dtype, device=ref.device)
        if "proj" not in cache:
            cache["proj"] = torch.from_numpy(self._proj_scale).to(dtype=ref.dtype, device=ref.device)
        return cache


