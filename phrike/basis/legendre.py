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

try:
    import mpmath
    _MPMATH_AVAILABLE = True
except ImportError:
    _MPMATH_AVAILABLE = False
    mpmath = None

try:
    from numba import njit, prange
    _NUMBA_AVAILABLE = True
except Exception:
    _NUMBA_AVAILABLE = False
    
    def njit(*args, **kwargs):  # type: ignore
        def inner(func):
            return func
        return inner
    
    def prange(*args, **kwargs):  # type: ignore
        return range(*args, **kwargs)

# Import scipy functions for high-precision LGL computation
from scipy.special import roots_jacobi, eval_legendre


# --- Numba JIT-compiled kernels for Legendre operations ---

@njit(cache=True, fastmath=True)
def _legendre_forward_kernel_numba(fw: np.ndarray, V: np.ndarray, proj_scale: np.ndarray) -> np.ndarray:
    """JIT-compiled kernel for Legendre forward transform.
    
    Args:
        fw: Weighted nodal values of shape (..., N)
        V: Vandermonde matrix of shape (N, N)
        proj_scale: Projection scaling factors of shape (N,)
        
    Returns:
        Modal coefficients of shape (..., N)
    """
    # Use matrix multiplication for efficiency
    a = fw @ V
    # Apply projection scaling - use explicit broadcasting
    N = proj_scale.shape[0]
    if fw.ndim == 1:
        # 1D case
        for i in range(N):
            a[i] *= proj_scale[i]
    else:
        # Multi-dimensional case
        for i in range(N):
            a[..., i] *= proj_scale[i]
    return a


@njit(cache=True, fastmath=True)
def _legendre_inverse_kernel_numba(F: np.ndarray, V: np.ndarray) -> np.ndarray:
    """JIT-compiled kernel for Legendre inverse transform.
    
    Args:
        F: Modal coefficients of shape (..., N)
        V: Vandermonde matrix of shape (N, N)
        
    Returns:
        Nodal values of shape (..., N)
    """
    return F @ V.T


@njit(cache=True, fastmath=True)
def _legendre_derivative_kernel_numba(f: np.ndarray, D: np.ndarray) -> np.ndarray:
    """JIT-compiled kernel for Legendre differentiation.
    
    Args:
        f: Nodal values of shape (..., N)
        D: Differentiation matrix of shape (N, N)
        
    Returns:
        Derivative values of shape (..., N)
    """
    return f @ D.T


@njit(cache=True, fastmath=True)
def _legendre_filter_kernel_numba(f: np.ndarray, Fn: np.ndarray) -> np.ndarray:
    """JIT-compiled kernel for Legendre spectral filtering.
    
    Args:
        f: Nodal values of shape (..., N)
        Fn: Filter matrix of shape (N, N)
        
    Returns:
        Filtered nodal values of shape (..., N)
    """
    return f @ Fn.T


@njit(cache=True, fastmath=True)
def _legendre_2d_derivative_kernel_numba(f: np.ndarray, D: np.ndarray, axis: int) -> np.ndarray:
    """JIT-compiled kernel for 2D Legendre differentiation.
    
    Args:
        f: Nodal values of shape (..., Ny, Nx)
        D: Differentiation matrix of shape (N, N)
        axis: Axis along which to differentiate (-1 for x, -2 for y)
        
    Returns:
        Derivative values of shape (..., Ny, Nx)
    """
    # Handle 3D arrays (components, Ny, Nx) or 2D arrays (Ny, Nx)
    if f.ndim == 3:
        ncomp, Ny, Nx = f.shape
        result = np.zeros_like(f)
        
        if axis == -1:  # x-derivative
            for c in range(ncomp):
                for i in range(Ny):
                    for j in range(Nx):
                        for k in range(Nx):
                            result[c, i, j] += f[c, i, k] * D[k, j]
        else:  # y-derivative (axis == -2)
            for c in range(ncomp):
                for i in range(Ny):
                    for j in range(Nx):
                        for k in range(Ny):
                            result[c, i, j] += f[c, k, j] * D[k, i]
        return result
    else:
        # Handle 2D arrays (Ny, Nx)
        Ny, Nx = f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        if axis == -1:  # x-derivative
            for i in range(Ny):
                for j in range(Nx):
                    for k in range(Nx):
                        result[i, j] += f[i, k] * D[k, j]
        else:  # y-derivative (axis == -2)
            for i in range(Ny):
                for j in range(Nx):
                    for k in range(Ny):
                        result[i, j] += f[k, j] * D[k, i]
        return result


@njit(cache=True, fastmath=True)
def _legendre_2d_filter_kernel_numba(f: np.ndarray, Fm: np.ndarray, axis: int) -> np.ndarray:
    """JIT-compiled kernel for 2D Legendre spectral filtering.
    
    Args:
        f: Nodal values of shape (..., Ny, Nx)
        Fm: Filter matrix of shape (N, N)
        axis: Axis along which to apply filter (-1 for x, -2 for y)
        
    Returns:
        Filtered nodal values of shape (..., Ny, Nx)
    """
    # Handle 3D arrays (components, Ny, Nx) or 2D arrays (Ny, Nx)
    if f.ndim == 3:
        ncomp, Ny, Nx = f.shape
        result = np.zeros_like(f)
        
        if axis == -1:  # x-filter
            for c in range(ncomp):
                for i in range(Ny):
                    for j in range(Nx):
                        for k in range(Nx):
                            result[c, i, j] += f[c, i, k] * Fm[k, j]
        else:  # y-filter (axis == -2)
            for c in range(ncomp):
                for i in range(Ny):
                    for j in range(Nx):
                        for k in range(Ny):
                            result[c, i, j] += f[c, k, j] * Fm[k, i]
        return result
    else:
        # Handle 2D arrays (Ny, Nx)
        Ny, Nx = f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        if axis == -1:  # x-filter
            for i in range(Ny):
                for j in range(Nx):
                    for k in range(Nx):
                        result[i, j] += f[i, k] * Fm[k, j]
        else:  # y-filter (axis == -2)
            for i in range(Ny):
                for j in range(Nx):
                    for k in range(Ny):
                        result[i, j] += f[k, j] * Fm[k, i]
        return result


@njit(cache=True, fastmath=True)
def _legendre_3d_derivative_kernel_numba(f: np.ndarray, D: np.ndarray, axis: int) -> np.ndarray:
    """JIT-compiled kernel for 3D Legendre differentiation.
    
    Args:
        f: Nodal values of shape (..., Nz, Ny, Nx)
        D: Differentiation matrix of shape (N, N)
        axis: Axis along which to differentiate (-1 for x, -2 for y, -3 for z)
        
    Returns:
        Derivative values of shape (..., Nz, Ny, Nx)
    """
    Nz, Ny, Nx = f.shape[-3], f.shape[-2], f.shape[-1]
    result = np.zeros_like(f)
    
    if axis == -1:  # x-derivative
        # Apply along x-axis manually
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Nx):
                        result[..., i, j, k] += f[..., i, j, l] * D[l, k]
    elif axis == -2:  # y-derivative
        # Apply along y-axis manually
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Ny):
                        result[..., i, j, k] += f[..., i, l, k] * D[l, j]
    else:  # z-derivative (axis == -3)
        # Apply along z-axis manually
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Nz):
                        result[..., i, j, k] += f[..., l, j, k] * D[l, i]
    
    return result


# --- Parallel Numba JIT-compiled kernels for Legendre operations ---

@njit(cache=True, fastmath=True, parallel=True)
def _legendre_forward_kernel_numba_parallel(fw: np.ndarray, V: np.ndarray, proj_scale: np.ndarray) -> np.ndarray:
    """JIT-compiled parallel kernel for Legendre forward transform.
    
    Args:
        fw: Weighted nodal values of shape (..., N)
        V: Vandermonde matrix of shape (N, N)
        proj_scale: Projection scaling factors of shape (N,)
        
    Returns:
        Modal coefficients of shape (..., N)
    """
    # Use matrix multiplication for efficiency
    a = fw @ V
    # Apply projection scaling
    for i in range(a.shape[-1]):
        a[..., i] *= proj_scale[i]
    return a


@njit(cache=True, fastmath=True, parallel=True)
def _legendre_inverse_kernel_numba_parallel(F: np.ndarray, V: np.ndarray) -> np.ndarray:
    """JIT-compiled parallel kernel for Legendre inverse transform.
    
    Args:
        F: Modal coefficients of shape (..., N)
        V: Vandermonde matrix of shape (N, N)
        
    Returns:
        Nodal values of shape (..., N)
    """
    return F @ V.T


@njit(cache=True, fastmath=True, parallel=True)
def _legendre_derivative_kernel_numba_parallel(f: np.ndarray, D: np.ndarray) -> np.ndarray:
    """JIT-compiled parallel kernel for Legendre differentiation.
    
    Args:
        f: Nodal values of shape (..., N)
        D: Differentiation matrix of shape (N, N)
        
    Returns:
        Derivative values of shape (..., N)
    """
    return f @ D.T


@njit(cache=True, fastmath=True, parallel=True)
def _legendre_filter_kernel_numba_parallel(f: np.ndarray, Fn: np.ndarray) -> np.ndarray:
    """JIT-compiled parallel kernel for Legendre spectral filtering.
    
    Args:
        f: Nodal values of shape (..., N)
        Fn: Filter matrix of shape (N, N)
        
    Returns:
        Filtered nodal values of shape (..., N)
    """
    return f @ Fn.T


@njit(cache=True, fastmath=True, parallel=True)
def _legendre_2d_derivative_kernel_numba_parallel(f: np.ndarray, D: np.ndarray, axis: int) -> np.ndarray:
    """JIT-compiled parallel kernel for 2D Legendre differentiation.
    
    Args:
        f: Nodal values of shape (..., Ny, Nx)
        D: Differentiation matrix of shape (N, N)
        axis: Axis along which to differentiate (-1 for x, -2 for y)
        
    Returns:
        Derivative values of shape (..., Ny, Nx)
    """
    # Handle 3D arrays (components, Ny, Nx) or 2D arrays (Ny, Nx)
    if f.ndim == 3:
        ncomp, Ny, Nx = f.shape
        result = np.zeros_like(f)
        
        if axis == -1:  # x-derivative
            for c in prange(ncomp):
                for i in range(Ny):
                    for j in range(Nx):
                        for k in range(Nx):
                            result[c, i, j] += f[c, i, k] * D[k, j]
        else:  # y-derivative (axis == -2)
            for c in prange(ncomp):
                for i in range(Ny):
                    for j in range(Nx):
                        for k in range(Ny):
                            result[c, i, j] += f[c, k, j] * D[k, i]
        return result
    else:
        # Handle 2D arrays (Ny, Nx)
        Ny, Nx = f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        if axis == -1:  # x-derivative
            for i in prange(Ny):
                for j in range(Nx):
                    for k in range(Nx):
                        result[i, j] += f[i, k] * D[k, j]
        else:  # y-derivative (axis == -2)
            for i in prange(Ny):
                for j in range(Nx):
                    for k in range(Ny):
                        result[i, j] += f[k, j] * D[k, i]
        return result


@njit(cache=True, fastmath=True, parallel=True)
def _legendre_2d_filter_kernel_numba_parallel(f: np.ndarray, Fm: np.ndarray, axis: int) -> np.ndarray:
    """JIT-compiled parallel kernel for 2D Legendre spectral filtering.
    
    Args:
        f: Nodal values of shape (..., Ny, Nx)
        Fm: Filter matrix of shape (N, N)
        axis: Axis along which to apply filter (-1 for x, -2 for y)
        
    Returns:
        Filtered nodal values of shape (..., Ny, Nx)
    """
    # Handle 3D arrays (components, Ny, Nx) or 2D arrays (Ny, Nx)
    if f.ndim == 3:
        ncomp, Ny, Nx = f.shape
        result = np.zeros_like(f)
        
        if axis == -1:  # x-filter
            for c in prange(ncomp):
                for i in range(Ny):
                    for j in range(Nx):
                        for k in range(Nx):
                            result[c, i, j] += f[c, i, k] * Fm[k, j]
        else:  # y-filter (axis == -2)
            for c in prange(ncomp):
                for i in range(Ny):
                    for j in range(Nx):
                        for k in range(Ny):
                            result[c, i, j] += f[c, k, j] * Fm[k, i]
        return result
    else:
        # Handle 2D arrays (Ny, Nx)
        Ny, Nx = f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        if axis == -1:  # x-filter
            for i in prange(Ny):
                for j in range(Nx):
                    for k in range(Nx):
                        result[i, j] += f[i, k] * Fm[k, j]
        else:  # y-filter (axis == -2)
            for i in prange(Ny):
                for j in range(Nx):
                    for k in range(Ny):
                        result[i, j] += f[k, j] * Fm[k, i]
        return result


@njit(cache=True, fastmath=True, parallel=True)
def _legendre_3d_derivative_kernel_numba_parallel(f: np.ndarray, D: np.ndarray, axis: int) -> np.ndarray:
    """JIT-compiled parallel kernel for 3D Legendre differentiation.
    
    Args:
        f: Nodal values of shape (..., Nz, Ny, Nx)
        D: Differentiation matrix of shape (N, N)
        axis: Axis along which to differentiate (-1 for x, -2 for y, -3 for z)
        
    Returns:
        Derivative values of shape (..., Nz, Ny, Nx)
    """
    if axis == -1:  # x-derivative
        return f @ D.T
    elif axis == -2:  # y-derivative
        # Apply along y-axis manually with parallel loops
        Nz, Ny, Nx = f.shape[-3], f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Ny):
                        result[..., i, j, k] += f[..., i, l, k] * D[l, j]
        return result
    else:  # z-derivative (axis == -3)
        # Apply along z-axis manually with parallel loops
        Nz, Ny, Nx = f.shape[-3], f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Nz):
                        result[..., i, j, k] += f[..., l, j, k] * D[l, i]
        return result


@njit(cache=True, fastmath=True, parallel=True)
def _legendre_3d_filter_kernel_numba_parallel(f: np.ndarray, Fm: np.ndarray, axis: int) -> np.ndarray:
    """JIT-compiled parallel kernel for 3D Legendre spectral filtering.
    
    Args:
        f: Nodal values of shape (..., Nz, Ny, Nx)
        Fm: Filter matrix of shape (N, N)
        axis: Axis along which to apply filter (-1 for x, -2 for y, -3 for z)
        
    Returns:
        Filtered nodal values of shape (..., Nz, Ny, Nx)
    """
    if axis == -1:  # x-filter
        return f @ Fm.T
    elif axis == -2:  # y-filter
        # Apply along y-axis manually with parallel loops
        Nz, Ny, Nx = f.shape[-3], f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Ny):
                        result[..., i, j, k] += f[..., i, l, k] * Fm[l, j]
        return result
    else:  # z-filter (axis == -3)
        # Apply along z-axis manually with parallel loops
        Nz, Ny, Nx = f.shape[-3], f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Nz):
                        result[..., i, j, k] += f[..., l, j, k] * Fm[l, i]
        return result


@njit(cache=True, fastmath=True)
def _legendre_3d_filter_kernel_numba(f: np.ndarray, Fm: np.ndarray, axis: int) -> np.ndarray:
    """JIT-compiled kernel for 3D Legendre spectral filtering.
    
    Args:
        f: Nodal values of shape (..., Nz, Ny, Nx)
        Fm: Filter matrix of shape (N, N)
        axis: Axis along which to apply filter (-1 for x, -2 for y, -3 for z)
        
    Returns:
        Filtered nodal values of shape (..., Nz, Ny, Nx)
    """
    if axis == -1:  # x-filter
        return f @ Fm.T
    elif axis == -2:  # y-filter
        # Apply along y-axis manually with loops
        Nz, Ny, Nx = f.shape[-3], f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Ny):
                        result[..., i, j, k] += f[..., i, l, k] * Fm[l, j]
        return result
    else:  # z-filter (axis == -3)
        # Apply along z-axis manually with loops
        Nz, Ny, Nx = f.shape[-3], f.shape[-2], f.shape[-1]
        result = np.zeros_like(f)
        
        for i in range(Nz):
            for j in range(Ny):
                for k in range(Nx):
                    for l in range(Nz):
                        result[..., i, j, k] += f[..., l, j, k] * Fm[l, i]
        return result


def _legendre_gauss_lobatto_nodes_weights(N: int, precision: str = "double") -> tuple[np.ndarray, np.ndarray]:
    """Return Legendre–Gauss–Lobatto nodes y in [-1, 1] and quadrature weights w.

    For N >= 2, the nodes are the endpoints ±1 and the (N-2) roots of
    d/dy P_{N-1}(y). The weights are

        w_j = 2 / [N(N-1) (P_{N-1}(y_j))^2].

    Uses high-precision computation (quadruple precision) for accuracy,
    then converts to the requested precision.

    Args:
        N: number of nodes (>= 2). If N == 1, returns y=[0], w=[2].
        precision: target precision ("single" or "double")

    Returns:
        (y, w): nodes and weights arrays of shape (N,) in target precision.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if N == 1:
        # Degenerate case: single node at center with full weight 2
        dtype = np.float32 if precision == "single" else np.float64
        return np.array([0.0], dtype=dtype), np.array([2.0], dtype=dtype)

    # Use high-precision computation if available
    if _MPMATH_AVAILABLE:
        # Compute in quadruple precision (32 digits) for maximum accuracy
        x_mp, w_mp = _gausslobatto_mpmath(N, precision=32)
        
        # Convert to numpy arrays
        y = np.array([float(xi) for xi in x_mp])
        w = np.array([float(wi) for wi in w_mp])
        
        # Convert to target precision
        if precision == "single":
            y = y.astype(np.float32)
            w = w.astype(np.float32)
        else:  # double precision
            y = y.astype(np.float64)
            w = w.astype(np.float64)
            
        return y, w
    
    # Fallback to original implementation if fast_lgl not available
    # Endpoints
    y0 = -1.0
    yN = 1.0

    if N == 2:
        dtype = np.float32 if precision == "single" else np.float64
        y = np.array([y0, yN], dtype=dtype)
        # P_{N-1} = P_1, P_1(±1) = ±1 → w0 = wN = 1
        # However, formula gives 2/[2*1*(±1)^2] = 1 each; sum weights = 2
        w = np.array([1.0, 1.0], dtype=dtype)
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
    
    # Convert to target precision
    if precision == "single":
        y = y.astype(np.float32)
        w = w.astype(np.float32)
    else:  # double precision
        y = y.astype(np.float64)
        w = w.astype(np.float64)
        
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


# High-precision LGL computation functions
def _gausslobatto_mpmath(n: int, precision: int = 32):
    """mpmath implementation of Gauss-Lobatto quadrature with arbitrary precision."""
    if not _MPMATH_AVAILABLE:
        raise ImportError("mpmath is required for high-precision computation but not installed")
    
    old_precision = mpmath.mp.dps
    mpmath.mp.dps = precision
    
    try:
        if n == 2:
            x = mpmath.matrix([-1, 1])
            w = mpmath.matrix([1, 1])
            return x, w
        elif n == 3:
            x = mpmath.matrix([-1, 0, 1])
            w = mpmath.matrix([1/3, 4/3, 1/3])
            return x, w
        else:
            # For n > 3, we need to compute Jacobi quadrature in high precision
            # This is a simplified version - a full implementation would use
            # mpmath's own quadrature routines
            
            # Get interior points using scipy (double precision)
            x_interior_scipy, w_interior_scipy = roots_jacobi(n - 2, 1.0, 1.0)
            P_values_scipy = eval_legendre(n - 1, x_interior_scipy)
            
            # Convert to mpmath for high precision arithmetic
            x_interior = mpmath.matrix([mpmath.mpf(float(x)) for x in x_interior_scipy])
            P_values = mpmath.matrix([mpmath.mpf(float(p)) for p in P_values_scipy])
            
            # Compute weights in high precision
            w_interior = mpmath.matrix([
                2 / (n * (n - 1) * p**2) for p in P_values
            ])
            
            # Add endpoints
            x = mpmath.matrix([-1] + list(x_interior) + [1])
            endpoint_weight = 2 / (n * (n - 1))
            w = mpmath.matrix([endpoint_weight] + list(w_interior) + [endpoint_weight])
            
            return x, w
    
    finally:
        mpmath.mp.dps = old_precision


def gausslobatto_mpmath(n: int, precision: int = 32):
    """Convenience function for mpmath backend with specified precision."""
    if not _MPMATH_AVAILABLE:
        raise ImportError("mpmath is required for mpmath backend but not installed")
    return _gausslobatto_mpmath(n, precision=precision)


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

    def __init__(self, N: int, Lx: float, *, bc: str = "dirichlet", precision: str = "double", 
                 use_parallel: bool = False) -> None:
        super().__init__(N=N, Lx=Lx, bc=bc)
        self.precision = precision
        self.use_parallel = use_parallel

        # Nodes and quadrature on reference domain [-1, 1]
        y, w_ref = _legendre_gauss_lobatto_nodes_weights(self.N, precision=self.precision)
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
        
        # Use JIT-compiled kernel for NumPy backend if available
        if _NUMBA_AVAILABLE:
            if self.use_parallel:
                return _legendre_forward_kernel_numba_parallel(fw, V, self._proj_scale)
            else:
                return _legendre_forward_kernel_numba(fw, V, self._proj_scale)
        else:
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
        
        # Use JIT-compiled kernel for NumPy backend if available
        if _NUMBA_AVAILABLE:
            if self.use_parallel:
                return _legendre_inverse_kernel_numba_parallel(F_arr, V)
            else:
                return _legendre_inverse_kernel_numba(F_arr, V)
        else:
            return np.tensordot(F_arr, V.T, axes=([-1], [0]))

    # Derivatives
    def dx(self, f: np.ndarray) -> np.ndarray:
        # Apply D_x along last axis
        if _TORCH_AVAILABLE and isinstance(f, torch.Tensor):  # type: ignore[arg-type]
            mats = self._get_torch_mats(f)
            return torch.matmul(f, mats["D_x"].T)
        
        # Use JIT-compiled kernel for NumPy backend if available
        if _NUMBA_AVAILABLE:
            # Ensure matrix has same dtype as input to avoid Numba dtype mismatch
            D_x_typed = self._D_x.astype(f.dtype)
            if self.use_parallel:
                return _legendre_derivative_kernel_numba_parallel(f, D_x_typed)
            else:
                return _legendre_derivative_kernel_numba(f, D_x_typed)
        else:
            return _as_last_axis_matrix_apply(self._D_x.T, f)

    def d2x(self, f: np.ndarray) -> np.ndarray:
        # Apply D_x twice; cache D2_x on device to avoid recomputation
        if _TORCH_AVAILABLE and isinstance(f, torch.Tensor):  # type: ignore[arg-type]
            mats = self._get_torch_mats(f)
            if "D2_x" not in mats:
                mats["D2_x"] = torch.matmul(mats["D_x"], mats["D_x"])  # (N,N)
            return torch.matmul(f, mats["D2_x"].T)
        
        # Use JIT-compiled kernel for NumPy backend if available
        if _NUMBA_AVAILABLE:
            # Precompute D2_x for efficiency and ensure dtype match
            D2_x = self._D_x @ self._D_x
            D2_x_typed = D2_x.astype(f.dtype)
            if self.use_parallel:
                return _legendre_derivative_kernel_numba_parallel(f, D2_x_typed)
            else:
                return _legendre_derivative_kernel_numba(f, D2_x_typed)
        else:
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
        
        # Use JIT-compiled kernel for NumPy backend if available
        if _NUMBA_AVAILABLE:
            # Ensure filter matrix has same dtype as input to avoid Numba dtype mismatch
            Fn_np_typed = Fn_np.astype(f.dtype)
            if self.use_parallel:
                return _legendre_filter_kernel_numba_parallel(f, Fn_np_typed)
            else:
                return _legendre_filter_kernel_numba(f, Fn_np_typed)
        else:
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
            cache["V"] = torch.from_numpy(self._V).to(dtype=ref.dtype, device=ref.device, non_blocking=True)
        if "D_x" not in cache:
            cache["D_x"] = torch.from_numpy(self._D_x).to(dtype=ref.dtype, device=ref.device, non_blocking=True)
        if "w_ref" not in cache:
            cache["w_ref"] = torch.from_numpy(self._w_ref).to(dtype=ref.dtype, device=ref.device, non_blocking=True)
        if "proj" not in cache:
            cache["proj"] = torch.from_numpy(self._proj_scale).to(dtype=ref.dtype, device=ref.device, non_blocking=True)
        return cache



class LegendreLobattoBasis2D:
    """Tensor-product LGL basis on [0,Lx]×[0,Ly] with nodal ops.

    Last two axes are (..., Ny, Nx). Derivatives via precomputed Dx,Dy.
    """

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float, *, bc: str = "dirichlet", precision: str = "double", 
                 use_parallel: bool = False) -> None:
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.bc = str(bc).lower()
        self.precision = precision
        self.use_parallel = use_parallel

        # Temporarily disabled 2D Legendre; keep constructor minimal to allow future re-implementation
        self._x = np.linspace(0.0, self.Lx, self.Nx)
        self._y = np.linspace(0.0, self.Ly, self.Ny)
        self._w_x = None
        self._w_y = None
        self._Dx = None
        self._Dy = None
        self._Vx = None
        self._Vy = None
        self._filter_cache_np = {}

    def quadrature_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("2D Legendre quadrature weights are not implemented yet.")

    def nodes(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._x, self._y

    def dx(self, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError("2D Legendre derivative dx is not implemented yet.")

    def dy(self, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError("2D Legendre derivative dy is not implemented yet.")

    def _filter_axis(self, f: np.ndarray, V: np.ndarray, p: int, alpha: float, axis: int) -> np.ndarray:
        raise NotImplementedError("2D Legendre spectral filter is not implemented yet.")

    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        raise NotImplementedError("2D Legendre spectral filter is not implemented yet.")


class LegendreLobattoBasis3D:
    def __init__(self, Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float, *, bc: str = "dirichlet", precision: str = "double", 
                 use_parallel: bool = False) -> None:
        self.Nx = int(Nx); self.Ny = int(Ny); self.Nz = int(Nz)
        self.Lx = float(Lx); self.Ly = float(Ly); self.Lz = float(Lz)
        self.bc = str(bc).lower()
        self.precision = precision
        self.use_parallel = use_parallel

        # Temporarily disabled 3D Legendre; keep minimal placeholders
        self._x = np.linspace(0.0, self.Lx, self.Nx)
        self._y = np.linspace(0.0, self.Ly, self.Ny)
        self._z = np.linspace(0.0, self.Lz, self.Nz)
        self._Dx = None
        self._Dy = None
        self._Dz = None
        self._Vx = None
        self._Vy = None
        self._Vz = None

    def nodes(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError("3D Legendre nodes are not implemented yet.")

    def dx(self, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError("3D Legendre derivative dx is not implemented yet.")

    def dy(self, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError("3D Legendre derivative dy is not implemented yet.")

    def dz(self, f: np.ndarray) -> np.ndarray:
        raise NotImplementedError("3D Legendre derivative dz is not implemented yet.")

    def _filter_axis(self, f: np.ndarray, V: np.ndarray, p: int, alpha: float, axis: int) -> np.ndarray:
        raise NotImplementedError("3D Legendre spectral filter is not implemented yet.")

    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        raise NotImplementedError("3D Legendre spectral filter is not implemented yet.")
