from __future__ import annotations

from typing import Optional

import numpy as np

try:  # Prefer pyfftw if available for performance
    import pyfftw.interfaces.scipy_fft as scipy_fft  # type: ignore
    from pyfftw.interfaces import cache as fftw_cache  # type: ignore

    _FFT_CACHE = True
except Exception:  # Fallback to SciPy's FFT
    from scipy import fft as scipy_fft  # type: ignore

    _FFT_CACHE = False
    fftw_cache = None  # type: ignore

from .base import Basis1D, _as_last_axis_matrix_apply
def _build_chebyshev_diff_matrix_y(y: np.ndarray) -> np.ndarray:
    """Chebyshev–Gauss–Lobatto first-derivative matrix with respect to y.

    Based on standard Trefethen formula on CGL nodes y_j = cos(pi*j/(N-1)).
    """
    N = y.shape[0]
    if N == 1:
        return np.zeros((1, 1))
    c = np.ones(N)
    c[0] = 2.0
    c[-1] = 2.0
    c = c * ((-1.0) ** np.arange(N))
    Y = y[:, None]
    dY = Y - Y.T
    D = (c[:, None] / c[None, :]) / dY
    np.fill_diagonal(D, 0.0)
    D = D - np.diag(np.sum(D, axis=1))
    return D

# Optional torch support (used only for input/output conversion)
try:  # pragma: no cover - torch optional
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False
    torch = None  # type: ignore


def _cheb_nodes_gl(N: int) -> np.ndarray:
    """Chebyshev-Gauss-Lobatto nodes on [-1, 1]."""
    if N == 1:
        return np.array([1.0])
    j = np.arange(N, dtype=float)
    return np.cos(np.pi * j / (N - 1))


def _dct1_none(a: np.ndarray, axis: int) -> np.ndarray:
    return scipy_fft.dct(a, type=1, axis=axis, norm=None)


def _idct1_none(A: np.ndarray, axis: int) -> np.ndarray:
    return scipy_fft.idct(A, type=1, axis=axis, norm=None)


class ChebyshevBasis1D(Basis1D):
    """Chebyshev basis on Chebyshev-Gauss-Lobatto nodes using DCT-I.

    Nodes: Chebyshev-Gauss-Lobatto points mapped to [0, Lx].
    Derivatives: computed via spectral recurrence relations.
    """

    def __init__(self, N: int, Lx: float, *, bc: str = "dirichlet", fft_workers: int = 1) -> None:
        super().__init__(N=N, Lx=Lx, bc=bc)
        self.fft_workers = int(fft_workers)

        # Chebyshev-Gauss-Lobatto nodes on [-1, 1]
        self._y = _cheb_nodes_gl(self.N)
        # Map to [0, Lx]: x = (1-y) * Lx/2
        self._x = (1.0 - self._y) * (self.Lx / 2.0)
        # Derivative scaling: y = 1 - 2x/Lx -> dy/dx = -2/Lx
        self._dy_dx = -2.0 / self.Lx

        # Precompute Chebyshev first- and second-derivative matrices in y
        self._D_y = _build_chebyshev_diff_matrix_y(self._y)
        self._D2_y = self._D_y @ self._D_y

        # Enable cache if using FFTW interface
        if _FFT_CACHE and fftw_cache is not None:
            try:
                fftw_cache.enable()
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

    # Grid and weights
    def nodes(self) -> np.ndarray:
        return self._x

    def quadrature_weights(self) -> Optional[np.ndarray]:
        """Clenshaw-Curtis quadrature weights for Chebyshev-Gauss-Lobatto nodes."""
        N = self.N
        if N == 1:
            return np.array([self.Lx], dtype=float)
        
        # Clenshaw-Curtis weights on [-1, 1]
        w = np.zeros(N)
        for j in range(N):
            if j == 0 or j == N - 1:
                w[j] = 1.0 / (N - 1)
            else:
                # Sum over even cosine terms
                w[j] = 2.0 / (N - 1)
                for k in range(1, (N - 1) // 2 + 1):
                    w[j] += 2.0 / (N - 1) * np.cos(2.0 * np.pi * k * j / (N - 1)) / (1.0 - 4.0 * k * k)
        
        # Scale to [0, Lx]
        return w * (self.Lx / 2.0)

    # Transforms (orthonormal DCT-I pair)
    def forward(self, f: np.ndarray) -> np.ndarray:
        return scipy_fft.dct(f, type=1, axis=-1, norm="ortho")

    def inverse(self, F: np.ndarray) -> np.ndarray:
        return scipy_fft.idct(F, type=1, axis=-1, norm="ortho")

    # Derivatives
    def dx(self, f: np.ndarray) -> np.ndarray:
        """Differentiate along x using Chebyshev collocation matrix.

        This uses the analytically correct differentiation matrix on the
        Chebyshev–Gauss–Lobatto nodes for d/dy and maps to d/dx via dy/dx.
        """
        is_torch = False
        if _TORCH_AVAILABLE and isinstance(f, (torch.Tensor,)):
            is_torch = True
            f_np = f.detach().cpu().numpy()
        else:
            f_np = np.asarray(f)

        df_dy = _as_last_axis_matrix_apply(self._D_y.T, f_np)
        df_dx = df_dy * self._dy_dx
        if is_torch:
            return torch.from_numpy(df_dx).to(dtype=f.dtype, device=f.device)
        return df_dx

    def d2x(self, f: np.ndarray) -> np.ndarray:
        """Second derivative using collocation matrix squared."""
        is_torch = False
        if _TORCH_AVAILABLE and isinstance(f, (torch.Tensor,)):
            is_torch = True
            f_np = f.detach().cpu().numpy()
        else:
            f_np = np.asarray(f)

        d2f_dy2 = _as_last_axis_matrix_apply(self._D2_y.T, f_np)
        d2f_dx2 = d2f_dy2 * (self._dy_dx ** 2)
        if is_torch:
            return torch.from_numpy(d2f_dx2).to(dtype=f.dtype, device=f.device)
        return d2f_dx2

    # Override filtering to use consistent (norm=None) Chebyshev coefficients
    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        # Handle optional torch input
        is_torch = False
        if _TORCH_AVAILABLE and isinstance(f, (torch.Tensor,)):
            is_torch = True
            f_np = f.detach().cpu().numpy()
        else:
            f_np = np.asarray(f)

        N = f_np.shape[-1]
        if N <= 1:
            return f

        # Forward to standard Chebyshev coefficients (no normalization)
        a = scipy_fft.dct(f_np, type=1, axis=-1, norm=None)
        a = a / (N - 1)
        a[..., 0] *= 0.5
        a[..., -1] *= 0.5

        # Exponential modal filter on Chebyshev index k
        k = np.arange(N, dtype=float)
        kmax = max(float(N - 1), 1.0)
        eta = k / kmax
        sigma = np.exp(-float(alpha) * eta**int(p))
        shape = (1,) * (a.ndim - 1) + (N,)
        a_filtered = a * sigma.reshape(shape)

        # Inverse using cosine-sum with doubled interior modes (norm=None)
        a_sum = a_filtered.copy()
        if N > 2:
            a_sum[..., 1:-1] *= 2.0
        f_filtered = scipy_fft.idct(a_sum, type=1, axis=-1, norm=None) / (N - 1)

        if is_torch:
            return torch.from_numpy(f_filtered).to(dtype=f.dtype, device=f.device)
        return f_filtered

    # --- 3/2 zero-padding dealiasing helpers (1D) ---
    def _coeffs_standard(self, f_np: np.ndarray) -> np.ndarray:
        """Return standard Chebyshev coefficients (norm=None) for f on N nodes.

        a = DCT-I(f, norm=None)/(N-1) with a0 and a_{N-1} halved.
        Works on the last axis.
        """
        N = f_np.shape[-1]
        a = scipy_fft.dct(f_np, type=1, axis=-1, norm=None)
        a = a / max(N - 1, 1)
        if N >= 1:
            a[..., 0] *= 0.5
            a[..., -1] *= 0.5
        return a

    def _inverse_from_coeffs(self, a: np.ndarray) -> np.ndarray:
        """Inverse from standard coefficients a using IDCT-I (norm=None).

        a has endpoints already halved; we double interior modes and divide by (N-1).
        """
        N = a.shape[-1]
        a_sum = a.copy()
        if N > 2:
            a_sum[..., 1:-1] *= 2.0
        return scipy_fft.idct(a_sum, type=1, axis=-1, norm=None) / max(N - 1, 1)

    def oversample(self, f: np.ndarray, M: int) -> np.ndarray:
        """Evaluate f on M Chebyshev–Lobatto nodes via zero-padded coefficients.

        Args:
            f: values on N nodes (last axis length N)
            M: oversampled number of nodes (M > N)
        Returns:
            f_os on M nodes (last axis length M)
        """
        f_np = np.asarray(f)
        N = f_np.shape[-1]
        a_N = self._coeffs_standard(f_np)
        # Zero-pad coefficients to length M
        shape_M = list(a_N.shape)
        shape_M[-1] = M
        a_M = np.zeros(shape_M, dtype=a_N.dtype)
        a_M[..., :N] = a_N
        # Evaluate on M nodes
        return self._inverse_from_coeffs(a_M)

    def project_to_N(self, f_os: np.ndarray, M: int) -> np.ndarray:
        """Project M-node values back to N Chebyshev nodes by truncating coeffs.

        Args:
            f_os: values on M nodes
            M: number of oversampled nodes provided
        Returns:
            f_N: values on original N nodes
        """
        fM = np.asarray(f_os)
        # Build coefficients at M
        a_M = scipy_fft.dct(fM, type=1, axis=-1, norm=None)
        a_M = a_M / max(M - 1, 1)
        if M >= 1:
            a_M[..., 0] *= 0.5
            a_M[..., -1] *= 0.5
        # Truncate to N and ensure endpoint convention for N
        N = self.N
        a_N = a_M[..., :N].copy()
        if N >= 1:
            a_N[..., 0] *= 1.0  # already halved from a_M
            a_N[..., -1] *= 0.5  # enforce halving at index N-1 for N-length
        return self._inverse_from_coeffs(a_N)


class ChebyshevBasis2D:
    """Tensor-product Chebyshev–Gauss–Lobatto basis on [0,Lx]×[0,Ly].

    Layout matches Grid2D convention: last two axes are (..., Ny, Nx).
    Provides forward/inverse tensor DCT-I, per-axis derivatives, and filtering.
    """

    def __init__(self, Nx: int, Ny: int, Lx: float, Ly: float, *, bc: str = "dirichlet") -> None:
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.bc = str(bc).lower()

        yx = _cheb_nodes_gl(self.Nx)
        yy = _cheb_nodes_gl(self.Ny)
        self._x = (1.0 - yx) * (self.Lx / 2.0)
        self._y = (1.0 - yy) * (self.Ly / 2.0)
        self._dy_dx = 2.0 / self.Lx
        self._dy_dy = 2.0 / self.Ly

        if _FFT_CACHE and fftw_cache is not None:
            try:
                fftw_cache.enable()
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        return self._x, self._y

    def forward(self, f: np.ndarray) -> np.ndarray:
        Ny, Nx = self.Ny, self.Nx
        F = _dct1_none(f, axis=-1)
        F = F / max(Nx - 1, 1)
        if Nx > 0:
            F[..., 0] *= 0.5
            F[..., -1] *= 0.5
        F = _dct1_none(F, axis=-2)
        F = F / max(Ny - 1, 1)
        if Ny > 0:
            F[..., 0, :] *= 0.5
            F[..., -1, :] *= 0.5
        return F

    def inverse(self, A: np.ndarray) -> np.ndarray:
        Ny, Nx = self.Ny, self.Nx
        B = A.copy()
        if Nx > 2:
            B[..., 1:-1] *= 2.0  # x interior modes per row
        if Ny > 2:
            B[..., 1:-1, :] *= 2.0  # y interior modes per column
        f = _idct1_none(B, axis=-2) / max(Ny - 1, 1)
        f = _idct1_none(f, axis=-1) / max(Nx - 1, 1)
        return f

    @staticmethod
    def _cheb_dx_1d(f: np.ndarray, dy_dx: float, axis: int) -> np.ndarray:
        f_move = np.moveaxis(f, axis, -1)
        N = f_move.shape[-1]
        if N <= 1:
            out = np.zeros_like(f_move)
            return np.moveaxis(out, -1, axis)
        a = _dct1_none(f_move, axis=-1) / (N - 1)
        a[..., 0] *= 0.5
        a[..., -1] *= 0.5
        b = np.zeros_like(a)
        if N >= 2:
            b[..., -2] = 2.0 * (N - 1) * a[..., -1]
            for k in range(N - 3, -1, -1):
                b[..., k] = b[..., k + 2] + 2.0 * (k + 1) * a[..., k + 1]
            b[..., 0] *= 0.5
        b_sum = b.copy()
        if N > 2:
            b_sum[..., 1:-1] *= 2.0
        df_dy = _idct1_none(b_sum, axis=-1) / (N - 1)
        df_dx = df_dy * dy_dx
        return np.moveaxis(df_dx, -1, axis)

    def dx(self, f: np.ndarray) -> np.ndarray:
        return self._cheb_dx_1d(f, self._dy_dx, axis=-1)

    def dy(self, f: np.ndarray) -> np.ndarray:
        return self._cheb_dx_1d(f, self._dy_dy, axis=-2)

    @staticmethod
    def _filter_axis(f: np.ndarray, p: int, alpha: float, axis: int) -> np.ndarray:
        f_move = np.moveaxis(f, axis, -1)
        N = f_move.shape[-1]
        if N <= 1:
            return f
        a = _dct1_none(f_move, axis=-1) / (N - 1)
        a[..., 0] *= 0.5
        a[..., -1] *= 0.5
        k = np.arange(N, dtype=float)
        kmax = max(float(N - 1), 1.0)
        sigma = np.exp(-float(alpha) * (k / kmax) ** int(p))
        a *= sigma.reshape((1,) * (a.ndim - 1) + (N,))
        a_sum = a.copy()
        if N > 2:
            a_sum[..., 1:-1] *= 2.0
        f_f = _idct1_none(a_sum, axis=-1) / (N - 1)
        return np.moveaxis(f_f, -1, axis)

    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        g = self._filter_axis(f, p=p, alpha=alpha, axis=-1)
        g = self._filter_axis(g, p=p, alpha=alpha, axis=-2)
        return g


class ChebyshevBasis3D:
    """Tensor-product Chebyshev basis on [0,Lx]×[0,Ly]×[0,Lz].

    Layout matches Grid3D convention: last three axes are (..., Nz, Ny, Nx).
    """

    def __init__(self, Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float, *, bc: str = "dirichlet") -> None:
        self.Nx = int(Nx)
        self.Ny = int(Ny)
        self.Nz = int(Nz)
        self.Lx = float(Lx)
        self.Ly = float(Ly)
        self.Lz = float(Lz)
        self.bc = str(bc).lower()

        yx = _cheb_nodes_gl(self.Nx)
        yy = _cheb_nodes_gl(self.Ny)
        yz = _cheb_nodes_gl(self.Nz)
        self._x = (1.0 - yx) * (self.Lx / 2.0)
        self._y = (1.0 - yy) * (self.Ly / 2.0)
        self._z = (1.0 - yz) * (self.Lz / 2.0)
        self._dy_dx = 2.0 / self.Lx
        self._dy_dy = 2.0 / self.Ly
        self._dy_dz = 2.0 / self.Lz

        if _FFT_CACHE and fftw_cache is not None:
            try:
                fftw_cache.enable()
                fftw_cache.set_keepalive_time(60.0)
            except Exception:
                pass

    def nodes(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._x, self._y, self._z

    def forward(self, f: np.ndarray) -> np.ndarray:
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        F = _dct1_none(f, axis=-1)
        F = F / max(Nx - 1, 1)
        if Nx > 0:
            F[..., :, :, 0] *= 0.5
            F[..., :, :, -1] *= 0.5
        F = _dct1_none(F, axis=-2)
        F = F / max(Ny - 1, 1)
        if Ny > 0:
            F[..., :, 0, :] *= 0.5
            F[..., :, -1, :] *= 0.5
        F = _dct1_none(F, axis=-3)
        F = F / max(Nz - 1, 1)
        if Nz > 0:
            F[..., 0, :, :] *= 0.5
            F[..., -1, :, :] *= 0.5
        return F

    def inverse(self, A: np.ndarray) -> np.ndarray:
        Nz, Ny, Nx = self.Nz, self.Ny, self.Nx
        B = A.copy()
        if Nx > 2:
            B[..., :, :, 1:-1] *= 2.0
        if Ny > 2:
            B[..., :, 1:-1, :] *= 2.0
        if Nz > 2:
            B[..., 1:-1, :, :] *= 2.0
        f = _idct1_none(B, axis=-3) / max(Nz - 1, 1)
        f = _idct1_none(f, axis=-2) / max(Ny - 1, 1)
        f = _idct1_none(f, axis=-1) / max(Nx - 1, 1)
        return f

    def dx(self, f: np.ndarray) -> np.ndarray:
        return ChebyshevBasis2D._cheb_dx_1d(f, self._dy_dx, axis=-1)

    def dy(self, f: np.ndarray) -> np.ndarray:
        return ChebyshevBasis2D._cheb_dx_1d(f, self._dy_dy, axis=-2)

    def dz(self, f: np.ndarray) -> np.ndarray:
        return ChebyshevBasis2D._cheb_dx_1d(f, self._dy_dz, axis=-3)

    def apply_spectral_filter(self, f: np.ndarray, *, p: int = 8, alpha: float = 36.0) -> np.ndarray:
        g = ChebyshevBasis2D._filter_axis(f, p=p, alpha=alpha, axis=-1)
        g = ChebyshevBasis2D._filter_axis(g, p=p, alpha=alpha, axis=-2)
        g = ChebyshevBasis2D._filter_axis(g, p=p, alpha=alpha, axis=-3)
        return g