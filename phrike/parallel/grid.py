"""Periodic 3D Fourier grid on a slab decomposition (this rank's part of it).

``SlabGrid3D`` plays the role of ``phrike.grid.Grid3D`` for the distributed
solvers.  It carries the global sizes under the usual names (``Nx``, ``Lx``,
``dx``, ...), this rank's coordinates, the wavenumbers of the retained modes
it owns and the distributed transform.  Dealiasing by the 2/3 rule is built
in: spectral arrays only ever hold retained modes.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from ..grid import _build_exponential_filter_rate, _spectral_dissipation_settings, _torch_dtypes
from .comm import Comm, Timers
from .decomp import SlabDecomposition
from .fft import SlabFFT


def _full_index(modes: np.ndarray, n: int) -> np.ndarray:
    """Index of integer wavenumbers in a length-``n`` FFT-ordered axis."""
    return np.where(modes >= 0, modes, modes + n)


class SlabGrid3D:
    """This rank's slab of a uniform periodic ``(Nz, Ny, Nx)`` grid."""

    def __init__(self, Nx: int, Ny: int, Nz: int, Lx: float, Ly: float, Lz: float,
                 comm: Optional[Comm] = None, device: str = "cpu", precision: str = "double",
                 filter_params: Optional[Dict[str, Any]] = None, fft_batch: Optional[int] = 4,
                 timers: Optional[Timers] = None):
        self.comm = comm if comm is not None else Comm.serial()
        self.Nx, self.Ny, self.Nz = int(Nx), int(Ny), int(Nz)
        self.Lx, self.Ly, self.Lz = float(Lx), float(Ly), float(Lz)
        self.dx, self.dy, self.dz = self.Lx / self.Nx, self.Ly / self.Ny, self.Lz / self.Nz
        self.dealias = True
        self.filter_params = filter_params
        self.device = torch.device(device)
        self.torch_device = str(device)
        self.backend = "torch"
        self._use_torch = True
        self.rdtype, self.cdtype, self.precision = _torch_dtypes(precision, self.device.type)
        self.timers = timers if timers is not None else Timers(enabled=False)
        self.comm.timers = self.timers

        d = self.decomp = SlabDecomposition(self.Nx, self.Ny, self.Nz, self.comm.size, self.comm.rank)
        self.fft = SlabFFT(d, self.comm, self.device, self.rdtype, batch=fft_batch, timers=self.timers)
        self.n_points = d.n_points
        self.inv_n = 1.0 / float(d.n_points)

        # Coordinates (float64 on the host; cast when a mesh is requested).
        self._x = np.arange(self.Nx) * self.dx
        self._y = np.arange(self.Ny) * self.dy
        self._z = (d.z0 + np.arange(d.nz)) * self.dz

        # Integer modes and physical wavenumbers of the retained modes.
        self.nx_modes = d.modes_x()
        self.ny_modes = d.modes_y_local()
        self.nz_modes = d.modes_z()
        to = self._to_device
        self.kx = to(2.0 * np.pi * self.nx_modes / self.Lx)
        self.ky = to(2.0 * np.pi * self.ny_modes / self.Ly)
        self.kz = to(2.0 * np.pi * self.nz_modes / self.Lz)
        self.kx3 = self.kx.reshape(1, 1, d.Kx)
        self.ky3 = self.ky.reshape(1, d.nky, 1)
        self.kz3 = self.kz.reshape(d.Kz, 1, 1)
        # Largest squared wavenumber that is actually evolved.
        self.k2_max = float((2.0 * np.pi * d.cx / self.Lx) ** 2 + (2.0 * np.pi * d.cy / self.Ly) ** 2
                            + (2.0 * np.pi * d.cz / self.Lz) ** 2)
        # Parseval weights of the half spectrum: kx > 0 stands for +-kx.
        w = np.full(d.Kx, 2.0)
        w[0] = 1.0
        self.weight3 = to(w).reshape(1, 1, d.Kx)

        # Timestep-aware Fourier dissipation, as Grid3D: lambda is a sum of
        # per-axis ramps, so exp(-h*lambda) is a product of 1D factors.
        enabled, p, rate, onset = _spectral_dissipation_settings(filter_params)
        self.spectral_dissipation_enabled = bool(enabled)
        if enabled:
            lx = _build_exponential_filter_rate(self.Nx, p, rate, True, onset)[self.nx_modes]
            ly = _build_exponential_filter_rate(self.Ny, p, rate, True, onset)[
                _full_index(self.ny_modes, self.Ny)]
            lz = _build_exponential_filter_rate(self.Nz, p, rate, True, onset)[
                _full_index(self.nz_modes, self.Nz)]
        else:
            lx, ly, lz = np.zeros(d.Kx), np.zeros(d.nky), np.zeros(d.Kz)
        self.lam1d = (to(lx), to(ly), to(lz))
        self.ones1d = tuple(torch.ones_like(a) for a in self.lam1d)

    # --- helpers ------------------------------------------------------------

    def _to_device(self, a) -> torch.Tensor:
        return torch.as_tensor(np.ascontiguousarray(a), dtype=self.rdtype, device=self.device)

    @property
    def min_spacing(self) -> float:
        return min(self.dx, self.dy, self.dz)

    @property
    def x(self) -> torch.Tensor:
        return self._to_device(self._x)

    @property
    def y(self) -> torch.Tensor:
        return self._to_device(self._y)

    @property
    def z(self) -> torch.Tensor:
        """z coordinates of this rank's planes."""
        return self._to_device(self._z)

    def xyz_mesh(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Coordinates ``(X, Y, Z)`` of this rank's slab, each ``(nz, Ny, Nx)``."""
        Z, Y, X = np.meshgrid(self._z, self._y, self._x, indexing="ij")
        return self._to_device(X), self._to_device(Y), self._to_device(Z)

    def k2(self) -> torch.Tensor:
        """Squared wavenumber of this rank's retained modes, ``(Kz, nky, Kx)``."""
        return self.kx3 * self.kx3 + self.ky3 * self.ky3 + self.kz3 * self.kz3

    def decay_factors(self, h: float):
        """Per-axis factors of ``exp(-h * lambda)`` (ones when dissipation is off)."""
        if not self.spectral_dissipation_enabled:
            return self.ones1d
        return tuple(torch.exp(-float(h) * lam) for lam in self.lam1d)

    def product3(self, factors) -> torch.Tensor:
        fx, fy, fz = factors
        return fz[:, None, None] * fy[None, :, None] * fx[None, None, :]

    # --- transforms ---------------------------------------------------------

    def forward(self, f: torch.Tensor, normalized: bool = True, out=None) -> torch.Tensor:
        return self.fft.forward(f, normalized=normalized, out=out)

    def inverse(self, S: torch.Tensor, out=None) -> torch.Tensor:
        return self.fft.inverse(S, out=out)

    def zeros_spectral(self, ncomp: int) -> torch.Tensor:
        return torch.zeros((ncomp,) + self.decomp.spectral_shape, dtype=self.cdtype, device=self.device)

    def zeros_physical(self, ncomp: int) -> torch.Tensor:
        return torch.zeros((ncomp,) + self.decomp.physical_shape, dtype=self.rdtype, device=self.device)

    # --- global reductions --------------------------------------------------

    def spectral_sum(self, density: torch.Tensor) -> float:
        """Sum over all modes of a real per-mode quantity (half-spectrum weighted)."""
        return self.comm.sum(float((density * self.weight3).sum().item()))

    def describe(self) -> str:
        return f"{self.decomp.describe()}; {self.precision} precision on {self.device}"
