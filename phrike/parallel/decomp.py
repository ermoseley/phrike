"""Slab decomposition of a periodic 3D Fourier grid and its retained modes.

Physical space is cut into z slabs: rank ``r`` owns ``nz`` consecutive planes
of the ``(Nz, Ny, Nx)`` grid.  Spectral space holds only the modes kept by the
2/3 rule, ``|n_i| <= (N_i - 1) // 3`` on every axis, in ``rfftn`` order: the
compact array is ``(Kz, Ky, Kx)`` with ``Kx = cx + 1`` and ``K = 2 c + 1`` on
the other two axes, and it is cut into ky slabs.  Each rank therefore owns
complete z columns in spectral space and complete (y, x) planes in physical
space, which is what one transpose per 3D transform needs.

Both cuts are balanced (sizes differ by at most one), so every pack and unpack
is a constant number of strided copies however many ranks there are.

A decomposition can also keep every mode (``full=True``): the spectral array is
then the whole ``rfftn`` half spectrum, ``Kx = Nx // 2 + 1`` and ``Ky = Ny``,
``Kz = Nz``.  Each axis is described by a head and a tail count, the number of
leading and trailing rows of an FFT-ordered axis that are kept (the retained
box has ``c + 1`` and ``c``; the full one ``N // 2 + 1`` and ``N - N // 2 - 1``).
The full decomposition is only used for the rare out-of-band content that the
accepted-state limiter creates (see ``compressible.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


def balanced_counts(n: int, parts: int) -> List[int]:
    """Split ``n`` items into ``parts`` consecutive blocks whose sizes differ by at most one."""
    if parts < 1:
        raise ValueError("parts must be positive")
    base, extra = divmod(int(n), int(parts))
    return [base + 1] * extra + [base] * (parts - extra)


def uniform_groups(counts: List[int]) -> List[Tuple[int, int, int, int]]:
    """Runs of equal block size as ``(first_block, n_blocks, size, first_item)``."""
    groups: List[Tuple[int, int, int, int]] = []
    item = 0
    i = 0
    while i < len(counts):
        j = i
        while j < len(counts) and counts[j] == counts[i]:
            j += 1
        if counts[i] > 0:
            groups.append((i, j - i, counts[i], item))
        item += counts[i] * (j - i)
        i = j
    return groups


def cutoff(n: int) -> int:
    """Largest retained integer mode of the 2/3 rule, as ``grid._build_filter_mask``."""
    return (int(n) - 1) // 3


def compact_modes(n: int) -> np.ndarray:
    """Integer wavenumbers of the retained modes of a full axis, in FFT order."""
    c = cutoff(n)
    return np.concatenate((np.arange(0, c + 1), np.arange(-c, 0))).astype(np.int64)


def axis_modes(head: int, tail: int) -> np.ndarray:
    """Integer wavenumbers of an FFT-ordered axis that keeps ``head`` leading and ``tail`` trailing rows.

    The Nyquist mode of an even axis is labelled ``+N/2``.
    """
    return np.concatenate((np.arange(0, head), np.arange(-tail, 0))).astype(np.int64)


@dataclass
class SlabDecomposition:
    """Global sizes, kept-mode sizes and this rank's two slabs."""

    Nx: int
    Ny: int
    Nz: int
    nprocs: int = 1
    rank: int = 0
    full: bool = False
    z_counts: List[int] = field(init=False)
    ky_counts: List[int] = field(init=False)

    def __post_init__(self) -> None:
        self.Nx, self.Ny, self.Nz = int(self.Nx), int(self.Ny), int(self.Nz)
        self.nprocs, self.rank = int(self.nprocs), int(self.rank)
        if min(self.Nx, self.Ny, self.Nz) < 1:
            raise ValueError("grid sizes must be positive")
        if not 0 <= self.rank < self.nprocs:
            raise ValueError("rank must lie in [0, nprocs)")
        self.full = bool(self.full)
        # The 2/3-rule cutoff, whether or not the modes above it are kept.
        self.cx, self.cy, self.cz = cutoff(self.Nx), cutoff(self.Ny), cutoff(self.Nz)
        self.Xh = self.Nx // 2 + 1
        if self.full:
            self.hx, self.hy, self.hz = self.Xh, self.Ny // 2 + 1, self.Nz // 2 + 1
            self.ty, self.tz = self.Ny - self.hy, self.Nz - self.hz
        else:
            self.hx, self.hy, self.hz = self.cx + 1, self.cy + 1, self.cz + 1
            self.ty, self.tz = self.cy, self.cz
        self.Kx, self.Ky, self.Kz = self.hx, self.hy + self.ty, self.hz + self.tz
        if self.nprocs > self.Nz or self.nprocs > self.Ky:
            what = ("Ny, the number of y modes" if self.full else
                    "Ky = 2*((Ny - 1)//3) + 1, the number of retained y modes")
            raise ValueError(
                f"A slab decomposition needs at most min(Nz, Ky) = "
                f"{min(self.Nz, self.Ky)} ranks for this grid (got {self.nprocs}); "
                f"{what}."
            )
        self.z_counts = balanced_counts(self.Nz, self.nprocs)
        self.ky_counts = balanced_counts(self.Ky, self.nprocs)
        self.z_starts = [int(v) for v in np.concatenate(([0], np.cumsum(self.z_counts)[:-1]))]
        self.ky_starts = [int(v) for v in np.concatenate(([0], np.cumsum(self.ky_counts)[:-1]))]

    # --- this rank ----------------------------------------------------------

    @property
    def nz(self) -> int:
        """Number of z planes this rank owns in physical space."""
        return self.z_counts[self.rank]

    @property
    def z0(self) -> int:
        return self.z_starts[self.rank]

    @property
    def nky(self) -> int:
        """Number of retained ky rows this rank owns in spectral space."""
        return self.ky_counts[self.rank]

    @property
    def ky0(self) -> int:
        return self.ky_starts[self.rank]

    @property
    def physical_shape(self) -> Tuple[int, int, int]:
        return (self.nz, self.Ny, self.Nx)

    @property
    def spectral_shape(self) -> Tuple[int, int, int]:
        return (self.Kz, self.nky, self.Kx)

    @property
    def n_points(self) -> int:
        return self.Nx * self.Ny * self.Nz

    @property
    def n_modes(self) -> int:
        return self.Kx * self.Ky * self.Kz

    # --- integer wavenumbers ------------------------------------------------

    def modes_x(self) -> np.ndarray:
        return np.arange(self.Kx, dtype=np.int64)

    def modes_y(self) -> np.ndarray:
        """Integer ky of every kept row (global compact order)."""
        return axis_modes(self.hy, self.ty)

    def modes_y_local(self) -> np.ndarray:
        return self.modes_y()[self.ky0:self.ky0 + self.nky]

    def modes_z(self) -> np.ndarray:
        return axis_modes(self.hz, self.tz)

    def describe(self) -> str:
        return (f"{self.Nx}x{self.Ny}x{self.Nz} grid, {self.Kx}x{self.Ky}x{self.Kz} "
                f"{'all' if self.full else 'retained'} modes, "
                f"{self.nprocs} slab(s): {min(self.z_counts)}-{max(self.z_counts)} z planes and "
                f"{min(self.ky_counts)}-{max(self.ky_counts)} ky rows per rank")
