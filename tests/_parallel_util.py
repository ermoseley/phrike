"""Shared helpers of the distributed-solver tests.

The tests run serially (``Comm.serial`` stand-in) and, unchanged, under
``mpirun -n P python -m pytest tests/test_parallel_*.py``: every rank then
runs the same test and compares its own slab against a reference it computes
for the whole grid.  ``tests/test_parallel_mpi.py`` launches those runs.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phrike.parallel import Comm  # noqa: E402
from phrike.parallel.decomp import compact_modes  # noqa: E402

COMM = Comm.auto()
if COMM.size > 1:
    torch.set_num_threads(1)

DISSIPATION = {"enabled": True, "p": 8, "rate": 300.0, "onset_fraction": 0.5}


def fits(shape) -> bool:
    """Whether the slab decomposition allows this grid for the current rank count."""
    Nx, Ny, Nz = shape
    return COMM.size <= min(Nz, 2 * ((Ny - 1) // 3) + 1)


def gmax(value: float) -> float:
    return COMM.max(float(value))


def shared_dir(tmp_path) -> str:
    """A directory every rank agrees on (rank 0's ``tmp_path``)."""
    return COMM.bcast(str(tmp_path) if COMM.is_root else None)


def compact_slice(F: np.ndarray, shape, decomp) -> np.ndarray:
    """Retained modes of a full ``fftn`` array (C, Nz, Ny, Nx) on this rank's ky slab."""
    Nx, Ny, Nz = shape
    mz, my = compact_modes(Nz), compact_modes(Ny)
    iz = np.where(mz >= 0, mz, mz + Nz)
    iy = np.where(my >= 0, my, my + Ny)
    return F[:, iz][:, :, iy][:, :, decomp.ky0:decomp.ky0 + decomp.nky, :decomp.Kx]


def full_wavenumbers(shape, lengths):
    """(KX, KY, KZ, 2/3 mask) of the full complex spectrum, each (Nz, Ny, Nx)."""
    Nx, Ny, Nz = shape
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=lengths[0] / Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=lengths[1] / Ny)
    kz = 2 * np.pi * np.fft.fftfreq(Nz, d=lengths[2] / Nz)
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    keep = [np.abs(np.rint(np.fft.fftfreq(n) * n)) <= (n - 1) // 3 for n in (Nz, Ny, Nx)]
    mask = keep[0][:, None, None] & keep[1][None, :, None] & keep[2][None, None, :]
    return KX, KY, KZ, mask.astype(float)
