"""Slab decomposition and the truncated distributed FFT against NumPy."""

import numpy as np
import pytest

from _parallel_util import COMM, compact_slice, fits, gmax, torch
from phrike.parallel import SlabDecomposition, SlabFFT
from phrike.parallel.decomp import balanced_counts, compact_modes, uniform_groups

SHAPES = [(16, 12, 10), (15, 9, 11), (20, 18, 14), (8, 8, 8), (24, 16, 32)]


def test_balanced_partition_and_groups():
    for n, parts in [(10, 3), (683, 128), (7, 7), (5, 1), (64, 8)]:
        counts = balanced_counts(n, parts)
        assert sum(counts) == n and max(counts) - min(counts) <= 1
        groups = uniform_groups(counts)
        assert len(groups) <= 2
        assert sum(nb * size for _, nb, size, _ in groups) == n
        for first, nb, size, item0 in groups:
            assert counts[first:first + nb] == [size] * nb and item0 == sum(counts[:first])


def test_retained_modes_match_grid_mask():
    from phrike.grid import _build_filter_mask

    for n in (8, 9, 10, 15, 16, 31, 32, 33, 64):
        kept = np.sort(np.rint(np.fft.fftfreq(n) * n)[_build_filter_mask(n, True) > 0])
        assert np.array_equal(np.sort(compact_modes(n)), kept)


def test_too_many_ranks_is_rejected():
    with pytest.raises(ValueError):
        SlabDecomposition(16, 16, 16, nprocs=12, rank=0)  # Ky = 11 retained rows


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 3e-5)])
@pytest.mark.parametrize("batch", [0, 2])
def test_transforms_match_numpy(shape, dtype, tol, batch):
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    Nx, Ny, Nz = shape
    d = SlabDecomposition(Nx, Ny, Nz, COMM.size, COMM.rank)
    fft = SlabFFT(d, COMM, "cpu", dtype, batch=batch)
    f = np.random.default_rng(1).standard_normal((3, Nz, Ny, Nx))
    F = np.fft.rfftn(f, axes=(-3, -2, -1))
    local = torch.as_tensor(f[:, d.z0:d.z0 + d.nz], dtype=dtype)

    spectrum = fft.forward(local)
    reference = compact_slice(F, shape, d)
    assert gmax(np.abs(spectrum.numpy() - reference).max() / np.abs(F).max()) < tol

    # The inverse of the retained modes is NumPy's irfftn of the zero-padded spectrum.
    mz, my = compact_modes(Nz), compact_modes(Ny)
    iz, iy = np.where(mz >= 0, mz, mz + Nz), np.where(my >= 0, my, my + Ny)
    padded = np.zeros_like(F)
    index = np.ix_(range(3), iz, iy, range(d.Kx))
    padded[index] = F[index]
    dealiased = np.fft.irfftn(padded, s=(Nz, Ny, Nx), axes=(-3, -2, -1))
    back = fft.inverse(fft.forward(local, normalized=True))
    assert gmax(np.abs(back.numpy() - dealiased[:, d.z0:d.z0 + d.nz]).max() / np.abs(dealiased).max()) < tol
    assert fft.forward_fields == 6 and fft.inverse_fields == 3


def test_exchange_modes_agree():
    """The point-to-point transpose is the same exchange as Alltoallv."""
    if COMM.size == 1:
        pytest.skip("needs more than one rank")
    from phrike.parallel import Comm

    shape = (16, 12, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    d = SlabDecomposition(*shape, COMM.size, COMM.rank)
    f = np.random.default_rng(2).standard_normal((2, 10, 12, 16))
    local = torch.as_tensor(f[:, d.z0:d.z0 + d.nz])
    results = []
    for mode in ("alltoallv", "p2p"):
        comm = Comm(COMM.mpi, transpose=mode)
        results.append(SlabFFT(d, comm, "cpu", torch.float64).forward(local))
    assert torch.equal(results[0], results[1])


FULL_SHAPES = [(16, 12, 10), (15, 9, 11), (20, 18, 14), (8, 8, 8), (9, 8, 7), (7, 10, 9), (6, 7, 8)]


def fits_full(shape) -> bool:
    Nx, Ny, Nz = shape
    return COMM.size <= min(Nz, Ny)


def test_full_decomposition_keeps_every_mode():
    for shape in [(16, 12, 10), (15, 9, 11), (8, 8, 8), (7, 10, 9)]:
        Nx, Ny, Nz = shape
        d = SlabDecomposition(Nx, Ny, Nz, 1, 0, full=True)
        assert (d.Kx, d.Ky, d.Kz) == (Nx // 2 + 1, Ny, Nz)
        assert (d.hy + d.ty, d.hz + d.tz) == (Ny, Nz)
        for modes, n in ((d.modes_y(), Ny), (d.modes_z(), Nz)):
            assert np.array_equal(np.abs(modes), np.abs(np.rint(np.fft.fftfreq(n) * n)).astype(int))
        # The default decomposition is the retained box, as before.
        r = SlabDecomposition(Nx, Ny, Nz, 1, 0)
        assert not r.full and np.array_equal(r.modes_y(), compact_modes(Ny))
        assert np.array_equal(r.modes_z(), compact_modes(Nz))
        assert (r.hy, r.ty, r.hz, r.tz) == (r.cy + 1, r.cy, r.cz + 1, r.cz)
    with pytest.raises(ValueError):
        SlabDecomposition(16, 12, 16, nprocs=13, rank=0, full=True)  # at most Ny = 12 ranks


@pytest.mark.parametrize("shape", FULL_SHAPES)
@pytest.mark.parametrize("dtype,tol", [(torch.float64, 1e-12), (torch.float32, 3e-5)])
@pytest.mark.parametrize("batch", [0, 2])
def test_full_transforms_match_numpy(shape, dtype, tol, batch):
    """The full decomposition is the plain distributed rfftn / irfftn, odd and even N."""
    if not fits_full(shape):
        pytest.skip("grid too small for this rank count")
    Nx, Ny, Nz = shape
    d = SlabDecomposition(Nx, Ny, Nz, COMM.size, COMM.rank, full=True)
    fft = SlabFFT(d, COMM, "cpu", dtype, batch=batch)
    rng = np.random.default_rng(3)
    f = rng.standard_normal((3, Nz, Ny, Nx))
    F = np.fft.rfftn(f, axes=(-3, -2, -1))
    local = torch.as_tensor(f[:, d.z0:d.z0 + d.nz], dtype=dtype)

    spectrum = fft.forward(local)
    assert tuple(spectrum.shape) == (3, Nz, d.nky, Nx // 2 + 1)
    assert gmax(np.abs(spectrum.numpy() - F[:, :, d.ky0:d.ky0 + d.nky]).max() / np.abs(F).max()) < tol

    back = fft.inverse(fft.forward(local, normalized=True))
    assert gmax(np.abs(back.numpy() - f[:, d.z0:d.z0 + d.nz]).max() / np.abs(f).max()) < tol

    # A Hermitian spectrum that is not a plain forward transform, as a damped one is.
    g = rng.standard_normal((2, Nz, Ny, Nx))
    damp = np.exp(-0.01 * (np.fft.fftfreq(Nz)[:, None, None] ** 2 * Nz ** 2
                           + np.fft.fftfreq(Ny)[None, :, None] ** 2 * Ny ** 2
                           + (np.arange(Nx // 2 + 1) ** 2)[None, None, :]))
    G = np.fft.rfftn(g, axes=(-3, -2, -1)) * damp
    expected = np.fft.irfftn(G, s=(Nz, Ny, Nx), axes=(-3, -2, -1)) * (Nx * Ny * Nz)  # inverse has no 1/N
    S = torch.as_tensor(G[:, :, d.ky0:d.ky0 + d.nky], dtype=fft.cdtype)
    out = fft.inverse(S)
    assert gmax(np.abs(out.numpy() - expected[:, d.z0:d.z0 + d.nz]).max() / np.abs(expected).max()) < tol
