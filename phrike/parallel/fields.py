"""Random band-limited vector fields that do not depend on the decomposition.

Each Fourier coefficient is a counter-based random number: a hash of
``(seed, component, nz, ny, nx)``.  A rank can therefore generate exactly the
modes it owns, and the field is the same for any number of ranks and for any
resolution that retains the band (a low-resolution test run starts from the
same large-scale field as the production run).

Band limits are integer mode magnitudes (cycles per box), as in
``problems/problem_list/turb3d.py``.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch

from .grid import SlabGrid3D

_MASK64 = np.uint64(0xFFFFFFFFFFFFFFFF)
_OFFSET = 1 << 20  # modes are stored as n + 2**20 in 21-bit fields


def _splitmix64(x: np.ndarray) -> np.ndarray:
    """SplitMix64 finalizer on uint64 arrays (wrapping arithmetic)."""
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def _uniform(mode_id: np.ndarray, stream: int) -> np.ndarray:
    """Uniform numbers in (0, 1) keyed by ``(mode_id, stream)``."""
    key = _splitmix64(np.asarray([stream], dtype=np.uint64))[0]
    bits = _splitmix64(mode_id ^ key)
    return ((bits >> np.uint64(11)).astype(np.float64) + 0.5) * (1.0 / 9007199254740992.0)


def gaussian_modes(seed: int, component: int, nz, ny, nx) -> np.ndarray:
    """Unit complex Gaussians ``g(n)`` with ``g(-n) = conj(g(n))`` on the ``nx = 0`` plane.

    ``nz, ny, nx`` are integer mode arrays (``nx >= 0``).  ``E|g|^2 = 1``; the
    ``n = 0`` mode is zero.
    """
    nz, ny, nx = (np.asarray(a, dtype=np.int64) for a in np.broadcast_arrays(nz, ny, nx))
    if np.any(nx < 0) or max(np.abs(nz).max(initial=0), np.abs(ny).max(initial=0),
                             nx.max(initial=0)) >= _OFFSET:
        raise ValueError("mode numbers out of range for the random-field hash")
    # On the nx = 0 plane the half with (ny < 0) or (ny == 0, nz < 0) mirrors the other.
    mirrored = (nx == 0) & ((ny < 0) | ((ny == 0) & (nz < 0)))
    mz = np.where(mirrored, -nz, nz)
    my = np.where(mirrored, -ny, ny)
    mode_id = ((mz + _OFFSET).astype(np.uint64) << np.uint64(42)) \
        | ((my + _OFFSET).astype(np.uint64) << np.uint64(21)) | nx.astype(np.uint64)
    stream = (int(seed) & 0xFFFFFFFF) * 16 + int(component) * 2
    u1 = _uniform(mode_id, stream)
    u2 = _uniform(mode_id, stream + 1)
    radius = np.sqrt(-np.log(u1))  # |g|^2 is exponential with mean 1
    g = radius * np.exp(2j * np.pi * u2)
    g = np.where(mirrored, np.conj(g), g)
    return np.where((nx == 0) & (ny == 0) & (nz == 0), 0.0, g)


def _envelope(mag: np.ndarray, kmin: float, kmax: float, spectrum: str, slope: float) -> np.ndarray:
    """Amplitude filter of ``turbulent_velocity_3d`` (square root of the power envelope)."""
    band = (mag >= kmin) & (mag <= kmax)
    if spectrum == "parabolic":
        power = np.where(band, np.clip((mag - kmin) * (kmax - mag), 0.0, None), 0.0)
    elif spectrum == "power_law":
        # Shell energy E(k) ~ k^s: per-mode power k^(s - 2), with cosine tapers at the band edges.
        safe = np.where(mag == 0.0, 1.0, mag)
        power = np.where(band, safe ** (slope - 2.0), 0.0)
        dk = kmax - kmin
        k1, k2 = kmin + 0.1 * dk, kmax - 0.1 * dk
        if dk > 0.0:
            low = np.where((mag >= kmin) & (mag < k1),
                           0.5 * (1.0 + np.cos(np.pi * (mag - kmin) / (k1 - kmin))), 1.0)
            high = np.where((mag > k2) & (mag <= kmax),
                            0.5 * (1.0 + np.cos(np.pi * (mag - k2) / (kmax - k2))), 1.0)
            power = power * low * high
    elif spectrum == "flat":
        power = band.astype(float)
    else:
        raise ValueError(f"Unknown spectrum_type: {spectrum}")
    return np.sqrt(power)


def random_vector_spectrum(grid: SlabGrid3D, seed: int, kmin: float, kmax: float,
                           rms: float, spectrum: str = "parabolic", slope: float = -5.0 / 3.0,
                           compressive_fraction: float = 0.0) -> torch.Tensor:
    """Retained spectrum ``(3, Kz, nky, Kx)`` (divided by N) of a random vector field.

    The field has zero mean and ``sqrt(<|v|^2>) = rms``.  ``compressive_fraction``
    mixes the compressive (``a``) and solenoidal (``1 - a``) projections; 0 gives
    a divergence-free field.
    """
    d = grid.decomp
    out = np.zeros((3,) + d.spectral_shape, dtype=np.complex128)
    m = int(min(np.floor(kmax), max(d.cx, d.cy, d.cz)))
    nz_all, ny_loc, nx_all = d.modes_z(), d.modes_y_local(), d.modes_x()
    iz = np.nonzero(np.abs(nz_all) <= m)[0]
    iy = np.nonzero(np.abs(ny_loc) <= m)[0]
    ix = np.nonzero(nx_all <= m)[0]
    local_power = 0.0
    if iz.size and iy.size and ix.size:
        NZ, NY, NX = np.meshgrid(nz_all[iz], ny_loc[iy], nx_all[ix], indexing="ij")
        mag = np.sqrt((NZ * NZ + NY * NY + NX * NX).astype(np.float64))
        filt = _envelope(mag, float(kmin), float(kmax), spectrum, float(slope))
        xi = np.stack([gaussian_modes(seed, c, NZ, NY, NX) * filt for c in range(3)])
        k = np.stack([2.0 * np.pi * NX / grid.Lx, 2.0 * np.pi * NY / grid.Ly, 2.0 * np.pi * NZ / grid.Lz])
        k2 = (k * k).sum(axis=0)
        along = k * ((k * xi).sum(axis=0) / np.where(k2 == 0.0, 1.0, k2))
        a = float(compressive_fraction)
        mixed = a * along + (1.0 - a) * (xi - along)
        weight = np.where(NX == 0, 1.0, 2.0)
        local_power = float((weight * (np.abs(mixed) ** 2).sum(axis=0)).sum())
        out[np.ix_(range(3), iz, iy, ix)] = mixed
    power = grid.comm.sum(local_power)  # <|v|^2> of the unscaled field
    if power > 0.0:
        out *= float(rms) / np.sqrt(power)
    return torch.as_tensor(out, dtype=grid.cdtype, device=grid.device)


def band_summary(grid: SlabGrid3D, kmin: float, kmax: float) -> Tuple[int, int]:
    """(largest retained integer mode on the tightest axis, requested upper band edge)."""
    d = grid.decomp
    return min(d.cx, d.cy, d.cz), int(np.floor(kmax))
