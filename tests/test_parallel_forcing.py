"""Band forcing: device synthesis of the compressible acceleration, spherical truncation.

Serially and, unchanged, under ``mpirun -n P``: every rank compares its own slab
with a reference built on the host, and global errors are reduced across ranks.
"""

import numpy as np
import pytest

from _parallel_util import COMM, fits, gmax, torch
from phrike.parallel.forcing import BandForcing, spherical_kcut2
from phrike.parallel.grid import SlabGrid3D
from phrike.parallel.incompressible import IncompressibleSlabStepper

TWO_PI = 2.0 * np.pi


def _host_acceleration(forcing, band):
    """The original synthesis: per-rank separable Fourier sums on the host (the reference)."""
    g = forcing.grid
    m = int(np.floor(forcing.kmax))
    cube = np.zeros((3, 2 * m + 1, 2 * m + 1, m + 1), dtype=np.complex128)
    nz, ny, nx = forcing.n[:, 0], forcing.n[:, 1], forcing.n[:, 2]
    scale = np.where(nx > 0, 2.0, 1.0)
    for c in range(3):
        cube[c, nz + m, ny + m, nx] = band[:, c] * scale
        plane = nx == 0
        cube[c, -nz[plane] + m, -ny[plane] + m, 0] = np.conj(band[plane, c])
    modes = np.arange(-m, m + 1)
    ex = np.exp(1j * np.outer(np.arange(m + 1) * (2.0 * np.pi / g.Lx), g._x))
    ey = np.exp(1j * np.outer(modes * (2.0 * np.pi / g.Ly), g._y))
    ez = np.exp(1j * np.outer(modes * (2.0 * np.pi / g.Lz), g._z))
    planes = np.einsum("cijk,kx->cijx", cube, ex)
    planes = np.einsum("cijx,jy->ciyx", planes, ey)
    return np.einsum("ciyx,iz->czyx", planes.real, ez.real) - np.einsum("ciyx,iz->czyx", planes.imag, ez.imag)


@pytest.mark.parametrize("zeta", [0.0, 0.5])
@pytest.mark.parametrize("lengths", [(TWO_PI, TWO_PI, TWO_PI), (1.0, 2.0, 1.5)])
def test_acceleration_equals_the_host_formula(zeta, lengths):
    shape = (24, 20, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid = SlabGrid3D(*shape, *lengths, comm=COMM, device="cpu", precision="double")
    forcing = BandForcing(grid, {"type": "ou", "kmin": 1.0, "kmax": 2.9, "amplitude": 0.8,
                                 "correlation_time": 0.3, "seed": 4, "compressive_fraction": zeta},
                          incompressible=False)
    forcing.advance(0.05)  # a generic state of the process
    for band in (None, forcing._sample() * 0.37):  # the OU state, and an explicit (scaled) band
        reference = _host_acceleration(forcing, forcing.f if band is None else band)
        field = forcing.acceleration(band)
        assert tuple(field.shape) == (3,) + grid.decomp.physical_shape
        scale = gmax(np.abs(reference).max())
        assert scale > 0.0
        assert gmax(np.abs(field.numpy() - reference).max()) < 1e-12 * scale


def test_acceleration_is_collective_for_ranks_without_band_modes():
    # kmax = 1.5 puts every mode on |ny| <= 1: ranks whose ky rows are all beyond that own none.
    shape = (32, 32, 32)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid = SlabGrid3D(*shape, TWO_PI, TWO_PI, TWO_PI, comm=COMM, device="cpu", precision="double")
    forcing = BandForcing(grid, {"type": "ou", "kmax": 1.5, "seed": 2}, incompressible=False)
    reference = _host_acceleration(forcing, forcing.f)
    assert gmax(np.abs(forcing.acceleration().numpy() - reference).max()) < 1e-12 * gmax(np.abs(reference).max())


def _expected_modes(kmin, kmax, lengths, kcut2):
    m = int(np.floor(kmax))
    out = set()
    for nz in range(-m, m + 1):
        for ny in range(-m, m + 1):
            for nx in range(0, m + 1):
                mag = np.sqrt(nz * nz + ny * ny + nx * nx)
                if not (kmin <= mag <= kmax) or not (nx > 0 or ny > 0 or (ny == 0 and nz > 0)):
                    continue
                k2 = (TWO_PI * nx / lengths[0]) ** 2 + (TWO_PI * ny / lengths[1]) ** 2 \
                    + (TWO_PI * nz / lengths[2]) ** 2
                if k2 <= kcut2:
                    out.add((nz, ny, nx))
    return out


LENGTHS = (TWO_PI, TWO_PI, np.pi)  # kz = 2 nz: the integer-mode box reaches beyond the ball


def _anisotropic_grid(comm=None):
    return SlabGrid3D(24, 24, 24, *LENGTHS, comm=comm or COMM, device="cpu", precision="double")


def test_forced_band_is_restricted_to_the_truncation_ball():
    grid = _anisotropic_grid()
    if not fits((24, 24, 24)):
        pytest.skip("grid too small for this rank count")
    cfg = {"type": "ou", "kmin": 3.5, "kmax": 4.5, "amplitude": 1.0, "seed": 1}
    kcut2 = spherical_kcut2(grid)
    full = BandForcing(grid, cfg, incompressible=True)
    ball = BandForcing(grid, cfg, incompressible=True, spherical_truncation=True)
    assert full.n_truncated == 0 and ball.n_truncated == full.M - ball.M > 0
    assert ball.k2.max() <= kcut2 and full.k2.max() > kcut2
    assert {tuple(int(v) for v in row) for row in ball.n} == _expected_modes(3.5, 4.5, LENGTHS, kcut2)
    # The kept band is renormalized: the rms acceleration is the requested one.
    mean_square = np.mean([2.0 * (np.abs(ball._sample()) ** 2).sum() for _ in range(300)])
    assert mean_square == pytest.approx(1.0, rel=0.1)
    assert "dropped" in ball.describe()


def test_empty_band_inside_the_truncation_is_an_error():
    # |n| = sqrt(3) modes all have nz = +-1, and kz = 16 lies far beyond kcut = 7.
    grid = SlabGrid3D(24, 24, 24, TWO_PI, TWO_PI, TWO_PI / 16.0, comm=COMM, device="cpu")
    cfg = {"type": "ou", "kmin": 1.6, "kmax": 1.8}
    assert BandForcing(grid, cfg, incompressible=True).M > 0
    with pytest.raises(ValueError, match="no modes inside the spherical truncation"):
        BandForcing(grid, cfg, incompressible=True, spherical_truncation=True)


def test_stepper_needs_a_forcing_built_for_its_truncation():
    grid = _anisotropic_grid()
    forcing = BandForcing(grid, {"type": "ou", "kmin": 3.5, "kmax": 4.5}, incompressible=True)
    with pytest.raises(ValueError, match="spherical_truncation"):
        IncompressibleSlabStepper(grid, 0.0, mhd=False, forcing=forcing, spherical_truncation=True)
    BandForcing(grid, {"type": "ou", "kmin": 3.5, "kmax": 4.5}, True, spherical_truncation=True)


@pytest.mark.parametrize("kind", ["constant_power", "ou", "white"])
def test_forcing_in_an_anisotropic_truncated_box_drives_only_coupled_modes(kind):
    shape = (24, 24, 24)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    power = 0.25
    grid = _anisotropic_grid()
    cfg = {"type": kind, "kmin": 3.5, "kmax": 4.5, "power": power, "amplitude": 1.0,
           "correlation_time": 0.2, "seed": 7}
    forcing = BandForcing(grid, cfg, incompressible=True, spherical_truncation=True)
    st = IncompressibleSlabStepper(grid, 0.0, mhd=False, scheme="rk4", forcing=forcing,
                                   spherical_truncation=True)
    V = np.random.default_rng(1).standard_normal((3, 24, 24, 24))
    d = grid.decomp
    st.seed(torch.as_tensor(V[:, d.z0:d.z0 + d.nz]))
    e0, dt, n = st.diagnostics()["total_energy"], 2e-4, 50
    for _ in range(n):
        st.step(dt=dt)
    outside = (grid.k2() > st.kcut2).expand_as(st.uhat.real)
    assert gmax(st.uhat.abs()[outside].max().item() if outside.any() else 0.0) == 0.0
    if kind == "constant_power":  # the whole power reaches the kept modes
        diag = st.diagnostics()
        assert diag["total_energy"] - e0 == pytest.approx(power * n * dt, rel=2e-3)
        assert diag["forcing_power"] == pytest.approx(power, rel=1e-2)
