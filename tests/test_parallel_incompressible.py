"""Incompressible visco-resistive MHD on the slab decomposition.

The nonlinear term is checked against an independent formulation (rotational
form with full complex FFTs), the time integration against exact solutions,
and the forcing against its energy budget.  Under MPI each rank checks its own
slab against the same whole-grid references.
"""

import numpy as np
import pytest

from _parallel_util import COMM, compact_slice, fits, full_wavenumbers, gmax, torch
from phrike.parallel import Comm
from phrike.parallel.fields import gaussian_modes, random_vector_spectrum
from phrike.parallel.forcing import BandForcing
from phrike.parallel.grid import SlabGrid3D
from phrike.parallel.incompressible import IncompressibleSlabStepper, lawson_coefficients, LAWSON_TABLES

TWO_PI = 2.0 * np.pi
CUBE = (TWO_PI, TWO_PI, TWO_PI)


def _make(shape, lengths=CUBE, nu=0.0, eta=0.0, mhd=True, scheme="rk4", comm=None, forcing=None, **kwargs):
    Nx, Ny, Nz = shape
    grid = SlabGrid3D(Nx, Ny, Nz, *lengths, comm=comm or COMM, device="cpu", precision="double", fft_batch=3)
    if isinstance(forcing, dict):
        forcing = BandForcing(grid, forcing, incompressible=True,
                              spherical_truncation=kwargs.get("spherical_truncation", False))
    return grid, IncompressibleSlabStepper(grid, nu, eta, mhd=mhd, scheme=scheme, forcing=forcing, **kwargs)


def _local(a, grid):
    d = grid.decomp
    return a[:, d.z0:d.z0 + d.nz]


def _project(V, shape, lengths):
    """Dealiased, solenoidal part of a 3-vector field on the whole grid."""
    KX, KY, KZ, mask = full_wavenumbers(shape, lengths)
    K = np.stack([KX, KY, KZ])
    k2 = (K ** 2).sum(axis=0)
    F = np.fft.fftn(V, axes=(-3, -2, -1)) * mask
    F = F - K * ((K * F).sum(axis=0) / np.where(k2 == 0, 1.0, k2))
    return np.fft.ifftn(F, axes=(-3, -2, -1)).real


def _rotational_rhs(u, b, shape, lengths):
    """du/dt = P[u x w + j x b], db/dt = curl(u x b), with full complex FFTs (divided by N)."""
    KX, KY, KZ, mask = full_wavenumbers(shape, lengths)
    K = np.stack([KX, KY, KZ])
    k2 = (K ** 2).sum(axis=0)
    fft = lambda f: np.fft.fftn(f, axes=(-3, -2, -1))
    ifft = lambda F: np.fft.ifftn(F, axes=(-3, -2, -1)).real
    curl = lambda F: np.stack([1j * (KY * F[2] - KZ * F[1]), 1j * (KZ * F[0] - KX * F[2]),
                               1j * (KX * F[1] - KY * F[0])])
    w, j = ifft(curl(fft(u))), ifft(curl(fft(b)))
    Nu = fft(np.cross(u, w, axis=0) + np.cross(j, b, axis=0)) * mask
    Nu = Nu - K * ((K * Nu).sum(axis=0) / np.where(k2 == 0, 1.0, k2))
    Nb = curl(fft(np.cross(u, b, axis=0)) * mask)
    return np.concatenate([Nu, Nb]) / u[0].size


@pytest.mark.parametrize("shape,lengths", [((16, 12, 10), CUBE), ((15, 18, 14), (1.0, 2.0, 1.5))])
def test_nonlinear_term_matches_rotational_form(shape, lengths):
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape, lengths)
    Nx, Ny, Nz = shape
    V = np.random.default_rng(3).standard_normal((6, Nz, Ny, Nx))
    st.seed(torch.as_tensor(_local(V, grid)))
    u, b = _project(V[:3], shape, lengths), _project(V[3:], shape, lengths)
    assert gmax(np.abs(st.physical().numpy() - _local(np.concatenate([u, b]), grid)).max()) < 1e-12
    reference = compact_slice(_rotational_rhs(u, b, shape, lengths), shape, grid.decomp)
    assert gmax(np.abs(st.rhs().numpy() - reference).max() / np.abs(reference).max()) < 1e-12


def test_hydro_nonlinear_term_matches_rotational_form():
    shape = (12, 16, 10)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape, mhd=False)
    Nx, Ny, Nz = shape
    V = np.random.default_rng(5).standard_normal((3, Nz, Ny, Nx))
    st.seed(torch.as_tensor(_local(V, grid)))
    u = _project(V, shape, CUBE)
    reference = compact_slice(_rotational_rhs(u, np.zeros_like(u), shape, CUBE)[:3], shape, grid.decomp)
    assert gmax(np.abs(st.rhs().numpy() - reference).max() / np.abs(reference).max()) < 1e-12


def _abc(grid, A=1.0, B=0.7, C=0.4):
    X, Y, Z = grid.xyz_mesh()
    return torch.stack([A * torch.sin(Z) + C * torch.cos(Y), B * torch.sin(X) + A * torch.cos(Z),
                        C * torch.sin(Y) + B * torch.cos(X)])


@pytest.mark.parametrize("scheme", ["rk2", "rk3", "rk4"])
def test_beltrami_flow_decays_exactly(scheme):
    """ABC flow with a parallel field: the nonlinear term is a pure gradient,
    so the fields only decay, exactly as exp(-nu t) for nu = eta."""
    shape = (16, 16, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    nu = 0.05
    grid, st = _make(shape, nu=nu, eta=nu, scheme=scheme, cfl=0.4)
    abc = _abc(grid)
    st.seed(torch.cat([abc, 0.5 * abc]))
    for _ in range(12):
        st.step()
    exact = torch.cat([abc, 0.5 * abc]) * np.exp(-nu * st.t)
    assert st.t > 0.3 and gmax((st.physical() - exact).abs().max().item()) < 1e-13


def _alfven_error(nsteps, scheme):
    """Transverse waves along x on a mean field with nu != eta: exact 2x2 linear solution."""
    import scipy.linalg

    shape, nu, eta, B0, m, T = (16, 8, 8), 0.02, 0.07, 1.3, 2, 0.5
    grid, st = _make(shape, nu=nu, eta=eta, scheme=scheme)
    X, _, _ = grid.xyz_mesh()
    k = float(m)
    u0 = torch.stack([torch.zeros_like(X), 0.3 * torch.cos(k * X), 0.1 * torch.sin(k * X)])
    b0 = torch.stack([torch.full_like(X, B0), 0.2 * torch.sin(k * X), -0.25 * torch.cos(k * X)])
    st.seed(torch.cat([u0, b0]))
    for _ in range(nsteps):
        st.step(dt=T / nsteps)
    propagator = scipy.linalg.expm(np.array([[-nu * k * k, 1j * k * B0], [1j * k * B0, -eta * k * k]]) * T)
    x = np.arange(shape[0]) * TWO_PI / shape[0]

    def evolve(u_line, b_line):
        a = propagator @ np.array([np.fft.rfft(u_line)[m], np.fft.rfft(b_line)[m]]) / len(x)
        return [2.0 * np.real(c * np.exp(1j * k * x)) for c in a]

    uy, by = evolve(0.3 * np.cos(k * x), 0.2 * np.sin(k * x))
    uz, bz = evolve(0.1 * np.sin(k * x), -0.25 * np.cos(k * x))
    V = st.physical().numpy()
    errors = [np.abs(V[c] - line[None, None, :]).max() for c, line in ((1, uy), (2, uz), (4, by), (5, bz))]
    errors += [np.abs(V[0]).max(), np.abs(V[3] - B0).max()]
    return gmax(max(errors))


@pytest.mark.parametrize("scheme,order,steps", [("rk4", 4, 20), ("rk3", 3, 40), ("rk2", 2, 40)])
def test_alfven_wave_convergence_order(scheme, order, steps):
    if not fits((16, 8, 8)):
        pytest.skip("grid too small for this rank count")
    coarse, fine = _alfven_error(steps, scheme), _alfven_error(2 * steps, scheme)
    assert abs(np.log2(coarse / fine) - order) < 0.3
    assert fine < {4: 1e-7, 3: 1e-5, 2: 1e-3}[order]


def test_lawson_tables_are_consistent():
    for name, table in LAWSON_TABLES.items():
        assert abs(sum(wa for wa, _ in table) - 1.0) < 1e-15, name
        coefs = lawson_coefficients(table, 2.0)
        assert coefs[-1][2] == 2.0 and all(c[3] >= 0.0 and c[4] >= 0.0 for c in coefs)


@pytest.mark.parametrize("order", [4, 6, 8])
def test_hyperdissipation_decays_a_mode_at_its_own_rate(order):
    """ABC flow at wavenumber 2 with a parallel field: the amplitude follows
    exp(-(nu k^2 + nu_h k^order) t) exactly, with different rates for u and b."""
    shape = (16, 16, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    nu, eta, nu_h, eta_h = 0.01, 0.02, 3e-3, 5e-3
    grid, st = _make(shape, nu=nu, eta=eta, cfl=0.4, hyper_nu=nu_h, hyper_eta=eta_h, hyper_order=order)
    X, Y, Z = grid.xyz_mesh()
    abc = torch.stack([torch.sin(2 * Z) + 0.4 * torch.cos(2 * Y), 0.7 * torch.sin(2 * X) + torch.cos(2 * Z),
                       0.4 * torch.sin(2 * Y) + 0.7 * torch.cos(2 * X)])
    st.seed(torch.cat([abc, 0.5 * abc]))
    # u and b decay at different rates, so b stops being parallel to u: only the
    # linear part is exact.  Check it on a state without the nonlinear coupling.
    for _ in range(8):
        st.step(dt=0.01)
    k2 = 4.0
    rate_u, rate_b = nu * k2 + nu_h * k2 ** (order // 2), eta * k2 + eta_h * k2 ** (order // 2)
    d = st.diagnostics()
    assert d["hyperviscous_dissipation"] == pytest.approx(nu_h * k2 ** (order // 2) * 2.0 * d["kinetic_energy"])
    assert d["hyperresistive_dissipation"] == pytest.approx(eta_h * k2 ** (order // 2) * 2.0 * d["magnetic_energy"])
    assert d["hyper_magnetic_prandtl"] == pytest.approx(nu_h / eta_h)
    # Equal rates keep the Beltrami state exact through the nonlinear term as well.
    grid, st = _make(shape, cfl=0.4, hyper_nu=nu_h, hyper_eta=nu_h, hyper_order=order)
    st.seed(torch.cat([abc, 0.5 * abc]))
    for _ in range(12):
        st.step()
    exact = torch.cat([abc, 0.5 * abc]) * np.exp(-nu_h * k2 ** (order // 2) * st.t)
    assert st.t > 0.1 and gmax((st.physical() - exact).abs().max().item()) < 1e-13
    assert rate_u != rate_b


def test_hyperdissipation_is_validated():
    grid = SlabGrid3D(8, 8, 8, *CUBE, comm=Comm.serial(), device="cpu")
    for bad in (2, 5, 6.5):
        with pytest.raises(ValueError):
            IncompressibleSlabStepper(grid, 0.0, hyper_nu=1e-3, hyper_order=bad)
    with pytest.raises(ValueError):
        IncompressibleSlabStepper(grid, 0.0, hyper_nu=-1e-3)
    # Without a coefficient the order is irrelevant and nothing changes.
    assert IncompressibleSlabStepper(grid, 0.01, 0.01).hyper_p == 0


def _helical_state(grid):
    X, Y, Z = grid.xyz_mesh()
    u = torch.stack([-2 * torch.sin(Y), 2 * torch.sin(X), torch.zeros_like(X)])
    b = 0.8 * torch.stack([-2 * torch.sin(2 * Y) + torch.sin(Z), 2 * torch.sin(X) + torch.sin(Z),
                           torch.sin(X) + torch.sin(Y)]) + _abc(grid)
    return torch.cat([u, b])


def test_ideal_invariants_are_conserved():
    """nu = eta = 0: the truncated system conserves energy, cross helicity and magnetic helicity."""
    shape = (24, 24, 24)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape)
    st.seed(_helical_state(grid))
    d0 = st.diagnostics()
    for _ in range(40):
        st.step(dt=2e-3)
    d1 = st.diagnostics()
    for key in ("total_energy", "cross_helicity", "magnetic_helicity"):
        assert abs(d1[key] - d0[key]) < 1e-11 * abs(d0[key]), key
    assert d1["max_div_b_hat"] < 1e-15 and d1["max_div_u_hat"] < 1e-15
    # The spectral energies are the physical-space means.
    V = st.physical()
    kinetic = COMM.sum(float((V[:3] ** 2).sum())) / grid.n_points / 2.0
    magnetic = COMM.sum(float((V[3:] ** 2).sum())) / grid.n_points / 2.0
    assert kinetic == pytest.approx(d1["kinetic_energy"], rel=1e-12)
    assert magnetic == pytest.approx(d1["magnetic_energy"], rel=1e-12)
    spectra = st.spectra()
    assert spectra["kinetic"].sum() == pytest.approx(d1["kinetic_energy"], rel=1e-12)
    assert spectra["magnetic"].sum() == pytest.approx(d1["magnetic_energy"], rel=1e-12)


def test_dissipation_rates_balance_the_energy_decay():
    shape = (24, 24, 24)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape, nu=0.03, eta=0.05)
    st.seed(_helical_state(grid))
    dt, energies, rates = 1e-3, [], []
    for _ in range(21):
        d = st.diagnostics()
        energies.append(d["total_energy"])
        rates.append(d["viscous_dissipation"] + d["resistive_dissipation"])
        st.step(dt=dt)
    # Simpson's rule for the dissipated energy over 20 steps.
    r = np.asarray(rates)
    dissipated = dt / 3.0 * (r[0] + 4.0 * r[1:-1:2].sum() + 2.0 * r[2:-1:2].sum() + r[-1])
    assert energies[0] - energies[-1] == pytest.approx(dissipated, rel=1e-8)


def test_hyperdissipation_rates_balance_the_energy_decay():
    shape = (24, 24, 24)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape, nu=0.002, eta=0.0, hyper_nu=2e-3, hyper_eta=4e-3, hyper_order=6)
    st.seed(_helical_state(grid))
    dt, energies, rates = 1e-3, [], []
    for _ in range(21):
        d = st.diagnostics()
        energies.append(d["total_energy"])
        rates.append(d["dissipation"])
        assert d["dissipation"] == pytest.approx(
            d["viscous_dissipation"] + d["resistive_dissipation"] + d["hyperviscous_dissipation"]
            + d["hyperresistive_dissipation"])
        st.step(dt=dt)
    assert rates[0] > 2.0 * (d["viscous_dissipation"] + d["resistive_dissipation"])  # the hyper terms matter
    r = np.asarray(rates)
    dissipated = dt / 3.0 * (r[0] + 4.0 * r[1:-1:2].sum() + 2.0 * r[2:-1:2].sum() + r[-1])
    assert energies[0] - energies[-1] == pytest.approx(dissipated, rel=1e-8)


def test_spherical_truncation_keeps_a_ball():
    shape = (24, 24, 24)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape, nu=0.01, eta=0.01, spherical_truncation=True)
    V = np.random.default_rng(1).standard_normal((6, 24, 24, 24))
    st.seed(torch.as_tensor(_local(V, grid)))
    for _ in range(3):
        st.step(dt=1e-3)
    outside = (grid.k2() > st.kcut2).expand_as(st.uhat.real)
    assert gmax(st.uhat.abs()[outside].max().item() if outside.any() else 0.0) == 0.0
    assert COMM.sum(float(st.uhat.abs()[~outside].sum())) > 0.0


def test_cfl_step_and_caps():
    shape = (16, 16, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape, nu=0.01, eta=0.01, cfl=0.5)
    st.seed(_helical_state(grid))
    V = st.physical().abs()
    rate = gmax(((V[0] + V[3]) / grid.dx + (V[1] + V[4]) / grid.dy + (V[2] + V[5]) / grid.dz).max().item())
    dt = st.step()
    assert st.rate == pytest.approx(rate, rel=1e-12) and dt == pytest.approx(0.5 / rate, rel=1e-12)
    assert st.step(dt_max=1e-4) == 1e-4
    quiet_grid, quiet = _make(shape, nu=0.01)
    with pytest.raises(RuntimeError):
        quiet.step()  # nothing bounds the step of a fluid at rest
    assert quiet.step(dt_max=0.1) == 0.1


def test_non_finite_state_is_reported():
    shape = (16, 16, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape)
    st.seed(_helical_state(grid))
    st.uhat[0, 0, 0, 1] = float("nan")
    with pytest.raises(FloatingPointError):
        st.step(dt=1e-3)


# --- random fields --------------------------------------------------------------


def test_random_field_is_independent_of_ranks_and_resolution():
    shape = (24, 20, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid = SlabGrid3D(*shape, *CUBE, comm=COMM, device="cpu")
    args = dict(seed=11, kmin=1.0, kmax=4.0, rms=0.7, spectrum="power_law", compressive_fraction=0.0)
    S = random_vector_spectrum(grid, **args)
    serial = SlabGrid3D(*shape, *CUBE, comm=Comm.serial(), device="cpu")
    reference = random_vector_spectrum(serial, **args)
    d = grid.decomp
    assert gmax((S - reference[:, :, d.ky0:d.ky0 + d.nky]).abs().max().item()) < 1e-15
    # rms, zero divergence, and a real field (the spectrum is Hermitian on kx = 0).
    assert grid.spectral_sum((S.real ** 2 + S.imag ** 2).sum(dim=0)) == pytest.approx(0.49, rel=1e-12)
    assert gmax((grid.kx3 * S[0] + grid.ky3 * S[1] + grid.kz3 * S[2]).abs().max().item()) < 1e-15
    assert gmax((grid.forward(grid.inverse(S)) - S).abs().max().item()) < 1e-14
    # The same modes at a higher resolution carry the same coefficients.
    fine = SlabGrid3D(48, 40, 32, *CUBE, comm=Comm.serial(), device="cpu")
    F = random_vector_spectrum(fine, **args).numpy()
    for nz in (1, -1, -4):
        coarse_iz, fine_iz = nz % serial.decomp.Kz, nz % fine.decomp.Kz
        for ny in (2, -3):
            coarse_iy, fine_iy = ny % serial.decomp.Ky, ny % fine.decomp.Ky
            assert np.allclose(F[:, fine_iz, fine_iy, 3], reference.numpy()[:, coarse_iz, coarse_iy, 3],
                               rtol=1e-12, atol=0.0)


def test_gaussian_modes_statistics_and_symmetry():
    nz, ny, nx = np.meshgrid(np.arange(-20, 21), np.arange(-20, 21), np.arange(0, 21), indexing="ij")
    g = gaussian_modes(5, 0, nz, ny, nx)
    assert abs(np.mean(np.abs(g) ** 2) - 1.0) < 0.02 and abs(np.mean(g)) < 0.02
    assert g[20, 20, 0] == 0.0
    assert np.allclose(g[::-1, ::-1, 0], np.conj(g[:, :, 0]))
    assert not np.allclose(g, gaussian_modes(6, 0, nz, ny, nx))
    assert not np.allclose(g, gaussian_modes(5, 1, nz, ny, nx))


# --- forcing --------------------------------------------------------------------


def _band_spectrum(grid, forcing, band):
    """Compact spectrum that holds the band values on this rank's modes."""
    S = grid.zeros_spectral(3)
    if forcing.n_local:
        S.reshape(3, -1)[:, forcing.index] = forcing._to_local(band)
    return S


@pytest.mark.parametrize("zeta", [0.0, 0.4])
def test_forcing_band_is_a_real_field_and_matches_its_synthesis(zeta):
    shape = (20, 18, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid = SlabGrid3D(*shape, 1.0, 2.0, 1.5, comm=COMM, device="cpu")
    forcing = BandForcing(grid, {"type": "ou", "kmin": 1.0, "kmax": 2.9, "amplitude": 0.8,
                                 "correlation_time": 0.3, "seed": 4, "compressive_fraction": zeta},
                          incompressible=False)
    S = _band_spectrum(grid, forcing, forcing.f)
    field = grid.inverse(S)
    assert gmax((grid.forward(field) - S).abs().max().item()) < 1e-14  # Hermitian on kx = 0
    assert gmax((forcing.acceleration() - field).abs().max().item()) < 1e-13
    div = (grid.kx3 * S[0] + grid.ky3 * S[1] + grid.kz3 * S[2]).abs().max().item()
    assert (gmax(div) < 1e-13) == (zeta == 0.0)
    # Every rank holds the same band and advances it identically.
    serial = BandForcing(SlabGrid3D(*shape, 1.0, 2.0, 1.5, comm=Comm.serial(), device="cpu"),
                         {"type": "ou", "kmin": 1.0, "kmax": 2.9, "amplitude": 0.8,
                          "correlation_time": 0.3, "seed": 4, "compressive_fraction": zeta},
                         incompressible=False)
    forcing.advance(0.05)
    serial.advance(0.05)
    assert np.array_equal(forcing.f, serial.f)


def test_forcing_amplitudes_have_the_requested_statistics():
    grid = SlabGrid3D(16, 16, 16, *CUBE, comm=Comm.serial(), device="cpu")
    ou = BandForcing(grid, {"type": "ou", "kmax": 2.5, "amplitude": 0.6, "correlation_time": 1.0},
                     incompressible=True)
    mean_square = np.mean([2.0 * (np.abs(ou._sample()) ** 2).sum() for _ in range(400)])
    assert mean_square == pytest.approx(0.36, rel=0.05)  # <|a|^2> = amplitude^2
    values = []
    for _ in range(400):  # the exact update keeps the stationary variance for any step
        ou.advance(0.7)
        values.append(2.0 * (np.abs(ou.f) ** 2).sum())
    assert np.mean(values) == pytest.approx(0.36, rel=0.1)
    white = BandForcing(grid, {"type": "white", "kmax": 2.5, "power": 0.3}, incompressible=True)
    injected = np.mean([(np.abs(white._sample()) ** 2).sum() for _ in range(400)])
    assert injected == pytest.approx(0.3, rel=0.05)


def test_power_law_forcing_follows_the_requested_spectrum():
    grid = SlabGrid3D(32, 32, 32, *CUBE, comm=Comm.serial(), device="cpu")
    index = -5.0 / 3.0
    ou = BandForcing(grid, {"type": "ou", "kmin": 2.0, "kmax": 8.0, "amplitude": 1.0, "profile": "power_law",
                            "spectral_index": index}, incompressible=True)
    mag = np.sqrt((ou.n.astype(float) ** 2).sum(axis=1))
    variance = np.mean([np.abs(ou._sample()) ** 2 for _ in range(300)], axis=0).sum(axis=1)  # per mode
    # Variance per mode ~ k^(index - 2): a straight line in the logarithms.
    slope = np.polyfit(np.log(mag), np.log(variance), 1)[0]
    assert slope == pytest.approx(index - 2.0, abs=0.08)
    # Per mode of a shell (the lattice does not hold exactly k^2 modes per shell).
    shell = lambda c: variance[np.abs(mag - c) <= 0.5].mean()
    assert shell(6.0) / shell(3.0) == pytest.approx(2.0 ** (index - 2.0), rel=0.12)
    assert 2.0 * variance.sum() == pytest.approx(1.0, rel=0.05)  # the rms acceleration is the amplitude
    assert "k^-1.667 spectrum" in ou.describe()
    alias = BandForcing(grid, {"type": "ou", "kmin": 2.0, "kmax": 8.0, "profile": "kolmogorov"},
                        incompressible=True)
    assert alias.profile == "power_law" and np.allclose(alias._envelope, ou._envelope)
    with pytest.raises(ValueError):
        BandForcing(grid, {"type": "ou", "profile": "gaussian"}, incompressible=True)


def test_constant_power_forcing_injects_exactly_that_power():
    shape = (16, 16, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    power = 0.25
    grid, st = _make(shape, forcing={"type": "constant_power", "kmin": 1.0, "kmax": 2.5, "power": power})
    st.seed(_helical_state(grid))
    e0, dt, n = st.diagnostics()["total_energy"], 2e-4, 50
    for _ in range(n):
        st.step(dt=dt)
    d = st.diagnostics()
    # Ideal run: the only energy source is the forcing (frozen per step, so first order in dt).
    assert d["total_energy"] - e0 == pytest.approx(power * n * dt, rel=2e-3)
    assert d["forcing_power"] == pytest.approx(power, rel=1e-2)


def test_white_noise_kicks_account_for_the_energy():
    shape = (16, 16, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(shape, forcing={"type": "white", "kmin": 1.0, "kmax": 2.5, "power": 0.5, "seed": 2})
    e0 = st.diagnostics()["total_energy"]
    assert e0 == 0.0
    for _ in range(60):
        st.step(dt=5e-3)
    d = st.diagnostics()
    # Ideal run from rest: the energy is what the kicks added (to RK accuracy) ...
    assert d["kinetic_energy"] == pytest.approx(d["injected_energy"], rel=1e-6)
    # ... and that is the requested power on average.
    assert d["injected_energy"] == pytest.approx(0.5 * 60 * 5e-3, rel=0.35)
    assert d["max_div_u_hat"] < 1e-15


def test_forced_run_is_independent_of_the_rank_count():
    shape = (16, 16, 16)
    if not fits(shape):
        pytest.skip("grid too small for this rank count")
    cfg = {"type": "ou", "kmin": 1.0, "kmax": 2.5, "amplitude": 1.5, "correlation_time": 0.05, "seed": 9}
    grid, st = _make(shape, nu=0.02, eta=0.01, forcing=dict(cfg))
    serial_grid, serial = _make(shape, nu=0.02, eta=0.01, forcing=dict(cfg), comm=Comm.serial())
    for stepper, g in ((st, grid), (serial, serial_grid)):
        stepper.set_spectrum(torch.cat([random_vector_spectrum(g, 1, 1.0, 4.0, 1.0),
                                        random_vector_spectrum(g, 2, 1.0, 4.0, 0.1)]))
        for _ in range(8):
            stepper.step()
    d = grid.decomp
    assert st.t == pytest.approx(serial.t, rel=1e-13)
    assert gmax((st.uhat - serial.uhat[:, :, d.ky0:d.ky0 + d.nky]).abs().max().item()) < 1e-13
    assert st.diagnostics()["forcing_power"] == pytest.approx(serial.diagnostics()["forcing_power"], rel=1e-10)


def test_forcing_state_round_trip():
    grid = SlabGrid3D(16, 16, 16, *CUBE, comm=Comm.serial(), device="cpu")
    cfg = {"type": "ou", "kmax": 2.5, "amplitude": 1.0, "correlation_time": 0.2, "seed": 3}
    a, b = BandForcing(grid, cfg, True), BandForcing(grid, cfg, True)
    for _ in range(3):
        a.advance(0.1)
    import json

    b.load_state_dict(json.loads(json.dumps(a.state_dict())))
    a.advance(0.1)
    b.advance(0.1)
    assert np.array_equal(a.f, b.f)
    with pytest.raises(ValueError):
        BandForcing(grid, dict(cfg, type="white", power=1.0), True).load_state_dict(a.state_dict())
    with pytest.raises(ValueError):
        BandForcing(grid, {"type": "constant_power", "power": 1.0}, incompressible=False)
