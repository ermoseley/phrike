"""Spectral and transfer diagnostics of the distributed solvers.

Every global quantity is compared with a NumPy computation on the gathered
whole-grid field; the transfer identities (conservation, antisymmetry, the
solver's own nonlinear term) hold to round-off.  Rank independence is checked
against a serial stepper built on the same (rank-independent) random field.
Under MPI every rank runs the same checks.
"""

import numpy as np
import pytest

from _parallel_util import COMM, fits, full_wavenumbers, torch
from phrike.parallel import Comm
from phrike.parallel.analysis import spectral_diagnostics, transfer_diagnostics
from phrike.parallel.compressible import CompressibleSlabStepper
from phrike.parallel.fields import random_vector_spectrum
from phrike.parallel.grid import SlabGrid3D
from phrike.parallel.incompressible import IncompressibleSlabStepper

TWO_PI = 2.0 * np.pi
CUBE = (TWO_PI, TWO_PI, TWO_PI)
SHAPE = (24, 20, 16)  # Nx, Ny, Nz: retained modes 8 x 13 x 11
BOXES = [CUBE, (1.0, 2.0, 1.5)]
VISC = dict(nu=2e-3, eta=3e-3, hyper_nu=1e-5, hyper_eta=2e-5)


def _make(shape=SHAPE, lengths=CUBE, mhd=True, comm=None, mean=(0.0, 0.0, 0.0), precision="double",
          seeds=(1, 2), **kwargs):
    """(grid, stepper) holding random solenoidal ``u`` and ``b`` (plus a uniform field ``mean``)."""
    comm = comm or COMM
    Nx, Ny, Nz = shape
    grid = SlabGrid3D(Nx, Ny, Nz, *lengths, comm=comm, device="cpu", precision=precision, fft_batch=3)
    st = IncompressibleSlabStepper(grid, mhd=mhd, **{"nu": 0.0, **kwargs})
    fields = [random_vector_spectrum(grid, seeds[0], 1.0, 9.0, 1.0, spectrum="flat")]
    if mhd:
        fields.append(random_vector_spectrum(grid, seeds[1], 1.0, 9.0, 0.7, spectrum="flat"))
    S = torch.cat(fields)
    if mhd and grid.decomp.ky0 == 0:
        for c in range(3):
            S[3 + c, 0, 0, 0] = mean[c]
    st.set_spectrum(S)
    return grid, st


def _gather(st):
    """The whole physical field ``(ncomp, Nz, Ny, Nx)`` on every rank."""
    local = st.physical().numpy()
    return np.concatenate(COMM.bcast(COMM.gather(local)), axis=1)


def _full_k(shape, lengths):
    KX, KY, KZ, mask = full_wavenumbers(shape, lengths)
    return np.stack([KX, KY, KZ]), mask


FFT = lambda f: np.fft.fftn(f, axes=(-3, -2, -1))
IFFT = lambda F: np.fft.ifftn(F, axes=(-3, -2, -1)).real


def _curl(F, K):
    return np.stack([1j * (K[1] * F[2] - K[2] * F[1]), 1j * (K[2] * F[0] - K[0] * F[2]),
                     1j * (K[0] * F[1] - K[1] * F[0])])


def _rel(a, b):
    return np.abs(np.asarray(a) - np.asarray(b)).max() / max(np.abs(b).max(), 1e-300)


def _close(a, b, tol=1e-12):
    assert np.shape(a) == np.shape(b)
    assert _rel(a, b) < tol, (_rel(a, b), a, b)


def _same_results(a, b, tol):
    assert set(a) == set(b)
    for key in a:
        if key == "attrs":
            assert set(a[key]) == set(b[key])
            for name, value in a[key].items():
                if isinstance(value, float):
                    assert value == pytest.approx(b[key][name], abs=1e-12)
                else:
                    assert value == b[key][name]
        else:
            assert np.shape(a[key]) == np.shape(b[key]), key
            fin = np.isfinite(b[key])
            assert np.array_equal(fin, np.isfinite(a[key])), key
            scale = max(np.abs(np.asarray(b[key])[fin]).max(initial=0.0), 1e-300)
            assert np.abs(np.asarray(a[key])[fin] - np.asarray(b[key])[fin]).max(initial=0.0) <= tol * scale, key


# --- shell spectra ------------------------------------------------------------------

@pytest.mark.parametrize("lengths", BOXES)
@pytest.mark.parametrize("mhd", [True, False])
def test_shell_sums_match_physical_means(lengths, mhd):
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(lengths=lengths, mhd=mhd, mean=(0.3, -0.2, 0.5))
    d = spectral_diagnostics(st, {"current_helicity": True})
    V = _gather(st)
    K, _ = _full_k(SHAPE, lengths)
    u = V[:3]
    _close(d["kinetic"].sum(), 0.5 * (u ** 2).sum(axis=0).mean())
    _close(d["kinetic_helicity"].sum(), (u * IFFT(_curl(FFT(u), K))).sum(axis=0).mean())
    ref = st.spectra()
    _close(d["k"], ref["k"])
    _close(d["kinetic"], ref["kinetic"])
    assert d["attrs"]["system"] == ("incompressible_mhd" if mhd else "incompressible_hydro")
    if not mhd:
        assert "magnetic" not in d and "cross_helicity" not in d
        return
    b = V[3:]
    k2 = (K ** 2).sum(axis=0)
    a = IFFT(_curl(FFT(b), K) / np.where(k2 == 0, 1.0, k2))
    jb = IFFT(_curl(FFT(b), K))
    _close(d["magnetic"].sum(), 0.5 * (b ** 2).sum(axis=0).mean())
    _close(d["magnetic"], ref["magnetic"])
    _close(d["cross_helicity"].sum(), (u * b).sum(axis=0).mean())
    _close(d["magnetic_helicity"].sum(), (a * b).sum(axis=0).mean())
    _close(d["current_helicity"].sum(), (jb * b).sum(axis=0).mean())
    assert abs(d["magnetic_helicity"][0]) < 1e-14  # the mean field has no gauge-invariant helicity


def test_dissipation_spectra_match_solver_rates():
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    _, st = _make(**VISC)
    d, diag = spectral_diagnostics(st, {}), st.diagnostics()
    assert diag["hyperviscous_dissipation"] > 0.0 and diag["hyperresistive_dissipation"] > 0.0
    _close(d["dissipation_kinetic"].sum(), diag["viscous_dissipation"] + diag["hyperviscous_dissipation"])
    _close(d["dissipation_magnetic"].sum(), diag["resistive_dissipation"] + diag["hyperresistive_dissipation"])
    _close(d["cross_helicity"].sum(), diag["cross_helicity"])
    _close(d["magnetic_helicity"].sum(), diag["magnetic_helicity"])
    assert d["dissipation_kinetic"].sum() + d["dissipation_magnetic"].sum() > 0.0


# --- anisotropy ---------------------------------------------------------------------

def _set_modes(grid, st, entries):
    """Replace the spectrum by single modes: ``(component offset, (nx, ny, nz), complex 3-vector)``."""
    S = torch.zeros_like(st.uhat)
    for offset, (nx, ny, nz), vec in entries:
        iz = np.nonzero(grid.nz_modes == nz)[0]
        iy = np.nonzero(grid.ny_modes == ny)[0]
        if iy.size:
            for c in range(3):
                S[offset + c, iz[0], iy[0], nx] += vec[c]
    st.set_spectrum(S)


def test_single_mode_lands_in_its_bin():
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    grid, st = _make()
    n = (3, 4, 2)  # k = (3, 4, 2) in a 2 pi box
    amp = 0.7
    _set_modes(grid, st, [(0, n, amp * np.array([4.0, -3.0, 0.0])),  # u perpendicular to k
                          (3, (1, 2, 3), np.array([2.0, -1.0, 0.0]) * 0.5),
                          (3, (0, 0, 0), np.array([0.6, 0.0, 0.0]))])  # mean field along x
    energy = (amp * 5.0) ** 2  # w = 2: shell sum is |u_k|^2
    d = spectral_diagnostics(st, {})
    assert d["attrs"]["axis"] == "auto: mean field"
    assert (d["attrs"]["axis_x"], d["attrs"]["axis_y"], d["attrs"]["axis_z"]) == (1.0, 0.0, 0.0)
    expect = np.zeros_like(d["kinetic_2d"])
    expect[3, 4] = energy  # k_par = 3, k_perp = sqrt(4^2 + 2^2) = 4.47 -> 4
    _close(d["kinetic_2d"], expect)
    assert d["k_par"].shape == (d["kinetic_2d"].shape[0],) and d["k_perp"].shape == (d["kinetic_2d"].shape[1],)
    _close(d["kinetic_par"].sum(), energy)
    _close(d["kinetic_perp"].sum(), energy)
    assert d["kinetic_par"][3] == pytest.approx(energy) and d["kinetic_perp"][4] == pytest.approx(energy)
    _close(d["magnetic_2d"].sum(), st.diagnostics()["magnetic_energy"])
    assert d["magnetic_2d"][0, 0] == pytest.approx(0.5 * 0.36)  # the mean field
    for axis, bin_ in (("z", (2, 5)), ("y", (4, 4)), ([1, 1, 0], (5, 2))):
        d = spectral_diagnostics(st, {"anisotropy": {"axis": axis}})
        expect = np.zeros_like(d["kinetic_2d"])
        expect[bin_] = energy
        _close(d["kinetic_2d"], expect)
    z = spectral_diagnostics(st, {"anisotropy": {"axis": "z"}})
    assert z["u_component_anisotropy"] == pytest.approx(0.0, abs=1e-14)  # u has no z component
    assert z["u_gradient_anisotropy"] == pytest.approx(4.0 / 25.0)  # k_par^2 / k_perp^2
    assert z["attrs"]["axis"] == "z"


def test_axis_selection_and_validation():
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(mean=(0.0, 0.0, 0.0))
    d = spectral_diagnostics(st, {})
    assert d["attrs"]["axis_z"] == 1.0 and "no mean field" in d["attrs"]["axis"]  # negligible mean: z
    grid, st = _make(mean=(0.0, 1.0, 1.0))
    d = spectral_diagnostics(st, {})
    assert d["attrs"]["axis_y"] == pytest.approx(np.sqrt(0.5)) and d["attrs"]["axis_z"] == pytest.approx(np.sqrt(0.5))
    _, hydro = _make(mhd=False)
    assert spectral_diagnostics(hydro, {})["attrs"]["axis_z"] == 1.0
    assert "magnetic_2d" not in spectral_diagnostics(hydro, {})
    assert "kinetic_2d" not in spectral_diagnostics(st, {"anisotropy": False})
    for bad in ("w", [0, 0, 0], [1, 2]):
        with pytest.raises(ValueError):
            spectral_diagnostics(st, {"anisotropy": {"axis": bad}})


@pytest.mark.parametrize("lengths", BOXES)
def test_anisotropy_matches_numpy(lengths):
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(lengths=lengths, mean=(0.0, 0.0, 0.4))
    d = spectral_diagnostics(st, {"anisotropy": {"axis": "auto"}})
    diag = st.diagnostics()
    assert d["attrs"]["axis_z"] == 1.0
    _close(d["kinetic_par"].sum(), diag["kinetic_energy"])
    _close(d["kinetic_perp"].sum(), diag["kinetic_energy"])
    _close(d["magnetic_par"].sum(), diag["magnetic_energy"])
    _close(d["magnetic_perp"].sum(), diag["magnetic_energy"])
    V = _gather(st)
    K, _ = _full_k(SHAPE, lengths)
    for name, f in (("u", V[:3]), ("b", V[3:])):
        F = FFT(f)
        mean = lambda x: x.mean()
        comp = mean(f[2] ** 2) / mean(f[0] ** 2 + f[1] ** 2)
        dz = IFFT(1j * K[2] * F)
        gz = mean((dz ** 2).sum(axis=0))
        gperp = sum(mean(IFFT(1j * K[j] * F[c]) ** 2) for j in (0, 1) for c in range(3))
        _close(d[f"{name}_component_anisotropy"], comp)
        _close(d[f"{name}_gradient_anisotropy"], gz / gperp)
    # The 2D spectrum, summed along k_par at fixed k_perp, equals a direct binning of all modes.
    dk = TWO_PI / max(lengths)
    kpar = np.abs(K[2])
    kperp = np.sqrt(K[0] ** 2 + K[1] ** 2)
    _, mask = _full_k(SHAPE, lengths)
    Fu = FFT(V[:3]) / V[0].size
    dens = 0.5 * (np.abs(Fu) ** 2).sum(axis=0)
    keep = mask > 0
    ip, iq = np.floor(kpar / dk + 0.5).astype(int), np.floor(kperp / dk + 0.5).astype(int)
    ref = np.zeros(d["kinetic_2d"].shape)
    np.add.at(ref, (ip[keep], iq[keep]), dens[keep])
    _close(d["kinetic_2d"], ref)


# --- compressible spectra -----------------------------------------------------------

def _compressible(system, lengths=CUBE, gamma=5.0 / 3.0):
    Nx, Ny, Nz = SHAPE
    grid = SlabGrid3D(Nx, Ny, Nz, *lengths, comm=COMM, device="cpu", precision="double", fft_batch=3)
    st = CompressibleSlabStepper(grid, system, gamma, 1e-10, 1e-12)
    d = grid.decomp
    z, y, x = np.meshgrid(*(np.arange(n) * 2 * np.pi / n for n in (Nz, Ny, Nx)), indexing="ij")
    rng = np.random.default_rng(4)
    rho = 1.0 + 0.2 * np.sin(x + z) * np.cos(2 * y) + 0.03 * rng.standard_normal(x.shape)
    u = [np.sin(y) * np.cos(z) + 0.1 * rng.standard_normal(x.shape), np.cos(x) * np.sin(z),
         0.5 * np.sin(2 * x + y) + 0.1 * rng.standard_normal(x.shape)]
    U = [rho] + [rho * c for c in u] + [10.0 * np.ones_like(x)]
    if system == "mhd":
        U += [0.3 + 0.2 * np.sin(y), 0.2 * np.cos(z + x), 0.2 * np.sin(x)]
    st.seed(torch.as_tensor(np.stack(U)[:, d.z0:d.z0 + d.nz].copy()))
    return grid, st


@pytest.mark.parametrize("lengths", BOXES)
@pytest.mark.parametrize("system", ["euler", "mhd"])
def test_compressible_spectra_match_physical_means(system, lengths):
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    grid, st = _compressible(system, lengths)
    U = np.concatenate(COMM.bcast(COMM.gather(st.physical().numpy())), axis=1)
    rho, m = U[0], U[1:4]
    d = spectral_diagnostics(st, {})
    # u = m / rho and sqrt(rho) u are not band limited: the spectra hold their 2/3-truncated parts.
    _, mask = _full_k(SHAPE, lengths)
    trunc = lambda f: IFFT(FFT(f) * mask)
    _close(d["kinetic"].sum(), 0.5 * (trunc(m / rho) ** 2).sum(axis=0).mean())
    _close(d["kinetic_density_weighted"].sum(), 0.5 * (trunc(m / np.sqrt(rho)) ** 2).sum(axis=0).mean())
    _close(d["density"].sum(), 0.5 * (rho ** 2).mean())
    assert d["density"][0] == pytest.approx(0.5 * rho.mean() ** 2)
    assert d["attrs"]["system"] == system
    _close(d["kinetic_par"].sum(), d["kinetic"].sum())
    _close(d["kinetic_perp"].sum(), d["kinetic"].sum())
    _close(d["k"], st.spectra()["k"])
    _close(d["kinetic"], st.spectra()["kinetic"])  # the stepper's own spectrum of u
    if system == "mhd":
        B = U[5:8]
        _close(d["magnetic"].sum(), 0.5 * (B ** 2).sum(axis=0).mean())
        _close(d["magnetic_2d"].sum(), d["magnetic"].sum())
        assert d["attrs"]["axis"] == "auto: mean field" and d["attrs"]["axis_x"] == 1.0
        _close(d["b_component_anisotropy"], (B[0] ** 2).mean() / (B[1] ** 2 + B[2] ** 2).mean())
        assert np.isfinite(d["u_gradient_anisotropy"])
    else:
        assert "magnetic" not in d and d["attrs"]["axis_z"] == 1.0
        with pytest.raises(NotImplementedError):
            transfer_diagnostics(st, {})


# --- transfers ----------------------------------------------------------------------

def _channels(mhd):
    return ["uu", "bb", "ub", "bu"] if mhd else ["uu"]


@pytest.mark.parametrize("lengths", BOXES)
def test_transfer_conservation_identities(lengths):
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    _, st = _make(lengths=lengths, mean=(0.2, 0.0, 0.3))
    t = transfer_diagnostics(st, {})
    scale = max(np.abs(t[f"T_{c}"]).max() for c in _channels(True))
    assert scale > 1e-3
    assert abs(t["T_uu"].sum()) < 1e-12 * scale
    assert abs(t["T_bb"].sum()) < 1e-12 * scale
    assert abs((t["T_ub"] + t["T_bu"]).sum()) < 1e-12 * scale
    for c in _channels(True):
        _close(t[f"Pi_{c}"], -np.cumsum(t[f"T_{c}"]))
    assert abs(t["Pi_uu"][-1]) < 1e-12 * scale
    assert t["k"].shape == t["T_uu"].shape == t["Pi_ub"].shape


def _numpy_unit_shell_sums(st, dens):
    """Sum a per-mode density (local numpy) over the unit shells, as ``spectra()``, over all ranks."""
    g = st.grid
    dk = TWO_PI / max(g.Lx, g.Ly, g.Lz)
    shell = np.floor(np.sqrt(g.k2().numpy()) / dk + 0.5).astype(np.int64)
    nbins = int(np.floor(np.sqrt(g.k2_max) / dk + 0.5)) + 1
    return COMM.allreduce(np.bincount(shell.reshape(-1), weights=dens.reshape(-1), minlength=nbins), "sum")


@pytest.mark.parametrize("lengths", BOXES)
@pytest.mark.parametrize("mhd", [True, False])
def test_transfer_matches_solver_nonlinear_term(lengths, mhd):
    """With nu = eta = 0 and no forcing, d/dt of each shell's energy is the solver's own term."""
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(lengths=lengths, mhd=mhd, mean=(0.2, 0.0, 0.3))
    t = transfer_diagnostics(st, {})
    uh, N = st.uhat.numpy(), st.rhs().numpy()
    w = grid.weight3.numpy()
    rate_u = _numpy_unit_shell_sums(st, w * (uh[:3].conj() * N[:3]).real.sum(axis=0))
    assert np.abs(rate_u).max() > 1e-3
    _close(t["T_uu"] + (t["T_ub"] if mhd else 0.0), rate_u)
    if mhd:
        rate_b = _numpy_unit_shell_sums(st, w * (uh[3:].conj() * N[3:]).real.sum(axis=0))
        _close(t["T_bb"] + t["T_bu"], rate_b)
    else:
        assert set(t) == {"k", "T_uu", "Pi_uu", "attrs"}


def _unit_edges(st):
    g = st.grid
    dk = TWO_PI / max(g.Lx, g.Ly, g.Lz)
    nbins = int(np.floor(np.sqrt(g.k2_max) / dk + 0.5)) + 1
    return np.concatenate(([0.0], np.arange(nbins) + 0.5))


@pytest.mark.parametrize("mhd", [True, False])
def test_transfer_matrix_antisymmetry_and_rows(mhd):
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    _, st = _make(lengths=BOXES[1], mhd=mhd, mean=(0.2, 0.0, 0.3))
    # Edges that reproduce the unit shells: the matrix rows then sum to the cheap-path transfer.
    t = transfer_diagnostics(st, {"transfer": {"matrix": True, "shells": _unit_edges(st)}})
    nb = t["T_uu"].size
    assert t["shell_edges"].shape == (nb + 1,)
    M = {c: t[f"T_{c}_matrix"] for c in _channels(mhd)}
    assert all(m.shape == (nb, nb) for m in M.values())
    scale = np.abs(M["uu"]).max()
    assert scale > 1e-4
    for c in _channels(mhd):
        _close(M[c].sum(axis=1), t[f"T_{c}"], 1e-11)
    assert np.abs(M["uu"] + M["uu"].T).max() < 1e-12 * scale
    if mhd:
        assert np.abs(M["bb"] + M["bb"].T).max() < 1e-12 * scale
        assert np.abs(M["ub"] + M["bu"].T).max() < 1e-12 * scale
        assert np.abs(M["ub"]).max() > 1e-4 * scale
    else:
        assert set(M) == {"uu"} and "T_bb_matrix" not in t


def _numpy_matrices(V, edges, lengths, mhd):
    """T_X(K, Q) from the gathered field with full FFTs and the advective form (u.grad) X_Q."""
    K, mask = _full_k(SHAPE, lengths)
    dk = TWO_PI / max(lengths)
    kk = np.sqrt((K ** 2).sum(axis=0)) / dk
    ns = edges.size - 1
    masks = [((kk >= edges[n]) & (kk < edges[n + 1])) * mask for n in range(ns)]
    u, b = V[:3], V[3:]
    Fu, Fb = FFT(u), (FFT(b) if mhd else None)
    adv = lambda a, F: np.stack([sum(a[j] * IFFT(1j * K[j] * F[i]) for j in range(3)) for i in range(3)])
    out = {c: np.zeros((ns, ns)) for c in _channels(mhd)}
    for q in range(ns):
        uq = IFFT(Fu * masks[q])
        bq = IFFT(Fb * masks[q]) if mhd else None
        for k_ in range(ns):
            uk = IFFT(Fu * masks[k_])
            out["uu"][k_, q] = -(uk * adv(u, FFT(uq))).sum(axis=0).mean()
            if mhd:
                bk = IFFT(Fb * masks[k_])
                out["bb"][k_, q] = -(bk * adv(u, FFT(bq))).sum(axis=0).mean()
                out["ub"][k_, q] = (uk * adv(b, FFT(bq))).sum(axis=0).mean()
                out["bu"][k_, q] = (bk * adv(b, FFT(uq))).sum(axis=0).mean()
    return out


@pytest.mark.parametrize("mhd", [True, False])
def test_transfer_matrix_matches_numpy(mhd):
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    grid, st = _make(mhd=mhd, mean=(0.2, 0.0, 0.3))
    t = transfer_diagnostics(st, {"transfer": {"matrix": True}})
    np.testing.assert_array_equal(t["shell_edges"], [0.0, 1.0, 2.0, 4.0, 8.0, 16.0])  # k up to ~10.5
    ref = _numpy_matrices(_gather(st), t["shell_edges"], CUBE, mhd)
    for c in _channels(mhd):
        _close(t[f"T_{c}_matrix"], ref[c], 1e-11)
    # Explicit edges that leave modes out drop their columns and rows; fields can be cached.
    edges = [1.0, 3.0, 7.0]
    cached = transfer_diagnostics(st, {"transfer": {"matrix": True, "shells": edges, "cache_fields": True,
                                                    "batch": 7}})
    ref2 = _numpy_matrices(_gather(st), np.array(edges), CUBE, mhd)
    for c in _channels(mhd):
        _close(cached[f"T_{c}_matrix"], ref2[c], 1e-11)
    with pytest.raises(ValueError):
        transfer_diagnostics(st, {"transfer": {"matrix": True, "shells": [2.0, 1.0]}})


# --- rank independence and precision ------------------------------------------------

def test_results_do_not_depend_on_the_rank_count():
    """Distributed results equal a serial stepper's on the same random field (serial under one rank)."""
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    kwargs = dict(lengths=BOXES[1], mean=(0.1, 0.4, 0.2), **VISC)
    config = {"transfer": {"matrix": True}, "current_helicity": True, "chunk_modes": 300}
    _, st = _make(**kwargs)
    _, ref = _make(comm=Comm.serial(), **kwargs)
    for fn in (spectral_diagnostics, transfer_diagnostics):
        _same_results(fn(st, config), fn(ref, config), 1e-11)
    _, st = _make(**kwargs)
    _same_results(transfer_diagnostics(st, {"chunk_modes": 10**9}), transfer_diagnostics(ref, {}), 1e-11)


def test_compressible_results_do_not_depend_on_the_rank_count():
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    _, st = _compressible("mhd", BOXES[1])
    grid = SlabGrid3D(*SHAPE, *BOXES[1], comm=Comm.serial(), device="cpu", precision="double", fft_batch=3)
    ref = CompressibleSlabStepper(grid, "mhd", 5.0 / 3.0, 1e-10, 1e-12)
    full = np.concatenate(COMM.bcast(COMM.gather(st.physical().numpy())), axis=1)
    ref.set_spectrum(grid.forward(torch.as_tensor(full), normalized=True))
    _same_results(spectral_diagnostics(st, {}), spectral_diagnostics(ref, {}), 1e-11)


def test_single_precision_agrees_with_double():
    if not fits(SHAPE):
        pytest.skip("grid too small for this rank count")
    _, hi = _make(mean=(0.0, 0.0, 0.3), **VISC)
    _, lo = _make(precision="single", mean=(0.0, 0.0, 0.3), **VISC)
    a, b = spectral_diagnostics(lo, {}), spectral_diagnostics(hi, {})
    for key in ("kinetic", "magnetic", "cross_helicity", "dissipation_kinetic", "kinetic_2d"):
        assert _rel(a[key], b[key]) < 2e-5, key
    ta, tb = transfer_diagnostics(lo, {}), transfer_diagnostics(hi, {})
    for key in ("T_uu", "T_bb", "T_ub", "T_bu"):
        assert _rel(ta[key], tb[key]) < 2e-4, key
