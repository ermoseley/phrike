"""Analysis output: spectrally coarsened fields, file formats, XDMF, diagnostics wiring, restart.

Every test runs serially and, unchanged, under ``mpirun -n P`` (see ``_parallel_util``):
each rank builds the same state, the output is written collectively, and rank 0
reads the files back and compares them with a NumPy reference computed from the
gathered fine field (full ``fftn``, mode truncation, ``ifftn`` on the coarse grid).
"""

import os
import sys
import types
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from _parallel_util import COMM, fits, gmax, shared_dir, torch
from phrike.parallel import Comm
from phrike.parallel.compressible import CompressibleSlabStepper
from phrike.parallel.driver import Simulation
from phrike.parallel.grid import SlabGrid3D
from phrike.parallel.incompressible import IncompressibleSlabStepper
from phrike.parallel.output import (OutputWriter, SpectralCoarsener, _H5Parts, _Layout, load_fields,
                                    parse_coarsen, parse_variables)

TWO_PI = 2.0 * np.pi
SHAPE = (24, 20, 16)  # (Nx, Ny, Nz)
LENGTHS = (TWO_PI, 3.0, 5.0)  # anisotropic: a swapped axis shows
GAMMA = 5.0 / 3.0

pytestmark = pytest.mark.skipif(not fits(SHAPE), reason="grid too small for this rank count")

h5py = pytest.importorskip("h5py")


# --- states and references ----------------------------------------------------------


def _grid(precision="double", shape=SHAPE):
    return SlabGrid3D(*shape, *LENGTHS, comm=COMM, device="cpu", precision=precision)


def _slab(field, grid):
    d = grid.decomp
    return torch.as_tensor(np.ascontiguousarray(field[:, d.z0:d.z0 + d.nz]))


def _gather(local):
    """The whole ``(C, Nz, Ny, Nx)`` array on every rank from this rank's slab."""
    return np.concatenate(COMM.bcast(COMM.gather(np.asarray(local))), axis=1)


def _incompressible(mhd=True, precision="double"):
    grid = _grid(precision)
    st = IncompressibleSlabStepper(grid, nu=0.01, eta=0.01, mhd=mhd, scheme="rk4", triton=False)
    Nx, Ny, Nz = SHAPE
    noise = np.random.default_rng(11).standard_normal((6 if mhd else 3, Nz, Ny, Nx))
    st.seed(_slab(noise, grid))
    return grid, st


def _smooth(rng, kmax=3):
    Nx, Ny, Nz = SHAPE
    F = np.fft.fftn(rng.standard_normal((Nz, Ny, Nx)))
    keep = [np.abs(np.fft.fftfreq(n, 1.0 / n)) <= kmax for n in (Nz, Ny, Nx)]
    F *= keep[0][:, None, None] * keep[1][None, :, None] * keep[2][None, None, :]
    f = np.fft.ifftn(F).real
    return f / np.abs(f).max()


def _compressible(system):
    grid = _grid()
    st = CompressibleSlabStepper(grid, system, GAMMA, 1e-10, 1e-12)
    rng = np.random.default_rng(5)
    rho = 1.0 + 0.3 * _smooth(rng)
    u = [0.3 * _smooth(rng) for _ in range(3)]
    B = [0.3 * _smooth(rng) for _ in range(3)] if system == "mhd" else []
    p = 1.0 + 0.2 * _smooth(rng)
    E = p / (GAMMA - 1.0) + 0.5 * rho * sum(c * c for c in u) + 0.5 * sum(c * c for c in B)
    U = np.stack([rho] + [rho * c for c in u] + [E] + B)
    st.seed(_slab(U, grid))
    return grid, st


def _retained(shape):
    Nx, Ny, Nz = shape
    axes = [np.abs(np.rint(np.fft.fftfreq(n) * n)) <= (n - 1) // 3 for n in (Nz, Ny, Nx)]
    return axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]


def _spectrum(f):
    """Retained modes of a real ``(Nz, Ny, Nx)`` field, divided by N, as ``fftn`` order."""
    Nz, Ny, Nx = f.shape
    return np.fft.fftn(f) / f.size * _retained((Nx, Ny, Nz))


def _synth(F, n_out=None):
    """The field summed from the spectrum ``F`` on the fine grid or, low-passed, on ``n_out = (Nx, Ny, Nz)``."""
    if n_out is None:
        return np.fft.ifftn(F).real * F.size
    Nz, Ny, Nx = F.shape
    Nxo, Nyo, Nzo = n_out
    out = np.zeros((Nzo, Nyo, Nxo), dtype=complex)
    modes = [np.arange(-((no - 1) // 2), (no - 1) // 2 + 1) for no in (Nzo, Nyo, Nxo)]
    out[np.ix_(*(m % no for m, no in zip(modes, (Nzo, Nyo, Nxo))))] = F[
        np.ix_(*(m % n for m, n in zip(modes, (Nz, Ny, Nx))))]
    return np.fft.ifftn(out).real * out.size


def _curl_spectrum(F3):
    """Spectrum of the curl of a vector given as three ``(Nz, Ny, Nx)`` spectra."""
    Nz, Ny, Nx = F3[0].shape
    k = [2 * np.pi * np.rint(np.fft.fftfreq(n) * n) / L for n, L in zip((Nx, Ny, Nz), LENGTHS)]
    kx, ky, kz = k[0][None, None, :], k[1][None, :, None], k[2][:, None, None]
    return [1j * (ky * F3[2] - kz * F3[1]), 1j * (kz * F3[0] - kx * F3[2]), 1j * (kx * F3[1] - ky * F3[0])]


def _on_root(fn):
    """``fn()`` evaluated on rank 0 and shared, so that a failure is seen by every rank."""
    return COMM.bcast(fn() if COMM.is_root else None)


def _writer(outdir, system, grid, fmt="hdf5", **fields):
    block = dict({"enabled": True, "precision": "double"}, **fields)
    config = {"io": {"outdir": outdir, "format": fmt, "fields": block}}
    return OutputWriter(grid, COMM, system, config)


def _error(a, b):
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


COARSE = [(12, 10, 8), (11, 9, 7), (17, 12, 5), (4, 4, 4), (24, 20, 16), (13, 20, 6), (5, 6, 4)]


# --- coarsening ------------------------------------------------------------------------


@pytest.mark.parametrize("n_out", COARSE)
def test_coarsened_fields_are_the_exact_spectral_lowpass(n_out):
    grid, st = _incompressible()
    coarsener = SpectralCoarsener(grid, n_out)
    assert sum(coarsener.layout.z_counts) == n_out[2]
    assert (coarsener.layout.nz > 0) == coarsener.active
    reference = _gather(st.physical().numpy())
    worst = 0.0
    for c in range(6):
        out = coarsener.to_physical(st.uhat[c:c + 1]).numpy()
        z0, nz = coarsener.layout.z0, coarsener.layout.nz
        assert out.shape == (nz, n_out[1], n_out[0])
        expected = _synth(_spectrum(reference[c]), n_out)[z0:z0 + nz]
        worst = max(worst, _error(out, expected) if nz else 0.0)
    assert gmax(worst) < 1e-12
    coarsener.release()


def test_more_ranks_than_coarse_slabs_leave_idle_ranks():
    grid, _ = _incompressible()
    coarsener = SpectralCoarsener(grid, (6, 5, 4))
    ranks = min(COMM.size, 4, 5)
    assert coarsener.active == (COMM.rank < ranks)
    assert coarsener.layout.z_counts == (list(coarsener.cd.z_counts) + [0] * (COMM.size - ranks))
    assert coarsener.to_physical(grid.zeros_spectral(1)).shape[0] == (coarsener.layout.nz)


def test_sub_communicator_is_the_first_ranks():
    sub = COMM.sub(2)
    if COMM.size <= 2:
        assert sub is COMM
    elif COMM.rank < 2:
        assert sub.size == 2 and sub.rank == COMM.rank
        assert sub.sum(1.0) == 2.0
    else:
        assert sub is None


def test_coarsening_accepts_only_valid_grids():
    shape = SHAPE
    assert parse_coarsen(None, shape) is None
    assert parse_coarsen(12, shape) == (12, 12, 12)
    assert parse_coarsen([12, 10, 8], shape) == (12, 10, 8)
    assert parse_coarsen({"factor": 4}, shape) == (6, 5, 4)
    for bad in (3, [12, 10], [12, 10, 17], 25, {"factor": 3}, {"factor": 0}, {"step": 2}, 8.5, True, [4, 4, 3]):
        with pytest.raises(ValueError):
            parse_coarsen(bad, shape)
    assert parse_variables("default", "imhd") == ["u", "b"]
    assert parse_variables(None, "mhd") == ["rho", "u", "p", "B"]
    assert parse_variables(["u", "omega", "u"], "ins") == ["u", "omega"]
    for system, bad in (("ins", ["b"]), ("imhd", ["rho"]), ("euler", ["B"]), ("mhd", ["velocity"])):
        with pytest.raises(ValueError):
            parse_variables([bad[0]], system)
    grid, _ = _incompressible()
    with pytest.raises(ValueError):
        _writer("unused", "imhd", grid, coarsen=17 + 20)
    with pytest.raises(ValueError):
        _writer("unused", "imhd", grid, precision="quarter")


# --- field files -----------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["hdf5", "npy"])
def test_full_resolution_equals_the_physical_state(tmp_path, fmt):
    outdir = shared_dir(tmp_path)
    grid, st = _incompressible()
    writer = _writer(outdir, "imhd", grid, fmt, coarsen=None, variables=["u", "b"])
    path = writer.write_fields(st, 3, 0.25, 12)
    assert os.path.basename(path).startswith("fields_000003")
    reference = _gather(st.physical().numpy())
    got = _on_root(lambda: load_fields(path))
    Nx, Ny, Nz = SHAPE
    assert got["t"] == 0.25 and got["step"] == 12 and got["attrs"]["system"] == "imhd"
    assert sorted(got["fields"]) == sorted(f"{v}_{c}" for v in "ub" for c in "xyz")
    err = 0.0
    for i, name in enumerate(["u_x", "u_y", "u_z", "b_x", "b_y", "b_z"]):
        assert got["fields"][name].shape == (Nz, Ny, Nx)
        err = max(err, _error(got["fields"][name], reference[i]))
    assert err < 1e-13
    assert got["attrs"]["dz"] == pytest.approx(LENGTHS[2] / Nz)
    assert _on_root(lambda: sorted(load_fields(path, ["b"])["fields"])) == ["b_x", "b_y", "b_z"]


@pytest.mark.parametrize("fmt", ["hdf5", "npy"])
@pytest.mark.parametrize("n_out", [(12, 10, 8), (11, 9, 7), (6, 5, 4)])
def test_coarse_file_holds_lowpass_fields_and_derived_curls(tmp_path, fmt, n_out):
    outdir = shared_dir(tmp_path)
    grid, st = _incompressible()
    writer = _writer(outdir, "imhd", grid, fmt, coarsen=list(n_out), variables=["u", "omega", "j"])
    path = writer.write_fields(st, 0, 0.0, 0)
    reference = _gather(st.physical().numpy())
    expected = {}
    for i, name in ((0, "u"), (3, "b")):
        spec = [_spectrum(reference[i + c]) for c in range(3)]
        expected[name] = [_synth(F, n_out) for F in spec]
        if name == "u":
            expected["omega"] = [_synth(F, n_out) for F in _curl_spectrum(spec)]
        else:
            expected["j"] = [_synth(F, n_out) for F in _curl_spectrum(spec)]
    got = _on_root(lambda: {k: v for k, v in load_fields(path)["fields"].items()})
    assert sorted(got) == sorted(f"{v}_{c}" for v in ("u", "omega", "j") for c in "xyz")
    err = 0.0
    for var in ("u", "omega", "j"):
        for c, ref in zip("xyz", expected[var]):
            assert got[f"{var}_{c}"].shape == (n_out[2], n_out[1], n_out[0])
            err = max(err, _error(got[f"{var}_{c}"], ref))
    assert err < 1e-12
    attrs = _on_root(lambda: load_fields(path)["attrs"])
    assert (attrs["Nx"], attrs["Ny"], attrs["Nz"]) == n_out and attrs["sim_Nx"] == SHAPE[0]
    assert attrs["dx"] == pytest.approx(LENGTHS[0] / n_out[0])
    assert bool(attrs["spectral_lowpass"])


@pytest.mark.parametrize("system", ["euler", "mhd"])
@pytest.mark.parametrize("n_out", [None, (12, 10, 8), (7, 8, 6)])
def test_compressible_primitives_and_derived_fields(tmp_path, system, n_out):
    outdir = shared_dir(tmp_path)
    grid, st = _compressible(system)
    names = ["rho", "u", "p", "m", "E", "omega"] + (["B", "j"] if system == "mhd" else [])
    writer = _writer(outdir, system, grid, "hdf5", coarsen=None if n_out is None else list(n_out),
                     variables=names)
    path = writer.write_fields(st, 1, 0.5, 3)
    U = _gather(st.physical().numpy())
    rho = U[0]
    u = [U[1 + c] / rho for c in range(3)]
    B = [U[5 + c] for c in range(3)] if system == "mhd" else [0 * rho] * 3
    p = (GAMMA - 1.0) * (U[4] - 0.5 * rho * sum(c * c for c in u) - 0.5 * sum(c * c for c in B))
    expected = {"rho": rho, "p": p, "E": U[4]}
    for c in range(3):
        expected[f"u_{'xyz'[c]}"] = u[c]
        expected[f"m_{'xyz'[c]}"] = U[1 + c]
        if system == "mhd":
            expected[f"B_{'xyz'[c]}"] = B[c]
    got = _on_root(lambda: load_fields(path)["fields"])
    if n_out is None:
        assert max(_error(got[k], v) for k, v in expected.items()) < 1e-12
    else:  # primitives are nonlinear: transformed on the fine grid, then low-passed with the retained modes
        assert max(_error(got[k], _synth(_spectrum(v), n_out)) for k, v in expected.items()) < 1e-12
    curls = {"omega": [_spectrum(c) for c in u]}
    if system == "mhd":
        curls["j"] = [_spectrum(c) for c in B]
    for var, spec in curls.items():
        for c, F in zip("xyz", _curl_spectrum(spec)):
            ref = _synth(F, n_out)
            assert _error(got[f"{var}_{c}"], ref) < 1e-12


def test_single_precision_grid_and_output_dtypes(tmp_path):
    outdir = shared_dir(tmp_path)
    grid, st = _incompressible(precision="single")
    assert grid.rdtype == torch.float32
    reference = _gather(st.physical().numpy().astype(np.float64))
    cases = (("single", np.float32, 2e-6), ("half", np.float16, 2e-3), ("double", np.float64, 2e-6))
    for k, (precision, dtype, tol) in enumerate(cases):
        writer = _writer(outdir, "imhd", grid, "hdf5", precision=precision, coarsen=[12, 10, 8],
                         variables=["u"])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = writer.write_fields(st, k, 0.0, 0)
        got = _on_root(lambda: load_fields(path))
        expected = _synth(_spectrum(reference[0]), (12, 10, 8))
        assert got["attrs"]["precision"] == precision
        assert COMM.bcast(got["fields"]["u_x"].dtype == dtype if COMM.is_root else None)
        assert _on_root(lambda: _error(got["fields"]["u_x"].astype(np.float64), expected)) < tol
        xmf = os.path.join(outdir, "fields", f"fields_{k:06d}.xmf")
        assert os.path.exists(xmf) == (precision != "half")
        if precision == "half" and COMM.is_root:
            assert any("XDMF" in str(w.message) for w in caught)
            assert "float16" in str(got["attrs"]["xdmf"])
    # Values beyond the float16 range are reported.
    st.uhat *= 1.0e6
    writer = _writer(outdir, "imhd", grid, "hdf5", precision="half", variables=["u"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        writer.write_fields(st, 9, 0.0, 0)
    assert COMM.rank != 0 or any("float16 maximum" in str(w.message) for w in caught)


def test_parts_file_holds_virtual_datasets_found_from_any_directory(tmp_path):
    """The serial-h5py layout: a part per rank, one stitched dataset per name (read from another cwd)."""
    outdir = shared_dir(tmp_path)
    Nx, Ny, Nz = SHAPE
    d = _grid().decomp
    layout = _Layout((Nz, Ny, Nx), d.z_counts, COMM.rank)
    field = np.random.default_rng(2).standard_normal((2, Nz, Ny, Nx)).astype(np.float32)
    stem = os.path.join(outdir, "stitched")
    handle = _H5Parts(stem, COMM, layout, np.float32)
    for i, name in enumerate(("a", "b")):
        handle.write(name, np.ascontiguousarray(field[i, d.z0:d.z0 + d.nz]))
    handle.close({"t": 1.5, "label": "x"})
    here = os.getcwd()
    try:
        os.chdir("/")
        with h5py.File(stem + ".h5", "r") as f:
            assert f["a"].shape == (Nz, Ny, Nx) and f["a"].is_virtual and f.attrs["t"] == 1.5
            assert np.array_equal(f["a"][...], field[0]) and np.array_equal(f["b"][...], field[1])
    finally:
        os.chdir(here)
    assert sorted(os.listdir(stem + ".parts")) == [f"part.{r:05d}.h5" for r in range(COMM.size)]


def test_writer_uses_parts_with_serial_h5py_and_one_file_otherwise(tmp_path):
    outdir = shared_dir(tmp_path)
    grid, st = _incompressible()
    path = _writer(outdir, "imhd", grid, "hdf5", variables=["u"]).write_fields(st, 0, 0.0, 0)
    stitched = COMM.size > 1 and not h5py.get_config().mpi
    assert os.path.isdir(path[:-3] + ".parts") == stitched
    assert _on_root(lambda: h5py.File(path, "r")["u_x"].is_virtual) == stitched


def test_xdmf_describes_the_datasets_and_the_series(tmp_path):
    outdir = shared_dir(tmp_path)
    grid, st = _incompressible()
    writer = _writer(outdir, "imhd", grid, "hdf5", precision="single", coarsen=[12, 10, 8],
                     variables=["u", "b", "omega"])
    for index in range(3):
        writer.write_fields(st, index, 0.1 * index, index)
    fields = os.path.join(outdir, "fields")

    def check():
        tree = ET.parse(os.path.join(fields, "fields_000001.xmf")).getroot()
        assert tree.tag == "Xdmf" and tree.get("Version") == "3.0"
        grid_xml = tree.find("Domain/Grid")
        top = grid_xml.find("Topology")
        assert top.get("TopologyType") == "3DCoRectMesh" and top.get("Dimensions") == "8 10 12"
        assert float(grid_xml.find("Time").get("Value")) == pytest.approx(0.1)
        geo = grid_xml.find("Geometry")
        assert geo.get("GeometryType") == "ORIGIN_DXDYDZ"
        items = [i.text.split() for i in geo.findall("DataItem")]
        assert items[0] == ["0", "0", "0"]
        assert [float(v) for v in items[1]] == pytest.approx([LENGTHS[0] / 12, LENGTHS[1] / 10, LENGTHS[2] / 8])
        attributes = grid_xml.findall("Attribute")
        scalars = [a for a in attributes if a.get("AttributeType") == "Scalar"]
        vectors = [a for a in attributes if a.get("AttributeType") == "Vector"]
        assert len(scalars) == 9 and sorted(a.get("Name") for a in vectors) == ["b", "omega", "u"]
        with h5py.File(os.path.join(fields, "fields_000001.h5"), "r") as h5:
            refs = [i for i in grid_xml.iter("DataItem") if i.get("Format") == "HDF"]
            assert len(refs) == 9 + 9
            for ref in refs:
                name, dataset = ref.text.split(":/")
                assert os.path.exists(os.path.join(fields, name)) and dataset in h5
                assert ref.get("Precision") == "4" and ref.get("Dimensions") == "8 10 12"
        join = vectors[0].find("DataItem")
        assert join.get("Function") == "JOIN($0, $1, $2)" and join.get("Dimensions") == "8 10 12 3"
        series = ET.parse(os.path.join(fields, "fields.xmf")).getroot().find("Domain/Grid")
        assert series.get("CollectionType") == "Temporal"
        assert [float(g.find("Time").get("Value")) for g in series.findall("Grid")] == pytest.approx([0, 0.1, 0.2])
        return True

    assert _on_root(check)
    # A partial rebuild (as after a restart from output 1) keeps only the outputs up to it.
    writer.rebuild_series(1)
    count = _on_root(lambda: len(ET.parse(os.path.join(fields, "fields.xmf")).getroot().findall("Domain/Grid/Grid")))
    assert count == 2


def test_small_files_roundtrip_in_both_formats(tmp_path):
    outdir = shared_dir(tmp_path)
    grid, _ = _incompressible()
    arrays = {"k": np.arange(5.0), "e": np.linspace(0, 1, 5) ** 2}
    attrs = {"t": 0.5, "step": 7, "note": "hello", "nested": None}
    for fmt in ("hdf5", "npy"):
        writer = _writer(outdir, "imhd", grid, fmt)
        path = writer.write_small(os.path.join(outdir, "small", f"d_{fmt}"), arrays, attrs)
        assert (path is not None) == COMM.is_root
        if COMM.is_root:
            if fmt == "hdf5":
                with h5py.File(path, "r") as f:
                    assert path.endswith(".h5") and np.array_equal(f["e"][...], arrays["e"])
                    assert f.attrs["step"] == 7 and f.attrs["note"] == "hello" and f.attrs["nested"] == "none"
            else:
                z = np.load(path)
                assert path.endswith(".npz") and np.array_equal(z["k"], arrays["k"]) and z["t"] == 0.5


@pytest.mark.parametrize("fmt", ["hdf5", "npy"])
def test_slices_and_spectra_are_written_in_the_chosen_format(tmp_path, fmt):
    outdir = shared_dir(tmp_path)
    for system in ("imhd", "mhd"):
        grid, st = _incompressible() if system == "imhd" else _compressible("mhd")
        writer = _writer(outdir, system, grid, fmt)
        writer.write_slice(st, 2, 0.5, 4)
        writer.write_spectra(st, 2, 0.5, 4)
        full = _gather(st.physical().numpy())
        zmid = SHAPE[2] // 2

        def read():
            stem = os.path.join(outdir, "slices", "slice_000002")
            names = ["ux", "uy", "uz", "bx", "by", "bz"] if system == "imhd" else \
                ["rho", "mx", "my", "mz", "E", "Bx", "By", "Bz"]
            if fmt == "hdf5":
                with h5py.File(stem + ".h5", "r") as f:
                    data = {n: f[n][...] for n in names}
                    t, zi = f.attrs["t"], f.attrs["z_index"]
                    with_spectra = h5py.File(os.path.join(outdir, "spectra", "spectra_000002.h5"), "r")
                    kinetic = with_spectra["kinetic"][...]
                    with_spectra.close()
            else:
                z = np.load(stem + ".npz")
                data, t, zi = {n: z[n] for n in names}, z["t"], z["z_index"]
                kinetic = np.load(os.path.join(outdir, "spectra", "spectra_000002.npz"))["kinetic"]
            err = max(_error(data[n], full[i][zmid]) for i, n in enumerate(names))
            return float(err), float(t), int(zi), float(kinetic.sum())

        err, t, zi, kinetic = _on_root(read)
        assert err < 1e-13 and t == 0.5 and zi == zmid and kinetic > 0.0
        COMM.barrier()


# --- driver ----------------------------------------------------------------------------


def _config(outdir, **io):
    Nx, Ny, Nz = SHAPE
    return {
        "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz, "Lx": TWO_PI, "Ly": TWO_PI, "Lz": TWO_PI},
        "physics": {"gamma": GAMMA, "viscosity": 0.02, "resistivity": 0.03,
                    "mhd": {"resistivity": 1e-3, "viscosity": 2e-3}},
        "integration": {"t0": 0.0, "t_end": 0.06, "cfl": 0.3, "scheme": "rk4", "output_interval": 0.03},
        "initial_conditions": {"vrms": 0.3, "kmin": 1.0, "kmax": 4.0, "b_rms": 0.1, "b_kmax": 3.0},
        "io": dict({"outdir": outdir, "format": "hdf5"}, **io),
        "monitoring": {"enabled": False},
    }


def _run(problem, config, **kwargs):
    sim = Simulation(problem, config, comm=COMM, backend="cpu", quiet=True, **kwargs)
    summary = sim.run()
    COMM.barrier()
    return sim, summary


def _fake_analysis(monkeypatch, calls, transfer_error=None):
    module = types.ModuleType("phrike.parallel.analysis")

    def spectral(stepper, config):
        calls.append(("spectral", dict(config)))
        return {"k": np.arange(4.0), "E": np.full(4, float(COMM.sum(1.0))), "attrs": {"label": "fake"}}

    def transfer(stepper, config):
        calls.append(("transfer", None))
        if transfer_error is not None:
            raise transfer_error
        return {"T": np.zeros(3)}

    module.spectral_diagnostics, module.transfer_diagnostics = spectral, transfer
    monkeypatch.setitem(sys.modules, "phrike.parallel.analysis", module)


def test_driver_writes_fields_and_diagnostics_on_their_schedules(tmp_path, monkeypatch):
    outdir = shared_dir(tmp_path)
    calls = []
    _fake_analysis(monkeypatch, calls)
    io = {"fields": {"enabled": True, "every": 2, "precision": "single", "coarsen": {"factor": 2},
                     "variables": ["u", "b", "j"]},
          "diagnostics": {"enabled": True, "every": 1, "bins": 7, "transfer": {"enabled": True, "every": 2}}}
    config = _config(outdir, **io)
    config["integration"]["t_end"] = 0.09  # outputs 0, 1, 2, 3
    sim, summary = _run("imhd_turb", config)
    if COMM.is_root:
        fields = sorted(os.listdir(os.path.join(outdir, "fields")))
        assert [n for n in fields if n.endswith(".h5")] == ["fields_000000.h5", "fields_000002.h5"]
        assert "fields.xmf" in fields and "fields_000002.xmf" in fields and "fields_000001.xmf" not in fields
        got = load_fields(os.path.join(outdir, "fields", "fields_000002.h5"))
        assert got["fields"]["u_x"].dtype == np.float32 and got["fields"]["u_x"].shape == (8, 10, 12)
        assert got["t"] == pytest.approx(0.06) and set(got["fields"]) >= {"j_z", "b_y"}
        diag = sorted(os.listdir(os.path.join(outdir, "diagnostics")))
        assert diag == [f"spectral_00000{i}.h5" for i in range(4)] + ["transfer_000000.h5", "transfer_000002.h5"]
        with h5py.File(os.path.join(outdir, "diagnostics", "spectral_000003.h5"), "r") as f:
            assert f["E"].shape == (4,) and f.attrs["label"] == "fake" and f.attrs["step"] == sim.step_count
            assert f.attrs["system"] == "imhd" and abs(f.attrs["t"] - 0.09) < 1e-12
    assert calls[0][1]["bins"] == 7 and calls[0][1]["transfer"] == {"enabled": True, "every": 2}
    assert [k for k, _ in calls].count("transfer") == 2
    assert {"out_fields", "out_diag"} <= set(summary["timers_max_seconds"])


def test_transfer_that_is_not_implemented_is_disabled_once(tmp_path, monkeypatch):
    outdir = shared_dir(tmp_path)
    calls = []
    _fake_analysis(monkeypatch, calls, transfer_error=NotImplementedError("compressible"))
    io = {"diagnostics": {"transfer": {"enabled": True, "every": 1}}}
    _run("mhd_turb3d", _config(outdir, **io))
    assert [k for k, _ in calls].count("transfer") == 1 and [k for k, _ in calls].count("spectral") == 3
    if COMM.is_root:
        assert not any(n.startswith("transfer") for n in os.listdir(os.path.join(outdir, "diagnostics")))


def test_missing_analysis_module_disables_diagnostics_with_one_warning(tmp_path, monkeypatch):
    outdir = shared_dir(tmp_path)
    monkeypatch.setitem(sys.modules, "phrike.parallel.analysis", None)  # an import raises ImportError
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run("imhd_turb", _config(outdir))
    if COMM.is_root:
        assert len([w for w in caught if "io.diagnostics is disabled" in str(w.message)]) == 1
        assert not os.path.exists(os.path.join(outdir, "diagnostics"))


def test_disabled_blocks_write_nothing_and_unknown_keys_are_reported(tmp_path, monkeypatch):
    outdir = shared_dir(tmp_path)
    _fake_analysis(monkeypatch, [])
    io = {"fields": {"enabled": False, "evrey": 2}, "diagnostics": {"enabled": False}, "slices": False,
          "spectra": False}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _run("imhd_turb", _config(outdir, **io))
    assert COMM.rank != 0 or any("io.fields.evrey" in str(w.message) for w in caught)
    if COMM.is_root:
        assert not any(os.path.exists(os.path.join(outdir, d)) for d in ("fields", "diagnostics", "slices", "spectra"))


def test_default_format_and_real_analysis_module(tmp_path):
    """No ``io.format``: HDF5 when h5py imports; diagnostics from the analysis module when it exists."""
    pytest.importorskip("phrike.parallel.analysis")
    outdir = shared_dir(tmp_path)
    config = _config(outdir)
    del config["io"]["format"]
    config["io"]["fields"] = {"enabled": True, "coarsen": 12, "variables": ["u"]}
    _run("imhd_turb", config)
    if COMM.is_root:
        assert os.path.exists(os.path.join(outdir, "slices", "slice_000002.h5"))
        assert os.path.exists(os.path.join(outdir, "spectra", "spectra_000002.h5"))
        assert os.path.exists(os.path.join(outdir, "fields", "fields_000002.h5"))
        assert any(n.startswith("spectral_000002") for n in os.listdir(os.path.join(outdir, "diagnostics")))


def test_restart_continues_the_numbering_and_rebuilds_the_series(tmp_path, monkeypatch):
    outdir = shared_dir(tmp_path)
    _fake_analysis(monkeypatch, [])
    io = {"fields": {"enabled": True, "coarsen": [12, 10, 8], "variables": ["u"]}}
    _run("imhd_turb", _config(outdir, **io))  # outputs 0, 1, 2
    fields = os.path.join(outdir, "fields")
    series = os.path.join(fields, "fields.xmf")

    def times():
        grids = ET.parse(series).getroot().findall("Domain/Grid/Grid")
        return [round(float(g.find("Time").get("Value")), 6) for g in grids]

    assert _on_root(times) == [0.0, 0.03, 0.06]
    if COMM.is_root:
        os.remove(series)
    COMM.barrier()
    sim = Simulation("imhd_turb", _config(outdir, **io), comm=COMM, backend="cpu", quiet=True,
                     restart=os.path.join(outdir, "snapshots", "snap_000001"))
    assert sim.out_index == 2
    # Rebuilt at start-up from the outputs the restart point includes (0 and 1), not the later one on disk.
    assert _on_root(times) == [0.0, 0.03]
    sim.run()
    COMM.barrier()
    assert _on_root(times) == [0.0, 0.03, 0.06]
    assert _on_root(lambda: sorted(n for n in os.listdir(fields) if n.endswith(".h5"))) == \
        [f"fields_00000{i}.h5" for i in range(3)]
    assert _on_root(lambda: load_fields(os.path.join(fields, "fields_000002.h5"))["t"]) == pytest.approx(0.06)


def test_npy_format_driver_run(tmp_path, monkeypatch):
    outdir = shared_dir(tmp_path)
    _fake_analysis(monkeypatch, [])
    io = {"format": "npy", "fields": {"enabled": True, "precision": "half", "variables": ["u", "omega"]}}
    _run("imhd_turb", _config(outdir, **io))
    if COMM.is_root:
        assert os.path.isdir(os.path.join(outdir, "fields", "fields_000002"))
        assert os.path.exists(os.path.join(outdir, "slices", "slice_000002.npz"))
        assert os.path.exists(os.path.join(outdir, "diagnostics", "spectral_000002.npz"))
        assert not [n for n in os.listdir(os.path.join(outdir, "fields")) if n.endswith(".xmf")]
        got = load_fields(os.path.join(outdir, "fields", "fields_000002"), ["omega"])
        assert sorted(got["fields"]) == ["omega_x", "omega_y", "omega_z"]
        assert got["fields"]["omega_x"].dtype == np.float16
        assert got["fields"]["omega_x"].shape == SHAPE[::-1]


def test_serial_communicator_handles_a_coarse_request():
    grid = SlabGrid3D(*SHAPE, *LENGTHS, comm=Comm.serial(), device="cpu")
    assert SpectralCoarsener(grid, (6, 5, 4)).sub is grid.comm
