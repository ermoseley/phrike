"""End-to-end runs of the distributed driver: problems, output, restart."""

import json
import os
import warnings

import numpy as np
import pytest

from _parallel_util import COMM, fits, gmax, shared_dir, torch
from phrike.parallel import Comm
from phrike.parallel.driver import Simulation
from phrike.parallel.grid import SlabGrid3D
from phrike.parallel.io import load_snapshot, read_meta, read_spectrum
from phrike.parallel.problems import get_problem, list_problems

TWO_PI = 2.0 * np.pi
SHAPE = (24, 20, 16)
OU = {"type": "ou", "kmin": 1.0, "kmax": 2.5, "amplitude": 1.0, "correlation_time": 0.1, "seed": 3}

pytestmark = pytest.mark.skipif(not fits(SHAPE), reason="grid too small for this rank count")


def _config(outdir, **sections):
    Nx, Ny, Nz = SHAPE
    config = {
        "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz, "Lx": TWO_PI, "Ly": TWO_PI, "Lz": TWO_PI},
        "physics": {"gamma": 5.0 / 3.0, "viscosity": 0.02, "resistivity": 0.03,
                    "mhd": {"resistivity": 1e-3, "viscosity": 2e-3}},
        "integration": {"t0": 0.0, "t_end": 0.06, "cfl": 0.3, "scheme": "rk4", "output_interval": 0.03},
        "initial_conditions": {"vrms": 0.3, "kmin": 1.0, "kmax": 4.0, "b_rms": 0.1, "b_kmax": 3.0},
        "io": {"outdir": outdir},
        "monitoring": {"enabled": False},
    }
    for name, block in sections.items():
        config[name] = dict(config.get(name, {}), **block) if isinstance(block, dict) else block
    return config


def _run(problem, config, **kwargs):
    sim = Simulation(problem, config, comm=COMM, backend="cpu", quiet=True, **kwargs)
    summary = sim.run()
    COMM.barrier()
    return sim, summary


def _snapshots(outdir):
    return sorted(os.listdir(os.path.join(outdir, "snapshots")))


@pytest.mark.parametrize("problem", list_problems())
def test_every_problem_runs(tmp_path, problem):
    outdir = shared_dir(tmp_path)
    sections = {}
    if problem in ("orszag_tang3d", "alfven3d"):
        sections["grid"] = {"Lx": 1.0, "Ly": 1.0, "Lz": 1.0}
        sections["integration"] = {"t_end": 0.01, "output_interval": 0.005}
    if problem == "tgv3d":
        sections["initial_conditions"] = {"mach": 0.3}
    sim, summary = _run(problem, _config(outdir, **sections))
    assert summary["steps"] >= 2 and sim.t == pytest.approx(sections.get("integration", {}).get("t_end", 0.06))
    assert sim.system == get_problem(problem).system
    assert _snapshots(outdir) == ["snap_000000", "snap_000001", "snap_000002"]
    meta = read_meta(os.path.join(outdir, "snapshots", "snap_000002"))
    assert meta["t"] == pytest.approx(sim.t) and meta["nprocs"] == COMM.size
    assert torch.isfinite(sim.stepper.physical()).all()
    if COMM.is_root:
        history = np.genfromtxt(os.path.join(outdir, "history.csv"), delimiter=",", names=True)
        assert len(history) == 3 and np.all(np.isfinite(history["kinetic_energy"]))
        with open(os.path.join(outdir, "timing.json")) as handle:
            timing = json.load(handle)
        assert timing["steps"] == summary["steps"] and timing["fft_fields_per_step"] > 0
        spectra = np.load(os.path.join(outdir, "spectra", "spectra_000002.npz"))
        assert spectra["kinetic"].sum() > 0.0
        slices = np.load(os.path.join(outdir, "slices", "slice_000002.npz"))
        assert slices[meta["fields"][0]].shape == (SHAPE[1], SHAPE[0])


def test_fft_count_per_step_is_minimal(tmp_path):
    """14 transforms per stage for incompressible MHD (6 inverse + 8 forward), 20 for compressible."""
    outdir = shared_dir(tmp_path)
    quiet_io = {"snapshots": False, "slices": False, "spectra": False}
    sim, summary = _run("imhd_turb", _config(os.path.join(outdir, "a"), io=quiet_io))
    assert summary["fft_fields_per_step"] == pytest.approx(4 * 14)
    sim, summary = _run("ins_turb", _config(os.path.join(outdir, "b"), io=quiet_io))
    assert summary["fft_fields_per_step"] == pytest.approx(4 * 8)
    sim, summary = _run("mhd_turb3d", _config(os.path.join(outdir, "c"), io=quiet_io))
    assert summary["fft_fields_per_step"] == pytest.approx(4 * 20)
    sim, summary = _run("turb3d", _config(os.path.join(outdir, "d"), io=quiet_io))
    assert summary["fft_fields_per_step"] == pytest.approx(4 * 14)


@pytest.mark.parametrize("problem,forcing", [("imhd_turb", OU), ("mhd_turb3d", OU), ("turb3d", None)])
def test_restart_reproduces_the_uninterrupted_run(tmp_path, problem, forcing):
    outdir = shared_dir(tmp_path)
    extra = {"forcing": dict(forcing)} if forcing else {}
    full, _ = _run(problem, _config(os.path.join(outdir, "full"), **extra))
    _run(problem, _config(os.path.join(outdir, "half"), integration={"t_end": 0.03}, **extra))
    resumed, summary = _run(problem, _config(os.path.join(outdir, "resumed"), **extra),
                            restart=os.path.join(outdir, "half", "snapshots", "snap_000001"))
    assert resumed.t == pytest.approx(full.t, rel=1e-13) and resumed.step_count == full.step_count
    scale = float(full.stepper.uhat.abs().max())
    assert gmax((resumed.stepper.uhat - full.stepper.uhat).abs().max().item()) < 1e-12 * scale
    # The restarted run continues the numbering and does not rewrite the state it started from.
    assert _snapshots(os.path.join(outdir, "resumed")) == ["snap_000002"]


def test_restart_from_checkpoint_keeps_the_output_times(tmp_path):
    outdir = shared_dir(tmp_path)
    config = _config(os.path.join(outdir, "a"), integration={"t_end": 0.05, "checkpoint_interval": 0.02},
                     io={"keep_checkpoints": 1})
    _run("imhd_turb", config)
    checkpoints = sorted(os.listdir(os.path.join(outdir, "a", "checkpoints")))
    assert checkpoints == ["chk_000001"]  # only the newest is kept
    meta = read_meta(os.path.join(outdir, "a", "checkpoints", "chk_000001"))
    assert meta["t"] == pytest.approx(0.04)
    resumed, _ = _run("imhd_turb", _config(os.path.join(outdir, "b"), integration={"t_end": 0.07}),
                      restart=os.path.join(outdir, "a", "checkpoints", "chk_000001"))
    times = [read_meta(os.path.join(outdir, "b", "snapshots", name))["t"]
             for name in _snapshots(os.path.join(outdir, "b"))]
    assert times == pytest.approx([0.06, 0.07])  # outputs stay on t0 + n * output_interval


def test_snapshot_reader_handles_other_rank_counts_and_resolutions(tmp_path):
    outdir = shared_dir(tmp_path)
    sim, _ = _run("imhd_turb", _config(outdir, integration={"t_end": 0.03}))
    path = os.path.join(outdir, "snapshots", "snap_000001")
    d = sim.grid.decomp
    # Same grid, read as if by a single rank.
    serial = SlabGrid3D(*SHAPE, TWO_PI, TWO_PI, TWO_PI, comm=Comm.serial(), device="cpu")
    whole = read_spectrum(path, serial, "imhd")
    assert gmax((whole[:, :, d.ky0:d.ky0 + d.nky] - sim.stepper.uhat).abs().max().item()) == 0.0
    # Finer grid: the old modes keep their values, the new ones start at zero.
    fine = SlabGrid3D(36, 30, 24, TWO_PI, TWO_PI, TWO_PI, comm=Comm.serial(), device="cpu")
    refined = read_spectrum(path, fine, "imhd")
    energy = lambda g, S: float(((S.real ** 2 + S.imag ** 2).sum(dim=0) * g.weight3).sum())
    assert energy(fine, refined) == pytest.approx(energy(serial, whole), rel=1e-13)
    assert refined[:, 2, 3, 1] == pytest.approx(whole[:, 2, 3, 1])
    assert refined[:, -2, -3, 1] == pytest.approx(whole[:, -2, -3, 1])
    # Coarser grid: modes that no longer fit are dropped.
    coarse = SlabGrid3D(12, 12, 12, TWO_PI, TWO_PI, TWO_PI, comm=Comm.serial(), device="cpu")
    reduced = read_spectrum(path, coarse, "imhd")
    assert reduced.shape[1:] == coarse.decomp.spectral_shape
    assert reduced[:, 1, -1, 2] == pytest.approx(whole[:, 1, -1, 2])
    with pytest.raises(ValueError):
        read_spectrum(path, serial, "mhd")
    if COMM.is_root:
        loaded = load_snapshot(path, physical=True)
        assert np.array_equal(loaded["spectrum"], whole.numpy())
        physical = serial.inverse(whole).numpy()
        assert np.abs(loaded["fields"] - physical).max() < 1e-13


def test_refined_restart_runs(tmp_path):
    outdir = shared_dir(tmp_path)
    _run("imhd_turb", _config(os.path.join(outdir, "coarse"), integration={"t_end": 0.03}, forcing=dict(OU)))
    config = _config(os.path.join(outdir, "fine"), grid={"Nx": 36, "Ny": 30, "Nz": 24}, forcing=dict(OU))
    sim, summary = _run("imhd_turb", config,
                        restart=os.path.join(outdir, "coarse", "snapshots", "snap_000001"))
    assert sim.t == pytest.approx(0.06) and summary["steps"] >= 1
    assert sim.grid.Nx == 36 and np.isfinite(sim.stepper.diagnostics()["total_energy"])


def test_configuration_is_checked(tmp_path):
    outdir = shared_dir(tmp_path)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Simulation("imhd_turb", _config(outdir, physics={"viscocity": 1.0}), comm=COMM, backend="cpu", quiet=True)
    assert COMM.rank != 0 or any("physics.viscocity" in str(w.message) for w in caught)
    with pytest.raises(NotImplementedError):
        Simulation("mhd_turb3d", _config(outdir, physics={"mhd": {"hall": {"enabled": True}}}),
                   comm=COMM, backend="cpu", quiet=True)
    with pytest.raises(ValueError):
        Simulation("imhd_turb", _config(outdir, integration={"scheme": "rk45"}), comm=COMM, backend="cpu", quiet=True)
    with pytest.raises(ValueError):
        Simulation("imhd_turb", _config(outdir, forcing={"type": "white"}), comm=COMM, backend="cpu", quiet=True)
    with pytest.raises(ValueError):
        Simulation("nonexistent", _config(outdir), comm=COMM, backend="cpu", quiet=True)
    with pytest.raises(ValueError):
        Simulation("turb3d", _config(outdir, grid={"dealias": False}), comm=COMM, backend="cpu", quiet=True)


def test_magnetic_prandtl_and_max_steps(tmp_path):
    outdir = shared_dir(tmp_path)
    config = _config(outdir, integration={"max_steps": 3, "t_end": 10.0, "output_interval": 5.0})
    del config["physics"]["resistivity"]
    del config["physics"]["mhd"]
    config["physics"]["magnetic_prandtl"] = 4.0
    sim, summary = _run("imhd_turb", config)
    assert summary["steps"] == 3 and sim.eta == pytest.approx(0.02 / 4.0)
    assert sim.stepper.diagnostics()["magnetic_prandtl"] == pytest.approx(4.0)
    # Stopping early leaves the output series alone and writes a restart point instead.
    assert _snapshots(outdir) == ["snap_000000"]
    meta = read_meta(os.path.join(outdir, "checkpoints", "chk_000000"))
    assert meta["step"] == 3 and meta["t"] == pytest.approx(sim.t)
    resumed, more = _run("imhd_turb", dict(config, integration=dict(config["integration"], max_steps=2)),
                         restart=os.path.join(outdir, "checkpoints", "chk_000000"))
    assert resumed.step_count == 5 and more["steps"] == 2


def test_wall_clock_limit_stops_with_a_checkpoint(tmp_path):
    outdir = shared_dir(tmp_path)
    sim, summary = _run("imhd_turb", _config(outdir, integration={"t_end": 1e9, "max_wall_minutes": 0.0}))
    assert summary["steps"] == 0 and sim.t == 0.0
    assert read_meta(os.path.join(outdir, "checkpoints", "chk_000000"))["step"] == 0


def test_cli_lists_problems(capsys):
    if COMM.size > 1:
        pytest.skip("serial only")
    from phrike.parallel.__main__ import main

    assert main(["--list"]) == 0
    assert "imhd_turb" in capsys.readouterr().out
