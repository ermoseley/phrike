"""Run the distributed-solver tests, and a restart across rank counts, under MPI.

The other ``test_parallel_*`` files are rank-aware: launched under ``mpirun``
every rank runs them and checks its own slab against a whole-grid reference.
This file launches those runs.  It is skipped when ``mpirun`` or ``mpi4py``
is missing, and inside an MPI launch.
"""

import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUITE = ["tests/test_parallel_fft.py", "tests/test_parallel_compressible.py",
         "tests/test_parallel_incompressible.py", "tests/test_parallel_driver.py",
         "tests/test_parallel_triton_emulated.py"]


def _launcher():
    from phrike.parallel.comm import launched_under_mpi

    if launched_under_mpi():
        return None
    exe = shutil.which("mpirun") or shutil.which("mpiexec")
    if exe is None:
        return None
    try:
        version = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30).stdout
        base = [exe] + (["--oversubscribe"] if "Open MPI" in version or "OpenRTE" in version else [])
        probe = subprocess.run(base + ["-n", "2", sys.executable, "-c",
                                       "from mpi4py import MPI; assert MPI.COMM_WORLD.Get_size() == 2"],
                               capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError):
        return None
    return base if probe.returncode == 0 else None


LAUNCHER = _launcher()
pytestmark = pytest.mark.skipif(LAUNCHER is None, reason="needs mpirun and mpi4py, outside an MPI launch")


def _env():
    env = dict(os.environ, PYTHONPATH=ROOT + os.pathsep + os.environ.get("PYTHONPATH", ""),
               OMP_NUM_THREADS="1", PHRIKE_MPI="1")
    return env


@pytest.mark.parametrize("ranks", [2, 3, 5, 7])
def test_distributed_suite_under_mpi(ranks):
    done = subprocess.run(LAUNCHER + ["-n", str(ranks), sys.executable, "-m", "pytest", "-q", "-x",
                                      "-p", "no:cacheprovider"] + SUITE,
                          cwd=ROOT, env=_env(), capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]


def test_distributed_suite_under_mpi_nccl_transport():
    """The same suite with the transpose on torch.distributed (Gloo on the CPU)."""
    env = dict(_env(), PHRIKE_MPI_TRANSPOSE="nccl")
    done = subprocess.run(LAUNCHER + ["-n", "3", sys.executable, "-m", "pytest", "-q", "-x",
                                      "-p", "no:cacheprovider"] + SUITE,
                          cwd=ROOT, env=env, capture_output=True, text=True, timeout=1800)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-2000:]


def _write_config(path, outdir, t_end):
    config = {
        "problem": "imhd_turb",
        "grid": {"Nx": 24, "Ny": 20, "Nz": 16, "Lx": 6.283185307179586, "Ly": 6.283185307179586,
                 "Lz": 6.283185307179586, "precision": "double"},
        "physics": {"viscosity": 0.02, "resistivity": 0.01},
        "integration": {"t0": 0.0, "t_end": t_end, "cfl": 0.5, "scheme": "rk4", "output_interval": 0.04},
        "forcing": {"type": "ou", "kmin": 1.0, "kmax": 2.5, "amplitude": 1.0, "correlation_time": 0.1, "seed": 5},
        "initial_conditions": {"vrms": 1.0, "kmin": 1.0, "kmax": 4.0, "b_rms": 0.1, "b_kmax": 3.0},
        "io": {"outdir": outdir}, "monitoring": {"enabled": False},
    }
    with open(path, "w") as handle:
        json.dump(config, handle)  # JSON is YAML


def _cli(ranks, config, *extra):
    command = [sys.executable, "-m", "phrike.parallel", "imhd_turb", "--config", config, "--backend", "cpu"]
    if ranks > 1:
        command = LAUNCHER + ["-n", str(ranks)] + command
        env = _env()
    else:
        env = dict(_env(), PHRIKE_MPI="0")
    done = subprocess.run(command + list(extra), cwd=ROOT, env=env, capture_output=True, text=True, timeout=900)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return done.stdout


def test_cli_restart_across_rank_counts(tmp_path):
    """Serial run == 3 ranks to t/2, then restarted on 2 ranks, to round-off."""
    from phrike.parallel.io import load_snapshot

    full, half, resumed = (str(tmp_path / name) for name in ("full", "half", "resumed"))
    for name, outdir, t_end in (("full", full, 0.08), ("half", half, 0.04), ("resumed", resumed, 0.08)):
        _write_config(str(tmp_path / f"{name}.yaml"), outdir, t_end)
    _cli(1, str(tmp_path / "full.yaml"))
    out = _cli(3, str(tmp_path / "half.yaml"))
    assert "3 ranks" in out
    out = _cli(2, str(tmp_path / "resumed.yaml"), "--restart", os.path.join(half, "snapshots", "snap_000001"))
    assert "restarted from" in out and "2 ranks" in out
    reference = load_snapshot(os.path.join(full, "snapshots", "snap_000002"))
    restarted = load_snapshot(os.path.join(resumed, "snapshots", "snap_000002"))
    assert restarted["meta"]["nprocs"] == 2 and reference["meta"]["nprocs"] == 1
    assert restarted["t"] == pytest.approx(reference["t"], rel=1e-13)
    scale = np.abs(reference["spectrum"]).max()
    assert np.abs(restarted["spectrum"] - reference["spectrum"]).max() < 1e-12 * scale
    a = np.atleast_1d(np.genfromtxt(os.path.join(full, "history.csv"), delimiter=",", names=True))
    b = np.atleast_1d(np.genfromtxt(os.path.join(resumed, "history.csv"), delimiter=",", names=True))
    assert a["kinetic_energy"][-1] == pytest.approx(b["kinetic_energy"][-1], rel=1e-12)
