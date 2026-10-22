"""Regression tests for problem orchestration, the CLI, snapshots, and tracers."""

import inspect
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from phrike import __version__, run_simulation
from phrike.equations import EulerEquations1D, EulerEquations2D
from phrike.grid import Grid1D
from phrike.io import (
    grid_boundary_condition,
    load_checkpoint,
    load_config,
    save_solution_snapshot,
)
from phrike.problems.base import BaseProblem, _MOMENTUM_SCALE_NOTE
from phrike.problems.problem_list.acoustic1d import Acoustic1DProblem
from phrike.problems.registry import ProblemRegistry
from phrike.tracers import FourierTracers1D, FourierTracers3D


def _config(tmp_path, **overrides):
    config = {
        "problem": "acoustic1d",
        "physics": {"gamma": 1.4},
        "integration": {
            "t0": 0.0,
            "t_end": 0.0,
            "cfl": 0.5,
            "output_interval": 1.0,
        },
        "grid": {
            "N": 8,
            "Lx": 1.0,
            "basis": "fourier",
            "precision": "double",
        },
        "io": {"outdir": str(tmp_path / "out")},
        "monitoring": {"enabled": False, "step_interval": 1},
    }
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key].update(value)
        else:
            config[key] = value
    return config


def _problem(tmp_path, **overrides):
    return Acoustic1DProblem(config=_config(tmp_path, **overrides))


def _state(n=8, velocity=0.0, pressure=1.0, gamma=1.4):
    equations = EulerEquations1D(gamma=gamma)
    rho = np.ones(n)
    velocity = np.full(n, velocity)
    pressure = np.full(n, pressure)
    return equations, equations.conservative(rho, velocity, pressure)


def _solver(equations, state_holder, grid=None):
    return SimpleNamespace(
        equations=equations,
        grid=grid if grid is not None else SimpleNamespace(),
        t=0.0,
        state=state_holder,
    )


class _MutableState:
    def __init__(self, equations, velocity):
        self.equations = equations
        self.gamma = equations.gamma
        self.velocity = velocity
        self.state = equations.conservative(
            np.ones(velocity.size), velocity, np.ones(velocity.size)
        )

    def primitive(self, _state):
        return self.equations.primitive(self.state)

    def conserved_quantities(self, _state):
        return self.equations.conserved_quantities(self.state)


def test_new_snapshot_is_json_and_records_identity(tmp_path):
    grid = Grid1D(N=8, Lx=1.0, basis="fourier", backend="numpy", dealias=False)
    equations, state = _state()
    provenance = {"config": {"problem": "acoustic1d", "grid": {"N": 8}}, "problem": "acoustic1d"}
    path = save_solution_snapshot(
        str(tmp_path), 0.0, state, grid, equations, provenance=provenance
    )

    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with np.load(path, allow_pickle=False) as raw:
            meta_raw = raw["meta"]
            meta = json.loads(str(meta_raw.item()))
    assert not [item for item in caught if issubclass(item.category, UserWarning)]
    assert meta["basis"] == "fourier"
    assert meta["dimension"] == 1
    assert meta["gamma"] == pytest.approx(1.4)
    assert meta["Lx"] == pytest.approx(1.0)
    assert meta["N"] == 8
    assert meta["phrike_version"] == __version__
    assert meta["problem"] == "acoustic1d"
    assert meta["config"]["grid"]["N"] == 8
    assert "created" in meta

    loaded = load_checkpoint(path)
    assert loaded["meta"]["basis"] == "fourier"
    problem = Acoustic1DProblem(
        config=_config(tmp_path), restart_from=path
    )
    with warnings_as_errors():
        problem.validate_restart_data(problem.create_grid(backend="numpy"), equations)


def test_restart_rejects_domain_basis_and_gamma_mismatches(tmp_path):
    grid = Grid1D(N=8, Lx=1.0, basis="fourier", backend="numpy", dealias=False)
    equations, state = _state()
    path = save_solution_snapshot(str(tmp_path), 0.0, state, grid, equations)

    longer = Acoustic1DProblem(
        config=_config(tmp_path, grid={"Lx": 2.0}), restart_from=path
    )
    with pytest.raises(ValueError, match="Lx"):
        longer.validate_restart_data(longer.create_grid(backend="numpy"), equations)

    legendre = Acoustic1DProblem(
        config=_config(tmp_path, grid={"basis": "legendre", "bc": "wall"}),
        restart_from=path,
    )
    with pytest.raises(ValueError, match="basis"):
        legendre.validate_restart_data(legendre.create_grid(backend="numpy"), equations)

    hot = Acoustic1DProblem(
        config=_config(tmp_path, physics={"gamma": 5.0 / 3.0}), restart_from=path
    )
    with pytest.raises(ValueError, match="gamma"):
        hot.validate_restart_data(hot.create_grid(backend="numpy"), hot.create_equations())


def test_boundary_aliases_are_stored_in_normalized_form():
    wall = Grid1D(
        N=8, Lx=1.0, basis="legendre", bc="reflective", backend="numpy", dealias=False
    )
    fixed = Grid1D(
        N=8, Lx=1.0, basis="legendre", bc="dirichlet", backend="numpy", dealias=False
    )
    assert grid_boundary_condition(wall) == "wall"
    assert grid_boundary_condition(fixed) == "fixed"


def test_legacy_pickled_snapshot_warns_and_still_loads(tmp_path):
    path = tmp_path / "legacy.npz"
    state = np.ones((3, 8))
    np.savez(
        path,
        t=np.array(0.1),
        U=state,
        meta={"N": 8, "Lx": 1.0, "gamma": 1.4},
    )
    with pytest.warns(UserWarning, match="pickled|legacy"):
        loaded = load_checkpoint(str(path))
    assert loaded["meta"]["Lx"] == pytest.approx(1.0)
    assert loaded["meta"]["gamma"] == pytest.approx(1.4)
    assert isinstance(loaded["meta"], dict)

    problem = _problem(tmp_path)
    problem.restart_data = loaded
    with pytest.warns(UserWarning, match="missing"):
        problem.validate_restart_data(problem.create_grid(backend="numpy"), EulerEquations1D(gamma=1.4))

    loaded["meta"]["Lx"] = 2.0
    with pytest.raises(ValueError, match="Lx"):
        problem.validate_restart_data(problem.create_grid(backend="numpy"), EulerEquations1D(gamma=1.4))


def test_git_lookup_failure_does_not_block_the_snapshot(tmp_path, monkeypatch):
    import phrike.io as io

    previous = dict(io._git_hash_cache)
    io._git_hash_cache["done"] = False
    io._git_hash_cache["value"] = None

    def fail(*_args, **_kwargs):
        raise OSError("git is not available")

    monkeypatch.setattr(io.subprocess, "run", fail)
    try:
        grid = Grid1D(N=8, Lx=1.0, basis="fourier", backend="numpy", dealias=False)
        equations, state = _state()
        path = save_solution_snapshot(str(tmp_path), 0.0, state, grid, equations)
        meta = load_checkpoint(path)["meta"]
        assert "git_hash" not in meta
        assert meta["phrike_version"] == __version__
    finally:
        io._git_hash_cache.clear()
        io._git_hash_cache.update(previous)


def test_pressure_statistics_use_pressure_not_sound_speed(tmp_path):
    problem = _problem(tmp_path)
    equations, state = _state()
    stats = problem.compute_density_stats(SimpleNamespace(equations=equations), state)
    assert stats["p_max"] == pytest.approx(1.0)
    assert stats["p_min"] == pytest.approx(1.0)
    sound = float(np.sqrt(1.4))
    assert stats["p_max"] < sound - 0.05

    equations2d = EulerEquations2D(gamma=1.4)
    rho = np.ones((2, 2))
    ux = np.zeros((2, 2))
    uy = np.full((2, 2), 3.0)
    pressure = np.full((2, 2), 1.25)
    state2d = equations2d.conservative(rho, ux, uy, pressure)
    stats2d = problem.compute_density_stats(
        SimpleNamespace(equations=equations2d), state2d
    )
    assert stats2d["p_max"] == pytest.approx(1.25)
    assert stats2d["p_min"] == pytest.approx(1.25)


def test_torch_pressure_statistics_include_pressure(tmp_path):
    torch = pytest.importorskip("torch")
    problem = _problem(tmp_path)
    equations = EulerEquations1D(gamma=1.4)
    rho = torch.ones(8, dtype=torch.float32)
    velocity = torch.zeros(8, dtype=torch.float32)
    pressure = torch.ones(8, dtype=torch.float32)
    state = equations.conservative(rho, velocity, pressure)
    stats = problem.compute_density_stats(SimpleNamespace(equations=equations), state)
    assert stats["p_max"] == pytest.approx(1.0)
    assert stats["p_min"] == pytest.approx(1.0)


def test_momentum_error_uses_a_physical_scale(tmp_path, capsys):
    problem = _problem(
        tmp_path,
        monitoring={
            "enabled": True,
            "step_interval": 1,
            "include_conservation": True,
            "include_time": False,
            "include_timestep": False,
            "include_velocity_stats": False,
            "include_density_stats": False,
        },
    )
    equations = EulerEquations1D(gamma=1.4)
    holder = _MutableState(equations, np.full(32, 1e-16))
    solver = SimpleNamespace(equations=holder, grid=SimpleNamespace(), t=0.0)
    problem.initialize_monitoring(solver, holder.state)
    holder.velocity = np.full(32, 1e-16 + 1e-6)
    holder.state = equations.conservative(
        np.ones(32), holder.velocity, np.ones(32)
    )
    errors = problem.compute_conservation_errors(solver, holder.state)
    expected = 3.2e-5 / (32.0 * np.sqrt(1.4))
    assert errors["momentum_error"] == pytest.approx(expected, rel=1e-8)
    assert errors["momentum_error"] < 1e-4
    assert errors["mass_error"] == pytest.approx(0.0, abs=1e-12)

    problem.output_monitoring_info(solver, holder.state, 1, 1e-3)
    printed = capsys.readouterr().out
    assert "momentum_error:" in printed
    assert _MOMENTUM_SCALE_NOTE in printed


def test_unknown_parsed_keys_warn_and_problem_keys_do_not(tmp_path):
    config = _config(tmp_path)
    config["integration"]["output_intervall"] = 0.2
    with pytest.warns(UserWarning, match=r"integration\.output_intervall"):
        Acoustic1DProblem(config=config)

    clean = _config(tmp_path)
    clean["initial_conditions"] = {"not_a_real_key": 1}
    clean["visualization"] = {"also_ignored": True}
    clean["output"] = {"format": "npz"}
    clean["boundary_conditions"] = {"x": "periodic"}
    with warnings_as_errors():
        Acoustic1DProblem(config=clean)

    warned = _config(tmp_path)
    warned["physics"]["mhd"] = {"enabled": False, "resistivity": 0.0}
    with pytest.warns(UserWarning, match=r"physics\.mhd\.enabled"):
        Acoustic1DProblem(config=warned)


def test_step_interval_and_conflicting_adaptive_blocks_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="step_interval"):
        _problem(tmp_path, monitoring={"step_interval": 0})
    with pytest.raises(ValueError, match="step_interval"):
        _problem(tmp_path, monitoring={"step_interval": 1.5})
    with pytest.raises(ValueError, match="integration.adaptive"):
        _problem(
            tmp_path,
            integration={"adaptive": {"enabled": False}},
            adaptive={"enabled": False},
        )


def test_empty_config_file_is_rejected(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_config(str(path))


def test_cli_video_flags_do_not_silently_override_yaml(tmp_path, monkeypatch):
    path = _write_video_config(tmp_path, enabled=False, quality="low")
    created = _install_cli_problem(monkeypatch)
    with pytest.warns(UserWarning, match="without --video"):
        _run_cli(monkeypatch, path, "--video-quality", "high")
    problem = created["problem"]
    assert problem.config["video"]["quality"] == "low"
    assert problem.generate_video is False

    path_on = _write_video_config(tmp_path, enabled=True, quality="low")
    created_on = _install_cli_problem(monkeypatch)
    with pytest.warns(UserWarning, match="without --video"):
        _run_cli(monkeypatch, path_on, "--video-quality", "high")
    problem_on = created_on["problem"]
    assert problem_on.config["video"]["quality"] == "high"
    assert problem_on.generate_video is True

    path_flag = _write_video_config(tmp_path, enabled=False, quality="low")
    created_flag = _install_cli_problem(monkeypatch)
    with warnings_as_errors():
        _run_cli(
            monkeypatch,
            path_flag,
            "--video",
            "--video-quality",
            "high",
            "--video-codec",
            "libx264",
        )
    problem_flag = created_flag["problem"]
    assert problem_flag.config["video"]["quality"] == "high"
    assert problem_flag.config["video"]["codec"] == "libx264"
    assert problem_flag.generate_video is True

    path_yaml = _write_video_config(tmp_path, enabled=True, quality="low")
    created_yaml = _install_cli_problem(monkeypatch)
    with warnings_as_errors():
        _run_cli(monkeypatch, path_yaml, "--video")
    problem_yaml = created_yaml["problem"]
    assert problem_yaml.config["video"]["quality"] == "low"
    assert problem_yaml.generate_video is True


def test_cli_reports_the_grid_precision(tmp_path, monkeypatch, capsys):
    path = tmp_path / "precision.yaml"
    path.write_text(
        "physics:\n  gamma: 1.4\nintegration:\n  t0: 0.0\n  t_end: 0.0\n  cfl: 0.5\n"
        "grid:\n  N: 8\n  Lx: 1.0\n  precision: double\n"
        f"io:\n  outdir: {tmp_path / 'out'}\n"
    )
    created = _install_cli_problem(monkeypatch, grid_precision="single")
    _run_cli(monkeypatch, path, "--verbose", "--precision", "double", "--backend", "numpy")
    problem = created["problem"]
    printed = capsys.readouterr().out
    assert "Precision: single" in printed
    assert "Precision: double" not in printed
    assert problem.precision == "double"
    assert problem.run_backend == "numpy"


def test_video_encoding_honours_crf_and_does_not_clobber_frames(tmp_path, monkeypatch, capsys):
    problem = _problem(
        tmp_path,
        video={"enabled": True, "codec": "libx264", "crf": 35, "quality": "high", "fps": 10},
    )
    frames = Path(problem.outdir) / "frames"
    frames.mkdir()
    (frames / "frame_0.100.png").write_bytes(b"time-name")
    (frames / "frame_00000000.png").write_bytes(b"index-name")
    count = problem._renumber_frames(str(frames))
    assert count == 2
    contents = {path.read_bytes() for path in frames.glob("frame_*.png")}
    assert contents == {b"time-name", b"index-name"}
    assert not (frames.parent / "frames.renumbering").exists()

    with pytest.warns(UserWarning, match="existing frame"):
        frames = Path(problem._prepare_frames_directory())
    assert list(frames.glob("frame_*.png")) == []
    (frames / "frame_00000000.png").write_bytes(b"frame")

    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(list(command))
        return SimpleNamespace(returncode=0, stdout="ffmpeg", stderr="")

    monkeypatch.setattr("phrike.problems.base.subprocess.run", fake_run)
    video = problem.generate_video(str(frames), "acoustic1d")
    assert video.endswith("acoustic1d.mp4")
    encoded = " ".join(" ".join(call) for call in calls)
    assert "frame_%08d.png" in encoded
    assert "%05d" not in encoded
    assert "-crf 35" in encoded
    assert "libx264" in encoded

    def missing(command, *args, **kwargs):
        raise FileNotFoundError(command[0])

    monkeypatch.setattr("phrike.problems.base.subprocess.run", missing)
    with pytest.warns(UserWarning, match="ffmpeg was not found"):
        assert problem.generate_video(str(frames), "acoustic1d") is None

    def fail(command, *args, **kwargs):
        if "-version" in command:
            return SimpleNamespace(returncode=0, stdout="ffmpeg", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="encoder failed")

    monkeypatch.setattr("phrike.problems.base.subprocess.run", fail)
    (frames / "frame_00000000.png").write_bytes(b"frame")
    with pytest.warns(UserWarning) as recorded:
        assert problem.generate_video(str(frames), "acoustic1d") is None
    messages = " ".join(str(item.message) for item in recorded)
    assert "frame_%08d.png" in messages
    assert "libx264" in messages
    manual = capsys.readouterr().out
    assert "frame_%08d.png" in manual
    assert "%05d" not in manual


def test_tracers_require_fourier_grids_and_keep_float64_positions(tmp_path):
    problem = _problem(tmp_path, tracers={"enabled": True, "num": 4})
    legendre = SimpleNamespace(N=8, Lx=1.0, _basis_name="legendre")
    with pytest.raises(ValueError, match="Fourier"):
        problem._create_tracers_from_config(legendre, None, None)

    mixed = SimpleNamespace(
        Nx=4, Ny=4, Lx=1.0, Ly=1.0, basis_x="fourier", basis_y="legendre"
    )
    with pytest.raises(ValueError, match="Fourier"):
        problem._create_tracers_from_config(mixed, None, None)

    cubic = SimpleNamespace(Nx=2, Ny=2, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0, basis="legendre")
    with pytest.raises(ValueError, match="Fourier"):
        problem._create_tracers_from_config(cubic, None, None)

    problem.config["tracers"] = {"enabled": True, "num": 8, "layout": "uniform", "mass": 1.0}
    fourier3d = SimpleNamespace(Nx=2, Ny=2, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0)
    tracers = problem._create_tracers_from_config(fourier3d, None, None)
    assert tracers.x.dtype == np.float64
    assert tracers.x.size == 8

    particles = FourierTracers3D(np.array([0.1]), np.array([0.2]), np.array([0.3]))
    with pytest.raises(ValueError, match="Fourier"):
        particles.step(SimpleNamespace(basis="legendre", Lx=1.0, Ly=1.0, Lz=1.0), None, None, 0.1)

    seen = {}

    class Grid:
        _basis_name = "fourier"
        Lx = 1.0

        def evaluate_fourier_at_points_1d(self, velocity, x):
            seen["dtype"] = np.asarray(x).dtype
            return np.zeros(np.asarray(x).shape, dtype=np.float32)

    class Equations:
        def primitive(self, _state):
            velocity = np.zeros(4, dtype=np.float32)
            return velocity, velocity, velocity, velocity

    tracers1d = FourierTracers1D(np.array([0.2, 0.4], dtype=np.float32))
    tracers1d._ensure_backend(np.zeros(4, dtype=np.float32))
    tracers1d.step(Grid(), Equations(), np.zeros((3, 4), dtype=np.float32), 0.01)
    assert tracers1d.x.dtype == np.float64
    assert seen["dtype"] == np.float32


def test_registry_uses_the_registered_problem_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    caller = _config(tmp_path)
    del caller["io"]
    caller["problem"] = {"name": "something_else", "description": "a wave"}
    original = {"name": "something_else", "description": "a wave"}
    with warnings_as_errors():
        created = ProblemRegistry.create_problem("acoustic1d", config=caller)
    assert caller["problem"] == original
    assert created.config["problem"] == "acoustic1d"
    assert Path(created.outdir).name == "acoustic1d"
    assert "{" not in created.outdir

    direct_config = _config(tmp_path)
    del direct_config["io"]
    direct_config["problem"] = {"name": "from_mapping", "description": "kept"}
    with warnings_as_errors():
        direct = Acoustic1DProblem(config=direct_config)
    assert Path(direct.outdir).name == "from_mapping"


def test_run_simulation_matches_the_cli_video_default_and_accepts_restart():
    assert inspect.signature(run_simulation).parameters["generate_video"].default is False
    assert "restart_from" in inspect.signature(run_simulation).parameters
    assert inspect.signature(BaseProblem.run).parameters["generate_video"].default is False


def test_run_records_effective_precision_and_provenance(tmp_path):
    problem = _problem(tmp_path)
    problem.precision = "double"
    grid = SimpleNamespace(precision="single", _use_torch=False)
    problem.create_grid = lambda *args, **kwargs: grid
    problem.create_equations = lambda: SimpleNamespace(gamma=1.4)
    problem.create_initial_conditions = lambda _grid: np.ones(3)
    problem.create_final_visualization = lambda _solver: None

    class Solver:
        def __init__(self, **kwargs):
            self.grid = kwargs["grid"]
            self.equations = kwargs["equations"]
            self.t = 0.0
            self.U = np.ones(3)

        def run(self, _state, **_kwargs):
            return []

    problem.get_solver_class = lambda: Solver
    solver, history = problem.run(backend="numpy", generate_video=False)
    assert solver is not None and history == []
    assert problem.precision == "single"
    assert grid._phrike_provenance["problem"] == "acoustic1d"
    assert grid._phrike_provenance["config"]["grid"]["N"] == 8


def test_clear_numba_cache_stays_inside_the_package(tmp_path, monkeypatch):
    root = tmp_path / "phrike"
    cache = root / "solver" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "kernel.nbi").write_text("cached")
    decoy = tmp_path / "home" / ".numba_cache"
    decoy.mkdir(parents=True)
    (decoy / "keep.txt").write_text("keep")

    BaseProblem.clear_numba_cache(root=root)
    assert not cache.exists()
    assert (decoy / "keep.txt").read_text() == "keep"

    cache.mkdir(parents=True)
    (cache / "kernel.nbi").write_text("cached")

    def fail(_path):
        raise OSError("busy")

    monkeypatch.setattr("phrike.problems.base.shutil.rmtree", fail)
    with pytest.warns(UserWarning, match="Could not remove"):
        BaseProblem.clear_numba_cache(root=root)
    assert (cache / "kernel.nbi").exists()
    assert (decoy / "keep.txt").read_text() == "keep"

    with pytest.warns(UserWarning, match="does not exist"):
        BaseProblem.clear_numba_cache(root=tmp_path / "missing")


def test_runtime_threads_are_read_from_the_config_before_numpy(tmp_path):
    from phrike.cli import _BLAS_THREAD_VARS, _apply_runtime_threads_from_argv

    path = tmp_path / "threads.yaml"
    path.write_text("runtime:\n  num_threads: 2\n")
    previous = {name: os.environ.get(name) for name in _BLAS_THREAD_VARS}
    try:
        for name in _BLAS_THREAD_VARS:
            os.environ.pop(name, None)
        _apply_runtime_threads_from_argv(["phrike", "--config", str(path), "acoustic1d"])
        for name in _BLAS_THREAD_VARS:
            assert os.environ[name] == "2"
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_tracer_density_mean_panel_is_labeled_as_a_mean(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    from phrike.tracer_density import _plot_tracer_density_3d_simple

    labels = []

    def savefig(self, *args, **kwargs):
        for axis in self.axes:
            labels.append(axis.get_ylabel())

    monkeypatch.setattr(plt.Figure, "savefig", savefig)
    rho = np.ones((2, 2, 2))
    _plot_tracer_density_3d_simple(
        rho, (0.0, 1.0, 0.0, 1.0, 0.0, 1.0), outpath=str(tmp_path / "density.png"), log=False
    )
    assert "Mean" in labels
    assert "Projection" not in labels


def test_three_dimensional_smoothing_uses_all_axes(tmp_path, capsys):
    pytest.importorskip("scipy")
    problem = _problem(
        tmp_path,
        initial_conditions_smoothing={
            "enabled": True,
            "kernel_size": 0.3,
            "mode": "wrap",
            "variables": ["density"],
            "diagnostic_output": True,
        },
    )
    grid = SimpleNamespace(Nx=8, Ny=4, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0, N=99)
    rho = np.zeros(8)
    rho[4] = 1.0
    problem.apply_initial_conditions_smoothing(rho, np.zeros(8), np.ones(8), grid)
    printed = capsys.readouterr().out
    assert "sigma=0.53" in printed

    grid1d = SimpleNamespace(N=8, Lx=1.0)
    problem.ic_smoothing_config["kernel_size"] = 0.5
    problem.apply_initial_conditions_smoothing(rho, np.zeros(8), np.ones(8), grid1d)
    assert "sigma=4.00" in capsys.readouterr().out


def test_restart_from_is_forwarded(monkeypatch):
    seen = {}

    def create_problem(name, config_path=None, config=None, restart_from=None):
        seen["name"] = name
        seen["restart_from"] = restart_from
        raise RuntimeError("stop before the run")

    monkeypatch.setattr(ProblemRegistry, "create_problem", staticmethod(create_problem))
    with pytest.raises(RuntimeError, match="stop"):
        run_simulation(
            "acoustic1d",
            config={"physics": {"gamma": 1.4}},
            restart_from="snapshot.npz",
            generate_video=False,
        )
    assert seen == {"name": "acoustic1d", "restart_from": "snapshot.npz"}


class warnings_as_errors:
    """Raise if a UserWarning is emitted inside the block."""

    def __enter__(self):
        import warnings

        self._warnings = warnings
        self._previous = warnings.filters[:]
        warnings.simplefilter("error", UserWarning)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._warnings.filters[:] = self._previous
        return False


def _write_video_config(tmp_path, enabled, quality):
    path = tmp_path / f"video-{enabled}-{quality}.yaml"
    path.write_text(
        "physics:\n  gamma: 1.4\n"
        "integration:\n  t0: 0.0\n  t_end: 0.0\n  cfl: 0.5\n"
        "grid:\n  N: 8\n  Lx: 1.0\n"
        f"io:\n  outdir: {tmp_path / 'cli-out'}\n"
        "video:\n"
        f"  enabled: {str(enabled).lower()}\n"
        f"  quality: {quality}\n"
        "  codec: mpeg4\n"
    )
    return path


def _install_cli_problem(monkeypatch, grid_precision="double"):
    holder = {}

    def create_problem(name, config_path=None, config=None, restart_from=None):
        del name, config, restart_from
        loaded = load_config(config_path)
        problem = SimpleNamespace(
            config=loaded,
            outdir=loaded["io"]["outdir"],
            precision=loaded["grid"].get("precision", "double"),
        )

        def run(backend, device, generate_video, debug):
            problem.generate_video = generate_video
            problem.run_backend = backend
            problem.run_device = device
            problem.run_debug = debug
            grid = SimpleNamespace(precision=grid_precision)
            return SimpleNamespace(t=0.0, grid=grid), []

        problem.run = run
        holder["problem"] = problem
        return problem

    monkeypatch.setattr(ProblemRegistry, "create_problem", staticmethod(create_problem))
    return holder


def _run_cli(monkeypatch, config_path, *extra):
    monkeypatch.setattr(
        sys,
        "argv",
        ["phrike", "acoustic1d", "--config", str(config_path), *extra],
    )
    from phrike.cli import main

    main()
