"""Regression tests for the MHD problems, their configs and the Alfven script."""

import importlib.util
import json
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from phrike.grid import Grid3D
from phrike.problems.problem_list.adsi2d import (
    ADSI2DProblem,
    NoGrowingModeError,
    adsi_growth_summary,
    adsi_reference,
)
from phrike.problems.problem_list.mhd_turb3d import banded_random_b_3d
from phrike.problems.registry import ProblemRegistry

ROOT = Path(__file__).resolve().parents[1]
MHD_CONFIGS = ["alfven1d", "alfven3d", "brio_wu1d", "mhd_turb3d", "adsi2d"]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config(name, tmp_path):
    with open(ROOT / "configs" / f"{name}.yaml") as stream:
        config = yaml.safe_load(stream)
    config["io"] = {"outdir": str(tmp_path)}
    return config


@pytest.mark.parametrize("name", MHD_CONFIGS)
def test_mhd_configs_construct_without_unknown_key_warnings(name, tmp_path):
    config = _config(name, tmp_path)
    assert isinstance(config["problem"], str)
    assert "num_threads" not in config.get("runtime", {})
    assert "fft_workers" not in config["grid"]
    assert config.get("video", {}).get("codec") in (None, "libx264")
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        problem = ProblemRegistry.get_problem_class(name)(config=config)
    assert problem.config["problem"] == name


def _tiny_adsi(tmp_path, ambipolar):
    config = _config("adsi2d", tmp_path)
    config["grid"].update(Nx=16, Ny=33, precision="double")
    config["integration"].update(t_end=0.04, output_interval=0.01, cfl=0.65)
    config["initial_conditions"].update(reference_points=33, seed_amplitude=1e-3)
    config["physics"]["mhd"]["ambipolar"]["enabled"] = ambipolar
    config["monitoring"]["enabled"] = False
    return ADSI2DProblem(config=config)


def test_ideal_adsi_control_reports_envelope_and_no_growth_claim(tmp_path):
    problem = _tiny_adsi(tmp_path, ambipolar=False)
    problem.run(backend="numpy")
    summary = json.loads((tmp_path / "adsi_growth.json").read_text())
    assert summary["ideal_control"] is True
    assert summary["growth_fit"] is None
    assert summary["reference_growth"] is None
    assert summary["samples"] >= 4
    assert summary["relative_amplitude_min"] <= 1.0 <= summary["relative_amplitude_max"]
    assert "no growing mode is expected" in summary["message"]


def test_ideal_limit_reference_fails_with_explanation(tmp_path):
    with pytest.raises(NoGrowingModeError, match="does not grow in ideal MHD"):
        adsi_reference(33, alpha=1e-6)
    config = _config("adsi2d", tmp_path)
    config["grid"].update(Nx=8, Ny=17, precision="double")
    config["initial_conditions"].update(reference_points=33, reference_alpha=1e-6)
    problem = ADSI2DProblem(config=config)
    grid = problem.create_grid(backend="numpy")
    with pytest.raises(ValueError, match="no growing eigenmode"):
        problem.create_initial_conditions(grid)


def test_growth_rate_is_reported_only_for_an_exponential_phase():
    t = np.linspace(0.0, 1.0, 21)
    clean = adsi_growth_summary(t, 2.0 * np.exp(0.45 * t), reference_growth=0.45)
    assert clean["growth_fit"]["rate"] == pytest.approx(0.45, rel=1e-9)
    # Oscillating transient (dip then recovery): an envelope, not an exponential.
    rejected = adsi_growth_summary(t, 1.5 + np.cos(2.0 * np.pi * t / 0.8),
                                   reference_growth=0.45)
    assert rejected["growth_fit"] is None
    assert "envelope only" in rejected["message"]
    # Ideal controls never get a fit, even for exponential data.
    ideal = adsi_growth_summary(t, np.exp(0.45 * t), ideal=True)
    assert ideal["growth_fit"] is None
    assert ideal["relative_amplitude_final"] == pytest.approx(np.exp(0.45))


def test_alfven_script_spatial_table_uses_one_fixed_dt(monkeypatch, capsys):
    script = _load_script("validate_alfven1d")
    steps = {}
    real_solver = script.SpectralSolverMHD1D

    class Recorder(real_solver):
        def step(self, U, dt):
            steps.setdefault(self.grid.N, []).append(dt)
            return super().step(U, dt)

    monkeypatch.setattr(script, "SpectralSolverMHD1D", Recorder)
    rows = script.spatial_convergence(Ns=(6, 12), t_end=0.02, dt=1e-3)
    assert [N for N, _ in rows] == [6, 12]
    assert set(steps) == {6, 12}
    first = np.asarray(steps[6])
    second = np.asarray(steps[12])
    assert len(first) == len(second) == 20
    np.testing.assert_allclose(first, 1e-3)
    np.testing.assert_allclose(second, first)
    assert all(error < 1e-10 for _, error in rows)


def test_alfven_script_temporal_order_is_fourth_at_fixed_n():
    script = _load_script("validate_alfven1d")
    rows = script.temporal_convergence(N=16, t_end=0.05,
                                       dts=(1e-2, 5e-3, 2.5e-3))
    order = np.log(rows[0][1] / rows[1][1]) / np.log(rows[0][0] / rows[1][0])
    assert 3.5 < order < 4.5


def test_alfven_script_help_does_not_run_the_diagnostic(capsys):
    script = _load_script("validate_alfven1d")
    with pytest.raises(SystemExit) as exit_info:
        script.parse_args(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    assert "--mps" in out
    assert "VALIDATION" not in out


def test_mhd_turb_band_uses_mode_numbers_and_b_is_divergence_free_in_any_box():
    grid = Grid3D(16, 16, 16, 1.0, 2.0, 0.5, dealias=True, backend="numpy")
    kmin, kmax = 2.0, 3.0
    b0 = (0.0, 0.0, 1.0)
    Bx, By, Bz = banded_random_b_3d(grid, b_rms=0.1, kmin=kmin, kmax=kmax, b0=b0)
    n = [np.fft.fftfreq(16, d=1.0 / 16) for _ in range(3)]
    nz, ny, nx = np.meshgrid(n[2], n[1], n[0], indexing="ij")
    hats = [np.fft.fftn(c - m) for c, m in zip((Bx, By, Bz), b0)]
    nmag = np.sqrt(nx**2 + ny**2 + nz**2)
    outside = (nmag < kmin) | (nmag > kmax)
    assert max(np.abs(h[outside]).max() for h in hats) < 1e-10
    divergence = (2j * np.pi) * (nx / 1.0 * hats[0] + ny / 2.0 * hats[1] + nz / 0.5 * hats[2])
    assert np.abs(divergence).max() < 1e-9 * max(np.abs(h).max() for h in hats)
