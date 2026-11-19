"""Restart a compressible MHD step whose accepted pressure hit the floor."""

import numpy as np
import pytest

from phrike.equations import MHDEquations1D
from phrike.grid import Grid1D
from phrike.problems.problem_list.brio_wu1d import BrioWu1DProblem
from phrike.solver import SpectralSolverMHD1D


DT = 1e-4
MHD_CONFIG = dict(viscosity=5e-3, bulk_viscosity=3e-3, resistivity=4e-3,
                  thermal_conductivity=6e-3, p_min=0.1)


def _solver(backend="numpy"):
    if backend == "torch":
        pytest.importorskip("torch")
    grid = Grid1D(N=48, Lx=1.0, basis="fourier", bc="periodic", dealias=True,
                  backend=backend, torch_device="cpu", precision="double")
    grid.mhd_config = MHD_CONFIG.copy()
    return SpectralSolverMHD1D(grid, MHDEquations1D(), spectral_fast_path=backend == "torch")


def _initial_state(solver):
    x = np.asarray(solver.grid.x)
    s, c = np.sin(2 * np.pi * x), np.cos(2 * np.pi * x)
    pressure = np.where(x < 0.5, 0.9, 0.02)
    U = solver.equations.conservative(1 + 0.07*s, 0.1*c, 0.05*s, 0.02*c, pressure,
                                      np.full_like(x, 0.75), 0.12*s, 0.04*c)
    return solver.grid.x.new_tensor(U) if solver.grid._use_torch else U


def _run_through_clamp(solver, **run_options):
    snapshots = []

    def save_first_step(step, dt, U):
        assert dt == DT
        if step == 0:
            assert solver.positivity.events > 0
            assert float(abs(U - solver.grid.project_dealiased(U)).max()) > 1e-2
            snapshots.append((solver.t, U.clone() if solver.grid._use_torch else U.copy()))

    # The output horizon is shorter than either stability bound.
    with pytest.warns(RuntimeWarning, match="Positivity clamp"):
        solver.run(_initial_state(solver), 0.0, 2*DT, output_interval=DT,
                   on_step=save_first_step, **run_options)
    assert solver._accepted_steps == 2
    return snapshots[0]


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_restart_after_pressure_clamp_matches_uninterrupted_step(backend):
    reference = _solver(backend)
    t, saved = _run_through_clamp(reference)
    restarted = _solver(backend)

    def check_restart(time, U):
        if time == t:
            np.testing.assert_array_equal(U, saved)
            if backend == "torch":
                assert not restarted.spectral.ready(U)

    with pytest.warns(RuntimeWarning, match="Positivity clamp"):
        restarted.run(saved, t, reference.t, restored=True, on_output=check_restart)
    assert restarted._accepted_steps == 1
    np.testing.assert_allclose(restarted.U, reference.U, rtol=1e-12, atol=0)

    fresh = _solver(backend)
    with pytest.warns(RuntimeWarning, match="Positivity clamp"):
        fresh.run(saved, t, reference.t)
    assert float(abs(fresh.U[4] - reference.U[4]).max()) > 1e-2


def test_fresh_input_still_gets_both_projections():
    solver = _solver()
    U = _initial_state(solver)
    U[5] += solver.grid.x  # A varying longitudinal field also exposes magnetic projection.
    projected = solver.grid.project_dealiased(solver._project_B(U))
    solver.run(U, 0.0, 0.0)
    np.testing.assert_array_equal(solver.U, projected)
    solver.run(U, 0.0, 0.0, restored=True)
    np.testing.assert_array_equal(solver.U, U)


def test_torch_restart_caches_only_a_band_limited_state():
    solver = _solver("torch")
    U = _initial_state(solver)
    solver.run(U, 0.0, 0.0)
    band_limited = solver.U
    solver.run(band_limited, 0.0, 0.0, restored=True)
    cached = solver.U
    np.testing.assert_array_equal(cached, band_limited)
    assert solver.spectral.ready(cached)

    solver.run(U, 0.0, 0.0, restored=True)
    np.testing.assert_array_equal(solver.U, U)
    assert not solver.spectral.ready(cached)
    assert not solver.spectral.ready(solver.U)


def test_problem_restart_from_clamped_checkpoint(tmp_path):
    reference = _solver()
    t, saved = _run_through_clamp(reference, checkpoint_interval=DT, outdir=str(tmp_path))
    problem = BrioWu1DProblem(config={
        "problem": "brio_wu1d",
        "grid": {"N": 48, "Lx": 1.0, "basis": "fourier", "bc": "periodic"},
        "physics": {"gamma": reference.equations.gamma, "mhd": MHD_CONFIG.copy()},
        "integration": {"t0": 0.0, "t_end": reference.t, "cfl": reference.cfl},
        "io": {"outdir": str(tmp_path / "restarted")},
    }, restart_from=str(tmp_path / f"snapshot_t{t:.6f}.npz"))
    assert problem.t0 == t
    np.testing.assert_array_equal(problem.restart_data["U"], saved)
    with pytest.warns(RuntimeWarning, match="Positivity clamp"):
        restarted, _ = problem.run(backend="numpy")
    assert restarted._accepted_steps == 1
    np.testing.assert_allclose(restarted.U, reference.U, rtol=1e-12, atol=0)
