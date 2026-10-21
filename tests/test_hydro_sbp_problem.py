from pathlib import Path

import numpy as np
import pytest
import yaml

from phrike.problems import ProblemRegistry


def _problem(tmp_path):
    path = Path(__file__).resolve().parents[1]/"configs/gaussian_wave2d_y_sbp.yaml"
    config = yaml.safe_load(path.read_text())
    config["io"]["outdir"] = str(tmp_path)
    return ProblemRegistry.create_problem("gaussian_wave2d_y", config=config)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("direction", ["up", "down"])
def test_configured_hydro_pulse_direction_and_backend(tmp_path, backend, direction):
    if backend == "torch":
        torch = pytest.importorskip("torch")
    problem = _problem(tmp_path)
    problem.config["initial_conditions"].update(direction=direction, y0=1.0, v0=0.05)
    grid = problem.create_grid(backend, "cpu")
    equations = problem.create_equations()
    U = problem.create_initial_conditions(grid)
    assert grid._mixed_open_y and grid.sbp_dissipation_strength == 0.01
    assert grid.bc_config == problem.config["grid"]["bc_config"]
    assert grid._bc_boundary_map["bottom"] == "open"
    assert grid.fft_workers == 1
    if backend == "torch":
        assert isinstance(U, torch.Tensor) and U.device.type == "cpu"
        assert U.dtype == grid.x.dtype
    initial = problem.convert_torch_to_numpy(U)[0]
    analytical = problem.convert_torch_to_numpy(problem.get_analytical_solution(grid, 0))[0]
    np.testing.assert_array_equal(initial, analytical)
    solver = problem.get_solver_class()(grid, equations, cfl=problem.cfl)
    t = 0.0
    while t < 0.2-1e-13:
        dt = min(solver.compute_dt(U), 0.2-t)
        U = solver.step(U, dt)
        t += dt
    actual = problem.convert_torch_to_numpy(U)[0]
    analytical = problem.convert_torch_to_numpy(problem.get_analytical_solution(grid, t))[0]
    error = np.linalg.norm(actual[0]-analytical[0])/np.linalg.norm(initial[0]-1)
    assert error < 0.003
    np.testing.assert_allclose(actual, np.broadcast_to(actual[..., :1], actual.shape), atol=1e-14)
    peak = np.asarray(grid.y)[np.argmax(actual[0, :, 0])]
    assert (peak-1.0)*(1 if direction == "up" else -1) > 0


def test_shipped_hydro_pulse_exits_open_boundaries(tmp_path):
    problem = _problem(tmp_path)
    grid = problem.create_grid("numpy")
    equations = problem.create_equations()
    U = problem.create_initial_conditions(grid)
    solver = problem.get_solver_class()(grid, equations, cfl=problem.cfl)
    rho0, p0 = 1.0, 0.6
    def acoustic_energy(state):
        _, _, vy, p = equations.primitive(state)
        return np.sum(grid.wy[:, None]*(rho0*vy**2+(p-p0)**2/rho0))
    initial_energy = acoustic_energy(U)
    t = 0.0
    while t < problem.t_end-1e-13:
        dt = min(solver.compute_dt(U), problem.t_end-t)
        U = solver.step(U, dt)
        t += dt
    assert acoustic_energy(U)/initial_energy < 1e-4
    rho, _, _, p = equations.primitive(U)
    assert rho.min() > 0 and p.min() > 0
