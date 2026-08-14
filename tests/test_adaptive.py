import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from phrike.adaptive import (
    AdaptiveTimeStepper,
    RK23,
    RK45,
    RK78,
)
from phrike.equations import EulerEquations2D, EulerEquations3D
from phrike.grid import Grid2D, Grid3D
from phrike.problems.problem_list.khi2d import KHI2DProblem
from phrike.solver import SpectralSolver2D, SpectralSolver3D


def _integrate_exponential(method, steps: int, estimate: str) -> float:
    state = np.array([1.0])
    dt = 1.0 / steps
    for _ in range(steps):
        high, low, _ = method.step(lambda value: value, state, dt)
        state = high if estimate == "high" else low
    return float(state[0])


@pytest.mark.parametrize("method_type", [RK23, RK45, RK78])
def test_embedded_tableaus_are_consistent(method_type):
    method = method_type()
    np.testing.assert_allclose(method.a.sum(axis=1), method.c, atol=5e-14)
    assert method.b_high.sum() == pytest.approx(1.0, abs=5e-14)
    assert method.b_low.sum() == pytest.approx(1.0, abs=5e-14)


def test_rkf78_is_the_complete_thirteen_stage_fehlberg_pair():
    method = RK78()

    assert len(method.c) == 13
    assert method.order_high == 8
    assert method.order_low == 7
    assert method.error_estimator_order == 8

    calls = 0

    def constant_rhs(state):
        nonlocal calls
        calls += 1
        return np.ones_like(state)

    high, low, error = method.step(constant_rhs, np.array([0.0]), 1.0)
    assert calls == 13
    np.testing.assert_allclose(high, [1.0], atol=5e-14)
    np.testing.assert_allclose(low, [1.0], atol=5e-14)
    assert error == pytest.approx(0.0, abs=5e-14)


def test_rkf78_high_and_embedded_estimates_have_expected_orders():
    method = RK78()

    high_errors = [
        abs(_integrate_exponential(method, steps, "high") - math.e) for steps in (2, 4)
    ]
    low_errors = [
        abs(_integrate_exponential(method, steps, "low") - math.e) for steps in (2, 4)
    ]

    # Halving dt approaches factors 2**8 and 2**7 respectively. These
    # conservative bounds avoid relying on asymptotic roundoff behavior.
    assert high_errors[0] / high_errors[1] > 180.0
    assert low_errors[0] / low_errors[1] > 100.0


def test_rkf78_controller_uses_eighth_order_error_scaling():
    class RecordingController:
        max_rejections = 3

        def __init__(self):
            self.order = None

        def compute_new_dt(self, dt, error, solution_scale, order):
            self.order = order
            return dt, True

    controller = RecordingController()
    stepper = AdaptiveTimeStepper(RK78(), controller)
    result = stepper.step(
        lambda value: value,
        np.array([1.0]),
        dt=0.1,
        solution_scale=1.0,
    )

    assert result.accepted
    assert controller.order == 8


def test_existing_khi_config_enables_rkf78(tmp_path):
    config_path = Path(__file__).parents[1] / "configs" / "khi2d.yaml"
    config = yaml.safe_load(config_path.read_text())
    config["io"]["outdir"] = str(tmp_path)
    config["grid"]["Nx"] = 8
    config["grid"]["Ny"] = 8

    problem = KHI2DProblem(config=config)

    assert problem.scheme == "rk78"
    assert problem.adaptive_config is not None
    assert problem.adaptive_config["enabled"] is True
    assert problem.adaptive_config["scheme"] == "rk78"

    solver = problem.get_solver_class()(
        grid=problem.create_grid(backend="cpu"),
        equations=problem.create_equations(),
        scheme=problem.scheme,
        cfl=problem.cfl,
        adaptive_config=problem.adaptive_config,
    )
    assert isinstance(solver.adaptive_stepper.method, RK78)


def test_all_rkf78_configs_enable_the_adaptive_method():
    config_dir = Path(__file__).parents[1] / "configs"
    rkf78_configs = []

    for config_path in config_dir.glob("*.yaml"):
        config = yaml.safe_load(config_path.read_text())
        if config.get("integration", {}).get("scheme") != "rk78":
            continue
        rkf78_configs.append(config_path.name)
        adaptive = config["integration"].get("adaptive") or config.get("adaptive")
        assert adaptive is not None, config_path.name
        assert adaptive.get("enabled") is True, config_path.name
        assert adaptive.get("scheme") == "rk78", config_path.name

    assert rkf78_configs


@pytest.mark.parametrize("dimensions", [2, 3])
def test_rkf78_solver_step_works_in_multiple_dimensions(dimensions):
    adaptive = {
        "enabled": True,
        "scheme": "rk78",
        "rtol": 1e-6,
        "atol": 1e-8,
    }
    if dimensions == 2:
        grid = Grid2D(Nx=6, Ny=6, Lx=1.0, Ly=1.0, backend="cpu")
        equations = EulerEquations2D()
        shape = (6, 6)
        state = equations.conservative(
            np.ones(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.ones(shape),
        )
        solver = SpectralSolver2D(
            grid, equations, scheme="rk78", adaptive_config=adaptive
        )
    else:
        grid = Grid3D(Nx=4, Ny=4, Nz=4, Lx=1.0, Ly=1.0, Lz=1.0, backend="cpu")
        equations = EulerEquations3D()
        shape = (4, 4, 4)
        state = equations.conservative(
            np.ones(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.ones(shape),
        )
        solver = SpectralSolver3D(
            grid, equations, scheme="rk78", adaptive_config=adaptive
        )

    stepped = solver.step(state, 1e-4)
    assert np.all(np.isfinite(stepped))


def test_embedded_methods_preserve_torch_backend_when_available():
    torch = pytest.importorskip("torch")
    state = torch.tensor([1.0], dtype=torch.float64)

    for method in (RK23(), RK45(), RK78()):
        high, low, error = method.step(lambda value: value, state, 0.01)
        assert isinstance(high, torch.Tensor)
        assert isinstance(low, torch.Tensor)
        assert high.device == state.device
        assert low.dtype == state.dtype
        assert math.isfinite(error)


def test_rkf78_solver_step_runs_on_torch_cpu_when_available():
    torch = pytest.importorskip("torch")
    grid = Grid2D(
        Nx=4,
        Ny=4,
        Lx=1.0,
        Ly=1.0,
        backend="torch",
        torch_device="cpu",
    )
    equations = EulerEquations2D()
    shape = (4, 4)
    state = equations.conservative(
        torch.ones(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        torch.zeros(shape, dtype=torch.float64),
        torch.ones(shape, dtype=torch.float64),
    )
    solver = SpectralSolver2D(
        grid,
        equations,
        scheme="rk78",
        adaptive_config={
            "enabled": True,
            "scheme": "rk78",
            "rtol": 1e-6,
            "atol": 1e-8,
        },
    )

    stepped = solver.step(state, 1e-4)
    assert isinstance(stepped, torch.Tensor)
    assert stepped.device.type == "cpu"
    assert torch.all(torch.isfinite(stepped))
