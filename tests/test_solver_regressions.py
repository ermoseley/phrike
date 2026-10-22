"""Regression tests for the time integrator and adaptive controller.

Numerical checks use the NumPy backend. Simulation files written by these
tests stay under the job scratch directory, not in the repository tree.
"""

import warnings

import numpy as np
import pytest

from phrike.adaptive import (
    AdaptiveStepController,
    AdaptiveStepResult,
    RK23,
    create_adaptive_stepper,
)
from phrike.equations import (
    EulerEquations1D,
    EulerEquations2D,
    EulerEquations3D,
    MHDEquations1D,
    MHDEquations2D,
)
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.solver import (
    SpectralSolver1D,
    SpectralSolver2D,
    SpectralSolver3D,
    SpectralSolverMHD,
    SpectralSolverMHD2D,
    SpectralSolverMHD3D,
    UnboundedTimeStep,
    _compute_rhs_2d,
    _compute_rhs_3d,
    _positivity_clamp,
    _positivity_clamp_2d,
)
import phrike.solver as solver_module


def _ones(shape):
    return np.ones(shape)


def _zeros(shape):
    return np.zeros(shape)


def _euler1d(n=16):
    grid = Grid1D(N=n, Lx=1.0, backend="numpy")
    equations = EulerEquations1D(gamma=1.4)
    state = equations.conservative(_ones(n), _zeros(n), _ones(n))
    return grid, equations, state


def _euler2d(n=8):
    grid = Grid2D(Nx=n, Ny=n, Lx=1.0, Ly=1.0, backend="numpy")
    equations = EulerEquations2D(gamma=1.4)
    shape = (n, n)
    state = equations.conservative(_ones(shape), _zeros(shape), _zeros(shape), _ones(shape))
    return grid, equations, state


def _euler3d(n=4):
    grid = Grid3D(Nx=n, Ny=n, Nz=n, Lx=1.0, Ly=1.0, Lz=1.0, backend="numpy")
    equations = EulerEquations3D(gamma=1.4)
    shape = (n, n, n)
    state = equations.conservative(
        _ones(shape), _zeros(shape), _zeros(shape), _zeros(shape), _ones(shape)
    )
    return grid, equations, state


def _mhd1d(n=16):
    grid = Grid1D(N=n, Lx=1.0, backend="numpy")
    equations = MHDEquations1D(gamma=1.4)
    state = equations.conservative(
        _ones(n), _zeros(n), _zeros(n), _zeros(n), _ones(n),
        _zeros(n), _zeros(n), _zeros(n),
    )
    return grid, equations, state


class _AcceptNan:
    """Duck-typed stepper that publishes a non-finite accepted state."""

    def step(self, _rhs, state, dt, _scale, limit_state=None):
        updated = np.array(state, copy=True)
        updated.ravel()[0] = np.nan
        return AdaptiveStepResult(
            U_new=updated,
            dt_used=dt,
            dt_next=dt,
            accepted=True,
            error=0.0,
            rejections=0,
        )


class _NanFlux1D(EulerEquations1D):
    def flux(self, U):
        return super().flux(U) * np.nan


class _NanFlux2D(EulerEquations2D):
    def flux(self, U):
        fx, fy = super().flux(U)
        return fx * np.nan, fy * np.nan


class _NanFlux3D(EulerEquations3D):
    def flux(self, U):
        fx, fy, fz = super().flux(U)
        return fx * np.nan, fy * np.nan, fz * np.nan


def test_nonfinite_initial_state_raises_before_the_loop():
    grid1, eqs1, state1 = _euler1d()
    grid2, eqs2, state2 = _euler2d()
    grid3, eqs3, state3 = _euler3d()
    gridm, eqsm, statem = _mhd1d()
    cases = {
        "1d-fixed": (SpectralSolver1D(grid1, eqs1, scheme="rk2"), state1),
        "1d-adaptive": (
            SpectralSolver1D(
                grid1, eqs1, scheme="rk45",
                adaptive_config={"enabled": True, "scheme": "rk45"},
            ),
            state1,
        ),
        "2d": (SpectralSolver2D(grid2, eqs2, scheme="rk2"), state2),
        "3d": (SpectralSolver3D(grid3, eqs3, scheme="rk2"), state3),
        "mhd": (SpectralSolverMHD(gridm, eqsm, scheme="rk2"), statem),
    }
    for name, (solver, state) in cases.items():
        broken = np.array(state, copy=True)
        broken.ravel()[0] = np.nan
        with pytest.raises(FloatingPointError, match=r"before time integration at step 0, t=0"):
            solver.run(broken, t0=0.0, t_end=0.01, output_interval=0.01)


def test_nonfinite_state_after_an_accepted_step_raises():
    grid, _, state = _euler1d()
    fixed = SpectralSolver1D(grid, _NanFlux1D(gamma=1.4), scheme="rk2")
    with pytest.raises(FloatingPointError, match=r"after accepted step at step 1, t="):
        fixed.run(state, t0=0.0, t_end=0.05, output_interval=1.0)

    grid2, _, state2 = _euler2d()
    solver2 = SpectralSolver2D(grid2, _NanFlux2D(gamma=1.4), scheme="rk2")
    with pytest.raises(FloatingPointError, match=r"after accepted step at step 1, t="):
        solver2.run(state2, t0=0.0, t_end=0.05, output_interval=1.0)

    grid3, _, state3 = _euler3d()
    solver3 = SpectralSolver3D(grid3, _NanFlux3D(gamma=1.4), scheme="rk2")
    with pytest.raises(FloatingPointError, match=r"after accepted step at step 1, t="):
        solver3.run(state3, t0=0.0, t_end=0.05, output_interval=1.0)

    gridm, eqsm, statem = _mhd1d()
    adaptive = SpectralSolverMHD(
        gridm, eqsm, scheme="rk45",
        adaptive_config={"enabled": True, "scheme": "rk45"},
    )
    adaptive.adaptive_stepper = _AcceptNan()
    with pytest.raises(FloatingPointError, match=r"after accepted step at step 1, t="):
        adaptive.run(statem, t0=0.0, t_end=0.05, output_interval=1.0)


def test_compute_dt_rejects_nonfinite_state_and_nonfinite_speed():
    grid, equations, state = _euler1d()
    solver = SpectralSolver1D(grid, equations, scheme="rk2")
    broken = np.array(state, copy=True)
    broken.ravel()[0] = np.inf
    with pytest.raises(FloatingPointError, match=r"while computing the time step at step 0"):
        solver.compute_dt(broken)

    class _BadSpeed(EulerEquations1D):
        def max_wave_speed(self, U):
            return float("nan")

    bad = SpectralSolver1D(grid, _BadSpeed(gamma=1.4), scheme="rk2")
    with pytest.raises(FloatingPointError, match=r"Non-finite maximum wave speed"):
        bad.compute_dt(state)

    class _NegativeSpeed(EulerEquations1D):
        def max_wave_speed(self, U):
            return -1.0

    negative = SpectralSolver1D(grid, _NegativeSpeed(gamma=1.4), scheme="rk2")
    with pytest.raises(FloatingPointError, match=r"Negative maximum wave speed"):
        negative.compute_dt(state)


def test_zero_wave_speed_is_not_a_silent_tiny_step():
    grid, equations, _ = _euler1d(n=32)
    resting = equations.conservative(_ones(32), _zeros(32), _zeros(32))
    solver = SpectralSolver1D(grid, equations, scheme="rk2", cfl=0.4)
    with pytest.raises(UnboundedTimeStep, match="zero"):
        solver.compute_dt(resting)

    with pytest.warns(RuntimeWarning, match="Positivity clamp"):
        history = solver.run(resting, t0=0.0, t_end=0.02, output_interval=0.02)
    assert solver.t == pytest.approx(0.02)
    assert history["time"][-1] == pytest.approx(0.02)
    assert len(history["time"]) <= 4

    adaptive = SpectralSolver1D(
        grid, equations, scheme="rk23", cfl=0.4,
        adaptive_config={"enabled": True, "scheme": "rk23", "rtol": 1e-4, "atol": 1e-6},
    )
    with pytest.warns(RuntimeWarning, match="Positivity clamp"):
        adaptive_history = adaptive.run(resting, t0=0.0, t_end=0.02, output_interval=0.02)
    assert adaptive.t == pytest.approx(0.02)
    assert len(adaptive_history["time"]) <= 4


def test_nonfinite_error_shrinks_and_then_raises():
    controller = AdaptiveStepController(
        min_dt_factor=0.5,
        max_dt_factor=5.0,
        max_rejections=4,
        dt_min=1e-8,
    )
    stepper = create_adaptive_stepper(
        "rk23",
        rtol=1e-6,
        atol=1e-8,
        min_dt_factor=0.5,
        max_dt_factor=5.0,
        max_rejections=4,
        dt_min=1e-8,
    )
    # The stepper built above owns its own controller; check the formula too.
    shrunk, accept = controller.compute_new_dt(1.0e-2, float("nan"), 1.0, order=3)
    assert accept is False
    assert shrunk == pytest.approx(5.0e-3)
    grown, _ = controller.compute_new_dt(1.0e-2, float("inf"), 1.0, order=3)
    assert grown < 1.0e-2

    dt = 1.0e-2
    returned = []

    def nan_rhs(_state):
        return np.array([np.nan])

    with pytest.raises(FloatingPointError, match="non-finite error"):
        for _ in range(6):
            result = stepper.step(nan_rhs, np.array([1.0]), dt, solution_scale=1.0)
            assert result.accepted is False
            assert result.U_new[0] == pytest.approx(1.0)
            assert result.dt_next < dt
            assert result.dt_next == pytest.approx(dt * 0.5)
            returned.append(result.dt_next)
            dt = result.dt_next
    assert returned == pytest.approx([5.0e-3, 2.5e-3, 1.25e-3])
    assert all(later < earlier for earlier, later in zip(returned, returned[1:]))


def test_max_rejections_must_be_an_integer_at_least_one():
    with pytest.raises(ValueError, match="max_rejections"):
        AdaptiveStepController(max_rejections=0)
    with pytest.raises(ValueError, match="max_rejections"):
        AdaptiveStepController(max_rejections=1.5)
    with pytest.raises(ValueError, match="max_rejections"):
        AdaptiveStepController(max_rejections=True)
    AdaptiveStepController(max_rejections=1)


def test_forced_acceptance_warns_and_is_flagged():
    stepper = create_adaptive_stepper(
        "rk23",
        rtol=0.0,
        atol=0.0,
        max_rejections=3,
        min_dt_factor=0.5,
        max_dt_factor=2.0,
        dt_min=1e-8,
    )
    state = np.array([1.0])
    dt = 0.2
    forced = None
    with pytest.warns(RuntimeWarning, match="exceeds the tolerance"):
        for _ in range(5):
            result = stepper.step(lambda value: value, state, dt, solution_scale=1.0)
            if result.forced:
                forced = result
                break
            dt = result.dt_next
            assert result.accepted is False
    assert forced is not None
    assert forced.accepted is True
    assert forced.forced is True
    assert stepper.forced_acceptances == 1


def test_rejected_step_below_dt_floor_raises():
    stepper = create_adaptive_stepper(
        "rk23",
        rtol=0.0,
        atol=0.0,
        max_rejections=100,
        min_dt_factor=0.1,
        dt_min=1.0e-4,
    )
    dt = 1.0e-3
    with pytest.raises(RuntimeError, match=r"dt_min=1.000e-04"):
        for _ in range(10):
            result = stepper.step(lambda value: value, np.array([1.0]), dt, 1.0)
            dt = result.dt_next


def test_scheme_names_are_validated():
    grid, equations, _ = _euler1d(n=8)
    with pytest.raises(ValueError, match="Unknown integration scheme"):
        SpectralSolver1D(grid, equations, scheme="bogus")
    with pytest.raises(ValueError, match="embedded method"):
        SpectralSolver1D(grid, equations, scheme="rk45")
    with pytest.raises(ValueError, match="different embedded methods"):
        SpectralSolver1D(
            grid, equations, scheme="rk78",
            adaptive_config={"enabled": True, "scheme": "rk45"},
        )

    inherited = SpectralSolver1D(
        grid, equations, scheme="rk23",
        adaptive_config={"enabled": True},
    )
    assert inherited.scheme == "rk23"
    assert inherited.adaptive_enabled is True

    default_embedded = SpectralSolver1D(
        grid, equations, scheme="rk4",
        adaptive_config={"enabled": True},
    )
    assert default_embedded.scheme == "rk45"
    assert default_embedded.adaptive_enabled is True

    solver_module._UNUSED_FALLBACK_WARNED = False
    with pytest.warns(RuntimeWarning, match="fallback_scheme is unused"):
        warned = SpectralSolver2D(
            *_euler2d_grid_eqs(),
            scheme="rk45",
            adaptive_config={
                "enabled": True,
                "scheme": "rk45",
                "fallback_scheme": "rk4",
            },
        )
    assert warned.scheme == "rk45"
    assert warned.adaptive_enabled is True

    with pytest.warns(RuntimeWarning, match="using fixed-step"):
        fallen = SpectralSolver1D(
            grid, equations, scheme="rk4",
            adaptive_config={
                "enabled": True,
                "scheme": "rk4",
                "fallback_scheme": "rk2",
            },
        )
    assert fallen.scheme == "rk2"
    assert fallen.adaptive_enabled is False
    assert fallen.adaptive_stepper is None

    with pytest.raises(ValueError, match="fallback_scheme"):
        SpectralSolver1D(
            grid, equations, scheme="rk4",
            adaptive_config={"enabled": True, "scheme": "rk2", "fallback_scheme": "rk45"},
        )


def _euler2d_grid_eqs():
    grid, equations, _ = _euler2d(n=8)
    return grid, equations


def test_two_dimensional_compute_dt_includes_artificial_viscosity():
    grid, equations, state = _euler2d(n=16)
    bare = SpectralSolver2D(grid, equations, scheme="rk4", cfl=0.4)
    viscous = SpectralSolver2D(
        grid, equations, scheme="rk4", cfl=0.4,
        artificial_viscosity_config={
            "enabled": True,
            "mode": "constant",
            "nu_constant": 1.0,
        },
    )
    hyperbolic = bare.compute_dt(state)
    limited = viscous.compute_dt(state)
    av_dt = 0.2 * grid.dx ** 2 / 1.0
    assert limited == pytest.approx(min(hyperbolic, av_dt))
    assert limited < hyperbolic


def test_unsupported_artificial_viscosity_raises():
    config = {"enabled": True, "mode": "constant", "nu_constant": 1.0}
    grid3, eqs3, _ = _euler3d()
    with pytest.raises(NotImplementedError, match="SpectralSolver3D"):
        SpectralSolver3D(grid3, eqs3, artificial_viscosity_config=config)

    grid2, eqs2, _ = _euler2d()
    with pytest.raises(NotImplementedError, match="2D MHD"):
        SpectralSolverMHD2D(grid2, MHDEquations2D(), artificial_viscosity_config=config)
    with pytest.raises(NotImplementedError, match="3D MHD"):
        SpectralSolverMHD3D(grid3, eqs3, artificial_viscosity_config=config)


def test_mhd1d_artificial_viscosity_still_limits_dt():
    grid, equations, state = _mhd1d(n=16)
    bare = SpectralSolverMHD(grid, equations, scheme="rk2", cfl=0.3)
    viscous = SpectralSolverMHD(
        grid, equations, scheme="rk2", cfl=0.3,
        artificial_viscosity_config={
            "enabled": True,
            "mode": "constant",
            "nu_constant": 1.0,
        },
    )
    hyperbolic = bare.compute_dt(state)
    limited = viscous.compute_dt(state)
    av_dt = 0.2 * grid.dx ** 2 / 1.0
    assert limited == pytest.approx(min(hyperbolic, av_dt))
    assert limited < hyperbolic


def test_positivity_clamp_warns_once_and_keeps_historical_floors():
    grid3, equations3, smooth3 = _euler3d()
    solver3 = SpectralSolver3D(grid3, equations3, scheme="rk4")
    stepped = solver3.step(smooth3, 1.0e-4)
    assert solver3.positivity.events == 0
    assert np.all(np.isfinite(stepped))

    # The 3D floor is a no-op on a positive state, so the step matches the
    # unclamped Runge-Kutta update followed by the usual dealias projection.
    unclamped = _unclamped_rk4_3d(grid3, equations3, smooth3, 1.0e-4)
    expected = grid3.apply_accepted_state_stabilization(unclamped, 0.5e-4)
    np.testing.assert_array_equal(stepped, expected)

    negative = np.array(smooth3, copy=True)
    negative[4] = -1.0
    with pytest.warns(RuntimeWarning, match="Positivity clamp"):
        floored = solver3.step(negative, 1.0e-4)
    _, _, _, _, pressure = equations3.primitive(floored)
    assert np.min(pressure) >= solver3.p_floor
    assert solver3.positivity.events >= 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solver3.step(negative, 1.0e-4)
    assert not [item for item in caught if issubclass(item.category, RuntimeWarning)]

    grid2, equations2, _ = _euler2d(n=4)
    low = equations2.conservative(
        np.full((4, 4), 1.0e-9),
        _zeros((4, 4)),
        _zeros((4, 4)),
        _ones((4, 4)),
    )
    clamped2 = _positivity_clamp_2d(low, equations2)
    assert np.min(clamped2[0]) == pytest.approx(1.0e-8)

    grid1, equations1, _ = _euler1d(n=4)
    low1 = equations1.conservative(np.full(4, 1.0e-12), _zeros(4), np.full(4, 1.0e-14))
    clamped1 = _positivity_clamp(low1, equations1)
    assert np.min(clamped1[0]) == pytest.approx(1.0e-10)


def _unclamped_rk4_3d(grid, equations, state, dt):
    k1 = _compute_rhs_3d(grid, equations, state)
    k2 = _compute_rhs_3d(grid, equations, state + 0.5 * dt * k1)
    k3 = _compute_rhs_3d(grid, equations, state + 0.5 * dt * k2)
    k4 = _compute_rhs_3d(grid, equations, state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def test_embedded_stage_limiter_floors_later_right_hand_sides():
    seen = []

    def rhs(state):
        seen.append(float(state[0]))
        derivative = np.zeros_like(state)
        derivative[0] = -100.0
        return derivative

    def limit(state):
        floored = np.array(state, copy=True)
        floored[0] = max(float(floored[0]), 1.0e-10)
        return floored

    RK23().step(rhs, np.array([1.0]), 1.0, limit_state=limit)
    assert seen[0] == pytest.approx(1.0)
    assert len(seen) == 4
    assert min(seen[1:]) >= 1.0e-10


def test_mhd_adaptive_stages_are_not_positivity_limited():
    grid, equations, state = _mhd1d(n=8)

    class _RecordLimit:
        def __init__(self):
            self.limit_state = "missing"

        def step(self, _rhs, current, dt, _scale, limit_state=None):
            self.limit_state = limit_state
            return AdaptiveStepResult(
                U_new=current,
                dt_used=dt,
                dt_next=dt,
                accepted=True,
                error=0.0,
                rejections=0,
            )

    solver = SpectralSolverMHD(
        grid, equations, scheme="rk45",
        adaptive_config={"enabled": True, "scheme": "rk45"},
    )
    recorder = _RecordLimit()
    solver.adaptive_stepper = recorder
    solver.step(state, 1.0e-4)
    assert recorder.limit_state is None


def test_outputs_and_checkpoints_land_on_requested_times_once(tmp_path):
    grid, equations, state = _euler1d(n=32)
    solver = SpectralSolver1D(grid, equations, scheme="rk2", cfl=0.4)
    dt_cfl = solver.compute_dt(state)
    output_interval = dt_cfl * 2.5
    t_end = output_interval * 2.0
    outdir = tmp_path / "output_timing"
    outdir.mkdir()
    seen = []

    def on_output(time, _state):
        seen.append(time)

    history = solver.run(
        state,
        t0=0.0,
        t_end=t_end,
        output_interval=output_interval,
        checkpoint_interval=t_end,
        outdir=str(outdir),
        on_output=on_output,
    )
    expected = [0.0, output_interval, t_end]
    np.testing.assert_allclose(history["time"], expected, atol=1.0e-12)
    np.testing.assert_allclose(seen, expected, atol=1.0e-12)
    snapshots = list(outdir.glob("snapshot_t*.npz"))
    assert len(snapshots) == 1
    assert snapshots[0].name == f"snapshot_t{t_end:.6f}.npz"


def test_run_installs_an_adaptive_dt_floor_unless_one_was_set():
    grid, equations, state = _euler1d(n=16)
    t_end = 1.0e-3
    automatic = SpectralSolver1D(
        grid, equations, scheme="rk45",
        adaptive_config={"enabled": True, "scheme": "rk45", "rtol": 1e-4, "atol": 1e-6},
    )
    automatic.run(state, t0=0.0, t_end=t_end, output_interval=t_end)
    assert automatic.adaptive_stepper.controller.dt_min == pytest.approx(1.0e-12 * t_end)

    chosen = SpectralSolver1D(
        grid, equations, scheme="rk45",
        adaptive_config={
            "enabled": True,
            "scheme": "rk45",
            "rtol": 1e-4,
            "atol": 1e-6,
            "dt_min": 1.0e-8,
        },
    )
    chosen.run(state, t0=0.0, t_end=t_end, output_interval=t_end)
    assert chosen.adaptive_stepper.controller.dt_min == pytest.approx(1.0e-8)


@pytest.mark.parametrize("dim", [2, 3])
def test_adaptive_proposal_is_capped_by_the_cfl_in_every_dimension(dim):
    adaptive = {
        "enabled": True,
        "scheme": "rk45",
        "rtol": 1e-2,
        "atol": 1e-4,
        "safety_factor": 0.9,
        "min_dt_factor": 0.1,
        "max_dt_factor": 5.0,
        "max_rejections": 4,
    }
    if dim == 2:
        grid, equations, state = _euler2d(n=8)
        solver = SpectralSolver2D(grid, equations, scheme="rk45", cfl=0.05, adaptive_config=adaptive)
    else:
        grid, equations, state = _euler3d(n=4)
        solver = SpectralSolver3D(grid, equations, scheme="rk45", cfl=0.05, adaptive_config=adaptive)
    cfl_dt = solver.compute_dt(state)
    used = []

    def on_step(_step, dt, _state):
        used.append(dt)

    t_end = cfl_dt * 6.0
    solver.run(state, t0=0.0, t_end=t_end, output_interval=t_end, on_step=on_step)
    assert solver.t == pytest.approx(t_end)
    assert len(used) >= 3
    assert max(used) <= cfl_dt * 1.05


def test_mhd2d_cfl_uses_minimum_node_spacing():
    grid = Grid2D(
        Nx=8, Ny=8, Lx=1.0, Ly=1.0,
        basis_x="legendre", basis_y="legendre", backend="numpy",
    )
    equations = MHDEquations2D()
    shape = (grid.Ny, grid.Nx)
    zeros = _zeros(shape)
    state = equations.conservative(
        _ones(shape), zeros, zeros, zeros, _ones(shape), zeros, zeros, zeros
    )
    solver = SpectralSolverMHD(grid, equations, scheme="rk2", cfl=0.35)
    speed = equations.max_wave_speed(state)
    spacing = min(grid.dx_min, grid.dy_min)
    assert spacing < min(grid.dx, grid.dy)
    dt = solver.compute_dt(state)
    assert dt == pytest.approx(0.35 * spacing / speed)
    assert dt < 0.35 * min(grid.dx, grid.dy) / speed


def test_boundary_exception_is_not_swallowed():
    class _Boom:
        def apply_boundary_conditions(self, _state, _equations):
            raise RuntimeError("bc failed")

    state = np.ones((4, 4, 4))
    with pytest.raises(RuntimeError, match="bc failed"):
        _compute_rhs_2d(_Boom(), EulerEquations2D(), state)


def test_grid_k2_max_uses_spacing_and_does_not_swallow_errors():
    from types import SimpleNamespace

    quiet = SimpleNamespace(
        Ny=4,
        dx=0.1,
        dy=0.1,
        dx_min=0.05,
        dy_min=0.05,
        k2=None,
        _mixed_open_y=False,
    )
    solver = SpectralSolverMHD(quiet, MHDEquations1D(), scheme="rk2")
    assert solver._k2_max == pytest.approx(2.0 * 4.0 / 0.05 ** 2)

    class _Raising:
        Ny = 4
        dx = 0.1
        dy = 0.1
        _mixed_open_y = False

        @property
        def k2(self):
            raise RuntimeError("no wavenumbers")

    with pytest.raises(RuntimeError, match="no wavenumbers"):
        SpectralSolverMHD(_Raising(), MHDEquations1D(), scheme="rk2")


def test_rk2_and_rk4_keep_their_temporal_orders_on_a_smooth_problem():
    grid = Grid1D(N=32, Lx=1.0, backend="numpy")
    equations = EulerEquations1D(gamma=1.4)
    x = np.asarray(grid.x)
    initial = equations.conservative(
        1.0 + 0.1 * np.sin(2.0 * np.pi * x),
        0.2 + 0.05 * np.cos(2.0 * np.pi * x),
        1.0 + 0.05 * np.sin(4.0 * np.pi * x),
    )
    t_end = 0.02

    def integrate(steps, scheme):
        solver = SpectralSolver1D(grid, equations, scheme=scheme)
        state = initial.copy()
        dt = t_end / steps
        for _ in range(steps):
            state = solver.step(state, dt)
        return state

    reference = integrate(64, "rk4")

    def rms(steps, scheme):
        return np.sqrt(np.mean((integrate(steps, scheme) - reference) ** 2))

    rk2_coarse, rk2_fine = rms(4, "rk2"), rms(8, "rk2")
    rk4_coarse, rk4_fine = rms(4, "rk4"), rms(8, "rk4")
    assert rk2_coarse > rk2_fine > 0.0
    assert rk4_coarse > rk4_fine > 0.0
    assert rk2_coarse / rk2_fine > 3.0
    assert rk4_coarse / rk4_fine > 8.0
