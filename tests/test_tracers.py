import numpy as np
import pytest

from phrike.adaptive import AdaptiveStepResult
from phrike.equations import (
    EulerEquations1D,
    EulerEquations2D,
    MHDEquations2D,
    MHDEquations3D,
)
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.io import load_checkpoint, save_solution_snapshot
from phrike.problems.base import BaseProblem
from phrike.solver import SpectralSolver2D
from phrike.tracer_density import periodic_simplex_density_3d
from phrike.tracers import FourierTracers1D, FourierTracers2D, FourierTracers3D


class _TracerProblem(BaseProblem):
    def create_grid(self, backend="numpy", device=None, debug=False):
        raise NotImplementedError

    def create_equations(self):
        raise NotImplementedError

    def create_initial_conditions(self, grid):
        raise NotImplementedError

    def create_visualization(self, solver, t, U):
        raise NotImplementedError

    def get_solver_class(self):
        raise NotImplementedError


def test_periodic_linear_interpolation_is_exact_at_nodes_and_for_constants():
    grid_1d = Grid1D(N=5, Lx=1.0, backend="cpu")
    field_1d = np.arange(grid_1d.N, dtype=np.float64)
    np.testing.assert_allclose(
        grid_1d.interpolate_periodic_linear_at_points_1d(field_1d, grid_1d.x),
        field_1d,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        grid_1d.interpolate_periodic_linear_at_points_1d(
            np.full(grid_1d.N, 3.5), [-0.1, 0.37, 1.2]
        ),
        3.5,
    )

    grid_2d = Grid2D(Nx=4, Ny=3, Lx=1.0, Ly=1.0, backend="cpu")
    field_2d = np.arange(grid_2d.Nx * grid_2d.Ny, dtype=np.float64).reshape(
        grid_2d.Ny, grid_2d.Nx
    )
    x_2d, y_2d = np.meshgrid(grid_2d.x, grid_2d.y, indexing="xy")
    np.testing.assert_allclose(
        grid_2d.interpolate_periodic_linear_at_points(
            field_2d, x_2d.ravel(), y_2d.ravel()
        ),
        field_2d.ravel(),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        grid_2d.interpolate_periodic_linear_at_points(
            np.full_like(field_2d, -2.0), [-0.1, 0.2], [0.3, 1.1]
        ),
        -2.0,
    )

    grid_3d = Grid3D(Nx=4, Ny=3, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0, backend="cpu")
    field_3d = np.arange(
        grid_3d.Nx * grid_3d.Ny * grid_3d.Nz, dtype=np.float64
    ).reshape(grid_3d.Nz, grid_3d.Ny, grid_3d.Nx)
    x_3d = np.broadcast_to(grid_3d.x[None, None, :], field_3d.shape).ravel()
    y_3d = np.broadcast_to(grid_3d.y[None, :, None], field_3d.shape).ravel()
    z_3d = np.broadcast_to(grid_3d.z[:, None, None], field_3d.shape).ravel()
    np.testing.assert_allclose(
        grid_3d.interpolate_periodic_linear_at_points(field_3d, x_3d, y_3d, z_3d),
        field_3d.ravel(),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        grid_3d.interpolate_periodic_linear_at_points(
            np.full_like(field_3d, 1.25), [-0.1, 1.1], [0.2, -0.4], [0.6, 1.5]
        ),
        1.25,
    )


def test_periodic_linear_interpolation_matches_torch_cpu():
    torch = pytest.importorskip("torch")

    grid_1d = Grid1D(N=5, Lx=1.0, backend="cpu")
    torch_grid_1d = Grid1D(
        N=5, Lx=1.0, backend="torch", torch_device="cpu"
    )
    field_1d = np.array([0.0, 1.0, -2.0, 3.0, 1.5])
    x_1d = np.array([-0.1, 0.17, 0.73, 1.05])
    expected_1d = grid_1d.interpolate_periodic_linear_at_points_1d(field_1d, x_1d)
    actual_1d = torch_grid_1d.interpolate_periodic_linear_at_points_1d(
        torch.as_tensor(field_1d), torch.as_tensor(x_1d)
    )
    np.testing.assert_allclose(actual_1d.numpy(), expected_1d)

    grid_2d = Grid2D(Nx=4, Ny=3, Lx=1.0, Ly=1.0, backend="cpu")
    torch_grid_2d = Grid2D(
        Nx=4, Ny=3, Lx=1.0, Ly=1.0, backend="torch", torch_device="cpu"
    )
    field_2d = np.arange(12, dtype=np.float64).reshape(3, 4)
    x_2d = np.array([-0.2, 0.1, 0.8])
    y_2d = np.array([0.3, 1.1, -0.25])
    expected_2d = grid_2d.interpolate_periodic_linear_at_points(
        field_2d, x_2d, y_2d
    )
    actual_2d = torch_grid_2d.interpolate_periodic_linear_at_points(
        torch.as_tensor(field_2d), torch.as_tensor(x_2d), torch.as_tensor(y_2d)
    )
    np.testing.assert_allclose(actual_2d.numpy(), expected_2d)

    grid_3d = Grid3D(Nx=4, Ny=3, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0, backend="cpu")
    torch_grid_3d = Grid3D(
        Nx=4,
        Ny=3,
        Nz=2,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        backend="torch",
        torch_device="cpu",
    )
    field_3d = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    x_3d = np.array([0.2, -0.1, 1.05])
    y_3d = np.array([0.4, 1.2, -0.3])
    z_3d = np.array([0.8, -0.2, 1.1])
    expected_3d = grid_3d.interpolate_periodic_linear_at_points(
        field_3d, x_3d, y_3d, z_3d
    )
    actual_3d = torch_grid_3d.interpolate_periodic_linear_at_points(
        torch.as_tensor(field_3d),
        torch.as_tensor(x_3d),
        torch.as_tensor(y_3d),
        torch.as_tensor(z_3d),
    )
    np.testing.assert_allclose(actual_3d.numpy(), expected_3d)


def test_batched_2d_fourier_evaluation_matches_resolved_modes():
    grid = Grid2D(Nx=12, Ny=10, Lx=1.0, Ly=1.0, backend="cpu")
    x_mesh, y_mesh = np.meshgrid(grid.x, grid.y, indexing="xy")
    field_x = np.sin(2.0 * np.pi * x_mesh) + 0.2 * np.cos(4.0 * np.pi * y_mesh)
    field_y = np.cos(2.0 * np.pi * y_mesh) - 0.3 * np.sin(4.0 * np.pi * x_mesh)
    x = np.array([0.123, 0.456, 0.789])
    y = np.array([0.234, 0.567, 0.891])

    value_x, value_y = grid.evaluate_fourier_at_points_batched_2d(
        field_x, field_y, x, y
    )

    expected_x = np.sin(2.0 * np.pi * x) + 0.2 * np.cos(4.0 * np.pi * y)
    expected_y = np.cos(2.0 * np.pi * y) - 0.3 * np.sin(4.0 * np.pi * x)
    np.testing.assert_allclose(value_x, expected_x, atol=2.0e-6)
    np.testing.assert_allclose(value_y, expected_y, atol=2.0e-6)


def test_fourier_interpolation_converges_spectrally_for_smooth_field():
    x = np.array([0.031, 0.123, 0.279, 0.417, 0.731, 0.913])
    y = np.array([0.071, 0.217, 0.389, 0.543, 0.817, 0.957])

    def field(x_position, y_position):
        return np.exp(
            0.35 * np.sin(2.0 * np.pi * x_position)
            + 0.2 * np.cos(2.0 * np.pi * y_position)
        )

    errors = []
    for n in (6, 8, 12):
        grid = Grid2D(
            Nx=n,
            Ny=n,
            Lx=1.0,
            Ly=1.0,
            backend="cpu",
            dealias=False,
        )
        x_mesh, y_mesh = np.meshgrid(grid.x, grid.y, indexing="xy")
        interpolated = grid.evaluate_fourier_at_points(
            field(x_mesh, y_mesh), x, y
        )
        errors.append(float(np.max(np.abs(interpolated - field(x, y)))))

    assert errors[1] < 0.01 * errors[0]
    assert errors[2] < 0.01 * errors[1]


def test_explicit_midpoint_tracers_are_second_order_with_two_position_calls():
    grid = Grid1D(N=32, Lx=1.0, backend="cpu")
    equations = EulerEquations1D()
    amplitude = 0.2
    velocity = amplitude * np.sin(2.0 * np.pi * grid.x)
    state = equations.conservative(
        np.ones(grid.N), velocity, np.ones(grid.N)
    )
    initial_x = 0.2
    final_time = 0.2
    exact = np.arctan(
        np.tan(np.pi * initial_x) * np.exp(2.0 * np.pi * amplitude * final_time)
    ) / np.pi
    errors = []
    for dt in (0.05, 0.025):
        tracer = FourierTracers1D(np.array([initial_x]))
        calls = 0
        original = grid.evaluate_fourier_at_points_1d

        def counted(field, position):
            nonlocal calls
            calls += 1
            return original(field, position)

        grid.evaluate_fourier_at_points_1d = counted
        for _ in range(round(final_time / dt)):
            tracer.step(grid, equations, state, dt, U_new=state)
        grid.evaluate_fourier_at_points_1d = original
        assert calls == 2 * round(final_time / dt)
        errors.append(abs(float(tracer.x_unwrapped[0]) - exact))

    observed_order = np.log2(errors[0] / errors[1])
    assert observed_order == pytest.approx(2.0, abs=0.15)


def test_constant_velocity_tracers_advance_with_variable_dt_and_wrap():
    grid = Grid2D(Nx=4, Ny=4, Lx=1.0, Ly=2.0, backend="cpu")
    equations = EulerEquations2D()
    shape = (grid.Ny, grid.Nx)
    state = equations.conservative(
        np.ones(shape),
        np.full(shape, 0.4),
        np.full(shape, -1.2),
        np.ones(shape),
    )
    tracers = FourierTracers2D(np.array([0.9]), np.array([0.2]))

    for dt in (0.125, 0.375):
        tracers.step(grid, equations, state, dt, U_new=state)

    np.testing.assert_allclose(tracers.x_unwrapped, [1.1])
    np.testing.assert_allclose(tracers.y_unwrapped, [-0.4])
    np.testing.assert_allclose(tracers.x, [0.1])
    np.testing.assert_allclose(tracers.y, [1.6])

    boundary_tracer = FourierTracers2D(np.array([0.0]), np.array([0.0]))
    tiny_velocity_state = equations.conservative(
        np.ones(shape),
        np.full(shape, -1.0e-20),
        np.zeros(shape),
        np.ones(shape),
    )
    boundary_tracer.step(
        grid, equations, tiny_velocity_state, 1.0, U_new=tiny_velocity_state
    )
    np.testing.assert_array_equal(boundary_tracer.x, [0.0])


def test_mhd_tracer_steps_work_in_two_and_three_dimensions():
    grid_2d = Grid2D(Nx=4, Ny=4, Lx=1.0, Ly=1.0, backend="cpu")
    equations_2d = MHDEquations2D()
    shape_2d = (grid_2d.Ny, grid_2d.Nx)
    state_2d = equations_2d.conservative(
        np.ones(shape_2d),
        np.full(shape_2d, 0.3),
        np.full(shape_2d, -0.2),
        np.zeros(shape_2d),
        np.ones(shape_2d),
        np.zeros(shape_2d),
        np.zeros(shape_2d),
        np.zeros(shape_2d),
    )
    tracers_2d = FourierTracers2D(np.array([0.9]), np.array([0.1]))
    tracers_2d.step(grid_2d, equations_2d, state_2d, 0.5, U_new=state_2d)
    np.testing.assert_allclose(tracers_2d.x, [0.05], atol=2.0e-7)
    np.testing.assert_allclose(tracers_2d.y, [0.0], atol=2.0e-7)

    grid_3d = Grid3D(Nx=4, Ny=3, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0, backend="cpu")
    equations_3d = MHDEquations3D()
    shape_3d = (grid_3d.Nz, grid_3d.Ny, grid_3d.Nx)
    state_3d = equations_3d.conservative(
        np.ones(shape_3d),
        np.full(shape_3d, 0.3),
        np.full(shape_3d, -0.2),
        np.full(shape_3d, 0.4),
        np.ones(shape_3d),
        np.zeros(shape_3d),
        np.zeros(shape_3d),
        np.zeros(shape_3d),
    )
    tracers_3d = FourierTracers3D(
        np.array([0.9]), np.array([0.1]), np.array([0.8])
    )
    tracers_3d.step(grid_3d, equations_3d, state_3d, 0.5, U_new=state_3d)
    np.testing.assert_allclose(tracers_3d.x, [0.05], atol=2.0e-7)
    np.testing.assert_allclose(tracers_3d.y, [0.0], atol=2.0e-7)
    wrapped_z = float(tracers_3d.z[0])
    assert min(abs(wrapped_z), abs(wrapped_z - 1.0)) < 2.0e-7


def test_periodic_simplex_density_preserves_uniform_lattice_mass():
    shape = (3, 4, 2)
    domain = (3.0, 2.0, 4.0)
    x, y, z = np.meshgrid(
        np.arange(shape[0]) * domain[0] / shape[0],
        np.arange(shape[1]) * domain[1] / shape[1],
        np.arange(shape[2]) * domain[2] / shape[2],
        indexing="ij",
    )
    mass = 0.25
    tracers = FourierTracers3D(
        x.ravel(), y.ravel(), z.ravel(), mass=mass, lattice_shape=shape
    )

    density, centroids = periodic_simplex_density_3d(tracers, domain)
    cell_volume = np.prod(
        [length / count for length, count in zip(domain, shape)]
    )

    assert density.shape == shape[::-1]
    assert centroids.shape == (*shape[::-1], 3)
    np.testing.assert_allclose(density, mass / cell_volume)
    assert density.sum() * cell_volume == pytest.approx(mass * np.prod(shape))


def test_snapshot_round_trip_preserves_tracer_topology(tmp_path):
    grid = Grid3D(Nx=2, Ny=2, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0, backend="cpu")
    equations = MHDEquations3D()
    state_shape = (grid.Nz, grid.Ny, grid.Nx)
    state = equations.conservative(
        np.ones(state_shape),
        np.zeros(state_shape),
        np.zeros(state_shape),
        np.zeros(state_shape),
        np.ones(state_shape),
        np.zeros(state_shape),
        np.zeros(state_shape),
        np.zeros(state_shape),
    )
    x, y, z = np.meshgrid(grid.x, grid.y, grid.z, indexing="ij")
    tracers = FourierTracers3D(
        x.ravel(), y.ravel(), z.ravel(), mass=0.125, lattice_shape=(2, 2, 2)
    )
    tracers.x_unwrapped = tracers.x_unwrapped + 1.0

    snapshot = save_solution_snapshot(
        str(tmp_path),
        0.25,
        U=state,
        grid=grid,
        equations=equations,
        tracers=tracers,
    )
    loaded = load_checkpoint(snapshot)

    np.testing.assert_array_equal(loaded["tracer_x"], tracers.x)
    np.testing.assert_array_equal(loaded["tracer_x_unwrapped"], tracers.x_unwrapped)
    np.testing.assert_array_equal(loaded["tracer_lattice_shape"], (2, 2, 2))
    assert float(loaded["tracer_mass"]) == pytest.approx(0.125)


@pytest.mark.parametrize(
    ("restart_data", "message"),
    (
        ({"tracer_y": [0.0], "tracer_z": [0.0]}, "missing tracer_x"),
        (
            {
                "tracer_x": [0.0],
                "tracer_y": [0.0],
                "tracer_z": [0.0],
                "tracer_x_unwrapped": [0.0],
            },
            "all or no unwrapped axes",
        ),
    ),
)
def test_malformed_tracer_restart_metadata_is_rejected(restart_data, message):
    problem = object.__new__(_TracerProblem)
    problem.config = {
        "tracers": {"enabled": True, "num": 1, "layout": "uniform"}
    }
    problem.restart_data = restart_data
    grid = Grid3D(Nx=2, Ny=2, Nz=2, Lx=1.0, Ly=1.0, Lz=1.0, backend="cpu")
    equations = MHDEquations3D()
    shape = (grid.Nz, grid.Ny, grid.Nx)
    state = equations.conservative(
        np.ones(shape),
        np.zeros(shape),
        np.zeros(shape),
        np.zeros(shape),
        np.ones(shape),
        np.zeros(shape),
        np.zeros(shape),
        np.zeros(shape),
    )

    with pytest.raises(ValueError, match=message):
        problem._create_tracers_from_config(grid, state, equations)


def test_rejected_adaptive_step_does_not_advance_tracers():
    class RejectThenAccept:
        def __init__(self):
            self.calls = 0

        def step(self, _rhs, state, dt, _scale):
            self.calls += 1
            if self.calls == 1:
                return AdaptiveStepResult(
                    U_new=state,
                    dt_used=dt,
                    dt_next=0.5 * dt,
                    accepted=False,
                    error=1.0,
                    rejections=1,
                )
            return AdaptiveStepResult(
                U_new=state,
                dt_used=dt,
                dt_next=dt,
                accepted=True,
                error=0.0,
                rejections=0,
            )

    grid = Grid2D(Nx=4, Ny=4, Lx=1.0, Ly=1.0, backend="cpu")
    equations = EulerEquations2D()
    shape = (grid.Ny, grid.Nx)
    state = equations.conservative(
        np.ones(shape), np.ones(shape), np.zeros(shape), np.ones(shape)
    )
    solver = SpectralSolver2D(grid, equations)
    solver.adaptive_enabled = True
    solver.adaptive_stepper = RejectThenAccept()
    tracers = FourierTracers2D(np.array([0.2]), np.array([0.3]))
    tracer_step = tracers.step
    tracer_calls = 0

    def count_tracer_step(*args, **kwargs):
        nonlocal tracer_calls
        tracer_calls += 1
        return tracer_step(*args, **kwargs)

    tracers.step = count_tracer_step
    solver.run(state, t0=0.0, t_end=0.005, tracers=tracers)

    assert solver.adaptive_stepper.calls == 3
    assert tracer_calls == 2
    np.testing.assert_allclose(tracers.x, [0.205])
    np.testing.assert_allclose(tracers.y, [0.3])
