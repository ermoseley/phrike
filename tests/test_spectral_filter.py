import numpy as np

from phrike.equations import EulerEquations1D
from phrike.grid import Grid1D, Grid2D
from phrike.solver import SpectralSolver1D


_FILTER = {
    "enabled": True,
    "p": 8,
    "e_folding_time_at_cutoff": 0.2,
}


def test_grid1d_operators_dealias_without_smooth_attenuation():
    grid = Grid1D(N=24, Lx=1.0, backend="numpy", filter_params=_FILTER)
    x = np.asarray(grid.x)
    retained_mode = grid.N // 3
    removed_mode = retained_mode + 1
    retained = np.sin(2.0 * np.pi * retained_mode * x)
    field = retained + 0.25 * np.cos(2.0 * np.pi * removed_mode * x)

    np.testing.assert_allclose(
        grid.dx1(field),
        2.0 * np.pi * retained_mode * np.cos(2.0 * np.pi * retained_mode * x),
        atol=2.0e-12,
    )
    points = np.array([0.037, 0.219, 0.563, 0.917])
    np.testing.assert_allclose(
        grid.evaluate_fourier_at_points_1d(field, points),
        np.sin(2.0 * np.pi * retained_mode * points),
        atol=5.0e-6,
    )


def test_grid2d_operators_dealias_without_smooth_attenuation():
    grid = Grid2D(
        Nx=24, Ny=24, Lx=1.0, Ly=1.0, backend="numpy", filter_params=_FILTER
    )
    x, y = np.meshgrid(np.asarray(grid.x), np.asarray(grid.y), indexing="xy")
    retained_mode = grid.Nx // 3
    retained = np.sin(2.0 * np.pi * (retained_mode * x + 3.0 * y))
    field = retained + 0.25 * np.cos(2.0 * np.pi * ((retained_mode + 1) * x + y))

    np.testing.assert_allclose(
        grid.dx1(field),
        2.0
        * np.pi
        * retained_mode
        * np.cos(2.0 * np.pi * (retained_mode * x + 3.0 * y)),
        atol=3.0e-12,
    )
    points_x = np.array([0.037, 0.219, 0.563, 0.917])
    points_y = np.array([0.071, 0.314, 0.689, 0.842])
    np.testing.assert_allclose(
        grid.evaluate_fourier_at_points(field, points_x, points_y),
        np.sin(2.0 * np.pi * (retained_mode * points_x + 3.0 * points_y)),
        atol=5.0e-6,
    )


def test_project_dealiased_is_idempotent_and_removes_unretained_modes():
    grid_1d = Grid1D(N=24, Lx=1.0, backend="numpy")
    x = np.asarray(grid_1d.x)
    low_1d = np.sin(2.0 * np.pi * 3.0 * x)
    field_1d = low_1d + 0.5 * np.cos(2.0 * np.pi * 9.0 * x)
    projected_1d = grid_1d.project_dealiased(field_1d)
    np.testing.assert_allclose(projected_1d, low_1d, atol=2.0e-12)
    np.testing.assert_allclose(
        grid_1d.project_dealiased(projected_1d), projected_1d, atol=2.0e-12
    )

    grid_2d = Grid2D(Nx=24, Ny=24, Lx=1.0, Ly=1.0, backend="numpy")
    x, y = np.meshgrid(np.asarray(grid_2d.x), np.asarray(grid_2d.y), indexing="xy")
    low_2d = np.sin(2.0 * np.pi * (3.0 * x + 2.0 * y))
    field_2d = low_2d + 0.5 * np.cos(2.0 * np.pi * (9.0 * x + y))
    projected_2d = grid_2d.project_dealiased(field_2d)
    np.testing.assert_allclose(projected_2d, low_2d, atol=2.0e-12)
    np.testing.assert_allclose(
        grid_2d.project_dealiased(projected_2d), projected_2d, atol=2.0e-12
    )


def test_dissipation_semigroup_and_mean_conservation_in_1d_and_2d():
    grid_1d = Grid1D(N=24, Lx=1.0, backend="numpy", filter_params=_FILTER)
    x = np.asarray(grid_1d.x)
    field_1d = 2.5 + 0.3 * np.cos(4.0 * np.pi * x) + 0.1 * np.sin(14.0 * np.pi * x)
    first_1d = grid_1d.apply_spectral_dissipation(field_1d, 0.07)
    split_1d = grid_1d.apply_spectral_dissipation(first_1d, 0.11)
    whole_1d = grid_1d.apply_spectral_dissipation(field_1d, 0.18)
    np.testing.assert_allclose(split_1d, whole_1d, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(np.mean(whole_1d), np.mean(field_1d), atol=2.0e-13)

    grid_2d = Grid2D(
        Nx=24, Ny=24, Lx=1.0, Ly=1.0, backend="numpy", filter_params=_FILTER
    )
    x, y = np.meshgrid(np.asarray(grid_2d.x), np.asarray(grid_2d.y), indexing="xy")
    field_2d = 2.5 + 0.3 * np.cos(2.0 * np.pi * (2.0 * x + y)) + 0.1 * np.sin(
        2.0 * np.pi * (3.0 * x - 4.0 * y)
    )
    first_2d = grid_2d.apply_spectral_dissipation(field_2d, 0.07)
    split_2d = grid_2d.apply_spectral_dissipation(first_2d, 0.11)
    whole_2d = grid_2d.apply_spectral_dissipation(field_2d, 0.18)
    np.testing.assert_allclose(split_2d, whole_2d, rtol=2.0e-13, atol=2.0e-13)
    np.testing.assert_allclose(np.mean(whole_2d), np.mean(field_2d), atol=2.0e-13)


def test_cutoff_mode_has_the_requested_e_folding_time():
    tau = 0.125
    grid = Grid1D(
        N=24,
        Lx=1.0,
        backend="numpy",
        filter_params={
            "enabled": True,
            "p": 4,
            "e_folding_time_at_cutoff": tau,
        },
    )
    mode = grid.N // 3
    field = np.cos(2.0 * np.pi * mode * np.asarray(grid.x))
    np.testing.assert_allclose(
        grid.apply_spectral_dissipation(field, tau),
        np.exp(-1.0) * field,
        atol=2.0e-12,
    )


def test_solver_dissipation_is_independent_of_step_partition_for_static_flow():
    grid = Grid1D(N=24, Lx=1.0, backend="numpy", filter_params=_FILTER)
    equations = EulerEquations1D(gamma=1.4)
    mode = grid.N // 3
    rho = 1.0 + 0.1 * np.cos(2.0 * np.pi * mode * np.asarray(grid.x))
    initial = equations.conservative(rho, np.zeros_like(rho), np.ones_like(rho))

    one_step = SpectralSolver1D(grid, equations, scheme="rk2").step(initial, 0.18)
    split_solver = SpectralSolver1D(grid, equations, scheme="rk2")
    two_steps = split_solver.step(initial, 0.07)
    two_steps = split_solver.step(two_steps, 0.11)

    np.testing.assert_allclose(two_steps, one_step, rtol=3.0e-13, atol=3.0e-13)
