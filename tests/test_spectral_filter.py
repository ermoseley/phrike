import numpy as np
import pytest

from phrike.adaptive import AdaptiveStepResult
from phrike.equations import EulerEquations1D
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.solver import SpectralSolver1D


_FILTER = {
    "enabled": True,
    "order": 8,
    "e_folding_time_at_cutoff": 0.2,
}


def test_grid1d_operators_dealias_without_smooth_attenuation():
    grid = Grid1D(N=24, Lx=1.0, backend="numpy", filter_params=_FILTER)
    x = np.asarray(grid.x)
    retained_mode = (grid.N - 1) // 3
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
    retained_mode = (grid.Nx - 1) // 3
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
    field_1d = low_1d + 0.5 * np.cos(2.0 * np.pi * 8.0 * x)
    projected_1d = grid_1d.project_dealiased(field_1d)
    np.testing.assert_allclose(projected_1d, low_1d, atol=2.0e-12)
    np.testing.assert_allclose(
        grid_1d.project_dealiased(projected_1d), projected_1d, atol=2.0e-12
    )

    grid_2d = Grid2D(Nx=24, Ny=24, Lx=1.0, Ly=1.0, backend="numpy")
    x, y = np.meshgrid(np.asarray(grid_2d.x), np.asarray(grid_2d.y), indexing="xy")
    low_2d = np.sin(2.0 * np.pi * (3.0 * x + 2.0 * y))
    field_2d = low_2d + 0.5 * np.cos(2.0 * np.pi * (8.0 * x + y))
    projected_2d = grid_2d.project_dealiased(field_2d)
    np.testing.assert_allclose(projected_2d, low_2d, atol=2.0e-12)
    np.testing.assert_allclose(
        grid_2d.project_dealiased(projected_2d), projected_2d, atol=2.0e-12
    )


def test_exact_two_thirds_rule_removes_divisible_by_three_boundary_alias():
    grid = Grid1D(N=24, Lx=1.0, backend="numpy")
    x = np.asarray(grid.x)
    unsafe_boundary_mode = grid.N // 3
    field = np.cos(2.0 * np.pi * unsafe_boundary_mode * x)

    np.testing.assert_allclose(grid.project_dealiased(field), 0.0, atol=2.0e-12)
    np.testing.assert_allclose(grid.convolve(field, field), 0.5, atol=2.0e-12)


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


def test_top_band_viscosity_leaves_low_modes_exact_and_damps_cutoff_one_efold():
    tau = 0.125
    grid = Grid1D(
        N=24,
        Lx=1.0,
        backend="numpy",
        filter_params={
            "enabled": True,
            "order": 2,
            "onset_fraction": 0.75,
            "e_folding_time_at_cutoff": tau,
        },
    )
    cutoff = (grid.N - 1) // 3
    low_mode = 4
    x = np.asarray(grid.x)
    low = np.sin(2.0 * np.pi * low_mode * x)
    high = np.cos(2.0 * np.pi * cutoff * x)
    np.testing.assert_allclose(
        grid.apply_spectral_dissipation(low + high, tau),
        low + np.exp(-1.0) * high,
        atol=2.0e-12,
    )


def test_solver_dissipation_is_independent_of_step_partition_for_static_flow():
    grid = Grid1D(N=24, Lx=1.0, backend="numpy", filter_params=_FILTER)
    equations = EulerEquations1D(gamma=1.4)
    mode = (grid.N - 1) // 3
    rho = 1.0 + 0.1 * np.cos(2.0 * np.pi * mode * np.asarray(grid.x))
    initial = equations.conservative(rho, np.zeros_like(rho), np.ones_like(rho))

    one_step = SpectralSolver1D(grid, equations, scheme="rk2").step(initial, 0.18)
    split_solver = SpectralSolver1D(grid, equations, scheme="rk2")
    two_steps = split_solver.step(initial, 0.07)
    two_steps = split_solver.step(two_steps, 0.11)

    np.testing.assert_allclose(two_steps, one_step, rtol=3.0e-13, atol=3.0e-13)


def test_grid3d_operators_and_tracer_sampling_use_only_hard_dealiasing():
    grid = Grid3D(
        Nx=12,
        Ny=12,
        Nz=12,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        backend="numpy",
        filter_params=_FILTER,
    )
    z, y, x = np.meshgrid(
        np.asarray(grid.z),
        np.asarray(grid.y),
        np.asarray(grid.x),
        indexing="ij",
    )
    retained_mode = (grid.Nx - 1) // 3
    phase = 2.0 * np.pi * (retained_mode * x + 2.0 * y + z)
    retained = np.sin(phase)
    field = retained + 0.25 * np.cos(
        2.0 * np.pi * ((retained_mode + 1) * x + y + z)
    )

    np.testing.assert_allclose(
        grid.dx1(field),
        2.0 * np.pi * retained_mode * np.cos(phase),
        atol=2.0e-11,
    )
    points_x = np.array([0.037, 0.219, 0.563, 0.917])
    points_y = np.array([0.071, 0.314, 0.689, 0.842])
    points_z = np.array([0.109, 0.287, 0.619, 0.931])
    np.testing.assert_allclose(
        grid.evaluate_fourier_at_points(
            field, points_x, points_y, points_z
        ),
        np.sin(
            2.0
            * np.pi
            * (
                retained_mode * points_x
                + 2.0 * points_y
                + points_z
            )
        ),
        atol=8.0e-6,
    )


def test_grid3d_projector_semigroup_and_mean_conservation():
    grid = Grid3D(
        Nx=12,
        Ny=12,
        Nz=12,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        backend="numpy",
        filter_params=_FILTER,
    )
    z, y, x = np.meshgrid(
        np.asarray(grid.z),
        np.asarray(grid.y),
        np.asarray(grid.x),
        indexing="ij",
    )
    low = 2.5 + 0.3 * np.cos(2.0 * np.pi * (2.0 * x + y + z))
    field = low + 0.1 * np.sin(2.0 * np.pi * (5.0 * x + y + z))
    projected = grid.project_dealiased(field)
    np.testing.assert_allclose(projected, low, atol=3.0e-12)
    np.testing.assert_allclose(
        grid.project_dealiased(projected), projected, atol=3.0e-12
    )

    split = grid.apply_spectral_dissipation(
        grid.apply_spectral_dissipation(projected, 0.07), 0.11
    )
    whole = grid.apply_spectral_dissipation(projected, 0.18)
    np.testing.assert_allclose(split, whole, rtol=3.0e-13, atol=3.0e-13)
    np.testing.assert_allclose(np.mean(whole), np.mean(projected), atol=3.0e-13)


@pytest.mark.parametrize("obsolete", ["alpha", "interval"])
def test_fourier_dissipation_rejects_obsolete_settings(obsolete):
    with pytest.raises(ValueError, match=obsolete):
        Grid1D(
            N=16,
            Lx=1.0,
            backend="numpy",
            filter_params={"enabled": True, obsolete: 1.0},
        )


def test_legendre_modal_filter_has_an_independent_configuration():
    grid = Grid1D(
        N=12,
        Lx=1.0,
        basis="legendre",
        bc="dirichlet",
        backend="numpy",
        filter_params={
            "spectral_dissipation": {"enabled": False},
            "modal_filter": {"enabled": True, "order": 8, "alpha": 24.0},
        },
    )
    assert not grid.spectral_dissipation_enabled
    assert grid.modal_filter_enabled
    assert grid.modal_filter_order == 8
    assert grid.modal_filter_alpha == 24.0


@pytest.mark.parametrize(
    ("grid", "forward_name", "inverse_name", "shape"),
    [
        (Grid1D(N=12, Lx=1.0, backend="numpy"), "rfft", "irfft", (12,)),
        (
            Grid2D(Nx=12, Ny=12, Lx=1.0, Ly=1.0, backend="numpy"),
            "fft2",
            "ifft2",
            (12, 12),
        ),
        (
            Grid3D(
                Nx=12,
                Ny=12,
                Nz=12,
                Lx=1.0,
                Ly=1.0,
                Lz=1.0,
                backend="numpy",
            ),
            "fftn",
            "ifftn",
            (12, 12, 12),
        ),
    ],
)
def test_disabled_dissipation_is_a_zero_fft_noop(
    monkeypatch, grid, forward_name, inverse_name, shape
):
    calls = {"forward": 0, "inverse": 0}
    forward = getattr(grid, forward_name)
    inverse = getattr(grid, inverse_name)

    def counted_forward(value):
        calls["forward"] += 1
        return forward(value)

    def counted_inverse(value):
        calls["inverse"] += 1
        return inverse(value)

    monkeypatch.setattr(grid, forward_name, counted_forward)
    monkeypatch.setattr(grid, inverse_name, counted_inverse)
    field = np.ones(shape)
    result = grid.apply_spectral_dissipation(field, 0.25)

    assert result is field
    assert calls == {"forward": 0, "inverse": 0}


def test_disabled_rk2_path_keeps_the_existing_fft_count(monkeypatch):
    grid = Grid1D(N=16, Lx=1.0, backend="numpy")
    equations = EulerEquations1D(gamma=1.4)
    initial = equations.conservative(
        np.ones(grid.N), np.zeros(grid.N), np.ones(grid.N)
    )
    calls = {"forward": 0, "inverse": 0}
    forward = grid.rfft
    inverse = grid.irfft

    def counted_forward(value):
        calls["forward"] += 1
        return forward(value)

    def counted_inverse(value):
        calls["inverse"] += 1
        return inverse(value)

    monkeypatch.setattr(grid, "rfft", counted_forward)
    monkeypatch.setattr(grid, "irfft", counted_inverse)
    SpectralSolver1D(grid, equations, scheme="rk2").step(initial, 0.01)

    # Two RHS derivatives plus the pre-existing accepted-state hard projection.
    assert calls == {"forward": 3, "inverse": 3}


def test_run_projects_the_initial_fourier_state_before_recording():
    grid = Grid1D(N=24, Lx=1.0, backend="numpy")
    equations = EulerEquations1D(gamma=1.4)
    x = np.asarray(grid.x)
    initial = equations.conservative(
        1.0 + 0.1 * np.cos(18.0 * np.pi * x),
        np.zeros(grid.N),
        np.ones(grid.N),
    )
    expected = grid.project_dealiased(initial)
    solver = SpectralSolver1D(grid, equations, scheme="rk2")

    solver.run(initial, t0=0.0, t_end=0.0)

    np.testing.assert_allclose(solver.U, expected, atol=2.0e-13)


class _AdaptiveIdentityStep:
    def __init__(self, accepted):
        self.accepted = accepted

    def step(self, _rhs, state, dt, _scale):
        return AdaptiveStepResult(
            U_new=state,
            dt_used=dt,
            dt_next=0.5 * dt,
            accepted=self.accepted,
            error=0.0 if self.accepted else 1.0,
            rejections=0 if self.accepted else 1,
        )


def test_adaptive_acceptance_completes_split_but_rejection_is_untouched():
    grid = Grid1D(N=24, Lx=1.0, backend="numpy", filter_params=_FILTER)
    equations = EulerEquations1D(gamma=1.4)
    mode = (grid.N - 1) // 3
    rho = 1.0 + 0.1 * np.cos(2.0 * np.pi * mode * np.asarray(grid.x))
    initial = equations.conservative(rho, np.zeros_like(rho), np.ones_like(rho))
    solver = SpectralSolver1D(grid, equations, scheme="rk2")
    solver.adaptive_enabled = True

    solver.adaptive_stepper = _AdaptiveIdentityStep(accepted=True)
    accepted = solver.step(initial, 0.18)
    expected = grid.apply_spectral_dissipation(initial, 0.18, project=True)
    np.testing.assert_allclose(accepted, expected, rtol=3.0e-13, atol=3.0e-13)

    solver.adaptive_stepper = _AdaptiveIdentityStep(accepted=False)
    rejected = solver.step(initial, 0.18)
    assert rejected is initial
    np.testing.assert_array_equal(rejected, initial)


def test_strang_split_with_rk2_is_second_order_for_smooth_flow():
    grid = Grid1D(
        N=32,
        Lx=1.0,
        backend="numpy",
        filter_params={
            "enabled": True,
            "order": 8,
            "e_folding_time_at_cutoff": 0.01,
        },
    )
    equations = EulerEquations1D(gamma=1.4)
    x = np.asarray(grid.x)
    initial = equations.conservative(
        1.0 + 0.15 * np.sin(6.0 * np.pi * x) + 0.04 * np.cos(14.0 * np.pi * x),
        0.2 + 0.05 * np.cos(4.0 * np.pi * x),
        1.0 + 0.1 * np.sin(8.0 * np.pi * x),
    )
    t_end = 0.05

    def integrate(steps, scheme):
        solver = SpectralSolver1D(grid, equations, scheme=scheme)
        state = initial.copy()
        dt = t_end / steps
        for _ in range(steps):
            state = solver.step(state, dt)
        return state

    reference = integrate(128, "rk4")
    coarse_error = np.sqrt(np.mean((integrate(4, "rk2") - reference) ** 2))
    fine_error = np.sqrt(np.mean((integrate(8, "rk2") - reference) ** 2))
    assert coarse_error / fine_error > 3.7


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_numpy_torch_dissipation_parity(device):
    torch = pytest.importorskip("torch")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("Metal/MPS is unavailable")
    numpy_grid = Grid2D(
        Nx=16,
        Ny=16,
        Lx=1.0,
        Ly=1.0,
        backend="numpy",
        filter_params=_FILTER,
    )
    torch_grid = Grid2D(
        Nx=16,
        Ny=16,
        Lx=1.0,
        Ly=1.0,
        backend="torch",
        torch_device=device,
        precision="single" if device == "mps" else "double",
        filter_params=_FILTER,
    )
    x, y = np.meshgrid(
        np.asarray(numpy_grid.x), np.asarray(numpy_grid.y), indexing="xy"
    )
    field = 2.5 + 0.3 * np.cos(2.0 * np.pi * (3.0 * x + 2.0 * y))
    expected = numpy_grid.apply_spectral_dissipation(field, 0.17, project=True)
    tensor = torch.as_tensor(
        field, dtype=torch_grid.x.dtype, device=torch_grid.x.device
    )
    actual = torch_grid.apply_spectral_dissipation(
        tensor, 0.17, project=True
    ).detach().cpu().numpy()
    tolerance = 2.0e-5 if device == "mps" else 3.0e-13
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
