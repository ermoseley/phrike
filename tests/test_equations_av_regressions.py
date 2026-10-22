"""Regressions for Euler diagnostics, wave speed, artificial viscosity and acoustics."""

import numpy as np
import pytest

from phrike.acoustic import AcousticParams, acoustic_exact, acoustic_ic
from phrike.artificial_viscosity import (
    ArtificialViscosityCoefficient,
    ArtificialViscosityConfig,
    GradientBasedSensor,
    SpectralArtificialViscosity,
)
from phrike.equations import (
    EulerEquations1D,
    EulerEquations2D,
    EulerEquations3D,
    MHDEquations1D,
    MHDEquations2D,
    MHDEquations3D,
    _max_wave_speed2d_kernel,
    _max_wave_speed3d_kernel,
    _max_wave_speed_kernel,
)
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.visualization import plot_fields


def _spectral_dx(field, length):
    n = field.size
    wavenumber = 2.0 * np.pi * np.fft.fftfreq(n, d=length / n)
    return np.fft.ifft(1j * wavenumber * np.fft.fft(field)).real


def _linearised_residual(params, t, n=64):
    """Max residual of the linearised Euler operator on acoustic_exact."""
    x = np.linspace(0.0, params.Lx, n, endpoint=False)
    step = 1.0e-6
    ahead = acoustic_exact(x, t + step, params)
    behind = acoustic_exact(x, t - step, params)
    time_derivative = tuple((p - m) / (2.0 * step) for p, m in zip(ahead, behind))
    rho, velocity, pressure = acoustic_exact(x, t, params)
    dx_rho = _spectral_dx(rho, params.Lx)
    dx_u = _spectral_dx(velocity, params.Lx)
    dx_p = _spectral_dx(pressure, params.Lx)
    sound2 = params.gamma * params.p0 / params.rho0
    residuals = (
        time_derivative[0] + params.u0 * dx_rho + params.rho0 * dx_u,
        time_derivative[1] + params.u0 * dx_u + dx_p / params.rho0,
        time_derivative[2] + params.u0 * dx_p + params.rho0 * sound2 * dx_u,
    )
    return max(float(np.max(np.abs(item))) for item in residuals)


def _hand_eigenmode(x, t, params):
    sound = np.sqrt(params.gamma * params.p0 / params.rho0)
    sign = 1.0 if params.direction == "right" else -1.0
    wavenumber = 2.0 * np.pi * params.mode_m / params.Lx
    phase = wavenumber * (x - (params.u0 + sign * sound) * t)
    amplitude = params.amplitude * params.rho0 if params.fractional else params.amplitude
    density_perturbation = amplitude * np.sin(phase)
    return (
        params.rho0 + density_perturbation,
        params.u0 + sign * sound * density_perturbation / params.rho0,
        params.p0 + sound * sound * density_perturbation,
    )


@pytest.mark.parametrize("direction", ["right", "left"])
@pytest.mark.parametrize("background", [0.0, 0.4])
def test_acoustic_exact_is_the_linear_eigenmode(direction, background):
    params = AcousticParams(
        rho0=1.0, p0=1.0, u0=background, amplitude=1.0e-4,
        mode_m=1, Lx=1.0, direction=direction, gamma=1.4,
    )
    x = np.linspace(0.0, params.Lx, 64, endpoint=False)
    for time in (0.0, 0.2):
        got = acoustic_exact(x, time, params)
        expected = _hand_eigenmode(x, time, params)
        for field, reference in zip(got, expected):
            np.testing.assert_allclose(field, reference, atol=1e-12, rtol=1e-12)
    assert _linearised_residual(params, 0.2) < 1e-8

    state = acoustic_ic(x, params)
    rho, velocity, pressure = acoustic_exact(x, 0.0, params)
    reconstructed = EulerEquations1D(gamma=params.gamma).conservative(rho, velocity, pressure)
    np.testing.assert_allclose(state, reconstructed, atol=1e-12, rtol=1e-12)


def test_acoustic_ic_velocity_follows_direction():
    x = np.linspace(0.0, 1.0, 32, endpoint=False)
    right = AcousticParams(u0=0.2, amplitude=1.0e-3, direction="right")
    left = AcousticParams(u0=0.2, amplitude=1.0e-3, direction="left")
    eq = EulerEquations1D(gamma=right.gamma)
    _, u_right, _, _ = eq.primitive(acoustic_ic(x, right))
    _, u_left, _, _ = eq.primitive(acoustic_ic(x, left))
    np.testing.assert_allclose(acoustic_ic(x, right)[0], acoustic_ic(x, left)[0])
    np.testing.assert_allclose(u_left - 0.2, -(u_right - 0.2), atol=1e-15)
    with pytest.raises(ValueError, match="direction"):
        acoustic_exact(x, 0.0, AcousticParams(direction="sideways"))


def _uniform(kind):
    """Uniform primitive state and the expected max wave speed (B = 0)."""
    gamma = 1.4
    sound = float(np.sqrt(gamma))
    ux, uy, uz = 0.25, 0.1, -0.05
    classes = {
        "euler1d": EulerEquations1D,
        "euler2d": EulerEquations2D,
        "euler3d": EulerEquations3D,
        "mhd1d": MHDEquations1D,
        "mhd2d": MHDEquations2D,
        "mhd3d": MHDEquations3D,
    }
    eq = classes[kind](gamma=gamma)
    if kind == "euler1d":
        state = eq.conservative(np.ones(16), np.full(16, ux), np.ones(16))
        speed = abs(ux) + sound
    elif kind == "euler2d":
        ones = np.ones((6, 8))
        state = eq.conservative(ones, np.full_like(ones, ux), np.full_like(ones, uy), ones)
        speed = max(abs(ux), abs(uy)) + sound
    elif kind == "euler3d":
        ones = np.ones((4, 4, 4))
        state = eq.conservative(
            ones, np.full_like(ones, ux), np.full_like(ones, uy), np.full_like(ones, uz), ones
        )
        speed = max(abs(ux), abs(uy), abs(uz)) + sound
    elif kind == "mhd1d":
        ones = np.ones(16)
        zeros = np.zeros(16)
        state = eq.conservative(ones, np.full(16, ux), zeros, zeros, ones, zeros, zeros, zeros)
        speed = abs(ux) + sound
    elif kind == "mhd2d":
        ones = np.ones((6, 8))
        zeros = np.zeros_like(ones)
        state = eq.conservative(
            ones, np.full_like(ones, ux), np.full_like(ones, uy), zeros, ones, zeros, zeros, zeros
        )
        speed = max(abs(ux), abs(uy)) + sound
    else:
        ones = np.ones((4, 4, 4))
        zeros = np.zeros_like(ones)
        state = eq.conservative(
            ones, np.full_like(ones, ux), np.full_like(ones, uy), np.full_like(ones, uz),
            ones, zeros, zeros, zeros,
        )
        speed = max(abs(ux), abs(uy), abs(uz)) + sound
    return eq, state, speed


def _write_cell(state, variable, value):
    out = state.copy()
    flat = out.reshape(out.shape[0], -1)
    flat[variable, flat.shape[1] // 3] = value
    return out


@pytest.mark.parametrize(
    "kind", ["euler1d", "euler2d", "euler3d", "mhd1d", "mhd2d", "mhd3d"]
)
def test_max_wave_speed_propagates_nonfinite_cells(kind):
    eq, state, speed = _uniform(kind)
    assert eq.max_wave_speed(state) == pytest.approx(speed)

    poisoned = _write_cell(state, 0, np.nan)
    assert np.isnan(eq.max_wave_speed(poisoned))

    energy = 2 if kind == "euler1d" else 3 if kind == "euler2d" else 4
    # Euler's sound speed is NaN for a negative thermal pressure. The MHD
    # fast-speed formula floors the discriminant, so a finite negative
    # pressure stays a finite speed; a non-finite conserved cell does not.
    if kind.startswith("euler"):
        assert np.isnan(eq.max_wave_speed(_write_cell(state, energy, -1.0)))
    # Inf energy is Inf for Euler and for the isotropic MHD bound, and NaN
    # for the 1D MHD fast speed (inf * 0 in the Alfven term). Either way the
    # bad cell is not dropped.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        assert not np.isfinite(eq.max_wave_speed(_write_cell(state, energy, np.inf)))


def test_numba_and_python_max_wave_speed_agree_on_nan():
    eq = EulerEquations1D()
    x = np.linspace(0.0, 1.0, 32, endpoint=False)
    velocity = 0.2 + 0.5 * np.sin(2.0 * np.pi * x)
    state = eq.conservative(np.ones_like(x), velocity, np.ones_like(x))
    rho, u, _p, sound = eq.primitive(state)
    expected = float(np.max(np.abs(u) + sound))
    assert eq.max_wave_speed(state) == pytest.approx(expected)
    assert _max_wave_speed_kernel.py_func(state, eq.gamma) == pytest.approx(expected)

    state[0, 4] = np.nan
    assert np.isnan(_max_wave_speed_kernel(state, eq.gamma))
    assert np.isnan(_max_wave_speed_kernel.py_func(state, eq.gamma))

    eq2 = EulerEquations2D()
    ones = np.ones((8, 8))
    state2 = eq2.conservative(ones, 0.2 * ones, -0.4 * ones, ones)
    state2[0, 1, 2] = np.nan
    assert np.isnan(_max_wave_speed2d_kernel(state2, eq2.gamma))
    assert np.isnan(_max_wave_speed2d_kernel.py_func(state2, eq2.gamma))

    eq3 = EulerEquations3D()
    ones3 = np.ones((4, 4, 4))
    state3 = eq3.conservative(ones3, 0.2 * ones3, ones3 * 0.1, -0.3 * ones3, ones3)
    state3[1, 0, 0, 1] = np.nan
    assert np.isnan(_max_wave_speed3d_kernel(state3, eq3.gamma))
    assert np.isnan(_max_wave_speed3d_kernel.py_func(state3, eq3.gamma))


def test_torch_max_wave_speed_matches_numpy_on_nan():
    torch = pytest.importorskip("torch")
    eq = EulerEquations2D()
    ones = np.ones((6, 8))
    state = eq.conservative(ones, 0.3 * ones, -0.2 * ones, ones)
    tensor = torch.tensor(state, dtype=torch.float64)
    assert eq.max_wave_speed(tensor) == pytest.approx(eq.max_wave_speed(state))
    tensor = tensor.clone()
    tensor[0, 1, 1] = float("nan")
    assert np.isnan(eq.max_wave_speed(tensor))

    mhd = MHDEquations1D(gamma=1.4)
    _eq, mhd_state, speed = _uniform("mhd1d")
    mhd_tensor = torch.tensor(mhd_state, dtype=torch.float64)
    assert mhd.max_wave_speed(mhd_tensor) == pytest.approx(speed)
    mhd_tensor = mhd_tensor.clone()
    mhd_tensor[0, 2] = float("nan")
    assert np.isnan(mhd.max_wave_speed(mhd_tensor))


def test_2d_euler_conserved_quantities_include_both_momenta():
    rng = np.random.default_rng(1)
    rho = rng.uniform(0.5, 1.5, (16, 16))
    ux = rng.uniform(-1.0, 1.0, rho.shape)
    uy = rng.uniform(-1.0, 1.0, rho.shape)
    pressure = rng.uniform(0.2, 2.0, rho.shape)
    eq = EulerEquations2D()
    state = eq.conservative(rho, ux, uy, pressure)
    totals = eq.conserved_quantities(state)
    assert totals["mass"] == pytest.approx(rho.sum())
    assert totals["momentum_x"] == pytest.approx((rho * ux).sum())
    assert totals["momentum_y"] == pytest.approx((rho * uy).sum())
    assert totals["momentum"] == pytest.approx(totals["momentum_x"])
    assert totals["energy"] == pytest.approx(state[3].sum())
    # The old unpacking used uy as pressure, so the energy was not sum(E).
    assert totals["energy"] != pytest.approx((rho * ux).sum())


def test_conserved_quantities_are_unweighted_grid_sums():
    eq = EulerEquations1D()
    state = eq.conservative(np.ones(32), np.zeros(32), np.ones(32))
    assert eq.conserved_quantities(state)["mass"] == pytest.approx(32.0)
    eq2 = EulerEquations2D()
    ones = np.ones((8, 16))
    state2 = eq2.conservative(ones, np.zeros_like(ones), np.zeros_like(ones), ones)
    assert eq2.conserved_quantities(state2)["mass"] == pytest.approx(ones.size)


def _relative_gradient(grid, field, epsilon):
    if hasattr(grid, "dz1"):
        grad = np.sqrt(grid.dx1(field) ** 2 + grid.dy1(field) ** 2 + grid.dz1(field) ** 2)
    elif hasattr(grid, "dy1"):
        grad = np.sqrt(grid.dx1(field) ** 2 + grid.dy1(field) ** 2)
    else:
        grad = np.abs(grid.dx1(field))
    return grad / (np.abs(field) + epsilon)


def test_pressure_sensor_uses_pressure_in_every_layout():
    epsilon = 1.0e-12
    config = ArtificialViscosityConfig(
        sensor_variable="pressure", epsilon=epsilon, s_ref=0.5
    )
    sensor = GradientBasedSensor(config)

    grid = Grid1D(32, 1.0, backend="numpy", dealias=False)
    x = np.asarray(grid.x)
    pressure = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
    state = EulerEquations1D().conservative(np.ones_like(x), np.zeros_like(x), pressure)
    got = sensor.compute_sensor(state, grid, EulerEquations1D())
    np.testing.assert_allclose(got, _relative_gradient(grid, pressure, epsilon), atol=1e-12)
    sound = np.sqrt(1.4 * pressure)
    assert np.max(np.abs(got - _relative_gradient(grid, sound, epsilon))) > 1.0e-2

    grid2 = Grid2D(12, 16, 1.0, 1.0, backend="numpy", dealias=False)
    X, Y = grid2.xy_mesh()
    uy = 0.8 * np.sin(2.0 * np.pi * X)
    pressure2 = np.ones_like(X)
    eq2 = EulerEquations2D()
    state2 = eq2.conservative(np.ones_like(X), np.zeros_like(X), uy, pressure2)
    assert np.max(np.abs(sensor.compute_sensor(state2, grid2, eq2))) < 1.0e-10
    pressure2 = 1.0 + 0.25 * np.sin(2.0 * np.pi * Y)
    state2 = eq2.conservative(np.ones_like(X), np.zeros_like(X), uy, pressure2)
    np.testing.assert_allclose(
        sensor.compute_sensor(state2, grid2, eq2),
        _relative_gradient(grid2, pressure2, epsilon),
        atol=1e-11,
    )

    grid3 = Grid3D(8, 8, 8, 1.0, 1.0, 1.0, backend="numpy", dealias=False)
    X, Y, Z = grid3.xyz_mesh()
    eq3 = EulerEquations3D()
    uy3 = 0.7 * np.sin(2.0 * np.pi * Y)
    state3 = eq3.conservative(
        np.ones_like(X), np.zeros_like(X), uy3, np.zeros_like(X), np.ones_like(X)
    )
    assert np.max(np.abs(sensor.compute_sensor(state3, grid3, eq3))) < 1.0e-10

    gm = Grid1D(32, 1.0, backend="numpy", dealias=False)
    gx = np.asarray(gm.x)
    mhd = MHDEquations1D()
    by = 0.4 * np.sin(2.0 * np.pi * gx)
    uy_m = 0.5 * np.cos(2.0 * np.pi * gx)
    zeros = np.zeros_like(gx)
    ones = np.ones_like(gx)
    quiet = mhd.conservative(ones, zeros, uy_m, zeros, ones, zeros, by, zeros)
    assert np.max(np.abs(sensor.compute_sensor(quiet, gm, mhd))) < 1.0e-10
    thermal = 1.0 + 0.2 * np.sin(2.0 * np.pi * gx)
    shocked = mhd.conservative(ones, zeros, uy_m, zeros, thermal, zeros, by, zeros)
    # Rebuild energy for the perturbed pressure. conservative() already did.
    np.testing.assert_allclose(
        sensor.compute_sensor(shocked, gm, mhd),
        _relative_gradient(gm, thermal, epsilon),
        atol=1e-11,
    )


def _peculiar_speed_sensor(grid, components):
    slopes = []
    fluctuations = []
    for velocity in components:
        if hasattr(grid, "dz1"):
            slope = np.sqrt(grid.dx1(velocity) ** 2 + grid.dy1(velocity) ** 2 + grid.dz1(velocity) ** 2)
        elif hasattr(grid, "dy1"):
            slope = np.sqrt(grid.dx1(velocity) ** 2 + grid.dy1(velocity) ** 2)
        else:
            slope = np.abs(grid.dx1(velocity))
        slopes.append(slope)
        fluctuations.append(velocity - np.mean(velocity))
    magnitude = np.sqrt(sum(item * item for item in slopes))
    reference = np.sqrt(np.mean(sum(item * item for item in fluctuations)))
    return magnitude / reference


def test_velocity_sensor_uses_components_and_a_global_scale():
    config = ArtificialViscosityConfig(sensor_variable="velocity_magnitude", epsilon=1.0e-12)
    sensor = GradientBasedSensor(config)
    grid = Grid1D(32, 1.0, backend="numpy", dealias=False)
    x = np.asarray(grid.x)
    wave = np.sin(2.0 * np.pi * x)
    eq = EulerEquations1D()
    resting = sensor.compute_sensor(
        eq.conservative(np.ones_like(x), wave, np.ones_like(x)), grid, eq
    )
    boosted = sensor.compute_sensor(
        eq.conservative(np.ones_like(x), wave + 10.0, np.ones_like(x)), grid, eq
    )
    expected = _peculiar_speed_sensor(grid, (wave,))
    np.testing.assert_allclose(resting, expected, atol=1e-10)
    np.testing.assert_allclose(boosted, resting, atol=1e-10)
    # x = 0 is a stagnation point of the unboosted wave. The sensor stays O(k).
    assert np.isfinite(resting).all()
    assert resting[0] == pytest.approx(2.0 * np.pi * np.sqrt(2.0))
    uniform = sensor.compute_sensor(
        eq.conservative(np.ones_like(x), np.full_like(x, 4.0), np.ones_like(x)), grid, eq
    )
    assert np.max(np.abs(uniform)) == 0.0

    grid2 = Grid2D(12, 16, 1.0, 2.0, backend="numpy", dealias=False)
    X, Y = grid2.xy_mesh()
    ux = 0.3 * np.sin(2.0 * np.pi * X)
    uy = 0.5 * np.sin(2.0 * np.pi * Y / 2.0)
    eq2 = EulerEquations2D()
    state = eq2.conservative(np.ones_like(X), ux, uy, np.ones_like(X))
    got = sensor.compute_sensor(state, grid2, eq2)
    np.testing.assert_allclose(got, _peculiar_speed_sensor(grid2, (ux, uy)), atol=1e-10)
    shifted = eq2.conservative(np.ones_like(X), ux + 4.0, uy - 2.0, np.ones_like(X))
    np.testing.assert_allclose(sensor.compute_sensor(shifted, grid2, eq2), got, atol=1e-10)

    grid3 = Grid3D(8, 8, 8, 1.0, 1.0, 1.0, backend="numpy", dealias=False)
    X, Y, Z = grid3.xyz_mesh()
    ux = 0.2 * np.sin(2.0 * np.pi * X)
    uy = 0.3 * np.cos(2.0 * np.pi * Y)
    uz = 0.4 * np.sin(2.0 * np.pi * Z)
    eq3 = EulerEquations3D()
    state3 = eq3.conservative(np.ones_like(X), ux, uy, uz, np.ones_like(X))
    np.testing.assert_allclose(
        sensor.compute_sensor(state3, grid3, eq3),
        _peculiar_speed_sensor(grid3, (ux, uy, uz)),
        atol=1e-10,
    )


def test_s_ref_is_applied_once():
    grid = Grid1D(64, 1.0, backend="numpy", dealias=False)
    x = np.asarray(grid.x)
    rho = 1.0 + 0.05 * np.sin(2.0 * np.pi * x)
    eq = EulerEquations1D()
    state = eq.conservative(rho, np.zeros_like(x), np.ones_like(x))
    config = ArtificialViscosityConfig(
        mode="sensor", sensor_variable="density", s_ref=0.5, s_min=0.01,
        p=2.0, nu_max=0.2, epsilon=1.0e-12,
    )
    raw = GradientBasedSensor(config).compute_sensor(state, grid, eq)
    np.testing.assert_allclose(raw, _relative_gradient(grid, rho, config.epsilon), atol=1e-12)
    # The duplicate division used to clip this peak down by another factor of s_ref.
    assert np.max(raw) > 0.3

    viscosity = ArtificialViscosityCoefficient(config).compute_viscosity(raw)
    active = np.where(raw > config.s_min, raw, 0.0)
    once = config.nu_max * np.clip(active / config.s_ref, 0.0, 1.0) ** config.p
    normalized = np.clip(raw / config.s_ref, 0.0, 1.0)
    twice = config.nu_max * np.clip(
        np.where(normalized > config.s_min, normalized, 0.0) / config.s_ref, 0.0, 1.0
    ) ** config.p
    np.testing.assert_allclose(viscosity, once, atol=1e-14)
    assert np.max(np.abs(viscosity - twice)) > 1.0e-2


def test_variable_weights_follow_the_state_layout():
    grid = Grid2D(16, 12, 1.0, 1.0, backend="numpy", dealias=False)
    X, Y = grid.xy_mesh()
    eq = EulerEquations2D()
    state = eq.conservative(
        np.ones_like(X),
        0.2 * np.sin(2.0 * np.pi * X),
        0.3 * np.sin(2.0 * np.pi * Y),
        1.0 + 0.1 * np.sin(2.0 * np.pi * X),
    )
    config = ArtificialViscosityConfig(
        mode="constant", nu_constant=0.01,
        variable_weights={
            "density": 1.0, "momentum_x": 1.0, "momentum_y": 1.0,
            "momentum_z": 10.0, "energy": 0.0,
        },
    )
    terms = SpectralArtificialViscosity(config).compute_viscosity_terms(state, grid, eq)
    assert np.max(np.abs(terms[3])) == pytest.approx(0.0, abs=1e-14)
    assert np.max(np.abs(terms[1])) > 1.0e-4

    grid3 = Grid3D(8, 8, 8, 1.0, 1.0, 1.0, backend="numpy", dealias=False)
    X, Y, Z = grid3.xyz_mesh()
    eq3 = EulerEquations3D()
    state3 = eq3.conservative(
        np.ones_like(X), np.zeros_like(X), np.zeros_like(X),
        0.25 * np.sin(2.0 * np.pi * Z), 1.0 + 0.1 * np.sin(2.0 * np.pi * X),
    )
    config3 = ArtificialViscosityConfig(
        mode="constant", nu_constant=0.01,
        variable_weights={"energy": 0.0, "momentum_z": 1.0},
    )
    terms3 = SpectralArtificialViscosity(config3).compute_viscosity_terms(state3, grid3, eq3)
    assert np.max(np.abs(terms3[4])) == pytest.approx(0.0, abs=1e-14)
    assert np.max(np.abs(terms3[3])) > 1.0e-4

    grid1 = Grid1D(32, 1.0, backend="numpy", dealias=False)
    gx = np.asarray(grid1.x)
    mhd = MHDEquations1D()
    ones = np.ones_like(gx)
    zeros = np.zeros_like(gx)
    magnetic = mhd.conservative(
        ones, zeros, zeros, zeros, ones,
        0.2 * np.sin(2.0 * np.pi * gx), 0.3 * np.cos(2.0 * np.pi * gx), zeros,
    )
    default = ArtificialViscosityConfig(mode="constant", nu_constant=0.01)
    assert default.variable_weights["bx"] == 1.0
    assert default.variable_weights["by"] == 1.0
    assert default.variable_weights["bz"] == 1.0
    default_terms = SpectralArtificialViscosity(default).compute_viscosity_terms(
        magnetic, grid1, mhd
    )
    assert np.max(np.abs(default_terms[6])) > 1.0e-4  # By is diffused by default

    caller = {"density": 2.0}
    partial = ArtificialViscosityConfig(variable_weights=caller)
    assert "bx" not in caller
    assert partial.variable_weights["bx"] == 1.0
    silent = ArtificialViscosityConfig(
        mode="constant", nu_constant=0.01, variable_weights={"by": 0.0}
    )
    silent_terms = SpectralArtificialViscosity(silent).compute_viscosity_terms(
        magnetic, grid1, mhd
    )
    assert np.max(np.abs(silent_terms[6])) == pytest.approx(0.0, abs=1e-14)
    assert np.max(np.abs(silent_terms[5])) > 1.0e-4


def test_non_string_viscosity_mode_raises():
    with pytest.raises(TypeError, match="mode"):
        ArtificialViscosityConfig(mode=0)
    with pytest.raises(TypeError, match="mode"):
        ArtificialViscosityConfig(mode=None)
    assert ArtificialViscosityConfig(mode="SENSOR").mode == "sensor"
    with pytest.raises(ValueError, match="Invalid mode"):
        ArtificialViscosityConfig(mode="nope")


def test_plot_fields_rejects_mhd_and_plots_1d_euler():
    import matplotlib
    matplotlib.use("Agg", force=True)
    mhd = MHDEquations1D()
    ones = np.ones(8)
    zeros = np.zeros(8)
    state = mhd.conservative(ones, zeros, zeros, zeros, ones, 0.2 * ones, zeros, zeros)
    with pytest.raises(ValueError, match="1D Euler"):
        plot_fields(None, state, mhd)

    eq2 = EulerEquations2D()
    ones2 = np.ones((4, 4))
    with pytest.raises(ValueError, match="1D Euler"):
        plot_fields(None, eq2.conservative(ones2, ones2 * 0, ones2 * 0, ones2), eq2)

    grid = Grid1D(16, 1.0, backend="numpy")
    x = np.asarray(grid.x)
    euler = EulerEquations1D().conservative(np.ones_like(x), np.zeros_like(x), np.ones_like(x))
    plot_fields(grid, euler, EulerEquations1D(), title="steady")


def test_torch_velocity_sensor_matches_numpy():
    torch = pytest.importorskip("torch")
    grid = Grid2D(8, 8, 1.0, 1.0, backend="numpy", dealias=False)
    grid_t = Grid2D(8, 8, 1.0, 1.0, backend="torch", torch_device="cpu", dealias=False)
    X, Y = grid.xy_mesh()
    ux = 0.25 * np.sin(2.0 * np.pi * X)
    uy = 0.4 * np.cos(2.0 * np.pi * Y)
    eq = EulerEquations2D()
    state = eq.conservative(np.ones_like(X), ux, uy, np.ones_like(X))
    config = ArtificialViscosityConfig(sensor_variable="velocity_magnitude")
    numpy_sensor = GradientBasedSensor(config).compute_sensor(state, grid, eq)
    tensor = torch.tensor(state, dtype=torch.float64)
    torch_sensor = GradientBasedSensor(config).compute_sensor(tensor, grid_t, eq)
    np.testing.assert_allclose(torch_sensor.detach().cpu().numpy(), numpy_sensor, atol=1e-10)
