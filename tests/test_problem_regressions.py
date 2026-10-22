"""Regression tests for hydro initial conditions and shipped configs."""

import os

os.environ.setdefault("MPLBACKEND", "Agg")

from pathlib import Path

import numpy as np
import pytest
import yaml

from phrike.equations import EulerEquations1D, EulerEquations2D, EulerEquations3D
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.problems.problem_list.acoustic1d import Acoustic1DProblem
from phrike.problems.problem_list.gaussian_wave1d import GaussianWave1DProblem
from phrike.problems.problem_list.gaussian_wave2d import GaussianWave2DProblem
from phrike.problems.problem_list.gaussian_wave2d_y import GaussianWave2DYProblem
from phrike.problems.problem_list.hse_1d import HSE1DProblem, hse_initial_conditions
from phrike.problems.problem_list.khi2d import kelvin_helmholtz_2d
from phrike.problems.problem_list.rti import rti_initial_conditions
from phrike.problems.problem_list.rti_1d import rti_1d_initial_conditions
from phrike.problems.problem_list.shu_osher1d import ShuOsher1DProblem
from phrike.problems.problem_list.tgv3d import taylor_green_vortex_3d
from phrike.problems.problem_list.turb3d import turbulent_velocity_3d

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path, problem, grid, initial, **extra):
    config = {
        "problem": problem,
        "grid": grid,
        "physics": {"gamma": 1.4},
        "integration": {
            "t0": 0.0,
            "t_end": 0.2,
            "cfl": 0.4,
            "scheme": "rk4",
        },
        "io": {"outdir": str(tmp_path)},
        "monitoring": {"enabled": False},
        "initial_conditions": initial,
    }
    config.update(extra)
    return config


def _shell_slope(ux, uy, uz, kmin, kmax):
    """Least-squares slope of the shell kinetic energy inside the untapered band."""
    velocity_hat = [np.fft.fftn(component) for component in (ux, uy, uz)]
    nz, ny, nx = ux.shape
    nx_mode = np.fft.fftfreq(nx, d=1.0 / nx).reshape(1, 1, nx)
    ny_mode = np.fft.fftfreq(ny, d=1.0 / ny).reshape(1, ny, 1)
    nz_mode = np.fft.fftfreq(nz, d=1.0 / nz).reshape(nz, 1, 1)
    mode = np.sqrt(nx_mode**2 + ny_mode**2 + nz_mode**2)
    power = sum(np.abs(hat) ** 2 for hat in velocity_hat)
    band = kmax - kmin
    k_lo = kmin + 0.1 * band
    k_hi = kmax - 0.1 * band
    shells = []
    energies = []
    for wavenumber in np.arange(np.ceil(k_lo + 1.0), np.floor(k_hi - 1.0) + 1e-9):
        mask = (mode >= wavenumber - 0.5) & (mode < wavenumber + 0.5)
        if int(mask.sum()) < 8:
            continue
        shells.append(wavenumber)
        energies.append(power[mask].sum())
    shells = np.asarray(shells)
    energies = np.asarray(energies)
    slope, _intercept = np.polyfit(np.log(shells), np.log(energies), 1)
    return float(slope)


def test_turb3d_solenoidal_field_and_shell_slope():
    equations = EulerEquations3D(gamma=1.4)
    grid = Grid3D(32, 32, 32, 1.0, 1.0, 1.0, dealias=True, backend="numpy")
    x, y, z = grid.xyz_mesh()
    state = turbulent_velocity_3d(
        x, y, z,
        alpha=0.0,
        spectrum_type="power_law",
        power_law_slope=-5.0 / 3.0,
        kmin=1.0,
        kmax=8.0,
        seed=1,
        vrms=0.1,
    )
    _rho, ux, uy, uz, _p = equations.primitive(state)
    divergence = grid.dx1(ux) + grid.dy1(uy) + grid.dz1(uz)
    rms = float(np.sqrt(np.mean(np.asarray(divergence) ** 2)))
    assert rms < 1e-10

    wide = Grid3D(24, 20, 28, 1.0, 2.0, 0.5, dealias=True, backend="numpy")
    x, y, z = wide.xyz_mesh()
    state = turbulent_velocity_3d(
        x, y, z, alpha=0.0, kmin=1.0, kmax=6.0, seed=2, vrms=0.1,
    )
    _rho, ux, uy, uz, _p = equations.primitive(state)
    divergence = wide.dx1(ux) + wide.dy1(uy) + wide.dz1(uz)
    rms = float(np.sqrt(np.mean(np.asarray(divergence) ** 2)))
    assert rms < 1e-10

    cube = Grid3D(32, 32, 32, 1.0, 1.0, 1.0, dealias=False, backend="numpy")
    x, y, z = cube.xyz_mesh()
    slope_target = -5.0 / 3.0
    state = turbulent_velocity_3d(
        x, y, z,
        alpha=0.0,
        spectrum_type="power_law",
        power_law_slope=slope_target,
        kmin=1.0,
        kmax=14.0,
        seed=0,
        vrms=1.0,
    )
    _rho, ux, uy, uz, _p = equations.primitive(state)
    measured = _shell_slope(ux, uy, uz, 1.0, 14.0)
    assert measured == pytest.approx(slope_target, abs=0.25)


def test_hydrostatic_pressure_uses_node_spacing(tmp_path):
    grid = Grid1D(N=32, Lx=2.0, basis="legendre", bc="dirichlet", backend="numpy")
    state = hse_initial_conditions(
        grid.x, rho_left=1.0, rho_right=1.0, pressure_left=2.5,
        gamma=1.4, gx=-1.0, x0=1.0, sigma=0.01,
    )
    rho, _u, pressure, _a = EulerEquations1D(gamma=1.4).primitive(state)
    x = np.asarray(grid.x)
    expected = 2.5 + (x - x[0]) * (-1.0)
    np.testing.assert_allclose(pressure, expected, atol=1e-12)
    slope = np.diff(pressure) / np.diff(x)
    np.testing.assert_allclose(slope, -1.0, atol=1e-12)
    assert float(np.min(pressure)) > 0.0

    with pytest.raises(ValueError, match="non-positive"):
        hse_initial_conditions(
            grid.x, 1.0, 1.0, 1.0, 1.4, -1.0, 1.0, 0.01,
        )

    problem = HSE1DProblem(config=_config(
        tmp_path,
        "hse_1d",
        {"N": 16, "Lx": 2.0, "basis": "legendre", "bc": "dirichlet"},
        {"left": {"rho": 1.0, "u": 0.0, "p": 2.5}, "right": {"rho": 1.0}},
        gravity={"enabled": True, "gx": -1.0},
    ))
    grid = problem.create_grid(backend="numpy")
    equations = problem.create_equations()
    state = problem.create_initial_conditions(grid)
    solver = problem.get_solver_class()(grid, equations, cfl=problem.cfl)
    problem.create_visualization(solver, 0.0, state)


def test_rti_gravity_sign_and_node_spacing():
    grid = Grid1D(N=32, Lx=2.0, basis="legendre", bc="dirichlet", backend="numpy")
    state = rti_1d_initial_conditions(
        grid.x, 2.0, 1.0, 2.5, 1.4, -1.0, 0.05, 0.0, 1.0,
    )
    rho, _u, pressure, _a = EulerEquations1D(gamma=1.4).primitive(state)
    x = np.asarray(grid.x)
    slope = np.diff(pressure) / np.diff(x)
    rho_mid = 0.5 * (rho[:-1] + rho[1:])
    np.testing.assert_allclose(slope, rho_mid * (-1.0), atol=1e-10)
    assert float(pressure[0]) > float(pressure[-1])
    assert float(np.min(pressure)) > 0.0
    assert float(rho[-1]) > float(rho[0])

    with pytest.raises(ValueError, match="not positive"):
        rti_1d_initial_conditions(
            grid.x, 2.0, 1.0, 2.5, 1.4, 1.0, 0.05, 0.0, 1.0,
        )

    grid2 = Grid2D(
        Nx=8, Ny=24, Lx=1.0, Ly=2.0,
        basis_x="legendre", basis_y="legendre", bc="dirichlet",
        backend="numpy", dealias=False,
    )
    x, y = grid2.xy_mesh()
    state = rti_initial_conditions(
        x, y, 2.0, 1.0, 2.5, 1.4, -1.0, 0.05, 0.0, 1.0,
    )
    rho, _u, _v, pressure = EulerEquations2D(gamma=1.4).primitive(state)
    y_line = np.asarray(y[:, 0])
    slope = np.diff(pressure[:, 0]) / np.diff(y_line)
    rho_mid = 0.5 * (rho[:-1, 0] + rho[1:, 0])
    np.testing.assert_allclose(slope, rho_mid * (-1.0), atol=1e-10)
    assert float(np.min(pressure)) > 0.0


def test_khi_perturbation_is_an_integer_mode():
    grid = Grid2D(
        Nx=64, Ny=64, Lx=1.0, Ly=1.0,
        basis_x="fourier", basis_y="fourier",
        backend="numpy", dealias=False,
    )
    x, y = grid.xy_mesh()
    with pytest.raises(ValueError, match="Lx"):
        kelvin_helmholtz_2d(x, y, perturb_kx=2)
    state = kelvin_helmholtz_2d(
        x, y, perturb_eps=0.01, perturb_kx=2, perturb_sigma=0.05,
        Lx=grid.Lx, Ly=grid.Ly,
    )
    _rho, _ux, uy, _p = EulerEquations2D(gamma=1.4).primitive(state)
    spectrum = np.mean(np.abs(np.fft.fft(uy, axis=-1)), axis=0)
    mode = spectrum[2]
    other = np.concatenate([spectrum[1:2], spectrum[3:32]])
    assert float(other.max() / mode) < 1e-8


def test_tgv_brachet_pressure_and_mach(tmp_path):
    grid = Grid3D(
        16, 16, 16, 2.0 * np.pi, 2.0 * np.pi, 2.0 * np.pi,
        backend="numpy", dealias=False,
    )
    x, y, z = grid.xyz_mesh()
    gamma = 1.4
    p0 = 1.0 / (gamma * 0.1**2)
    state = taylor_green_vortex_3d(x, y, z, rho0=1.0, p0=p0, U0=1.0, k=1, gamma=gamma)
    _rho, _ux, _uy, _uz, pressure = EulerEquations3D(gamma=gamma).primitive(state)
    expected = p0 + (1.0 / 16.0) * (np.cos(2.0 * x) + np.cos(2.0 * y)) * (
        np.cos(2.0 * z) + 2.0
    )
    np.testing.assert_allclose(pressure, expected, atol=1e-12)
    mach = 1.0 / np.sqrt(gamma * p0)
    assert mach == pytest.approx(0.1)

    shipped = yaml.safe_load((ROOT / "configs" / "tgv3d.yaml").read_text())
    assert shipped["initial_conditions"]["mach"] == pytest.approx(0.1)
    assert "artificial_viscosity" not in shipped


def test_acoustic_eigenmode_translates(tmp_path):
    config = _config(
        tmp_path,
        "acoustic1d",
        {"N": 64, "Lx": 1.0, "basis": "fourier", "dealias": True},
        {
            "rho0": 1.0, "u0": 0.0, "p0": 1.0,
            "amplitude": 1.0e-4, "k": 1,
            "wave": "acoustic", "direction": "right",
        },
    )
    problem = Acoustic1DProblem(config=config)
    grid = problem.create_grid(backend="numpy")
    equations = problem.create_equations()
    state = problem.create_initial_conditions(grid)
    rho, velocity, pressure, _sound = equations.primitive(state)
    sound = np.sqrt(problem.gamma * 1.0 / 1.0)
    density_prime = rho - 1.0
    np.testing.assert_allclose(velocity, sound * density_prime, atol=1e-14)
    np.testing.assert_allclose(pressure - 1.0, sound**2 * density_prime, atol=1e-14)

    solver = problem.get_solver_class()(grid, equations, cfl=problem.cfl)
    time = 0.0
    t_end = 0.2
    while time < t_end - 1e-13:
        step = min(solver.compute_dt(state), t_end - time)
        state = solver.step(state, step)
        time += step
    reference = problem.get_analytical_solution(grid, time)
    rho, velocity, pressure, _sound = equations.primitive(state)
    np.testing.assert_allclose(rho, reference["rho"], atol=1e-7)
    np.testing.assert_allclose(velocity, reference["u"], atol=1e-7)
    np.testing.assert_allclose(pressure, reference["p"], atol=1e-7)

    entropy_dir = tmp_path / "entropy"
    entropy = Acoustic1DProblem(config=_config(
        entropy_dir,
        "acoustic1d",
        {"N": 64, "Lx": 1.0, "basis": "fourier", "dealias": True},
        {
            "rho0": 1.0, "u0": 0.0, "p0": 1.0,
            "amplitude": 1.0e-3, "k": 1, "wave": "entropy",
        },
    ))
    grid = entropy.create_grid(backend="numpy")
    equations = entropy.create_equations()
    state = entropy.create_initial_conditions(grid)
    rho0, u0, p0, _sound = equations.primitive(state)
    assert float(np.max(np.abs(u0))) == pytest.approx(0.0, abs=1e-14)
    assert float(np.max(np.abs(p0 - 1.0))) == pytest.approx(0.0, abs=1e-14)
    solver = entropy.get_solver_class()(grid, equations, cfl=entropy.cfl)
    time = 0.0
    while time < 0.1 - 1e-13:
        step = min(solver.compute_dt(state), 0.1 - time)
        state = solver.step(state, step)
        time += step
    rho, velocity, pressure, _sound = equations.primitive(state)
    np.testing.assert_allclose(rho, rho0, atol=1e-10)
    np.testing.assert_allclose(velocity, 0.0, atol=1e-10)
    np.testing.assert_allclose(pressure, 1.0, atol=1e-10)


def test_gaussian_wave2d_direction_shape_and_width(tmp_path):
    problem = GaussianWave2DProblem(config=_config(
        tmp_path,
        "gaussian_wave2d",
        {
            "Nx": 48, "Ny": 32, "Lx": 1.0, "Ly": 1.5,
            "basis_x": "fourier", "basis_y": "fourier", "dealias": False,
        },
        {
            "rho0": 1.0, "u0": 0.0, "v0": 0.0, "p0": 1.0,
            "amplitude": 0.01, "sigma": 0.08, "x0": 0.5, "y0": 0.75,
            "angle": 45.0,
        },
    ))
    grid = problem.create_grid(backend="numpy")
    state = problem.create_initial_conditions(grid)
    assert state.shape == (4, grid.Ny, grid.Nx)
    rho, velocity_x, velocity_y, _pressure = EulerEquations2D(gamma=1.4).primitive(state)
    reference = problem.get_analytical_solution(grid, 0.0)
    np.testing.assert_allclose(rho, reference["rho"], atol=1e-12)
    np.testing.assert_allclose(velocity_x, reference["u"], atol=1e-12)

    peak = np.unravel_index(int(np.argmax(rho)), rho.shape)
    density_prime = float(rho[peak] - 1.0)
    sound = np.sqrt(1.4)
    direction = np.sqrt(0.5)
    assert float(velocity_x[peak]) == pytest.approx(-sound * direction * density_prime)
    assert float(velocity_y[peak]) == pytest.approx(sound * direction * density_prime)

    ix = int(np.argmin(np.abs(np.asarray(grid.x) - (0.5 + 0.08))))
    iy = int(np.argmin(np.abs(np.asarray(grid.y) - 0.75)))
    dx = float(grid.x[ix] - 0.5)
    expected = 0.01 * np.exp(-(dx**2) / (2.0 * 0.08**2))
    assert float(rho[iy, ix] - 1.0) == pytest.approx(expected, rel=1e-6, abs=1e-8)

    later = problem.get_analytical_solution(grid, 0.05)
    iy0, ix0 = np.unravel_index(int(np.argmax(reference["rho"])), rho.shape)
    iy1, ix1 = np.unravel_index(int(np.argmax(later["rho"])), rho.shape)
    assert float(grid.x[ix1]) < float(grid.x[ix0])
    assert float(grid.y[iy1]) > float(grid.y[iy0])


def test_fourier_gaussian_rejects_periodic_bc_name(tmp_path):
    config = _config(
        tmp_path,
        "gaussian_wave2d",
        {
            "Nx": 8, "Ny": 8, "Lx": 1.0, "Ly": 1.0,
            "basis_x": "fourier", "basis_y": "fourier", "bc": "periodic",
        },
        {"amplitude": 1.0e-3, "angle": 45.0},
    )
    problem = GaussianWave2DProblem(config=config)
    with pytest.warns(UserWarning, match="periodic"):
        grid = problem.create_grid(backend="numpy")
    assert grid.basis_x == "fourier" and grid.basis_y == "fourier"


def test_gaussian_wave1d_reference_is_periodic_only(tmp_path):
    periodic = GaussianWave1DProblem(config=_config(
        tmp_path / "periodic",
        "gaussian_wave1d",
        {"N": 32, "Lx": 1.0, "basis": "fourier"},
        {"amplitude": 1.0e-3, "sigma": 0.05, "x0": 0.5},
    ))
    grid = periodic.create_grid(backend="numpy")
    reference = periodic.get_analytical_solution(grid, 0.1)
    assert isinstance(reference, dict) and "rho" in reference

    walls = GaussianWave1DProblem(config=_config(
        tmp_path / "walls",
        "gaussian_wave1d",
        {"N": 32, "Lx": 1.0, "basis": "legendre", "bc": "dirichlet"},
        {"amplitude": 1.0e-3, "sigma": 0.05, "x0": 0.5},
    ))
    grid = walls.create_grid(backend="numpy")
    assert walls.get_analytical_solution(grid, 0.1) is None
    assert walls.get_wave_parameters()["x0"] == pytest.approx(0.5)


def test_gaussian_wave2d_y_uses_base_constructor_and_gamma(tmp_path):
    config = _config(
        tmp_path,
        "gaussian_wave2d_y",
        {
            "Nx": 8, "Ny": 16, "Lx": 1.0, "Ly": 1.0,
            "basis_x": "fourier", "basis_y": "fourier",
        },
        {"amplitude": 1.0e-3, "sigma": 0.05, "y0": 0.5, "direction": "up"},
    )
    config.pop("physics")
    config["equations"] = {"gamma": 1.6}
    problem = GaussianWave2DYProblem(None, config)
    assert problem.gamma == pytest.approx(1.6)
    grid = problem.create_grid(backend="numpy")
    state = problem.create_initial_conditions(grid)
    reference = problem.get_analytical_solution(grid, 0.0)
    np.testing.assert_array_equal(np.asarray(state), np.asarray(reference))


def test_shu_osher_rankine_hugoniot_and_diagnostics(tmp_path):
    config = _config(
        tmp_path / "chebyshev",
        "shu_osher1d",
        {"N": 32, "Lx": 1.0, "basis": "chebyshev", "bc": "fixed", "dealias": False},
        {"x_shock": 0.5},
    )
    config.pop("physics")
    config["equations"] = {"gamma": 1.6}
    problem = ShuOsher1DProblem(config=config)
    assert problem.gamma == pytest.approx(1.6)
    assert problem.u_post == pytest.approx(problem.u_L)
    assert problem.p_post == pytest.approx(problem.p_L)
    shock_speed = problem.shock_speed
    left_flux = problem.rho_L * (shock_speed - problem.u_L)
    right_flux = problem.rho_R * (shock_speed - problem.u_R)
    assert left_flux == pytest.approx(right_flux)
    assert not hasattr(problem, "get_analytical_solution")

    grid = problem.create_grid(backend="numpy")
    equations = problem.create_equations()
    state = problem.create_initial_conditions(grid)
    rho, velocity, pressure, _sound = equations.primitive(state)
    assert float(rho[0]) == pytest.approx(problem.rho_L)
    assert float(velocity[0]) == pytest.approx(problem.u_L)
    assert float(pressure[0]) == pytest.approx(problem.p_L)
    assert float(rho[-1]) != pytest.approx(float(rho[0]))
    assert float(velocity[-1]) == pytest.approx(0.0)

    captured = {}

    def _capture(_solver, conserved):
        captured["state"] = conserved
        return {"mass_error": 0.0}

    problem.compute_conservation_errors = _capture
    holder = type("Holder", (), {"U": state})()
    diagnostics = problem.compute_diagnostics(holder, 0.1)
    assert captured["state"] is state
    assert "l2_error" not in diagnostics
    assert diagnostics["shock_position"] == pytest.approx(
        problem.x_shock + problem.shock_speed * 0.1
    )

    periodic = ShuOsher1DProblem(config=_config(
        tmp_path / "fourier",
        "shu_osher1d",
        {"N": 16, "Lx": 1.0, "basis": "fourier"},
        {},
    ))
    with pytest.warns(UserWarning, match="seam"):
        periodic.create_grid(backend="numpy")


def test_shipped_hydro_configs_match_solver_convention():
    def load(name):
        return yaml.safe_load((ROOT / "configs" / name).read_text())

    sod = load("sod.yaml")
    assert sod["problem"] == "sod"
    assert sod["grid"]["basis"] == "chebyshev"
    assert sod["grid"]["bc"] == "fixed"
    assert sod["io"]["outdir"] == "outputs/sod"

    assert load("rti_athena.yaml")["gravity"]["gy"] == pytest.approx(-1.0)
    assert load("rti.yaml")["gravity"]["gy"] == pytest.approx(-1.0)
    assert load("rti_1d_athena.yaml")["gravity"]["gx"] == pytest.approx(-1.0)
    hse = load("hse_1d_test.yaml")
    assert hse["gravity"]["gx"] == pytest.approx(-1.0)
    assert hse["initial_conditions"]["left"]["p"] == pytest.approx(2.5)

    shu = load("shu_osher1d.yaml")
    assert shu["grid"]["basis"] == "chebyshev"
    assert shu["grid"]["bc"] == "fixed"
    assert shu["artificial_viscosity"]["mode"] == "constant"
    assert "safety_factor" not in shu.get("adaptive", {})
    assert shu["integration"].get("safety_factor") is None

    acoustic = load("acoustic1d.yaml")
    assert acoustic["initial_conditions"]["wave"] == "acoustic"
    assert load("khi2d_4k.yaml")["io"]["outdir"] == "outputs/khi2d_4k"
    assert load("khi2d.yaml")["io"]["outdir"] == "outputs/khi2d"
