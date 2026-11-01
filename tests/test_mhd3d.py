"""Small-grid 3D MHD regressions with fixed RK4 timesteps."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from phrike.equations import MHDEquations2D, MHDEquations3D
from phrike.grid import Grid2D, Grid3D
from phrike.problems import OrszagTang3DProblem, ProblemRegistry
from phrike.problems.problem_list.alfven3d import circularly_polarized_alfven_3d
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d
from phrike.problems.problem_list.orszag_tang3d import orszag_tang_3d
from phrike.solver import SpectralSolverMHD2D, SpectralSolverMHD3D

ROOT = Path(__file__).resolve().parents[1]
# No damping: compare the ideal equations independently of dimensional filters.
FILTERS = {"spectral_dissipation": {"enabled": False}}


def _grid(n=16, **kwargs):
    return Grid3D(n, n, n, 1.0, 1.0, 1.0, backend="numpy",
                  filter_params=FILTERS, **kwargs)


def _relative_errors(equations, actual, exact):
    actual, exact = equations.primitive(actual), equations.primitive(exact)
    errors = []
    for indices in ((5, 6, 7), (1, 2, 3)):
        reference = np.array([exact[i] for i in indices])
        delta = np.array([actual[i] for i in indices]) - reference
        errors.append(np.sqrt(np.sum(delta ** 2) / np.sum(reference ** 2)))
    return np.array(errors)


def test_flux_rotational_symmetry():
    X, Y, Z = _grid(8).xyz_mesh()
    rng = np.random.default_rng(23)
    fields = [a + 0.05 * np.sin(2 * np.pi * (X + Y))
              + 0.03 * np.cos(2 * np.pi * Z) for a in rng.uniform(0.3, 1.0, 8)]
    equations = MHDEquations3D()
    U = equations.conservative(*fields)
    _, fy, fz = equations.flux(U)
    errors = []
    for expected, swap in ((fy, [0, 2, 1, 3, 4, 6, 5, 7]),
                           (fz, [0, 3, 2, 1, 4, 7, 6, 5])):
        rotated = equations.flux(U[swap])[0][swap]
        errors.append(np.max(np.abs(expected - rotated)))
        np.testing.assert_allclose(expected, rotated, rtol=5e-15, atol=5e-15)
    print(f"Flux y/z rotation max differences: {errors}")


def test_ot_epsilon_zero_matches_2d():
    g2 = Grid2D(32, 32, 1.0, 1.0, backend="numpy", filter_params=FILTERS)
    g3 = Grid3D(32, 32, 4, 1.0, 1.0, 1.0, backend="numpy", filter_params=FILTERS)
    u2 = orszag_tang_2d(*g2.xy_mesh())
    u3 = orszag_tang_3d(*g3.xyz_mesh(), epsilon=0.0)
    np.testing.assert_array_equal(u3, np.repeat(u2[:, None], 4, axis=1))
    s2 = SpectralSolverMHD2D(g2, MHDEquations2D(), scheme="rk4")
    s3 = SpectralSolverMHD3D(g3, MHDEquations3D(), scheme="rk4")
    for _ in range(10):
        u2, u3 = s2.step(u2, 0.001), s3.step(u3, 0.001)
    error = np.max(np.abs(u3 - u2[:, None]))
    print(f"OT 2D vs 3D after 10 steps: max={error:.6e}")
    np.testing.assert_allclose(u3, np.repeat(u2[:, None], 4, axis=1), rtol=0, atol=1e-12)


@pytest.mark.parametrize("direction", [1, -1])
def test_diagonal_alfven_exact_solution_and_rk4_convergence(direction):
    grid = _grid()
    equations = MHDEquations3D()
    X, Y, Z = grid.xyz_mesh()
    params = dict(Y=Y, Z=Z, orientation="diagonal", direction=direction)
    period = 1.0 / np.sqrt(3.0)
    errors = []
    for steps in (32, 64):
        solver = SpectralSolverMHD3D(grid, equations, scheme="rk4")
        U = circularly_polarized_alfven_3d(X, **params)
        dt = period / steps
        for step in range(steps):
            U = solver.step(U, dt)
            if step + 1 == steps // 4:
                # Also detect a reversed wave, which a full-period check misses.
                exact = circularly_polarized_alfven_3d(X, t=period / 4, **params)
                np.testing.assert_allclose(U, exact, rtol=0, atol=3e-6)
        exact = circularly_polarized_alfven_3d(X, t=period, **params)
        error = _relative_errors(equations, U, exact)
        print(f"Alfven direction={direction}, dt={dt:.12e}: relative L2 B/u={error}")
        errors.append(error)
    # The measured coarse-step velocity error is ~8e-5; the fine run is <1e-5.
    assert np.max(errors[0]) < 1e-4
    assert np.max(errors[1]) < 1e-5
    ratios = errors[0] / errors[1]
    print(f"Alfven direction={direction} B/u error ratios: {ratios}")
    assert np.all((ratios >= 12) & (ratios <= 20))


def test_ot_divergence_and_conservation():
    grid = _grid()
    solver = SpectralSolverMHD3D(grid, MHDEquations3D(), scheme="rk4")
    U = orszag_tang_3d(*grid.xyz_mesh())
    initial = np.sum(U[:5], axis=(1, 2, 3))
    scale = np.array([initial[0]] * 4 + [initial[4]])
    max_div = solver.max_div_b(U)
    max_drift = np.zeros(5)
    for _ in range(8):
        U = solver.step(U, 0.001)
        max_div = max(max_div, solver.max_div_b(U))
        drift = np.abs(np.sum(U[:5], axis=(1, 2, 3)) - initial) / scale
        max_drift = np.maximum(max_drift, drift)
    print(f"OT max div B={max_div:.6e}; mass/momenta/energy drifts={max_drift}")
    assert max_div < 1e-12
    assert np.max(max_drift) < 1e-12
    assert solver.positivity.events == 0


@pytest.mark.parametrize("precision", ["double", "single"])
def test_ot_divergence_with_spectral_dissipation_is_round_off(precision):
    """Dissipation scales each mode by one scalar, so it commutes with the projection.

    max|div B| is absolute: it sits at eps * k_max * max|B| of the working precision,
    about 1e-14 in double and 1e-6 in single (Metal/MPS) on this 32^3 grid.
    """
    if precision == "single":
        pytest.importorskip("torch")
        kwargs = {"backend": "torch", "torch_device": "cpu"}
    else:
        kwargs = {"backend": "numpy"}
    dissipation = {"spectral_dissipation": {"enabled": True, "order": 8, "rate": 300.0,
                                            "onset_fraction": 0.5}}
    grid = Grid3D(32, 32, 32, 1.0, 1.0, 1.0, precision=precision,
                  filter_params=dissipation, **kwargs)
    solver = SpectralSolverMHD3D(grid, MHDEquations3D(), scheme="rk4")
    U = solver._project_B(orszag_tang_3d(*grid.xyz_mesh()))
    max_div = 0.0
    for _ in range(4):
        U = solver.step(U, 0.002)
        max_div = max(max_div, solver.max_div_b(U))
    B = np.asarray(U[5:8].cpu() if hasattr(U, "cpu") else U[5:8])
    eps = np.finfo(B.dtype).eps
    k_max = 2.0 * np.pi * (32 // 3) * np.sqrt(3.0)
    scale = eps * k_max * np.max(np.sqrt(np.sum(B * B, axis=0)))
    print(f"{precision}: max div B={max_div:.3e}, eps*k_max*max|B|={scale:.3e}")
    assert max_div < 10.0 * scale


def test_torch_cpu_matches_numpy():
    torch = pytest.importorskip("torch")
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        gn = _grid()
        gt = Grid3D(16, 16, 16, 1.0, 1.0, 1.0, backend="torch",
                    torch_device="cpu", precision="double", filter_params=FILTERS)
        un, ut = orszag_tang_3d(*gn.xyz_mesh()), orszag_tang_3d(*gt.xyz_mesh())
        sn = SpectralSolverMHD3D(gn, MHDEquations3D(), scheme="rk4")
        st = SpectralSolverMHD3D(gt, MHDEquations3D(), scheme="rk4")
        for _ in range(4):
            un, ut = sn.step(un, 0.001), st.step(ut, 0.001)
        error = np.max(np.abs(un - ut.numpy()))
        print(f"Torch vs NumPy OT max difference={error:.6e}")
        np.testing.assert_allclose(ut.numpy(), un, rtol=0, atol=1e-10)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("direction", [1, -1])
def test_default_alfven_ic_unchanged(direction):
    X, _, _ = _grid(8).xyz_mesh()
    rho0, p0, B0, b1, m, Lx, gamma = 1.3, 0.2, 0.7, 0.12, 2, 1.4, 1.4
    # origin/main at 5036f8b: x-only t=0 formula, including operation order.
    phi = (2.0 * np.pi * m / Lx) * X - 0.0
    rho = np.full_like(X, rho0); p = np.full_like(X, p0)
    Bx = np.full_like(X, B0); By = b1 * np.cos(phi); Bz = b1 * np.sin(phi)
    inv = 1.0 / np.sqrt(rho0)
    expected = MHDEquations3D(gamma=gamma).conservative(
        rho, np.zeros_like(X), -float(direction) * By * inv,
        -float(direction) * Bz * inv, p, Bx, By, Bz,
    )
    params = dict(rho0=rho0, p0=p0, B0=B0, b1=b1, m=m, Lx=Lx,
                  gamma=gamma, direction=direction)
    np.testing.assert_array_equal(circularly_polarized_alfven_3d(X, **params), expected)
    np.testing.assert_array_equal(
        circularly_polarized_alfven_3d(X, orientation="x", **params), expected,
    )


def test_diagonal_alfven_requires_cube_and_known_orientation():
    X, Y, Z = _grid(8).xyz_mesh()
    with pytest.raises(ValueError, match="cubic box"):
        circularly_polarized_alfven_3d(X, Y=Y, Z=Z, orientation="diagonal", Ly=2.0)
    with pytest.raises(ValueError, match="orientation"):
        circularly_polarized_alfven_3d(X, orientation="unknown")


@pytest.mark.parametrize("name", ["orszag_tang3d", "alfven3d_diagonal"])
def test_problem_configs_construct(name, tmp_path):
    config = yaml.safe_load((ROOT / "configs" / f"{name}.yaml").read_text())
    config["grid"].update(Nx=8, Ny=8, Nz=8)
    config["io"]["outdir"] = str(tmp_path)
    problem = ProblemRegistry.create_problem(config["problem"], config=config)
    if name == "orszag_tang3d":
        assert isinstance(problem, OrszagTang3DProblem)
    grid = problem.create_grid("numpy")
    assert grid.mhd_config == config["physics"]["mhd"]
    U = problem.create_initial_conditions(grid)
    assert U.shape == (8, 8, 8, 8)
    assert isinstance(problem.get_solver_class()(grid, problem.create_equations()),
                      SpectralSolverMHD3D)
