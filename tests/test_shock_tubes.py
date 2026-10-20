from pathlib import Path

import numpy as np
import pytest
import yaml

from phrike.problems import ProblemRegistry


@pytest.mark.parametrize("basis", ["legendre", "chebyshev"])
def test_brio_wu_initial_state_and_viscous_induction(basis, tmp_path):
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/brio_wu1d.yaml").read_text()
    )
    config["grid"].update(N=17, basis=basis)
    config["initial_conditions"]["smoothing"]["enabled"] = False
    config["io"]["outdir"] = str(tmp_path)
    problem = ProblemRegistry.create_problem("brio_wu1d", config=config)
    grid = problem.create_grid("numpy")
    equations = problem.create_equations()
    U = problem.create_initial_conditions(grid)
    rho, ux, uy, uz, pressure, bx, by, bz = equations.primitive(U)
    left = grid.x < 0.5
    assert equations.gamma == 2.0
    np.testing.assert_array_equal(rho, np.where(left, 1.0, 0.125))
    np.testing.assert_allclose(pressure, np.where(left, 1.0, 0.1), atol=2e-16)
    np.testing.assert_array_equal(bx, 0.75)
    np.testing.assert_array_equal(by, np.where(left, 1.0, -1.0))
    np.testing.assert_array_equal([ux, uy, uz, bz], np.zeros((4, grid.N)))
    solver = problem.get_solver_class()(
        grid, equations, artificial_viscosity_config=problem.artificial_viscosity_config
    )
    inviscid = problem.get_solver_class()(grid, equations)
    difference = solver._compute_rhs(U) - inviscid._compute_rhs(U)
    assert np.max(np.abs(difference[6])) > 0.0
    np.testing.assert_array_equal(difference[5], 0.0)
    advanced = solver.step(U, min(solver.compute_dt(U), 1e-6))
    np.testing.assert_allclose(advanced[5], 0.75, atol=1e-15)
    np.testing.assert_allclose(advanced[:, [0, -1]], U[:, [0, -1]], atol=1e-15)


@pytest.mark.parametrize("basis", ["legendre", "chebyshev"])
@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_polynomial_flux_overintegration_removes_quadratic_alias(basis, backend):
    from types import SimpleNamespace
    from phrike.grid import Grid1D

    kwargs = {"torch_device": "cpu"} if backend == "torch" else {}
    if backend == "torch":
        torch = pytest.importorskip("torch")
    grid = Grid1D(N=17, Lx=1.0, basis=basis, bc="fixed", backend=backend, **kwargs)
    mode = np.zeros(17)
    mode[11] = 1.0
    values = grid._basis.inverse(mode)
    state = np.array([np.ones(17), values, np.full(17, 100.0)])
    if backend == "torch":
        state = torch.as_tensor(state, dtype=grid.x.dtype)

    def flux(U):
        result = U * 0.0
        result[1] = U[1] ** 2
        return result

    equations = SimpleNamespace(gamma=1.4, flux=flux)
    actual = grid.flux_derivative(state, equations)
    if backend == "torch":
        actual = actual.numpy()
    if basis == "chebyshev":
        expected = np.zeros(17)
    else:
        product = np.polynomial.legendre.legmul(mode, mode)[:17]
        expected = -2.0 * np.polynomial.legendre.legval(
            grid._basis._y, np.polynomial.legendre.legder(product)
        )
    np.testing.assert_allclose(actual[1], expected, atol=2e-10)
    grid.dealias = False
    aliased = grid.flux_derivative(state, equations)
    if backend == "torch":
        aliased = aliased.numpy()
    assert np.max(np.abs(aliased[1] - expected)) > 1.0


@pytest.mark.parametrize("basis", ["legendre", "chebyshev"])
@pytest.mark.parametrize("mhd", [False, True])
def test_padded_shock_flux_uses_admissible_conservative_states(basis, mhd):
    from types import SimpleNamespace
    from phrike.equations import EulerEquations1D, MHDEquations1D
    from phrike.grid import Grid1D

    grid = Grid1D(N=32, Lx=1.0, basis=basis, bc="fixed", backend="numpy")
    left = grid.x < 0.5
    rho, pressure = np.where(left, 1.0, 0.125), np.where(left, 1.0, 0.1)
    zero = np.zeros(32)
    equations = MHDEquations1D(gamma=2.0) if mhd else EulerEquations1D()
    if mhd:
        state = equations.conservative(rho, zero, zero, zero, pressure,
                                       zero + 0.75, np.where(left, 1.0, -1.0), zero)
    else:
        state = equations.conservative(rho, zero, pressure)
    original = state.copy()
    evaluated = []

    def checked_flux(U):
        primitive = equations.primitive(U)
        assert np.all(primitive[0] > 0.0)
        assert np.all(primitive[4 if mhd else 2] > 0.0)
        evaluated.append(U.shape[-1])
        return equations.flux(U)

    derivative = grid.flux_derivative(
        state, SimpleNamespace(gamma=equations.gamma, flux=checked_flux)
    )
    assert np.all(np.isfinite(derivative))
    assert evaluated == [48]
    np.testing.assert_array_equal(state, original)
    if mhd:
        np.testing.assert_array_equal(derivative[5], 0.0)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_legendre_padded_flux_single_precision(backend):
    from phrike.equations import EulerEquations1D
    from phrike.grid import Grid1D

    if backend == "torch":
        torch = pytest.importorskip("torch")
    grid = Grid1D(N=17, Lx=1.0, basis="legendre", bc="fixed", precision="single",
                  backend=backend, torch_device="cpu" if backend == "torch" else None)
    x = grid.x
    equations = EulerEquations1D()
    state = equations.conservative(1.0 + 0.1*x, x*0.1, 1.0 + 0.2*x)
    derivative = grid.flux_derivative(state, equations)
    assert derivative.dtype == state.dtype
    if backend == "torch":
        assert derivative.device == state.device
        assert torch.isfinite(derivative).all()
    else:
        assert np.isfinite(derivative).all()
