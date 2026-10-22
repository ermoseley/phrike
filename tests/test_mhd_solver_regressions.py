"""Regression tests for MHD positivity, stalled steps and weighted diagnostics.

All runs use the NumPy backend and are small enough for a few seconds.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import yaml

from phrike.equations import (
    EulerEquations1D,
    MHDEquations1D,
    MHDEquations2D,
)
from phrike.grid import Grid1D, Grid2D
from phrike.problems import ProblemRegistry
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d
from phrike.solver import (
    SpectralSolver1D,
    SpectralSolverMHD2D,
    StalledTimeStep,
    _cap_proposal,
    conserved_diagnostics,
)

GAMMA = 5.0 / 3.0


def _orszag_tang(n, spectral_dissipation=None):
    filters = {"spectral_dissipation": spectral_dissipation or {"enabled": False}}
    grid = Grid2D(
        Nx=n, Ny=n, Lx=1.0, Ly=1.0, dealias=True, filter_params=filters,
        backend="numpy", precision="double", basis_x="fourier", basis_y="fourier",
    )
    grid.mhd_config = {}
    equations = MHDEquations2D(gamma=GAMMA)
    X, Y = grid.xy_mesh()
    U0 = orszag_tang_2d(X, Y, gamma=GAMMA)
    return grid, equations, SpectralSolverMHD2D(grid, equations, scheme="rk4", cfl=0.25), U0


def test_accepted_state_is_positive_after_the_hard_projection():
    # A sharp, nearly evacuated density step rings below zero under the 2/3
    # Fourier projection. The accepted state must be floored afterwards, with
    # the energy rebuilt from the projected B and velocity.
    n = 32
    grid, equations, solver, _ = _orszag_tang(n)
    X, _ = grid.xy_mesh()
    rho = np.where(X < 0.5, 0.01, 1.0)
    zeros = np.zeros_like(rho)
    U = equations.conservative(
        rho, 0.1 + zeros, zeros, zeros, 1.0 + zeros, 0.5 + zeros, zeros, zeros
    )
    ringing = equations.primitive(grid.apply_accepted_state_stabilization(U, 0.0))
    assert ringing[0].min() < solver.rho_min, "the case must ring below the floor"

    with pytest.warns(RuntimeWarning, match="Positivity clamp") as record:
        U_new = solver._fixed_step(U, 1e-8)
    assert len([w for w in record if "Positivity clamp" in str(w.message)]) == 1
    assert solver.positivity.events > 0
    rho_new, ux, uy, uz, p, Bx, By, Bz = equations.primitive(U_new)
    assert np.all(np.isfinite(U_new))
    assert rho_new.min() >= solver.rho_min
    assert p.min() >= solver.p_min
    assert solver.max_div_b(U_new) < 1e-12
    energy = (
        p / (GAMMA - 1.0)
        + 0.5 * rho_new * (ux * ux + uy * uy + uz * uz)
        + 0.5 * (Bx * Bx + By * By + Bz * Bz)
    )
    np.testing.assert_allclose(U_new[4], energy, rtol=1e-13)


def test_positive_state_is_returned_untouched_by_the_accepted_limiter():
    _, equations, solver, U0 = _orszag_tang(16)
    assert solver._limit_accepted(U0) is U0
    assert solver.positivity.events == 0


def test_orszag_tang_with_spectral_dissipation_stays_positive():
    dissipation = {"enabled": True, "order": 8, "rate": 100.0, "onset_fraction": 0.5}
    grid, equations, solver, U0 = _orszag_tang(64, dissipation)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        history = solver.run(U0, 0.0, 0.5, output_interval=0.25)
    rho, _, _, _, p, *_ = equations.primitive(solver.U)
    assert solver.t == pytest.approx(0.5)
    assert np.all(np.isfinite(solver.U))
    # Rebuilding p from the total energy leaves round-off around the floor.
    assert rho.min() >= solver.rho_min
    assert p.min() > 0.0
    assert max(history["div_b_max"]) < 1e-11
    energy = np.asarray(history["energy"])
    assert abs(energy[-1] / energy[0] - 1.0) < 0.05


def test_ideal_orszag_tang_without_dissipation_raises_instead_of_crawling():
    _, _, solver, U0 = _orszag_tang(64)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(StalledTimeStep, match="Time step collapsed") as caught:
            solver.run(U0, 0.0, 0.5, output_interval=0.25)
    assert isinstance(caught.value, RuntimeError)
    assert solver.t < 0.5
    assert solver._accepted_steps < 5000


class _SteppedLimit:
    """Solver stand-in whose stability limit follows a script."""

    def __init__(self, limits):
        self.limits = list(limits)
        self.U = None
        self.t = 0.0
        self._accepted_steps = 0
        self._dt_reference = None

    def _stability_limit(self, U):
        return self.limits.pop(0)


def test_stall_check_ignores_steady_and_moderately_shrinking_steps():
    solver = _SteppedLimit([1e-3, 5e-4, 2e-5, 1e-4, 1e-3])
    for _ in range(5):
        dt = _cap_proposal(solver, float("inf"), 1.0, None, None)
        solver.t += dt
    assert solver._dt_reference == pytest.approx(1e-3)


def test_stall_check_uses_the_uncapped_step_not_the_output_horizon():
    solver = _SteppedLimit([1e-3, 1e-3])
    _cap_proposal(solver, float("inf"), 1.0, None, None)
    solver.t = 0.1 - 1e-13  # a legitimately tiny remainder before an output
    assert _cap_proposal(solver, float("inf"), 1.0, 0.1, None) < 1e-12


@pytest.mark.parametrize(
    "limits",
    [
        [1e-3, 1e-12],  # collapse below 1e-8 of the initial bound
        [1e-3, 1e-7],  # 1e-4 of the initial bound with 1e7 steps to go
    ],
)
def test_stall_check_raises_on_a_collapsed_step_bound(limits):
    solver = _SteppedLimit(limits)
    _cap_proposal(solver, float("inf"), 1.0, None, None)
    with pytest.raises(StalledTimeStep, match="collapsed"):
        _cap_proposal(solver, float("inf"), 1.0, None, None)


def _sod_problem(tmp_path, basis="chebyshev", n=48):
    config = yaml.safe_load(
        (Path(__file__).resolve().parents[1] / "configs/sod.yaml").read_text()
    )
    config["grid"].update(N=n, basis=basis)
    config["integration"]["t_end"] = 0.05
    config["monitoring"] = {"enabled": True, "include_conservation": True}
    config["io"]["outdir"] = str(tmp_path)
    return ProblemRegistry.create_problem("sod", config=config)


@pytest.mark.parametrize("basis", ["chebyshev", "legendre"])
def test_non_fourier_diagnostics_use_quadrature_weights(basis, tmp_path):
    problem = _sod_problem(tmp_path, basis)
    grid = problem.create_grid("numpy")
    equations = problem.create_equations()
    U0 = problem.create_initial_conditions(grid)
    solver = problem.get_solver_class()(
        grid, equations, scheme="rk4", cfl=problem.cfl,
        artificial_viscosity_config=problem.artificial_viscosity_config,
    )
    problem.initialize_monitoring(solver, U0)
    history = solver.run(U0, 0.0, 0.05, output_interval=0.05)

    rho, u, p, _ = equations.primitive(solver.U)
    w = grid.wx
    np.testing.assert_allclose(history["mass"][-1], np.sum(w * rho), rtol=1e-13)
    np.testing.assert_allclose(
        history["energy"][-1],
        np.sum(w * (p / (equations.gamma - 1.0) + 0.5 * rho * u * u)),
        rtol=1e-13,
    )
    weighted_drift = history["mass"][-1] / history["mass"][0] - 1.0
    node_sum_drift = rho.sum() / equations.primitive(U0)[0].sum() - 1.0
    assert abs(weighted_drift) < 1e-4
    assert node_sum_drift != pytest.approx(weighted_drift, rel=0.1)

    errors = problem.compute_conservation_errors(solver, solver.U)
    assert errors["mass_error"] == pytest.approx(weighted_drift, rel=1e-6, abs=1e-14)
    energy_drift = history["energy"][-1] / history["energy"][0] - 1.0
    assert errors["energy_error"] == pytest.approx(energy_drift, rel=1e-6, abs=1e-14)
    assert abs(errors["energy_error"]) < 1e-3


def test_non_fourier_mhd_diagnostics_are_integrals():
    grid = Grid1D(N=17, Lx=2.0, basis="chebyshev", bc="fixed", backend="numpy")
    equations = MHDEquations1D(gamma=2.0)
    ones = np.ones(grid.N)
    U = equations.conservative(
        0.5 * ones, 0.2 * ones, 0.0 * ones, 0.0 * ones, 1.0 * ones,
        0.75 * ones, 0.3 * ones, 0.0 * ones,
    )
    cons = conserved_diagnostics(grid, equations, U)
    assert set(cons) == set(equations.conserved_quantities(U))
    assert cons["mass"] == pytest.approx(0.5 * 2.0, rel=1e-13)
    assert cons["momentum_x"] == pytest.approx(0.5 * 0.2 * 2.0, rel=1e-13)
    assert cons["magnetic_energy"] == pytest.approx(0.5 * (0.75**2 + 0.3**2) * 2.0, rel=1e-13)
    assert cons["cross_helicity"] == pytest.approx((0.2 * 0.75) * 2.0, rel=1e-13)


def test_fourier_diagnostics_are_unchanged_grid_sums():
    grid = Grid1D(N=32, Lx=1.0, basis="fourier", backend="numpy")
    equations = EulerEquations1D(gamma=1.4)
    x = np.asarray(grid.x)
    rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * x)
    U = equations.conservative(rho, 0.3 + 0.0 * x, 1.0 + 0.1 * np.cos(2.0 * np.pi * x))
    assert conserved_diagnostics(grid, equations, U) == equations.conserved_quantities(U)

    solver = SpectralSolver1D(grid, equations, scheme="rk4", cfl=0.4)
    history = solver.run(U, 0.0, 0.02, output_interval=0.02)
    final = equations.conserved_quantities(solver.U)
    assert history["mass"][-1] == final["mass"]
    assert history["energy"][-1] == final["energy"]
    assert history["mass"][-1] == float(np.sum(equations.primitive(solver.U)[0]))
