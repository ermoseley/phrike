import numpy as np
import pytest

from phrike.adaptive import AdaptiveStepResult
from phrike.equations import EulerEquations1D, MHDEquations1D
from phrike.grid import Grid1D
from phrike.solver import SpectralSolver1D, SpectralSolverMHD, _compute_rhs



@pytest.fixture(params=["legendre", "chebyshev"])
def basis(request):
    return request.param


@pytest.fixture(params=[False, True], ids=["hydro", "mhd"])
def mhd(request):
    return request.param


def _case(basis, mhd, bc="fixed", **kwargs):
    grid = Grid1D(N=9, Lx=1.0, basis=basis, bc=bc, backend="numpy", **kwargs)
    x = np.asarray(grid.x)
    eqs = MHDEquations1D(gamma=1.4) if mhd else EulerEquations1D(gamma=1.4)
    rho, ux, pressure = 1.0 + 0.2*x, 0.1 + 0.03*x, 1.0 + 0.1*x
    if mhd:
        U = eqs.conservative(rho, ux, 0.2 + 0.01*x, -0.1 + 0.02*x,
                             pressure, 0.7 + 0*x, 0.3 + 0.1*x, -0.2 + 0.04*x)
    else:
        U = eqs.conservative(rho, ux, pressure)
    solver = SpectralSolverMHD(grid, eqs, scheme="rk2") if mhd else SpectralSolver1D(grid, eqs, scheme="rk2")
    return grid, eqs, U, solver


def _pressure(U, mhd):
    energy = 4 if mhd else 2
    kinetic = np.sum(U[1:energy]**2, axis=0) / (2*U[0])
    magnetic = 0.5*np.sum(U[5:8]**2, axis=0) if mhd else 0.0
    return 0.4*(U[energy] - kinetic - magnetic)


@pytest.mark.parametrize("bc", ["fixed", "dirichlet"])
def test_fixed_captures_first_endpoints_and_paired_values(basis, mhd, bc):
    grid, eqs, U, _ = _case(basis, mhd, bc)
    original = U.copy()
    grid.apply_boundary_conditions(U, eqs)
    changed = U*1.03
    result = grid.apply_boundary_conditions(changed, eqs)
    np.testing.assert_array_equal(U, original)
    np.testing.assert_array_equal(changed, U*1.03)
    np.testing.assert_allclose(result[:, [0, -1]], U[:, [0, -1]], atol=2e-15)
    grid.set_dirichlet_bc_values({"rho": (1.7, 1.8), "momentum": (0.3, -0.4), "p": (2.1, 2.2)})
    result = grid.apply_boundary_conditions(changed, eqs)
    np.testing.assert_allclose(result[0, [0, -1]], [1.7, 1.8])
    np.testing.assert_allclose(result[1, [0, -1]], [0.3, -0.4])
    np.testing.assert_allclose(_pressure(result, mhd)[[0, -1]], [2.1, 2.2])
    if mhd:
        np.testing.assert_allclose(result[[2, 3, 5, 6, 7]][:, [0, -1]], U[[2, 3, 5, 6, 7]][:, [0, -1]])


@pytest.mark.parametrize("bc", ["outflow", "neumann", "wall"])
def test_weak_boundaries_leave_the_state_to_the_boundary_flux(basis, mhd, bc):
    grid, eqs, U, _ = _case(basis, mhd, bc)
    original = U.copy()
    result = grid.apply_boundary_conditions(U, eqs)
    np.testing.assert_array_equal(result, original)
    np.testing.assert_array_equal(U, original)


def test_mixed_left_and_right_conditions(basis, mhd):
    bc = {"left": {"default": "outflow", "density": "fixed", "momentum": "wall"}, "right": "fixed"}
    grid, eqs, U, _ = _case(basis, mhd, bc)
    grid.apply_boundary_conditions(U, eqs)
    changed = 1.03*U
    result = grid.apply_boundary_conditions(changed, eqs)
    # Fixed density is imposed strongly; the weak momentum and pressure keep
    # the evolved endpoint values.
    assert result[0, 0] == U[0, 0]
    assert result[1, 0] == changed[1, 0]
    np.testing.assert_allclose(_pressure(result, mhd)[0], _pressure(changed, mhd)[0])
    np.testing.assert_allclose(result[:, -1], U[:, -1], atol=2e-15)
    np.testing.assert_array_equal(changed, 1.03*U)


def _mass_rate(grid, eqs, U, mhd):
    if mhd:
        solver = SpectralSolverMHD(grid, eqs, scheme="rk2")
        rhs = solver._compute_rhs(U)
    else:
        rhs = _compute_rhs(grid, eqs, U)
    return float(np.sum(np.asarray(grid.wx) * rhs[0]))


@pytest.mark.parametrize("dealias", [False, True])
def test_wall_boundary_flux_conserves_mass_exactly(basis, mhd, dealias):
    grid, eqs, U, _ = _case(basis, mhd, "wall", dealias=dealias)
    grid.initialize_boundary_conditions(U, eqs)
    assert abs(_mass_rate(grid, eqs, U, mhd)) < 1e-12


def test_outflow_boundary_flux_vanishes_on_the_reference_state(basis, mhd):
    grid, eqs, _, _ = _case(basis, mhd, "outflow")
    x = np.asarray(grid.x)
    if mhd:
        U = eqs.conservative(1 + 0*x, 0.1 + 0*x, 0.2 + 0*x, -0.1 + 0*x, 1 + 0*x,
                             0.7 + 0*x, 0.3 + 0*x, -0.2 + 0*x)
        rhs = SpectralSolverMHD(grid, eqs, scheme="rk2")._compute_rhs(U)
    else:
        U = eqs.conservative(1 + 0*x, 0.1 + 0*x, 1 + 0*x)
        rhs = _compute_rhs(grid, eqs, U)
    np.testing.assert_allclose(rhs, 0.0, atol=1e-11)


def test_acoustic_pulse_leaves_through_outflow_and_reflects_off_walls(basis):
    gamma, rho0, p0, amplitude = 1.4, 1.0, 1.0, 1e-3
    c = np.sqrt(gamma * p0 / rho0)

    def remaining(bc):
        grid = Grid1D(N=64, Lx=1.0, basis=basis, bc=bc, backend="numpy", dealias=False)
        eqs = EulerEquations1D(gamma=gamma)
        x = np.asarray(grid.x)
        d = amplitude * np.exp(-((x - 0.5) / 0.05) ** 2)
        U0 = eqs.conservative(rho0 + d, c * d / rho0, p0 + c * c * d)
        solver = SpectralSolver1D(grid, eqs, scheme="rk4", cfl=0.3)
        solver.run(U0, t0=0.0, t_end=1.0, output_interval=1.0)
        rho, u, p, _ = eqs.primitive(solver.U)
        w = np.asarray(grid.wx)
        energy = np.sum(w * ((p - p0) ** 2 / (rho0 * c * c) + rho0 * u ** 2))
        initial = np.sum(w * (2.0 * rho0 * (c * d) ** 2))
        return np.sqrt(energy / initial)

    assert remaining("outflow") < 1e-4
    assert abs(remaining("wall") - 1.0) < 1e-3


def test_diffusion_has_zero_boundary_flux(basis):
    grid = Grid1D(N=17, Lx=2.0, basis=basis, bc="outflow", backend="numpy")
    x = np.asarray(grid.x)
    f = np.exp(np.sin(3.0 * x)) + x ** 3
    w = np.asarray(grid.wx)
    assert abs(np.sum(w * grid.diffusion(f))) < 1e-9
    np.testing.assert_allclose(grid.diffusion(f)[1:-1], grid.laplacian(f)[1:-1])


@pytest.mark.parametrize("kwargs", [
    {"basis": "chebyshev", "bc": "unknown"},
    {"basis": "legendre", "bc": {"left": "outflow", "top": "wall"}},
    {"basis": "chebyshev", "bc": {"temperature": "fixed"}},
    {"basis": "fourier", "bc": "outflow"},
    {"basis": "chebyshev", "bc": "periodic"},
])
def test_invalid_boundary_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        Grid1D(N=9, Lx=1.0, backend="numpy", **kwargs)


def test_torch_cpu_grid_initializes_and_boundary_rhs_preserve_input(basis, mhd):
    torch = pytest.importorskip("torch")
    _, eqs, U, _ = _case(basis, mhd, "wall")
    grid = Grid1D(N=9, Lx=1.0, basis=basis, bc="wall", backend="torch", torch_device="cpu")
    assert isinstance(grid.x, torch.Tensor)
    assert grid.x.device.type == "cpu"
    state = torch.as_tensor(U, dtype=grid.x.dtype)
    original = state.clone()
    bounded = grid.apply_boundary_conditions(state, eqs)
    assert bounded.device == state.device
    assert bounded.dtype == state.dtype
    assert torch.equal(bounded, state)
    rhs = SpectralSolverMHD(grid, eqs)._compute_rhs(state) if mhd else _compute_rhs(grid, eqs, state)
    assert torch.isfinite(rhs).all()
    assert torch.equal(state, original)


def test_stage_rhs_preserves_input(basis, mhd):
    grid, eqs, U, solver = _case(basis, mhd, "wall")
    original = U.copy()
    rhs = solver._compute_rhs(U) if mhd else _compute_rhs(grid, eqs, U)
    assert np.all(np.isfinite(rhs))
    np.testing.assert_array_equal(U, original)


class _EndpointChangingStepper:
    def __init__(self, accepted):
        self.accepted = accepted

    def step(self, rhs, state, dt, scale):
        rhs(state)
        return AdaptiveStepResult(U_new=state*1.01, dt_used=dt, dt_next=dt,
                                  accepted=self.accepted, error=0.0, rejections=0)


def test_accepted_steps_restore_fixed_boundaries_and_rejection_is_untouched(basis, mhd):
    grid, eqs, U, solver = _case(basis, mhd, filter_params={"enabled": True, "alpha": 3.0})
    original = U.copy()
    grid.initialize_boundary_conditions(U, eqs)
    fixed = solver.step(U, 1e-6)
    np.testing.assert_allclose(fixed[:, [0, -1]], U[:, [0, -1]], atol=3e-15)
    solver.adaptive_enabled = True
    solver.adaptive_stepper = _EndpointChangingStepper(True)
    accepted = solver.step(U, 1e-6)
    np.testing.assert_allclose(accepted[:, [0, -1]], U[:, [0, -1]], atol=3e-15)
    solver.adaptive_stepper = _EndpointChangingStepper(False)
    assert solver.step(U, 1e-6) is U
    np.testing.assert_array_equal(U, original)


def test_uniform_state_preservation_and_mhd_projection(basis, mhd):
    grid, eqs, U, solver = _case(basis, mhd, "outflow")
    U = np.repeat(U[:, :1], grid.N, axis=1)
    np.testing.assert_allclose(solver.step(U, 1e-5), U, atol=2e-13)
    if mhd:
        perturbed = U.copy()
        perturbed[5] += 0.03*np.asarray(grid.x)
        perturbed[4] += 0.5*(perturbed[5]**2 - U[5]**2)
        projected = solver._project_B(perturbed)
        assert solver.max_div_b(projected) < 2e-12
        np.testing.assert_allclose(_pressure(projected, True), _pressure(perturbed, True), atol=2e-15)


@pytest.mark.parametrize("values", [(0.5, 1.0), (0.5, 0.5)])
def test_incompatible_fixed_longitudinal_field_rejected(basis, values):
    grid, eqs, U, _ = _case(basis, True)
    with pytest.raises(ValueError, match="Bx"):
        grid.set_dirichlet_bc_values({"bx": values})
        grid.apply_boundary_conditions(U, eqs)


def test_sensor_viscosity_respects_diffusion_coefficient_bound():
    from phrike.artificial_viscosity import ArtificialViscosityConfig, ArtificialViscosityCoefficient
    from phrike.solver import _artificial_viscosity_dt

    grid, _, _, solver = _case("chebyshev", False)
    config = ArtificialViscosityConfig(s_ref=0.1, nu_max=0.001)
    viscosity = ArtificialViscosityCoefficient(config).compute_viscosity(np.array([0.0, 0.5, 1.0]))
    assert viscosity.max() <= config.nu_max
    from types import SimpleNamespace
    dt = _artificial_viscosity_dt(grid, SimpleNamespace(config=config))
    diffusion = grid._basis.d2x(np.eye(grid.N)).T[1:-1, 1:-1]
    eigenvalues = np.linalg.eigvals(diffusion)
    assert np.max(np.abs(1 + dt*viscosity.max()*eigenvalues)) <= 1.0
