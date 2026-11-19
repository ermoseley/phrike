import numpy as np
import pytest

from phrike.equations import MHDEquations2D
from phrike.grid import Grid2D
from phrike.nonideal import MidpointAmbipolar
from phrike.solver import SpectralSolverMHD2D


def _grid(mixed=False, **kwargs):
    return Grid2D(Nx=12, Ny=12, Lx=2*np.pi, Ly=2*np.pi,
                  basis_y="sbp" if mixed else "fourier", **kwargs)


def _state(grid, amplitude=0.02):
    x, y = np.meshgrid(np.asarray(grid.x), np.asarray(grid.y))
    U = np.zeros((8, grid.Ny, grid.Nx))
    U[0] = 1
    U[5] = 1
    U[7] = amplitude*np.cos(x)
    U[4] = 2.0+0.5*(U[5:8]**2).sum(axis=0)
    return U


@pytest.mark.parametrize("eta_max", [None, 0.5], ids=["uncapped", "capped"])
def test_mixed_operator_is_weighted_dissipative_and_preserves_divergence(eta_max):
    grid = _grid(mixed=True, backend="numpy")
    ad = MidpointAmbipolar(grid, {"eta": 0.7, "eta_max": eta_max})
    rng = np.random.default_rng(3)
    a, b, reference = rng.normal(size=(3, 3, grid.Ny, grid.Nx))
    La, Lb = ad._operator(a, reference), ad._operator(b, reference)
    w = grid.wy[:, None]
    np.testing.assert_allclose((a*Lb*w).sum(), (b*La*w).sum(), atol=2e-11)
    assert (a*La*w).sum() <= 0
    # The magnetic work is the negative of the heating from every current in the retained band.
    B = grid.project_dealiased(a)
    dx, dy = grid.dx1(B), grid.dy1(B)
    J = np.stack([dy[2], -dx[2], dx[1]-dy[0]])
    eta = ad.eta
    if eta_max is not None:
        eta = np.minimum(eta, eta_max/(reference**2).sum(axis=0))
        assert eta.min() < ad.eta == eta.max()
    heating = (eta*w*(np.cross(J, reference, axis=0)**2).sum(axis=0)).sum()
    np.testing.assert_allclose((a*La*w).sum(), -heating, rtol=1e-12, atol=0)
    np.testing.assert_allclose(grid.divergence(La[0], La[1])[1:-1], 0, atol=1e-13)
    Bx, By = grid.project_solenoidal(La[0], La[1])
    np.testing.assert_allclose(grid.divergence(Bx, By), 0, atol=1e-12)
    np.testing.assert_allclose(ad._operator(np.ones_like(a), reference), 0, atol=1e-13)
    precondition = ad._precondition(reference, 0.2)
    np.testing.assert_allclose((a*precondition(b)).sum(),
                               (b*precondition(a)).sum(), atol=1e-12)
    assert (a*precondition(a)).sum() > 0
    U = _state(grid)
    result = ad.step(U, None, 0.02)
    np.testing.assert_allclose((result[4]*w).sum(), (U[4]*w).sum(), atol=1e-12)
    assert ((result[5:8]**2-U[5:8]**2)*w).sum() < 0


def test_mixed_operator_dissipates_perpendicular_face_currents():
    grid = Grid2D(Nx=12, Ny=9, Lx=1.0, Ly=1.0, basis_y="sbp", backend="numpy")
    ad = MidpointAmbipolar(grid, {"eta": 1.0})
    U = np.zeros((8, grid.Ny, grid.Nx))
    U[0] = 1
    B = U[5:8]
    B[0] = np.array([1, 1/59, 1, 0, 7/8, 0, 1, 1/59, 1])[:, None]
    U[4] = 2.0+0.5*B[0]**2
    Jz = -grid.dy1(B[0])
    np.testing.assert_allclose(Jz[1:-1], 0, atol=1e-13)
    reference = np.zeros_like(B)
    reference[0] = 0.7
    w = grid.wy[:, None]
    work = (w*B*ad._operator(B, reference)).sum()
    assert work < 0
    np.testing.assert_allclose(work, -ad.eta*0.7**2*(w*Jz**2).sum(), rtol=1e-12, atol=0)
    reference[0], reference[2] = 0, 0.7
    np.testing.assert_allclose(ad._operator(B, reference), 0, atol=1e-13)

    result = ad.step(U, None, 0.01)
    assert (w*result[5:8]**2).sum() < (w*B**2).sum()
    np.testing.assert_allclose((w*(result[4]-U[4])).sum(), 0, atol=1e-12)


def test_mixed_ambipolar_step_keeps_div_b_at_round_off():
    grid = _grid(mixed=True, backend="numpy")
    grid.mhd_config = {"ambipolar": {"enabled": True, "eta": 0.7}}
    solver = SpectralSolverMHD2D(grid, MHDEquations2D())
    x, y = grid.xy_mesh()
    one, zero = np.ones_like(x), np.zeros_like(x)
    U = solver.equations.conservative(one, zero, zero, zero, one, 0.7+0.1*np.sin(y),
                                      0.2*np.sin(x), 0.1*np.cos(x+y))
    assert solver.max_div_b(U) < 1e-13
    result = solver.step(U, 0.02)
    assert solver.ambipolar.total_solves == 4
    assert solver.max_div_b(result) < 1e-13


def test_linear_resolvent_decay_and_conservative_heating():
    grid = _grid(backend="numpy")
    U = _state(grid, amplitude=1e-5)
    ad = MidpointAmbipolar(grid, {"eta": 0.4, "midpoint_tol": 1e-11})
    h = 0.15
    result = ad.step(U, None, h)
    expected = U[7]/(1+0.4*h+0.5*(0.4*h)**2)
    np.testing.assert_allclose(result[7], expected, rtol=2e-9, atol=1e-14)
    np.testing.assert_allclose(result[4].sum(), U[4].sum(), atol=1e-12)
    heat = result[4]-0.5*(result[5:8]**2).sum(axis=0)-2.0
    assert heat.mean() > 0
    assert heat.min() > -1e-13
    assert ad.last_residual <= 1e-11
    assert ad.total_solves == 2


def test_nonlinear_coefficient_prediction_has_second_order_time_accuracy():
    grid = _grid(backend="numpy")
    initial = _state(grid, amplitude=0.3)

    def evolve(n):
        ad = MidpointAmbipolar(grid, {"eta": 0.4, "midpoint_tol": 1e-12})
        U = initial.copy()
        for _ in range(n):
            U = ad.step(U, None, 0.2/n)
        return U[5:8]

    reference = evolve(64)
    errors = [np.linalg.norm(evolve(n)-reference) for n in (2, 4, 8)]
    assert errors[0]/errors[1] > 3.4
    assert errors[1]/errors[2] > 3.4


def test_cg_failure_is_reported():
    grid = _grid(mixed=True, backend="numpy")
    U = _state(grid, amplitude=0.2)
    U[7] *= 1+0.2*np.sin(np.asarray(grid.y))[:, None]
    ad = MidpointAmbipolar(grid, {"eta": 1.0, "midpoint_maxiter": 1,
                                 "midpoint_tol": 1e-12})
    with pytest.raises(RuntimeError, match="did not converge"):
        ad.step(U, None, 0.1)


def test_torch_cpu_matches_numpy_on_mixed_grid():
    torch = pytest.importorskip("torch")
    numpy_grid = _grid(mixed=True, backend="numpy")
    torch_grid = _grid(mixed=True, backend="torch", torch_device="cpu")
    U = _state(numpy_grid)
    cfg = {"eta": 0.4, "midpoint_tol": 1e-10}
    expected = MidpointAmbipolar(numpy_grid, cfg).step(U, None, 0.02)
    actual = MidpointAmbipolar(torch_grid, cfg).step(torch.as_tensor(U), None, 0.02)
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-12)
