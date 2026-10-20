import numpy as np
import pytest

from phrike.equations import EulerEquations2D, MHDEquations2D
from phrike.grid import Grid2D
from phrike.solver import SpectralSolver2D, SpectralSolverMHD2D, _compute_rhs_2d


def test_mixed_derivatives_and_sbp_identity():
    grid = Grid2D(24, 33, 1.0, 2.0, basis_y="sbp", backend="numpy")
    X, Y = grid.xy_mesh()
    np.testing.assert_allclose(grid.dx1(np.sin(2*np.pi*X)),
                               2*np.pi*np.cos(2*np.pi*X), atol=4e-14)
    np.testing.assert_allclose(grid.dy1(Y**2), 2*Y, atol=4e-14)
    D = grid.dy1(np.eye(grid.Ny))
    boundary = np.zeros_like(D)
    boundary[0, 0], boundary[-1, -1] = -1, 1
    np.testing.assert_allclose(grid.wy[:, None]*D + D.T*grid.wy[None, :],
                               boundary, atol=4e-15)
    # Modifying the upper boundary cannot wrap into the lower derivative.
    far = Y*0
    far[-1] = 1.0
    np.testing.assert_array_equal(grid.dy1(far)[:4], 0.0)


@pytest.mark.parametrize("mhd", [False, True])
def test_open_boundaries_preserve_pressure_and_input(mhd):
    grid = Grid2D(16, 17, 1.0, 2.0, basis_y="sbp", backend="numpy")
    X, Y = grid.xy_mesh()
    rho, p = 1+0.1*Y, 2+0.2*Y
    eq = MHDEquations2D() if mhd else EulerEquations2D()
    if mhd:
        U = eq.conservative(rho, X*0+.1, Y*.2, Y*.3, p,
                            X*0+.4, Y*0+.2, X*0+.3)
    else:
        U = eq.conservative(rho, X*0+.1, Y*.2, p)
    original = U.copy()
    bounded = grid.apply_boundary_conditions(U, eq)
    np.testing.assert_array_equal(U, original)
    # Open endpoints evolve; their exterior state enters through a flux penalty.
    np.testing.assert_array_equal(bounded, U)
    index = 4 if mhd else 3
    np.testing.assert_allclose(eq.primitive(bounded)[index], p)
    totals = grid.conserved_quantities(U, eq)
    assert totals["mass"] == pytest.approx(grid.Lx*(grid.Ly+0.05*grid.Ly**2))
    assert totals["energy"] == pytest.approx(np.sum(U[index]*grid.wy[:, None])*grid.dx)
    rhs = grid.apply_boundary_flux(np.zeros_like(U), U, eq, eq.flux(U)[1])
    np.testing.assert_array_equal(rhs, 0)


def test_acoustic_pulse_leaves_open_boundary_without_recurrence():
    grid = Grid2D(4, 65, 1.0, 2.0, basis_y="sbp", backend="numpy")
    X, Y = grid.xy_mesh()
    eq = EulerEquations2D(gamma=5/3)
    dp = 1e-4*np.exp(-0.5*((Y-0.5)/0.1)**2)
    U = eq.conservative(1+dp, X*0, dp, 0.6+dp)
    solver = SpectralSolver2D(grid, eq, cfl=0.25)
    initial_energy = np.sum(grid.wy[:, None]*2*dp**2)
    t = 0.0
    while t < 4.0-1e-13:
        dt = min(solver.compute_dt(U), 4.0-t)
        U = solver.step(U, dt)
        t += dt
    rho, _, uy, p = eq.primitive(U)
    energy = np.sum(grid.wy[:, None]*(uy**2+(p-0.6)**2))
    assert energy/initial_energy < 1e-4
    assert rho.min() > 0 and p.min() > 0


def test_mixed_explicit_diffusion_respects_sbp_stability_bound():
    grid = Grid2D(8, 33, 1.0, 2.0, basis_y="sbp", backend="numpy")
    grid.mhd_config = {"resistivity": 1.0, "project_each_substage": False}
    eq = MHDEquations2D()
    X, _ = grid.xy_mesh()
    one, zero = X*0+1, X*0
    U = eq.conservative(one, zero, zero, zero, one, zero+0.3, zero, zero)
    solver = SpectralSolverMHD2D(grid, eq, cfl=0.35)
    D = grid._sbp_dy(np.eye(grid.Ny))
    root = np.sqrt(grid.wy)
    D = root[:, None]*D/root[None, :]
    eigenvalues, eigenvectors = np.linalg.eigh(D.T@D)
    U[7] = 1e-7*(eigenvectors[:, -1]/root)[:, None]
    dt = solver.compute_dt(U)
    assert dt*eigenvalues[-1] <= 1.0
    result = solver.step(U, dt)
    assert np.linalg.norm(result[7]) < np.linalg.norm(U[7])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_transverse_flow_enters_hydro_cfl(backend):
    if backend == "torch":
        pytest.importorskip("torch")
    grid = Grid2D(8, 17, 1.0, 2.0, basis_y="sbp", backend=backend,
                  torch_device="cpu" if backend == "torch" else None)
    X, _ = grid.xy_mesh()
    eq = EulerEquations2D(gamma=5/3)
    U = eq.conservative(X*0+1, X*0, X*0+3, X*0+0.6)
    solver = SpectralSolver2D(grid, eq, cfl=0.35)
    assert solver.compute_dt(U) == pytest.approx(0.35*min(grid.dx, grid.dy)/4)


@pytest.mark.parametrize("mhd", [False, True])
def test_transverse_nonlinear_rhs_stays_in_retained_fourier_band(mhd):
    grid = Grid2D(24, 17, 1.0, 2.0, basis_y="sbp", fft_workers=1)
    X, Y = grid.xy_mesh()
    zero, one = X*0, X*0+1
    vy = 0.01*Y*np.cos(14*np.pi*X)
    eq = MHDEquations2D() if mhd else EulerEquations2D()
    U = (eq.conservative(one, zero, vy, zero, one, zero+0.3, zero, zero+0.3)
         if mhd else eq.conservative(one, zero, vy, one))
    U = grid.project_dealiased(U)
    rhs = (SpectralSolverMHD2D(grid, eq)._compute_rhs(U) if mhd
           else _compute_rhs_2d(grid, eq, U))
    spectrum = np.fft.fft(rhs, axis=-1)/grid.Nx
    excluded = abs(np.fft.fftfreq(grid.Nx)*grid.Nx) > grid.Nx/3
    np.testing.assert_allclose(spectrum[..., excluded], 0, atol=1e-14)


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_mixed_projection_allows_normal_flux_and_preserves_divergence(backend):
    if backend == "torch":
        torch = pytest.importorskip("torch")
    grid = Grid2D(16, 17, 1.0, 2.0, basis_y="sbp", backend=backend,
                  torch_device="cpu" if backend == "torch" else None)
    rng = np.random.default_rng(8)
    B = rng.normal(size=(2, 17, 16))
    if backend == "torch":
        B = torch.as_tensor(B, dtype=grid.x.dtype)
    Bx, By = grid.project_solenoidal(*grid.project_dealiased(B))
    divergence = grid.divergence(Bx, By)
    if backend == "torch":
        divergence = divergence.numpy()
    assert np.max(abs(divergence)) < 3e-11
    constant = grid.y[:, None]*0 + grid.x[None, :]*0 + .7
    bx, by = grid.project_solenoidal(constant*0, constant)
    if backend == "torch":
        bx, by = bx.numpy(), by.numpy()
    np.testing.assert_allclose(bx, 0.0, atol=1e-14)
    np.testing.assert_allclose(by, .7, atol=1e-14)
