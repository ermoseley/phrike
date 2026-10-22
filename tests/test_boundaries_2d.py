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


@pytest.mark.parametrize("mhd", [False, True])
def test_sbp_dissipation_controls_stationary_grid_scale_shear(mhd):
    grid = Grid2D(4, 33, 1.0, 2.0, basis_y="sbp", backend="numpy",
                  filter_params={"sbp_dissipation": {"enabled": True, "strength": 2.0}})
    X, _ = grid.xy_mesh()
    one, zero = X*0+1, X*0
    ux = 1e-4*np.broadcast_to((-1.0)**np.arange(grid.Ny)[:, None], X.shape)
    eq = MHDEquations2D() if mhd else EulerEquations2D()
    U = (eq.conservative(one, ux, zero, zero, one, zero+.3, zero, zero+.3)
         if mhd else eq.conservative(one, ux, zero, one))
    solver = (SpectralSolverMHD2D(grid, eq, cfl=.65) if mhd
              else SpectralSolver2D(grid, eq, cfl=.65))
    dt = solver.compute_dt(U)
    assert dt*grid.sbp_dissipation_rate_bound(eq.max_wave_speed(U)) <= 1.0+1e-14
    result = solver.step(U, dt)
    assert np.sum(grid.wy[:, None]*result[1]**2) < np.sum(grid.wy[:, None]*U[1]**2)
    assert eq.primitive(result)[4 if mhd else 3].min() > 0.0


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
    grid = Grid2D(24, 17, 1.0, 2.0, basis_y="sbp", fft_workers=1, backend="numpy")
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


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("mhd", [False, True])
@pytest.mark.parametrize("basis_y", ["sbp", "fourier"])
def test_split_advection_preserves_momentum_and_other_components(backend, mhd, basis_y):
    if backend == "torch":
        torch = pytest.importorskip("torch")
    grid = Grid2D(8, 33, 1.0, 2.0, basis_y=basis_y, backend=backend,
                  torch_device="cpu" if backend == "torch" else None)
    X, Y = grid.xy_mesh()
    sin, cos = (torch.sin, torch.cos) if backend == "torch" else (np.sin, np.cos)
    rho = 1+.12*cos(2*np.pi*Y/grid.Ly)+.03*cos(2*np.pi*X)
    vx, vy = .4+.2*sin(16*np.pi*Y/grid.Ly), .3+.06*cos(18*np.pi*Y/grid.Ly)
    eq = MHDEquations2D() if mhd else EulerEquations2D()
    U = (eq.conservative(rho, vx, vy, .2+.1*cos(14*np.pi*Y/grid.Ly),
                         X*0+2, X*0+.3, X*0+.2, X*0+.1)
         if mhd else eq.conservative(rho, vx, vy, X*0+2))
    original = np.asarray(U).copy()
    rhs = U*.01
    before = np.asarray(rhs).copy()
    assert grid.apply_split_advection(rhs, U) is rhs
    np.testing.assert_array_equal(U, original)
    if basis_y == "fourier":
        np.testing.assert_array_equal(rhs, before)
        return
    momentum = slice(1, 4 if mhd else 3)
    change = np.asarray(rhs)-before
    np.testing.assert_allclose(np.sum(change[momentum]*grid.wy[:, None],
                                     axis=(-2, -1))*grid.dx, 0, atol=2e-14)
    unchanged = [0, 4, 5, 6, 7] if mhd else [0, 3]
    np.testing.assert_array_equal(np.asarray(rhs)[unchanged], before[unchanged])


@pytest.mark.parametrize("backend", ["numpy", "torch"])
@pytest.mark.parametrize("mhd", [False, True])
def test_split_advection_obeys_sbp_kinetic_energy_boundary_balance(backend, mhd):
    if backend == "torch":
        torch = pytest.importorskip("torch")
    grid = Grid2D(8, 33, 1.0, 2.0, basis_y="sbp", backend=backend,
                  torch_device="cpu" if backend == "torch" else None)
    X, Y = grid.xy_mesh()
    sin, cos = (torch.sin, torch.cos) if backend == "torch" else (np.sin, np.cos)
    rho = 1+.12*cos(2*np.pi*Y/grid.Ly)
    vx = .4+.2*sin(16*np.pi*Y/grid.Ly)
    vy = .3+.06*cos(18*np.pi*Y/grid.Ly)+.02*Y
    eq = MHDEquations2D() if mhd else EulerEquations2D()
    U = (eq.conservative(rho, vx, vy, .2+.1*cos(14*np.pi*Y/grid.Ly),
                         X*0+2, X*0+.3, X*0+.2, X*0+.1)
         if mhd else eq.conservative(rho, vx, vy, X*0+2))
    momentum = slice(1, 4 if mhd else 3)
    velocity, q = U[momentum]/U[0], U[2]
    rhs = U*0
    rhs[0], rhs[momentum] = -grid.dy1(q), -grid.dy1(q*velocity)
    conservative = np.asarray(rhs).copy()
    grid.apply_split_advection(rhs, U)
    velocity, q = np.asarray(velocity), np.asarray(q)
    speed2 = np.sum(velocity**2, axis=0)
    expected = -.5*grid.dx*np.sum(q[-1]*speed2[-1]-q[0]*speed2[0])
    corrected = np.asarray(rhs)
    rate = np.sum(grid.dx*grid.wy[:, None]*(np.sum(velocity*corrected[momentum],
                  axis=0)-.5*speed2*corrected[0]))
    assert rate == pytest.approx(expected, rel=1e-12, abs=1e-13)
    conservative_rate = np.sum(grid.dx*grid.wy[:, None]*(np.sum(
        velocity*conservative[momentum], axis=0)-.5*speed2*conservative[0]))
    assert abs(conservative_rate-expected) > 1e-4
