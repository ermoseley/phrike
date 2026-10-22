import itertools

import numpy as np
import pytest

from phrike.equations import MHDEquations1D, MHDEquations2D, MHDEquations3D
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.io import load_checkpoint, save_solution_snapshot
from phrike.nonideal import Hall, MidpointAmbipolar
from phrike.problems.problem_list.adsi_shear import (
    ADSIShearProblem, adsi_incompressible_roots, adsi_shear_matrix, adsi_shear_mode, adsi_shear_vector,
)
from phrike.problems.problem_list.hall_mri1d import (
    HallMRI1DProblem, hall_mri_fastest, hall_mri_matrix, hall_mri_mode,
)
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d
from phrike.solver import SpectralSolverMHD1D, SpectralSolverMHD2D, SpectralSolverMHD3D


def grid1d(**kwargs):
    return Grid1D(N=24, Lx=2*np.pi, backend="numpy", **kwargs)


def state(grid, B, rho=None):
    one = np.ones_like(grid.x)
    return MHDEquations1D().conservative(one if rho is None else rho,
                                         0*one, 0*one, 0*one, one, *B)


def solver(grid, **blocks):
    grid.mhd_config = blocks
    return SpectralSolverMHD1D(grid=grid, equations=MHDEquations1D())


def solver2d(grid, **blocks):
    grid.mhd_config = blocks
    return SpectralSolverMHD2D(grid=grid, equations=MHDEquations2D())


@pytest.mark.parametrize("method,eta", [("reference", 0.2), ("reference", -0.2),
                                        ("inertia", 0.2), ("inertia", -0.2)])
def test_circular_hall_rotation(method, eta):
    g = grid1d()
    amp, length, dt = 0.02, 0.3, 0.1
    U = state(g, [np.ones_like(g.x), amp*np.cos(g.x), amp*np.sin(g.x)])
    hall = Hall(g, dict(eta_h=eta, method=method, length=length, safety=0.25))
    V = hall.step(U, None, dt)
    omega = eta/np.sqrt(1+amp**2)
    if method == "inertia":
        omega /= 1+length**2
    assert np.max(np.abs(V[6]-amp*np.cos(g.x+omega*dt))) < 2e-10
    assert np.max(np.abs(V[7]-amp*np.sin(g.x+omega*dt))) < 2e-10
    eq = MHDEquations1D()
    assert np.max(np.abs(eq.primitive(V)[4]-eq.primitive(U)[4])) < 1e-14
    assert np.array_equal(V[:4], U[:4]) and np.array_equal(V[5], U[5])


@pytest.mark.parametrize("dim", [2, 3])
def test_circular_rotation_in_2d_and_3d(dim):
    n = 12
    if dim == 2:
        g = Grid2D(Nx=n, Ny=n, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
        eq, X = MHDEquations2D(), np.meshgrid(g.x, g.y)[0]
    else:
        g = Grid3D(Nx=n, Ny=n, Nz=n, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, backend="numpy")
        eq, X = MHDEquations3D(), g.xyz_mesh()[0]
    one = np.ones_like(X)
    amp, eta, length, dt = 0.02, 0.2, 0.3, 0.1
    U = eq.conservative(one, 0*one, 0*one, 0*one, one, one, amp*np.cos(X), amp*np.sin(X))
    V = Hall(g, dict(eta_h=eta, method="inertia", length=length)).step(U, eq, dt)
    omega = eta/np.sqrt(1+amp**2)/(1+length**2)
    assert np.max(np.abs(V[6]-amp*np.cos(X+omega*dt))) < 2e-10
    assert np.max(np.abs(V[7]-amp*np.sin(X+omega*dt))) < 2e-10


@pytest.mark.parametrize("method", ["reference", "inertia"])
def test_oblique_circular_rotation_is_axis_permutation_invariant(method):
    """B = k_hat + amp (e1 cos + e2 sin)(k.x) has constant |B| and curl b = -|k| b, so the Hall
    subflow rotates it exactly at omega = eta |k|^2/|B| (divided by 1 + l^2 |k|^2 for inertia).
    The distinct, signed components (2, 1, -1) are permuted over the axes: any kx/ky/kz mix-up
    in the symbols, which the symmetric k = (1, 1, 1) cannot see, changes the result."""
    n, amp, eta, length, dt = 12, 0.02, 0.05, 0.3, 0.1
    g = Grid3D(Nx=n, Ny=n, Nz=n, Lx=2*np.pi, Ly=2*np.pi, Lz=2*np.pi, backend="numpy")
    eq, (X, Y, Z) = MHDEquations3D(), g.xyz_mesh()
    one = np.ones_like(X)
    hall = Hall(g, dict(eta_h=eta, method=method, length=length))
    for k in itertools.permutations((2.0, 1.0, -1.0)):
        k = np.array(k)
        kn = np.linalg.norm(k)
        khat = k/kn
        e1 = np.cross(khat, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(khat, e1)
        phase = k[0]*X+k[1]*Y+k[2]*Z
        field = lambda a: np.array([khat[i]+amp*(e1[i]*np.cos(a)+e2[i]*np.sin(a)) for i in range(3)])
        omega = eta*kn**2/np.sqrt(1+amp**2)
        if method == "inertia":
            omega /= 1+length**2*kn**2
        U = eq.conservative(one, 0*one, 0*one, 0*one, one, *field(phase))
        V = hall.step(U, eq, dt)
        assert np.max(np.abs(V[5:8]-field(phase+omega*dt))) < 1e-9, k


@pytest.mark.parametrize("force_free", [True, False])
def test_hall_equilibria(force_free):
    g = grid1d()
    zero = g.x*0
    B = np.array([zero, np.cos(g.x), np.sin(g.x)] if force_free
                 else [zero, 1+0.1*np.cos(g.x), zero])
    U = state(g, B)
    for method in ("reference", "inertia"):
        hall = Hall(g, dict(eta_h=1.0, method=method, length=0.3, length_mode="local"))
        E = hall.electric_field(B, U[0])
        if not force_free:
            assert np.max(np.abs(E)) > 0.05
        V = hall.step(U, None, 0.03)
        assert np.max(np.abs(V-U)) < 2e-13


def test_variable_inertia_spd_and_energy():
    g = grid1d()
    B = np.array([1+0*g.x, 0.1*np.cos(g.x)+0.03*np.sin(2*g.x),
                  0.1*np.sin(g.x)+0.02*np.cos(3*g.x)])
    rho = 1+0.2*np.cos(g.x)
    length2 = 0.3**2/rho
    hall = Hall(g, dict(eta_h=0.2, eta_h_form="proportional_to_B_over_rho", method="inertia",
                        length=0.3, length_mode="local", cg_tol=1e-11))
    rhs = hall.curl(B)
    x, iterations, residual = hall.inertia_solve(rhs, length2, tol=1e-11)
    assert iterations > 0 and residual < 1e-11
    assert np.linalg.norm(x+hall.curl(length2*hall.curl(x))-rhs) < 2e-11
    constant, _, _ = hall.inertia_solve(rhs, 0.09)
    cg, _, _ = hall.inertia_solve(rhs, 0*length2+0.09)
    assert np.max(np.abs(cg-constant)) < 1e-11
    U = state(g, B, rho)
    V = hall.step(U, None, 0.01)
    assert abs(hall.modified_energy(V[5:8], length2) - hall.modified_energy(B, length2)) < 1e-10
    eq = MHDEquations1D()
    assert np.max(np.abs(eq.primitive(V)[4]-eq.primitive(U)[4])) < 1e-14
    assert hall.last_stats["cg_calls"] > 0 and hall.total["cg_residual"] <= 1e-11
    with pytest.raises(RuntimeError, match="failed"):
        hall.inertia_solve(rhs, length2, tol=1e-15, maxiter=1)


def test_variable_inertia_2d_preserves_curl_free_part():
    g = Grid2D(Nx=24, Ny=24, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
    X, Y = np.meshgrid(g.x, g.y)
    hall = Hall(g, dict(eta_h=1.0, method="inertia"))
    potential = np.sin(X)*np.cos(Y)
    gradient = np.array([g.dx1(potential)+1, g.dy1(potential), 0*X])
    length2 = 0.04/(0.01+np.exp(-((X-np.pi)**2+(Y-np.pi)**2)))
    fixed, _, _ = hall.inertia_solve(gradient, length2)
    assert np.array_equal(fixed, gradient)
    rhs = gradient+hall.curl(np.array([0*X, 0*X, np.cos(X+Y)]))
    value, _, residual = hall.inertia_solve(rhs, length2)
    assert residual <= 1e-10
    change = value-rhs
    assert np.max(np.abs(g.divergence(change[0], change[1]))) < 2e-11
    assert np.max(np.abs(change.mean(axis=(1, 2)))) < 1e-13


def test_hall_length_ratio_bounds_and_only_raises():
    g = grid1d()
    rho = 0.01+np.exp(-(g.x-np.pi)**2)
    U = state(g, np.array([1+0*g.x, 0*g.x, 0*g.x]), rho)
    cfg = dict(eta_h=1.0, method="inertia", length=0.3, length_mode="local", rho0=1.0)
    free = Hall(g, cfg).length_squared(U, 0.1)
    assert np.array_equal(Hall(g, dict(cfg, length_ratio=None)).length_squared(U, 0.1), free)
    assert np.sqrt(free.max()/free.min()) > 4
    for ratio in (1., 2., 4.):
        capped = Hall(g, dict(cfg, length_ratio=ratio)).length_squared(U, 0.1)
        assert np.sqrt(capped.max()/capped.min()) <= ratio*(1+1e-12)
        assert np.all(capped >= free) and capped.max() == free.max()
    for bad in (0.5, np.inf, np.nan):
        with pytest.raises(ValueError, match="length_ratio"):
            Hall(g, dict(cfg, length_ratio=bad))


def test_automatic_length_is_clipped_at_zero_and_scales_with_dt():
    g = grid1d()
    U = state(g, [1+0*g.x, 0.01*np.cos(g.x), 0.01*np.sin(g.x)])
    hall = Hall(g, dict(eta_h=2.0, method="inertia", safety=0.5))
    k2max = hall.kmax2(U[5:8])
    assert hall.length_squared(U, 0.1*0.5/(2.0*k2max)) == 0.0
    assert hall.length_squared(U, 0.1) == pytest.approx(2.0*0.1/0.5-1.0/k2max)
    assert Hall(g, dict(eta_h=-2.0, method="inertia")).length_squared(U, 0.1) == pytest.approx(
        hall.length_squared(U, 0.1))


def test_hall_eta_form_is_case_insensitive_and_validated():
    g = grid1d()
    cfg = dict(eta_h=0.3, b0=2.0, rho0=3.0)
    B = np.array([1+0*g.x, 0.3*np.cos(g.x), 0.3*np.sin(g.x)])
    rho = 1+0.2*np.cos(g.x)
    E = [Hall(g, dict(cfg, eta_h_form=form)).electric_field(B, rho)
         for form in ("proportional_to_B_over_rho", "PROPORTIONAL_TO_b_over_RHO")]
    assert np.array_equal(E[0], E[1])
    assert np.max(np.abs(E[0])) > 0
    with pytest.raises(ValueError, match="eta_h_form"):
        Hall(g, dict(cfg, eta_h_form="B_over_rho"))
    with pytest.raises(ValueError, match="method"):
        Hall(g, dict(cfg, method="implicit"))


def test_reservoir_energy_is_modified_energy_minus_magnetic():
    g = grid1d()
    hall = Hall(g, dict(eta_h=1.0, method="inertia"))
    B = np.array([1+0*g.x, 0.1*np.cos(g.x), 0.1*np.sin(2*g.x)])
    length2 = 0.09/(1+0.2*np.cos(g.x))
    reservoir = hall.reservoir_energy(B, length2)
    assert reservoir > 0
    assert hall.reservoir_energy(B, 0.0) == 0
    assert abs(hall.modified_energy(B, length2) - np.mean(0.5*(B**2).sum(axis=0)) - reservoir) < 1e-14


def test_ideal_cfl_and_reference_internal_subcycling():
    g = grid1d()
    U = state(g, [1+0*g.x, 0.01*np.cos(g.x), 0.01*np.sin(g.x)])
    reference = solver(g, hall=dict(enabled=True, eta_h=10., method="reference"))
    dt = reference.compute_dt(U)
    reference.step(U, dt)
    assert reference.hall.last_stats["substeps"] > 1
    inertia = solver(g, hall=dict(enabled=True, eta_h=10., method="inertia"))
    assert inertia.compute_dt(U) == dt
    inertia.step(U, dt)
    assert inertia.hall.last_stats["substeps"] <= 2
    assert inertia.hall.total["calls"] == 2


def test_absent_hall_leaves_fixed_step_unchanged():
    g = grid1d()
    U = state(g, [1+0*g.x, 0.1*np.cos(g.x), 0.1*np.sin(g.x)])
    plain = solver(g)
    assert plain.hall is None and plain.shearing_sheet is None
    dt = plain.compute_dt(U)
    expected = plain._project_B(plain._rk_substages(g.apply_spectral_dissipation(U, 0.5*dt), dt))
    expected = g.apply_boundary_conditions(
        g.apply_accepted_state_stabilization(expected, 0.5*dt), plain.equations)
    assert np.array_equal(plain.step(U, dt), expected)
    assert solver(g, hall=dict(enabled=False, eta_h=1.0)).hall is None
    # A Hall block with eta_h = 0 returns copies, so the step is bitwise unchanged.
    assert np.array_equal(solver(g, hall=dict(enabled=True, eta_h=0.0)).step(U, dt), expected)


def test_hall_split_is_palindromic_around_the_ideal_step():
    g = Grid2D(Nx=16, Ny=16, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
    eq = MHDEquations2D()
    X, Y = np.meshgrid(g.x, g.y)
    one = np.ones_like(X)
    U = eq.conservative(one, 0.05*np.sin(Y), 0*one, 0*one, one, 1+0.1*np.sin(Y),
                        0.1*np.cos(X), 0.1*np.sin(X+Y))
    g.mhd_config = dict(hall=dict(enabled=True, eta_h=0.4, method="inertia", length=0.2),
                        ambipolar=dict(enabled=True, eta=0.3, midpoint_tol=1e-12))
    s = SpectralSolverMHD2D(grid=g, equations=eq)
    dt, h = 0.02, 0.01
    V = s.hall.step(U, eq, h)
    V = s.ambipolar.step(V, eq, h)
    V = s._project_B(s._rk_substages(V, dt))
    V = s.hall.step(s.ambipolar.step(V, eq, h), eq, h)
    V = g.apply_accepted_state_stabilization(V, h)
    assert np.max(np.abs(s.step(U, dt)-V)) < 1e-12


def test_hall_rejects_adaptive_stepping_and_non_fourier_grids():
    g = grid1d()
    g.mhd_config = dict(hall=dict(enabled=True, eta_h=1.0))
    with pytest.raises(ValueError, match="fixed-step"):
        SpectralSolverMHD1D(grid=g, equations=MHDEquations1D(),
                            adaptive_config=dict(enabled=True))
    mixed = Grid2D(Nx=12, Ny=12, Lx=1.0, Ly=1.0, basis_y="sbp", backend="numpy")
    with pytest.raises(ValueError, match="periodic Fourier"):
        Hall(mixed, dict(eta_h=1.0))


def test_torch_cpu_matches_numpy_in_2d():
    torch = pytest.importorskip("torch")
    eq = MHDEquations2D()
    numpy_grid = Grid2D(Nx=16, Ny=16, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
    torch_grid = Grid2D(Nx=16, Ny=16, Lx=2*np.pi, Ly=2*np.pi, backend="torch", torch_device="cpu")
    X, Y = np.meshgrid(numpy_grid.x, numpy_grid.y)
    one = np.ones_like(X)
    rho = 1+0.2*np.cos(X+Y)
    U = eq.conservative(rho, 0*one, 0*one, 0*one, one, 1+0.1*np.sin(Y),
                        0.1*np.cos(X), 0.1*np.sin(X+Y))
    for cfg in (dict(eta_h=0.4, method="reference"),
                dict(eta_h=0.4, method="inertia", length=0.2),
                dict(eta_h=0.4, method="inertia", length=0.2, length_mode="local", cg_tol=1e-11)):
        expected = Hall(numpy_grid, cfg).step(U, eq, 0.05)
        actual = Hall(torch_grid, cfg).step(torch.as_tensor(U), eq, 0.05)
        np.testing.assert_allclose(actual.numpy(), expected, atol=2e-11)


def test_shear_profile_eigenvalue():
    from scripts.hall_viability import shear_mode
    sigma, by, dby, residual = shear_mode(96, 6*np.pi, 0.)
    assert abs(sigma-1) < 1e-12 and residual < 1e-10
    assert np.max(np.abs(by-1)) < 1e-12 and np.max(np.abs(dby)) < 1e-12
    fine, _, _, residual = shear_mode(128, 6*np.pi, 0.01)
    coarse, _, _, _ = shear_mode(64, 6*np.pi, 0.01)
    assert 0 < fine < 1 and abs(fine/coarse-1) < 0.01 and residual < 1e-10


@pytest.mark.parametrize("radial", [1, 2])
def test_shearing_sheet_sources_and_energy_bookkeeping(radial):
    g = grid1d()
    one = np.ones_like(g.x)
    eq = MHDEquations1D()
    U = eq.conservative(1.3*one, 0.2*one, 0.4*one, -0.7*one, 2.0*one, 0.9*one, 0.3*one, -0.2*one)
    omega, shear = 0.8, 1.1
    diff = (solver(g, shearing_sheet=dict(omega=omega, shear=shear, radial_axis=radial))
            ._compute_rhs(U) - solver(g)._compute_rhs(U))
    # phi = z_v x r: +z for radial y, -y for radial z (vertical is x).
    phi, sign = 3-radial, 1 if radial == 1 else -1
    rho, u, B = U[0], U[1:4]/U[0], U[5:8]
    ur, uf, Br, Bf = u[radial], sign*u[phi], B[radial], sign*B[phi]
    expected = np.zeros_like(U)
    expected[1+radial] = 2*omega*rho*uf
    expected[1+phi] = sign*(-(2*omega-shear)*rho*ur)
    expected[5+phi] = sign*(-shear*Br)
    expected[4] = shear*(rho*ur*uf-Br*Bf)
    np.testing.assert_allclose(diff, expected, atol=1e-13)
    # The sources move energy between kinetic, magnetic and total, never internal.
    pressure_rate = diff[4]-(u*diff[1:4]).sum(axis=0)-(B*diff[5:8]).sum(axis=0)
    assert np.max(np.abs(pressure_rate)) < 1e-13


def test_shearing_sheet_defaults_and_validation():
    g = grid1d()
    sheet = solver(g, shearing_sheet=dict(omega=2.0)).shearing_sheet
    assert sheet["shear"] == 3.0 and (sheet["radial"], sheet["azimuthal"], sheet["sign"]) == (1, 2, 1)
    assert solver(g, shearing_sheet=dict(omega=0.0, shear=1.0)).shearing_sheet["shear"] == 1.0
    for bad in (dict(omega=1.0, vertical_axis=1), dict(omega=1.0, radial_axis=0),
                dict(omega=float("nan"))):
        with pytest.raises(ValueError):
            solver(g, shearing_sheet=bad)
    # 2D: vertical x and radial y are the grid axes, azimuthal z is ignorable.
    g2 = Grid2D(Nx=8, Ny=8, Lx=1.0, Ly=1.0, backend="numpy")
    sheet = solver2d(g2, shearing_sheet=dict(omega=1.0)).shearing_sheet
    assert (sheet["radial"], sheet["azimuthal"], sheet["sign"]) == (1, 2, 1)
    for bad in (dict(omega=1.0, vertical_axis=1), dict(omega=1.0, radial_axis=2)):
        with pytest.raises(ValueError):
            solver2d(g2, shearing_sheet=bad)
    g3 = Grid3D(Nx=8, Ny=8, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0, backend="numpy")
    g3.mhd_config = dict(shearing_sheet=dict(omega=1.0))
    with pytest.raises(NotImplementedError):
        SpectralSolverMHD3D(grid=g3, equations=MHDEquations3D())


def test_shearing_sheet_2d_sources_match_1d_and_are_local():
    g1 = grid1d()
    g2 = Grid2D(Nx=24, Ny=8, Lx=2*np.pi, Ly=1.0, backend="numpy")
    sheet = dict(omega=0.8, shear=1.1)
    x = g1.x
    U1 = MHDEquations1D().conservative(1.3+0.1*np.cos(x), 0.2+0.05*np.sin(x), 0.4*np.ones_like(x),
                                       -0.7+0.1*np.cos(2*x), 2.0+0*x, 0.9+0*x, 0.3*np.cos(x),
                                       -0.2+0.1*np.sin(x))
    d1 = solver(g1, shearing_sheet=sheet)._compute_rhs(U1)-solver(g1)._compute_rhs(U1)
    # A state independent of r: every row of the 2D sources is the 1D source.
    U2 = np.repeat(U1[:, None, :], g2.Ny, axis=1)
    d2 = solver2d(g2, shearing_sheet=sheet)._compute_rhs(U2)-solver2d(g2)._compute_rhs(U2)
    for j in range(g2.Ny):
        np.testing.assert_allclose(d2[:, j, :], d1, atol=1e-13)
    # A state that depends on r and x: the sources stay pointwise (vertical x, radial y, phi = +z).
    X, Y = np.meshgrid(g2.x, g2.y)
    one = np.ones_like(X)
    V = MHDEquations2D().conservative(1.3+0.1*np.cos(X+Y), 0.2*one, 0.4+0.1*np.sin(Y), -0.7+0.1*np.cos(X),
                                      2.0*one, 0.9*one, 0.3*np.cos(Y), -0.2+0.1*np.sin(X-Y))
    d = solver2d(g2, shearing_sheet=sheet)._compute_rhs(V)-solver2d(g2)._compute_rhs(V)
    omega, shear = sheet["omega"], sheet["shear"]
    expected = np.zeros_like(V)
    expected[2] = 2*omega*V[3]
    expected[3] = -(2*omega-shear)*V[2]
    expected[7] = -shear*V[6]
    expected[4] = shear*(V[2]*V[3]/V[0]-V[6]*V[7])
    np.testing.assert_allclose(d, expected, atol=1e-13)


def test_ideal_mri_growth_and_hall_neutral_curves():
    kv = np.sqrt(15.0)/4.0
    s, _ = hall_mri_mode(kv, omega=1.0, shear=1.5, field=1.0)
    assert abs(s.real-0.75) < 1e-12 and abs(s.imag) < 1e-12
    k, s, _ = hall_mri_fastest(1.0, 1.5, 1.0)
    assert abs(k-kv) < 1e-3 and abs(s.real-0.75) < 1e-6
    # Keplerian marginal stability: w^2 = 3 - 2 h and w^2 = -h/2 with h = eta_H sgn(B0) k^2.
    for eta, k_c in ((0.5, np.sqrt(1.5)), (-0.25, np.sqrt(6.0))):
        inside = hall_mri_mode(0.9*k_c, 1.0, 1.5, 1.0, eta_h=eta)[0].real
        outside = hall_mri_mode(1.1*k_c, 1.0, 1.5, 1.0, eta_h=eta)[0].real
        assert inside > 0.05 and outside < 1e-9
    # Reversing the field flips sgn(B0) and the vertical tension together, so only the
    # sign of eta_H (Omega . B0) matters.
    assert hall_mri_mode(0.7, 1.0, 1.5, -1.0, eta_h=-0.5)[0].real == pytest.approx(
        hall_mri_mode(0.7, 1.0, 1.5, 1.0, eta_h=0.5)[0].real)
    assert abs(hall_mri_mode(0.7, 1.0, 1.5, 1.0, eta_h=-0.5)[0].real
               - hall_mri_mode(0.7, 1.0, 1.5, 1.0, eta_h=0.5)[0].real) > 1e-2


def test_hall_shear_instability_closed_form_and_sign():
    S, eta = 1.0, 2.0
    for k in (0.3, 0.5, 1.0):
        x, h = k*k, eta*k*k
        z = (-(2*x+h*(h-S))+abs(h)*np.sqrt(4*x+(h-S)**2))/2
        assert hall_mri_mode(k, 0.0, S, 1.0, eta_h=eta)[0].real == pytest.approx(np.sqrt(z))
        assert hall_mri_mode(k, 0.0, S, 1.0, eta_h=-eta)[0].real < 1e-9


@pytest.mark.parametrize("hall", [None, dict(eta_h=0.5, method="reference"),
                                  dict(eta_h=-0.25, method="inertia", length=0.2)])
def test_hall_mri_eigenmode_grows_at_the_linear_rate(hall, tmp_path):
    mhd = dict(enabled=True, shearing_sheet=dict(omega=1.0, q=1.5))
    if hall is not None:
        mhd["hall"] = dict(hall, enabled=True)
    problem = HallMRI1DProblem(config=dict(
        problem="hall_mri1d", grid=dict(N=16, Lx="auto", precision="double"),
        physics=dict(gamma=5.0/3.0, mhd=mhd),
        integration=dict(t0=0.0, t_end=3.0, cfl=0.3, scheme="rk4"),
        initial_conditions=dict(rho0=1.0, B0=1.0, beta=100.0, mode=1, amplitude=1e-6),
        io=dict(outdir=str(tmp_path))))
    g = problem.create_grid("numpy")
    U = problem.create_initial_conditions(g)
    s = SpectralSolverMHD1D(grid=g, equations=problem.create_equations(), cfl=0.3)
    steps = int(np.ceil(3.0/s.compute_dt(U)))
    dt, amplitude = 3.0/steps, [problem.amplitude_history[0][1]]
    for i in range(steps):
        U = s.step(U, dt)
        if i == steps//2-1:
            amplitude.append(problem._amplitude(U))
    amplitude.append(problem._amplitude(U))
    growth = np.log(amplitude[2]/amplitude[1])/(dt*(steps-steps//2))
    assert growth == pytest.approx(problem.linear_mode["growth"], rel=0.02)
    assert problem.linear_mode["growth"] > 0.1


def test_float32_inertia_cg_tolerance_is_floored_at_roundoff():
    g = Grid2D(Nx=24, Ny=24, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
    X, Y = np.meshgrid(g.x, g.y)
    hall = Hall(g, dict(eta_h=1.0, method="inertia"))
    B = np.array([1+0*X, 0.1*np.cos(X+Y), 0.1*np.sin(Y)])
    rhs = hall.curl(B).astype(np.float32)
    length2 = (0.01*(1+0.5*np.sin(X))).astype(np.float32)
    # A float32 residual of 1e-10 is unreachable; the request is raised to 100 eps.
    x, iterations, residual = hall.inertia_solve(rhs, length2, tol=1e-10)
    assert x.dtype == np.float32 and iterations > 0
    assert residual <= 100*np.finfo(np.float32).eps
    # Double precision keeps the requested tolerance.
    _, _, residual = hall.inertia_solve(rhs.astype(np.float64), length2.astype(np.float64), tol=1e-10)
    assert residual <= 1e-10


@pytest.mark.parametrize("length_mode", ["constant", "local"])
def test_hall_inertia_restart_is_bitwise(length_mode, tmp_path):
    n, half = 16, 3
    hall = dict(enabled=True, eta_h=0.5, eta_h_form="proportional_to_B_over_rho", method="inertia",
                length_mode=length_mode, length_ratio=4.0, b0=1/np.sqrt(4*np.pi),
                rho0=25/(36*np.pi), cg_tol=1e-12)
    filters = dict(spectral_dissipation=dict(enabled=True, order=2, onset_fraction=0.8,
                                             e_folding_time_at_cutoff=0.05))

    def make():
        g = Grid2D(Nx=n, Ny=n, Lx=1.0, Ly=1.0, backend="numpy", filter_params=filters)
        g.mhd_config = dict(resistivity=0.0, viscosity=0.0, hall=dict(hall))
        return g, SpectralSolverMHD2D(grid=g, equations=MHDEquations2D(gamma=5.0/3.0), cfl=0.25)

    g, s = make()
    U0 = orszag_tang_2d(*g.xy_mesh(), gamma=5.0/3.0)
    dt = s.compute_dt(U0)
    U = U0
    for _ in range(2*half):
        U = s.step(U, dt)
    assert s.hall.total["substeps"] > 0
    assert length_mode == "constant" or s.hall.total["cg_calls"] > 0
    # Hall is stateless: a fresh grid, solver and Hall object rebuilt from the same
    # config continue exactly from a saved snapshot.
    g, s = make()
    V = U0
    for _ in range(half):
        V = s.step(V, dt)
    path = save_solution_snapshot(str(tmp_path), half*dt, U=V, grid=g, equations=s.equations)
    restart = load_checkpoint(path)
    assert restart["t"] == half*dt
    g, s = make()
    V = restart["U"]
    for _ in range(half):
        V = s.step(V, dt)
    assert np.array_equal(V, U)


def test_hall_mri_restart_rebuilds_mode_metadata_and_history(tmp_path):
    def config():
        return dict(
            problem="hall_mri1d", grid=dict(N=16, Lx="auto", precision="double"),
            physics=dict(gamma=5.0/3.0, mhd=dict(enabled=True,
                                                 shearing_sheet=dict(omega=1.0, q=1.5))),
            integration=dict(t0=0.0, t_end=1.0, cfl=0.3, scheme="rk4"),
            initial_conditions=dict(rho0=1.0, B0=1.0, beta=100.0, mode=1, amplitude=1e-6),
            io=dict(outdir=str(tmp_path)), monitoring=dict(enabled=True))

    fresh = HallMRI1DProblem(config=config())
    g = fresh.create_grid("numpy")
    s = SpectralSolverMHD1D(grid=g, equations=fresh.create_equations(), cfl=0.3)
    U = s.step(fresh.create_initial_conditions(g), 0.1)
    snapshot = save_solution_snapshot(str(tmp_path), 0.1, U=U, grid=g, equations=s.equations)
    restarted = HallMRI1DProblem(config=config(), restart_from=snapshot)
    g2 = restarted.create_grid("numpy")  # restart never calls create_initial_conditions
    s2 = SpectralSolverMHD1D(grid=g2, equations=restarted.create_equations(), cfl=0.3)
    restarted.initialize_monitoring(s2, restarted.restart_data["U"])
    assert restarted.linear_mode == fresh.linear_mode
    (t, amplitude), = restarted.amplitude_history
    assert t == 0.1 and amplitude == pytest.approx(fresh._amplitude(U), rel=1e-14)


def test_adsi_shear_ideal_mri_and_hall_limits():
    s, _ = adsi_shear_mode(np.sqrt(15.0)/4.0, 0.0, 1.0, 1.5, 1.0, 0.0, p0=100.0)
    assert abs(s.real-0.75) < 1e-10 and abs(s.imag) < 1e-10
    # For k along B0 the transverse block is hall_mri_matrix (also with the inertia length).
    for eta, length in ((0.5, 0.0), (-0.25, 0.0), (0.5, 0.3)):
        k = 0.8
        hall = np.linalg.eigvals(hall_mri_matrix(k, 1.0, 1.5, 1.0, eta_h=eta, length=length))
        full = np.linalg.eigvals(adsi_shear_matrix(k, 0.0, 1.0, 1.5, 1.0, 0.0, p0=100.0,
                                                   eta_h=eta, length=length))
        assert max(min(abs(h-f) for f in full) for h in hall) < 1e-10


def test_adsi_shear_incompressible_quartic_and_destabilization_rule():
    # Kunz and Balbus (2004) quartic for B_R = 0 against the compressible matrix at huge sound speed.
    for omega, shear, bphi, eta, kv, kr in ((0.0, 1.0, 1.0, 2.0, 0.7, -1.4), (0.0, 1.0, 0.3, 3.0, 0.5, -3.0),
                                            (1.0, 1.5, 1.0, 1.0, 0.6, 0.3), (1.0, 1.5, 1.0, 1.0, 0.6, -0.3)):
        roots = adsi_incompressible_roots(kv, kr, omega, shear, 1.0, bphi, eta=eta)
        eigen = np.linalg.eigvals(adsi_shear_matrix(kv, kr, omega, shear, 1.0, bphi, p0=1e6, eta=eta))
        assert max(min(abs(r-e) for e in eigen) for r in roots) < 1e-4
    # Pure shear, Am = 1/(rho eta S) = 0.5: unstable iff S k_r k_v B_v B_phi < 0 and |k_r B_phi/(k_v B_v)| > Am.
    growth = lambda kr, bphi, eta: np.linalg.eigvals(
        adsi_shear_matrix(1.0, kr, 0.0, 1.0, 1.0, bphi, p0=1e6, eta=eta)).real.max()
    for bphi in (1.0, -1.0):
        assert 0.01 < growth(-2.0*bphi, bphi, 2.0) <= 0.5+1e-9
        assert growth(2.0*bphi, bphi, 2.0) < 1e-6
        assert growth(-0.25*bphi, bphi, 2.0) < 1e-6
        assert growth(-2.0*bphi, bphi, 0.0) < 1e-6
    # k along the vertical axis decouples (u_r, b_r) from the shear-driven rows: stable at any eta.
    assert max(growth(0.0, bphi, eta) for bphi in (0.3, 1.0, -1.0) for eta in (0.0, 0.5, 5.0)) < 1e-6


def test_adsi_shear_growth_never_exceeds_half_the_shear():
    rng = np.random.default_rng(7)
    for _ in range(60):
        omega, shear = rng.choice([0.0, 1.0]), rng.choice([1.0, 1.5, -0.7])
        s, _ = adsi_shear_mode(rng.uniform(0.2, 3.0), rng.uniform(-3.0, 3.0), omega, shear, 1.0,
                               rng.uniform(-2.0, 2.0), p0=rng.uniform(0.5, 50.0), eta=rng.uniform(0.0, 5.0),
                               eta_h=rng.uniform(-2.0, 2.0), length=rng.choice([0.0, 0.3]))
        assert s.real <= 0.5*abs(shear)+1e-9
    # Ideal pure shear is stable for oblique k as well.
    for kv, kr, bphi in ((0.7, 1.1, 0.8), (1.0, -2.0, 0.3), (0.4, 0.9, 1.5)):
        assert np.linalg.eigvals(adsi_shear_matrix(kv, kr, 0.0, 1.0, 1.0, bphi, p0=5.0)).real.max() < 1e-6


def test_adsi_shear_seed_roundtrip_and_box(tmp_path):
    problem = ADSIShearProblem(config=dict(
        problem="adsi_shear", grid=dict(Nx=16, Ny=16, Lx="auto", Ly="auto", precision="double"),
        physics=dict(gamma=5.0/3.0, mhd=dict(enabled=True, shearing_sheet=dict(omega=0.0, shear=1.0),
                                              ambipolar=dict(enabled=True, eta=2.0))),
        integration=dict(t0=0.0, t_end=1.0, cfl=0.3, scheme="rk4"),
        initial_conditions=dict(rho0=1.0, Bv=1.0, Bphi=1.0, beta=10.0, mode_v=1, mode_r=1,
                                amplitude=1e-6),
        io=dict(outdir=str(tmp_path))))
    g = problem.create_grid("numpy")
    mode = problem.linear_mode
    assert mode["growth"] > 0.01 and mode["k_r"]*mode["k_v"] < 0.0  # S B_v B_phi > 0 needs k_r k_v < 0
    assert g.Lx == pytest.approx(2.0*np.pi/mode["k_v"]) and g.Ly == pytest.approx(2.0*np.pi/abs(mode["k_r"]))
    U = problem.create_initial_conditions(g)
    assert problem.amplitude_history[0][1] == pytest.approx(1e-6*np.sqrt(2.0), rel=1e-6)
    rho, ux, uy, uz, p, bx, by, bz = problem.create_equations().primitive(U)
    c = adsi_shear_vector(np.array([rho, ux, uy, uz, bx, by, bz, p]), mode["k_v"], mode["k_r"],
                          problem._mode_v, problem._mode_r)
    np.testing.assert_allclose(c, 1e-6*np.sqrt(2.0)*problem._vector, rtol=1e-5, atol=1e-12)


def hall_cap_state(g):
    """1D state with a low-density spot at x = pi, where eta_H ~ |B|/rho is large."""
    B = np.array([1+0*g.x, 0.3*np.cos(g.x), 0.3*np.sin(g.x)])
    rho = 0.02+0.98*(1-np.exp(-(g.x-np.pi)**2))
    return B, rho, state(g, B, rho)


@pytest.mark.parametrize("form", ["constant", "proportional_to_B_over_rho"])
@pytest.mark.parametrize("eta", [0.2, -0.2])
@pytest.mark.parametrize("cap", [0.1, 1.0, 50.0])
def test_hall_eta_max_clamps_magnitude_and_keeps_sign(form, eta, cap):
    g = grid1d()
    B, rho, _ = hall_cap_state(g)
    cfg = dict(eta_h=eta, eta_h_form=form)
    free = Hall(g, cfg).eta_field(B, rho)
    capped = Hall(g, dict(cfg, eta_max=cap)).eta_field(B, rho)
    below = np.abs(free) < cap
    assert np.array_equal(capped[below], free[below])
    assert np.array_equal(capped, np.sign(free)*np.minimum(np.abs(free), cap))
    assert np.all(np.sign(capped) == np.sign(eta)) and np.max(np.abs(capped)) <= cap
    if form == "proportional_to_B_over_rho" and cap == 1.0:
        assert below.any() and not below.all()


@pytest.mark.parametrize("form", ["constant", "proportional_to_B_over_rho"])
def test_hall_eta_max_electric_field_uses_the_capped_coefficient(form):
    g = grid1d()
    B, rho, _ = hall_cap_state(g)
    cfg = dict(eta_h=-0.2, eta_h_form=form)
    free = Hall(g, cfg)
    # Without a cap the expressions are unchanged, including the analytic |B| cancellation.
    J = free.curl(B)
    cross = np.stack([J[1]*B[2]-J[2]*B[1], J[2]*B[0]-J[0]*B[2], J[0]*B[1]-J[1]*B[0]])
    modulus = np.sqrt((B*B).sum(axis=0))
    original = ((free.eta/np.maximum(modulus, 1e-30))*cross if form == "constant"
                else (free.eta*free.rho0/free.b0/np.maximum(rho, 1e-30))*cross)
    assert np.array_equal(free.electric_field(B, rho), original)
    for cap in (0.5, 50.0):
        hall = Hall(g, dict(cfg, eta_max=cap))
        expected = hall.eta_field(B, rho)*np.cross(hall.curl(B), B, axis=0)/modulus
        np.testing.assert_allclose(hall.electric_field(B, rho), expected, rtol=1e-13, atol=1e-15)
    # A cap above every |eta_H| reproduces the uncapped field.
    np.testing.assert_allclose(Hall(g, dict(cfg, eta_max=1e6)).electric_field(B, rho),
                               free.electric_field(B, rho), rtol=1e-13, atol=1e-15)


def test_hall_eta_max_reduces_substeps_and_automatic_length():
    g = grid1d()
    B, rho, U = hall_cap_state(g)
    cfg = dict(eta_h=0.2, eta_h_form="proportional_to_B_over_rho", method="reference")
    free, capped = Hall(g, cfg), Hall(g, dict(cfg, eta_max=0.5))
    free.step(U, None, 0.05)
    capped.step(U, None, 0.05)
    assert 0 < capped.last_stats["substeps"] < free.last_stats["substeps"]
    inertia = dict(cfg, method="inertia", safety=0.5)
    dt = 0.05
    l2_free = Hall(g, inertia).length_squared(U, dt)
    hall = Hall(g, dict(inertia, eta_max=0.5))
    l2_capped = hall.length_squared(U, dt)
    assert l2_capped < l2_free
    assert l2_capped == pytest.approx(max(0.0, 0.5*dt/0.5-1.0/hall.kmax2(B)))
    # Constant form: the rate uses min(|eta_h|, eta_max).
    const = Hall(g, dict(eta_h=2.0, method="reference", eta_max=0.5))
    const.step(U, None, 0.05)
    plain = Hall(g, dict(eta_h=0.5, method="reference"))
    plain.step(U, None, 0.05)
    assert const.last_stats["substeps"] == plain.last_stats["substeps"]


def test_hall_eta_max_absent_or_inactive_leaves_step_unchanged():
    g = grid1d()
    B, rho, U = hall_cap_state(g)
    cfg = dict(eta_h=0.2, eta_h_form="proportional_to_B_over_rho", method="inertia", length=0.1)
    expected = Hall(g, cfg).step(U, None, 0.05)
    assert np.array_equal(Hall(g, dict(cfg, eta_max=None)).step(U, None, 0.05), expected)
    np.testing.assert_allclose(Hall(g, dict(cfg, eta_max=1e6)).step(U, None, 0.05), expected,
                               rtol=1e-9, atol=1e-11)


@pytest.mark.parametrize("bad", [0.0, -1.0, np.nan, np.inf])
def test_hall_ambipolar_and_resistivity_caps_are_validated(bad):
    g = grid1d()
    with pytest.raises(ValueError, match="eta_max"):
        Hall(g, dict(eta_h=1.0, eta_max=bad))
    g2 = Grid2D(Nx=12, Ny=12, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
    with pytest.raises(ValueError, match="eta_max"):
        MidpointAmbipolar(g2, dict(eta=1.0, eta_max=bad))
    with pytest.raises(ValueError, match="resistivity_max"):
        solver(g, resistivity=1.0, resistivity_max=bad)


def ambipolar_state(g):
    X, Y = np.meshgrid(g.x, g.y)
    U = np.zeros((8, g.Ny, g.Nx))
    U[0] = 1
    U[5], U[6], U[7] = 1+0.3*np.sin(Y), 0.2*np.cos(X), 0.3*np.sin(X+Y)
    U[4] = 2.0+0.5*(U[5:8]**2).sum(axis=0)
    return U


def test_ambipolar_eta_max_below_everywhere_is_the_uniform_coefficient_operator():
    g = Grid2D(Nx=12, Ny=12, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
    X, Y = np.meshgrid(g.x, g.y)
    b = 1.2  # |reference| is uniform, so eta c |B|^2 = eta_max everywhere
    reference = b*np.array([np.cos(X), np.sin(X)*np.cos(Y), np.sin(X)*np.sin(Y)])
    cap, rng = 0.3, np.random.default_rng(5)
    B = rng.normal(size=(3, g.Ny, g.Nx))
    capped = MidpointAmbipolar(g, dict(eta=5.0, eta_max=cap))
    uniform = MidpointAmbipolar(g, dict(eta=cap/b**2))
    np.testing.assert_allclose(capped._operator(B, reference), uniform._operator(B, reference),
                               rtol=1e-10, atol=1e-10)
    V = rng.normal(size=B.shape)
    np.testing.assert_allclose(capped._precondition(reference, 0.2)(V),
                               uniform._precondition(reference, 0.2)(V), rtol=1e-10, atol=1e-10)


def test_ambipolar_eta_max_absent_or_inactive_leaves_step_unchanged():
    g = Grid2D(Nx=12, Ny=12, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
    U = ambipolar_state(g)
    cfg = dict(eta=0.4, midpoint_tol=1e-12)
    expected = MidpointAmbipolar(g, cfg).step(U, None, 0.05)
    assert np.array_equal(MidpointAmbipolar(g, dict(cfg, eta_max=None)).step(U, None, 0.05), expected)
    reference = U[5:8]
    big = MidpointAmbipolar(g, dict(cfg, eta_max=1e9))
    assert np.array_equal(big._operator(U[5:8]*0.9, reference),
                          MidpointAmbipolar(g, cfg)._operator(U[5:8]*0.9, reference))
    np.testing.assert_allclose(big.step(U, None, 0.05), expected, rtol=1e-9, atol=1e-10)


def test_ambipolar_eta_max_keeps_the_operator_spd_and_cg_convergent():
    mixed = Grid2D(Nx=12, Ny=12, Lx=2*np.pi, Ly=2*np.pi, basis_y="sbp", backend="numpy")
    rng = np.random.default_rng(3)
    a, b, reference = rng.normal(size=(3, 3, mixed.Ny, mixed.Nx))
    b2 = (reference*reference).sum(axis=0)
    ad = MidpointAmbipolar(mixed, dict(eta=1.0, eta_max=float(np.median(b2))))
    assert 0 < np.count_nonzero(ad._coupling(b2) < 1) < b2.size
    La, Lb = ad._operator(a, reference), ad._operator(b, reference)
    w = mixed.wy[:, None]
    np.testing.assert_allclose((a*Lb*w).sum(), (b*La*w).sum(), atol=2e-11)
    assert (a*La*w).sum() <= 0
    precondition = ad._precondition(reference, 0.2)
    np.testing.assert_allclose((a*precondition(b)).sum(), (b*precondition(a)).sum(), atol=1e-12)
    assert (a*precondition(a)).sum() > 0
    g = Grid2D(Nx=12, Ny=12, Lx=2*np.pi, Ly=2*np.pi, backend="numpy")
    U = ambipolar_state(g)
    cfg = dict(eta=2.0, midpoint_tol=1e-12)
    free, limited = MidpointAmbipolar(g, cfg), MidpointAmbipolar(g, dict(cfg, eta_max=0.8))
    V0, V1 = free.step(U, None, 0.05), limited.step(U, None, 0.05)
    assert limited.last_residual <= 1e-12 and limited.total_solves == 2
    magnetic = lambda V: (V[5:8]**2).sum()
    assert magnetic(V1) < magnetic(U) and magnetic(V0) < magnetic(V1)


def test_resistivity_max_clamps_rhs_and_timestep():
    g = grid1d()
    U = state(g, [1+0*g.x, 0.3*np.cos(g.x), 0.3*np.sin(g.x)])
    plain = solver(g, resistivity=50.0)
    capped = solver(g, resistivity=50.0, resistivity_max=20.0)
    reference = solver(g, resistivity=20.0)
    inactive = solver(g, resistivity=0.1, resistivity_max=20.0)
    assert capped.resistivity == 20.0 and inactive.resistivity == 0.1 and plain.resistivity == 50.0
    assert np.array_equal(capped._compute_rhs(U), reference._compute_rhs(U))
    assert not np.array_equal(capped._compute_rhs(U), plain._compute_rhs(U))
    assert capped.compute_dt(U) == reference.compute_dt(U) > plain.compute_dt(U)
