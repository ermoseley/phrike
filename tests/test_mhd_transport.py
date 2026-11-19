"""Physics of the compressible visco-resistive MHD transport terms (``phrike.transport``).

Backend parity is tested elsewhere; these cases check the operators themselves:
Galilean invariance, the decay rates of shear and longitudinal modes, local
viscous and Ohmic heating, the discrete energy closure, exact decaying
solutions, conduction, the shearing-sheet heating, closed polynomial and mixed
boxes, and the diffusive time-step bounds.
"""

import numpy as np
import pytest

from phrike.equations import MHDEquations1D, MHDEquations2D, MHDEquations3D
from phrike.grid import Grid1D, Grid2D, Grid3D
from phrike.solver import SpectralSolverMHD1D, SpectralSolverMHD2D, SpectralSolverMHD3D
from phrike.transport import Transport, heating, strain_from_gradient

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

TWO_PI = 2.0 * np.pi
GAMMA = 5.0 / 3.0
FULL = Transport(nu=0.03, nu_bulk=0.02, eta=0.05, kappa=0.04)


def _solver1d(transport, N=16, basis="fourier", dealias=True, **cfg):
    periodic = basis == "fourier"
    grid = Grid1D(N=N, Lx=TWO_PI if periodic else 1.0, basis=basis, bc="periodic" if periodic else "wall",
                  dealias=dealias if periodic else False, backend="numpy")
    grid.mhd_config = dict(transport.as_dict(), **cfg)
    return SpectralSolverMHD1D(grid, MHDEquations1D(gamma=GAMMA))


def _solver3d(transport, shape=(16, 12, 10), backend="numpy", fast=False):
    Nx, Ny, Nz = shape
    kwargs = dict(backend=backend, torch_device="cpu", precision="double") if backend == "torch" else {
        "backend": "numpy"}
    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=TWO_PI, Ly=TWO_PI, Lz=TWO_PI, dealias=True, **kwargs)
    grid.mhd_config = dict(transport.as_dict(), divergence_clean="projection")
    return SpectralSolverMHD3D(grid, MHDEquations3D(gamma=GAMMA), spectral_fast_path=fast)


def _rhs(solver, U):
    return np.asarray(solver._transport_rhs(U, solver.transport))


def _band_limited(grid, rng, n):
    """``n`` random smooth fields with unit maximum (the retained band only)."""
    shape = (n,) + tuple(np.asarray(grid.x if not hasattr(grid, "xyz_mesh") else grid.xyz_mesh()[0]).shape)
    f = np.asarray(grid.project_dealiased(rng.normal(size=shape)))
    return f / np.abs(f).max(axis=tuple(range(1, f.ndim)), keepdims=True)


def _random_state_3d(solver, seed=0):
    g, eqs = solver.grid, solver.equations
    f = _band_limited(g, np.random.default_rng(seed), 8)
    rho = 1.0 + 0.4 * f[0]
    u = 0.3 * f[1:4]
    p = 1.0 + 0.3 * f[4]
    B = 0.5 * f[5:8]
    return eqs.conservative(rho, *u, p, *B)


# --- Galilean invariance -------------------------------------------------------

def test_uniform_velocity_has_no_transport_terms_at_variable_density():
    """The old nu lap(rho u) momentum term gives nu V lap(rho) here."""
    solver = _solver3d(FULL)
    X, Y, Z = (np.asarray(a) for a in solver.grid.xyz_mesh())
    rho = 1.0 + 0.3 * np.cos(X) * np.cos(2 * Y) * np.sin(Z)
    one = np.ones_like(rho)
    U = solver.equations.conservative(rho, 0.7 * one, -0.4 * one, 0.25 * one, 0.8 * rho,
                                      0.2 * one, 0.1 * one, -0.3 * one)  # T = p/rho uniform
    assert np.abs(_rhs(solver, U)).max() < 1e-12


@pytest.mark.parametrize("basis", ["chebyshev", "legendre"])
def test_uniform_velocity_has_no_transport_terms_on_polynomial_grids(basis):
    solver = _solver1d(FULL, N=17, basis=basis)
    x = np.asarray(solver.grid.x)
    rho = 1.0 + 0.3 * np.cos(3.0 * x)
    one = np.ones_like(x)
    U = solver.equations.conservative(rho, 0.7 * one, -0.4 * one, 0.25 * one, 0.8 * rho,
                                      0.2 * one, 0.1 * one, -0.3 * one)
    assert np.abs(_rhs(solver, U)).max() < 1e-9


def test_boost_leaves_the_momentum_rhs_and_shifts_the_energy_rhs():
    solver = _solver3d(FULL)
    U = _random_state_3d(solver)
    V = np.array([0.3, -0.2, 0.5])
    boosted = U.copy()
    boosted[1:4] += U[0] * V[:, None, None, None]
    boosted[4] += (U[1:4] * V[:, None, None, None]).sum(0) + 0.5 * U[0] * (V ** 2).sum()
    R, Rb = _rhs(solver, U), _rhs(solver, boosted)
    scale = np.abs(R).max()
    assert np.abs(Rb[1:4] - R[1:4]).max() < 1e-11 * scale
    assert np.abs(Rb[5:8] - R[5:8]).max() < 1e-11 * scale
    assert np.abs(Rb[4] - R[4] - (V[:, None, None, None] * R[1:4]).sum(0)).max() < 1e-11 * scale


# --- decay rates and local heating -----------------------------------------------

@pytest.mark.parametrize("longitudinal", [False, True])
def test_oblique_mode_decay_rates(longitudinal):
    tr = Transport(nu=0.03, nu_bulk=0.02)
    solver = _solver3d(tr)
    X, Y, Z = (np.asarray(a) for a in solver.grid.xyz_mesh())
    k = np.array([1.0, 2.0, -1.0])
    e = k / np.linalg.norm(k) if longitudinal else np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    phase = np.sin(k[0] * X + k[1] * Y + k[2] * Z)
    one = np.ones_like(X)
    u = [0.1 * e[i] * phase for i in range(3)]
    U = solver.equations.conservative(one, *u, one, 0 * one, 0 * one, 0 * one)
    rate = (4.0 * tr.nu / 3.0 + tr.nu_bulk if longitudinal else tr.nu) * float(k @ k)
    R = _rhs(solver, U)
    assert np.abs(R[1:4] + rate * np.stack(u)).max() < 1e-12


def test_ohmic_heating_is_local():
    """B_y = sin x: R_By = -eta sin x, R_E = eta cos 2x, internal heating eta cos^2 x."""
    eta = 0.05
    solver = _solver1d(Transport(eta=eta))
    x = np.asarray(solver.grid.x)
    one, zero = np.ones_like(x), np.zeros_like(x)
    U = solver.equations.conservative(one, zero, zero, zero, one, zero, np.sin(x), zero)
    R = _rhs(solver, U)
    assert np.abs(R[6] + eta * np.sin(x)).max() < 1e-13
    assert np.abs(R[4] - eta * np.cos(2 * x)).max() < 1e-13
    assert np.abs(R[4] - U[6] * R[6] - eta * np.cos(x) ** 2).max() < 1e-13


def test_viscous_heating_is_local():
    """u_y = A sin x at unit density: R_my = -nu A sin x and the heating is nu A^2 cos^2 x."""
    nu, A = 0.03, 0.2
    solver = _solver1d(Transport(nu=nu, nu_bulk=0.02))
    x = np.asarray(solver.grid.x)
    one, zero = np.ones_like(x), np.zeros_like(x)
    U = solver.equations.conservative(one, zero, A * np.sin(x), zero, one, zero, zero, zero)
    R = _rhs(solver, U)
    assert np.abs(R[2] + nu * A * np.sin(x)).max() < 1e-13
    assert np.abs(R[4] - (A * np.sin(x)) * R[2] - nu * A * A * np.cos(x) ** 2).max() < 1e-13


def test_energy_closure_on_a_variable_density_state():
    """Total energy is conserved and sum(R_E - u.R_m - B.R_B) is the volume heating."""
    solver = _solver3d(FULL)
    g = solver.grid
    U = _random_state_3d(solver, seed=1)
    R = _rhs(solver, U)
    rho = U[0]
    u = [U[1] / rho, U[2] / rho, U[3] / rho]
    B = [U[5], U[6], U[7]]
    d = (g.dx1, g.dy1, g.dz1)
    G = [[np.asarray(d[a](u[b])) for b in range(3)] for a in range(3)]
    D, theta = strain_from_gradient(G)
    J = [np.asarray(d[1](B[2]) - d[2](B[1])), np.asarray(d[2](B[0]) - d[0](B[2])),
         np.asarray(d[0](B[1]) - d[1](B[0]))]
    viscous, ohmic = heating(rho, FULL, D=D, theta=theta, J=J)
    total = float(np.sum(viscous + ohmic))
    work = float(np.sum(R[4] - sum(u[i] * R[1 + i] for i in range(3)) - sum(B[i] * R[5 + i] for i in range(3))))
    assert abs(float(np.sum(R[4]))) < 1e-12 * total
    assert np.abs(R[1:4].sum(axis=(1, 2, 3))).max() < 1e-12 * total
    assert total > 0.0 and work == pytest.approx(total, rel=1e-11)


# --- conduction and the shearing sheet --------------------------------------------

def test_conduction_is_kappa_laplacian_of_temperature():
    kappa = 0.04
    solver = _solver1d(Transport(kappa=kappa))
    x = np.asarray(solver.grid.x)
    one, zero = np.ones_like(x), np.zeros_like(x)
    U = solver.equations.conservative(2.0 * one, zero, zero, zero, 2.0 * (1.0 + 0.1 * np.cos(2 * x)),
                                      zero, zero, zero)
    R = _rhs(solver, U)
    assert np.abs(R[4] + kappa * 4.0 * 0.1 * np.cos(2 * x)).max() < 1e-13
    assert np.abs(R[:4]).max() < 1e-14 and np.abs(R[5:]).max() < 1e-14


def test_conduction_damps_the_checkerboard_without_dealiasing():
    """Two real first derivatives drop the Nyquist mode; the spectral Laplacian does not."""
    kappa = 0.04
    solver = _solver1d(Transport(kappa=kappa), N=8, dealias=False)
    one, zero = np.ones(8), np.zeros(8)
    T = 1.0 + 0.1 * (-1.0) ** np.arange(8)
    U = solver.equations.conservative(one, zero, zero, zero, T, zero, zero, zero)
    R = _rhs(solver, U)
    assert np.abs(R[4] + kappa * 16.0 * (T - 1.0)).max() < 1e-12


def test_shearing_sheet_viscous_heating_of_the_background_flow():
    nu, omega, q = 0.01, 1.0, 1.5
    solver = _solver1d(Transport(nu=nu), shearing_sheet={"omega": omega, "q": q})
    one, zero = np.ones(16), np.zeros(16)
    U = solver.equations.conservative(one, zero, zero, zero, one, zero, zero, zero)
    R = np.asarray(solver._compute_rhs(U))
    assert np.abs(R[4] - nu * (q * omega) ** 2).max() < 1e-14
    assert np.abs(np.delete(R, 4, axis=0)).max() < 1e-14


# --- exact decaying solutions -------------------------------------------------------

def _circular_modes(X, Y, Z, U0=0.2, B0=0.3, p0=1.0):
    """Parallel circular velocity and magnetic modes along z: force-free and advection-free."""
    one = np.ones_like(Z) if isinstance(Z, np.ndarray) else torch.ones_like(Z)
    cos, sin = (np.cos, np.sin) if isinstance(Z, np.ndarray) else (torch.cos, torch.sin)
    return [one, U0 * cos(Z), U0 * sin(Z), 0 * one, p0 * one, B0 * cos(Z), B0 * sin(Z), 0 * one]


def _circular_exact(t, tr, U0=0.2, B0=0.3, p0=1.0):
    fu, fb = np.exp(-tr.nu * t), np.exp(-tr.eta * t)
    p = p0 + (GAMMA - 1.0) * (0.5 * U0 ** 2 * (1.0 - fu * fu) + 0.5 * B0 ** 2 * (1.0 - fb * fb))
    return U0 * fu, B0 * fb, p


@pytest.mark.parametrize("backend,fast", [("numpy", False)] + ([("torch", True)] if torch is not None else []))
def test_circular_modes_decay_exactly(backend, fast):
    solver = _solver3d(FULL, shape=(8, 8, 16), backend=backend, fast=fast)
    eqs = solver.equations
    X, Y, Z = solver.grid.xyz_mesh()
    U = eqs.conservative(*_circular_modes(X, Y, Z))
    if fast:
        assert solver.spectral is not None
    U = solver.grid.project_dealiased(solver._project_B(U))
    dt, n = 0.01, 50
    for _ in range(n):
        U = solver.step(U, dt)
    Un = U.detach().cpu().numpy() if hasattr(U, "detach") else np.asarray(U)
    u_amp, b_amp, p = _circular_exact(n * dt, FULL)
    Z = np.asarray(Z.detach().cpu() if hasattr(Z, "detach") else Z)
    rho, ux, uy, uz, pn, Bx, By, Bz = eqs.primitive(Un)
    assert np.abs(rho - 1.0).max() < 1e-12
    assert np.abs(ux - u_amp * np.cos(Z)).max() < 1e-10 and np.abs(uy - u_amp * np.sin(Z)).max() < 1e-10
    assert np.abs(Bx - b_amp * np.cos(Z)).max() < 1e-10 and np.abs(By - b_amp * np.sin(Z)).max() < 1e-10
    assert np.abs(pn - p).max() < 1e-10


# --- closed boxes ---------------------------------------------------------------------

@pytest.mark.parametrize("basis", ["chebyshev", "legendre"])
def test_polynomial_box_conserves_energy_and_dissipates(basis):
    """Any positive state, however rough: no flux leaves, and the heating is the integrated work."""
    solver = _solver1d(FULL, N=17, basis=basis)
    eqs = solver.equations
    rng = np.random.default_rng(7)
    rho = 0.2 + rng.uniform(size=17)
    u = rng.normal(size=(3, 17))
    B = np.vstack([0.4 * np.ones(17), rng.normal(size=(2, 17))])
    p = 0.5 + rng.uniform(size=17)
    U = eqs.conservative(rho, *u, p, *B)
    R = _rhs(solver, U)
    w = np.asarray(solver.grid.wx)
    ops = solver._poly_ops
    Du, DB = ops.grad(u), ops.grad(B)
    kinetic = float(np.sum(w * (u * R[1:4]).sum(0)))
    magnetic = float(np.sum(w * (B * R[5:8]).sum(0)))
    viscous = FULL.nu * rho * (4.0 / 3.0 * Du[0] ** 2 + Du[1] ** 2 + Du[2] ** 2) + FULL.nu_bulk * rho * Du[0] ** 2
    ohmic = FULL.eta * (DB[1] ** 2 + DB[2] ** 2)
    assert abs(float(np.sum(w * R[4]))) < 1e-12 * float(np.sum(w * np.abs(R[4])))
    assert np.abs((w * R[1:4]).sum(1)).max() < 1e-12 * float(np.sum(w * np.abs(R[1:4])))
    assert np.abs(R[5]).max() == 0.0
    assert kinetic == pytest.approx(-float(np.sum(w * viscous)), rel=1e-10) and kinetic < 0.0
    assert magnetic == pytest.approx(-float(np.sum(w * ohmic)), rel=1e-10) and magnetic < 0.0


@pytest.mark.parametrize("basis,sizes,fine", [("chebyshev", (9, 17, 33), 1e-6), ("legendre", (7, 9, 13), 1e-9)])
def test_polynomial_viscous_operator_converges(basis, sizes, fine):
    """A smooth state with no flux through the walls: spectral convergence of R_m."""
    nu = 0.03
    errors = []
    for N in sizes + (65,):
        solver = _solver1d(Transport(nu=nu), N=N, basis=basis)
        x = np.asarray(solver.grid.x)
        rho = 1.0 + 0.2 * np.cos(np.pi * x)
        uy, duy = np.cos(np.pi * x), -np.pi * np.sin(np.pi * x)
        exact = nu * (-0.2 * np.pi * np.sin(np.pi * x) * duy + rho * (-np.pi ** 2) * np.cos(np.pi * x))
        zero = np.zeros_like(x)
        U = solver.equations.conservative(rho, zero, uy, zero, rho, zero, zero, zero)
        errors.append(np.abs(_rhs(solver, U)[2] - exact).max())
    assert errors[1] < 0.1 * errors[0] and errors[2] < 0.1 * errors[1]  # before round-off dominates
    assert errors[3] < fine


def test_mixed_grid_viscous_work_is_dissipative():
    grid = Grid2D(Nx=12, Ny=13, Lx=TWO_PI, Ly=TWO_PI, basis_y="sbp", backend="numpy")
    grid.mhd_config = FULL.as_dict()
    solver = SpectralSolverMHD2D(grid, MHDEquations2D(gamma=GAMMA))
    rng = np.random.default_rng(5)
    shape = (grid.Ny, grid.Nx)
    rho = 0.3 + rng.uniform(size=shape)
    m = grid.project_dealiased(rng.normal(size=(3,) + shape))  # retained momentum, as the stepper keeps it
    B = grid.project_dealiased(0.3 * rng.normal(size=(3,) + shape))
    p = 0.5 + rng.uniform(size=shape)
    U = solver.equations.conservative(rho, *(m / rho), p, *B)
    R = _rhs(solver, U)
    w = np.asarray(grid.wy)[:, None]
    u = U[1:4] / U[0]
    assert float(np.sum(w * (u * R[1:4]).sum(0))) < 0.0
    assert abs(float(np.sum(w * R[4]))) < 1e-12 * float(np.sum(w * np.abs(R[4])))
    assert np.abs((w * R[1:4]).sum(axis=(1, 2))).max() < 1e-12 * float(np.sum(w * np.abs(R[1:4])))


# --- time-step bounds --------------------------------------------------------------------

def _momentum_operator(solver, rho, p):
    """d R_m / d m at frozen density (3N x 3N): the viscous operator the step must keep stable."""
    eqs = solver.equations
    N = rho.size
    zero = np.zeros(N)
    base = eqs.conservative(rho, zero, zero, zero, p, zero, zero, zero)
    A = np.empty((3 * N, 3 * N))
    for c in range(3):
        for j in range(N):
            U = base.copy()
            U[1 + c, j] = 1.0
            A[:, c * N + j] = _rhs(solver, U)[1:4].reshape(-1)
    return A


def test_viscous_step_bound_holds_at_large_density_contrast():
    nu = 1.0
    solver = _solver1d(Transport(nu=nu), N=48)
    x = np.asarray(solver.grid.x)
    rho, p = 1.0 + 0.999 * np.cos(x), 0.01 * np.ones(48)
    fastest = np.abs(np.linalg.eigvals(_momentum_operator(solver, rho, p))).max()
    k2 = solver._diffusive_k2()
    assert k2 == pytest.approx(15.0 ** 2)  # (48 - 1) // 3 on a 2 pi box
    assert fastest > 4.0 / 3.0 * nu * k2  # the uniform-density bound would be unstable
    zero = np.zeros(48)
    U = solver.equations.conservative(rho, zero, zero, zero, p, zero, zero, zero)
    assert solver._stability_limit(U) * fastest <= solver.cfl_diff * (1.0 + 1e-12)
    uniform = _solver1d(Transport(nu=nu), N=48, viscous_step_bound="uniform")
    assert uniform._stability_limit(U) == pytest.approx(solver._stability_limit(U) * rho.max() / rho.min())


@pytest.mark.parametrize("basis", ["chebyshev", "legendre"])
def test_polynomial_step_bound_is_the_operator_spectral_radius(basis):
    eta = 0.05
    solver = _solver1d(Transport(eta=eta), N=17, basis=basis)
    eqs = solver.equations
    one, zero = np.ones(17), np.zeros(17)
    A = np.empty((17, 17))
    for j in range(17):
        By = zero.copy()
        By[j] = 1.0
        A[:, j] = _rhs(solver, eqs.conservative(one, zero, zero, zero, one, zero, By, zero))[6]
    fastest = np.abs(np.linalg.eigvals(A)).max()
    assert eta * solver._diffusive_k2() == pytest.approx(fastest, rel=1e-8)


# --- configuration ------------------------------------------------------------------------

@pytest.mark.parametrize("cfg", [{"resistivity": float("inf"), "resistivity_max": 1.0},
                                 {"viscosity": -1.0}, {"thermal_conductivity": float("nan")},
                                 {"bulk_viscosity": float("inf")}, {"resistivity": 1.0, "resistivity_max": 0.0}])
def test_invalid_coefficients_are_rejected(cfg):
    with pytest.raises(ValueError):
        Transport.from_config(cfg)


def test_resistivity_cap_and_step_bound_option():
    assert Transport.from_config({"resistivity": 3.0, "resistivity_max": 1.0}).eta == 1.0
    with pytest.raises(ValueError, match="viscous_step_bound"):
        _solver1d(Transport(nu=0.1), viscous_step_bound="bogus")


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_euler_slab_stepper_rejects_transport():
    from phrike.parallel import Comm
    from phrike.parallel.compressible import CompressibleSlabStepper
    from phrike.parallel.grid import SlabGrid3D

    grid = SlabGrid3D(8, 8, 8, TWO_PI, TWO_PI, TWO_PI, comm=Comm(None), device="cpu", precision="double")
    with pytest.raises(ValueError):
        CompressibleSlabStepper(grid, "euler", 1.4, 1e-10, 1e-12, transport=Transport(nu=0.1))


# --- review cases: mixed grids, single precision, Nyquist, warm-up, history -----------------

def test_mixed_grid_viscous_step_bound():
    """The SBP closure breaks the longitudinal 4/3: the mixed-grid bound must use 2 nu + nu_b."""
    nu = 1.0
    grid = Grid2D(Nx=24, Ny=17, Lx=0.5, Ly=1.0, basis_y="sbp", backend="numpy")
    grid.mhd_config = {"viscosity": nu}
    solver = SpectralSolverMHD2D(grid, MHDEquations2D(gamma=GAMMA))
    shape = (grid.Ny, grid.Nx)
    n = grid.Ny * grid.Nx
    one, zero = np.ones(shape), np.zeros(shape)
    base = solver.equations.conservative(one, zero, zero, zero, one, zero, zero, zero)
    A = np.empty((3 * n, 3 * n))
    for c in range(3):
        for j in range(n):
            U = base.copy()
            U[1 + c].flat[j] = 1.0
            A[:, c * n + j] = _rhs(solver, U)[1:4].reshape(-1)
    fastest = np.abs(np.linalg.eigvals(A)).max()
    k2 = solver._diffusive_k2()
    assert fastest > 4.0 / 3.0 * nu * k2  # the periodic bound would be unsafe here
    assert solver._stability_limit(base) * fastest <= solver.cfl_diff * (1.0 + 1e-12)


def test_mixed_grid_resistivity_damps_face_currents():
    """A field whose SBP derivative vanishes inside but not at the faces still decays."""
    eta = 1.0
    grid = Grid2D(Nx=12, Ny=9, Lx=1.0, Ly=1.0, basis_y="sbp", backend="numpy")
    grid.mhd_config = {"resistivity": eta}
    solver = SpectralSolverMHD2D(grid, MHDEquations2D(gamma=GAMMA))
    shape = (grid.Ny, grid.Nx)
    Bx = np.repeat(np.array([1, 1 / 59, 1, 0, 7 / 8, 0, 1, 1 / 59, 1])[:, None], grid.Nx, axis=1)
    one, zero = np.ones(shape), np.zeros(shape)
    U = solver.equations.conservative(one, zero, zero, zero, 2.0 - 0.5 * Bx ** 2, Bx, zero, zero)
    R = _rhs(solver, U)
    w = np.asarray(grid.wy)[:, None]
    Jz = -np.asarray(grid.dy1(Bx))
    magnetic = float(np.sum(w * Bx * R[5]))
    assert magnetic == pytest.approx(-eta * float(np.sum(w * Jz ** 2)), rel=1e-12) and magnetic < -1.0
    assert abs(float(np.sum(w * R[4]))) < 1e-12 * float(np.sum(w * np.abs(R[4])))


def test_tiny_resistivity_on_a_mixed_grid_stays_finite():
    grid = Grid2D(Nx=12, Ny=9, Lx=1.0, Ly=1.0, basis_y="sbp", backend="numpy")
    grid.mhd_config = {"resistivity": 1e-300}
    solver = SpectralSolverMHD2D(grid, MHDEquations2D(gamma=GAMMA))
    shape = (grid.Ny, grid.Nx)
    y = np.asarray(grid.y)[:, None] * np.ones(shape)
    one, zero = np.ones(shape), np.zeros(shape)
    U = solver.equations.conservative(one, zero, zero, zero, one, 0.1 * np.cos(np.pi * y), zero, zero)
    assert np.isfinite(_rhs(solver, U)).all()


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_single_precision_polynomial_grid_keeps_a_constant_state():
    grid = Grid1D(N=128, Lx=1.0, basis="chebyshev", bc="wall", dealias=False, backend="torch",
                  torch_device="cpu", precision="single")
    grid.mhd_config = FULL.as_dict()
    solver = SpectralSolverMHD1D(grid, MHDEquations1D(gamma=GAMMA))
    one = torch.ones(128, dtype=torch.float32)
    U = solver.equations.conservative(one, 0.3 * one, -0.2 * one, 0.1 * one, one, 0.4 * one, 0.2 * one, 0 * one)
    assert float(solver._transport_rhs(U, solver.transport).abs().max()) == 0.0


def test_unmasked_even_fourier_axis_warns_about_the_nyquist_mode():
    solver = _solver1d(Transport(nu=0.1), N=16, dealias=False)
    x = np.asarray(solver.grid.x)
    one, zero = np.ones_like(x), np.zeros_like(x)
    U = solver.equations.conservative(one, zero, 0.1 * np.sin(x), zero, one, zero, zero, zero)
    with pytest.warns(RuntimeWarning, match="Nyquist"):
        solver._compute_rhs(U)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("transport,L", [(Transport(nu=0.01, nu_bulk=0.01, eta=0.01), TWO_PI),
                                         (Transport(nu=0.01, nu_bulk=0.01, eta=0.01, kappa=0.01), TWO_PI),
                                         (Transport(kappa=1000.0), 1.0)])
def test_warm_up_keeps_the_triton_tier_with_transport(transport, L):
    """Nine derivative rows go back through the scratch transforms, and a strong
    diffusivity does not make the warm-up state need the limiter."""
    from _triton_emulator import installed
    from phrike.parallel import Comm
    from phrike.parallel.compressible import CompressibleSlabStepper
    from phrike.parallel.grid import SlabGrid3D

    with installed():
        grid = SlabGrid3D(12, 12, 12, L, L, L, comm=Comm(None), device="cpu", precision="single")
        st = CompressibleSlabStepper(grid, "mhd", GAMMA, 1e-10, 1e-12, transport=transport, triton="force")
        assert st.warm_up(embedded=True)


@pytest.mark.skipif(torch is None, reason="phrike.parallel needs torch")
def test_appended_history_gains_new_columns(tmp_path):
    from phrike.parallel.io import History

    path = str(tmp_path / "history.csv")
    old = History(path, True)
    old.write({"t": 0.0, "energy": 1.0})
    old.write({"t": 1.0, "energy": 0.9})
    resumed = History(path, True, append=True)
    resumed.write({"t": 2.0, "energy": 0.8, "viscous_dissipation": 0.05})
    lines = open(path).read().splitlines()
    assert lines[0] == "t,energy,viscous_dissipation"
    assert lines[1].endswith(",nan") and lines[2].endswith(",nan") and lines[3].endswith("0.05")


def test_mixed_grid_resistive_step_keeps_div_b_at_round_off():
    """The weak curl leaves div B in the face rows of the RHS; the stage projection removes it."""
    grid = Grid2D(Nx=16, Ny=17, Lx=1.0, Ly=1.0, basis_y="sbp", backend="numpy")
    grid.mhd_config = {"resistivity": 0.05, "viscosity": 0.02}
    solver = SpectralSolverMHD2D(grid, MHDEquations2D(gamma=GAMMA))
    shape = (grid.Ny, grid.Nx)
    x = np.asarray(grid.x)[None, :] * np.ones(shape)
    y = np.asarray(grid.y)[:, None] * np.ones(shape)
    one, zero = np.ones(shape), np.zeros(shape)
    Bx = 0.2 + 0.1 * np.cos(2 * np.pi * x) * np.cos(np.pi * y)
    By = 0.1 * np.sin(2 * np.pi * x) * np.sin(np.pi * y) * 2.0 / 1.0 * 0.5
    U = solver.equations.conservative(one, 0.05 * np.sin(2 * np.pi * x), zero, zero, one, Bx, By, 0.1 * one)
    U = solver._project_B(U)
    start = solver.max_div_b(U)
    for _ in range(3):
        U = solver.step(U, 1e-3)
    assert np.isfinite(U).all() and solver.max_div_b(U) < max(1e-10, 10 * start)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_multi_rank_kernel_failures_abort_but_a_diverging_state_raises():
    """A non-finite state is judged on reduced statistics (the same on every rank) and raises;
    a kernel failure on one rank of several aborts the job instead of desynchronizing it."""
    from phrike.parallel import Comm
    from phrike.parallel.compressible import CompressibleSlabStepper
    from phrike.parallel.grid import SlabGrid3D

    grid = SlabGrid3D(8, 8, 8, TWO_PI, TWO_PI, TWO_PI, comm=Comm(None), device="cpu", precision="double")
    st = CompressibleSlabStepper(grid, "mhd", GAMMA, 1e-10, 1e-12, triton=False)

    class TwoRanks:
        size, rank, is_root = 2, 1, False

        def abort(self, code=1):
            raise SystemExit(code)

    st.comm = TwoRanks()
    with pytest.raises(FloatingPointError):
        st._disable_triton(FloatingPointError("Non-finite solution after a step"))
    with pytest.raises(SystemExit):
        st._disable_triton(RuntimeError("kernel failed"))
