import numpy as np

from phrike.grid import Grid3D
from phrike.equations import EulerEquations3D
from phrike.initial_conditions import taylor_green_vortex_3d
from phrike.solver import SpectralSolver3D


def test_grid3d_dx_matches_analytic():
    Nx, Ny, Nz = 32, 16, 8
    Lx = 2.0 * np.pi
    Ly = 1.0
    Lz = 1.0
    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dealias=True)

    # Build function f(x) = sin(kx * x), broadcast across y,z
    kx = 3
    x = grid.x  # 1D
    f1d = np.sin(kx * (2.0 * np.pi) * x / Lx)
    # Target shape (Nz, Ny, Nx)
    f = np.tile(f1d, (Nz, Ny, 1))
    dfdx_true_1d = (kx * (2.0 * np.pi) / Lx) * np.cos(kx * (2.0 * np.pi) * x / Lx)
    dfdx_true = np.tile(dfdx_true_1d, (Nz, Ny, 1))

    dfdx_spec = grid.dx1(f)
    err = np.max(np.abs(dfdx_spec - dfdx_true))
    assert err < 1e-8


def test_solver3d_runs_cpu():
    Nx, Ny, Nz = 16, 16, 8
    L = 2.0 * np.pi
    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=L, Ly=L, Lz=L, dealias=True, filter_params={"enabled": True, "p": 8, "alpha": 36.0})
    eqs = EulerEquations3D(gamma=1.4)

    X, Y, Z = grid.xyz_mesh()
    U0 = taylor_green_vortex_3d(X, Y, Z, rho0=1.0, p0=100.0, U0=1.0, k=1, gamma=1.4)
    solver = SpectralSolver3D(grid=grid, equations=eqs, scheme="rk4", cfl=0.25)
    history = solver.run(U0, t0=0.0, t_end=0.05, output_interval=0.05)

    assert solver.U is not None
    rho, ux, uy, uz, p = eqs.primitive(solver.U)
    assert np.isfinite(rho).all()
    assert np.isfinite(ux).all()
    assert np.isfinite(uy).all()
    assert np.isfinite(uz).all()
    assert np.isfinite(p).all()


