import numpy as np

from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.solver import SpectralSolver1D


def test_fft_derivative_sine():
    N = 256
    Lx = 2.0 * np.pi
    grid = Grid1D(N=N, Lx=Lx, dealias=False)
    f = np.sin(grid.x)
    df = grid.dx1(f)
    err = np.linalg.norm(df - np.cos(grid.x), ord=np.inf)
    assert err < 1e-6


def test_conservation_short_run():
    N = 256
    Lx = 1.0
    grid = Grid1D(N=N, Lx=Lx, dealias=True)
    eqs = EulerEquations1D(gamma=1.4)
    # Smooth initial condition to avoid strong shocks in very short run
    rho = 1.0 + 1e-3 * np.sin(2.0 * np.pi * grid.x / Lx)
    u = np.zeros_like(grid.x)
    p = np.ones_like(grid.x)
    U0 = eqs.conservative(rho, u, p)

    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme="rk4", cfl=0.4)
    history = solver.run(U0, t0=0.0, t_end=0.01, output_interval=0.01)

    # Check near conservation over short time
    m0, e0 = history["mass"][0], history["energy"][0]
    m1, e1 = history["mass"][-1], history["energy"][-1]
    assert abs(m1 - m0) / m0 < 1e-8
    assert abs(e1 - e0) / e0 < 1e-6


