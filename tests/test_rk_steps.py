import numpy as np

from spectralhydro.grid import Grid1D
from spectralhydro.equations import EulerEquations1D
from spectralhydro.solver import SpectralSolver1D


def test_rk_advances_time():
    N = 64
    Lx = 1.0
    grid = Grid1D(N=N, Lx=Lx)
    eqs = EulerEquations1D()
    rho = np.ones(N)
    u = np.zeros(N)
    p = np.ones(N)
    U0 = eqs.conservative(rho, u, p)

    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme="rk2", cfl=0.5)
    solver.run(U0, t0=0.0, t_end=0.01, output_interval=0.005)
    assert solver.t >= 0.01


