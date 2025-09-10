import numpy as np

from spectralhydro.grid import Grid1D
from spectralhydro.equations import EulerEquations1D
from spectralhydro.initial_conditions import sod_shock_tube
from spectralhydro.solver import SpectralSolver1D


def test_sod_runs_without_crash():
    N = 512
    Lx = 1.0
    grid = Grid1D(N=N, Lx=Lx, dealias=True)
    eqs = EulerEquations1D(gamma=1.4)
    left = {"rho": 1.0, "u": 0.0, "p": 1.0}
    right = {"rho": 0.125, "u": 0.0, "p": 0.1}
    U0 = sod_shock_tube(grid.x, x0=0.5, left=left, right=right, gamma=1.4)

    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme="rk4", cfl=0.4)
    solver.run(U0, t0=0.0, t_end=0.02, output_interval=0.01)
    assert solver.U is not None
    rho, u, p, _ = eqs.primitive(solver.U)
    assert np.all(np.isfinite(rho)) and np.all(np.isfinite(u)) and np.all(np.isfinite(p))


