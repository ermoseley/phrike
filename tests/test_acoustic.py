import numpy as np

from hydra.grid import Grid1D
from hydra.equations import EulerEquations1D
from hydra.solver import SpectralSolver1D
from hydra.acoustic import AcousticParams, acoustic_ic, acoustic_exact


def test_acoustic_accuracy_small_amplitude():
    N = 256
    Lx = 1.0
    grid = Grid1D(N=N, Lx=Lx, dealias=True)
    eqs = EulerEquations1D(gamma=1.4)
    params = AcousticParams(rho0=1.0, p0=1.0, u0=0.0, amplitude=1e-7, mode_m=2, Lx=Lx, direction="right", fractional=True, gamma=1.4)

    U0 = acoustic_ic(grid.x, params)

    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme="rk4", cfl=0.6)
    t_end = 0.5
    solver.run(U0, t0=0.0, t_end=t_end, output_interval=t_end)

    rho_a, u_a, p_a = acoustic_exact(grid.x, t_end, params)
    rho, u, p, _ = eqs.primitive(solver.U)

    def rel_l2(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    assert rel_l2(rho, rho_a) < 1e-6
    assert rel_l2(u, u_a) < 1e-6
    assert rel_l2(p, p_a) < 1e-6


