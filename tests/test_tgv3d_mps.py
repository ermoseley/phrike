import os
import pytest

import numpy as np

from phrike.grid import Grid3D
from phrike.equations import EulerEquations3D
from phrike.initial_conditions import taylor_green_vortex_3d
from phrike.solver import SpectralSolver3D


def torch_available_mps():
    try:
        import torch  # type: ignore
    except Exception:
        return False
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


@pytest.mark.skipif(not torch_available_mps(), reason="Torch MPS not available on this system")
def test_tgv3d_runs_on_mps():
    try:
        import torch  # type: ignore
    except Exception:
        pytest.skip("Torch not installed")

    Nx, Ny, Nz = 32, 32, 16
    Lx = Ly = Lz = 2.0 * np.pi

    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dealias=True, filter_params={"enabled": True, "p": 8, "alpha": 36.0}, backend="torch", torch_device="mps")
    eqs = EulerEquations3D(gamma=1.4)

    X, Y, Z = grid.xyz_mesh()
    U0 = taylor_green_vortex_3d(X, Y, Z, rho0=1.0, p0=100.0, U0=1.0, k=1, gamma=1.4)

    solver = SpectralSolver3D(grid=grid, equations=eqs, scheme="rk4", cfl=0.2)
    history = solver.run(U0, t0=0.0, t_end=0.1, output_interval=0.05)

    assert solver.U is not None
    rho, ux, uy, uz, p = eqs.primitive(solver.U)
    # Check basic finiteness and reasonable ranges
    import torch  # type: ignore
    assert torch.isfinite(rho).all()
    assert torch.isfinite(ux).all()
    assert torch.isfinite(uy).all()
    assert torch.isfinite(uz).all()
    assert torch.isfinite(p).all()


