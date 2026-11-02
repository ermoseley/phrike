"""The 2.5D MHD fast path (MHD2DSpectral) must reproduce the reference
SpectralSolverMHD step to round-off and keep div B at round-off."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phrike.adaptive import create_adaptive_method  # noqa: E402
from phrike.equations import MHDEquations2D  # noqa: E402
from phrike.grid import Grid2D  # noqa: E402
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d  # noqa: E402
from phrike.solver import SpectralSolverMHD2D  # noqa: E402

_DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _setup(device, shape=(48, 40), dissipation=True, eta=0.0, nu=0.0, precision="double"):
    Nx, Ny = shape
    grid = Grid2D(
        Nx=Nx, Ny=Ny, Lx=1.0, Ly=1.0, dealias=True,
        filter_params={"enabled": True, "p": 8, "rate": 300.0, "onset_fraction": 0.5} if dissipation else None,
        backend="torch", torch_device=device, precision=precision,
    )
    grid.mhd_config = {"resistivity": eta, "viscosity": nu, "divergence_clean": "projection",
                       "project_each_substage": True}
    eqs = MHDEquations2D(gamma=5.0 / 3.0)
    X, Y = grid.xy_mesh()
    U0 = orszag_tang_2d(X, Y, gamma=eqs.gamma).to(grid.kx.dtype)
    U0[3] = U0[3] + 0.05 * U0[0] * torch.sin(2 * np.pi * X)  # exercise uz and Bz
    U0[7] = U0[7] + 0.05 * torch.cos(2 * np.pi * Y)
    return grid, eqs, U0


def _steps(solver, U, n, dt):
    U = solver.grid.project_dealiased(solver._project_B(U))
    for _ in range(n):
        U = solver.step(U, dt)
    return U


def _assert_close(A, B, rtol):
    A, B = A.detach().cpu().numpy(), B.detach().cpu().numpy()
    scale = np.abs(B).max()
    assert np.abs(A - B).max() <= rtol * scale, np.abs(A - B).max() / scale


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("dissipation,eta,nu", [(True, 0.0, 0.0), (False, 2e-3, 1e-3)])
def test_fast_step_matches_reference(device, scheme, dissipation, eta, nu):
    grid, eqs, U0 = _setup(device, dissipation=dissipation, eta=eta, nu=nu)
    fast = SpectralSolverMHD2D(grid, eqs, scheme=scheme, spectral_fast_path=True)
    ref = SpectralSolverMHD2D(grid, eqs, scheme=scheme, spectral_fast_path=False)
    assert fast.spectral is not None
    Uf = _steps(fast, U0, 4, 1e-3)
    Ur = _steps(ref, U0, 4, 1e-3)
    _assert_close(Uf, Ur, 1e-11)
    assert fast.max_div_b(Uf) < 1e-11


@pytest.mark.parametrize("device", _DEVICES)
def test_adaptive_run_matches_reference(device):
    grid, eqs, U0 = _setup(device)
    cfg = {"enabled": True, "scheme": "rk45", "rtol": 1e-8, "atol": 1e-12}
    out = []
    for fast in (True, False):
        solver = SpectralSolverMHD2D(grid, eqs, adaptive_config=cfg, spectral_fast_path=fast)
        solver.run(U0, t0=0.0, t_end=0.02, output_interval=1.0)
        out.append((solver.U, solver.t))
    assert out[0][1] == pytest.approx(out[1][1], rel=1e-12)
    _assert_close(out[0][0], out[1][0], 1e-9)


@pytest.mark.parametrize("device", _DEVICES)
def test_embedded_step_matches_reference(device):
    grid, eqs, U0 = _setup(device)
    solver = SpectralSolverMHD2D(grid, eqs, spectral_fast_path=True)
    fast = solver.spectral
    U = fast.seed(grid.project_dealiased(solver._project_B(U0)))
    method = create_adaptive_method("rk45")
    dt, h = 1e-3, 5e-4
    U_new, error = fast.embedded_step(U, dt, method.a, method.b_high, method.b_low, h)
    R_high, _, R_err = method.step(solver._compute_rhs, U, dt)
    ref = solver._limit_accepted(grid.apply_accepted_state_stabilization(
        solver._project_B(solver._clamp(R_high)), h))
    _assert_close(U_new, ref, 1e-12)
    assert abs(error - R_err) <= 1e-13 * float(U.abs().max())
