"""1D fast paths (Euler1DSpectral, MHD1DSpectral) against the reference steps."""

import numpy as np
import pytest

from phrike.adaptive import create_adaptive_method
from phrike.equations import EulerEquations1D, MHDEquations1D
from phrike.grid import Grid1D
from phrike.solver import SpectralSolver1D, SpectralSolverMHD1D, _compute_rhs

try:
    import torch

    _DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
except Exception:  # pragma: no cover
    torch = None
    _DEVICES = []

_DISSIPATION = {"enabled": True, "p": 8, "rate": 50.0}


def _grid(N, backend="numpy", device=None, dissipation=True):
    return Grid1D(N=N, Lx=1.0, basis="fourier", bc="periodic", dealias=True,
                  filter_params=_DISSIPATION if dissipation else None,
                  backend=backend, torch_device=device, precision="double")


def _euler_state(grid, noise=1e-2):
    x = np.arange(grid.N) / grid.N
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.2 * np.sin(2 * np.pi * x) + noise * rng.standard_normal(x.shape)
    u = 0.3 * np.cos(2 * np.pi * x)
    p = 1.0 + 0.1 * np.sin(4 * np.pi * x)
    eqs = EulerEquations1D(gamma=1.4)
    return eqs, eqs.conservative(rho, u, p)


def _as(grid, U):
    if getattr(grid, "_use_torch", False):
        return torch.as_tensor(np.asarray(U), dtype=grid.x.dtype, device=grid.x.device)
    return U


def _to_np(a):
    return a.detach().cpu().numpy() if torch is not None and isinstance(a, torch.Tensor) else np.asarray(a)


def _assert_close(A, B, rtol):
    A, B = _to_np(A), _to_np(B)
    scale = np.abs(B).max()
    assert np.abs(A - B).max() <= rtol * scale, np.abs(A - B).max() / scale


def _steps(solver, U, n, dt):
    for _ in range(n):
        U = solver.step(U, dt)
    return U


@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("dissipation", [True, False])
def test_euler1d_fast_matches_reference_numpy(scheme, dissipation):
    grid = _grid(96, dissipation=dissipation)
    eqs, U0 = _euler_state(grid)
    Uf = _steps(SpectralSolver1D(grid, eqs, scheme=scheme, spectral_fast_path=True), U0, 4, 1e-3)
    Ur = _steps(SpectralSolver1D(grid, eqs, scheme=scheme, spectral_fast_path=False), U0, 4, 1e-3)
    _assert_close(Uf, Ur, 1e-12)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("device", _DEVICES)
def test_euler1d_fast_matches_reference_torch(device):
    grid = _grid(256, backend="torch", device=device)
    eqs, U0 = _euler_state(grid)
    U0 = _as(grid, U0)
    Uf = _steps(SpectralSolver1D(grid, eqs, spectral_fast_path=True), U0, 4, 1e-3)
    Ur = _steps(SpectralSolver1D(grid, eqs, spectral_fast_path=False), U0, 4, 1e-3)
    _assert_close(Uf, Ur, 1e-12)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("device", _DEVICES)
def test_euler1d_adaptive_matches_reference(device):
    grid = _grid(256, backend="torch", device=device)
    eqs, U0 = _euler_state(grid)
    out = []
    for fast in (True, False):
        solver = SpectralSolver1D(grid, eqs, spectral_fast_path=fast,
                                  adaptive_config={"enabled": True, "scheme": "rk45", "rtol": 1e-8, "atol": 1e-12})
        solver.run(_as(grid, U0), t0=0.0, t_end=0.02, output_interval=1.0)
        out.append((solver.U, solver.t))
    assert out[0][1] == pytest.approx(out[1][1], rel=1e-12)
    _assert_close(out[0][0], out[1][0], 1e-9)


@pytest.mark.skipif(torch is None, reason="torch not installed")
@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("eta", [0.0, 2e-3])
def test_mhd1d_fast_matches_reference(device, eta):
    grid = _grid(256, backend="torch", device=device)
    grid.mhd_config = {"resistivity": eta, "divergence_clean": "projection", "project_each_substage": True}
    eqs = MHDEquations1D(gamma=5.0 / 3.0)
    x = grid.x
    U0 = eqs.conservative(1.0 + 0.1 * torch.sin(2 * np.pi * x), 0.1 * torch.cos(2 * np.pi * x),
                          0.05 * torch.sin(4 * np.pi * x), 0.02 * torch.cos(2 * np.pi * x),
                          torch.ones_like(x), 0.75 + 0 * x, torch.cos(2 * np.pi * x), torch.sin(2 * np.pi * x))
    fast = SpectralSolverMHD1D(grid, eqs, spectral_fast_path=True)
    ref = SpectralSolverMHD1D(grid, eqs, spectral_fast_path=False)
    assert fast.spectral is not None

    def run(solver):
        U = solver.grid.project_dealiased(solver._project_B(U0))
        return _steps(solver, U, 4, 1e-3)

    _assert_close(run(fast), run(ref), 1e-11)
