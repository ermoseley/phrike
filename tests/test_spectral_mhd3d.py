"""The 3D MHD fast path (phrike/spectral_mhd3d.py) must reproduce the reference
SpectralSolverMHD step to round-off and keep div B at round-off."""

import warnings

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phrike.adaptive import create_adaptive_method  # noqa: E402
from phrike.equations import MHDEquations3D  # noqa: E402
from phrike.grid import Grid3D  # noqa: E402
from phrike.problems.problem_list.orszag_tang3d import orszag_tang_3d  # noqa: E402
from phrike.solver import SpectralSolverMHD3D  # noqa: E402

_CUDA = torch.cuda.is_available()
_DEVICES = ["cpu"] + (["cuda"] if _CUDA else [])


def _setup(device, shape=(16, 12, 10), dissipation=True, eta=0.0, nu=0.0, substage=True,
           precision="double", noise=0.0):
    Nx, Ny, Nz = shape
    grid = Grid3D(
        Nx=Nx, Ny=Ny, Nz=Nz, Lx=1.0, Ly=1.0, Lz=1.0, dealias=True,
        filter_params={"enabled": True, "p": 8, "rate": 300.0, "onset_fraction": 0.5} if dissipation else None,
        backend="torch", torch_device=device, precision=precision,
    )
    grid.mhd_config = {"resistivity": eta, "viscosity": nu, "divergence_clean": "projection",
                       "project_each_substage": substage}
    eqs = MHDEquations3D(gamma=5.0 / 3.0)
    U0 = orszag_tang_3d(*grid.xyz_mesh(), gamma=eqs.gamma, epsilon=0.2).to(grid.kx.dtype)
    if noise:
        g = torch.Generator(device="cpu").manual_seed(0)
        U0[0] = U0[0] * (1.0 + noise * torch.randn(U0[0].shape, generator=g, dtype=U0.dtype).to(U0.device))
    return grid, eqs, U0


def _solver(grid, eqs, fast, **kw):
    return SpectralSolverMHD3D(grid, eqs, spectral_fast_path=fast, **kw)


def _steps(solver, U, n, dt):
    U = solver.grid.project_dealiased(solver._project_B(U))
    for _ in range(n):
        U = solver.step(U, dt)
    return U


def _assert_close(A, B, rtol):
    A, B = A.detach().cpu().numpy(), B.detach().cpu().numpy()
    scale = np.abs(B).max()
    assert np.abs(A - B).max() <= rtol * scale, np.abs(A - B).max() / scale


def _rtol(precision):
    return 1e-5 if precision == "single" else 1e-11


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("dissipation,eta,nu,substage", [
    (True, 0.0, 0.0, True),
    (False, 0.0, 0.0, True),
    (True, 2e-3, 1e-3, False),
])
def test_fast_step_matches_reference(device, scheme, dissipation, eta, nu, substage):
    grid, eqs, U0 = _setup(device, dissipation=dissipation, eta=eta, nu=nu, substage=substage, noise=1e-2)
    fast = _solver(grid, eqs, True, scheme=scheme)
    ref = _solver(grid, eqs, False, scheme=scheme)
    assert fast.spectral is not None
    Uf = _steps(fast, U0, 4, 2e-3)
    Ur = _steps(ref, U0, 4, 2e-3)
    _assert_close(Uf, Ur, _rtol("double"))
    assert fast.max_div_b(Uf) < 1e-12


@pytest.mark.skipif(not _CUDA, reason="CUDA required")
def test_fast_step_matches_reference_fp32():
    grid, eqs, U0 = _setup("cuda", precision="single")
    Uf = _steps(_solver(grid, eqs, True), U0, 4, 2e-3)
    Ur = _steps(_solver(grid, eqs, False), U0, 4, 2e-3)
    _assert_close(Uf, Ur, _rtol("single"))


@pytest.mark.parametrize("device", _DEVICES)
def test_run_and_adaptive_match_reference(device):
    grid, eqs, U0 = _setup(device)
    cfg = {"enabled": True, "scheme": "rk45", "rtol": 1e-8, "atol": 1e-12}
    out = []
    for fast in (True, False):
        solver = _solver(grid, eqs, fast, adaptive_config=cfg)
        flags = []
        step = solver.adaptive_stepper.step

        def recording(*args, _step=step, _flags=flags, **kwargs):
            result = _step(*args, **kwargs)
            _flags.append(result.accepted)
            return result

        solver.adaptive_stepper.step = recording
        hist = solver.run(U0, t0=0.0, t_end=0.03, output_interval=1.0)
        out.append((solver.U, solver.t, flags, hist["div_b_max"][-1]))
    (Uf, tf, ff, divf), (Ur, tr, fr, _) = out
    assert ff == fr and len(fr) > 1
    assert tf == pytest.approx(tr, rel=1e-12)
    _assert_close(Uf, Ur, 1e-9)
    assert divf < 1e-12


@pytest.mark.parametrize("device", _DEVICES)
def test_embedded_step_matches_reference(device):
    grid, eqs, U0 = _setup(device)
    solver = _solver(grid, eqs, True)
    fast = solver.spectral
    U = fast.seed(grid.project_dealiased(solver._project_B(U0)))
    method = create_adaptive_method("rk45")
    dt, h = 2e-3, 1e-3
    U_new, error = fast.embedded_step(U, dt, method.a, method.b_high, method.b_low, h)
    R_high, _, R_err = method.step(solver._compute_rhs, U, dt)
    ref = solver._limit_accepted(grid.apply_accepted_state_stabilization(
        solver._project_B(solver._clamp(R_high)), h))
    _assert_close(U_new, ref, 1e-12)
    assert abs(error - R_err) <= 1e-13 * float(U.abs().max())
    assert fast.ready(U)  # an uncommitted attempt leaves the cache alone


def test_positivity_clamp_falls_back_to_reference():
    grid, eqs, U0 = _setup("cpu", dissipation=False)
    U0 = U0.clone()
    U0[4, :3, :3, :3] = -1.0  # clearly negative thermal pressure: the clamp must act
    fast = _solver(grid, eqs, True)
    ref = _solver(grid, eqs, False)
    U_in = fast.spectral.seed(U0)  # a cached state, so the fast path is attempted
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Uf = fast.step(U_in, 1e-4)
        Ur = ref.step(U_in.clone(), 1e-4)
    _assert_close(Uf, Ur, 1e-12)
    assert fast.positivity.events == ref.positivity.events >= 1


def test_unsupported_options_keep_reference_path():
    grid, eqs, _ = _setup("cpu")
    grid.mhd_config = dict(grid.mhd_config, divergence_clean="none")
    assert _solver(grid, eqs, True)._fast() is None
    numpy_grid = Grid3D(Nx=8, Ny=8, Nz=8, Lx=1.0, Ly=1.0, Lz=1.0, backend="numpy")
    assert _solver(numpy_grid, eqs, None).spectral is None
