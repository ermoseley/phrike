"""The 2D Euler fast path (Euler2DSpectral) must reproduce the reference
SpectralSolver2D step to round-off."""

import warnings

import numpy as np
import pytest

from phrike.adaptive import create_adaptive_method
from phrike.equations import EulerEquations2D
from phrike.grid import Grid2D
from phrike.solver import SpectralSolver2D, _compute_rhs_2d

try:
    import torch

    _CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None
    _CUDA = False

_DISSIPATION = {"enabled": True, "p": 2, "rate": 5.0, "onset_fraction": 0.8}


def _grid(shape, dealias=True, dissipation=True, backend="numpy", device=None, precision="double"):
    Nx, Ny = shape
    return Grid2D(
        Nx=Nx, Ny=Ny, Lx=1.0, Ly=1.0, dealias=dealias,
        filter_params=_DISSIPATION if dissipation else None,
        backend=backend, torch_device=device, precision=precision,
    )


def _state(grid, noise=1e-2):
    """Shear layer plus white noise (content above the dealias cutoff)."""
    y, x = np.meshgrid(*(np.arange(n) / n for n in (grid.Ny, grid.Nx)), indexing="ij")
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.5 * np.tanh((np.abs(y - 0.5) - 0.25) / 0.05) + noise * rng.standard_normal(x.shape)
    ux = 0.5 * np.tanh((np.abs(y - 0.5) - 0.25) / 0.05)
    uy = 0.01 * np.sin(4 * np.pi * x)
    p = 2.5 + noise * rng.standard_normal(x.shape)
    eqs = EulerEquations2D(gamma=1.4)
    return eqs, eqs.conservative(rho, ux, uy, p)


def _as(grid, U):
    if getattr(grid, "_use_torch", False):
        return torch.as_tensor(U, dtype=grid.kx.dtype, device=grid.kx.device)
    return U


def _run(grid, eqs, U, fast, scheme="rk4", steps=3, dt=1e-3):
    solver = SpectralSolver2D(grid, eqs, scheme=scheme, spectral_fast_path=fast)
    U = _as(grid, U)
    for _ in range(steps):
        U = solver.step(U, dt)
    return U


def _run_adaptive(grid, eqs, U0, fast, scheme="rk45", rtol=1e-6, t_end=0.02):
    solver = SpectralSolver2D(
        grid, eqs, spectral_fast_path=fast,
        adaptive_config={"enabled": True, "scheme": scheme, "rtol": rtol, "atol": 1e-12},
    )
    flags = []
    step = solver.adaptive_stepper.step

    def recording_step(*args, **kwargs):
        result = step(*args, **kwargs)
        flags.append(result.accepted)
        return result

    solver.adaptive_stepper.step = recording_step
    solver.run(_as(grid, U0), t0=0.0, t_end=t_end, output_interval=1.0)
    return solver.U, solver.t, flags


def _to_np(a):
    return a.detach().cpu().numpy() if torch is not None and isinstance(a, torch.Tensor) else a


def _assert_close(A, B, rtol=1e-12):
    A, B = _to_np(A), _to_np(B)
    scale = np.abs(B).max()
    assert np.abs(A - B).max() <= rtol * scale, np.abs(A - B).max() / scale


@pytest.mark.parametrize("scheme", ["rk2", "rk4"])
@pytest.mark.parametrize("shape,dealias,dissipation", [
    ((32, 24), True, True),
    ((27, 19), True, False),
    ((24, 24), False, True),
])
def test_fast_step_matches_reference_numpy(scheme, shape, dealias, dissipation):
    grid = _grid(shape, dealias, dissipation)
    eqs, U0 = _state(grid)
    _assert_close(_run(grid, eqs, U0, True, scheme), _run(grid, eqs, U0, False, scheme))


def test_fast_rhs_matches_reference_numpy():
    grid = _grid((32, 28))
    eqs, U0 = _state(grid)
    solver = SpectralSolver2D(grid, eqs, spectral_fast_path=True)
    _assert_close(solver.spectral.rhs(U0), _compute_rhs_2d(grid, eqs, U0))


def test_positivity_clamp_falls_back_to_reference_numpy():
    grid = _grid((16, 16), dissipation=False)
    eqs, U0 = _state(grid, noise=0.0)
    U0 = np.array(U0, copy=True)
    U0[3, :3, :3] = -1.0  # negative pressure: the clamp must act
    fast = SpectralSolver2D(grid, eqs, spectral_fast_path=True)
    ref = SpectralSolver2D(grid, eqs, spectral_fast_path=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Uf = fast.step(U0, 1e-4)
        Ur = ref.step(U0, 1e-4)
    np.testing.assert_array_equal(Uf, Ur)
    assert fast.positivity.events == ref.positivity.events >= 1


def test_adaptive_numpy_matches_reference():
    grid = _grid((32, 24))
    eqs, U0 = _state(grid)
    U_fast, t_fast, flags_fast = _run_adaptive(grid, eqs, U0, True)
    U_ref, t_ref, flags_ref = _run_adaptive(grid, eqs, U0, False)
    assert flags_fast == flags_ref and len(flags_ref) > 1
    assert t_fast == pytest.approx(t_ref, rel=1e-12)
    _assert_close(U_fast, U_ref, rtol=1e-9)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_fast_step_matches_reference_torch_cpu():
    grid = _grid((32, 24), backend="torch", device="cpu")
    eqs, U0 = _state(grid)
    _assert_close(_run(grid, eqs, U0, True), _run(grid, eqs, U0, False))


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton kernels")
@pytest.mark.parametrize("dealias,dissipation", [(True, True), (True, False), (False, True)])
def test_triton_rk4_matches_reference(dealias, dissipation):
    grid = _grid((64, 48), dealias=dealias, dissipation=dissipation, backend="torch", device="cuda")
    eqs, U0 = _state(grid)
    solver = SpectralSolver2D(grid, eqs)
    if solver.spectral._triton is None:
        pytest.skip("Triton path unavailable")
    _assert_close(_run(grid, eqs, U0, True, steps=4), _run(grid, eqs, U0, False, steps=4))


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton kernels")
def test_triton_rk4_matches_reference_fp32():
    grid = _grid((64, 48), backend="torch", device="cuda", precision="single")
    eqs, U0 = _state(grid)
    _assert_close(_run(grid, eqs, U0, True, steps=4), _run(grid, eqs, U0, False, steps=4), rtol=1e-5)


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton kernels")
@pytest.mark.parametrize("scheme", ["rk45", "rk78"])
def test_triton_adaptive_run_matches_reference(scheme):
    grid = _grid((64, 48), backend="torch", device="cuda")
    eqs, U0 = _state(grid)
    U_fast, t_fast, flags_fast = _run_adaptive(grid, eqs, U0, True, scheme=scheme, rtol=1e-10)
    U_ref, t_ref, flags_ref = _run_adaptive(grid, eqs, U0, False, scheme=scheme, rtol=1e-10)
    assert flags_fast == flags_ref
    if scheme == "rk45":  # rk78 meets this tolerance without rejections
        assert not all(flags_ref)
    assert t_fast == pytest.approx(t_ref, rel=1e-12)
    _assert_close(U_fast, U_ref, rtol=1e-9)


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton kernels")
def test_triton_embedded_step_matches_reference():
    grid = _grid((64, 48), backend="torch", device="cuda")
    eqs, U0 = _state(grid)
    spectral = SpectralSolver2D(grid, eqs).spectral
    U = spectral.dissipate(_as(grid, U0), 0.0, project=True)
    assert spectral.embedded_ready(U)
    method = create_adaptive_method("rk45")
    dt, h = 1e-3, 5e-4
    U_new, error = spectral.embedded_step(U, dt, method.a, method.b_high, method.b_low, h)
    R_high, _, R_err = method.step(lambda V: _compute_rhs_2d(grid, eqs, V), U, dt)
    _assert_close(U_new, grid.apply_accepted_state_stabilization(R_high, h))
    assert abs(error - R_err) <= 1e-13 * float(U.abs().max())
