"""The bandwidth-lean 3D Euler path (phrike/spectral3d.py) must reproduce the
reference per-derivative complex-FFT path to round-off."""

import warnings

import numpy as np
import pytest

from phrike.adaptive import create_adaptive_method
from phrike.equations import EulerEquations3D
from phrike.grid import Grid3D
from phrike.solver import SpectralSolver3D, _compute_rhs_3d

try:
    import torch

    _CUDA = torch.cuda.is_available()
except Exception:  # pragma: no cover
    torch = None
    _CUDA = False

_DISSIPATION = {"enabled": True, "p": 8, "rate": 40.0}


def _grid(shape, dealias=True, dissipation=True, backend="numpy", device=None, precision="double"):
    Nx, Ny, Nz = shape
    return Grid3D(
        Nx=Nx, Ny=Ny, Nz=Nz, Lx=2 * np.pi, Ly=2 * np.pi, Lz=2 * np.pi,
        dealias=dealias, filter_params=_DISSIPATION if dissipation else None,
        backend=backend, torch_device=device, precision=precision,
    )


def _state(grid, seed=0, noise=1e-2):
    """Taylor-Green-like flow plus white noise (content above the dealias cutoff)."""
    Nz, Ny, Nx = grid.Nz, grid.Ny, grid.Nx
    z, y, x = np.meshgrid(*(np.arange(n) * 2 * np.pi / n for n in (Nz, Ny, Nx)), indexing="ij")
    rng = np.random.default_rng(seed)
    rho = 1.0 + 0.1 * np.cos(x) * np.cos(z) + noise * rng.standard_normal(x.shape)
    ux = np.sin(x) * np.cos(y) * np.cos(z)
    uy = -np.cos(x) * np.sin(y) * np.cos(z)
    uz = 0.3 * np.sin(z) * np.cos(y)
    p = 10.0 + noise * rng.standard_normal(x.shape)
    eqs = EulerEquations3D(gamma=1.4)
    return eqs, eqs.conservative(rho, ux, uy, uz, p)


def _as(grid, U):
    if getattr(grid, "_use_torch", False):
        return torch.as_tensor(U, dtype=grid.kx.dtype, device=grid.kx.device)
    return U


def _run(grid, eqs, U, fast, scheme="rk4", steps=3, dt=2e-3):
    solver = SpectralSolver3D(grid, eqs, scheme=scheme, spectral_fast_path=fast)
    U = _as(grid, U)
    for _ in range(steps):
        U = solver.step(U, dt)
    return U


def _run_adaptive(grid, eqs, U0, fast, rtol=1e-6, t_end=0.05):
    """Adaptive RK45 run; returns (U, t, accepted flags of every attempt)."""
    solver = SpectralSolver3D(
        grid, eqs, spectral_fast_path=fast,
        adaptive_config={"enabled": True, "scheme": "rk45", "rtol": rtol, "atol": 1e-12},
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
    ((16, 12, 10), True, True),
    ((15, 9, 11), True, False),
    ((12, 12, 12), False, True),
])
def test_fast_step_matches_reference_numpy(scheme, shape, dealias, dissipation):
    grid = _grid(shape, dealias, dissipation)
    eqs, U0 = _state(grid)
    _assert_close(_run(grid, eqs, U0, True, scheme), _run(grid, eqs, U0, False, scheme))


def test_fast_rhs_matches_reference_numpy():
    grid = _grid((16, 14, 12))
    eqs, U0 = _state(grid)
    solver = SpectralSolver3D(grid, eqs, spectral_fast_path=True)
    _assert_close(solver.spectral.rhs(U0), _compute_rhs_3d(grid, eqs, U0))


def test_fast_path_defaults():
    eqs = EulerEquations3D()
    assert SpectralSolver3D(_grid((8, 8, 8)), eqs).spectral is None
    if torch is not None:
        assert SpectralSolver3D(_grid((8, 8, 8), backend="torch", device="cpu"), eqs).spectral is not None

    class Custom(EulerEquations3D):
        pass

    assert SpectralSolver3D(_grid((8, 8, 8)), Custom(), spectral_fast_path=True).spectral is None


def test_positivity_clamp_falls_back_to_reference_numpy():
    grid = _grid((8, 8, 8), dissipation=False)
    eqs, U0 = _state(grid, noise=0.0)
    U0 = np.array(U0, copy=True)
    U0[4, 0, 0, 0] = -1.0  # negative pressure: the clamp must act
    fast = SpectralSolver3D(grid, eqs, spectral_fast_path=True)
    ref = SpectralSolver3D(grid, eqs, spectral_fast_path=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        Uf = fast.step(U0, 1e-4)
        Ur = ref.step(U0, 1e-4)
    np.testing.assert_array_equal(Uf, Ur)
    assert fast.positivity.events == ref.positivity.events >= 1


def test_adaptive_rk45_numpy_matches_reference():
    grid = _grid((12, 10, 8))
    eqs, U0 = _state(grid)
    U_fast, t_fast, flags_fast = _run_adaptive(grid, eqs, U0, True)
    U_ref, t_ref, flags_ref = _run_adaptive(grid, eqs, U0, False)
    assert flags_fast == flags_ref and len(flags_ref) > 1
    assert t_fast == pytest.approx(t_ref, rel=1e-12) and t_ref == pytest.approx(0.05)
    # The step-size controller turns round-off in the error estimate into tiny
    # dt differences, so agreement is close to, not at, round-off.
    _assert_close(U_fast, U_ref, rtol=1e-9)


@pytest.mark.skipif(torch is None, reason="torch not installed")
def test_fast_step_matches_reference_torch_cpu():
    grid = _grid((12, 10, 8), backend="torch", device="cpu")
    eqs, U0 = _state(grid)
    _assert_close(_run(grid, eqs, U0, True), _run(grid, eqs, U0, False))


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton kernels")
@pytest.mark.parametrize("dealias,dissipation", [(True, True), (True, False), (False, True)])
def test_triton_rk4_matches_reference(dealias, dissipation):
    grid = _grid((32, 24, 20), dealias=dealias, dissipation=dissipation, backend="torch", device="cuda")
    eqs, U0 = _state(grid)
    solver = SpectralSolver3D(grid, eqs)
    if solver.spectral._triton is None:
        pytest.skip("Triton path unavailable")
    # The first step sees a non-band-limited input (all modes); later steps
    # reuse the cached spectrum and visit only the retained modes.
    _assert_close(_run(grid, eqs, U0, True, steps=4), _run(grid, eqs, U0, False, steps=4))


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton kernels")
def test_triton_rk4_matches_reference_fp32():
    grid = _grid((32, 24, 20), backend="torch", device="cuda", precision="single")
    eqs, U0 = _state(grid)
    _assert_close(_run(grid, eqs, U0, True, steps=4), _run(grid, eqs, U0, False, steps=4), rtol=1e-5)


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton kernels")
def test_triton_adaptive_run_matches_reference():
    # A tight tolerance forces rejected attempts through the spectral embedded path.
    grid = _grid((32, 24, 20), backend="torch", device="cuda")
    eqs, U0 = _state(grid)
    U_fast, t_fast, flags_fast = _run_adaptive(grid, eqs, U0, True, rtol=1e-10)
    U_ref, t_ref, flags_ref = _run_adaptive(grid, eqs, U0, False, rtol=1e-10)
    assert not all(flags_ref), "expected some rejected attempts"
    assert flags_fast == flags_ref
    assert t_fast == pytest.approx(t_ref, rel=1e-12) and t_ref == pytest.approx(0.05)
    _assert_close(U_fast, U_ref, rtol=1e-9)


@pytest.mark.skipif(not _CUDA, reason="CUDA required for the Triton kernels")
@pytest.mark.parametrize("method_name", ["rk23", "rk45"])
def test_triton_embedded_step_matches_reference(method_name):
    grid = _grid((32, 24, 20), backend="torch", device="cuda")
    eqs, U0 = _state(grid)
    spectral = SpectralSolver3D(grid, eqs).spectral
    if spectral._triton is None:
        pytest.skip("Triton path unavailable")
    # Make U band-limited and cached, as after any completed step.
    U = spectral.dissipate(_as(grid, U0), 0.0, project=True)
    assert spectral.embedded_ready(U)
    method = create_adaptive_method(method_name)
    dt, h = 3e-3, 1.5e-3
    U_new, error = spectral.embedded_step(U, dt, method.a, method.b_high, method.b_low, h)

    rhs = lambda V: _compute_rhs_3d(grid, eqs, V)
    R_high, R_low, R_err = method.step(rhs, U, dt)
    _assert_close(U_new, grid.apply_accepted_state_stabilization(R_high, h))
    # The reference subtracts two O(|U|) states, so it is only good to round-off
    # of |U|; the spectral path sums dt*(b - b_low)*k directly.
    assert abs(error - R_err) <= 1e-13 * float(U.abs().max())
    # An uncommitted attempt leaves the cached state usable.
    assert spectral.embedded_ready(U)
