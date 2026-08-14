from types import SimpleNamespace

import pytest

import phrike.grid as grid_module
from phrike.grid import Grid1D, Grid2D, Grid3D, resolve_backend


class _Availability:
    def __init__(self, available: bool):
        self._available = available

    def is_available(self) -> bool:
        return self._available


def _fake_torch(*, mps: bool, cuda: bool):
    return SimpleNamespace(
        backends=SimpleNamespace(mps=_Availability(mps)),
        cuda=_Availability(cuda),
        float32="float32",
        float64="float64",
        complex64="complex64",
        complex128="complex128",
    )


def test_auto_backend_prefers_metal_then_cuda(monkeypatch):
    monkeypatch.setattr(grid_module, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(grid_module, "torch", _fake_torch(mps=True, cuda=True))
    assert resolve_backend("auto") == ("torch", "mps")

    monkeypatch.setattr(grid_module, "torch", _fake_torch(mps=False, cuda=True))
    assert resolve_backend("auto") == ("torch", "cuda")


def test_auto_backend_falls_back_to_numpy_cpu(monkeypatch):
    monkeypatch.setattr(grid_module, "_TORCH_AVAILABLE", False)
    monkeypatch.setattr(grid_module, "torch", None)

    assert resolve_backend("auto") == ("numpy", None)
    assert resolve_backend("cpu") == ("numpy", None)
    with pytest.raises(ImportError, match="requires PyTorch"):
        resolve_backend("torch")


def test_explicit_backend_aliases_and_device_validation(monkeypatch):
    monkeypatch.setattr(grid_module, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(grid_module, "torch", _fake_torch(mps=True, cuda=True))

    assert resolve_backend("metal") == ("torch", "mps")
    assert resolve_backend("cuda") == ("torch", "cuda")
    assert resolve_backend("torch", "metal") == ("torch", "mps")
    assert resolve_backend("torch", "cpu") == ("torch", "cpu")
    assert resolve_backend("numpy", "cpu") == ("numpy", None)
    assert resolve_backend("cpu", "cpu") == ("numpy", None)

    with pytest.raises(ValueError, match="only supports CPU"):
        resolve_backend("numpy", "cuda")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_backend("metal", "cuda")


def test_explicit_unavailable_accelerator_fails_clearly(monkeypatch):
    monkeypatch.setattr(grid_module, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(grid_module, "torch", _fake_torch(mps=False, cuda=False))

    with pytest.raises(RuntimeError, match="Metal/MPS"):
        resolve_backend("metal")
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_backend("cuda")
    assert resolve_backend("torch") == ("torch", "cpu")


def test_metal_uses_single_precision(monkeypatch):
    fake = _fake_torch(mps=True, cuda=False)
    monkeypatch.setattr(grid_module, "_TORCH_AVAILABLE", True)
    monkeypatch.setattr(grid_module, "torch", fake)

    real_dtype, complex_dtype, precision = grid_module._torch_dtypes("double", "mps")
    assert (real_dtype, complex_dtype, precision) == (
        fake.float32,
        fake.complex64,
        "single",
    )


def test_grid_default_uses_numpy_when_no_accelerator_is_installed():
    grid = Grid1D(N=8, Lx=1.0)
    assert grid.backend == "numpy"
    assert grid.torch_device is None
    assert not grid._use_torch


def test_torch_cpu_is_supported_in_every_grid_dimension():
    torch = pytest.importorskip("torch")
    grids = (
        Grid1D(N=8, Lx=1.0, backend="torch", torch_device="cpu"),
        Grid2D(
            Nx=4,
            Ny=4,
            Lx=1.0,
            Ly=1.0,
            backend="torch",
            torch_device="cpu",
        ),
        Grid3D(
            Nx=4,
            Ny=4,
            Nz=4,
            Lx=1.0,
            Ly=1.0,
            Lz=1.0,
            backend="torch",
            torch_device="cpu",
        ),
    )

    for grid in grids:
        assert grid.backend == "torch"
        assert grid.torch_device == "cpu"
        assert grid.x.device.type == "cpu"
        assert grid.x.dtype == torch.float64
