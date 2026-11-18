"""A NumPy interpreter for the subset of Triton that PHRIKE's kernels use.

Triton needs CUDA, so the GPU tier of the solvers cannot run in CI or on a
laptop.  The kernel bodies, however, are ordinary Python that calls
``triton.language``; this module provides stand-ins for ``triton`` and
``triton.language`` that execute a kernel block by block on the tensors'
memory.  With them installed (``installed()``), the launch code, the strides
and offsets, the complex arithmetic and the in-place buffer reuse of the
Triton tier are executed and can be compared with the Torch tier.

What this cannot check is whether the Triton compiler accepts a kernel or how
the GPU rounds; ``python -m phrike.parallel.selftest`` does that on the target
machine.

Blocks run one after another.  That equals the parallel result for kernels in
which a block only writes elements that no other block reads, which is the
contract of every kernel here.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import types

import numpy as np
import torch

_KERNEL_MODULES = ("phrike.spectral3d_triton", "phrike.parallel.kernels_triton")
_state = {"pid": 0}


class Arr(np.ndarray):
    """ndarray with Triton's ``.to(dtype)``."""

    def __new__(cls, a):
        return np.asarray(a).view(cls)

    def to(self, dtype):
        return Arr(np.asarray(self).astype(dtype))


class _PtrType:
    def __init__(self, element_ty):
        self.element_ty = element_ty


class Ptr:
    """A tensor argument: flat storage plus element offsets."""

    def __init__(self, base: np.ndarray, index=0):
        self.base = base
        self.index = index

    @property
    def dtype(self):
        return _PtrType(self.base.dtype)

    def __add__(self, offset):
        return Ptr(self.base, self.index + np.asarray(offset))

    __radd__ = __add__


def _load(ptr, mask=None, other=None):
    index = np.asarray(ptr.index)
    if mask is None:
        return Arr(ptr.base[index])
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), index.shape)
    values = ptr.base[np.where(mask, index, 0)]
    return Arr(np.where(mask, values, 0 if other is None else other).astype(ptr.base.dtype))


def _store(ptr, value, mask=None):
    index = np.asarray(ptr.index)
    value = np.broadcast_to(np.asarray(value), index.shape)
    if mask is None:
        ptr.base[index] = value
        return
    mask = np.broadcast_to(np.asarray(mask, dtype=bool), index.shape)
    ptr.base[index[mask]] = value[mask]


def _atomic_add(ptr, value):
    ptr.base[np.asarray(ptr.index)] += value


def _language() -> types.ModuleType:
    tl = types.ModuleType("triton.language")
    tl.constexpr = object
    tl.int32, tl.int64 = np.int32, np.int64
    tl.uint32 = np.uint32
    tl.float32, tl.float64 = np.float32, np.float64
    tl.program_id = lambda axis: Arr(np.int32(_state["pid"]))
    tl.arange = lambda start, stop: Arr(np.arange(start, stop, dtype=np.int32))
    tl.load, tl.store, tl.atomic_add = _load, _store, _atomic_add
    tl.where = lambda c, a, b: Arr(np.where(c, a, b))
    tl.sum = lambda x, axis=None: Arr(np.sum(x, axis=axis))
    tl.max = lambda x, axis=None: Arr(np.max(x, axis=axis))
    tl.abs = lambda x: Arr(np.abs(x))
    tl.sqrt = lambda x: Arr(np.sqrt(x))
    tl.exp = lambda x: Arr(np.exp(x))
    tl.maximum = lambda a, b: Arr(np.fmax(a, b))  # like the GPU's fmax: a NaN operand is dropped
    tl.minimum = lambda a, b: Arr(np.fmin(a, b))
    tl.zeros = lambda shape, dtype: Arr(np.zeros(shape, dtype=dtype))
    tl.split = lambda v: (Arr(v[..., 0]), Arr(v[..., 1]))
    tl.join = lambda a, b: Arr(np.stack([np.asarray(a), np.asarray(b)], axis=-1))
    tl.static_range = range
    return tl


def _convert(arg):
    if isinstance(arg, torch.Tensor):
        if arg.device.type != "cpu" or not arg.is_contiguous():
            raise ValueError("the Triton emulator needs contiguous CPU tensors")
        return Ptr(arg.detach().numpy().reshape(-1))
    if isinstance(arg, bool):
        return arg
    if isinstance(arg, (int, np.integer)):
        # Triton types a Python int as int32 when it fits; wrap-around on
        # un-widened products is part of what the kernels must avoid.
        return Arr(np.int32(arg) if abs(int(arg)) < 2 ** 31 else np.int64(arg))
    if isinstance(arg, float):
        return np.float32(arg)  # as Triton: Python floats reach a kernel as fp32
    return arg


class _Launcher:
    def __init__(self, fn, grid):
        self.fn, self.grid = fn, grid

    def __call__(self, *args, **kwargs):
        kwargs.pop("num_warps", None)
        converted = [_convert(a) for a in args]
        with np.errstate(all="ignore"):
            for pid in range(int(self.grid[0])):
                _state["pid"] = pid
                self.fn(*converted, **kwargs)


class JITFunction:
    def __init__(self, fn):
        self.fn = fn
        self.__name__ = getattr(fn, "__name__", "kernel")

    def __call__(self, *args, **kwargs):  # a helper called from inside a kernel
        return self.fn(*args, **kwargs)

    def __getitem__(self, grid):
        return _Launcher(self.fn, grid)


def _triton() -> types.ModuleType:
    module = types.ModuleType("triton")
    module.__version__ = "emulated"
    module.jit = JITFunction
    module.cdiv = lambda a, b: -(-int(a) // int(b))
    module.language = _language()
    return module


@contextlib.contextmanager
def installed():
    """Import PHRIKE's kernel modules against the emulator, then restore everything."""
    names = ("triton", "triton.language") + _KERNEL_MODULES
    saved = {name: sys.modules.get(name) for name in names}
    attrs = {}
    for name in _KERNEL_MODULES:
        parent, _, child = name.rpartition(".")
        package = importlib.import_module(parent)
        attrs[name] = (package, child, getattr(package, child, None))
    fake = _triton()
    try:
        sys.modules["triton"] = fake
        sys.modules["triton.language"] = fake.language
        for name in _KERNEL_MODULES:
            sys.modules.pop(name, None)
            package, child, _ = attrs[name]
            if hasattr(package, child):
                delattr(package, child)
            importlib.import_module(name)
        yield fake
    finally:
        for name in names:
            if saved[name] is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved[name]
        for name, (package, child, original) in attrs.items():
            if original is None:
                if hasattr(package, child):
                    delattr(package, child)
            else:
                setattr(package, child, original)
