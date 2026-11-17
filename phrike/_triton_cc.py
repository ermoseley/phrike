"""Keep Triton's helper build away from compilers it cannot drive."""

from __future__ import annotations

import os
import shutil

# Compilers that reject the GNU flags Triton passes (``-Wno-psabi``).
_UNUSABLE = frozenset({"nvc", "nvc++", "pgcc", "pgc++", "icc", "icx", "xlc", "craycc"})


def ensure_triton_compiler() -> None:
    """Point ``CC`` at gcc or clang when it names a compiler Triton cannot use.

    Triton builds a small C helper with ``$CC`` the first time a kernel runs.
    The NVIDIA HPC SDK modules, which CUDA-aware MPI builds usually need, set
    ``CC=nvc``; that build then fails and every Triton kernel with it, which
    leaves the solvers on the much slower Torch tier.  ``PHRIKE_TRITON_CC``
    names the compiler to use instead of the automatic choice.
    """
    override = os.environ.get("PHRIKE_TRITON_CC")
    if override:
        os.environ["CC"] = override
        return
    cc = os.environ.get("CC")
    if not cc:
        return  # Triton then looks for gcc or clang itself
    if os.path.basename(cc.split()[0]) not in _UNUSABLE:
        return
    replacement = shutil.which("gcc") or shutil.which("clang")
    if replacement:
        os.environ["CC"] = replacement
