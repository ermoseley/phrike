"""On a CUDA machine, the distributed solvers' self-test must pass.

It compares every Triton kernel and both solver tiers with their Torch
counterparts (see ``phrike/parallel/selftest.py``).  Without CUDA this is
skipped; the Torch tier is covered by the other ``test_parallel_*`` files.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("precision", ["double", "single"])
def test_selftest_passes_on_cuda(precision):
    from phrike.parallel.selftest import main

    assert main(["--backend", "cuda", "--size", "32", "--precision", precision]) == 0
