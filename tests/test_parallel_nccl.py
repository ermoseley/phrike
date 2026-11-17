"""The torch.distributed ("nccl") transpose, on the CPU through Gloo.

``tests/_parallel_nccl_worker.py`` runs under ``mpirun`` here: with
``PHRIKE_MPI_TRANSPOSE=nccl`` the distributed FFT and a few solver steps equal
the ``alltoallv`` result bit for bit, and when the group cannot start on one
rank every rank falls back to ``alltoallv`` together.  Skipped when ``mpirun``
or ``mpi4py`` is missing, and inside an MPI launch.  The CUDA/NCCL backend
itself can only be exercised on a GPU cluster (``python -m
phrike.parallel.selftest --transpose nccl``).
"""

import os
import subprocess
import sys

import pytest

pytest.importorskip("torch")

from test_parallel_mpi import LAUNCHER, ROOT, _env  # noqa: E402

pytestmark = pytest.mark.skipif(LAUNCHER is None, reason="needs mpirun and mpi4py, outside an MPI launch")
WORKER = os.path.join(ROOT, "tests", "_parallel_nccl_worker.py")


def _run(ranks, mode, **env):
    done = subprocess.run(LAUNCHER + ["-n", str(ranks), sys.executable, WORKER, mode], cwd=ROOT,
                          env=dict(_env(), **env), capture_output=True, text=True, timeout=600)
    assert done.returncode == 0 and "OK" in done.stdout, done.stdout[-3000:] + done.stderr[-3000:]
    return done.stdout, done.stderr


@pytest.mark.parametrize("ranks", [2, 3, 5])
def test_nccl_transport_equals_alltoallv_bit_for_bit(ranks):
    out, _ = _run(ranks, "equal", PHRIKE_MPI_TRANSPOSE="nccl")
    assert f"{ranks} ranks, nccl (torch.distributed gloo" in out


def test_fallback_when_a_rank_cannot_start_the_group():
    out, _ = _run(3, "fallback", PHRIKE_NCCL_TEST_FAIL_RANK="1")
    assert out.count("warning: transpose 'nccl' is unavailable") == 1  # once, on the root
    assert "alltoallv" in out and "rank 1: forced failure" in out


def test_fallback_when_init_fails_after_the_preflight():
    """One rank never joins: the others time out in init and all end on alltoallv."""
    out, _ = _run(3, "fallback", PHRIKE_NCCL_TEST_FAIL_RANK="2:init", PHRIKE_NCCL_TIMEOUT="6")
    assert out.count("warning: transpose 'nccl' is unavailable") == 1


def test_one_rank_needs_no_group():
    from phrike.parallel import Comm
    import torch

    comm = Comm.serial()
    comm.transpose = "nccl"
    send = torch.arange(6.0)
    recv = torch.zeros(6)
    comm.exchange(send, [3], recv, [3], 2)
    assert torch.equal(recv, send)
    comm.init_transport(torch.device("cpu"))
    assert comm.transpose == "nccl"


def test_unknown_transport_is_rejected():
    from phrike.parallel import Comm

    with pytest.raises(ValueError):
        Comm(None, transpose="rdma")
