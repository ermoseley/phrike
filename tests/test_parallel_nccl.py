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


@pytest.mark.parametrize("batch,overlap", [("1", "1"), ("2", "1"), ("1", "0")])
def test_pipelined_batches_equal_the_blocking_exchange(batch, overlap):
    """Batches in flight behind the transforms give the same bits as one exchange at a time."""
    _run(3, "equal", PHRIKE_MPI_TRANSPOSE="nccl", PHRIKE_TEST_FFT_BATCH=batch, PHRIKE_FFT_OVERLAP=overlap)


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


def test_auto_transport_resolves_by_device():
    """The default takes nccl only for several CUDA ranks; everything else is alltoallv."""
    from phrike.parallel import Comm
    import torch

    comm = Comm.serial()
    assert comm.transpose == "auto"
    assert comm.resolve_transport(torch.device("cpu")) == "alltoallv"
    comm = Comm.serial()
    assert comm.resolve_transport(torch.device("cuda:0")) == "alltoallv"  # one rank exchanges nothing
    # An exchange resolves it too.
    comm = Comm.serial()
    recv = torch.zeros(4)
    comm.exchange(torch.arange(4.0), [2], recv, [2], 2)
    assert torch.equal(recv, torch.arange(4.0))
    assert Comm(None, transpose="p2p").resolve_transport(torch.device("cuda:0")) == "p2p"


def test_gpu_placement_check_refuses_shared_gpus_and_disables_p2p_when_isolated(monkeypatch):
    from phrike.parallel.comm import Comm

    comm = Comm.serial()
    shared = [(None, "n1", "cuda", ("A", ("A", "B"))), (None, "n1", "cuda", ("A", ("A", "B")))]
    assert "share GPU A" in comm._check_placement(shared)[0]
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
    visible = [(None, "n1", "cuda", ("A", ("A", "B"))), (None, "n1", "cuda", ("B", ("A", "B")))]
    assert comm._check_placement(visible) == [] and "NCCL_P2P_DISABLE" not in os.environ
    isolated = [(None, "n1", "cuda", ("A", ("A",))), (None, "n1", "cuda", ("B", ("B",))),
                (None, "n2", "cuda", ("A", ("A",)))]
    assert comm._check_placement(isolated) == [] and os.environ.get("NCCL_P2P_DISABLE") == "1"
    monkeypatch.delenv("NCCL_P2P_DISABLE", raising=False)
