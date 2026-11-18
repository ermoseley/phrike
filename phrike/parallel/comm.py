"""MPI layer of the slab-decomposed solvers.

``Comm`` wraps an ``mpi4py`` communicator and degrades to a serial stand-in
(one rank, no ``mpi4py`` import) so the same solver code runs on a laptop, on
one GPU and under ``mpirun``/``srun`` with one rank per GPU.

Only three kinds of traffic exist:

* ``exchange``: the all-to-all of a distributed FFT transpose.  Buffers are
  1-D real Torch tensors.  Three transports exist (``transpose``; the default
  ``auto`` takes ``nccl`` for several CUDA ranks and ``alltoallv`` otherwise):
  ``alltoallv`` and ``p2p`` hand the buffers to MPI by address, so a
  CUDA-aware MPI moves device memory directly; otherwise the data is staged
  through pinned host memory.  Counts are given in rows of ``unit`` reals and
  sent with a contiguous derived datatype, which keeps every count and
  displacement far below 2**31 for any grid.  ``nccl`` moves the device
  buffers with ``torch.distributed`` (NCCL on CUDA tensors, Gloo on CPU
  tensors), which needs neither a CUDA-aware MPI nor host staging; MPI is
  used only to start it and to agree that it works.
* ``allreduce``: a handful of float64 scalars per step (wave speed, counts).
* ``bcast``/``gather`` of small Python objects on the output path.
"""

from __future__ import annotations

import atexit
import os
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch is required by the solvers
    torch = None  # type: ignore


_LAUNCHER_ENV = (
    "OMPI_COMM_WORLD_SIZE", "PMI_RANK", "PMI_SIZE", "PMIX_RANK", "PALS_RANKID",
    "MPI_LOCALRANKID", "MV2_COMM_WORLD_RANK",
)


def launched_under_mpi() -> bool:
    """True when the process looks like one task of an MPI launch.

    Importing ``mpi4py`` calls ``MPI_Init``, which fails on some login nodes,
    so serial runs never import it.  ``PHRIKE_MPI=1`` forces MPI and
    ``PHRIKE_MPI=0`` forces the serial stand-in.
    """
    flag = os.environ.get("PHRIKE_MPI")
    if flag is not None:
        return flag.strip().lower() not in ("0", "false", "no", "off", "")
    if any(name in os.environ for name in _LAUNCHER_ENV):
        return True
    try:
        return int(os.environ.get("SLURM_STEP_NUM_TASKS", "1")) > 1
    except ValueError:
        return False


def _import_mpi():
    """Import ``mpi4py.MPI``, asking for funneled threading before MPI starts.

    Only the main thread calls MPI.  mpi4py's default (``MPI_THREAD_MULTIPLE``)
    slows down or disables the fast paths of several MPI libraries.
    """
    import mpi4py

    if "mpi4py.MPI" not in sys.modules:
        mpi4py.rc.thread_level = "funneled"
    from mpi4py import MPI

    return MPI


def _env_flag(name: str) -> Optional[bool]:
    value = os.environ.get(name)
    if value is None or value.strip().lower() in ("", "auto"):
        return None
    return value.strip().lower() not in ("0", "false", "no", "off")


TRANSPORTS = ("alltoallv", "p2p", "nccl")


class _TorchDistributed:
    """The process-wide ``torch.distributed`` group behind the ``nccl`` transport.

    One default group exists per process, so it is shared by every ``Comm``
    and created on the first exchange (or by ``Comm.init_transport``).  A
    failed start is remembered, so that every later ``Comm`` falls back
    without trying again.
    """

    backend: Optional[str] = None  # "nccl" or "gloo" once the group is up
    owned = False                  # we created it, so we destroy it
    failed: Optional[str] = None   # why the transport is unavailable
    warned = False
    atexit_registered = False


_TD = _TorchDistributed


def _destroy_group() -> None:
    """Destroy the process group this module created (idempotent)."""
    if _TD.owned:
        _TD.owned = False
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                dist.destroy_process_group()
        except Exception:  # noqa: BLE001 - exit-time cleanup must not raise
            pass
    _TD.backend = None


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host if host in ("127.0.0.1", "localhost") else "", 0))
        return int(sock.getsockname()[1])


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return float(default)


class Timers:
    """Accumulated wall-clock sections; ``sync`` makes GPU sections honest."""

    def __init__(self, enabled: bool = True, sync: bool = False):
        self.enabled = bool(enabled)
        self.sync = bool(sync)
        self.total: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}

    def _now(self) -> float:
        if self.sync and torch is not None and torch.cuda.is_available():
            # The current stream only: an exchange in flight on NCCL's stream must
            # not be waited for here, or timing a run would undo its overlap.
            torch.cuda.current_stream().synchronize()
        return time.perf_counter()

    @contextmanager
    def __call__(self, name: str):
        if not self.enabled:
            yield
            return
        start = self._now()
        try:
            yield
        finally:
            self.total[name] = self.total.get(name, 0.0) + (self._now() - start)
            self.calls[name] = self.calls.get(name, 0) + 1

    def reset(self) -> None:
        self.total.clear()
        self.calls.clear()


class _Completed:
    """Handle of an exchange that finished inside ``exchange_begin``."""

    overlapped = False

    def wait(self) -> None:
        return None


class _InFlight:
    """Handle of an all-to-all running on the ``torch.distributed`` group.

    Until ``wait`` returns, the send buffer must not be written and the
    receive buffer must not be read.  On CUDA ``wait`` orders the current
    stream after the exchange without blocking the host, so kernels issued
    between ``exchange_begin`` and ``wait`` run while the data moves.
    """

    overlapped = True

    def __init__(self, work, timers: "Timers"):
        self._work = work
        self._timers = timers

    def wait(self) -> None:
        work, self._work = self._work, None
        if work is not None:
            with self._timers("mpi_exchange"):  # with ``sync``: the time not hidden behind compute
                work.wait()


_DONE = _Completed()


class Comm:
    """An MPI communicator, or a one-rank stand-in when ``mpi_comm`` is None."""

    def __init__(self, mpi_comm: Any = None, gpu_aware: Optional[bool] = None,
                 transpose: Optional[str] = None):
        self.mpi = mpi_comm
        self.MPI = None
        if mpi_comm is None:
            self.rank, self.size = 0, 1
        else:
            self.MPI = _import_mpi()
            self.rank, self.size = int(mpi_comm.Get_rank()), int(mpi_comm.Get_size())
        if gpu_aware is None:
            gpu_aware = _env_flag("PHRIKE_GPU_AWARE_MPI")
        if gpu_aware is None:
            # Cray MPICH announces device support through this variable; other
            # libraries cannot be probed safely in-process (see
            # ``python -m scripts.check_gpu_aware_mpi``), so the default is host staging.
            gpu_aware = os.environ.get("MPICH_GPU_SUPPORT_ENABLED", "0").strip() == "1"
        self.gpu_aware = bool(gpu_aware)
        transpose = (transpose or os.environ.get("PHRIKE_MPI_TRANSPOSE") or "auto").lower()
        if transpose != "auto" and transpose not in TRANSPORTS:
            raise ValueError("transpose must be 'auto', 'alltoallv', 'p2p' or 'nccl'")
        # "auto" becomes a transport once the device is known (``resolve_transport``).
        self.transpose = transpose
        self.transpose_requested = transpose
        self.transport_note = ""  # why a requested transport is not in use
        self.timers = Timers(enabled=False)
        self._row_types: Dict[Any, Any] = {}
        self._host: Dict[Any, Any] = {}
        self.bytes_sent = 0

    # --- construction -------------------------------------------------------

    @classmethod
    def serial(cls) -> "Comm":
        return cls(None)

    @classmethod
    def world(cls, **kwargs) -> "Comm":
        return cls(_import_mpi().COMM_WORLD, **kwargs)

    @classmethod
    def auto(cls, **kwargs) -> "Comm":
        """``world`` under an MPI launcher, otherwise the serial stand-in."""
        if launched_under_mpi():
            return cls.world(**kwargs)
        return cls(None, **kwargs)

    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def sub(self, size: int) -> Optional["Comm"]:
        """Communicator over the first ``size`` ranks (collective); ``None`` on the others.

        Returns ``self`` when ``size`` covers every rank.  A real sub-communicator
        is not COMM_WORLD, so it never uses the ``nccl`` transport: it exchanges
        with ``alltoallv`` (or ``p2p`` if that was requested) and shares this
        communicator's timers.
        """
        size = int(size)
        if size < 1:
            raise ValueError("a sub-communicator needs at least one rank")
        if self.mpi is None or size >= self.size:
            return self
        color = 0 if self.rank < size else self.MPI.UNDEFINED
        mpi = self.mpi.Split(color, self.rank)
        if color == self.MPI.UNDEFINED:
            return None
        child = Comm(mpi, gpu_aware=self.gpu_aware,
                     transpose="p2p" if self.transpose == "p2p" else "alltoallv")
        child.timers = self.timers
        return child

    # --- devices ------------------------------------------------------------

    def node_rank(self) -> int:
        """Rank of this process among the ranks that share its node."""
        if self.mpi is None:
            return 0
        shared = self.mpi.Split_type(self.MPI.COMM_TYPE_SHARED, key=self.rank)
        try:
            return int(shared.Get_rank())
        finally:
            shared.Free()

    def pick_device(self, backend: str = "auto") -> str:
        """Torch device for this rank: one rank per visible GPU on each node."""
        backend = (backend or "auto").lower()
        if backend in ("cpu", "numpy"):
            return "cpu"
        if backend in ("mps", "metal"):
            return "mps"
        cuda = torch is not None and torch.cuda.is_available()
        if backend in ("cuda", "torch", "auto") and cuda:
            count = torch.cuda.device_count()
            index = self.node_rank() % count
            if self.mpi is not None:
                # With one GPU bound per task every rank sees "cuda:0", so
                # compare physical devices rather than local indices.
                host = self.MPI.Get_processor_name()
                try:
                    uuid = str(torch.cuda.get_device_properties(index).uuid)
                except Exception:  # pragma: no cover - older torch
                    uuid = f"{host}:{index}"
                devices = self.mpi.allgather((host, uuid))
                distinct = len(set(devices))
                if self.is_root and distinct < len(devices):
                    print(f"[phrike.parallel] warning: {len(devices)} ranks share {distinct} GPUs; "
                          f"run one rank per GPU for full speed.", flush=True)
            return f"cuda:{index}"
        if backend == "cuda":
            raise RuntimeError("CUDA was requested but is not available")
        return "cpu"

    # --- small collectives --------------------------------------------------

    def allreduce(self, values: Sequence[float], op: str = "sum") -> np.ndarray:
        """Reduce float64 scalars over all ranks (``sum``, ``max`` or ``min``)."""
        local = np.asarray(values, dtype=np.float64).reshape(-1)
        if self.mpi is None:
            return local.copy()
        out = np.empty_like(local)
        mpi_op = {"sum": self.MPI.SUM, "max": self.MPI.MAX, "min": self.MPI.MIN}[op]
        with self.timers("mpi_reduce"):
            self.mpi.Allreduce(local, out, op=mpi_op)
        return out

    def sum(self, value: float) -> float:
        return float(self.allreduce([value], "sum")[0])

    def max(self, value: float) -> float:
        return float(self.allreduce([value], "max")[0])

    def min(self, value: float) -> float:
        return float(self.allreduce([value], "min")[0])

    def bcast(self, obj: Any, root: int = 0) -> Any:
        return obj if self.mpi is None else self.mpi.bcast(obj, root=root)

    def gather(self, obj: Any, root: int = 0) -> Optional[List[Any]]:
        return [obj] if self.mpi is None else self.mpi.gather(obj, root=root)

    def barrier(self) -> None:
        if self.mpi is not None:
            self.mpi.Barrier()

    def abort(self, code: int = 1) -> None:
        if self.mpi is not None:
            self.mpi.Abort(code)
        raise SystemExit(code)

    def install_abort_hook(self) -> None:
        """Abort every rank when one raises; a lone traceback would hang the job."""
        if self.mpi is None:
            return
        comm = self

        def hook(kind, value, tb):
            sys.stderr.write(f"[phrike.parallel] rank {comm.rank} failed:\n")
            traceback.print_exception(kind, value, tb)
            sys.stderr.flush()
            comm.mpi.Abort(1)

        sys.excepthook = hook

    # --- the NCCL transport -------------------------------------------------

    def _fall_back(self, reason: str) -> None:
        self.transpose = "alltoallv"
        self.transport_note = reason
        if self.is_root and not _TD.warned:
            _TD.warned = True
            print(f"[phrike.parallel] warning: transpose 'nccl' is unavailable ({reason}); "
                  f"using 'alltoallv' on every rank.", flush=True)

    def _preflight(self, kind: str) -> Optional[str]:
        """Rank-local reason why torch.distributed cannot be started, or None."""
        fail_rank = os.environ.get("PHRIKE_NCCL_TEST_FAIL_RANK", "")  # tests only
        if fail_rank.strip() and int(fail_rank.split(":")[0]) == self.rank \
                and not fail_rank.endswith(":init"):
            return "forced failure (PHRIKE_NCCL_TEST_FAIL_RANK)"
        try:
            import torch.distributed as dist
        except Exception as exc:  # noqa: BLE001
            return f"torch.distributed cannot be imported ({exc})"
        if not dist.is_available():
            return "this torch has no torch.distributed"
        if kind == "cuda":
            if not torch.cuda.is_available() or not dist.is_nccl_available():
                return "this torch has no NCCL or no CUDA device"
        elif kind != "cpu":
            return f"no torch.distributed backend for {kind} tensors"
        elif not dist.is_gloo_available():
            return "this torch has no Gloo backend"
        if dist.is_initialized():
            if dist.get_rank() != self.rank or dist.get_world_size() != self.size:
                return "a torch.distributed group with different ranks already exists"
        elif self.MPI.Comm.Compare(self.mpi, self.MPI.COMM_WORLD) not in (self.MPI.IDENT, self.MPI.CONGRUENT):
            return "the MPI communicator is not COMM_WORLD, so torch ranks would differ"
        return None

    def resolve_transport(self, device) -> str:
        """Turn the ``auto`` transport into a concrete one for tensors on ``device``.

        Several CUDA ranks use ``nccl``: on one node it measured 170 to 300
        GB/s per rank against 5 to 8 GB/s for host-staged MPI, and it needs no
        CUDA support in MPI.  CPU runs, a single rank, and runs that ask for
        device buffers in MPI (``gpu_aware``) use ``alltoallv``.
        """
        if self.transpose == "auto":
            cuda = torch.device(device).type == "cuda"
            use_nccl = cuda and self.mpi is not None and self.size > 1 and not self.gpu_aware
            self.transpose = "nccl" if use_nccl else "alltoallv"
        return self.transpose

    def init_transport(self, device) -> None:
        """Start the ``nccl`` transport for tensors on ``device``; collective.

        Does nothing for the other transports, on one rank, or when the group
        is already up.  Every rank must call it (``exchange`` does so on its
        first use).  If the group cannot start on any rank, all ranks switch
        to ``alltoallv`` together.
        """
        self.resolve_transport(device)
        if self.transpose != "nccl" or self.mpi is None or self.size == 1:
            return
        if _TD.failed is not None:
            self._fall_back(_TD.failed)
            return
        if _TD.backend is not None:
            return
        import datetime

        device = torch.device(device)
        kind = device.type
        backend = "nccl" if kind == "cuda" else "gloo"
        # Rank-local checks first and an MPI agreement before any torch call:
        # init_process_group blocks until every rank joins, so a rank that
        # cannot start must be known to all before they enter it.
        problem = self._preflight(kind)
        votes = self.mpi.allgather((problem, self.MPI.Get_processor_name(), kind))
        problems = [f"rank {r}: {v[0]}" for r, v in enumerate(votes) if v[0] is not None]
        if not problems and len({v[2] for v in votes}) > 1:
            problems = ["ranks use different device kinds"]
        if problems:
            _TD.failed = problems[0]
            self._fall_back(_TD.failed)
            return
        init_method = None
        if self.is_root:
            try:
                master = os.environ.get("MASTER_ADDR")
                same_node = len({v[1] for v in votes}) == 1
                host = master or ("127.0.0.1" if same_node else votes[0][1])
                port = os.environ.get("MASTER_PORT") or _free_port(host)
                init_method = f"tcp://{host}:{int(port)}"
            except Exception:  # noqa: BLE001 - reported as an init failure below
                init_method = None
        init_method = self.mpi.bcast(init_method, root=0)
        import torch.distributed as dist

        error = None
        if init_method is None:
            error = "rank 0 could not choose a rendezvous address"
        elif dist.is_initialized():
            _TD.owned = False  # someone else's group: use it, never destroy it
        else:
            fail_rank = os.environ.get("PHRIKE_NCCL_TEST_FAIL_RANK", "")
            try:
                if fail_rank.endswith(":init") and int(fail_rank.split(":")[0]) == self.rank:
                    raise RuntimeError("forced init failure (PHRIKE_NCCL_TEST_FAIL_RANK)")
                options = {"backend": backend, "init_method": init_method, "rank": self.rank,
                           "world_size": self.size,
                           "timeout": datetime.timedelta(seconds=_env_float("PHRIKE_NCCL_TIMEOUT", 600.0))}
                if kind == "cuda":
                    try:
                        dist.init_process_group(device_id=device, **options)
                    except TypeError:  # older torch without device_id
                        dist.init_process_group(**options)
                else:
                    dist.init_process_group(**options)
                _TD.owned = True
            except Exception as exc:  # noqa: BLE001
                error = f"init_process_group({backend}) failed: {type(exc).__name__}: {exc}"
        if error is None:
            try:
                # One small collective now, so that a broken NCCL shows up here.
                probe = torch.arange(self.size, dtype=torch.float32, device=device) + self.rank * self.size
                out = torch.empty_like(probe)
                dist.all_to_all_single(out, probe)
                if kind == "cuda":
                    torch.cuda.synchronize(device)
                expected = torch.arange(self.size, dtype=torch.float32) * self.size + self.rank
                if not torch.equal(out.cpu(), expected):
                    error = "the NCCL probe exchange returned wrong data"
            except Exception as exc:  # noqa: BLE001
                error = f"probe exchange failed: {type(exc).__name__}: {exc}"
        verdicts = self.mpi.allgather(error)
        bad = [f"rank {r}: {e}" for r, e in enumerate(verdicts) if e is not None]
        if bad:
            _destroy_group()
            _TD.failed = bad[0]
            self._fall_back(_TD.failed)
            return
        _TD.backend = backend
        if _TD.owned and not _TD.atexit_registered:
            _TD.atexit_registered = True
            atexit.register(_destroy_group)  # runs before mpi4py finalizes MPI

    def _use_torch_distributed(self, device) -> bool:
        """Whether this exchange goes through the process group (may start it)."""
        if self.transpose == "auto":
            self.resolve_transport(device)
        if self.transpose != "nccl" or self.mpi is None or self.size == 1:
            return False
        if _TD.backend is None:
            self.init_transport(device)
            if self.transpose != "nccl":
                return False
        return _TD.backend == {"cuda": "nccl", "cpu": "gloo"}.get(device.type)

    def close(self) -> None:
        """Free the derived datatypes and destroy the torch.distributed group."""
        for datatype in self._row_types.values():
            try:
                datatype.Free()
            except Exception:  # noqa: BLE001
                pass
        self._row_types.clear()
        _destroy_group()

    # --- the transpose ------------------------------------------------------

    def _row_type(self, dtype, unit: int):
        key = (str(dtype), int(unit))
        cached = self._row_types.get(key)
        if cached is None:
            base = self.MPI.FLOAT if dtype == torch.float32 else self.MPI.DOUBLE
            cached = base.Create_contiguous(int(unit))
            cached.Commit()
            self._row_types[key] = cached
        return cached

    def _mpi_buffer(self, tensor):
        """MPI buffer over a tensor's memory (host or device), by address."""
        nbytes = tensor.numel() * tensor.element_size()
        factory = getattr(self.MPI, "buffer", None) or getattr(self.MPI, "memory")
        return factory.fromaddress(tensor.data_ptr(), nbytes)

    def _host_buffer(self, key: str, n: int, dtype):
        buf = self._host.get((key, str(dtype)))
        if buf is None or buf.numel() < n:
            try:
                buf = torch.empty(n, dtype=dtype, pin_memory=True)
            except Exception:
                buf = torch.empty(n, dtype=dtype)
            self._host[(key, str(dtype))] = buf
        return buf

    def exchange(self, send, scounts: Sequence[int], recv, rcounts: Sequence[int],
                 unit: int) -> None:
        """All-to-all exchange of 1-D real tensors.

        ``scounts[q]`` rows go to rank ``q`` and ``rcounts[p]`` rows arrive from
        rank ``p``, consecutively in rank order; a row is ``unit`` reals.
        """
        self.exchange_begin(send, scounts, recv, rcounts, unit).wait()

    def can_overlap(self, device) -> bool:
        """Whether ``exchange_begin`` returns before the data has moved (the nccl transport)."""
        return self._use_torch_distributed(torch.device(device))

    def exchange_begin(self, send, scounts: Sequence[int], recv, rcounts: Sequence[int],
                       unit: int, overlap: bool = True):
        """Start the exchange and return a handle whose ``wait`` completes it.

        With the ``nccl`` transport (and ``overlap``) the exchange runs in the
        background: work issued before ``wait`` overlaps with it, provided it
        leaves ``send`` and ``recv`` alone.  The MPI transports finish here
        and return a completed handle.
        """
        nsend = int(sum(scounts)) * unit
        nrecv = int(sum(rcounts)) * unit
        if send.numel() < nsend or recv.numel() < nrecv:
            raise ValueError("exchange buffers are smaller than the requested counts")
        if self.mpi is None:
            if nsend != nrecv:
                raise ValueError("serial exchange needs matching send and receive sizes")
            recv[:nrecv].copy_(send[:nsend])
            return _DONE
        if len(scounts) != self.size or len(rcounts) != self.size:
            raise ValueError("exchange counts must have one entry per rank")
        me = self.rank
        scounts = np.asarray(scounts, dtype=np.int64)
        rcounts = np.asarray(rcounts, dtype=np.int64)
        if scounts[me] != rcounts[me]:
            raise ValueError("a rank's block to itself must have matching counts")
        sdispls = np.concatenate(([0], np.cumsum(scounts)[:-1]))
        rdispls = np.concatenate(([0], np.cumsum(rcounts)[:-1]))
        s0, r0, n0 = int(sdispls[me]) * unit, int(rdispls[me]) * unit, int(scounts[me]) * unit
        if self._use_torch_distributed(send.device):
            # The group's all-to-all takes one contiguous buffer per side, and
            # the self block sits inside it, so the group copies that block on
            # the device itself (a local copy, no extra buffer).  The exchange
            # is ordered after everything already issued on the current stream;
            # ``wait`` orders the stream after it, so ``recv`` is valid from then on.
            import torch.distributed as dist

            self.bytes_sent += (nsend - n0) * send.element_size()
            work = dist.all_to_all_single(recv[:nrecv], send[:nsend],
                                          output_split_sizes=(rcounts * unit).tolist(),
                                          input_split_sizes=(scounts * unit).tolist(), async_op=True)
            handle = _InFlight(work, self.timers)
            if not overlap:
                handle.wait()
            return handle
        # This rank's block to itself never goes through MPI or the host.
        if n0:
            recv[r0:r0 + n0].copy_(send[s0:s0 + n0])
        if self.size == 1:
            return _DONE
        on_device = send.device.type != "cpu"
        if send.device.type == "cuda":
            # MPI reads the buffer now; every kernel that writes it must be done.
            torch.cuda.synchronize(send.device)
        staged = on_device and not (self.gpu_aware and send.device.type == "cuda")
        if staged:
            with self.timers("mpi_stage"):
                hsend = self._host_buffer("s", nsend, send.dtype)
                if s0:
                    hsend[:s0].copy_(send[:s0])
                if nsend > s0 + n0:
                    hsend[s0 + n0:nsend].copy_(send[s0 + n0:nsend])
                hrecv = self._host_buffer("r", nrecv, recv.dtype)
            sbuf_t, rbuf_t = hsend, hrecv
        else:
            sbuf_t, rbuf_t = send, recv
        row = self._row_type(send.dtype, unit)
        sc, rc = scounts.copy(), rcounts.copy()
        sc[me] = rc[me] = 0
        self.bytes_sent += (nsend - n0) * send.element_size()
        with self.timers("mpi_exchange"):
            if self.transpose != "p2p":  # also nccl on a tensor the group cannot carry
                self.mpi.Alltoallv(
                    [self._mpi_buffer(sbuf_t), (sc.tolist(), sdispls.tolist()), row],
                    [self._mpi_buffer(rbuf_t), (rc.tolist(), rdispls.tolist()), row],
                )
            else:
                self._exchange_p2p(sbuf_t, sc, sdispls, rbuf_t, rc, rdispls, row, unit)
        if staged:
            with self.timers("mpi_stage"):
                if r0:
                    recv[:r0].copy_(hrecv[:r0])
                if nrecv > r0 + n0:
                    recv[r0 + n0:nrecv].copy_(hrecv[r0 + n0:nrecv])
        return _DONE

    def _exchange_p2p(self, send, scounts, sdispls, recv, rcounts, rdispls, row, unit) -> None:
        """The same exchange with one nonblocking message per peer."""
        me = self.rank
        requests = []
        keep = []
        for p in range(self.size):
            if p == me or rcounts[p] == 0:
                continue
            piece = recv[rdispls[p] * unit:(rdispls[p] + rcounts[p]) * unit]
            buf = self._mpi_buffer(piece)
            keep.append((piece, buf))
            requests.append(self.mpi.Irecv([buf, int(rcounts[p]), row], source=p, tag=77))
        for q in range(self.size):
            if q == me or scounts[q] == 0:
                continue
            piece = send[sdispls[q] * unit:(sdispls[q] + scounts[q]) * unit]
            buf = self._mpi_buffer(piece)
            keep.append((piece, buf))
            requests.append(self.mpi.Isend([buf, int(scounts[q]), row], dest=q, tag=77))
        self.MPI.Request.Waitall(requests)
        del keep

    # --- reporting ----------------------------------------------------------

    def describe(self) -> str:
        if self.mpi is None:
            return "serial (no MPI)"
        library = self.MPI.Get_library_version().splitlines()[0][:70]
        if self.transpose == "auto":
            return f"{self.size} ranks, transport chosen on the first exchange (nccl on CUDA); {library}"
        if self.transpose == "nccl":
            if _TD.backend is not None:
                detail = f"torch.distributed {_TD.backend}, no host staging"
            elif self.size == 1:
                detail = "unused on one rank"
            else:
                detail = "torch.distributed starts on the first exchange"
            return f"{self.size} ranks, nccl ({detail}); {library}"
        mode = "device buffers (GPU-aware MPI)" if self.gpu_aware else "host staging"
        note = f" [nccl requested but unavailable: {self.transport_note}]" if self.transport_note else ""
        return f"{self.size} ranks, {self.transpose}, {mode}{note}; {library}"
