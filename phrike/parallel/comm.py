"""MPI layer of the slab-decomposed solvers.

``Comm`` wraps an ``mpi4py`` communicator and degrades to a serial stand-in
(one rank, no ``mpi4py`` import) so the same solver code runs on a laptop, on
one GPU and under ``mpirun``/``srun`` with one rank per GPU.

Only three kinds of traffic exist:

* ``exchange``: the all-to-all of a distributed FFT transpose.  Buffers are
  1-D real Torch tensors.  They are handed to MPI by address, so a
  CUDA-aware MPI moves device memory directly; otherwise the data is staged
  through pinned host memory.  Counts are given in rows of ``unit`` reals and
  sent with a contiguous derived datatype, which keeps every count and
  displacement far below 2**31 for any grid.
* ``allreduce``: a handful of float64 scalars per step (wave speed, counts).
* ``bcast``/``gather`` of small Python objects on the output path.
"""

from __future__ import annotations

import os
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


class Timers:
    """Accumulated wall-clock sections; ``sync`` makes GPU sections honest."""

    def __init__(self, enabled: bool = True, sync: bool = False):
        self.enabled = bool(enabled)
        self.sync = bool(sync)
        self.total: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}

    def _now(self) -> float:
        if self.sync and torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()
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
        transpose = (transpose or os.environ.get("PHRIKE_MPI_TRANSPOSE") or "alltoallv").lower()
        if transpose not in ("alltoallv", "p2p"):
            raise ValueError("MPI transpose must be 'alltoallv' or 'p2p'")
        self.transpose = transpose
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
        nsend = int(sum(scounts)) * unit
        nrecv = int(sum(rcounts)) * unit
        if send.numel() < nsend or recv.numel() < nrecv:
            raise ValueError("exchange buffers are smaller than the requested counts")
        if self.mpi is None:
            if nsend != nrecv:
                raise ValueError("serial exchange needs matching send and receive sizes")
            recv[:nrecv].copy_(send[:nsend])
            return
        if len(scounts) != self.size or len(rcounts) != self.size:
            raise ValueError("exchange counts must have one entry per rank")
        me = self.rank
        scounts = np.asarray(scounts, dtype=np.int64)
        rcounts = np.asarray(rcounts, dtype=np.int64)
        if scounts[me] != rcounts[me]:
            raise ValueError("a rank's block to itself must have matching counts")
        sdispls = np.concatenate(([0], np.cumsum(scounts)[:-1]))
        rdispls = np.concatenate(([0], np.cumsum(rcounts)[:-1]))
        # This rank's block to itself never goes through MPI or the host.
        s0, r0, n0 = int(sdispls[me]) * unit, int(rdispls[me]) * unit, int(scounts[me]) * unit
        if n0:
            recv[r0:r0 + n0].copy_(send[s0:s0 + n0])
        if self.size == 1:
            return
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
            if self.transpose == "alltoallv":
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
        mode = "device buffers (GPU-aware MPI)" if self.gpu_aware else "host staging"
        return f"{self.size} ranks, {self.transpose}, {mode}; {library}"
