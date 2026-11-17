"""Check whether the MPI library can send and receive GPU memory directly.

    srun -n 2 python -m scripts.check_gpu_aware_mpi      # or mpirun -n 2

The distributed solvers hand device pointers to MPI when
``parallel.gpu_aware_mpi: true`` (or ``PHRIKE_GPU_AWARE_MPI=1``).  An MPI
without CUDA support crashes on that, usually with a segmentation fault, so
the default is to stage through pinned host memory.  This script tries the
direct path on a small buffer and then times both paths.  If it prints
"device buffers work", turn the option on; if it dies after "trying device
buffers", leave it off (and see docs/parallel.rst for how to get a
GPU-aware MPI).  If neither path is usable or fast, ``parallel.transpose: nccl``
(``PHRIKE_MPI_TRANSPOSE=nccl``) exchanges device buffers with NCCL and needs no
MPI CUDA support.
"""

import sys
import time

import torch
from mpi4py import MPI

from phrike.parallel.comm import Comm


def main() -> int:
    world = MPI.COMM_WORLD
    rank, size = world.Get_rank(), world.Get_size()
    if size < 2:
        print("run this with at least two ranks")
        return 2
    if not torch.cuda.is_available():
        print("no CUDA device visible to this rank")
        return 2
    staged = Comm(world, gpu_aware=False)
    device = staged.pick_device("cuda")
    if rank == 0:
        print(f"{MPI.Get_library_version().splitlines()[0]}", flush=True)
        print(f"{size} ranks; rank 0 on {device} ({torch.cuda.get_device_name(torch.device(device))})", flush=True)

    def exchange(comm, n):
        """Every rank sends n rows to every rank; returns (seconds, correct)."""
        send = torch.arange(n * size, dtype=torch.float32, device=device) + 1000.0 * rank
        recv = torch.empty_like(send)
        torch.cuda.synchronize()
        world.Barrier()
        start = time.perf_counter()
        comm.exchange(send, [n] * size, recv, [n] * size, unit=1)
        torch.cuda.synchronize()
        elapsed = world.allreduce(time.perf_counter() - start, op=MPI.MAX)
        expected = torch.cat([torch.arange(n, dtype=torch.float32, device=device) + rank * n + 1000.0 * p
                              for p in range(size)])
        ok = world.allreduce(bool(torch.equal(recv, expected)), op=MPI.LAND)
        return elapsed, ok

    _, ok = exchange(staged, 1024)
    if rank == 0:
        print(f"host staging: {'correct' if ok else 'WRONG DATA'}", flush=True)
        print("trying device buffers (a crash right after this line means MPI is not GPU-aware) ...", flush=True)
    direct = Comm(world, gpu_aware=True)
    _, ok_direct = exchange(direct, 1024)
    if rank == 0:
        print(f"device buffers work: {'correct data' if ok_direct else 'WRONG DATA - do not enable'}", flush=True)
    if not (ok and ok_direct):
        return 1
    n = (64 * 1024 * 1024) // (4 * size)  # about 64 MB sent per rank
    for label, comm in (("host staging", staged), ("device buffers", direct)):
        exchange(comm, n)  # warm up
        elapsed = min(exchange(comm, n)[0] for _ in range(3))
        if rank == 0:
            print(f"  {label:15s} {64.0 / elapsed:8.0f} MB/s per rank", flush=True)
    if rank == 0:
        print("set parallel.gpu_aware_mpi: true (or PHRIKE_GPU_AWARE_MPI=1) to use device buffers", flush=True)
        print("or use parallel.transpose: nccl (PHRIKE_MPI_TRANSPOSE=nccl), which needs no MPI CUDA support;"
              " compare with python -m scripts.bench_parallel --transpose nccl", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
