"""One rank of the nccl-transport checks launched by ``test_parallel_nccl.py``.

    mpirun -n 3 python tests/_parallel_nccl_worker.py equal|fallback

``equal``: with the nccl transport (Gloo on the CPU) the distributed FFT and a
few solver steps equal the ``alltoallv`` result bit for bit, with the batches
pipelined behind the exchange (the default) and with ``PHRIKE_FFT_OVERLAP=0``;
``PHRIKE_TEST_FFT_BATCH`` sets the fields per batch.  ``fallback``:
with the group forced to fail on one rank every rank ends up on ``alltoallv``,
warns once on the root, and still computes the same answer.  Prints ``OK`` on
the root when every check passed; any failure raises, which aborts the job.
"""

import os
import sys

import numpy as np
import torch

from phrike.parallel import Comm
from phrike.parallel.comm import _import_mpi
from phrike.parallel.compressible import CompressibleSlabStepper
from phrike.parallel.fields import random_vector_spectrum
from phrike.parallel.grid import SlabGrid3D
from phrike.parallel.incompressible import IncompressibleSlabStepper

TWO_PI = 2.0 * np.pi
SHAPE = (24, 20, 16)
BATCH = int(os.environ.get("PHRIKE_TEST_FFT_BATCH", "4"))
DEPTHS = []  # pipeline depth of every grid built


def _grid(comm, precision):
    grid = SlabGrid3D(*SHAPE, TWO_PI, TWO_PI, TWO_PI, comm=comm, device="cpu", precision=precision,
                      fft_batch=BATCH)
    DEPTHS.append(grid.fft._depth())
    return grid


def results(comm, precision):
    """FFT round trip, 3 incompressible-MHD steps and 2 compressible-MHD steps."""
    grid = _grid(comm, precision)
    g = torch.Generator().manual_seed(5 + comm.rank)
    f = torch.randn((3,) + grid.decomp.physical_shape, generator=g, dtype=torch.float64).to(grid.rdtype)
    spectrum = grid.forward(f, normalized=True)
    out = {"spectrum": spectrum, "round_trip": grid.inverse(spectrum)}

    st = IncompressibleSlabStepper(grid, 0.02, 0.03, mhd=True, triton=False)
    st.set_spectrum(torch.cat([random_vector_spectrum(grid, 1, 1.0, 4.0, 1.0),
                               random_vector_spectrum(grid, 2, 1.0, 4.0, 0.3)]))
    for _ in range(3):
        st.step(dt=2e-3)
    out["imhd"] = st.uhat.clone()

    X, Y, Z = grid.xyz_mesh()
    rho = 1.0 + 0.2 * torch.cos(X) * torch.cos(Z)
    u = [torch.sin(X) * torch.cos(Y), -torch.cos(X) * torch.sin(Y), 0.3 * torch.sin(Z)]
    B = [torch.sin(Z), torch.sin(X), torch.cos(Y)]
    e = 2.0 / (5.0 / 3.0 - 1.0) + 0.5 * rho * sum(c * c for c in u) + 0.5 * sum(c * c for c in B)
    ch = CompressibleSlabStepper(grid, "mhd", 5.0 / 3.0, 1e-10, 1e-12, triton=False)
    ch.seed(torch.stack([rho] + [rho * c for c in u] + [e] + B))
    for _ in range(2):
        ch.step(1e-3, "rk4")
    out["mhd"] = ch.physical()
    return out


def same(a, b):
    return all(torch.equal(a[k], b[k]) for k in a)


def agree(mpi, ok, what):
    everyone = mpi.allgather(bool(ok))
    if not all(everyone):
        raise SystemExit(f"{what}: failed on ranks {[r for r, v in enumerate(everyone) if not v]}")


def main(mode):
    mpi = _import_mpi().COMM_WORLD
    reference = Comm(mpi, transpose="alltoallv")
    for precision in ("double", "single"):
        want = results(reference, precision)
        if mode == "equal":
            comm = Comm.auto()  # transpose from PHRIKE_MPI_TRANSPOSE
            assert comm.transpose_requested == "nccl", "run with PHRIKE_MPI_TRANSPOSE=nccl"
            got = results(comm, precision)
            agree(mpi, comm.transpose == "nccl", "nccl stayed active")
            overlap = os.environ.get("PHRIKE_FFT_OVERLAP", "1") != "0"
            agree(mpi, DEPTHS[-2:] == [1, 2 if overlap else 1], "the pipeline follows the transport")
            agree(mpi, same(want, got), f"nccl equals alltoallv ({precision})")
            agree(mpi, comm.bytes_sent > 0, "nccl counted its traffic")
        else:
            comm = Comm(mpi, transpose="nccl")
            got = results(comm, precision)
            agree(mpi, comm.transpose == "alltoallv", "fell back to alltoallv on every rank")
            agree(mpi, bool(comm.transport_note), "fallback has a reason")
            agree(mpi, same(want, got), f"fallback equals alltoallv ({precision})")
            if comm.is_root and precision == "double":
                print(f"fallback reason: {comm.transport_note}")
                print(comm.describe())
    if mode == "equal" and mpi.Get_rank() == 0:
        print(comm.describe())
    comm.close()
    if mpi.Get_rank() == 0:
        print("OK")


if __name__ == "__main__":
    main(sys.argv[1])
