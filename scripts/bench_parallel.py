"""Throughput and weak-scaling benchmark of the distributed solvers.

    python -m scripts.bench_parallel --problem imhd_turb --points-per-rank 256
    srun -n 8  python -m scripts.bench_parallel --problem imhd_turb --points-per-rank 256
    srun -n 64 python -m scripts.bench_parallel --problem imhd_turb --points-per-rank 256

Weak scaling keeps the work per rank fixed: with P ranks the grid is the cube
whose side is the FFT-friendly even number nearest to
``points_per_rank * P**(1/3)`` (or pass ``--grid Nx Ny Nz``).  Perfect weak
scaling is a constant "Mzc/s/rank" (10^6 zone-cycles per second per rank) as P
grows, apart from the log N factor of the transforms.  FFT sizes with factors
of 3, 5 and 7 are slower than powers of two, which bends that curve;
``--pow2-per-rank N`` keeps every side a power of two instead: ``N^3`` points
per rank on the grid ``N x N x N`` whose sides are doubled in turn (z, y, x)
for each doubling of the rank count, e.g. 512^3, 512x512x1024, 512x1024x1024,
1024^3 for 1, 2, 4 and 8 ranks.  The timer breakdown
separates the local transforms (``fft_xy``, ``fft_z``), packing (``fft_pack``)
and the exchange (``mpi_exchange``, plus ``mpi_stage`` when MPI is given host
copies of device buffers).  ``--transpose nccl`` moves device buffers with
NCCL (Gloo on the CPU); the header names the transport in use and the
"exchange" line gives the effective bandwidth per rank (bytes sent per step
over the exchange time per step).  Under about 1 GB/s/rank the transport is
badly configured (see docs/parallel.rst).
``--forcing ou|white|constant_power`` adds large-scale forcing to the timed
steps (constant_power is incompressible only); its cost is reported as
``forcing`` (and ``forcing_kick`` for the compressible momentum/energy update).
``--hyper`` replaces the Laplacian dissipation of the incompressible solvers by
sixth-order hyperdissipation.

With the ``nccl`` transport the exchange of a batch of fields runs behind the
transforms of its neighbours.  The exchange line then reports the time the
step was blocked waiting for data, not the time the data took; ``--overlap 0``
runs one exchange at a time, which is the mode that measures the transport
(``--transform-only`` does only that: round trips of ``--fields`` fields, no
solver, a few seconds to start).

One JSON line per run is appended to ``--out`` for plotting. The bandwidth field
is null for pipelined transforms because the wait measures only unhidden exchange time.
"""

import argparse
import json
import sys
import time

import numpy as np
import torch

from phrike.parallel.comm import Comm
from phrike.parallel.driver import Simulation
from phrike.parallel.problems import COMPRESSIBLE, get_problem

TWO_PI = 2.0 * np.pi


def friendly(n: float) -> int:
    """Nearest even integer of the form 2^a 3^b 5^c 7^d."""
    best = None
    limit = int(2 * n) + 16
    a = 2
    while a <= limit:
        b = a
        while b <= limit:
            c = b
            while c <= limit:
                d = c
                while d <= limit:
                    if best is None or abs(d - n) < abs(best - n):
                        best = d
                    d *= 7
                c *= 5
            b *= 3
        a *= 2
    return int(best)


def pow2_grid(n: int, ranks: int):
    """(Nx, Ny, Nz) with ``n**3`` points per rank and power-of-two sides."""
    if n < 1 or n & (n - 1) or ranks & (ranks - 1):
        raise ValueError("--pow2-per-rank needs a power of two and a power-of-two rank count")
    sides, axis = [n, n, n], 2
    while ranks > 1:
        sides[axis] *= 2
        axis = (axis - 1) % 3
        ranks //= 2
    return tuple(sides)


def count_nodes(comm) -> int:
    if comm.mpi is None:
        return 1
    return len(set(comm.mpi.allgather(comm.MPI.Get_processor_name())))


def pipelines_batches(fft, fields: int) -> bool:
    """The NCCL stream can overlap only when a transform spans multiple batches."""
    return 0 < fft.batch < fields and fft._depth() > 1


def transform_only(args, comm, shape) -> int:
    """Round trips of the distributed transform alone: the transport without the solver."""
    from phrike.parallel.comm import Timers
    from phrike.parallel.grid import SlabGrid3D

    device = comm.pick_device(args.backend)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    comm.init_transport(torch.device(device))
    timers = Timers(enabled=True, sync=True)
    Nx, Ny, Nz = shape
    grid = SlabGrid3D(Nx, Ny, Nz, TWO_PI, TWO_PI, TWO_PI, comm=comm, device=device, precision=args.precision,
                      fft_batch=args.fft_batch, timers=timers,
                      fft_overlap=None if args.overlap is None else args.overlap == "1")
    gen = torch.Generator().manual_seed(1 + comm.rank)
    f = torch.randn((args.fields,) + grid.decomp.physical_shape, generator=gen, dtype=torch.float32).to(
        dtype=grid.rdtype, device=grid.device)
    S = grid.forward(f, normalized=True)
    del f

    def sync():
        if grid.device.type == "cuda":
            torch.cuda.synchronize()
        comm.barrier()

    for _ in range(max(args.warmup, 1)):
        S = grid.forward(grid.inverse(S), normalized=True)
    sync()
    timers.reset()
    comm.bytes_sent = 0
    start = time.perf_counter()
    for _ in range(args.steps):
        S = grid.forward(grid.inverse(S), normalized=True)
    sync()
    wall = comm.max(time.perf_counter() - start)
    names = comm.union_names(timers.total)
    sections = dict(zip(names, comm.allreduce([timers.total.get(n, 0.0) for n in names], "max")))
    sent = comm.sum(comm.bytes_sent) / comm.size / args.steps
    exchange = sections.get("mpi_exchange", 0.0) / args.steps
    exchange_gbs = sent / exchange / 1e9 if exchange > 0 else float("nan")
    overlapped = pipelines_batches(grid.fft, args.fields)
    nodes = count_nodes(comm)
    # One round trip is 2 * fields field transforms.
    per_field = wall / args.steps / (2 * args.fields)
    result = {
        "problem": "transform", "ranks": comm.size, "nodes": nodes, "grid": [Nx, Ny, Nz],
        "precision": args.precision, "fields": args.fields, "fft_batch": args.fft_batch,
        "transpose": comm.transpose, "gpu_aware_mpi": comm.gpu_aware, "overlapped": overlapped,
        "seconds_per_field_transform": per_field,
        "megapoints_per_second_per_rank": grid.n_points / comm.size / per_field / 1e6,
        "megabytes_sent_per_round_trip_per_rank": sent / 1e6,
        "exchange_gb_per_second_per_rank": None if overlapped else exchange_gbs,
        "seconds_by_section": {n: float(v) / args.steps for n, v in sections.items()}, "label": args.label,
    }
    if comm.is_root:
        print(f"transform {Nx}x{Ny}x{Nz} on {comm.size} rank(s) on {nodes} node(s), {args.precision}, "
              f"{args.fields} fields in batches of {args.fft_batch}, {comm.describe()}")
        print(f"  {1e3 * per_field:.3f} ms per field transform   "
              f"{result['megapoints_per_second_per_rank']:.1f} Mpoints/s/rank   "
              f"{sent / 1e6:.0f} MB sent per round trip per rank")
        if comm.size > 1:
            what = "blocked" if overlapped else "exchange"
            rate = "" if overlapped else f"{result['exchange_gb_per_second_per_rank']:.2f} GB/s per rank, "
            print(f"  exchange [{comm.transpose}{', behind the transforms' if overlapped else ''}]: {rate}"
                  f"{what} {exchange:.4f} s of {wall / args.steps:.4f} s per round trip")
        for name in names:
            print(f"    {name:14s} {sections[name] / args.steps:9.4f} s/round trip")
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    comm.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", default="imhd_turb")
    parser.add_argument("--points-per-rank", type=int, default=128, help="cube root of the grid points per rank")
    parser.add_argument("--pow2-per-rank", type=int, metavar="N",
                        help="N^3 points per rank on a grid with power-of-two sides")
    parser.add_argument("--grid", type=int, nargs=3, metavar=("NX", "NY", "NZ"), help="explicit grid instead")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3, help="untimed steps (kernel builds, FFT plans)")
    parser.add_argument("--precision", choices=["single", "double"], default="single")
    parser.add_argument("--scheme", default="rk4")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--fft-batch", type=int, default=1)
    parser.add_argument("--transpose", choices=["auto", "alltoallv", "p2p", "nccl"], default=None,
                        help="exchange transport (default: PHRIKE_MPI_TRANSPOSE, or auto: nccl on "
                             "several CUDA ranks, alltoallv otherwise)")
    parser.add_argument("--gpu-aware-mpi", choices=["0", "1"], default=None)
    parser.add_argument("--overlap", choices=["0", "1"], default=None,
                        help="1: pipeline the batches behind the exchange (default with nccl); 0: one at a time")
    parser.add_argument("--transform-only", action="store_true",
                        help="time forward + inverse transforms of --fields fields instead of solver steps")
    parser.add_argument("--fields", type=int, default=8, help="fields per round trip with --transform-only")
    parser.add_argument("--hyper", action="store_true",
                        help="sixth-order hyperdissipation instead of the Laplacians (incompressible)")
    parser.add_argument("--label", default="", help="free text copied to the output record")
    parser.add_argument("--no-triton", action="store_true", help="use the torch tier")
    parser.add_argument("--forcing", choices=["none", "ou", "white", "constant_power"], default="none",
                        help="large-scale forcing on 1 <= |n| <= 2.5 included in the timed steps")
    parser.add_argument("--out", default="bench_parallel.jsonl")
    args = parser.parse_args()

    aware = None if args.gpu_aware_mpi is None else args.gpu_aware_mpi == "1"
    comm = Comm.auto(gpu_aware=aware, transpose=args.transpose)
    comm.install_abort_hook()
    if args.grid:
        Nx, Ny, Nz = args.grid
    elif args.pow2_per_rank:
        Nx, Ny, Nz = pow2_grid(args.pow2_per_rank, comm.size)
    else:
        Nx = Ny = Nz = friendly(args.points_per_rank * comm.size ** (1.0 / 3.0))
    system = get_problem(args.problem).system
    if args.forcing == "constant_power" and system in COMPRESSIBLE:
        parser.error("--forcing constant_power needs an incompressible problem")
    config = {
        "grid": {"Nx": Nx, "Ny": Ny, "Nz": Nz, "Lx": TWO_PI, "Ly": TWO_PI, "Lz": TWO_PI,
                 "precision": args.precision},
        "physics": {"gamma": 5.0 / 3.0, "viscosity": 1e-3, "resistivity": 1e-3},
        "integration": {"t_end": 1e30, "cfl": 0.3 if system in COMPRESSIBLE else 0.5, "scheme": args.scheme},
        "initial_conditions": {"vrms": 0.3, "kmin": 1.0, "kmax": 4.0, "b_rms": 0.1, "b_kmax": 4.0,
                               "alpha": 0.0, "mach": 0.3},
        "io": {"outdir": "bench_parallel_out", "snapshots": False, "slices": False, "spectra": False},
        "monitoring": {"enabled": False},
        "parallel": {"fft_batch": args.fft_batch, "triton": not args.no_triton},
    }
    if args.overlap is not None:
        config["parallel"]["overlap"] = args.overlap == "1"
    if args.hyper:
        if system in COMPRESSIBLE:
            parser.error("--hyper needs an incompressible problem")
        # The cutoff sits at the retained k_max of the finest direction, as in a production run.
        kmax = (min(Nx, Ny, Nz) - 1) // 3
        config["physics"].update({"viscosity": 0.0, "resistivity": 0.0, "hyper_order": 6,
                                  "hyperviscosity": 30.0 / kmax ** 6, "hyperresistivity": 30.0 / kmax ** 6})
    if args.transform_only:
        return transform_only(args, comm, (Nx, Ny, Nz))
    if args.forcing != "none":
        config["forcing"] = {"type": args.forcing, "kmin": 1.0, "kmax": 2.5, "amplitude": 0.1,
                             "power": 1e-3, "correlation_time": 1.0, "seed": 1}
    # The Simulation binds this rank's CUDA device and then starts the transport.
    sim = Simulation(args.problem, config, comm=comm, backend=args.backend, quiet=True, profile=True)
    st, grid = sim.stepper, sim.grid

    def step():
        if sim.compressible:
            dt = sim._compressible_limit()
            st.step(dt, args.scheme)
            sim._compressible_kick(dt)  # no-op without forcing
        else:
            st.step()

    def sync():
        if grid.device.type == "cuda":
            torch.cuda.synchronize()
        comm.barrier()

    for _ in range(args.warmup):
        step()
    sync()
    sim.timers.reset()
    grid.fft.forward_fields = grid.fft.inverse_fields = 0
    comm.bytes_sent = 0
    if grid.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    for _ in range(args.steps):
        step()
    sync()
    wall = comm.max(time.perf_counter() - start)

    names = comm.union_names(sim.timers.total)
    worst = comm.allreduce([sim.timers.total.get(n, 0.0) for n in names], "max")
    sections = dict(zip(names, worst))
    # Bytes this rank sent over the time it spent in the exchange itself (staging copies excluded).
    exchange_seconds = sections.get("mpi_exchange", 0.0) / args.steps
    sent_per_step = comm.sum(comm.bytes_sent) / comm.size / args.steps
    exchange_gbs = sent_per_step / exchange_seconds / 1e9 if exchange_seconds > 0 else float("nan")
    peak = comm.max(torch.cuda.max_memory_allocated() / 2 ** 30 if grid.device.type == "cuda" else 0.0)
    per_rank = grid.n_points / comm.size
    overlapped = pipelines_batches(grid.fft, st.nflux if sim.compressible else st.nprod)
    nodes = count_nodes(comm)
    result = {
        "problem": args.problem, "system": system, "ranks": comm.size, "grid": [Nx, Ny, Nz],
        "precision": args.precision, "scheme": args.scheme, "forcing": args.forcing, "device": str(grid.device),
        "tier": "triton" if getattr(st, "_triton", None) is not None else "torch",
        "gpu_aware_mpi": comm.gpu_aware, "transpose": comm.transpose,
        "transpose_requested": comm.transpose_requested, "fft_batch": args.fft_batch,
        "overlapped": overlapped, "nodes": nodes, "hyper": bool(args.hyper), "label": args.label,
        "steps": args.steps, "seconds_per_step": wall / args.steps,
        "mzc_per_second_per_rank": per_rank * args.steps / wall / 1e6,
        "fft_fields_per_step": (grid.fft.forward_fields + grid.fft.inverse_fields) / args.steps,
        "megabytes_sent_per_step_per_rank": comm.sum(comm.bytes_sent) / comm.size / args.steps / 1e6,
        "exchange_gb_per_second_per_rank": None if overlapped else exchange_gbs,
        "peak_gpu_memory_gib": peak,
        "seconds_per_step_by_section": {n: float(v) / args.steps for n, v in zip(names, worst)},
    }
    result["forcing_seconds_per_step"] = sum(v for n, v in result["seconds_per_step_by_section"].items()
                                             if n.startswith("forcing"))
    if comm.is_root:
        print(f"{args.problem} [{system}] {Nx}x{Ny}x{Nz} on {comm.size} rank(s) on {nodes} node(s), "
              f"{args.precision}, {result['tier']} tier, {comm.describe()}")
        print(f"  {result['seconds_per_step']:.4f} s/step   {result['mzc_per_second_per_rank']:.2f} Mzc/s/rank   "
              f"{result['megabytes_sent_per_step_per_rank']:.0f} MB sent/step/rank   "
              f"peak GPU memory {peak:.2f} GiB")
        if comm.size > 1:
            if overlapped:
                print(f"  exchange [{comm.transpose}, behind the transforms]: {sent_per_step / 1e6:.0f} MB sent per "
                      f"step, step blocked on it for {exchange_seconds:.4f} s "
                      f"({100 * exchange_seconds / result['seconds_per_step']:.1f}%)")
            else:
                slow = "" if not exchange_gbs < 1.0 else "   <-- under 1 GB/s: the transport is misconfigured"
                print(f"  exchange [{comm.transpose}]: {exchange_gbs:.2f} GB/s per rank "
                      f"({sent_per_step / 1e6:.0f} MB sent in {exchange_seconds:.4f} s per step){slow}")
        if args.forcing != "none":
            print(f"  {args.forcing} forcing: {result['forcing_seconds_per_step']:.4f} s/step "
                  f"({100 * result['forcing_seconds_per_step'] / result['seconds_per_step']:.1f}% of the step)")
        for name in names:
            share = result["seconds_per_step_by_section"][name] / result["seconds_per_step"]
            print(f"    {name:14s} {result['seconds_per_step_by_section'][name]:9.4f} s/step  {100 * share:5.1f}%")
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    comm.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
