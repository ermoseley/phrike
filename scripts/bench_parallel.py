"""Throughput and weak-scaling benchmark of the distributed solvers.

    python -m scripts.bench_parallel --problem imhd_turb --points-per-rank 256
    srun -n 8  python -m scripts.bench_parallel --problem imhd_turb --points-per-rank 256
    srun -n 64 python -m scripts.bench_parallel --problem imhd_turb --points-per-rank 256

Weak scaling keeps the work per rank fixed: with P ranks the grid is the cube
whose side is the FFT-friendly even number nearest to
``points_per_rank * P**(1/3)`` (or pass ``--grid Nx Ny Nz``).  Perfect weak
scaling is a constant "Mzc/s/rank" (10^6 zone-cycles per second per rank) as P
grows, apart from the log N factor of the transforms.  The timer breakdown
separates the local transforms (``fft_xy``, ``fft_z``), packing (``fft_pack``)
and MPI (``mpi_exchange``, plus ``mpi_stage`` when device buffers are off).
``--forcing ou|white|constant_power`` adds large-scale forcing to the timed
steps (constant_power is incompressible only); its cost is reported as
``forcing`` (and ``forcing_kick`` for the compressible momentum/energy update).

One JSON line per run is appended to ``--out`` for plotting.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--problem", default="imhd_turb")
    parser.add_argument("--points-per-rank", type=int, default=128, help="cube root of the grid points per rank")
    parser.add_argument("--grid", type=int, nargs=3, metavar=("NX", "NY", "NZ"), help="explicit grid instead")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3, help="untimed steps (kernel builds, FFT plans)")
    parser.add_argument("--precision", choices=["single", "double"], default="single")
    parser.add_argument("--scheme", default="rk4")
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--fft-batch", type=int, default=4)
    parser.add_argument("--transpose", choices=["alltoallv", "p2p"], default=None)
    parser.add_argument("--gpu-aware-mpi", choices=["0", "1"], default=None)
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
    if args.forcing != "none":
        config["forcing"] = {"type": args.forcing, "kmin": 1.0, "kmax": 2.5, "amplitude": 0.1,
                             "power": 1e-3, "correlation_time": 1.0, "seed": 1}
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

    names = sorted(sim.timers.total)
    worst = comm.allreduce([sim.timers.total[n] for n in names], "max")
    peak = comm.max(torch.cuda.max_memory_allocated() / 2 ** 30 if grid.device.type == "cuda" else 0.0)
    per_rank = grid.n_points / comm.size
    result = {
        "problem": args.problem, "system": system, "ranks": comm.size, "grid": [Nx, Ny, Nz],
        "precision": args.precision, "scheme": args.scheme, "forcing": args.forcing, "device": str(grid.device),
        "tier": "triton" if getattr(st, "_triton", None) is not None else "torch",
        "gpu_aware_mpi": comm.gpu_aware, "transpose": comm.transpose, "fft_batch": args.fft_batch,
        "steps": args.steps, "seconds_per_step": wall / args.steps,
        "mzc_per_second_per_rank": per_rank * args.steps / wall / 1e6,
        "fft_fields_per_step": (grid.fft.forward_fields + grid.fft.inverse_fields) / args.steps,
        "megabytes_sent_per_step_per_rank": comm.sum(comm.bytes_sent) / comm.size / args.steps / 1e6,
        "peak_gpu_memory_gib": peak,
        "seconds_per_step_by_section": {n: float(v) / args.steps for n, v in zip(names, worst)},
    }
    result["forcing_seconds_per_step"] = sum(v for n, v in result["seconds_per_step_by_section"].items()
                                             if n.startswith("forcing"))
    if comm.is_root:
        print(f"{args.problem} [{system}] {Nx}x{Ny}x{Nz} on {comm.size} rank(s), {args.precision}, "
              f"{result['tier']} tier, {comm.describe()}")
        print(f"  {result['seconds_per_step']:.4f} s/step   {result['mzc_per_second_per_rank']:.2f} Mzc/s/rank   "
              f"{result['megabytes_sent_per_step_per_rank']:.0f} MB sent/step/rank   "
              f"peak GPU memory {peak:.2f} GiB")
        if args.forcing != "none":
            print(f"  {args.forcing} forcing: {result['forcing_seconds_per_step']:.4f} s/step "
                  f"({100 * result['forcing_seconds_per_step'] / result['seconds_per_step']:.1f}% of the step)")
        for name in names:
            share = result["seconds_per_step_by_section"][name] / result["seconds_per_step"]
            print(f"    {name:14s} {result['seconds_per_step_by_section'][name]:9.4f} s/step  {100 * share:5.1f}%")
        with open(args.out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(result) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
