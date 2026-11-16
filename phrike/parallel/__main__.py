"""Command line of the distributed solvers.

    mpirun -n 8 python -m phrike.parallel imhd_turb --config configs/parallel/imhd_driven_turb.yaml
    python -m phrike.parallel --list
"""

from __future__ import annotations

import argparse
import sys


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m phrike.parallel",
        description="PHRIKE distributed (MPI, one rank per GPU) spectral solvers",
    )
    parser.add_argument("problem", nargs="?", help="problem name (see --list)")
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--list", action="store_true", help="list the distributed problems and exit")
    parser.add_argument("--backend", default="auto", choices=["auto", "cuda", "cpu", "torch", "mps", "metal"],
                        help="device: auto uses one CUDA device per rank when available, else the CPU")
    parser.add_argument("--precision", choices=["single", "double"], help="override grid.precision")
    parser.add_argument("--restart", help="snapshot or checkpoint directory to restart from")
    parser.add_argument("--outdir", help="override io.outdir")
    parser.add_argument("--max-steps", type=int, help="stop after this many steps (writes a checkpoint)")
    parser.add_argument("--max-wall-minutes", type=float,
                        help="stop and write a checkpoint after this much wall-clock time")
    parser.add_argument("--profile", action="store_true",
                        help="synchronize the device around timed sections (accurate timing.json)")
    args = parser.parse_args(argv)

    from .problems import get_problem, list_problems

    if args.list or not args.problem:
        for name in list_problems():
            problem = get_problem(name)
            print(f"{name:20s} [{problem.system:5s}] {problem.description}")
        return 0
    if not args.config:
        parser.error("--config is required")

    from ..io import load_config
    from .comm import Comm
    from .driver import Simulation

    config = load_config(args.config)
    par = config.get("parallel") or {}
    comm = Comm.auto(gpu_aware=par.get("gpu_aware_mpi"), transpose=par.get("transpose"))
    comm.install_abort_hook()
    sim = Simulation(args.problem, config, comm=comm, backend=args.backend, restart=args.restart,
                     outdir=args.outdir, precision=args.precision, max_steps=args.max_steps,
                     max_wall_minutes=args.max_wall_minutes, profile=args.profile or None)
    sim.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
