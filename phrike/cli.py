"""Command-line interface for SpectralHydro."""

import argparse
import sys

from .problems import ProblemRegistry


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="PHRIKE (Pseudo-spectral Hydrodynamical solver for Realistic Integration of physiKal Environments)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  phrike sod --config configs/sod.yaml
  phrike khi2d --backend torch --device mps
  phrike tgv3d --config configs/tgv3d.yaml --backend torch --device cuda
  phrike turb3d --no-video
        """,
    )

    # Problem selection
    available_problems = ProblemRegistry.list_problems()
    parser.add_argument(
        "problem",
        choices=available_problems,
        help=f"Problem to run. Available: {', '.join(available_problems)}",
    )

    # Configuration
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Backend options
    parser.add_argument(
        "--backend",
        type=str,
        default="numpy",
        choices=["numpy", "torch"],
        help="Array backend (default: numpy)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device: cpu|mps|cuda (if backend=torch)",
    )

    # Output options
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument("--outdir", type=str, help="Override output directory")

    # Restart options
    parser.add_argument(
        "--restart-from",
        type=str,
        help="Restart simulation from checkpoint file (e.g., snapshot_t0.100000.npz)",
    )

    # Cache management
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear numba cache before running (one-time fix for package rename issues)",
    )

    # Verbosity
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    try:
        # Clear cache if requested
        if args.clear_cache:
            from .problems.base import BaseProblem

            BaseProblem.clear_numba_cache()
            if args.verbose:
                print("Cleared numba cache")

        # Create problem instance
        problem = ProblemRegistry.create_problem(
            name=args.problem, config_path=args.config, restart_from=args.restart_from
        )

        # Override output directory if specified
        if args.outdir:
            problem.outdir = args.outdir
            from phrike.io import ensure_outdir

            ensure_outdir(problem.outdir)

        # Run simulation
        if args.verbose:
            print(f"Running {args.problem} problem...")
            print(f"Backend: {args.backend}")
            if args.device:
                print(f"Device: {args.device}")
            print(f"Output directory: {problem.outdir}")

        solver, history = problem.run(
            backend=args.backend, device=args.device, generate_video=not args.no_video
        )

        if args.verbose:
            print(f"Simulation completed at t={solver.t:.6f}")
            print(f"Results saved to: {problem.outdir}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
