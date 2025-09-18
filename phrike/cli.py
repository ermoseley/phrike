"""Command-line interface for SpectralHydro."""

import argparse
import os
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
  phrike sod --video-quality high --video-fps 60
  phrike khi2d --video-codec libx264 --video-quality medium
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
    # Basis options (1D only currently)
    parser.add_argument(
        "--basis",
        type=str,
        default=None,
        choices=["fourier", "chebyshev", "legendre"],
        help="Spectral basis (1D problems): fourier|chebyshev|legendre",
    )
    parser.add_argument(
        "--bc",
        type=str,
        default=None,
        help="Boundary condition for non-periodic basis (e.g., dirichlet, reflective)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device: cpu|mps|cuda (if backend=torch)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["single", "double"],
        default="double",
        help="Floating point precision: single|double (default: double)",
    )

    # Output options
    parser.add_argument("--no-video", action="store_true", help="Skip video generation")
    parser.add_argument("--outdir", type=str, help="Override output directory")
    
    # Video quality options
    parser.add_argument(
        "--video-quality",
        type=str,
        choices=["low", "medium", "high"],
        default="high",
        help="Video quality setting (default: high)",
    )
    parser.add_argument("--video-fps", type=int, help="Video frames per second")
    parser.add_argument("--video-codec", type=str, help="Video codec (e.g., libx264, h264_videotoolbox)")

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

    # Debug and verbosity
    parser.add_argument("--debug", action="store_true", help="Debug mode")
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
        
        # Optional runtime threading control via config: runtime.num_threads: int|"auto"
        try:
            runtime_cfg = problem.config.get("runtime", {})
            num_threads_cfg = runtime_cfg.get("num_threads", None)
            if num_threads_cfg is not None:
                if isinstance(num_threads_cfg, str) and num_threads_cfg.strip().lower() == "auto":
                    num_threads = os.cpu_count() or 1
                else:
                    num_threads = int(num_threads_cfg)
                # Set common BLAS/OpenMP env vars before heavy ops
                for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
                    os.environ[var] = str(num_threads)
                # Torch threading (CPU codepaths)
                try:
                    import torch  # type: ignore

                    torch.set_num_threads(num_threads)
                    try:
                        torch.set_num_interop_threads(num_threads)
                    except Exception:
                        pass
                except Exception:
                    pass
                if args.verbose:
                    print(f"Threads: {num_threads} (from runtime.num_threads)")
        except Exception:
            # Non-fatal: continue with defaults
            pass
        
        # Override video settings if specified
        if not args.no_video:
            if 'video' not in problem.config:
                problem.config['video'] = {}
            
            if args.video_quality:
                problem.config['video']['quality'] = args.video_quality
            if args.video_fps:
                problem.config['video']['fps'] = args.video_fps
            if args.video_codec:
                problem.config['video']['codec'] = args.video_codec

        # Run simulation
        if args.verbose:
            print(f"Running {args.problem} problem...")
            print(f"Backend: {args.backend}")
            if args.device:
                print(f"Device: {args.device}")
            print(f"Precision: {args.precision}")
            print(f"Output directory: {problem.outdir}")

        # Inject CLI overrides into problem config if provided
        if args.basis or args.bc or args.precision:
            if "grid" not in problem.config:
                problem.config["grid"] = {}
            if args.basis:
                problem.config["grid"]["basis"] = args.basis
            if args.bc:
                problem.config["grid"]["bc"] = args.bc
            if args.precision:
                problem.config["grid"]["precision"] = args.precision
                # Ensure in-memory attribute reflects CLI override applied post-init
                try:
                    problem.precision = args.precision
                except Exception:
                    pass

        solver, history = problem.run(
            backend=args.backend, device=args.device, generate_video=not args.no_video, debug=args.debug
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
