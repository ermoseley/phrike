"""Command-line interface for PHRIKE."""

import argparse
import os
import sys
import warnings


_BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _apply_runtime_threads_from_argv(argv=None) -> None:
    """Set BLAS/OpenMP thread counts before NumPy is imported.

    OpenMP and BLAS read these variables when they are first loaded.
    ``python -m phrike`` and the ``phrike`` console script import this
    module before NumPy, so ``runtime.num_threads`` in ``--config`` takes
    effect. Importing PHRIKE first, then calling :func:`main`, cannot
    resize a BLAS pool that is already initialized. Torch's own thread
    count is updated later in :func:`main`.
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default=None)
    try:
        known, _ = parser.parse_known_args(argv)
    except SystemExit:
        return
    if not known.config or not os.path.isfile(known.config):
        return
    try:
        import yaml
    except ImportError:
        return
    try:
        with open(known.config, "r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        return
    if not isinstance(loaded, dict):
        return
    runtime = loaded.get("runtime") or {}
    if not isinstance(runtime, dict):
        return
    num_threads = _parse_num_threads(runtime.get("num_threads", None))
    if num_threads is None:
        return
    for var in _BLAS_THREAD_VARS:
        os.environ[var] = str(num_threads)


def _parse_num_threads(num_threads_cfg):
    """Return a positive thread count, or None when the setting is absent or invalid."""
    if num_threads_cfg is None:
        return None
    if isinstance(num_threads_cfg, str) and num_threads_cfg.strip().lower() == "auto":
        return os.cpu_count() or 1
    try:
        num_threads = int(num_threads_cfg)
    except (TypeError, ValueError):
        warnings.warn(
            f"Ignoring runtime.num_threads={num_threads_cfg!r}",
            UserWarning,
            stacklevel=2,
        )
        return None
    if num_threads < 1:
        warnings.warn(
            f"Ignoring runtime.num_threads={num_threads}; it must be positive",
            UserWarning,
            stacklevel=2,
        )
        return None
    return num_threads


def _video_flag_overrides(args) -> dict:
    """CLI video settings that were explicitly passed. Defaults stay None."""
    overrides = {}
    if getattr(args, "video_quality", None) is not None:
        overrides["quality"] = args.video_quality
    if getattr(args, "video_fps", None) is not None:
        overrides["fps"] = args.video_fps
    if getattr(args, "video_codec", None) is not None:
        overrides["codec"] = args.video_codec
    return overrides


# Must run before the package import below, which imports NumPy.
_apply_runtime_threads_from_argv()

from .problems import ProblemRegistry  # noqa: E402


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="phrike",
        description="PHRIKE: pseudo-spectral solver for compressible hydrodynamics and MHD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  phrike sod --config configs/sod.yaml
  phrike khi2d --backend metal
  phrike tgv3d --config configs/tgv3d.yaml --backend cuda
  phrike sod --backend cpu
  phrike turb3d --video
  phrike sod --video --video-quality high --video-fps 60
  phrike khi2d --video --video-codec libx264 --video-quality medium
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
        default="auto",
        choices=["auto", "metal", "cuda", "cpu", "numpy", "torch"],
        help=(
            "Compute backend (default: auto; prefers Metal/MPS, then CUDA, "
            "then NumPy CPU)"
        ),
    )
    # Basis options (1D only currently)
    parser.add_argument(
        "--basis",
        type=str,
        default=None,
        choices=["fourier", "legendre", "chebyshev"],
        help="Spectral basis (1D problems): fourier|legendre|chebyshev",
    )
    parser.add_argument(
        "--bc",
        type=str,
        default=None,
        help="1D boundary condition: periodic (Fourier), fixed, outflow, or wall",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "mps", "metal", "cuda"],
        help="Explicit Torch device when --backend=torch",
    )
    parser.add_argument(
        "--precision",
        type=str,
        choices=["single", "double"],
        default=None,
        help=(
            "Override YAML precision: single|double "
            "(Metal/MPS executes in single precision)"
        ),
    )

    # Output options
    parser.add_argument("--video", action="store_true", help="Enable video generation (default: disabled)")
    parser.add_argument("--outdir", type=str, help="Override output directory")
    
    # Video quality options
    parser.add_argument(
        "--video-quality",
        type=str,
        choices=["low", "medium", "high"],
        default=None,
        help="Override YAML video quality (low, medium, or high)",
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
        
        runtime_cfg = problem.config.get("runtime") or {}
        if runtime_cfg and not isinstance(runtime_cfg, dict):
            warnings.warn("runtime must be a mapping; ignoring it", UserWarning, stacklevel=1)
            runtime_cfg = {}
        num_threads = _parse_num_threads(runtime_cfg.get("num_threads", None))
        if num_threads is not None:
            # Too late to change an already initialized BLAS pool. The import
            # hook above covers the console-script path. Torch is set here.
            for var in _BLAS_THREAD_VARS:
                os.environ[var] = str(num_threads)
            try:
                import torch  # type: ignore
            except ImportError:
                torch = None
            if torch is not None:
                try:
                    torch.set_num_threads(num_threads)
                except (RuntimeError, ValueError) as exc:
                    warnings.warn(
                        f"Torch thread count was not changed: {exc}",
                        UserWarning,
                        stacklevel=1,
                    )
                try:
                    torch.set_num_interop_threads(num_threads)
                except RuntimeError as exc:
                    warnings.warn(
                        "Torch inter-op threads were not changed "
                        f"({exc}). Set runtime.num_threads before Torch starts.",
                        UserWarning,
                        stacklevel=1,
                    )
            if args.verbose:
                print(f"Threads: {num_threads} (from runtime.num_threads)")

        # YAML video settings apply first. CLI flags override them only when
        # the flag was actually passed (the defaults are None, not "high").
        video_cfg = problem.config.get("video") or {}
        if video_cfg and not isinstance(video_cfg, dict):
            warnings.warn("video must be a mapping; ignoring it", UserWarning, stacklevel=1)
            video_cfg = {}
            problem.config["video"] = {}
        yaml_video = bool(video_cfg.get("enabled", False))
        generate_video = bool(args.video or yaml_video)
        overrides = _video_flag_overrides(args)
        if overrides and not args.video:
            if generate_video:
                warnings.warn(
                    "Video options were given without --video; they override "
                    "the YAML video settings because video.enabled is true",
                    UserWarning,
                    stacklevel=1,
                )
            else:
                warnings.warn(
                    "Video options were given without --video and video "
                    "generation is off; the options have no effect",
                    UserWarning,
                    stacklevel=1,
                )
                overrides = {}
        if generate_video and overrides:
            problem.config.setdefault("video", {})
            problem.config["video"].update(overrides)

        # Inject CLI overrides into problem config if provided
        if args.basis or args.bc or args.precision:
            if "grid" not in problem.config or not isinstance(problem.config["grid"], dict):
                problem.config["grid"] = {}
            if args.basis:
                problem.config["grid"]["basis"] = args.basis
            if args.bc:
                problem.config["grid"]["bc"] = args.bc
            if args.precision:
                problem.config["grid"]["precision"] = args.precision
                problem.precision = args.precision

        if args.verbose:
            print(f"Running {args.problem} problem...")
            print(f"Backend: {args.backend}")
            if args.device:
                print(f"Device: {args.device}")
            print(f"Output directory: {problem.outdir}")

        solver, history = problem.run(
            backend=args.backend, device=args.device, generate_video=generate_video, debug=args.debug
        )

        if args.verbose:
            print(f"Precision: {solver.grid.precision}")
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
