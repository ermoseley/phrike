"""PHRIKE package initialization."""

from typing import Optional

# Register all problems
from .problems import register  # noqa: F401

__all__ = [
    "__version__",
    "run_simulation",
]

__version__ = "0.1.0"


def run_simulation(
    problem_name: str,
    config_path: Optional[str] = None,
    config: Optional[dict] = None,
    backend: str = "auto",
    device: Optional[str] = None,
    precision: Optional[str] = None,
    generate_video: bool = False,
    debug: bool = False,
    restart_from: Optional[str] = None,
):
    """Convenience function to run a simulation.

    Video generation is off unless ``generate_video`` is true, matching the
    command-line default.

    Args:
        problem_name: Name of the problem to run
        config_path: Path to YAML configuration file
        config: Configuration dictionary
        backend: Compute backend. ``auto`` prefers Metal/MPS, then CUDA,
            then NumPy CPU. Explicit aliases are ``metal``, ``cuda``, ``cpu``,
            ``numpy``, and ``torch``.
        device: Optional Torch device ('cpu', 'mps'/'metal', 'cuda')
        precision: Optional floating point precision override
            ('single' or 'double'). Metal/MPS executes in single precision.
        generate_video: Whether to generate video from frames. Default is
            off, matching the CLI.
        debug: Debug mode - error if specified backend/device is not available
        restart_from: Optional checkpoint path. Domain, basis, boundary
            condition, dimension, and gamma must match the snapshot.

    Returns:
        Tuple of (solver, history)
    """
    from .problems import ProblemRegistry

    problem = ProblemRegistry.create_problem(
        name=problem_name,
        config_path=config_path,
        config=config,
        restart_from=restart_from,
    )

    # Override precision in config if provided
    if precision:
        if "grid" not in problem.config or not isinstance(problem.config["grid"], dict):
            problem.config["grid"] = {}
        problem.config["grid"]["precision"] = precision
        problem.precision = precision

    return problem.run(backend=backend, device=device, generate_video=generate_video, debug=debug)
