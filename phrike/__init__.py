"""PHRIKE package initialization."""

# Register all problems
from .problems import register  # noqa: F401

__all__ = [
    "__version__",
    "run_simulation",
]

__version__ = "0.1.0"


def run_simulation(
    problem_name: str,
    config_path: str = None,
    config: dict = None,
    backend: str = "auto",
    device: str = None,
    precision: str = None,
    generate_video: bool = True,
    debug: bool = False,
):
    """Convenience function to run a simulation.

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
        generate_video: Whether to generate video from frames
        debug: Debug mode - error if specified backend/device is not available

    Returns:
        Tuple of (solver, history)
    """
    from .problems import ProblemRegistry

    problem = ProblemRegistry.create_problem(
        name=problem_name, config_path=config_path, config=config
    )

    # Override precision in config if provided
    if precision:
        if "grid" not in problem.config:
            problem.config["grid"] = {}
        problem.config["grid"]["precision"] = precision
        # Also update in-memory attribute to ensure downstream getters see override
        try:
            problem.precision = precision
        except Exception:
            pass

    return problem.run(backend=backend, device=device, generate_video=generate_video, debug=debug)
