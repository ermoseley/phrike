"""Hydra package initialization."""

# Register all problems
from .problems import register  # noqa: F401

__all__ = [
    "__version__",
    "run_simulation",
]

__version__ = "0.1.0"


def run_simulation(problem_name: str, config_path: str = None, config: dict = None, 
                  backend: str = "numpy", device: str = None, generate_video: bool = True):
    """Convenience function to run a simulation.
    
    Args:
        problem_name: Name of the problem to run
        config_path: Path to YAML configuration file
        config: Configuration dictionary
        backend: Array backend ('numpy' or 'torch')
        device: Torch device ('cpu', 'mps', 'cuda')
        generate_video: Whether to generate video from frames
        
    Returns:
        Tuple of (solver, history)
    """
    from .problems import ProblemRegistry
    
    problem = ProblemRegistry.create_problem(
        name=problem_name,
        config_path=config_path,
        config=config
    )
    
    return problem.run(
        backend=backend,
        device=device,
        generate_video=generate_video
    )


