"""Problem registry for dynamic problem loading."""

from typing import Dict, Type, List, Optional
from .base import BaseProblem


class ProblemRegistry:
    """Registry for managing available problems."""

    _problems: Dict[str, Type[BaseProblem]] = {}

    @classmethod
    def register(cls, name: str, problem_class: Type[BaseProblem]) -> None:
        """Register a problem class.

        Args:
            name: Problem name (e.g., 'sod', 'khi2d', 'tgv3d')
            problem_class: Problem class that inherits from BaseProblem
        """
        cls._problems[name] = problem_class

    @classmethod
    def get_problem_class(cls, name: str) -> Type[BaseProblem]:
        """Get a problem class by name.

        Args:
            name: Problem name

        Returns:
            Problem class

        Raises:
            KeyError: If problem name is not registered
        """
        if name not in cls._problems:
            available = ", ".join(cls._problems.keys())
            raise KeyError(f"Unknown problem '{name}'. Available problems: {available}")
        return cls._problems[name]

    @classmethod
    def create_problem(
        cls,
        name: str,
        config_path: Optional[str] = None,
        config: Optional[dict] = None,
        restart_from: Optional[str] = None,
    ) -> BaseProblem:
        """Create a problem instance.

        Args:
            name: Problem name
            config_path: Path to YAML configuration file
            config: Configuration dictionary
            restart_from: Path to checkpoint file for restart

        Returns:
            Problem instance
        """
        problem_class = cls.get_problem_class(name)
        return problem_class(
            config_path=config_path, config=config, restart_from=restart_from
        )

    @classmethod
    def list_problems(cls) -> List[str]:
        """List all registered problem names.

        Returns:
            List of problem names
        """
        return list(cls._problems.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a problem is registered.

        Args:
            name: Problem name

        Returns:
            True if problem is registered
        """
        return name in cls._problems
