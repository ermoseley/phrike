Development Guide
=================

This guide is for developers who want to contribute to PHRIKE or extend its functionality.

Development Setup
-----------------

1. **Clone the repository**:

   .. code-block:: bash

      git clone https://github.com/your-username/phrike.git
      cd phrike

2. **Install in development mode**:

   .. code-block:: bash

      pip install -e .[dev,docs]

3. **Install pre-commit hooks** (optional):

   .. code-block:: bash

      pre-commit install

Code Structure
--------------

PHRIKE is organized into several main modules:

- **``phrike/equations.py``**: Euler equation implementations (1D, 2D, 3D)
- **``phrike/grid.py``**: Spectral grid classes and FFT operations
- **``phrike/solver.py``**: Time integration and solver classes
- **``phrike/problems/``**: Problem-specific implementations
- **``phrike/io.py``**: Input/output utilities
- **``phrike/visualization.py``**: Plotting and visualization
- **``phrike/cli.py``**: Command-line interface

### Adding New Problems

To add a new problem, create a new class inheriting from ``BaseProblem``:

.. code-block:: python

   from phrike.problems.base import BaseProblem
   from phrike.grid import Grid1D
   from phrike.equations import EulerEquations1D
   from phrike.solver import SpectralSolver1D
   
   class MyProblem(BaseProblem):
       """My custom problem."""
       
       def create_grid(self, backend="numpy", device=None):
           """Create the computational grid."""
           N = int(self.config["grid"]["N"])
           Lx = float(self.config["grid"]["Lx"])
           
           return Grid1D(
               N=N, Lx=Lx, dealias=True,
               backend=backend, torch_device=device
           )
       
       def create_equations(self):
           """Create the equation system."""
           return EulerEquations1D(gamma=self.gamma)
       
       def create_initial_conditions(self, grid):
           """Create initial conditions."""
           # Your initial conditions here
           rho = np.ones_like(grid.x)
           u = np.zeros_like(grid.x)
           p = np.ones_like(grid.x)
           
           return self.equations.conservative(rho, u, p)
       
       def create_visualization(self, solver, t, U):
           """Create visualization for current state."""
           # Your visualization code here
           pass

Then register the problem in ``phrike/problems/register.py``:

.. code-block:: python

   from .my_problem import MyProblem
   
   ProblemRegistry.register("my_problem", MyProblem)

### Adding New Equations

To add new equation systems, create a new class:

.. code-block:: python

   from dataclasses import dataclass
   from typing import Tuple
   import numpy as np
   
   @dataclass
   class MyEquations:
       """My custom equation system."""
       
       def primitive(self, U):
           """Convert conservative to primitive variables."""
           # Your implementation here
           pass
       
       def conservative(self, *args):
           """Convert primitive to conservative variables."""
           # Your implementation here
           pass
       
       def flux(self, U):
           """Compute flux vector."""
           # Your implementation here
           pass
       
       def max_wave_speed(self, U):
           """Compute maximum wave speed."""
           # Your implementation here
           pass

### Adding New Grid Types

To add new grid types, create a new class:

.. code-block:: python

   from dataclasses import dataclass
   import numpy as np
   
   @dataclass
   class MyGrid:
       """My custom grid type."""
       
       def __post_init__(self):
           """Initialize grid after creation."""
           # Your initialization code here
           pass
       
       def fft(self, f):
           """Forward FFT."""
           # Your FFT implementation here
           pass
       
       def ifft(self, F):
           """Inverse FFT."""
           # Your IFFT implementation here
           pass
       
       def derivative(self, f):
           """Compute derivative."""
           # Your derivative implementation here
           pass

Testing
-------

PHRIKE uses pytest for testing. Run tests with:

.. code-block:: bash

   # Run all tests
   pytest
   
   # Run specific test file
   pytest tests/test_1d_solver.py
   
   # Run with coverage
   pytest --cov=phrike
   
   # Run with verbose output
   pytest -v

### Writing Tests

Create test files in the ``tests/`` directory:

.. code-block:: python

   import pytest
   import numpy as np
   from phrike.equations import EulerEquations1D
   
   class TestMyFeature:
       def test_basic_functionality(self):
           """Test basic functionality."""
           eqs = EulerEquations1D(gamma=1.4)
           
           # Test data
           rho = np.ones(10)
           u = np.zeros(10)
           p = np.ones(10)
           
           # Test conversion
           U = eqs.conservative(rho, u, p)
           rho_out, u_out, p_out, _ = eqs.primitive(U)
           
           # Assertions
           np.testing.assert_allclose(rho, rho_out)
           np.testing.assert_allclose(u, u_out)
           np.testing.assert_allclose(p, p_out)
       
       def test_edge_cases(self):
           """Test edge cases."""
           # Your edge case tests here
           pass

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test complete workflows
- **Validation tests**: Test against analytical solutions
- **Performance tests**: Test performance characteristics

Code Quality
------------

PHRIKE enforces high code quality standards:

### Formatting

Use Black for code formatting:

.. code-block:: bash

   black phrike/ --line-length 88

### Linting

Use Ruff for linting:

.. code-block:: bash

   ruff check phrike/ --fix

### Type Checking

Use mypy for type checking:

.. code-block:: bash

   mypy phrike/ --ignore-missing-imports

### Documentation

Follow these documentation standards:

1. **Docstrings**: Use Google-style docstrings
2. **Type hints**: Add type annotations to all functions
3. **Comments**: Explain complex algorithms and logic
4. **Examples**: Include usage examples in docstrings

Example docstring:

.. code-block:: python

   def my_function(param1: float, param2: str) -> np.ndarray:
       """Brief description of the function.
       
       Longer description explaining what the function does,
       its purpose, and any important details.
       
       Args:
           param1: Description of param1
           param2: Description of param2
       
       Returns:
           Description of return value
       
       Raises:
           ValueError: When invalid input is provided
       
       Example:
           >>> result = my_function(1.0, "test")
           >>> print(result.shape)
           (10,)
       """
       # Implementation here
       pass

Performance Optimization
------------------------

### Profiling

Use Python's built-in profiler:

.. code-block:: python

   import cProfile
   import pstats
   
   # Profile a function
   cProfile.run('my_function()', 'profile_output')
   
   # Analyze results
   p = pstats.Stats('profile_output')
   p.sort_stats('cumulative').print_stats(10)

### Memory Profiling

Use memory_profiler for memory analysis:

.. code-block:: python

   from memory_profiler import profile
   
   @profile
   def my_function():
       # Your code here
       pass

### Numba Optimization

Use Numba for performance-critical code:

.. code-block:: python

   from numba import njit
   
   @njit(cache=True, fastmath=True)
   def fast_function(data):
       """Numba-accelerated function."""
       # Your optimized code here
       return result

### GPU Optimization

Optimize GPU code for PyTorch:

.. code-block:: python

   import torch
   
   def gpu_optimized_function(data):
       """GPU-optimized function."""
       # Use in-place operations when possible
       data *= 2.0
       
       # Minimize CPU-GPU transfers
       result = data.sum()
       
       # Use appropriate data types
       data = data.float()  # Use float32 for MPS
       
       return result

Documentation
-------------

### Building Documentation

Build documentation locally:

.. code-block:: bash

   cd docs
   make html
   
   # View in browser
   open _build/html/index.html

### Documentation Structure

- **API Reference**: Auto-generated from docstrings
- **User Guide**: How to use PHRIKE
- **Examples**: Code examples and tutorials
- **Development Guide**: This guide
- **Changelog**: Version history

### Writing Documentation

Follow these guidelines:

1. **Use reStructuredText**: For Sphinx documentation
2. **Include code examples**: Show how to use features
3. **Keep it up to date**: Update docs when code changes
4. **Use clear language**: Write for your audience
5. **Include diagrams**: Use matplotlib for plots

Release Process
---------------

### Version Numbering

Use semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features, backward compatible
- **PATCH**: Bug fixes, backward compatible

### Release Checklist

1. **Update version** in ``pyproject.toml``
2. **Update changelog** in ``CHANGELOG.md``
3. **Run tests** to ensure everything works
4. **Build documentation** and check for errors
5. **Create release** on GitHub
6. **Publish to PyPI** (if applicable)

### Creating a Release

.. code-block:: bash

   # Update version
   vim pyproject.toml
   
   # Commit changes
   git add pyproject.toml CHANGELOG.md
   git commit -m "Bump version to 0.2.0"
   git tag v0.2.0
   git push origin main --tags
   
   # Create GitHub release
   gh release create v0.2.0 --generate-notes

Contributing
------------

### Getting Started

1. **Fork the repository** on GitHub
2. **Create a feature branch**: ``git checkout -b feature/my-feature``
3. **Make your changes** and add tests
4. **Run the test suite** to ensure everything works
5. **Submit a pull request** with a clear description

### Pull Request Guidelines

1. **Write clear commit messages**
2. **Include tests** for new functionality
3. **Update documentation** as needed
4. **Follow code style** guidelines
5. **Describe changes** in the PR description

### Code Review Process

1. **Automated checks** must pass
2. **Manual review** by maintainers
3. **Address feedback** and make changes
4. **Merge** when approved

### Issue Reporting

When reporting issues:

1. **Use the issue template**
2. **Provide minimal reproduction** code
3. **Include system information**
4. **Describe expected vs actual** behavior
5. **Add relevant labels**

### Feature Requests

When requesting features:

1. **Check existing issues** first
2. **Describe the use case** clearly
3. **Explain the expected behavior**
4. **Consider implementation** complexity
5. **Be patient** with responses
