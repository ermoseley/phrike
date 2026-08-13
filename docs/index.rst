PHRIKE Documentation
===================

PHRIKE is a high-performance pseudo-spectral hydrodynamics solver for
compressible Euler equations. It supports 1D, 2D, and 3D simulations with
automatic Metal/MPS, CUDA, and NumPy CPU backend selection.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   user_guide
   api_reference
   examples
   development
   changelog

Features
--------

* **Multi-dimensional**: 1D, 2D, and 3D Euler equation solvers
* **Spectral Accuracy**: Pseudo-spectral methods with exponential convergence
* **Portable Backends**: Metal/MPS preferred, with CUDA and CPU support
* **Adaptive Integration**: RK23, RK45, and full 13-stage Fehlberg RKF78
* **High Performance**: Numba JIT compilation and FFTW integration
* **Comprehensive Testing**: Extensive test suite with validation problems
* **Easy to Use**: YAML configuration and simple Python API

Quick Example
-------------

.. code-block:: python

   import phrike
   
   # Run a 1D Sod shock tube simulation
   solver, history = phrike.run_simulation(
       problem_name="sod",
       config_path="configs/sod.yaml",
       backend="cpu"
   )
   
   # Run with GPU acceleration
   solver, history = phrike.run_simulation(
       problem_name="khi2d",
       config_path="configs/khi2d.yaml",
       backend="metal"  # use "cuda" on NVIDIA systems
   )

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
