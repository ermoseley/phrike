SpectralHydro: Pseudo-spectral Hydrodynamics (1D core)
=====================================================

SpectralHydro is a modular pseudo-spectral solver for compressible flows. The initial release implements a 1D solver for the Euler equations using FFT-based spatial derivatives with RK2/RK4 time integration. The design is dimension-agnostic and intended to be extended to 2D/3D with MPI support.

Quick start
-----------

1. Install dependencies (works with your default Python/Conda):

   pip install -e .

2. Run the Sod shock tube example:

   python examples/run_sod.py --config configs/sod.yaml

Project structure
-----------------

- spectralhydro/
  - grid.py: spatial grids and FFT transforms
  - equations.py: Euler equations in conservative form
  - solver.py: time integration (RK2/RK4), CFL control
  - initial_conditions.py: Sod, sinusoidal, etc.
  - io.py: save/load state, checkpoints
  - visualization.py: plotting of fields and spectra
- examples/run_sod.py: driver for the Sod shock tube
- configs/sod.yaml: configuration for the example
- tests/: unit tests for grid, solver, and conservation

License
-------

Stanford University


