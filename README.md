HYDRA (HYDRodynamics GPU-Accelerated solver)
=====================================================

HYDRA is a modular pseudo-spectral solver for compressible flows. The initial release implements a 1D solver for the Euler equations using FFT-based spatial derivatives with RK2/RK4 time integration. The design is dimension-agnostic and intended to be extended to 2D/3D with MPI support.

Quick start
-----------

1. Install dependencies (works with your default Python/Conda):

   pip install -e .

2. Run the Sod shock tube example:

   python examples/run_sod.py --config configs/sod.yaml

Backends (NumPy vs Torch)
-------------------------

SpectralHydro supports two array/FFT backends:

- NumPy/SciPy (CPU, default)
- Torch (CPU, Apple Silicon via MPS, or CUDA if available)

Install Torch (optional):

```
pip install torch torchvision torchaudio
```

Use the `--backend` flag in examples to select the backend, and optionally `--device` for Torch:

```
# NumPy (default)
python examples/run_khi2d.py --config configs/khi2d.yaml --backend numpy

# Torch on CPU
python examples/run_khi2d.py --config configs/khi2d.yaml --backend torch --device cpu

# Torch on Apple Silicon GPU (MPS)
python examples/run_khi2d.py --config configs/khi2d.yaml --backend torch --device mps

# Torch on CUDA (Linux/NVIDIA)
python examples/run_khi2d.py --config configs/khi2d.yaml --backend torch --device cuda
```

Notes:

- Torch backend uses `torch.fft` and vectorized ops. No numba is used on Torch.
- On macOS, CuPy is not supported for GPU; Torch via MPS is the recommended GPU path.
- If Torch is not installed or the requested device is unavailable, use the NumPy backend.

Project structure
-----------------

- hydro/
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


