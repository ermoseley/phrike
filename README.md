HYDRA (HYDRodynamics GPU-Accelerated solver)
=====================================================

HYDRA is a modular pseudo-spectral solver for compressible flows. The initial release implements 1D, 2D, and 3D solvers for the Euler equations using FFT-based spatial derivatives with RK2/RK4 time integration. The design is dimension-agnostic and supports both CPU (NumPy) and GPU (Torch) backends.

Quick start
-----------

1. Install dependencies (works with your default Python/Conda):

   pip install -e .

2. Run simulations using the unified CLI:

   # 1D Sod shock tube
   python -m hydra sod --config configs/sod.yaml
   
   # 2D Kelvin-Helmholtz instability
   python -m hydra khi2d --config configs/khi2d.yaml
   
   # 3D Taylor-Green vortex
   python -m hydra tgv3d --config configs/tgv3d.yaml
   
   # 3D turbulent velocity field
   python -m hydra turb3d --config configs/turb3d.yaml
   
   # 3D turbulent simulation with monitoring (monitoring enabled by default)
   python examples/run_turb3d_with_monitoring.py

Backends (NumPy vs Torch)
-------------------------

Hydra supports two array/FFT backends:

- NumPy/SciPy (CPU, default)
- Torch (CPU, Apple Silicon via MPS, or CUDA if available)

Install Torch (optional):

```
pip install torch torchvision torchaudio
```

Use the `--backend` flag to select the backend, and optionally `--device` for Torch:

```
# NumPy (default)
python -m hydra khi2d --config configs/khi2d.yaml --backend numpy

# Torch on CPU
python -m hydra khi2d --config configs/khi2d.yaml --backend torch --device cpu

# Torch on Apple Silicon GPU (MPS)
python -m hydra khi2d --config configs/khi2d.yaml --backend torch --device mps

# Torch on CUDA (Linux/NVIDIA)
python -m hydra khi2d --config configs/khi2d.yaml --backend torch --device cuda
```

Notes:

- Torch backend uses `torch.fft` and vectorized ops. No numba is used on Torch.
- On macOS, CuPy is not supported for GPU; Torch via MPS is the recommended GPU path.
- If Torch is not installed or the requested device is unavailable, use the NumPy backend.

CLI Interface
-------------

Hydra provides a unified command-line interface for all problems:

```bash
# Basic usage
python -m hydra <problem> [options]

# Available problems
python -m hydra --help

# Common options
python -m hydra sod --config configs/sod.yaml --backend torch --device mps --no-video -v
```

**Available Problems:**
- `sod` - 1D Sod shock tube
- `khi2d` - 2D Kelvin-Helmholtz instability  
- `tgv3d` - 3D Taylor-Green vortex
- `turb3d` - 3D turbulent velocity field

**CLI Options:**
- `--config CONFIG` - Path to YAML configuration file
- `--backend {numpy,torch}` - Array backend (default: numpy)
- `--device DEVICE` - Torch device: cpu|mps|cuda (if backend=torch)
- `--no-video` - Skip video generation
- `--outdir OUTDIR` - Override output directory
- `-v, --verbose` - Verbose output

Programmatic API
----------------

You can also use Hydra programmatically:

```python
from hydra import run_simulation

# Using config file
solver, history = run_simulation("sod", config_path="configs/sod.yaml")

# Using config dictionary
config = {
    "problem": "sod",
    "grid": {"N": 512, "Lx": 1.0},
    "integration": {"t_end": 0.2, "cfl": 0.4},
    "initial_conditions": {
        "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
        "right": {"rho": 0.125, "u": 0.0, "p": 0.1}
    }
}
solver, history = run_simulation("sod", config=config)
```

Architecture
------------

Hydra uses a modular, extensible architecture:

- **Unified CLI**: Single entry point for all problems with consistent interface
- **Problem Registry**: Dynamic loading of problems with easy extensibility
- **Base Problem Class**: Common functionality shared across all problems
- **Multiple Backends**: NumPy (CPU) and Torch (CPU/GPU) support
- **Configuration-Driven**: YAML-based configuration for all parameters
- **Type-Safe**: Full type hints and validation throughout

**Adding New Problems:**
1. Inherit from `BaseProblem` in `hydra/problems/`
2. Implement required methods: `create_grid()`, `create_equations()`, etc.
3. Register with `ProblemRegistry.register("name", ProblemClass)`
4. Create YAML config file in `configs/`

Project structure
-----------------

- hydra/
  - cli.py: unified command-line interface
  - problems/: problem-specific implementations
    - base.py: base problem class with common functionality
    - registry.py: problem registry for dynamic loading
    - sod.py, khi2d.py, tgv3d.py, turb3d.py: specific problems
  - grid.py: spatial grids and FFT transforms
  - equations.py: Euler equations in conservative form
  - solver.py: time integration (RK2/RK4), CFL control
  - initial_conditions.py: Sod, sinusoidal, etc.
  - io.py: save/load state, checkpoints
  - visualization.py: plotting of fields and spectra
- configs/: YAML configuration files for each problem
- examples/: example scripts and usage demonstrations
- tests/: unit tests for grid, solver, and conservation

License
-------

Stanford University


