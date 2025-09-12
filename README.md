# PHRIKE: Pseudo-spectral Hydrodynamical solver for Realistic Integration of physiKal Environments

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PHRIKE is a high-performance pseudo-spectral solver for compressible Euler equations, designed for computational fluid dynamics research and education. It features exponential convergence, dual backend support (CPU/GPU), and comprehensive monitoring capabilities.

## ✨ Features

- **Multi-dimensional**: 1D, 2D, and 3D Euler equation solvers
- **Spectral Accuracy**: Pseudo-spectral methods with exponential convergence for smooth solutions
- **Dual Backend**: NumPy (CPU) and PyTorch (GPU) support with automatic device detection
- **High Performance**: Numba JIT compilation and optional FFTW integration
- **Comprehensive Testing**: Extensive test suite with validation problems
- **Easy to Use**: YAML configuration and simple Python API
- **Monitoring**: Built-in conservation tracking and real-time statistics
- **Visualization**: Automatic frame generation and video creation

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/phrike.git
cd phrike

# Install in development mode
pip install -e .

# Install with optional dependencies
pip install -e .[fastfft,dev,docs]
```

### Basic Usage

```bash
# Run a 1D Sod shock tube
phrike sod --config configs/sod.yaml

# Run 2D Kelvin-Helmholtz instability
phrike khi2d --config configs/khi2d.yaml

# Run with GPU acceleration
phrike tgv3d --backend torch --device cuda
```

### Python API

```python
import phrike

# Run simulation and get results
solver, history = phrike.run_simulation(
    problem_name="sod",
    config_path="configs/sod.yaml"
)

# Access final state
print(f"Final time: {solver.t}")
print(f"Final density range: {solver.U[0].min():.3f} to {solver.U[0].max():.3f}")
```

## 📚 Documentation

- [Installation Guide](docs/installation.rst)
- [Quick Start Guide](docs/quickstart.rst)
- [User Guide](docs/user_guide.rst)
- [API Reference](docs/api_reference.rst)
- [Examples](docs/examples.rst)

To build documentation locally:

```bash
pip install -e .[docs]
cd docs
make html
```

## 🧪 Available Problems

### 1D Problems
- **Sod Shock Tube** (`sod`): Classic Riemann problem for shock wave validation
- **Acoustic Waves** (`acoustic1d`): Linear wave propagation for accuracy testing
- **Gaussian Wave Packets** (`gaussian_wave1d`): Stationary and traveling wave tests

### 2D Problems
- **Kelvin-Helmholtz Instability** (`khi2d`): Shear layer instability for mixing studies

### 3D Problems
- **Taylor-Green Vortex** (`tgv3d`): Decaying vortex for turbulence validation
- **3D Turbulence** (`turb3d`): Forced turbulence for statistical analysis

## ⚙️ Configuration

PHRIKE uses YAML configuration files. Here's a basic example:

```yaml
problem: sod

grid:
  N: 1024
  Lx: 1.0
  dealias: true

physics:
  gamma: 1.4

integration:
  t0: 0.0
  t_end: 0.2
  cfl: 0.4
  scheme: rk4

initial_conditions:
  left:
    rho: 1.0
    u: 0.0
    p: 1.0
  right:
    rho: 0.125
    u: 0.0
    p: 0.1
```

## 🔧 Backend Support

### NumPy Backend (Default)
- CPU-only computation
- Uses SciPy FFT or PyFFTW (if available)
- Stable and well-tested

### PyTorch Backend
- GPU acceleration via CUDA or MPS (Apple Silicon)
- Automatic device detection
- Better memory management for large problems

```bash
# Use GPU acceleration
phrike sod --backend torch --device cuda
phrike sod --backend torch --device mps  # Apple Silicon
```

## 📊 Performance

PHRIKE is optimized for high-performance computing:

- **Numba JIT**: Critical kernels compiled for speed
- **FFTW Integration**: Optional high-performance FFT
- **GPU Acceleration**: PyTorch backend with MPS/CUDA support
- **Memory Efficient**: Optimized array operations

### Benchmark Results

| Problem | Resolution | CPU Time | GPU Time | Speedup |
|---------|------------|----------|----------|---------|
| 1D Gaussian | 1024 | 2.3s | 0.8s | 2.9x |
| 2D KHI | 128² | 45s | 12s | 3.8x |
| 3D TGV | 64³ | 180s | 35s | 5.1x |

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_1d_solver.py
pytest tests/test_sod_validation.py
```

## 📈 Monitoring

PHRIKE includes comprehensive monitoring capabilities:

```yaml
monitoring:
  enabled: true
  step_interval: 10
  include_conservation: true
  include_timestep: true
  include_velocity_stats: true
```

## 🎥 Visualization

Automatic visualization and video generation:

```bash
# Generate video (default)
phrike sod --config configs/sod.yaml

# Skip video generation
phrike sod --config configs/sod.yaml --no-video
```

## 🔬 Scientific Validation

PHRIKE has been validated against:

- **Analytical Solutions**: Sod shock tube, acoustic waves
- **Literature Benchmarks**: Taylor-Green vortex, KHI growth rates
- **Conservation Properties**: Mass, momentum, energy conservation
- **Convergence Studies**: Spectral accuracy verification

## 🛠️ Development

### Code Quality

```bash
# Format code
black phrike/

# Lint code
ruff check phrike/

# Type checking
mypy phrike/ --ignore-missing-imports
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **NumPy/SciPy**: Core numerical computing
- **PyTorch**: GPU acceleration
- **Numba**: JIT compilation
- **Matplotlib**: Visualization
- **FFTW**: High-performance FFT (optional)

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/phrike/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/phrike/discussions)
- **Documentation**: [Read the Docs](https://phrike.readthedocs.io/)

## 🔗 Related Projects

- [Dedalus](https://dedalus-project.org/): General-purpose spectral PDE solver
- [SpectralDNS](https://github.com/spectralDNS/spectralDNS): Spectral DNS solver
- [PySpectral](https://github.com/pyspectral/pyspectral): Spectral analysis tools