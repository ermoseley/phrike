# Phrike Installation Guide

## Quick Installation

### Basic Installation (Recommended)
```bash
pip install -e .
```

This installs phrike with all core dependencies including mpmath for high-precision quadrature.

### Optional Dependencies

#### For Fast FFT (Optional)
```bash
pip install -e .[fastfft]
```

#### For PyTorch Backend (Optional)
```bash
pip install -e .[torch]
```

#### For High-Precision Computing (Optional)
```bash
pip install -e .[high-precision]
```

#### For Development (Optional)
```bash
pip install -e .[dev]
```

#### For Documentation (Optional)
```bash
pip install -e .[docs]
```

### All Optional Dependencies
```bash
pip install -e .[fastfft,torch,high-precision,dev,docs]
```

## Usage

After installation, you can run phrike as a command-line tool:

```bash
# Run a problem
phrike khi2d --config configs/khi2d.yaml

# Run RTI problem with high-precision quadrature
phrike rti --config configs/rti.yaml

# List available problems
phrike --help
```

## Dependencies

### Core Dependencies (Always Installed)
- `numpy>=1.22` - Numerical computing
- `scipy>=1.10` - Scientific computing
- `matplotlib>=3.5` - Plotting
- `PyYAML>=6.0` - Configuration files
- `numba>=0.58` - JIT compilation
- `mpmath>=1.3.0` - High-precision arithmetic

### Optional Dependencies
- `pyfftw>=0.13` - Fast FFT (faster than scipy.fft)
- `torch>=2.0.0` - PyTorch backend for GPU acceleration
- `sympy>=1.14.0` - Symbolic mathematics (for high-precision)

## High-Precision Features

Phrike now supports high-precision Legendre-Gauss-Lobatto quadrature:

```python
from fastgaussquadrature_python import gausslobatto, gausslobatto_mpmath

# Double precision (default)
x, w = gausslobatto(64, backend='numpy')

# Quadruple precision (32 digits)
x, w = gausslobatto(64, backend='mpmath', precision=32)

# Arbitrary precision (64 digits)
x, w = gausslobatto_mpmath(64, precision=64)
```

## Troubleshooting

### mpmath Import Error
If you get an import error for mpmath, install it:
```bash
pip install mpmath>=1.3.0
```

### PyTorch Installation Issues
For PyTorch installation issues, see the [PyTorch installation guide](https://pytorch.org/get-started/locally/).

### Development Installation
For development, install in editable mode:
```bash
git clone <repository-url>
cd phrike
pip install -e .[dev]
```
