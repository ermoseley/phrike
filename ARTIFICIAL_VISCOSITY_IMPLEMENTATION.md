# Artificial Viscosity Implementation for PHRIKE

## Overview

This document describes the implementation of gradient-based artificial viscosity for the PHRIKE pseudo-spectral CFD solver. The artificial viscosity module provides stabilization for shocks and steep gradients while preserving smooth regions of the solution.

## Features

- **Multi-dimensional Support**: Works with 1D, 2D, and 3D simulations
- **Gradient-based Sensing**: Uses spectral derivatives to detect non-smooth regions
- **Tunable Parameters**: Comprehensive configuration options for different problem types
- **Spectral Consistency**: Uses pseudo-spectral differentiation throughout
- **Periodic Boundary Conditions**: Fully compatible with spectral methods
- **High-order Time Integration**: Compatible with RK2, RK4, and adaptive schemes
- **Diagnostic Output**: Optional visualization of viscosity application

## Architecture

### Core Components

1. **`ArtificialViscosityConfig`**: Configuration dataclass with all tunable parameters
2. **`GradientBasedSensor`**: Computes smoothness sensor using gradient magnitude
3. **`ArtificialViscosityCoefficient`**: Calculates viscosity coefficient from sensor
4. **`SpectralArtificialViscosity`**: Main class orchestrating the complete process

### Integration Points

- **Solver Integration**: Seamlessly integrated into `SpectralSolver1D/2D/3D`
- **Configuration System**: YAML configuration support via `BaseProblem`
- **RHS Computation**: Added to right-hand side computation functions
- **Time Integration**: Compatible with all time-stepping schemes

## Mathematical Formulation

### Smoothness Sensor

For a variable $q$ (density, pressure, or velocity magnitude), the smoothness sensor is defined as:

$$s = \frac{|\nabla q|}{|q| + \epsilon}$$

where:
- $\nabla q$ is the gradient of $q$ computed using spectral differentiation
- $\epsilon$ is a small number to avoid division by zero (default: $10^{-12}$)

### Viscosity Coefficient

The local viscosity coefficient is computed as:

$$\nu = \nu_{\max} \left(\frac{s}{s_{\text{ref}}}\right)^p \cdot H(s - s_{\min})$$

where:
- $\nu_{\max}$ is the maximum viscosity coefficient
- $s_{\text{ref}}$ is the reference smoothness threshold
- $p$ is the exponent controlling sharpness of the transition
- $H(s - s_{\min})$ is the Heaviside function (applies viscosity only where $s > s_{\min}$)

### Viscosity Terms

The artificial viscosity is applied as:

$$\frac{\partial U_i}{\partial t} = \text{RHS}_i + \nabla \cdot (\nu \nabla U_i)$$

where $U_i$ are the conservative variables and the divergence is computed using spectral differentiation.

## Configuration

### YAML Configuration

```yaml
artificial_viscosity:
  enabled: true                    # Enable/disable artificial viscosity
  nu_max: 1e-3                    # Maximum viscosity coefficient
  s_ref: 1.0                      # Reference smoothness threshold
  s_min: 0.1                      # Minimum smoothness for applying viscosity
  p: 2.0                          # Exponent for smoothness scaling
  epsilon: 1e-12                  # Small number to avoid division by zero
  sensor_variable: "density"       # Variable to use for smoothness sensing
  diagnostic_output: false         # Output diagnostic information
  variable_weights:               # Weights for different conserved variables
    density: 1.0
    momentum_x: 1.0
    momentum_y: 1.0
    momentum_z: 1.0
    energy: 1.0
```

### Parameter Guidelines

#### 1D Problems (Shock Tubes, Acoustic Waves)
- `nu_max`: 1e-3 to 1e-2
- `s_ref`: 0.5 to 1.0
- `s_min`: 0.05 to 0.2
- `p`: 1.5 to 2.0
- `sensor_variable`: "density" or "pressure"

#### 2D Problems (KHI, Vortex Dynamics)
- `nu_max`: 5e-4 to 2e-3
- `s_ref`: 0.3 to 0.8
- `s_min`: 0.02 to 0.1
- `p`: 1.0 to 1.5
- `sensor_variable`: "density"

#### 3D Problems (TGV, Turbulence)
- `nu_max`: 2e-4 to 1e-3
- `s_ref`: 0.2 to 0.5
- `s_min`: 0.01 to 0.05
- `p`: 1.0 to 1.2
- `sensor_variable`: "velocity_magnitude"

## Usage Examples

### Basic Usage

```python
from phrike.artificial_viscosity import create_artificial_viscosity

# Create artificial viscosity module
av_config = {
    "enabled": True,
    "nu_max": 1e-3,
    "s_ref": 1.0,
    "s_min": 0.1,
    "p": 2.0
}
artificial_viscosity = create_artificial_viscosity(av_config)

# Use in solver
solver = SpectralSolver1D(
    grid=grid,
    equations=equations,
    artificial_viscosity_config=av_config
)
```

### Configuration File Usage

```bash
# Run with artificial viscosity
phrike sod --config configs/sod_artificial_viscosity.yaml

# Run comparison
python examples/run_sod_artificial_viscosity.py --compare
```

### Programmatic Usage

```python
import phrike

# Run simulation with artificial viscosity
solver, history = phrike.run_simulation(
    problem_name="sod",
    config_path="configs/sod_artificial_viscosity.yaml"
)

# Access artificial viscosity diagnostics
if solver.artificial_viscosity and solver.artificial_viscosity.config.diagnostic_output:
    diagnostics = solver.artificial_viscosity.get_diagnostics()
    sensor = diagnostics["sensor"]
    viscosity = diagnostics["viscosity"]
```

## Implementation Details

### Spectral Differentiation

The module uses the existing spectral differentiation methods from the grid classes:
- 1D: `grid.dx1()`
- 2D: `grid.dx1()`, `grid.dy1()`
- 3D: `grid.dx1()`, `grid.dy1()`, `grid.dz1()`

### Performance Optimizations

- **Numba JIT Compilation**: Critical kernels are compiled for speed
- **Vectorized Operations**: All computations use NumPy vectorization
- **Memory Efficiency**: Minimal memory allocation during computation
- **GPU Compatibility**: Works with PyTorch backend

### Error Handling

- **Finite Value Checks**: All computations are checked for finite values
- **Shape Validation**: Array shapes are validated at runtime
- **Graceful Degradation**: Falls back to zero viscosity if errors occur

## Testing

The implementation includes comprehensive tests covering:

- **Unit Tests**: Individual component functionality
- **Integration Tests**: Solver integration
- **Multi-dimensional Tests**: 1D, 2D, 3D functionality
- **Performance Tests**: Conservation and smoothness properties
- **Configuration Tests**: YAML parsing and validation

Run tests with:
```bash
pytest tests/test_artificial_viscosity.py -v
```

## Validation

### Conservation Properties

The artificial viscosity implementation preserves:
- **Mass Conservation**: Viscosity terms are conservative
- **Momentum Conservation**: Proper treatment of momentum equations
- **Energy Conservation**: Consistent with energy equation

### Smooth Solution Preservation

- **Low Damping**: Smooth regions experience minimal damping
- **Threshold Control**: `s_min` parameter controls activation threshold
- **Gradient-based**: Only activates where gradients are large

### Shock Stabilization

- **Shock Capturing**: Effectively stabilizes shock waves
- **Gradient Detection**: Accurately identifies non-smooth regions
- **Appropriate Viscosity**: Provides sufficient but not excessive damping

## Performance Impact

### Computational Overhead

- **RHS Computation**: ~20-30% increase in RHS computation time
- **Memory Usage**: Minimal additional memory requirements
- **Scalability**: Scales well with problem size

### Benefits

- **Stability**: Enables stable simulation of challenging problems
- **Accuracy**: Preserves solution accuracy in smooth regions
- **Robustness**: Reduces sensitivity to initial conditions and parameters

## Troubleshooting

### Common Issues

1. **Excessive Damping**: Reduce `nu_max` or increase `s_min`
2. **Insufficient Stabilization**: Increase `nu_max` or decrease `s_min`
3. **Poor Gradient Detection**: Try different `sensor_variable`
4. **Numerical Issues**: Check `epsilon` parameter

### Debugging

Enable diagnostic output to visualize viscosity application:
```yaml
artificial_viscosity:
  diagnostic_output: true
```

This creates plots showing:
- Smoothness sensor values
- Viscosity coefficient distribution
- Viscosity terms for each variable
- Sensor vs viscosity correlation

## Future Enhancements

### Planned Features

- **Adaptive Viscosity**: Dynamic adjustment based on solution evolution
- **Multi-scale Sensing**: Different sensors for different length scales
- **Advanced Sensors**: Pressure-based and entropy-based sensors
- **GPU Optimization**: Enhanced GPU performance

### Research Directions

- **Machine Learning**: ML-based viscosity prediction
- **Multi-physics**: Extension to MHD and other systems
- **High-order Methods**: Integration with spectral element methods

## References

1. Jameson, A., Schmidt, W., & Turkel, E. (1981). Numerical solutions of the Euler equations by finite volume methods using Runge-Kutta time-stepping schemes.
2. Shu, C.-W. (1998). Essentially non-oscillatory and weighted essentially non-oscillatory schemes for hyperbolic conservation laws.
3. Lele, S. K. (1992). Compact finite difference schemes with spectral-like resolution.

## Contact

For questions or issues related to the artificial viscosity implementation, please:
- Open an issue on GitHub
- Contact the development team
- Check the documentation and examples
