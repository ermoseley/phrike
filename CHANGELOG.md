# Changelog

All notable changes to PHRIKE will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive Sphinx documentation
- API reference with auto-generated docs
- User guide with detailed usage instructions
- Examples guide with code samples
- Development guide for contributors
- Performance benchmarks and scaling analysis
- GPU acceleration support (PyTorch backend)
- Real-time monitoring system
- Checkpoint/restart functionality
- Video generation capabilities
- Spectral filtering options
- Multi-threaded FFT support

### Changed
- Renamed from "Hydra" to "PHRIKE"
- Improved code formatting with Black
- Enhanced type annotations
- Better error handling and validation
- Optimized performance with Numba JIT
- Updated README with comprehensive information

### Fixed
- Backend compatibility issues between NumPy and PyTorch
- Memory management for large problems
- Conservation tracking accuracy
- FFT performance on different platforms
- Video generation reliability

## [0.1.0] - 2024-01-01

### Added
- Initial release of PHRIKE
- 1D, 2D, and 3D Euler equation solvers
- Pseudo-spectral spatial discretization
- RK2 and RK4 time integration schemes
- NumPy backend for CPU computation
- Basic problem implementations:
  - Sod shock tube (1D)
  - Kelvin-Helmholtz instability (2D)
  - Taylor-Green vortex (3D)
  - 3D turbulence
- YAML configuration system
- Command-line interface
- Basic visualization and plotting
- Test suite with validation problems
- Documentation and examples

### Technical Details
- **Spatial Discretization**: Pseudo-spectral methods with FFT
- **Time Integration**: Runge-Kutta schemes (RK2, RK4)
- **Dealiasing**: 2/3-rule and exponential filtering
- **Conservation**: Mass, momentum, and energy tracking
- **Backend**: NumPy with SciPy FFT
- **Performance**: Numba JIT compilation for critical kernels
- **Testing**: Pytest-based test suite
- **Documentation**: Sphinx with ReadTheDocs integration

### Known Issues
- Limited GPU support (CPU-only)
- Basic monitoring capabilities
- Limited visualization options
- No checkpoint/restart functionality
- Basic error handling

## [0.0.1] - 2023-12-01

### Added
- Initial development version
- Basic 1D solver implementation
- Sod shock tube problem
- Simple configuration system
- Basic testing framework

---

## Version History Summary

### v0.1.0 (Current)
- **Focus**: Core functionality and stability
- **Features**: Multi-dimensional solvers, basic problems
- **Target**: Research and education use

### v0.2.0 (Planned)
- **Focus**: Performance and usability
- **Features**: GPU acceleration, advanced monitoring
- **Target**: Production simulations

### v0.3.0 (Planned)
- **Focus**: Advanced features and validation
- **Features**: More problem types, advanced visualization
- **Target**: Comprehensive CFD toolkit

### v1.0.0 (Planned)
- **Focus**: Maturity and stability
- **Features**: Full feature set, extensive validation
- **Target**: Production-ready CFD solver

---

## Migration Guide

### From v0.0.x to v0.1.0

**Breaking Changes:**
- Package renamed from "hydra" to "phrike"
- Configuration file format updated
- API changes in problem classes

**Migration Steps:**
1. Update import statements:
   ```python
   # Old
   import hydra
   
   # New
   import phrike
   ```

2. Update configuration files:
   ```yaml
   # Old format
   problem_name: sod
   
   # New format
   problem: sod
   ```

3. Update CLI usage:
   ```bash
   # Old
   python -m hydra sod
   
   # New
   phrike sod
   ```

### From v0.1.x to v0.2.0

**New Features:**
- GPU acceleration support
- Enhanced monitoring system
- Checkpoint/restart functionality

**Migration Steps:**
1. Install PyTorch for GPU support:
   ```bash
   pip install torch
   ```

2. Update configuration for monitoring:
   ```yaml
   monitoring:
     enabled: true
     step_interval: 10
   ```

3. Use new GPU backend:
   ```python
   solver, history = phrike.run_simulation(
       problem_name="sod",
       backend="torch",
       device="cuda"
   )
   ```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to PHRIKE.

## Support

- **Issues**: [GitHub Issues](https://github.com/your-username/phrike/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/phrike/discussions)
- **Documentation**: [Read the Docs](https://phrike.readthedocs.io/)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
