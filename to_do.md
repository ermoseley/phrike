Add new basis functions
    - Chebyshev + Filtering near discontinuities
Compare our Chebyshev/Legendre implementation to Dedalus. What aren't we getting right, if anything?
Run a convection problem. Set pressure BCs for a stratified medium.
Compare Dedalus to PHRIKE


After the fork:
Add turbulent driving
Add MPI
Add MHD

Install PyTorch with CUDA support:
Apply to chebyshev_so...
cu118
Consider using float32 for large problems:
Modify the grid initialization to use torch.float32 instead of torch.float64
This would roughly double the effective memory and speed
Enable multi-GPU support (future enhancement):
The current architecture could be extended for multi-GPU simulations
PyTorch's DistributedDataParallel could be integrated
Optimize FFT operations:
Consider using cuFFT directly for even better performance
Batch multiple FFT operations when possible


Longer term:
Upgrade to spectral element method (SEM)

## Standard Next Steps for PHRIKE Development:

### 1. Code Quality & Documentation 📚
- [ ] API Documentation: Generate comprehensive docs (Sphinx, ReadTheDocs)
- [ ] Code Comments: Add detailed docstrings to all public functions
- [ ] Type Hints: Complete type annotations for better IDE support
- [ ] Code Style: Run `black`, `flake8`, `mypy` for consistent formatting
- [ ] README: Update with installation, examples, and performance benchmarks

### 2. Testing & Validation ✅
- [ ] Unit Tests: Test individual functions (equations, solvers, utilities)
- [ ] Integration Tests: Test complete workflows end-to-end
- [ ] Regression Tests: Ensure changes don't break existing functionality
- [ ] Validation Tests: Compare against known analytical solutions
- [ ] CI/CD: Set up GitHub Actions for automated testing

### 3. Scientific Validation 🔬
- [ ] Benchmark Problems: Implement standard CFD test cases
  - [ ] 1D: Sod shock tube, Woodward-Colella blast wave
  - [ ] 2D: Double Mach reflection, Rayleigh-Taylor instability
  - [ ] 3D: Taylor-Green vortex, decaying turbulence
- [ ] Literature Comparison: Compare results with published papers
- [ ] Convergence Studies: Verify theoretical convergence rates
- [ ] Conservation Tests: Ensure mass/momentum/energy conservation

### 4. Performance Optimization ⚡
- [ ] Profiling: Identify bottlenecks with `cProfile`, `line_profiler`
- [ ] Memory Optimization: Reduce memory footprint, optimize data structures
- [ ] Algorithm Improvements: Implement better time-stepping, FFT optimizations
- [ ] Parallelization: Add OpenMP for CPU, optimize GPU kernels
- [ ] Caching: Implement smart caching for repeated computations

### 5. Usability & Features 🚀
- [ ] Configuration System: Improve YAML configuration with validation
- [ ] Visualization: Better plotting tools, 3D visualization
- [ ] Output Formats: Support HDF5, VTK, NetCDF for data exchange
- [ ] Restart Capability: Robust checkpoint/restart system
- [ ] Error Handling: Better error messages and recovery
- [ ] Logging: Structured logging system

### 6. Distribution & Deployment 📦
- [ ] Package Management: Proper `pyproject.toml`, versioning
- [ ] Dependencies: Pin versions, handle optional dependencies
- [ ] Installation: `pip install`, conda packages
- [ ] Docker: Container images for reproducible environments
- [ ] Documentation: User guide, developer guide, API reference

### 7. Community & Collaboration 👥
- [ ] Open Source: Clear license, contribution guidelines
- [ ] Issue Tracking: GitHub issues for bugs and feature requests
- [ ] Code Review: Pull request templates, review process
- [ ] Examples: Jupyter notebooks, tutorial scripts
- [ ] Papers: Write and publish methodology papers