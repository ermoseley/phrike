Add new basis functions
    - Legendre + Filtering near discontinuities
Compare our Legendre implementation to Dedalus. What aren't we getting right, if anything?
Run a convection problem. Set pressure BCs for a stratified medium.
RTI with legendre polynomials
problems/problem_list
move all initial_conditions.py into the problems so it is self-contained.
Fix video so that it uses ffmpeg-python
  - It should also have an enbled: flag. 

After the fork:
Add turbulent driving
Add MPI
Add MHD

I'll add a minimal-spacing based CFL for Legendre grids and re-enable adaptive stepping using that as the initial dt. I'll compute dx_min/dy_min in Grid2D and use them in SpectralSolver2D.compute_dt. Then I'll switch the config back to adaptive=true.


Short answer: the main culprit is CFL underestimation on the Legendre grid, compounded by missing energy-stable boundary treatment for non-periodic Legendre in 2D.

What I see in the code:
- CFL error for Legendre: compute_dt used L/N spacings. On Legendre LGL nodes, min spacing is O(π^2/N^2) near endpoints, so the stable dt is ~100× smaller than using L/N. I’ve added dx_min/dy_min to Grid2D and changed compute_dt to use min(dx_min, dy_min); adaptive is re-enabled. You can see dt ~ 4e-5 now.
- No proper boundary treatment in 2D Legendre: we’re doing strong-form collocation (D·F) without SAT/numerical fluxes and we removed manual BC enforcement. For Euler with walls/outflow this is not energy-stable and blows up quickly even with a correct dt.
- Aliasing/split form: nonlinear fluxes on collocation grids alias; we have modal filtering, but no split-form/flux-differencing. That further destabilizes Legendre compared to periodic Fourier KHI.
- Monitoring uses plain sums; on Legendre, integrals should be quadrature-weighted. The huge “mass_error” is not meaningful.

Minimal fixes to stabilize:
- Keep adaptive, but use min node spacing (done). Consider smaller cfl (0.02–0.05) initially.
- Enforce BCs every RK substage for Legendre:
  - Bottom: v=0; recompute energy from p and u at the wall row.
  - Top/sides: zero-gradient copy.
- Strengthen Legendre modal filtering each step and add a tiny constant viscosity.
- Optionally add a light sponge in the top 10–15% to stabilize the outflow.

Longer-term correct fix:
- Switch to SBP-SAT or split-form/flux-differencing for 2D Legendre so boundaries and nonlinearity are energy-stable.

I looked through the RTI path and the new 2D Legendre integration. Here are the most likely causes of the rapid blow-up, in order of impact:

- Critical CFL mis-estimation for Legendre grid
  - compute_dt uses grid.dx = Lx/Nx and grid.dy = Ly/Ny, but Legendre nodes are clustered near boundaries; the true minimum spacing is O(L/N^2). With Nx=64, Ny=128, the smallest Δy is ~2/128^2 ≈ 1.2e-4, but we’re stepping like Δy ≈ 1/64 ≈ 1.6e-2, exceeding the stable CFL by ~100x at the endpoints.
  - Fix: In the 2D solver, compute dt with min(np.diff(grid.x)) and min(np.diff(grid.y)) instead of L/N. Start with smaller cfl (e.g., 0.02–0.05) for Legendre.

- Missing energy-stable boundary treatment for non-periodic Legendre
  - The 2D Legendre path applies plain nodal derivatives (D·F) with no SAT/numerical flux at boundaries. For Euler, this is not energy stable and will blow up quickly with outflow/wall BCs.
  - KHI works because Fourier is periodic, no boundaries to treat.
  - Minimal stopgap: enforce BCs each RK stage (bottom v=0; copy ghost values for Neumann), and recompute energy consistently after enforcing the wall condition. Proper fix requires SBP-SAT/flux differencing.

- Aliasing/nonlinear stability in Legendre
  - The Euler nonlinear fluxes alias on a collocation grid; Fourier has 2/3-rule dealiasing, but we don’t have an equivalent for Legendre. Modal filtering helps, but without split-form/flux-differencing it’s still fragile.
  - Mitigate: stronger modal filtering every stage and/or a small constant viscosity. Longer-term: skew-symmetric/split-form flux or flux-differencing.

- Monitoring is misleading with Legendre
  - Conservation metrics (mass, energy) are computed as simple sums. On Legendre grids, integrals should use quadrature weights; current mass_error prints are not trustworthy and can look astronomical even if fields are finite.
  - Fix: when basis is Legendre, weight sums by tensor-product LGL weights.

- Gravity/ICs are not the primary culprit
  - The hydrostatic ICs integrate dp/dy = ρ gy correctly for the given ρ(y). The immediate instability and NaNs trace to the two issues above (CFL and boundary/numerical stability), not to the ICs or gravity source.

What I’d change first, minimally:
- In 2D dt calculation, use Δx_min = min(diff(grid.x)), Δy_min = min(diff(grid.y)).
- Reintroduce BC enforcement each RHS/RK stage for the Legendre grid (bottom reflective, top/sides Neumann), including consistent energy recomputation at the wall.
- Strengthen modal filter (alpha 100+, p 8–10, every step) and optionally add tiny constant viscosity.
- Lower cfl to 0.02–0.05 for Legendre until we implement SAT/flux-differencing.

If you want, I can implement the corrected dt for Legendre grids and re-add per-stage BC enforcement now.

Install PyTorch with CUDA support:
Apply to legendre_so...
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