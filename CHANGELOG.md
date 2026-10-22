# Changelog

All notable changes to PHRIKE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Changed

- Document the equations that are integrated, the shared YAML keys, the
  CFL definition, and the spectral-dissipation rate. The adaptive Fehlberg
  integrator is named `rk78` (Fehlberg 7(8)). The strings `RKF78` and
  `rkf78` are not accepted.
- Require Python 3.10 or newer, matching the versions exercised in CI.
- Include `configs/` and the Sphinx sources in the sdist, and check those
  paths in CI. Remove the unused SymPy extra. Declare optional FINUFFT
  for off-grid Fourier evaluation.
- Run the test suite in CI against a CPU-only PyTorch wheel. Lint is not
  a required check.

## 0.1.0

- 1D, 2D, and 3D compressible Euler and ideal MHD solvers on Fourier,
  Legendre, and 1D Chebyshev grids, with NumPy and optional PyTorch
  backends (CPU, CUDA, and Metal/MPS).
- Ideal MHD with Helmholtz projection of the magnetic field, plus
  Alfvén-wave, Brio–Wu, Orszag–Tang, ambipolar-diffusion shear, and 3D
  MHD turbulence problems.
- Midpoint-resolvent ambipolar diffusion for 2D MHD.
- 1D fixed, outflow, and wall boundaries, and 2D Fourier-x/SBP-y grids
  with open transverse faces.
- Hard 2/3 Fourier dealiasing and timestep-aware spectral dissipation.
- Lagrangian tracer particles and tracer-density reconstruction.
- Fixed-step RK2 and RK4, and embedded RK23, RK45, and rk78
  (Fehlberg 7(8)).
- Source-tree post-processing and diagnostic helpers under `scripts/`.
