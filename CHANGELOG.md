# Changelog

All notable changes to PHRIKE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Added

- Ideal MHD solvers in 1D, 2D, and 3D with Helmholtz projection of the
  magnetic field, plus Alfvén-wave, Brio–Wu, Orszag–Tang, and 3D MHD
  turbulence problems.
- Midpoint-resolvent ambipolar diffusion for 2D MHD and the single-layer
  ambipolar-diffusion shear instability (`adsi2d`) problem.
- A 1D Chebyshev–Lobatto basis, 1D fixed/outflow/wall boundary conditions,
  and 2D Fourier-x/SBP-y grids with approximately open transverse faces.
- Timestep-aware spectral dissipation (`integration.spectral_dissipation`)
  alongside the hard 2/3 Fourier dealias projection.
- Lagrangian tracer particles and tracer-density reconstruction.
- Explicit backend selection for NumPy CPU, PyTorch CPU, CUDA, and Metal/MPS.
- An RKF78 adaptive integrator with tests for its tableau and controller.

### Changed

- Consolidated post-processing, benchmark, and validation helpers under
  `scripts/`.
- Tightened packaging, documentation, and continuous-integration checks.

## 0.1.0

- Initial research-code version: 1D/2D/3D compressible Euler solvers on
  Fourier and Legendre grids with NumPy and PyTorch backends.
