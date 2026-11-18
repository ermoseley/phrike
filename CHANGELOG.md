# Changelog

All notable changes to PHRIKE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Added

- `phrike.parallel`: the 3D Euler and MHD spectral fast path on many GPUs
  (MPI slab decomposition, one rank per GPU, truncated distributed FFT), run
  with `python -m phrike.parallel`. With one rank the step equals the
  reference solver to round-off; any rank count gives the one-rank result.
  Fixed and embedded Runge-Kutta, the positivity limiter, per-rank spectral
  snapshots, and restart on a different rank count or resolution.
- Incompressible visco-resistive MHD and Navier-Stokes solvers on the same
  decomposition (`imhd_turb`, `imhd_orszag_tang`, `imhd_alfven`, `ins_turb`,
  `ins_taylor_green`): exact integrating factor for viscosity and
  resistivity, RK2/RK3/RK4, 14 transforms per stage for MHD. The solver
  also supports even-order hyperdissipation and power-law OU forcing.
  The 288³ and 2048³ fluctuation-dynamo examples use the same physical coefficients.
- Large-scale forcing for driven turbulence (`forcing.type`: `ou`, `white`,
  `constant_power`) and random initial fields that do not depend on the rank
  count. The compressible acceleration is synthesized on the device from the
  modes each rank owns; the band respects the spherical truncation.
- `phrike.parallel`: `parallel.transpose: nccl` (or `PHRIKE_MPI_TRANSPOSE=nccl`,
  `--transpose nccl`) exchanges the FFT transposes with `torch.distributed`
  (NCCL on CUDA, Gloo on CPU), rendezvousing through MPI and needing neither a
  CUDA-aware MPI nor host staging. If the group cannot start on any rank, all
  ranks fall back to `alltoallv` together. Multi-GPU runs select NCCL by default
  and overlap FFT batches. The benchmark reports exposed wait time with overlap
  and effective exchange bandwidth with overlap disabled.
- `phrike.parallel`: selectable analysis output. `io.format` (`auto`, `hdf5`,
  `npy`) sets the format of fields, slices, spectra and diagnostics, and the
  new `io.fields` block writes physical-space fields (`u`, `b`, `omega`, `j`;
  compressible `rho`, `u`, `p`, `B`, ...) in double, single or half precision,
  one variable at a time. `io.fields.coarsen` writes them on a coarser grid as
  the exact spectral low-pass of the field, redistributed from the fine to the
  coarse decomposition with one all-to-all. HDF5 output is one file (parallel
  h5py) or per-rank parts stitched by virtual datasets (serial h5py), with
  per-output XDMF files and a `fields.xmf` time series that is rebuilt after a
  restart; `phrike.parallel.output.load_fields` reads either format.
  `io.diagnostics` calls the analysis module (`phrike.parallel.analysis`) at the
  output times. Slices of incompressible runs no longer build the physical
  state, and slices and spectra are `.h5` by default when h5py imports (set
  `io.format: npy` for the previous `.npz` files). Restart snapshots are unchanged.
- `python -m phrike.parallel.selftest`, `scripts/bench_parallel.py` (with
  `--forcing`),
  `scripts/check_gpu_aware_mpi.py` and a Slurm template.

### Fixed

- `phrike.parallel` compressible MHD: after the accepted-state positivity
  limiter clamps, the next step now follows the reference (the clamped state's
  out-of-band content is damped, enters every stage and is removed at the end
  of that step) instead of deviating by percent-level errors. The content is
  saved in snapshots and restored on a restart with the same grid and rank
  count. `strict_clamp` now defaults to true, so floor violations before the
  final dissipation are counted as in the serial solver.
- `phrike.parallel` Triton tier: warm-up exercises the kernels on each rank's
  real local shapes, and a kernel failure after warm-up aborts the job instead
  of leaving one rank on a different collective sequence (a hang).
- 1D `outflow` and `wall` boundaries on Chebyshev/Legendre grids are now
  imposed weakly with an SBP/Rusanov boundary flux (far-field and mirror
  exterior states), consistent with the overintegrated flux and with a
  zero-flux closure for diffusion terms. The previous copy-the-neighbour
  closure was unstable: a smooth acoustic pulse grew instead of leaving.
- With overintegration, the 1D wall flux no longer leaks mass and energy at a
  discontinuity next to the wall.
- Checkpoints store the 1D and mixed-grid boundary reference, so a restarted
  `outflow` run keeps its original far-field state.

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
