# Changelog

All notable changes to PHRIKE are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/).

## Unreleased

### Changed (physics)

- Compressible MHD viscosity and resistivity are now the standard
  visco-resistive MHD on every compressible path (serial reference, GPU fast
  paths and `phrike.parallel`). `viscosity` is the kinematic shear viscosity of
  the Newtonian stress (dynamic viscosity `rho nu`, with the `-2/3 div u` part),
  which enters the momentum flux; the energy flux gains the viscous work
  `-sigma.u` and the Ohmic Poynting flux `eta J x B`, so viscous and Ohmic
  heating are deposited where the energy is dissipated; induction uses
  `-curl(eta J)`. The old `nu lap(rho u)` momentum term, which was not Galilean
  invariant and heated the gas in the wrong place, is removed (the incompressible
  solvers keep `nu lap u`, exact there). New optional keys: `bulk_viscosity`
  and `thermal_conductivity` (heat flux `-kappa grad(p/rho)`). The diffusive
  time step uses `(4/3 nu + nu_bulk) rho_max/rho_min` (new key
  `viscous_step_bound: uniform` drops the density ratio) and the largest
  retained wavenumber in serial runs too; on 1D polynomial grids it uses the
  spectral radius of the actual operator, and on mixed Fourier/SBP grids
  `2 nu + nu_bulk`. On 1D Chebyshev grids the transport terms use a
  summation-by-parts gradient in the Clenshaw-Curtis weights, so they conserve
  energy and are dissipative. On mixed open-y grids every transport flux,
  including the induction term's electric field, uses the SBP adjoint at the
  faces, so face currents are dissipated too. A fixed distributed
  `integration.dt` above the stability limit now logs a warning, and an unmasked
  even Fourier axis with transport warns that its Nyquist mode is not damped.
  Results of runs with nonzero `viscosity` or `resistivity` change.
- `phrike.parallel`: the warm-up settles the kernel tier collectively, and a
  kernel failure during a multi-rank step aborts all ranks instead of letting
  one rank switch tiers; the Torch-tier fluxes are formed a few planes at a
  time and the heating diagnostics a few rows at a time (lower peak memory);
  restarts from checkpoints written before this change log a notice, and
  `history.csv` gains the new columns when appended.

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
