GPU performance
===============

This page records how the 3D Euler and 3D MHD solvers were profiled and sped
up on GPUs.
The benchmark is ``scripts/bench_hydro3d.py``: a Taylor-Green vortex with
dealiasing on, spectral dissipation (``p = 8``, ``rate = 40``), and the step
capped at CFL 0.2, run through the same loop body as ``SpectralSolver3D.run``.
Throughput is in Mzups per accepted step, that is, 10\ :sup:`6` cells advanced
by one completed time step per second. The script also prints rejected attempts
and simulated time per wall-clock second.

Hardware: Princeton Stellar, one NVIDIA A100-PCIE-40GB (1.55 TB/s HBM2),
PyTorch 2.14 with CUDA 12.6.

Results
-------

Reference path versus fast path, Mzups per accepted step:

.. list-table::
   :header-rows: 1

   * - Scheme
     - Precision
     - 128\ :sup:`3`
     - 256\ :sup:`3`
   * - RK4
     - fp64
     - 40 → 237 (5.9x)
     - 41 → 289 (7.0x)
   * - RK4
     - fp32
     - 81 → 421 (5.2x)
     - 80 → 524 (6.6x)
   * - RK45 (error controlled)
     - fp64
     - 23 → 134 (6.0x)
     - 23 → 160 (6.8x)

Turning spectral dissipation off removes two inverse transforms per step (see
below). Without it the RK4 fp64 step at 256\ :sup:`3` measured 44 → 350 Mzups
(8.0x). The fast path uses about half the memory of the reference path, so
384\ :sup:`3` in fp64 and 512\ :sup:`3` in fp32 (675 Mzups) fit on the 40 GB
card, where the reference path runs out of memory. For large grids set
``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``.

On an Apple M-series laptop (MPS, fp32, 128\ :sup:`3`, RK4) the fast path runs
at 11 Mzups eager and 16-23 Mzups with the ``torch.compile`` Metal backend,
which is the default on MPS. The reference path runs at 4.6 Mzups.

Choosing a time scheme
~~~~~~~~~~~~~~~~~~~~~~

The stepper caps every embedded-RK proposal at the CFL step. RK45 therefore
takes the same step as RK4 while evaluating the right-hand side 7 times
instead of 4. FSAL does not apply, because the accepted state is dissipated
and projected. For these CFL-limited runs RK4 at a CFL near its stability
limit is the cheaper choice. With 2/3 dealiasing, the largest derivative
eigenvalue gives an RK4 limit of roughly CFL 0.7 in 3D, 0.95 in 2D and 1.35 in
1D. The shipped configurations use 0.2-0.25.

Profiles
--------

The reference path (256\ :sup:`3`, fp64, 392 ms per step before split-step
dissipation was added) spent 40% of its time in cuFFT and 60% in elementwise
kernels. Those were complex mask and derivative multiplies, ``.real`` copies,
``torch.stack`` and additions. Nsight Compute showed every hot kernel, the
cuFFT passes included, already at 82-89% of DRAM bandwidth. Speed therefore
had to come from moving fewer bytes, not from tuning kernels. Each RHS
evaluation took 15 forward and 15 inverse full complex FFTs, one pair per
flux component.

In the fast path (48 ms per step without dissipation), cuFFT passes take 76%
of the step at 74-84% of DRAM bandwidth. The fused flux kernel takes 12% at
80%, the Triton stage kernel 8% at 86%, and the wave-speed reduction 1%.

What the fast path does
-----------------------

``phrike/spectral3d.py`` and ``phrike/spectral3d_triton.py`` implement it:

- Real-to-complex transforms, with half-spectrum storage.
- The flux divergence is formed in spectral space, so each stage needs one
  inverse transform per conserved variable (5) instead of one per flux
  component (15).
- Nine flux transforms instead of fifteen. The momentum flux
  ``rho u_i u_j + p delta_ij`` is symmetric (6 fields) and the energy flux
  ``(E + p) u_i`` has 3. The mass flux is the momentum, whose spectrum is
  already in the stage state.
- RK stages are combined in spectral space. The end-of-step dissipation and
  dealias projection become a multiply in the final stage, and the pre-step
  dissipation half step scales the cached spectrum in place.
- On CUDA, one Triton kernel per stage forms the spectral RHS of all five
  variables and applies the update. Once the state is band-limited it visits
  only the 8/27 of modes the 2/3 rule keeps; ``run()`` projects the initial
  state, so that holds on every step. Any other input is stepped over all
  modes, as in the reference path.
- Spectral buffers use the ``norm="forward"`` convention, so no separate
  normalization pass is needed. The state spectrum is cached between steps,
  keyed on tensor identity and ``_version``.
- Pointwise work is fused with ``torch.compile``: fluxes together with
  positivity-floor counts, and the finite check together with the wave speed.
  RK weights are passed as device scalars, so a new ``dt`` does not trigger a
  recompile.
- Embedded RK (rk23, rk45, rk78) runs on the same machinery. Stage right-hand
  sides are stored on the retained modes only. The error estimate costs one
  inverse transform, and the cached state is untouched until an attempt is
  accepted.
- The positivity clamp leaves a positive state untouched. The fast path counts
  floor violations of every limited stage and of the pre-projection state on
  the device. If any value would be clamped, it returns ``None`` and the solver
  repeats that step on the reference path. Checking the pre-projection state
  costs one extra inverse transform per step when dissipation is on.

The fast path is on by default for Torch grids (CUDA, MPS, CPU) and off for
NumPy grids, which keep the bitwise reference regressions. Use
``SpectralSolver3D(..., spectral_fast_path=True/False)`` to choose explicitly,
or set ``PHRIKE_COMPILE=0`` to disable only the compiled and Triton kernels.
Subclassed equations and runs with gravity keep the reference path.

Correctness
-----------

``tests/test_spectral3d.py`` compares the fast path with the reference to
round-off: 1e-12 relative in fp64 and 1e-5 in fp32. It covers:

- RK2 and RK4 with dissipation on and off, and with dealiasing off.
- Odd and anisotropic grids, and a white-noise state with content above the
  cutoff.
- NumPy, Torch CPU, and CUDA with Triton.
- The positivity-clamp fallback, which is bitwise equal to the reference.
- The Triton embedded step against the reference rk23 and rk45 at fixed
  ``dt``.
- Whole adaptive runs, on NumPy and on CUDA with forced rejections. They
  produce identical accept and reject sequences, and the states agree to 1e-9.
  They are not closer because the controller turns round-off in the reference
  error estimate, which is a difference of two O(\|U\|) states, into tiny
  ``dt`` changes.

3D MHD
------

``SpectralSolverMHD`` steps 3D runs on Torch grids through
``phrike/spectral_mhd3d.py``. The benchmark is ``scripts/bench_mhd3d.py``: the
3D Orszag-Tang vortex with the settings in ``configs/orszag_tang3d.yaml``
(projection after every substage, dissipation ``rate = 300`` with onset 0.5,
CFL 0.25). On the A100, in Mzups per accepted step:

.. list-table::
   :header-rows: 1

   * - Scheme
     - Precision
     - 64\ :sup:`3`
     - 128\ :sup:`3`
     - 256\ :sup:`3`
   * - RK4
     - fp64
     - 12 → 95 (7.9x)
     - 19 → 166 (8.7x)
     - 19 → 195 (10.0x)
   * - RK4
     - fp32
     - 17 → 123 (7.4x)
     - 40 → 301 (7.5x)
     - 38 → 363 (9.6x)
   * - RK45
     - fp64
     -
     -
     - 14 → 107 (7.9x)

Peak memory halves (256\ :sup:`3` fp64: 22.5 → 11.4 GB). 384\ :sup:`3` fp32
now fits (19 GB, 309 Mzups). The reference step spent 36% of its 887 ms in
cuFFT and 64% in elementwise kernels, all near DRAM bandwidth. Each stage
transformed all 24 flux components, projected B with six more transforms, and
rebuilt the conserved state in the clamp. The fast step (78 ms of GPU time at
256\ :sup:`3` fp64) spends 75% in cuFFT, 12% in the fused flux kernel and 8% in
the Triton stage kernel. About 9 ms of the wall time is the device copy that
``torch.fft.irfftn`` makes of its input. That copy is needed because cuFFT's
complex-to-real transform overwrites the input, and splitting the transform to
avoid it was slower.

How the fast path works:

- Each stage transforms 12 flux fields: the symmetric Maxwell-stress momentum
  flux (6), the energy flux (3) and the antisymmetric induction tensor
  ``A_ij = u_i B_j - u_j B_i`` (3). The mass flux is the stage momentum, and
  resistive and viscous Laplacians enter the spectral RHS directly.
- The spectral RHS of B is ``i k_j A_ji``, which is solenoidal because ``A`` is
  antisymmetric. A stage state built from a solenoidal, band-limited state is
  therefore already projected to round-off. The Leray projection of B is
  applied once per step, in the final stage, so div B stays at round-off
  (about 1e-13 in fp64 at 256\ :sup:`3`).
- As in the hydro path, floor violations of every limited state are counted on
  the device. A step whose clamps would act is repeated on the reference path.
  Its output seeds the spectral cache only when the accepted-state clamp left
  it band-limited.
- The first step of a run, and runs with Hall or ambipolar terms, artificial
  viscosity, ``divergence_clean: none`` or other equation classes, use the
  reference path.

``tests/test_spectral_mhd3d.py`` checks RK2/RK4 with dissipation on and off,
resistivity and viscosity, and ``project_each_substage`` off, at 1e-11 in fp64
and 1e-5 in fp32 on Torch CPU and CUDA. It also covers whole adaptive runs
(the same accept and reject sequence), the embedded step at fixed ``dt``, the
clamp fallback, and div B.

Remaining headroom
------------------

FFT passes are about three quarters of the step and run near DRAM bandwidth,
so this algorithm on cuFFT is close to its roofline. Two changes could
plausibly give another 1.3-1.5x:

- Fuse the flux evaluation with the x-direction FFT passes, which needs a
  device-side FFT.
- Prune the transforms to the retained modes. A prototype using torch's
  strided 2D FFTs saved about 19% of the forward transform in fp64 and nothing
  in fp32. Raw cuFFT advanced-layout plans were slower.

An H100 (3.35 TB/s) should give roughly 2x over these A100 numbers. Grids of
64\ :sup:`3` and smaller are partly bound by kernel launches. CUDA graphs gave
only about 10% there, so they are not used.

Reproducing
-----------

.. code-block:: bash

   python scripts/bench_hydro3d.py --N 256 --device cuda --precision double
   python scripts/bench_hydro3d.py --N 256 --device cuda --precision double --legacy
   python scripts/bench_hydro3d.py --N 256 --device cuda --scheme rk45
   nsys profile -t cuda,nvtx -o prof python scripts/bench_hydro3d.py --N 256 --steps 3 --nvtx
