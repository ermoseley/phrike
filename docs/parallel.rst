Multi-GPU runs and incompressible MHD
=====================================

``phrike.parallel`` runs the 3D periodic Fourier solvers on many GPUs, one MPI
rank per GPU, and adds an incompressible visco-resistive MHD solver.  The same
code runs on a single device when no MPI launcher is present.

.. code-block:: bash

   python -m phrike.parallel --list
   python -m phrike.parallel imhd_turb --config configs/parallel/imhd_driven_turb.yaml
   srun -n 64 python -m phrike.parallel imhd_turb --config configs/parallel/imhd_driven_turb.yaml

Systems and problems
--------------------

.. list-table::
   :header-rows: 1

   * - System
     - Problems
     - Stepper
   * - compressible Euler (``euler``)
     - ``tgv3d``, ``turb3d``
     - ``CompressibleSlabStepper``: the fast path of
       :doc:`performance` on a slab decomposition
   * - compressible MHD (``mhd``)
     - ``orszag_tang3d``, ``alfven3d``, ``mhd_turb3d``
     - same
   * - incompressible Navier-Stokes (``ins``)
     - ``ins_taylor_green``, ``ins_turb``
     - ``IncompressibleSlabStepper``
   * - incompressible MHD (``imhd``)
     - ``imhd_turb``, ``imhd_orszag_tang``, ``imhd_alfven``
     - same

The compressible steppers take the reference solver's step: with one rank the
result equals ``SpectralSolverMHD3D`` and ``SpectralSolver3D`` to round-off,
and with any number of ranks it equals the one-rank result to round-off
(``tests/test_parallel_compressible.py``).  Hall, ambipolar, shearing-sheet,
gravity and artificial-viscosity terms are not available here and raise.

Decomposition and transforms
----------------------------

Physical space is cut into z slabs, spectral space into ky slabs.  Spectral
arrays hold only the modes kept by the 2/3 rule, ``|n_i| <= (N_i - 1) // 3``,
which is 8/27 of the half spectrum.  A forward transform is a batched 2D
real-to-complex FFT in (y, x) on each rank, truncation, one all-to-all
exchange, and a 1D FFT in z; the inverse runs the other way with zero padding.
Truncating before the exchange and padding after it means a transform
communicates 4/9 of the grid's data, once.

What that implies for scaling:

* Under weak scaling (fixed points per GPU) every rank does the same local
  work and sends its whole truncated block once per transform, so the bytes
  per rank per step are constant.  The step needs one small ``MPI_Allreduce``.
  Throughput per GPU is then constant up to the :math:`\log N` factor of the
  transforms, provided the network delivers a constant bandwidth per node.
* The number of ranks is limited to ``min(Nz, Ky)`` with
  ``Ky = 2 ((Ny - 1) // 3) + 1``; keep at least 8 planes per rank.  For
  :math:`2048^3` that is a few hundred GPUs.  Beyond it a pencil (2D)
  decomposition is needed, at the price of a second exchange per transform.
* Transforms per step: 80 for compressible MHD with RK4 (12 flux fields
  forward and 8 state fields inverse per stage), 56 for compressible Euler,
  56 for incompressible MHD (8 forward, 6 inverse) and 32 for incompressible
  Navier-Stokes.  Split-step ``spectral_dissipation`` adds one inverse
  transform of the state per step.

Use FFT-friendly sizes (products of 2, 3, 5, 7).  ``timing.json`` in the output
directory and ``python -m scripts.bench_parallel`` report seconds per step,
zone-cycles per second per rank, megabytes sent per step and the time in the
local transforms, packing and MPI.

MPI and GPUs
------------

Install ``mpi4py`` against the cluster's MPI (``pip install --no-binary=mpi4py
mpi4py`` with that MPI's ``mpicc`` on the path).  Launch one rank per GPU; a
rank takes device ``node_rank % visible_devices``, so both ``srun
--gpus-per-task=1`` and several ranks sharing a node's device list work.

By default the exchange buffers are staged through pinned host memory, which
works with any MPI.  With a CUDA-aware MPI the device buffers go to MPI
directly: run ``srun -n 2 python -m scripts.check_gpu_aware_mpi``, and if it
reports that device buffers work set ``parallel.gpu_aware_mpi: true`` or
``PHRIKE_GPU_AWARE_MPI=1``.  Do this before a large run; staging costs two
extra copies of everything communicated.

``python -m phrike.parallel.selftest`` (serially or under ``srun``) checks the
distributed FFT, every Triton kernel against its Torch counterpart, both
solver tiers against each other and an exact solution, on the device and MPI
the run will use.  A kernel that fails to build is replaced by Torch ops at
run time with a warning, so run the self-test once per machine to know which
tier you are getting.

Parallel options
~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Key
     - Meaning
   * - ``parallel.gpu_aware_mpi``
     - pass device memory to MPI (default: host staging; ``PHRIKE_GPU_AWARE_MPI``)
   * - ``parallel.transpose``
     - ``alltoallv`` (default) or ``p2p`` (nonblocking point-to-point; ``PHRIKE_MPI_TRANSPOSE``)
   * - ``parallel.fft_batch``
     - fields transformed and exchanged per pass (default 4).  Lower values cut
       the transient memory of a transform; ``0`` takes all fields at once
   * - ``parallel.triton``
     - ``false`` forces the Torch tier (``PHRIKE_TRITON=0``)
   * - ``parallel.strict_clamp``
     - compressible only: also count floor violations before the final modal
       dissipation, as the reference limiter does (one more inverse transform
       of the state per step)
   * - ``parallel.profile``
     - synchronize the device around timed sections (``--profile``)

Memory per grid point is about 27 real fields for incompressible MHD and 44
for compressible MHD at ``fft_batch: 4``, that is roughly 110 and 180 bytes
in single precision: an 80 GB device holds about :math:`900^3` and
:math:`760^3` points.  Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
for large grids.

Incompressible MHD
------------------

With unit density and the magnetic field in Alfvén units,
:math:`\mathbf{b} = \mathbf{B}/\sqrt{\mu_0\rho}`,

.. math::

   \partial_t \mathbf{u} &= -\nabla\cdot(\mathbf{u}\mathbf{u} - \mathbf{b}\mathbf{b})
       - \nabla\left(p + \tfrac{1}{2} b^2\right) + \nu\nabla^2\mathbf{u} + \mathbf{f},
       \qquad \nabla\cdot\mathbf{u} = 0, \\
   \partial_t \mathbf{b} &= \nabla\times(\mathbf{u}\times\mathbf{b}) + \eta\nabla^2\mathbf{b},
       \qquad \nabla\cdot\mathbf{b} = 0.

A uniform mean field is the :math:`k = 0` mode of :math:`\mathbf{b}`
(``initial_conditions.b0``); it may be zero.  The Alfvén Mach number reported
in ``history.csv`` is :math:`u_\mathrm{rms}/b_\mathrm{rms}`
(``alfven_mach``) and :math:`u_\mathrm{rms}/|\mathbf{b}_0|`
(``alfven_mach_mean_field``).  Nothing in the method assumes a strong field,
so super-Alfvénic and field-free runs need no special treatment.

Method:

* The unknowns are the retained Fourier modes of :math:`\mathbf{u}` and
  :math:`\mathbf{b}`.  The nonlinearity is exactly quadratic, so the 2/3 rule
  removes aliasing exactly; with :math:`\nu = \eta = 0` the truncated system
  conserves energy, cross helicity and magnetic helicity, which the tests
  check to :math:`10^{-11}`.
* Each stage transforms :math:`\mathbf{u}`, :math:`\mathbf{b}` to physical
  space, forms eight products in one pass and transforms them back.  The
  momentum flux :math:`\mathbf{u}\mathbf{u} - \mathbf{b}\mathbf{b}` is
  symmetric and its isotropic part is a pressure, so five fields carry it; the
  induction tensor :math:`\mathbf{u}\mathbf{b} - \mathbf{b}\mathbf{u}` is
  antisymmetric (three fields), which makes :math:`\partial_t\mathbf{b}`
  solenoidal to round-off.  Pressure is eliminated by the Leray projection.
* Viscosity and resistivity are integrated exactly with an integrating factor
  (Lawson Runge-Kutta), so there is no diffusive step limit for any
  :math:`\nu`, :math:`\eta` or magnetic Prandtl number.  ``integration.scheme``
  is ``rk4`` (default), ``rk3`` (Heun) or ``rk2`` (midpoint).
* The step is :math:`\Delta t = \mathrm{cfl} / \max_{\mathbf{x}} \sum_i
  (|u_i| + |b_i|)/\Delta x_i`, evaluated on fields the first stage transforms
  anyway.  Linear stability on the retained modes needs ``cfl <= 1.35`` for
  RK4 and ``0.83`` for RK3; the default is 0.5.  RK2 has no stable interval on
  the imaginary axis and relies on the physical diffusion.

``physics.viscosity`` is :math:`\nu`; :math:`\eta` is ``physics.resistivity``
or ``viscosity / physics.magnetic_prandtl``.  ``physics.spherical_truncation``
keeps :math:`|\mathbf{k}| \le k_\mathrm{max}` instead of the full 2/3 box.
``history.csv`` reports the energies, both dissipation rates, the forcing
power, cross and magnetic helicity, the Taylor-scale Reynolds number and
``kmax_eta`` :math:`= k_\mathrm{max}(\nu^3/\varepsilon_\nu)^{1/4}`, which
should stay above about 1.5 in a resolved run.

Forcing
-------

The ``forcing`` block drives the velocity on a shell of low wavenumbers
``kmin <= |n| <= kmax`` (cycles per box):

.. list-table::
   :header-rows: 1

   * - ``type``
     - Behaviour
   * - ``ou``
     - Ornstein-Uhlenbeck acceleration of rms ``amplitude`` and correlation
       time ``correlation_time``; frozen within a step and advanced exactly
       between steps
   * - ``white``
     - delta-correlated kicks with mean energy injection rate ``power``
   * - ``constant_power``
     - :math:`\hat{\mathbf{f}}_k = P\,\hat{\mathbf{u}}_k / (2 E_\mathrm{band})`:
       injects exactly ``power`` (incompressible solvers only)

The band holds a few hundred modes at most, so every rank generates all of it
from a counter-based stream keyed by ``seed`` and the update index.  A forced
run is therefore the same for any rank count, and it restarts exactly.  In
the incompressible solvers the acceleration is a term of every Runge-Kutta
stage and costs no transform.  In the compressible solvers it is a velocity
kick after each step, at fixed density and internal energy, synthesized on
each slab by separable Fourier sums; it costs eight transforms per step.
``compressive_fraction`` (compressible only) mixes compressive and solenoidal
forcing.

Random initial fields (``turb3d``, ``mhd_turb3d``, ``imhd_turb``, ``ins_turb``)
are also counter-based per Fourier mode: they do not depend on the rank count
or on the resolution, as long as the band is retained.  They are not the
realizations of the serial ``turb3d`` and ``mhd_turb3d`` problems, which draw
white noise on the whole grid.

Output and restart
------------------

.. code-block:: text

   outdir/
     run_info.json  history.csv  timing.json
     snapshots/snap_000012/meta.json, spec.00000.npy, ...   one file per rank
     checkpoints/chk_000003/...                             same format
     spectra/spectra_000012.npz                             shell-summed E(k)
     slices/slice_000012.npz                                the z mid-plane

Snapshots hold the retained spectrum divided by :math:`N`, the solver's own
state: 0.3 of a real field per component.  ``phrike.parallel.io.load_snapshot``
assembles one in a single process and can return the fields in real space.

``--restart DIR`` accepts any snapshot or checkpoint of the same system.  The
rank count may differ, and so may the resolution: modes are matched by integer
wavenumber, so a run can be spun up coarse and refined.  A restart continues
the output numbering and keeps the outputs on ``t0 + n * output_interval``.
``--max-wall-minutes`` (or ``integration.max_wall_minutes``) and
``--max-steps`` stop with a checkpoint instead of an output.

Differences from the serial solvers
-----------------------------------

* The explicit diffusive limit of the compressible MHD stepper uses the
  largest retained wavenumber, not the grid's Nyquist wavenumber, so it is
  2.25 times larger.
* A state that the positivity limiter changed is stored as its retained
  modes; the content above the cutoff that the limiter created is dropped at
  the start of the next step rather than at its end.
* The initial projection of :math:`\mathbf{B}` holds the thermal pressure
  fixed against the dealiased, projected field; the serial ``run`` projects
  before it dealiases.  The two agree when the initial field is band-limited.
