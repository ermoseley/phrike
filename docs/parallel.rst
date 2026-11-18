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

* At fixed points per GPU the total size of each rank's truncated block
  stays nearly constant. With :math:`P` equal slabs, the fraction sent to
  other ranks is :math:`1 - 1/P`. How much crosses the network depends on
  rank placement. The adaptive step also needs a small CFL all-reduce.
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
local transforms, packing and the exchange; the benchmark also gives the
exchange bandwidth per rank (see "Checking the transport").

MPI and GPUs
------------

Install ``mpi4py`` against the cluster's MPI (``pip install --no-binary=mpi4py
mpi4py`` with that MPI's ``mpicc`` on the path).  Launch one rank per GPU; a
rank takes device ``node_rank % visible_devices``, so both ``srun
--gpus-per-task=1`` and several ranks sharing a node's device list work.

Transports
~~~~~~~~~~

Every distributed transform does one all-to-all, so the transport decides the
scaling.  ``parallel.transpose`` (or ``PHRIKE_MPI_TRANSPOSE``, or ``--transpose``
in ``scripts/bench_parallel.py`` and the self-test) defaults to ``auto``. Multiple CUDA ranks use
``nccl``. One rank, CPU runs, and explicit GPU-aware MPI use ``alltoallv``.
The available transports are

``nccl``
   Moves the device buffers with ``torch.distributed``: NCCL for CUDA tensors
   (NVLink and InfiniBand), Gloo for CPU tensors.  There is no host staging and
   no dependence on MPI's CUDA support; MPI is only used to start the group and
   to check that it works.  **Use this on GPU clusters.**  On first use rank 0
   picks a free TCP port on its host and broadcasts the address; set
   ``MASTER_ADDR`` (and ``MASTER_PORT``) to choose it yourself, for example when
   the compute nodes' host names do not resolve from each other.  The CUDA
   device is bound before the group starts, and the group is destroyed at exit.
   If the group cannot start on any rank, all ranks switch to ``alltoallv``
   together, with one warning on rank 0, and the run continues; the run header,
   ``run_info.json`` and the benchmark report the transport in use.  The
   timeout of the group (10 minutes by default) can be changed with
   ``PHRIKE_NCCL_TIMEOUT`` in seconds.  Ranks must be MPI ``COMM_WORLD``
   ranks, one per GPU.

``alltoallv``
   One ``MPI_Alltoallv``.  Device buffers are staged through pinned host memory
   (two extra copies of everything communicated), or passed to MPI directly with
   ``gpu_aware_mpi``.  Use it for CPU runs and when ``nccl`` is unavailable.

``p2p``
   The same exchange as one nonblocking send and receive per peer.  Some MPI
   libraries are faster this way when ranks are many and blocks are small.

Install ``mpi4py`` against the cluster's MPI (``pip install --no-binary=mpi4py
mpi4py`` with that MPI's ``mpicc`` on the path).  Launch one rank per GPU; a
rank takes device ``node_rank % visible_devices``, so both ``srun
--gpus-per-task=1`` and several ranks sharing a node's device list work.

With ``alltoallv`` or ``p2p``, device buffers are staged through pinned host
memory by default, which works with any MPI.  With a CUDA-aware MPI they go to
MPI directly: run ``srun -n 2 python -m scripts.check_gpu_aware_mpi``, and if
it reports that device buffers work set ``parallel.gpu_aware_mpi: true`` or
``PHRIKE_GPU_AWARE_MPI=1``.  Do this before a large run.  Device buffers in MPI
can crash (a segmentation fault in the CUDA IPC path) on a library that is not
configured for them; ``nccl`` avoids the question.

Checking the transport
~~~~~~~~~~~~~~~~~~~~~~

With ``--overlap 0``, ``python -m scripts.bench_parallel`` reports effective
exchange bandwidth per rank from the off-rank bytes sent and the time in
``mpi_exchange``. Host copies are timed separately in ``mpi_stage``. With two
batches in flight the timer measures only exposed wait time, so the bandwidth
field is null and throughput is measured from the complete step.
Host staging itself is fast (47 to 56 GB/s on four H200 GPUs); the transport
is what to look at.  If the bench shows ``alltoallv`` on host buffers under
about 1 GB/s per rank, the MPI transport is misconfigured, not slow by nature.
Measured on four H200 GPUs on S3DF, the CUDA-aware path segfaulted and the
default UCX path ran at about 45 MB/s per rank, while forcing the plain shared
memory transport on one node,

.. code-block:: bash

   OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader srun ...

restored 5.5 GB/s per rank, which is about what host MPI can do there.  Rather
than tune MPI, use ``nccl``, which exchanges device buffers over NVLink and
InfiniBand.  Compare the transports on the machine you will run on:

.. code-block:: bash

   for t in alltoallv nccl; do
     srun -n 4 python -m scripts.bench_parallel --transpose $t --points-per-rank 256
   done

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
     - ``auto`` (default), ``alltoallv``, ``p2p``, or ``nccl``
       (``PHRIKE_MPI_TRANSPOSE``)
   * - ``parallel.fft_batch``
     - fields transformed and exchanged per pass (default 1). Larger batches use more
       temporary memory. ``0`` takes all fields at once
   * - ``parallel.overlap``
     - pipeline two field batches on the NCCL transport (default). Set
       ``false`` or ``PHRIKE_FFT_OVERLAP=0`` to time the sequential exchange.
       The pipeline uses a second pair of exchange buffers.
   * - ``parallel.triton``
     - ``false`` forces the Torch tier (``PHRIKE_TRITON=0``)
   * - ``parallel.strict_clamp``
     - compressible only, default ``true``: also count floor violations before
       the final modal dissipation, as the reference limiter does (one more
       inverse transform of the state per step when the dissipation is on).
       ``false`` skips that count: faster, but a step the reference would limit
       can pass unlimited
   * - ``parallel.profile``
     - synchronize the device around timed sections (``--profile``)

Memory per grid point is about 27 real fields for incompressible MHD and 44
for compressible MHD at ``fft_batch: 4``, that is roughly 110 and 180 bytes
in single precision: an 80 GB device holds about :math:`900^3` and
:math:`760^3` points.  Set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``
for large grids (unverified with a CUDA-aware MPI, whose CUDA IPC may not work
with expandable segments; ``nccl`` is not affected as far as is known).

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
``history.csv`` reports the energies, the dissipation rates, the forcing
power, cross and magnetic helicity, the Taylor-scale Reynolds number and
``kmax_eta`` :math:`= k_\mathrm{max}(\nu^3/\varepsilon_\nu)^{1/4}`, which
should stay above about 1.5 in a resolved run.

Hyperdissipation
~~~~~~~~~~~~~~~~

``physics.hyperviscosity`` :math:`\nu_h` and ``physics.hyperresistivity``
:math:`\eta_h` add :math:`-\nu_h(-\nabla^2)^p\mathbf{u}` and
:math:`-\eta_h(-\nabla^2)^p\mathbf{b}` with :math:`2p =`
``physics.hyper_order`` (an even integer of at least 4; the default 6 is
sixth-order hyperdissipation, :math:`\nu_h k^6`).  A mode then decays at
:math:`\nu k^2 + \nu_h k^{2p}`: the Laplacian terms may be zero or kept
alongside.  Without ``hyperresistivity``, ``magnetic_prandtl`` sets
:math:`\eta_h = \nu_h/\mathrm{Pm}` as it does for :math:`\eta`.  The
integrating factor covers these terms too, so they never limit the step,
whatever the order.  The Triton stage kernel has branches for orders 4 to 10;
other orders run on the Torch tier.

.. code-block:: yaml

   physics:
     viscosity: 0.0
     hyperviscosity: 3.666e-11     # 1 / 2.728e10
     hyperresistivity: 3.666e-11
     hyper_order: 6

``history.csv`` then also holds ``hyperviscous_dissipation``,
``hyperresistive_dissipation``, their sum with the Laplacian rates
(``dissipation``), ``hyper_magnetic_prandtl`` and ``kmax_eta_hyper``
:math:`= k_\mathrm{max}(\nu_h^3/\varepsilon)^{1/(6p-2)}`, the analogue of
``kmax_eta`` for the hyperviscous cutoff.  ``taylor_reynolds`` uses the
viscosity that would dissipate at the measured rate,
:math:`\varepsilon/\langle\omega^2\rangle`, which is :math:`\nu` itself
without hyperviscosity.  ``configs/parallel/imhd_dynamo_hyper.yaml`` runs a
:math:`288^3` fluctuation dynamo.  For a resolution comparison,
``configs/parallel/imhd_dynamo_hyper_2048.yaml`` uses the same physical
coefficients, forcing and seeds at :math:`2048^3`, with one field per FFT batch
and overlapping NCCL exchanges on eight GPUs.  Hyperdissipation is available in the incompressible solvers.

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

``profile`` shapes the band: ``parabolic`` (default, zero at both edges),
``flat`` (the same variance for every mode) or ``power_law``, for which the
shell-integrated spectrum of the forcing follows :math:`k^{s}` with
:math:`s =` ``spectral_index`` (default :math:`-5/3`, a Kolmogorov band; the
variance per mode is then :math:`k^{s-2}`).  ``amplitude`` and ``power`` keep
their meaning for every profile.

The band holds a few hundred modes at most, so every rank generates all of it
from a counter-based stream keyed by ``seed`` and the update index.  A forced
run is therefore the same for any rank count, and it restarts exactly.  In
the incompressible solvers the acceleration is a term of every Runge-Kutta
stage and costs no transform.  In the compressible solvers it is a velocity
kick after each step, at fixed density and internal energy: each rank puts the
band on the retained modes it owns and one distributed inverse transform of
three fields gives the acceleration on the device, so the cost follows the
local data and weak scaling is kept.  With the kick's projection back onto
the retained modes it costs eleven field transforms per step.
``compressive_fraction`` (compressible only) mixes compressive and solenoidal
forcing.

With ``physics.spherical_truncation: true`` (incompressible solvers) the band
is restricted to the modes the solver keeps; in an anisotropic box modes with
:math:`|k| > k_\mathrm{cut}` are dropped and reported, a band left empty is an
error, and ``constant_power`` still injects exactly ``power``.

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
     snapshots/snap_000012/meta.json, spec.00000.npy, ...   restart format, one file per rank
     checkpoints/chk_000003/...                             same format
     spectra/spectra_000012.h5                              shell-summed E(k)
     slices/slice_000012.h5                                 the z mid-plane
     fields/fields_000012.h5, fields_000012.xmf, fields.xmf physical-space fields (io.fields)
     diagnostics/spectral_000012.h5, transfer_000012.h5     analysis module (io.diagnostics)

Every file except the snapshots and checkpoints is analysis output.  The
restart format below is always written at full precision whatever the options.

.. code-block:: yaml

   io:
     format: auto        # auto | hdf5 | npy: fields, slices, spectra, diagnostics
     snapshots: true     # restart-format spectral snapshots at every output
     slices: true
     spectra: true
     fields:
       enabled: false
       every: 1          # every n-th output (integration.output_interval)
       precision: single # double | single | half
       coarsen: null     # null | N_out | [Nx_out, Ny_out, Nz_out] | {factor: 4}
       variables: default
     diagnostics:
       enabled: true
       every: 1
       transfer: {enabled: false, every: 5}

Formats
~~~~~~~

``io.format: auto`` is HDF5 when ``h5py`` imports and ``npy`` otherwise; slices,
spectra and diagnostics are then ``.h5`` or ``.npz`` files written by rank 0.
A field output is a set of ``(Nz, Ny, Nx)`` datasets, one per scalar
component, named ``u_x, u_y, u_z, b_x, ..., rho, p``.  The file attributes
hold ``t``, ``step``, the grid sizes and box lengths (of the output grid and
of the simulation), the system, the precision, the coarsening, the code
commit, and a note on units: magnetic fields carry no :math:`4\pi` or
:math:`\mu_0`, and nodes sit at :math:`i L/N` without the periodic image.

HDF5 field files come in two layouts that a reader cannot tell apart.  With an
``h5py`` built against MPI (and more than one rank) every rank writes its z
slab into the one ``fields_NNNNNN.h5``: the file and datasets are created
collectively and the writes are independent, so ranks with an empty slab do
nothing.  With serial ``h5py`` each rank writes
``fields_NNNNNN.parts/part.RRRRR.h5`` and rank 0 writes ``fields_NNNNNN.h5``
holding one virtual dataset per component that stitches the parts; keep the
``.parts`` directory with the file.  One rank always writes one plain file.
``npy`` writes ``fields_NNNNNN/<name>.RRRRR.npy`` per rank and a ``meta.json``
(written last: a directory that has it is complete).

``phrike.parallel.output.load_fields`` assembles either format in one process::

   from phrike.parallel.output import load_fields
   out = load_fields("outputs/run/fields/fields_000012.h5", ["u", "omega_z"])
   out["t"], out["attrs"], out["fields"]["u_x"]      # (Nz, Ny, Nx) array

``variables`` takes dataset names or whole vectors and limits the memory used
to what is asked for; the result also holds the coordinates ``x, y, z``.

ParaView
~~~~~~~~

Each HDF5 field output gets a ``fields_NNNNNN.xmf`` (XDMF 3, ``3DCoRectMesh``
with ``ORIGIN_DXDYDZ``): every dataset is a scalar attribute, and ``u``, ``b``,
``B``, ``omega``, ``j`` (and ``m``) are also vector attributes joined from
their three components.  ``fields/fields.xmf`` is a temporal collection of all
of them; rank 0 rewrites it at each field output and rebuilds it from the
files that exist after a restart, so open ``fields.xmf`` to see the series.
The XDMF reader needs ``float32`` or ``float64``: with ``precision: half`` the
HDF5 files hold NumPy ``float16`` (readable with h5py), no ``.xmf`` is written
and one warning says so.

Precision
~~~~~~~~~

The cast to ``double``, ``single`` or ``half`` happens on the device, before the
copy to the host, so a lower precision also shrinks the host and I/O traffic.
``half`` keeps about three significant digits and a range of
:math:`6\times10^{-8}` to 65504: values above that are stored as ``inf`` (the
run warns) and values below lose digits, so normalise or use ``single`` for
fields with a wide dynamic range, such as a weak seed magnetic field next to
the velocity.

Coarsening
~~~~~~~~~~

``coarsen`` writes the fields on a coarser grid of ``N_out`` points per axis:
an integer for all axes, ``[Nx_out, Ny_out, Nz_out]``, or ``{factor: f}``
(every size divided by ``f``, which must divide it).  Each size must lie
between 4 and the simulation's.  This is the exact spectral low-pass, not
point sampling and not averaging: the output is the field summed from the
modes with :math:`|n_i| \le (N_{\mathrm{out},i} - 1)//2` that the solver also
retains, evaluated on the coarse grid (the coarse Nyquist modes are zero).
Nothing aliases, and the coarse field equals the coarse samples of the fine
one whenever the fine field has no content above the coarse cutoff.

The ky rows of the retained spectrum of one variable are sent from their
owners to the owners of the coarse decomposition in one all-to-all, and the
coarse inverse transform runs on ``min(ranks, Nz_out, Ny_out)`` ranks (a
sub-communicator, so it uses ``alltoallv`` even with ``transpose: nccl``); the
remaining ranks take part in the exchange and write nothing.  Only the kept
modes move (at most one variable's retained spectrum, much less for a strong
coarsening), and one variable is in flight at a time.  Incompressible fields and their spectral
curls come straight from ``uhat``.  The compressible ``rho``, ``m``, ``E`` and
``B`` come from the spectrum too; the primitives ``u = m / rho`` and ``p`` are
formed in physical space on the fine slabs, forward-transformed with the
truncated transform and then coarsened.  Without ``coarsen`` each variable is
inverse-transformed (or, compressible, taken from the resident state) and
written on the simulation grid.

``variables`` is ``default`` (incompressible ``u`` and ``b``; compressible
``rho``, ``u``, ``p`` and ``B``) or a list: ``u``, ``b``, ``omega`` (curl of
``u``), ``j`` (curl of ``b``) for the incompressible systems, and ``rho``,
``u``, ``p``, ``m``, ``E``, ``B``, ``omega`` and ``j`` for the compressible
ones.  Curls are spectral.

Sizes at :math:`2048^3` single precision: a full-resolution output is 4.3 GB per
field and rank on 8 ranks, 34 GB per scalar component and 0.2 TB for the six
components of ``u`` and ``b``, so write it rarely.  For routine output use
``coarsen: 512`` (0.54 GB per component, 3.2 GB for ``u`` and ``b``) or
``1024`` (4.3 GB per component), and ``half`` halves either.  Writing is
timed under ``out_fields`` in ``timing.json``.  Each rank holds one component
(4.3 GB at full resolution) plus the transform's workspace while it writes,
and a coarsened output needs far less.

Diagnostics
~~~~~~~~~~~

``phrike.parallel.analysis`` computes spectral turbulence diagnostics from the
retained modes, on the device, followed by one small ``allreduce``; the
binning runs over chunks of kz planes of at most ``io.diagnostics.chunk_modes``
modes (default ``2**23``).  Shells are those of ``stepper.spectra()`` (unit
width in :math:`|k| L_\mathrm{max}/2\pi`, centred on integers) and every
spectrum sums over shells to the volume mean quoted.

``spectral_diagnostics`` (no transform for the incompressible solvers):

* ``kinetic``, ``magnetic``: :math:`\langle|u|^2\rangle/2`, :math:`\langle|b|^2\rangle/2`;
  ``kinetic_helicity`` (:math:`\langle u\cdot\omega\rangle`), ``cross_helicity``
  (:math:`\langle u\cdot b\rangle`), ``magnetic_helicity`` (:math:`\langle a\cdot b\rangle`,
  Coulomb gauge) and, with ``current_helicity: true``, :math:`\langle j\cdot b\rangle`;
* ``dissipation_kinetic``, ``dissipation_magnetic``: :math:`D(k^2)|\hat u_k|^2`
  with :math:`D = \nu k^2 + \nu_h k^{2p}` (``eta`` for b), summing to the
  dissipation rates in ``history.csv``;
* anisotropy about an axis :math:`\hat e` (``anisotropy.axis``: ``auto`` is the
  mean magnetic field when :math:`|\langle b\rangle| > 10^{-8} b_\mathrm{rms}`
  and z otherwise; or ``x``, ``y``, ``z``, a 3-vector; ``anisotropy: false``
  disables it): ``kinetic_2d`` and ``magnetic_2d`` in
  :math:`(k_\parallel, k_\perp)` on unit bins, their reductions ``*_par`` and
  ``*_perp``, the component anisotropy :math:`\langle(u\cdot\hat e)^2\rangle/\langle|u_\perp|^2\rangle`
  and the gradient anisotropy :math:`\langle|(\hat e\cdot\nabla)u|^2\rangle/\langle|\nabla_\perp u|^2\rangle`
  (and for b); the axis used is in ``attrs``;
* compressible runs: spectra of :math:`u = m/\rho`, of :math:`\sqrt\rho\,u`
  (``kinetic_density_weighted``), of the density and of B, with the same
  anisotropy outputs (six forward transforms).

``transfer_diagnostics`` (incompressible only) uses the solver's equations,
:math:`\partial_t u = -(u\cdot\nabla)u + (b\cdot\nabla)b - \nabla P` and
:math:`\partial_t b = (b\cdot\nabla)u - (u\cdot\nabla)b`, with :math:`X_K` the
field filtered to shell K::

    T_uu(K,Q) = -<u_K . (u.grad) u_Q>     T_bb(K,Q) = -<b_K . (u.grad) b_Q>
    T_ub(K,Q) =  <u_K . (b.grad) b_Q>     T_bu(K,Q) =  <b_K . (b.grad) u_Q>

``T_ub`` is the Lorentz-force transfer from magnetic shell Q to kinetic shell
K and ``T_bu`` the stretching of :math:`b_Q` into magnetic shell K.

* Always: the shell transfers ``T_uu``, ``T_bb``, ``T_ub``, ``T_bu`` (unfiltered
  field in the Q slot) and the fluxes :math:`\Pi_X(k) = -\sum_{K\le k} T_X(K)`
  on the unit shells, for 9 inverse and 21 forward transforms (3 and 6 without
  b), about two steps' worth.  ``T_uu + T_ub`` and ``T_bb + T_bu`` are the
  nonlinear rates of change of the shell energies.
* ``transfer.matrix: true`` adds ``T_X_matrix[K, Q]`` and ``shell_edges``
  (default logarithmic in base 2: ``[0,1), [1,2), [2,4), ...``; or
  ``transfer.shells``).  Each non-empty Q shell costs 36 forward and 24 inverse
  transforms with about 6 physical fields alive, or 36 forward and 6 inverse
  with ``transfer.cache_fields: true`` (about 10 fields).  The matrices are
  antisymmetric (``T_ub(K,Q) = -T_bu(Q,K)``) and each row sums to the shell
  transfer.  At 2048³ use them sparingly (every several outputs).

The driver calls ``spectral_diagnostics(stepper, config)`` every
``io.diagnostics.every`` outputs and, when ``transfer.enabled``,
``transfer_diagnostics(stepper, config)`` every ``transfer.every`` outputs, with
the ``io.diagnostics`` block as ``config``.  Both are collective and return
the same small arrays on every rank; rank 0 writes them to
``diagnostics/spectral_NNNNNN`` and ``diagnostics/transfer_NNNNNN`` (time in
``out_diag``).  A missing analysis module disables diagnostics with one
warning, and ``NotImplementedError`` from the transfer diagnostics (for the
compressible systems) disables only the transfer.

Restart snapshots
~~~~~~~~~~~~~~~~~

Snapshots hold the retained spectrum divided by :math:`N`, the solver's own
state: 0.3 of a real field per component.  ``phrike.parallel.io.load_snapshot``
assembles one in a single process and can return the fields in real space.
A compressible MHD snapshot written right after the accepted-state limiter
clamped also holds that state's out-of-band residual (``resid.*.npy``, physical
space, five components per rank); see below.

``--restart DIR`` accepts any snapshot or checkpoint of the same system.  The
rank count may differ, and so may the resolution: modes are matched by integer
wavenumber, so a run can be spun up coarse and refined.  The out-of-band
residual of a clamped state is stored per rank in physical space, so it is read
back only when the grid and the rank count are unchanged, and then the restart
reproduces the uninterrupted run exactly.  After a restart on another rank
count or resolution it is dropped with a ``RuntimeWarning`` and the first step
differs from the uninterrupted run by that residual.  A restart continues
the output numbering (fields, slices, spectra and diagnostics included) and
keeps the outputs on ``t0 + n * output_interval``; ``fields.xmf`` is rebuilt
from the field outputs up to the restart point.
``--max-wall-minutes`` (or ``integration.max_wall_minutes``) and
``--max-steps`` stop with a checkpoint instead of an output.

Differences from the serial solvers
-----------------------------------

* The explicit diffusive limit of the compressible MHD stepper,
  ``cfl_diff / (max(nu, eta) k_max^2)``, takes :math:`k_\mathrm{max}` from the
  largest retained mode, :math:`(N - 1) // 3`, not from the grid's Nyquist mode
  :math:`N/2` as the serial solver does.  The retained modes are the only ones
  the Laplacian acts on, so this is the limit that applies.  The step it
  allows is larger by :math:`(N/2)^2 / ((N - 1)//3)^2`: 2.25 for large grids,
  up to 3 for small ones.  A diffusion-limited run therefore takes fewer,
  larger steps than the serial solver and its results agree with it to the
  time-stepping error, not to round-off; set ``cfl_diff`` 2.25 times smaller
  to reproduce the serial step.
* When the MHD accepted-state limiter clamps, the clamped state has content
  above the cutoff.  The reference keeps it through the next step: the
  dissipation half-step damps the whole spectrum, every stage is evaluated on
  the full physical state, and the accepted-state projection removes it at the
  end.  The stepper keeps that content as a residual (physical space) next to
  the retained spectrum and does the same, using a full-spectrum distributed
  transform that is built only while a residual exists (the retained-mode
  decomposition allows ``min(Nz, Ky)`` ranks, the full one ``min(Nz, Ny)``).
  The residual is saved in snapshots (see Output and restart); a forcing kick
  carries it along unchanged.
* With the Triton tier, ``warm_up`` also runs every kernel variant on this
  rank's real local shapes (Triton specializes on sizes such as a row count of
  1) and then makes later kernel failures fatal: the driver aborts all ranks
  instead of one rank falling back to Torch ops in the middle of a collective
  step.
* The initial projection of :math:`\mathbf{B}` holds the thermal pressure
  fixed against the dealiased, projected field; the serial ``run`` projects
  before it dealiases.  The two agree when the initial field is band-limited.
