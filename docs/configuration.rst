Configuration reference
=======================

YAML files are read by ``BaseProblem`` and then by the grid and solver
constructors. Problem-specific ``initial_conditions``
keys live in the problem classes and in ``configs/``; they are not repeated
here. Integrator names are parsed as described below.

A top-level ``problem`` value is used as the default output-directory name
and as the video filename. It should be the registered problem name. A
mapping is not normalized; configs that store a mapping also set
``io.outdir``, which is the path that is used.

Required blocks are ``integration`` (``t0``, ``t_end``, ``cfl``), ``grid``,
and either ``equations.gamma`` or ``physics.gamma``.

A key that is not on the parser's allow-list for a recognized section
produces a ``UserWarning`` (``Unknown configuration key '...' is ignored``).
The value stays in the mapping. ``physics.mhd.enabled`` warns and is not
read.

Integrators
-----------

Every solver uses one resolver. Names are
stripped and lowercased. The accepted names are:

* ``rk2`` and ``rk4`` — fixed-step methods. The default ``integration.scheme``
  is ``rk4``.
* ``rk23`` — Bogacki–Shampine, 3rd-order step, embedded 2nd-order estimate.
* ``rk45`` — Dormand–Prince, 5th-order step, embedded 4th-order estimate.
* ``rk78`` — Fehlberg 7(8), 13 stages. The accepted step uses the 8th-order
  weights and the embedded estimate is 7th order. ``RK78`` is the same name.
  ``rkf78`` and ``RKF78`` are not accepted. The controller exponent is
  ``order_low + 1``.

Any other string raises ``ValueError``. An embedded name with adaptive
stepping left off also raises ``ValueError``. ``rk23``, ``rk45``, and
``rk78`` run only when adaptive stepping is enabled (see
``configs/sod_mixed_bc_example.yaml`` for a 1D adaptive RK45 example).

If adaptive stepping is on and ``adaptive.scheme`` is omitted, an embedded
``integration.scheme`` is used; otherwise the embedded method defaults to
``rk45``. If both keys name embedded methods, they must name the same one.
If ``adaptive.scheme`` is ``rk2`` or ``rk4``, adaptive stepping stays off
and the solver warns and uses ``fallback_scheme``.

``fallback_scheme`` must be ``rk2`` or ``rk4`` when it is set. The default
is ``rk4``. It is not used to retry a rejected step. When an embedded
method is selected, a ``RuntimeWarning`` says that ``fallback_scheme`` is
ignored.

The adaptive block may be ``integration.adaptive`` or a top-level
``adaptive`` mapping. The nested key is preferred when both are present
and the nested value is non-empty.

.. list-table:: Adaptive keys
   :header-rows: 1
   :widths: 28 18 54

   * - Key
     - Default
     - Meaning
   * - ``enabled``
     - ``false``
     - Turn the embedded stepper on.
   * - ``scheme``
     - ``rk45``
     - ``rk23``, ``rk45``, or ``rk78``.
   * - ``rtol``
     - ``1e-6``
     - Relative tolerance.
   * - ``atol``
     - ``1e-8``
     - Absolute tolerance.
   * - ``safety_factor``
     - ``0.9``
     - Step-size controller factor.
   * - ``min_dt_factor``
     - ``0.1``
     - Smallest accepted step-size ratio.
   * - ``max_dt_factor``
     - ``5.0``
     - Largest accepted step-size ratio.
   * - ``max_rejections``
     - ``10``
     - Rejection limit stored for the controller.
   * - ``dt_min``
     - set in ``run``
     - Absolute step floor. If omitted, ``run`` sets
       ``1e-12 * abs(t_end - t0)`` when that span is positive, and
       ``1e-14`` when it is not. The integrator reads this key. The
       unknown-key check does not list it yet, so a YAML value warns.
   * - ``fallback_scheme``
     - ``rk4``
     - Fixed scheme used only when adaptive stepping was requested
       with ``rk2`` or ``rk4``. Must itself be ``rk2`` or ``rk4``.

Ambipolar diffusion rejects an enabled adaptive block.

Integration
-----------

``integration.t0``, ``t_end``, and ``cfl`` are required. Optional keys:

* ``scheme`` — default ``rk4``. See above.
* ``output_interval`` — default ``0.1``.
* ``checkpoint_interval`` — default ``0``. A non-positive value disables
  checkpoints.
* ``spectral_dissipation`` — Fourier grids only. Keys: ``enabled``
  (default false), ``order`` or ``p`` (default 8, even, at least 2),
  exactly one of ``rate`` and ``e_folding_time_at_cutoff`` when enabled,
  and ``onset_fraction`` (default 0, in ``[0, 1)``). ``rate`` is the
  cutoff-mode damping rate; the e-folding time is ``1/rate``. See
  :doc:`equations`. ``alpha`` and ``interval`` raise ``ValueError``.
* ``modal_filter`` — Legendre and Chebyshev grids. Keys: ``enabled``
  (default false), ``order`` or ``p`` (default 8), ``alpha`` (default
  36, dimensionless). This is not the Fourier rate.
* ``sbp_dissipation`` — Fourier-:math:`x` / SBP-:math:`y` only. Keys:
  ``enabled`` (default false) and ``strength`` (default 0, finite and
  nonnegative). Enabling it on another grid raises ``ValueError``.

``integration.spectral_filter`` raises ``ValueError``. Use
``spectral_dissipation`` or ``modal_filter``.

A polynomial grid ignores ``spectral_dissipation``. A Fourier grid ignores
``modal_filter``. A mixed SBP grid ignores ``spectral_dissipation``.

Grid
----

``grid.precision`` is ``double`` (default) or ``single``. Metal/MPS always
runs in single precision. ``grid.fft_workers``, when set, is the FFT worker
count. Otherwise that count comes from ``runtime.num_threads`` or from the
CPU count. ``grid.dealias`` defaults to true in the problem constructors
that read it.

One dimension reads ``N``, ``Lx``, ``basis`` (``fourier``, ``legendre``, or
``chebyshev``; default ``fourier``), and ``bc``. ``bc`` may be a global
name, a per-variable mapping, or a ``left``/``right`` mapping. Names and
aliases are listed in :doc:`validation`. ``sod`` also reads ``grid.x0``
(default half the domain).

``grid.bc_values`` is read by ``sod`` and ``brio_wu1d``. Each entry is a
``(left, right)`` pair of finite numbers. Accepted keys are ``density``
(``rho``), ``momentum`` (``momentum_x``, ``mom``, ``velocity``, ``u``),
``pressure`` (``p``), and, for MHD, ``momentum_y``, ``momentum_z``, ``bx``,
``by``, and ``bz``. Momentum values are momentum density. The two ``bx``
values must be equal.

Two dimensions read ``Nx``, ``Ny``, ``Lx``, ``Ly``, ``basis_x``,
``basis_y``, and ``bc``. ``basis_y: sbp`` with ``basis_x: fourier`` selects
the open transverse grid. ``grid.bc_config`` is read from YAML only by
``gaussian_wave2d_y``. ``adsi2d`` hard-codes periodic ``left``/``right`` and
open ``bottom``/``top`` and does not read ``bc_config``. On that mixed grid
the side values must stay periodic in :math:`x` and open in :math:`y`;
``outflow`` and ``neumann`` are aliases of ``open``.

Three dimensions read ``Nx``, ``Ny``, ``Nz``, ``Lx``, ``Ly``, ``Lz``, and
``dealias``.

Physics
-------

``equations.gamma`` and ``physics.gamma`` are the same number. One of them
is required.

``physics.mhd`` is copied onto the grid by ``alfven1d``, ``alfven3d``,
``brio_wu1d``, ``orszag_tang2d``, ``mhd_turb3d``, and ``adsi2d``. The flag
``physics.mhd.enabled`` is not read; the problem class selects the MHD
solver. Keys, all optional:

* ``resistivity`` — default 0. Magnetic diffusivity. Units and the
  Laplacian it multiplies are in :doc:`equations`.
* ``viscosity`` — default 0. Diffusivity applied to momentum density.
* ``cfl_diff`` — default 1. Parabolic timestep factor.
* ``divergence_clean`` — default ``projection``. ``none`` skips the
  Helmholtz projection. Mixed open-:math:`y` MHD requires ``projection``.
* ``project_each_substage`` — default true.
* ``rho_min`` — default ``1e-10``. ``p_min`` — default ``1e-12``.
* ``ambipolar`` — two-dimensional MHD only. ``enabled`` (default false),
  ``eta`` (alias ``alpha``, default 1), ``method`` (must be
  ``midpoint_resolvent``), ``preconditioner`` or
  ``midpoint_preconditioner`` (``fourier_tensor``, ``none``, or
  ``identity``; default ``fourier_tensor``), ``midpoint_tol`` (default
  ``2e-5`` in float32 and ``1e-10`` otherwise), and ``midpoint_maxiter``
  (default 200).

Artificial viscosity, gravity, and smoothing
--------------------------------------------

Top-level ``artificial_viscosity`` is optional. Keys: ``enabled`` (default
false), ``mode`` (``sensor`` or ``constant``; default ``sensor``),
``nu_constant`` (default 0), ``nu_max`` (default ``1e-3``), ``s_ref``
(default 1), ``s_min`` (default 0.1), ``p`` (default 2), ``epsilon``
(default ``1e-12``), ``sensor_variable`` (default ``density``),
``variable_weights``, and ``diagnostic_output`` (default false). The 1D and 2D Euler solvers apply it and include it in the timestep.
The 1D MHD solver applies it. Enabling it for 2D or 3D MHD, or for the
3D Euler solver, raises ``NotImplementedError``.

Top-level ``gravity`` is optional and affects the Euler solvers only.
Keys: ``enabled`` (default false), ``gx``, ``gy``, ``gz`` (default 0), and
``type`` (default ``linear``, stored and not consulted). One dimension
adds only ``gx``.

Top-level ``initial_conditions_smoothing`` is a Gaussian smoother, separate
from any ``initial_conditions.smoothing`` block a problem defines. Keys:
``enabled`` (default false), ``kernel_size`` (default 0), ``mode``
(default ``wrap``), ``variables`` (default density, velocity, and
pressure), and ``diagnostic_output`` (default false).

I/O, monitoring, video, runtime, tracers
----------------------------------------

``io.outdir`` is the output directory. If it is omitted, the directory is
``outputs/<problem>``. The command-line ``--outdir`` flag replaces it.

``monitoring`` defaults to on. Keys: ``enabled`` (default true),
``step_interval`` (default 10), ``output_file`` (a file name under the
output directory; stdout if omitted), ``include_conservation``,
``include_timestep``, ``include_time``, ``include_velocity_stats``, and
``include_density_stats`` (all default true). The MHD block of the monitor
prints ``plasma_beta`` as defined in :doc:`equations`.

``video.enabled`` is read by the command-line interface only. The CLI
renders video when ``--video`` is passed or when ``video.enabled`` is
true; ``--video-fps``, ``--video-quality`` and ``--video-codec`` override
the YAML values only for a run that renders video. ``phrike.run_simulation``
and ``BaseProblem.run`` default ``generate_video`` to false. A video run
clears ``<outdir>/frames`` first. Frame rendering reads
``video.frame_dpi`` in the problem visualizers that draw frames; the
fallback used there is 120 or 150, depending on the problem.

The encoder uses ``fps`` (default 30), ``codec`` (default ``libx264``),
``quality`` (``low``, ``medium``, or ``high``; default ``high``), ``crf``
(default from ``quality``: 28, 23, 18), ``preset``, ``tune``,
``bitrate``, ``maxrate``, ``bufsize``, ``pix_fmt`` (default
``yuv420p``), ``scale`` (default 1), and ``width`` and ``height``
(default 0, meaning unused). ``profile`` (default ``high``) and ``level``
(default ``4.1``) are passed to every encoder except ``mpeg4`` and
``libvpx-vp9``. If the requested
codec fails, the encoder warns and tries ``libx264``,
``h264_videotoolbox``, ``h264_nvenc``, ``mpeg4`` and ``libvpx-vp9`` in
that order; if all fail, the frames are kept and the equivalent ffmpeg
command is printed.

The 3D turbulence frame also reads ``projection_type`` (default
``column_density``; any other value selects a slice), ``column_axis``
(default ``z``), ``column_thickness`` (default 0.2), ``slice_axis``
(default ``z``), ``slice_position`` (default 0.5), ``colorbar_scale``
(``linear`` or ``log``; default ``linear``), ``colorbar_min``,
``colorbar_max``, and ``colorbar_fixed`` (default false; a fixed range
is used only when both limits are set).

``runtime.num_threads`` is an integer or ``auto``. The command-line
interface applies it to OpenMP, BLAS, and Torch CPU thread counts. The
grid FFT worker count still prefers ``grid.fft_workers``.

``tracers`` is used on Fourier grids only. A non-Fourier grid ignores the
block. Keys: ``enabled`` (default false), ``num`` (default 100, must be
positive), ``layout`` (``random`` or ``uniform``), ``seed``, ``mass`` (a
number, or ``auto`` for mean density times volume divided by particle
count), ``plot_density`` (3D final plot only), and ``density_grid`` (three
positive integers; otherwise the plot chooses its own shape).
