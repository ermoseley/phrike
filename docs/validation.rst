Validation and limitations
==========================

Run the maintained automated checks with:

.. code-block:: bash

   python -m pytest -q

The current suite exercises adaptive Runge-Kutta coefficients and controller
order, backend selection, unavailable-device errors, and Torch CPU behavior.
It does not constitute a complete validation suite for every equation set,
configuration, or accelerator.

On Apple Silicon with PyTorch installed, the default backend is Metal/MPS
in single precision. Tests pass an explicit backend into the grid. See
:doc:`installation`. The timestep and unit conventions used below are
defined in :doc:`equations`.

PHRIKE is therefore best treated as experimental research software. Numerical
results should be accompanied by their configuration, commit, backend,
precision, and problem-specific convergence or conservation evidence. The
non-periodic polynomial paths remain exploratory, and 3D Legendre support is not
implemented.

1D bases and boundary conditions
--------------------------------

``grid.basis`` accepts ``fourier``, ``legendre``, and ``chebyshev`` in 1D.
Fourier is periodic on a uniform grid. Both polynomial bases use Lobatto nodes
including the endpoints, physical quadrature weights, and the minimum node
spacing in the advective and explicit-viscosity timestep limits. Chebyshev
transforms, derivatives, and filters currently use dense matrices.

``grid.bc`` accepts a global name, a per-variable mapping, or independent
``left`` and ``right`` specifications. On a polynomial grid it defaults to
``wall``. Supported names have the following meaning:

* ``fixed`` (alias ``dirichlet``) holds the initial endpoint density, momenta,
  thermal pressure, and magnetic field, imposed strongly at the endpoint node.
  Set paired overrides with ``grid.bc_values``. Unlike older versions,
  ``dirichlet`` means fixed data; use ``wall`` to request an impermeable
  endpoint.
* ``outflow`` (aliases ``open``, ``neumann``) is a far-field closure imposed
  weakly: the endpoint right-hand side receives the SBP/Rusanov penalty
  ``-(F*(u_b, g) - F(u_b)) . n / w_b`` against the initial endpoint state
  ``g``, the same closure as the open faces of the 2D Fourier/SBP grid.
  Outgoing waves leave; incoming characteristics relax to the initial state.
* ``wall`` (aliases ``reflective``, ``reflecting``) uses the same penalty
  against the mirror state with reversed normal momentum, so the mass flux
  through the wall vanishes exactly. This is a kinematic wall; no additional
  magnetic conducting-wall model is imposed.
* ``periodic`` requires Fourier. Non-periodic BCs on Fourier are rejected.

The penalty uses the endpoint value of the flux polynomial that the volume
term differentiates (with 3/2 overintegration this differs from the flux of
the endpoint state), and diffusion terms (artificial viscosity, explicit
resistivity and viscosity) use a zero-flux endpoint closure. Together these
keep the summation-by-parts balance: with ``wall`` endpoints the total mass
changes only at round-off. Earlier versions copied neighbouring nodal values
instead; that closure is unstable for the global polynomial derivative.

On a smooth right-going acoustic pulse (amplitude 1e-3, N=64, t=1) the
remaining acoustic amplitude after the pulse has left through ``outflow`` is
1.5e-6 (Chebyshev) and 1.2e-7 (Legendre); the copying closure grew the pulse
seventy-fold on Chebyshev and produced NaN on Legendre. A ``wall`` returns the
reflected pulse with the same amplitude. Near a discontinuity the global
basis spreads Gibbs error to every node: on the shipped ``sod.yaml`` and
``brio_wu1d.yaml`` settings at N=64 and N=48, the nodes next to a ``fixed``
endpoint already deviate from the initial density by up to 3% (Sod) and 10%
(Brio–Wu, Legendre) at t=0.02, before any wave arrives. ``outflow`` endpoints
now stay within 0.5% of their initial density in those cases, and ``wall``
endpoints stay within a factor of two of the error at the nodes next to a
``fixed`` endpoint.

For example:

.. code-block:: yaml

   grid:
     N: 96
     Lx: 1.0
     basis: chebyshev
     bc:
       left: fixed
       right:
         default: outflow
         momentum: wall
     bc_values:
       density: [1.0, 0.125]
       momentum: [0.0, 0.0]
       pressure: [1.0, 0.1]

Variable keys are ``density`` (``rho``), ``momentum`` (``momentum_x``, ``mom``,
``velocity``, ``u``), ``pressure`` (``p``), and for MHD ``momentum_y``,
``momentum_z``, ``bx``, ``by``, ``bz``. Momentum aliases specify momentum
density, not velocity. Unspecified mapping entries default to ``wall``;
use ``default`` for another fallback. Fixed ``bx`` overrides must agree on
both sides and match the longitudinal interior field. Energy is rebuilt
from thermal pressure and all kinetic/magnetic components.

Boundary application returns a copy and is performed for RHS evaluations
and after accepted filtering, so rejected RK steps leave the input intact.
Fixed values are captured anew at each ``solver.run``; explicit overrides
are preferable when restarting if the original boundary data must persist.
Lobatto MHD projection retains the quadrature-weighted mean longitudinal field.

These polynomial-boundary checks cover NumPy and Torch CPU; they do not
qualify polynomial runs on CUDA or Metal/MPS hardware.

1D shock runs
--------------

The examples run sharp Sod to ``t=0.2`` and sharp Brio-Wu to ``t=0.1``:

.. code-block:: bash

   python -m phrike sod --config configs/sod_chebyshev_1d.yaml --backend cpu
   python -m phrike brio_wu1d --config configs/brio_wu1d.yaml --backend cpu
   # Repeat either with --basis legendre.

Both use N=192, NumPy CPU double precision, fixed reservoir endpoints,
CFL=0.2 (see :doc:`equations`), RK4, and no initial smoothing. Sod uses
constant viscosity 0.001;
Brio-Wu uses 0.0015. Both disable the accepted-step modal filter and enable
``grid.dealias``. For polynomial grids this now evaluates nonlinear fluxes
on a 3/2-sized grid and projects their derivatives onto the original basis.
This suppresses quadratic aliasing; rational MHD/Euler fluxes are not claimed
to be exactly integrated by a finite padding factor.

Sharp conservative polynomials can give negative density or thermal pressure
at extra nodes even when the original nodal data are positive. At those
extra nodes only, flux evaluation uses the piecewise-linear conservative
state between the adjacent original nodes. This is a convex interpolation
of admissible nodal states. The evolved state, its energy, and the configured
sharp initial data are not clipped or smoothed by this interpolation.

Constant-viscosity terms are batched and the Legendre second-derivative
matrix is cached. For constant viscosity with globally fixed endpoints, the
diffusion timestep uses a row-norm bound on the interior Laplacian matrix;
other boundary/viscosity configurations retain the spacing-based limit.

The Brio-Wu state uses gamma=2, Bx=0.75, By=+1/-1, rho=1/0.125, p=1/0.1,
and zero velocities/Bz, following the
`Athena shock-tube input <https://github.com/PrincetonUniversity/athena/blob/main/inputs/mhd/athinput.bw>`_.
Two sampling methods are used on this page.

* The N=192 Chebyshev mean-absolute table, the quoted Sod velocity extrema
  (minimum about -0.0142 and maximum 0.9324), and the matching Legendre
  mean-absolute errors use 2001 uniformly spaced samples after linear
  interpolation from the Lobatto nodes. Those figures include
  interpolation error.
* The Dedalus comparison, the higher-resolution successive-difference
  table, the N=512 errors against the inviscid and finite-volume
  references, and the halved-viscosity L1 errors use evaluation of the
  conservative polynomial on 5001 common uniform points, with primitive
  fields recovered after that evaluation.

Sod was compared with its exact Euler Riemann solution. Brio-Wu was
compared with an independent 2048-cell primitive-minmod MUSCL/HLL
finite-volume solution using SSPRK2 and outflow BCs. That reference
stayed positive without floors; its 1024-to-2048 density difference was
0.00150. It is an approximate MHD reference. The finite-volume driver is
not in this repository.

The earlier N=96 examples combined insufficient resolution and nonlinear
aliasing with an overly strong per-step filter in Sod. Changing resolution,
viscosity and flux evaluation together gives the following measured
Chebyshev comparison; it is not an ablation of their separate contributions:

.. list-table:: Mean absolute primitive-profile errors, before / after
   :header-rows: 1

   * - Problem
     - Density
     - Normal velocity
     - Pressure
     - Transverse field By
   * - Sod
     - 0.01817 / 0.00765
     - 0.04753 / 0.01100
     - 0.01911 / 0.00587
     - --
   * - Brio-Wu
     - 0.01672 / 0.01353
     - 0.02842 / 0.02354
     - 0.01763 / 0.01548
     - 0.02116 / 0.02001

The improved Chebyshev runs needed no evolved-stage pressure-floor
repairs, and accepted nodal densities stayed positive. Minimum raw
pressure is the smallest thermal pressure recovered on an evolved
Runge-Kutta stage before the positivity clamp (``p_min`` of 1e-12 on
these 1D updates). It is not the minimum pressure on the final Lobatto
nodes, and it is not evaluated on the extra flux nodes. The recorded
stage minima were 0.08118 (Sod) and 0.04312 (Brio-Wu). The script that
recorded them is not in this repository. The extra-node interpolation
safeguard is separate from this stage-positivity check. Brio-Wu max
absolute div(B) was 3.64e-12.
The relative quadrature-integrated mass and total-energy changes were
below 4e-7 in these two Chebyshev runs; this is a case-specific check.
Sod's final velocity minimum improved from about -0.36 to -0.0142;
its maximum was 0.9324 versus the exact 0.9275. Small plateau ripples and
viscous broadening remain. These runs establish neither strict monotonicity
nor resolution convergence.

Repeating the same setup on Legendre gave mean density/velocity/pressure
errors 0.00766/0.01114/0.00589 for Sod and 0.01353/0.02357/0.01553 for
Brio-Wu, also without pressure-floor activations. The minimum raw
pressures, in the same stage sense as above, were 0.08119 and 0.04318.
Brio-Wu max absolute div(B) was 1.21e-12.

A viscous Dedalus 3.0.3 Sod comparison was run outside this repository.
Its driver is not included. That run used N=256, viscosity 0.0003,
initial tanh width 0.01, primitive-variable diffusion, wall/Neumann tau
constraints, and RK443, through ``t=0.2``. A timestep of 1e-5 was
qualified against 1e-6 through ``t=0.005``; the maximum primitive
differences were below 3.4e-8. The full run stayed finite and positive.
A filtered variant of that external script failed on its first step with
a singular system and provides no comparison curve.

Evaluating the saved spectral profiles on 5001 common uniform samples gave
density/velocity/pressure L1 errors 0.01097/0.02973/0.01115 for that Dedalus
configuration and 0.00755/0.01090/0.00584 for the improved N=192 PHRIKE case.
This comparison uses the actual Dedalus solution, with coordinates and fields
at the same scale; no analytical surrogate or plot smoothing was used.
Dedalus retained visible plateau ringing. These configurations use different
diffusion equations, boundary constraints, viscosities, and initial smoothing;
the errors against sharp inviscid Sod do not establish matched-solver parity.

An additional production PHRIKE run used the same N=256, viscosity 0.0003,
tanh width 0.01, and wall endpoints. It stayed positive, with L1 errors
0.00769/0.01463/0.00692 on the same polynomial sampling. Both low-viscosity
profiles ring: PHRIKE has a stronger downstream velocity wake, while its
shock position is closer to inviscid Sod. The velocity-midpoint shock positions
were 0.83618 (Dedalus), 0.85087 (PHRIKE), and 0.85043 (exact).
Matching these input parameters does not equate primitive and conservative
diffusion or the tau and endpoint-extrapolation boundary treatments.

Higher-resolution checks
~~~~~~~~~~~~~~~~~~~~~~~~

Sharp Chebyshev runs at N=192, 256, 384, and 512 held viscosity, initial
data, and fixed endpoints constant. These runs use the 5001-point
polynomial sampling described above, rather than the 2001-point linear
interpolation of the mean-absolute table.

.. list-table:: L1 differences between successive resolutions
   :header-rows: 1

   * - Resolution pair
     - Sod density
     - Sod velocity
     - Brio-Wu density
     - Brio-Wu normal velocity
   * - 192 to 256
     - 3.05e-4
     - 1.56e-3
     - 4.01e-4
     - 2.40e-3
   * - 256 to 384
     - 1.52e-4
     - 5.96e-4
     - 2.07e-4
     - 1.05e-3
   * - 384 to 512
     - 4.39e-5
     - 1.10e-4
     - 6.41e-5
     - 2.54e-4

The profiles approach a viscous solution: N=512 density errors against the
inviscid/FV references remain 0.00762 (Sod) and 0.01354 (Brio-Wu). Increasing
resolution alone does not remove this regularization bias. Sod's sampled
velocity undershoot decreased from -0.0197 at N=192 to -0.000955 at N=384.
These finite refinement checks do not establish an asymptotic spectral order.

The production explicit Sod solver was run at N=192 and 256. A separate
split-diffusion study, whose driver is not included in this repository,
used exact constant diffusion of the interior Laplacian with an affine
fixed-boundary lift, Strang-split around the production RK4 flux solver.
That arrangement avoids the clustered-grid explicit diffusion restriction
at higher resolution. It leaves MHD Bx undiffused. Its N=192 profiles
agreed with explicit evolution to within 1.1e-6 in every primitive L1
component. A smooth control with no fallback gave a 3.94-fold error
reduction on timestep halving. At N=512, doubling the study timestep
changed primitive profiles by at most 1.80e-7 (Sod) and 6.35e-8
(Brio-Wu), below the measured resolution differences.

All split runs recorded no nonpositive density/pressure stages or invalid
diffusion half-steps. The extra-node interpolation safeguard remained active;
N=512 used 100 (Sod) and 1336 (Brio-Wu) replacement-node evaluations over the
whole run. Maximum absolute div(B) was 4.69e-11 in the refined Brio-Wu runs.
That split integrator is not part of the production solver and cannot be
rerun from this tree. These runs do not qualify a new production time
scheme or establish wall-clock scaling.

A separate N=512 sensitivity run halved viscosity to 0.0005 for Sod and
0.00075 for Brio-Wu. Density/normal-velocity/pressure L1 errors fell to
0.00481/0.00589/0.00339 and 0.00919/0.01519/0.00972, respectively; Brio-Wu
By error fell to 0.01266. These split runs also needed no stage repairs or
invalid diffusion halves. They sharpen the profiles but do not establish
resolution convergence at the lower viscosity. The Brio-Wu compound-wave
structure still differs visibly from the approximate finite-volume reference.

Focused tests cover polynomial transforms, quadrature, derivatives,
overintegrated mode products, positive extra-node shock states, original-state
immutability, constant longitudinal field, fixed/wall/extrapolating BCs,
accepted/rejected RK paths, and NumPy/Torch CPU precision behavior.
Accelerator parity has not been established.

2D mixed boundaries and ambipolar diffusion
-------------------------------------------

``Grid2D(basis_x="fourier", basis_y="sbp")`` uses a periodic x direction and
uniform, endpoint-inclusive y nodes. The y derivative is diagonal-norm SBP4(2):
fourth order in the interior and second order at the boundary. Its quadrature
weights are used for mixed-grid conservation diagnostics and magnetic
projection. The hard 2/3 projection acts only in x. There is no periodic y
transform or smooth y filter on this path.

Hydro and MHD momentum advection use a conservative split form in y. It
preserves the SBP-weighted momentum integral and gives the advective
kinetic-energy boundary balance before the x projection, avoiding a spurious
energy source from the discrete product-rule error. This identity excludes
pressure, magnetic stresses, boundary penalties and dissipation; it is not a
stability or positivity guarantee for the complete projected system.

For nonlinear mixed-grid runs, ``integration.sbp_dissipation`` can enable
conservative high-order numerical damping with ``strength: 0.01``. It applies
the negative SBP-weighted transpose product of third forward differences to
all conserved variables inside the RK right-hand side. The interior
checkerboard damping rate is ``strength * max_wave_speed / dy``; the term
vanishes as fifth order in the interior and second order at the endpoints
under refinement. It preserves SBP-weighted totals and constants before
boundary fluxes and magnetic projection. This is numerical regularization,
separate from physical ambipolar diffusion; its strength and grid dependence
must be checked for nonlinear results. The timestep includes its damping
bound. The ADSI example enables it because centered y derivatives otherwise
admit nearly alternating computational modes that nonlinear terms can feed.

On this grid, ``left`` and ``right`` are ``periodic`` and ``bottom`` and
``top`` are ``open``. ``outflow`` and ``neumann`` are aliases for this
open flux closure, not derivative constraints. Only ``gaussian_wave2d_y``
reads ``grid.bc_config`` from YAML. ``adsi2d`` hard-codes periodic x and
open y and does not read ``bc_config``. Endpoints evolve through an
SBP/Rusanov
boundary flux against the initial ambient exterior state. Unlike endpoint
copying, this lets outgoing acoustic energy leave without a later recurrence.
The exterior is recaptured by ``solver.run``; restart boundary values therefore
come from the restart state. Walls and other orientations are not implemented
on this new path.

MHD requires ``divergence_clean: projection``. The weighted projection allows
normal magnetic flux and holds thermal pressure fixed while changing magnetic
and total energy. ``project_each_substage: false`` projects accepted states;
intermediate RK stages need not be solenoidal. This is not constrained transport
or an exactly non-reflecting MHD boundary. Outgoing oblique MHD packets retain
some reflected energy, so boundary-distance checks remain necessary.

``physics.mhd.ambipolar`` enables constant-coupling ambipolar diffusion with
``eta`` multiplying ``(B**2 I - B B.T) curl(B)``. Only two-dimensional MHD is
supported. Set ``method: midpoint_resolvent``; optional solver controls are
``midpoint_tol``, ``midpoint_maxiter``, and ``preconditioner: fourier_tensor``
(or ``none``). Adaptive RK integration is rejected when this split operator
is enabled. Explicit resistivity and viscosity retain separate diffusion CFL
limits, including the SBP derivative bound on mixed grids.

The ambipolar step predicts the coefficient field at half time, then applies
``(I - h L + h**2 L**2 / 2)**(-1)`` with that field frozen. An H-weighted
increment CG solve uses a checked true relative residual and raises on failure.
Fourier transforms in the preconditioner do not impose periodic physical y
boundaries. Strang half-steps around the ideal RK update give second-order
combined time integration. Total energy evolves through the corresponding
conservative diffusive Poynting flux, allowing magnetic energy to heat the gas.

On a periodic 32 x 32 Fourier grid, take a smooth field with
:math:`|k| \le 2` plus :math:`B_x = 1`, amplitude 0.3, :math:`\eta = 0.4`
and :math:`T = 0.4`. Against a fine-step reference of the same
2/3-dealiased discretization, the midpoint-resolvent step converges at
second order in :math:`\Delta t`: the errors are 1.28e-4, 3.44e-5, 8.94e-6,
2.28e-6 and 5.74e-7 for 8, 16, 32, 64 and 128 steps (successive ratios
3.72, 3.85, 3.92 and 3.97). Against an :math:`N = 128` reference the error
saturates at 3.8e-5, which matches the 3.6e-5 error of the 2/3-band
truncation itself. At fixed :math:`N` the attainable accuracy is therefore
set by the retained Fourier band, as for the rest of the solver, not by the
time step.

At open y faces the diffusive closure sets tangential ambipolar electric field
to zero. This differs from transparent diffusion and from the ideal-MHD open
flux. The boundary tensor is projected on both sides to preserve weighted
symmetry and dissipation; its constitutive accuracy can be lower at endpoints
with nonzero normal magnetic field. Localized modes require a wider-domain
control before interpreting their growth.

The focused tests check the SBP identity, outgoing acoustic-pulse recurrence,
projection, explicit diffusion stability, weighted ambipolar dissipation and
energy balance, nonlinear second-order time accuracy, CG failure handling,
and NumPy/Torch CPU agreement. These checks alone do not establish full MHD
boundary transparency or general accelerator parity.

Hydro acoustic pulse
~~~~~~~~~~~~~~~~~~~~

The same Fourier-x/SBP-y grid and KO6 damping are available to the two-dimensional
Euler solver. A small acoustic pulse travels toward the open upper y boundary:

.. code-block:: bash

   python -m phrike gaussian_wave2d_y --config configs/gaussian_wave2d_y_sbp.yaml --backend numpy
   python -m phrike gaussian_wave2d_y --config configs/gaussian_wave2d_y_sbp.yaml --backend torch --device cpu

The example explicitly enables ``integration.sbp_dissipation`` with strength
0.01. Its sound speed is one, and its initially upward-moving pulse reaches
the upper boundary near t=1.5. The run continues to t=4, when a periodic-y
pulse would have returned to its initial position twice. Snapshots and a final
figure are written to ``outputs/gaussian_wave2d_y_sbp``. This is a smooth, small
amplitude hydro example, not a shock test or a claim of exactly non-reflecting
boundaries. ``direction: up`` means positive y; ``down`` means negative y.

ADSI example
~~~~~~~~~~~~

.. code-block:: bash

   python -m phrike adsi2d --config configs/adsi2d.yaml --backend metal

This is a 2.5D single layer: all three vector components are retained, while
fields depend only on x and y. The flow is out of plane,
``U_z = 0.5*tanh((y - Ly/2)/0.05)``, with ``B=(b,0,b)`` and
``b=1/sqrt(4*pi)``. Density is one, thermal pressure is 60, gamma is 5/3, and
the ambipolar coupling coefficient is one. Periodic x has length one; the
shipped transverse domain has length two. This geometry suppresses the usual
flow-parallel Kelvin--Helmholtz mode while retaining ambipolar shear coupling.
With the :math:`\mu_0 = 1` convention in :doc:`equations`,
:math:`B^2 = 2b^2 = 1/(2\pi)`, the magnetic pressure :math:`B^2/2` is
:math:`1/(4\pi)`, and the plasma beta :math:`p/(B^2/2)` is :math:`240\pi`.
The Alfven speed :math:`\sqrt{B^2/\rho}` is :math:`1/\sqrt{2\pi}`.

The seed is a compressible linear eigenmode computed independently on a padded
periodic perturbation domain, with the same localized single-layer shear.
Initial in-plane magnetic perturbations use a discrete vector potential.
The reference growth is 0.450648772464. At the shipped :math:`L_y = 2`,
doubling the reference resolution (257 to 513 points) changes it by 3e-13.
Doubling the transverse domain to :math:`L_y = 4` at fixed grid spacing
(257 points at :math:`L_y = 2` against 513 at :math:`L_y = 4`) changes it by
+4.2e-9. At a fixed 257 points the same doubling changes it by -3.8e-8, but
that comparison also coarsens the spacing: at :math:`L_y = 4`, going from 257
to 513 points alone changes the growth by 4.2e-8. The 4.2e-9 figure is
therefore the domain sensitivity. This is a padded linear reference, not an
exact solution of the production open boundary conditions.

Ambipolar diffusion is what drives the instability: the shear does not grow in
ideal MHD. The ideal-MHD control evolves the same ambipolar eigenmode seed with
``physics.mhd.ambipolar.enabled: false`` (keep ``reference_alpha: 1`` so the
initial data are identical). It is a stable control that is expected not to
grow, and it has no reference eigenvalue. Asking ``reference_alpha`` for an
ideal-limit eigenmode (for example :math:`10^{-6}`) fails with an explicit
"no growing eigenmode" error, because none exists.

Every run writes ``adsi_growth.json`` and prints the envelope of the seeded
mode: the largest :math:`|B_y|` amplitude of the first x Fourier coefficient,
relative to its initial value, with its minimum and maximum over the outputs.
For an ambipolar run, a growth rate is fitted to the logarithm of that
amplitude over the last half of the outputs and reported only if the window is
a clean exponential (monotone, rms deviation from a straight line at most 0.02
in :math:`\ln` amplitude); otherwise only the envelope is reported. Ideal runs
never receive a fitted rate. The reference eigenvalue is quoted only when the
run's ``eta`` equals ``reference_alpha``.

For orientation, these runs used this checkout on NumPy float64, with
:math:`N_y = 2N_x + 1`, ``seed_amplitude`` 0.001, ``cfl`` 0.65 and the
shipped SBP dissipation and ambipolar settings, and are not the table below.
The ambipolar fit is 0.197 on 32 x 65 (the amplitude grows by only 1.09 by
t = 0.5) and 0.400 on 64 x 129 (amplitude 1.23 at t = 0.5; the reference
growth is 0.4506). The ideal control at 64 x 129 dips to 0.08 of its initial
amplitude near t = 0.15 and returns to 2.4 by t = 0.5. At 32 x 65 it dips to
0.19 near t = 0.15, rises to 3.5 near t = 1, and falls back to 0.5 by
t = 2. This oscillation of the seed is a transient whose phase and size depend
on resolution; it is not growth, and no rate is fitted to it.

Measured MPS refinement
~~~~~~~~~~~~~~~~~~~~~~~

These rows are not a run of the shipped ``configs/adsi2d.yaml``. They were
recorded on MPS float32 with the split-advection solver. Relative to that
file they override ``t_end`` from 5.0 to 0.5, ``cfl`` from 0.35 to 0.65,
and ``seed_amplitude`` from 0.0003 to 0.001, and they set
``Ny = 2*Nx + 1`` on four grids instead of the single 128 x 257 grid.
SBP dissipation strength stays 0.01, the ambipolar tolerance stays 1e-5,
and the preconditioner stays ``fourier_tensor``. The physical parameters
in the ADSI example above are unchanged. The driver that applied these
overrides is not in this repository. All four grids reached t=0.5.
Refinement reduced both the mesh spacing and the timestep; this measures
coupled space/time behavior.

The relative errors below compare the complex first x Fourier coefficients
with their initial profiles multiplied by ``exp(lambda*t)``, using the
physical SBP-weighted norm in y. They retain amplitude and phase errors.
Velocity and magnetic groups use the norm of all three corresponding primitive
components, computed in host float64 from the saved states.

.. list-table:: Linear-mode relative errors at t=0.5
   :header-rows: 1

   * - Grid
     - B_y
     - Velocity group
     - Magnetic group
   * - 32 x 65
     - 0.0885034
     - 0.256596
     - 0.218142
   * - 64 x 129
     - 0.0136797
     - 0.0583893
     - 0.0507484
   * - 128 x 257
     - 0.00155027
     - 0.00515733
     - 0.00476882
   * - 256 x 513
     - 0.00202771
     - 0.00124618
     - 0.00162361

Adjacent-grid differences in the physical B_y mode, without independently
renormalizing its amplitude, were 0.0770476, 0.0126664, and 0.00118707. The
32/64/128 sequence gives an apparent coupled refinement rate of 2.60. This is
a coarse-range measurement: the shear width spans only 1.6 y cells on the
coarsest grid. The B_y reference error improves strongly through 128, then worsens
at 256; an asymptotic order is not established.

A NumPy float64 run at 128 had B_y reference error 0.000916713 and differed from
MPS by 0.00108509, comparable to the finest adjacent-grid difference. Precision,
effective CG termination, finite amplitude, and discretization remain mixed
in this comparison; the finest B_y order is withheld.

The weak compressible components are less accurately reproduced. At 128,
density-mode error is 1209% of the expected mode: its initial norm is only
3.479e-7 and its absolute error is 5.268e-6. Pressure-mode error is 8.37%.
The supporting CPU run gives 9.39% and 0.103%, respectively. These comparisons
do not qualify the full primitive mode. Host float64 recovery cannot restore
perturbations lost during float32 evolution; the contributions from precision,
projection, finite amplitude and solve tolerances have not been isolated.

These measurements concern early mode evolution. They do not establish
nonlinear saturation convergence or independence from numerical damping.

Hall term
---------

``physics.mhd.hall`` adds Hall induction ``dB/dt = -curl E_H`` with
``E_H = eta_H (J x B)/|B|`` and a signed ``eta_h`` (``J = curl B``, Heaviside
units). It is implemented for 1D, 2D, and 3D periodic Fourier grids; mixed-open
and polynomial grids and adaptive RK stepping are rejected. Density, momentum,
and internal energy are frozen during a Hall call, so total energy changes only
by the change of ``0.5 |B|**2``. The curl uses the grid's dealias mask with zero
Nyquist symbols. The step is Hall/2, ambipolar/2, RK, projection, ambipolar/2,
Hall/2, which is palindromic; without a ``hall`` block it is unchanged.

.. code-block:: yaml

   physics:
     mhd:
       hall:
         enabled: true
         eta_h: 0.03             # signed
         eta_h_form: constant    # or proportional_to_B_over_rho (uses b0, rho0)
         method: inertia         # reference | inertia
         length: 0.05            # inertia only; omit for the automatic length
         length_mode: constant   # constant | local
         length_ratio: 4.0       # local only: raise l so max(l)/min(l) <= ratio
         safety: 0.5
         cg_tol: 1.0e-10

``reference`` is nonlinear RK4 subcycled to ``h <= safety/(max|eta_H| kmax**2)``
of the masked curl. ``inertia`` solves ``(I + curl l**2 curl) dB/dt = -curl E_H``
with ``l`` frozen per call, which divides the Hall rate of solenoidal modes by
``1 + l**2 k**2`` and leaves curl-free parts unchanged. The automatic
``l**2 = max(0, max|eta_H| dt/safety - 1/kmax**2)`` uses the Hall call's ``dt``
(half the solver step) and is zero when the Hall rate is already resolved. A
uniform ``l`` uses an exact Fourier inverse; ``l(x)`` uses preconditioned CG with
the mean-``l`` Fourier preconditioner and raises if it does not converge. Without
``length_ratio`` the CG iteration count grows with the dynamic range of ``l``.

The regularization is a scale filter, not a frequency filter: for a uniform,
resolved, unfiltered mode both the whistler and ion-cyclotron branches are
off by about ``(k l)**2``. With automatic ``l`` at fixed resolution the error
converges to true Hall at first order in ``dt`` in the regularized regime, while
a small ``dt`` recovers the unregularized Hall step. For fixed ``l(x)`` a Hall call
conserves ``0.5 (|B|**2 + l**2 |curl B|**2)``; the second term is the reservoir
(``Hall.reservoir_energy``). The ideal step and a changing automatic ``l``
exchange energy with it, so total energy is not conserved in coupled runs. The outer
step keeps the ideal CFL limit. ``scripts/hall_viability.py`` reports these
pieces (``--suite waves|invariants|convergence|energy|cost|shear|mri``) and must
be run on a cluster.

Shearing sheet and Hall-MRI
---------------------------

``physics.mhd.shearing_sheet`` (``omega``, ``shear`` or ``q`` with ``S = q omega``
and default ``q = 1.5``, ``vertical_axis`` 0, ``radial_axis`` 1) adds the
axisymmetric local-disc terms for the velocity ``u`` relative to
``v0 = -S r phi`` inside every RK stage:
``d(rho u_r)/dt += 2 Omega rho u_phi``, ``d(rho u_phi)/dt -= (2 Omega - S) rho u_r``,
``dB_phi/dt -= S B_r``, ``dE/dt += S (rho u_r u_phi - B_r B_phi)``, with
``phi = z_v x r`` taken from the right-handed basis (vertical x, radial y gives
``+z``). Internal energy is unchanged and ``B_phi`` is generated along ``phi``,
so ``div B`` is untouched. Fields must not vary along ``r``: only 1D grids with
``vertical_axis: 0`` are accepted. ``omega: 0`` with ``shear != 0`` is pure shear.

For ``k`` along ``B0`` the ``hall_mri1d`` problem seeds the fastest-growing
eigenvector of the 4x4 system in ``phrike/problems/problem_list/hall_mri1d.py``,
with ``h = eta_H sgn(B0) k**2`` and ``eta_H -> eta_H/(1 + l**2 k**2)`` for the
inertia method. The sign that matters is that of ``eta_H (Omega . B0)``. It
reproduces Balbus-Hawley at ``h = 0`` (0.75 Omega at ``k v_A = sqrt(15) Omega/4``)
and, with ``v_H**2 = 2 Omega eta_H sgn(B0)``, Balbus and Terquem (2001) eqs. 45
and 57 for ``eta = 0`` (checked by hand, not by a published table): the neutral
Keplerian wavenumbers are ``(k v_A)**2 = 3 Omega**2 - 2 Omega h`` and
``-Omega h/2``. Wardle (1999) and Sano and Stone (2002) were not compared.
