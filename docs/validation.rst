Validation and limitations
==========================

Run the maintained automated checks with:

.. code-block:: bash

   python -m pytest -q

The current suite exercises adaptive Runge-Kutta coefficients and controller
order, backend selection, unavailable-device errors, and Torch CPU behavior.
It does not constitute a complete validation suite for every equation set,
configuration, or accelerator.

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
  thermal pressure, and magnetic field. Set paired overrides with
  ``grid.bc_values``. Unlike older versions, ``dirichlet`` means fixed data;
  use ``wall`` to request an impermeable endpoint.
* ``outflow`` (aliases ``open``, ``neumann``) copies adjacent nodal density,
  momenta, thermal pressure, and field values. This is endpoint extrapolation,
  not an exact polynomial derivative constraint or a characteristic
  non-reflecting boundary.
* ``wall`` (aliases ``reflective``, ``reflecting``) sets normal momentum to
  zero and extrapolates the other fields. This is a kinematic wall; no
  additional magnetic conducting-wall model is imposed.
* ``periodic`` requires Fourier. Non-periodic BCs on Fourier are rejected.

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

1D shock runs (2026-10-20)
--------------------------

The examples run sharp Sod to ``t=0.2`` and sharp Brio-Wu to ``t=0.1``:

.. code-block:: bash

   python -m phrike sod --config configs/sod_chebyshev_1d.yaml --backend cpu
   python -m phrike brio_wu1d --config configs/brio_wu1d.yaml --backend cpu
   # Repeat either with --basis legendre.

Both use N=192, NumPy CPU double precision, fixed reservoir endpoints,
CFL=0.2, RK4, and no initial smoothing. Sod uses constant viscosity 0.001;
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
Comparisons used 2001 uniformly spaced samples after linear interpolation
from the Lobatto nodes. Sod was compared with its exact Euler Riemann
solution. Brio-Wu was compared with an independent 2048-cell primitive-minmod
MUSCL/HLL finite-volume solution using SSPRK2 and outflow BCs. That reference
stayed positive without floors; its 1024-to-2048 density difference was
0.00150. It is an approximate MHD reference.

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
repairs, and accepted nodal densities stayed positive. Minimum raw pressures were 0.08118 (Sod) and 0.04312
(Brio-Wu). The extra-node interpolation safeguard is separate from this
stage-positivity check. Brio-Wu max absolute div(B) was 3.64e-12.
The relative quadrature-integrated mass and total-energy changes were
below 4e-7 in these two Chebyshev runs; this is a case-specific check.
Sod's final velocity minimum improved from about -0.36 to -0.0142;
its maximum was 0.9324 versus the exact 0.9275. Small plateau ripples and
viscous broadening remain. These runs establish neither strict monotonicity
nor resolution convergence.

Repeating the same setup on Legendre gave mean density/velocity/pressure
errors 0.00766/0.01114/0.00589 for Sod and 0.01353/0.02357/0.01553 for
Brio-Wu, also without pressure-floor activations. The minimum raw pressures
were 0.08119 and 0.04318; Brio-Wu max absolute div(B) was 1.21e-12.

The local viscous Dedalus Sod script was also run with Dedalus 3.0.3 through
``t=0.2``: N=256, viscosity 0.0003, initial tanh width 0.01, primitive-variable
diffusion, wall/Neumann tau constraints, and RK443. A timestep of 1e-5 was
qualified against the original 1e-6 through ``t=0.005``; the maximum primitive
differences were below 3.4e-8. The full run stayed finite and positive.
The local filtered variant failed on its first step with a singular system
and provides no comparison curve.

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

Sharp Chebyshev runs at N=192, 256, 384, and 512 held viscosity, initial data,
and fixed endpoints constant. Profiles were evaluated from the conservative
polynomial on 5001 common uniform points before recovering primitive fields;
linear interpolation would introduce its own resolution-dependent error.

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

The production explicit Sod solver was run at N=192 and 256. To avoid the
clustered-grid explicit diffusion restriction at higher resolution, a temporary
study driver used exact constant diffusion of the interior Laplacian with an
affine fixed-boundary lift, Strang-split around the production RK4 flux solver.
It leaves MHD Bx undiffused. Its N=192 profiles agreed with explicit evolution
to within 1.1e-6 in every primitive L1 component. A smooth, fallback-free control
gave a 3.94-fold error reduction on timestep halving. At N=512, doubling the
study timestep changed primitive profiles by at most 1.80e-7 (Sod) and 6.35e-8
(Brio-Wu), below the measured resolution differences.

All split runs recorded no nonpositive density/pressure stages or invalid
diffusion half-steps. The extra-node interpolation safeguard remained active;
N=512 used 100 (Sod) and 1336 (Brio-Wu) replacement-node evaluations over the
whole run. Maximum absolute div(B) was 4.69e-11 in the refined Brio-Wu runs.
The temporary integrator is not part of the production solver; these runs do
not qualify a new production time scheme or establish wall-clock scaling.

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

Specify ``left``/``right`` as ``periodic`` and ``bottom``/``top`` as ``open``
in ``bc_config``; ``outflow`` and ``neumann`` are aliases for this open flux
closure, not derivative constraints. Endpoints evolve through an SBP/Rusanov
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

The seed is a compressible linear eigenmode computed independently on a padded
periodic perturbation domain, with the same localized single-layer shear.
Initial in-plane magnetic perturbations use a discrete vector potential.
The reference growth is 0.450648772464; doubling reference resolution changes
it by approximately 5e-13, and doubling its transverse domain changes it by
4.2e-9. This is a padded linear reference, not an exact solution of the
production open boundary conditions. Set ``reference_alpha: 1`` when disabling
ambipolar diffusion to retain identical initial data in an ideal-MHD control.

Measured MPS refinement
~~~~~~~~~~~~~~~~~~~~~~~

A local study on 2026-10-21 used the published ``b02275c`` solver with MPS
float32, CPU fallback disabled, CFL 0.65, seed amplitude 0.001, and SBP
dissipation strength 0.01. All four grids reached t=0.5 with the physical
parameters above unchanged and ``Ny = 2*Nx + 1``. The ambipolar increment solve
used tolerance 1e-5 and the Fourier-tensor preconditioner. Refinement reduced
both the mesh spacing and timestep; this measures coupled space/time behavior.

The relative error below compares the complex first x Fourier coefficient of
``B_y`` with its initial profile multiplied by ``exp(lambda*t)``, using the
physical SBP-weighted norm in y. It retains amplitude and phase errors.

.. list-table:: Fundamental-mode error at t=0.5
   :header-rows: 1

   * - Grid
     - Relative error
   * - 32 x 65
     - 0.129218
   * - 64 x 129
     - 0.0175508
   * - 128 x 257
     - 0.00180458
   * - 256 x 513
     - 0.00203268

Adjacent-grid differences in the physical mode, without independently
renormalizing its amplitude, were 0.114586, 0.0161964, and 0.00133268. The
32/64/128 sequence gives an apparent coupled refinement rate of 2.82. This is
a coarse-range measurement: the shear width spans only 1.6 y cells on the
coarsest grid. The reference error improves strongly through 128, then worsens
at 256; an asymptotic order is not established.

A NumPy float64 run at 128 had reference error 0.00121653 and differed from
MPS by 0.00109970, comparable to the finest adjacent-grid difference. Precision,
effective CG termination, finite amplitude, and discretization remain mixed
in this comparison. Halving the seed did not produce clean harmonic scaling.
Doubling the y domain gave a centered-core difference of 0.00039937, including
an initial-profile difference of 0.00031384; this only checks early domain
sensitivity. With AD disabled and exactly the same AD-mode initial state, the
signed mode projection reversed sign and turned over by t=1.5. Its transient
``log(abs(amplitude))/t`` is not an ideal-MHD instability growth rate.

These measurements concern early mode evolution. They do not establish
nonlinear saturation convergence or independence from numerical damping.
The separate coarse saturation continuation uses strength 0.02 after a
clearly linear t=0.5 restart and therefore has different numerical controls.
