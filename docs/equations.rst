Governing equations
===================

PHRIKE integrates nondimensional equations. There is no separate vacuum
permeability: ideal MHD is implemented with :math:`\mu_0 = 1`. Magnetic
pressure is :math:`B^2/2`, with
:math:`B^2 = B_x^2 + B_y^2 + B_z^2`, and not :math:`B^2/(8\pi)`.
The adiabatic index :math:`\gamma` is ``equations.gamma`` or
``physics.gamma``.

Compressible Euler
------------------

The conserved state is density, momentum density, and total energy density.
In one dimension,

.. math::

   U = (\rho,\ \rho u,\ E), \qquad
   E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho u^2.

The flux is :math:`(\rho u,\ \rho u^2 + p,\ (E + p) u)`, and the update is
:math:`\partial_t U = -\partial_x F`. In two and three dimensions the same
closure is used with the corresponding momentum components. The sound speed
used by the timestep is :math:`a = \sqrt{\gamma p / \rho}`.

An optional constant gravity source is added only on the Euler right-hand
side. One dimension uses ``gx``: momentum gains :math:`\rho g_x` and energy
gains :math:`\rho g_x u`. Two and three dimensions add the analogous
components. The MHD right-hand side does not include this source.

Ideal MHD
---------

Every MHD dimension stores eight fields, including the out-of-plane
components:

.. math::

   U = (\rho,\ \rho u_x,\ \rho u_y,\ \rho u_z,\ E,\ B_x,\ B_y,\ B_z),

.. math::

   E = \frac{p}{\gamma - 1} + \frac{1}{2}\rho |\mathbf{u}|^2
       + \frac{1}{2} B^2.

Thermal pressure is recovered by subtracting kinetic and magnetic energy
from :math:`E`. The total pressure in the momentum and energy fluxes is

.. math::

   p_\star = p + \frac{1}{2} B^2.

The momentum flux is :math:`\rho\mathbf{u}\mathbf{u} + p_\star I - \mathbf{B}\mathbf{B}`.
The energy flux is :math:`(E + p_\star)\mathbf{u} - \mathbf{B}(\mathbf{u}\cdot\mathbf{B})`.
The induction flux is the tensor :math:`\mathbf{u}\mathbf{B} - \mathbf{B}\mathbf{u}`,
equivalent to :math:`\partial_t\mathbf{B} = \nabla\times(\mathbf{u}\times\mathbf{B})`.
In one dimension only the :math:`x`-flux is used, and the :math:`B_x` flux
row is identically zero.

The Alfvén speed consistent with :math:`\mu_0 = 1` is
:math:`\lvert\mathbf{B}\rvert/\sqrt{\rho}`. The directional Alfvén speed
along a unit normal :math:`\mathbf{n}` is
:math:`\lvert\mathbf{B}\cdot\mathbf{n}\rvert/\sqrt{\rho}`.

After Runge–Kutta updates the solver can Helmholtz-project :math:`\mathbf{B}`
and rebuild :math:`E` while holding the thermal pressure fixed. That
projection is the discrete solenoidal constraint. It is controlled by
``physics.mhd.divergence_clean`` (see :doc:`configuration`).

Wave speeds and CFL
-------------------

The hyperbolic timestep is

.. math::

   \Delta t = \mathrm{cfl}\ \frac{\Delta x_{\min}}{s_{\max}},

where :math:`s_{\max}` is the value returned by ``max_wave_speed`` and
:math:`\Delta x_{\min}` is the smallest positive node spacing on the axes
the grid has. A recorded ``dx_min``, ``dy_min``, or ``dz_min`` is used
for that axis; otherwise the solver uses ``dx``, ``dy``, or ``dz``.

A non-finite or negative wave speed raises ``FloatingPointError``. A zero
wave speed adds no hyperbolic bound. ``compute_dt`` then raises
``UnboundedTimeStep`` unless a viscous or diffusive bound is finite.
During ``run``, the step is also limited by the time remaining until the
next output, checkpoint, or ``t_end``. A zero wave speed does not fall
back to a fixed :math:`10^{-6}` step.

``max_wave_speed`` is:

* Euler, 1D: :math:`\max(\lvert u\rvert + a)`.
* Euler, 2D: :math:`\max(\max(\lvert u_x\rvert, \lvert u_y\rvert) + a)`.
* Euler, 3D: :math:`\max(\max(\lvert u_x\rvert, \lvert u_y\rvert, \lvert u_z\rvert) + a)`.
* MHD, 1D: :math:`\max(\lvert u_x\rvert + c_{f,x})`, the exact fast magnetosonic speed
  along :math:`x`. With :math:`a^2 = \gamma p/\rho`,
  :math:`b^2 = B^2/\rho`, and :math:`c_{A,x}^2 = B_x^2/\rho`,

  .. math::

     c_{f,x}^2 = \frac{1}{2}\left(a^2 + b^2
     + \sqrt{(a^2 + b^2)^2 - 4 a^2 c_{A,x}^2}\right).

* MHD, 2D and 3D: the largest Cartesian speed
  :math:`\max_i \lvert u_i\rvert` plus :math:`\sqrt{a^2 + b^2}`. This is an
  upper bound on the fast speed, using the total Alfvén speed
  :math:`\lvert\mathbf{B}\rvert/\sqrt{\rho}`. The 3D class also exposes an exact
  directional fast speed, but the timestep calls ``max_wave_speed``.

The spacing in that formula is:

* 1D Fourier: :math:`L_x/N`. 1D Legendre or Chebyshev: the minimum
  Lobatto spacing, stored as ``grid.dx``.
* 2D and 3D Fourier: :math:`L/N` on each periodic axis. The Euler solvers
  use the minimum of those spacings. A Legendre axis uses its minimum node
  spacing when the grid has recorded it.
* Fourier-:math:`x` / SBP-:math:`y`: :math:`\Delta y = L_y/(N_y - 1)`.

Further limits are applied only where the solver implements them.

* 1D and 2D Euler, and 1D MHD when artificial viscosity is enabled, also
  take the artificial-viscosity limit below.
* 2D Euler on a Fourier-:math:`x` / SBP-:math:`y` grid, when the wave
  speed is positive, also takes :math:`1/r_{\mathrm{SBP}}`. The factor
  in front of that bound is 1, not ``cfl``.
* 3D Euler has no parabolic limit in ``compute_dt``.
* MHD explicit resistivity or viscosity, and SBP dissipation on a mixed
  open-:math:`y` grid, use

  .. math::

     r = \max(\eta_m, \nu)\, k^2_{\max} + r_{\mathrm{SBP}},

  .. math::

     \Delta t_{\mathrm{diff}} = \mathrm{cfl\_diff}\,/\, r,

  and the step is the minimum of the hyperbolic and diffusive values.
  :math:`r_{\mathrm{SBP}}` is included only on mixed open-:math:`y` grids.
  ``cfl_diff`` defaults to 1. :math:`k^2_{\max}` is the largest squared
  Fourier wavenumber on a periodic grid, :math:`4/\Delta x^2` on a 1D
  polynomial grid, and a matrix-norm bound on the SBP derivative for mixed
  open-:math:`y` grids. Ambipolar diffusion is not part of this explicit
  limit.

The SBP bound is ``strength * speed / min(w_y)``. The weights :math:`w_y`
are the SBP-4(2) quadrature weights, so the endpoint weight is
:math:`(17/48)\Delta y`. The interior checkerboard decay rate of the same
operator is ``strength * speed / dy``, which is a smaller number than the
timestep bound.

Viscosity, resistivity, and ambipolar diffusion
-----------------------------------------------

Explicit MHD resistivity ``physics.mhd.resistivity``
(:math:`\eta_m`, units of length squared over time) adds
:math:`\eta_m\nabla^2\mathbf{B}` to the three magnetic components. Explicit
viscosity ``physics.mhd.viscosity`` (:math:`\nu`, the same units) adds
:math:`\nu\nabla^2(\rho\mathbf{u})` to the three momentum components. Both
operators act through the grid Laplacian. Neither term is added to the
total-energy component, so a loss of kinetic or magnetic energy reappears
as thermal pressure when :math:`p` is recovered from :math:`E`.

Artificial-viscosity coefficients ``nu_constant`` and ``nu_max`` have those
same units of length squared over time. On a 1D polynomial grid in constant
mode the term is ``nu * weight * laplacian(U)`` on each conserved row. In
the other implemented cases it is :math:`\nabla\cdot(\nu\nabla U_i)`. When the mode is ``constant``, the basis is not Fourier, the boundary
condition is the global name ``fixed``, and the grid is one-dimensional
with more than two nodes, the timestep is
:math:`1/(\nu\, w\, L)`, where :math:`L` is a row-norm bound of the interior
second-derivative matrix. Otherwise the artificial-viscosity timestep is
:math:`0.2\,\Delta x^2/(\nu w)`, where :math:`w` is the largest variable
weight and is at least 1. Sensor mode uses ``nu_max`` in that limit.
Constant mode uses ``nu_constant``, or ``nu_max`` when ``nu_constant``
is 0. On 1D MHD the artificial-viscosity term is not added to the
longitudinal magnetic component.

Ambipolar diffusion is implemented for two-dimensional eight-field MHD.
With :math:`\mathbf{J} = \nabla\times\mathbf{B}`,

.. math::

   \mathbf{E} = \eta\,(B^2 I - \mathbf{B}\mathbf{B}^{T})\,\mathbf{J}.

The induction update is :math:`-\nabla\times\mathbf{E}`. The coefficient
:math:`\eta` is ``physics.mhd.ambipolar.eta`` (``alpha`` is accepted as an
alias). Its units are length squared over time per :math:`B^2`, so that
:math:`\eta B^2` is a diffusivity. The time discretization is a frozen
midpoint resolvent, Strang-split around the ideal Runge–Kutta step, not a
forward-Euler evaluation of this term. Total energy is updated with the
discrete Poynting flux of that electric field. The open-:math:`y` closure
sets the tangential diffusive electric field to zero. Details of the
resolvent and the boundary tensor are in :doc:`validation`.

Plasma beta
-----------

Monitoring reports

.. math::

   \beta = \frac{\langle p\rangle}{\langle B^2/2\rangle}.

On a mixed open-:math:`y` grid the averages are SBP-weighted means.
Otherwise they are unweighted means of the nodal values. For a uniform
state this is :math:`2p/B^2`. It is not the Gaussian-unit ratio
:math:`8\pi p/B^2`.

The ADSI example uses :math:`\mathbf{B} = (b, 0, b)` with
:math:`b = 1/\sqrt{4\pi}`, :math:`p = 60`, and :math:`\rho = 1`. Then
:math:`B^2 = 1/(2\pi)`, the magnetic pressure is :math:`1/(4\pi)`,
:math:`\beta = 240\pi`, and the Alfvén speed is :math:`1/\sqrt{2\pi}`.

Spectral dissipation
--------------------

On a pure Fourier grid, ``integration.spectral_dissipation`` multiplies
each Fourier coefficient by :math:`\exp(-\Delta t\,\lambda)`. The integer
FFT mode index is :math:`\hat k`, and
:math:`\eta_{\hat k} = \lvert\hat k\rvert / k_{\mathrm{cut}}`. With dealiasing,
:math:`k_{\mathrm{cut}} = (N - 1)//3`; without it,
:math:`k_{\mathrm{cut}}` is the largest integer mode. Retained dealias
modes are those with :math:`\lvert\hat k\rvert \le (N - 1)//3`,
using the rounded integer FFT index.

.. math::

   \lambda(\hat k) = \mathrm{rate}\ 
   \max\left(\frac{\eta_{\hat k} - f}{1 - f},\ 0\right)^{p}.

``rate`` is :math:`\lambda` at the cutoff mode :math:`\eta = 1`. The
e-folding time of that mode is ``1/rate``. ``e_folding_time_at_cutoff``
is that time and is converted by ``rate = 1/tau``. ``onset_fraction``
is :math:`f` and must lie in :math:`[0, 1)`. ``order`` (alias ``p``)
must be even and at least 2. An accepted step applies the factor twice,
each time with :math:`\Delta t/2`, so the cutoff mode is multiplied by
:math:`\exp(-\Delta t_{\mathrm{step}}\,\mathrm{rate})`.

In two and three pure Fourier dimensions, :math:`\lambda` is the sum of
the one-dimensional profiles on each axis. A mode at the cutoff of one
axis and at zero on the others has :math:`\lambda = \mathrm{rate}`. A
mode at the cutoff of every axis has :math:`\lambda = \mathrm{rate}`
times the number of axes.

Polynomial grids do not use this rate. They use
``integration.modal_filter`` on an accepted step:
:math:`\sigma(k) = \exp(-\alpha (k/k_{\max})^{p})` with
:math:`k_{\max} = N - 1`. ``alpha`` is dimensionless. It is not a
damping rate and it is not converted from an e-folding time.
Fourier-:math:`x` / SBP-:math:`y` grids ignore spectral dissipation and
use ``integration.sbp_dissipation`` instead.
