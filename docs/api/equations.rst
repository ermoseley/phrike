Equations Module
================

The equations module provides implementations of the compressible Euler equations in 1D, 2D, and 3D.

EulerEquations1D
~~~~~~~~~~~~~~~~

.. autoclass:: phrike.equations.EulerEquations1D
   :members:
   :undoc-members:
   :show-inheritance:

EulerEquations2D
~~~~~~~~~~~~~~~~

.. autoclass:: phrike.equations.EulerEquations2D
   :members:
   :undoc-members:
   :show-inheritance:

EulerEquations3D
~~~~~~~~~~~~~~~~

.. autoclass:: phrike.equations.EulerEquations3D
   :members:
   :undoc-members:
   :show-inheritance:

Numba Kernels
~~~~~~~~~~~~~

The module also includes Numba-accelerated kernels for performance-critical operations:

.. autofunction:: phrike.equations._primitive_kernel
.. autofunction:: phrike.equations._flux_kernel
.. autofunction:: phrike.equations._max_wave_speed_kernel
.. autofunction:: phrike.equations._primitive2d_kernel
.. autofunction:: phrike.equations._flux2d_kernel
.. autofunction:: phrike.equations._max_wave_speed2d_kernel
.. autofunction:: phrike.equations._primitive3d_kernel
.. autofunction:: phrike.equations._flux3d_kernel
.. autofunction:: phrike.equations._max_wave_speed3d_kernel
