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
non-periodic Legendre paths remain exploratory, and 3D Legendre support is not
implemented.
