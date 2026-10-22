Installation
============

PHRIKE requires Python 3.10 or newer. Continuous integration tests 3.10
and 3.11. Clone the source repository and install it into an isolated
environment:

.. code-block:: bash

   git clone https://github.com/ermoseley/phrike.git
   cd phrike
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

The package is installed from source. This repository does not claim a
PyPI release.

Optional extras
---------------

* ``torch`` installs PyTorch for the CPU, CUDA, and Metal/MPS backends.
  On Apple Silicon, if PyTorch is installed and a grid is left on the
  default backend ``auto``, the grid uses Metal/MPS. That device runs in
  single precision, including when the configuration asks for double
  precision. Select ``--backend cpu`` or ``backend="numpy"`` for a
  NumPy run. Tests pass an explicit backend into the grid. The default
  CI job does not install PyTorch. A separate job installs a CPU-only
  PyTorch wheel and runs the suite there. CUDA and Metal hardware are
  not exercised in CI.
* ``fastfft`` installs pyFFTW. Fourier transforms use it when the import
  succeeds, and otherwise use SciPy.
* ``finufft`` installs FINUFFT. Off-grid evaluation of Fourier fields,
  including tracer sampling, uses it when the import succeeds, and
  otherwise uses a direct sum.
* ``high-precision`` installs mpmath, which is already a required
  dependency, for high-precision Gauss–Lobatto nodes. Nothing in PHRIKE
  imports SymPy, so SymPy is not an extra.

``mpmath`` is imported by the Legendre quadrature path. A missing
``mpmath`` install raises ``ImportError`` only when that path is used.
