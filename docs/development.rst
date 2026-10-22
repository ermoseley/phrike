Development
===========

Install the development and documentation dependencies:

.. code-block:: bash

   python -m pip install -e ".[dev,docs]"

Before submitting a change, run the focused test suite, build the
documentation with warnings treated as errors, and build a wheel:

.. code-block:: bash

   python -m pytest -q
   python -m sphinx -W -b html docs docs/_build/html
   python -m build --wheel --outdir dist

The source-tree post-processing, benchmark, and numerical-diagnostic tools are
documented in ``scripts/README.md``. They are intentionally separate from the
stable ``phrike`` command-line interface.

Python 3.10 or newer matches the interpreters used in CI. On Apple
Silicon with PyTorch installed, the default backend is Metal/MPS in
single precision. Tests pass an explicit backend (``backend="numpy"``,
or a Torch device chosen in the test). A laptop with Metal available
does not match the default CI job, which does not install PyTorch.

Keep generated documentation, local run directories, output files, and Numba
caches out of version control. See the repository's CONTRIBUTING.md for the
expected numerical-validation context.
