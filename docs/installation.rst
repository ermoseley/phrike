Installation
============

PHRIKE requires Python 3.9 or newer. Clone the source repository and install
it into an isolated environment:

.. code-block:: bash

   git clone https://github.com/ermoseley/phrike.git
   cd phrike
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -e ".[dev]"

Accelerator backends use the optional Torch dependency:

.. code-block:: bash

   python -m pip install -e ".[torch]"

The package is installed from source; this repository does not claim a PyPI
release.
