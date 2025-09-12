Installation
============

PHRIKE requires Python 3.9 or higher and can be installed from source.

Dependencies
------------

PHRIKE depends on the following packages:

* **Core Dependencies**:
  - `numpy >= 1.22` - Array operations
  - `scipy >= 1.10` - Scientific computing
  - `matplotlib >= 3.5` - Visualization
  - `PyYAML >= 6.0` - Configuration files
  - `numba >= 0.58` - JIT compilation

* **Optional Dependencies**:
  - `pyfftw >= 0.13` - High-performance FFT (install with ``pip install phrike[fastfft]``)
  - `torch` - GPU acceleration (install separately)
  - `pytest >= 7.0` - Testing (install with ``pip install phrike[dev]``)

Installation Methods
--------------------

From Source
~~~~~~~~~~~

1. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/your-username/phrike.git
      cd phrike

2. Install in development mode:

   .. code-block:: bash

      pip install -e .

3. Install with optional dependencies:

   .. code-block:: bash

      pip install -e .[fastfft,dev]

Using pip
~~~~~~~~~

.. code-block:: bash

   pip install phrike

Using conda
~~~~~~~~~~~

.. code-block:: bash

   conda install -c conda-forge numpy scipy matplotlib pyyaml numba
   pip install phrike

Verification
------------

After installation, verify PHRIKE is working correctly:

.. code-block:: python

   import phrike
   print(f"PHRIKE version: {phrike.__version__}")
   
   # List available problems
   from phrike.problems import ProblemRegistry
   print("Available problems:", ProblemRegistry.list_problems())

GPU Support
-----------

For GPU acceleration, install PyTorch:

.. code-block:: bash

   # For CUDA (Linux/Windows)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # For MPS (Apple Silicon Macs)
   pip install torch torchvision torchaudio

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

1. **ImportError: No module named 'numba'**
   - Install numba: ``pip install numba``

2. **FFT performance issues**
   - Install pyfftw: ``pip install pyfftw``

3. **GPU not detected**
   - Verify PyTorch installation: ``python -c "import torch; print(torch.cuda.is_available())"``

4. **Memory issues on large problems**
   - Use smaller resolutions or enable spectral filtering
   - Consider using the Torch backend for better memory management

Performance Tips
~~~~~~~~~~~~~~~~

* Use ``pyfftw`` for better FFT performance
* Enable multi-threading with ``fft_workers`` in configuration
* Use GPU backend for large 3D problems
* Enable spectral filtering for stability at high resolutions
