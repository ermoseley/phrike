Quick start
===========

Run a CPU acoustic-wave smoke test from the repository root:

.. code-block:: bash

   python -m phrike acoustic1d --config configs/acoustic1d.yaml \
     --backend cpu --outdir /tmp/phrike-acoustic

The command writes its outputs to the chosen directory. The command-line
interface leaves video off unless ``--video`` is supplied or
``video.enabled`` is true. ``run_simulation`` and ``BaseProblem.run``
also default ``generate_video`` to false.

List the available problem names and options with:

.. code-block:: bash

   python -m phrike --help

For a system with a configured accelerator backend, select it explicitly:

.. code-block:: bash

   python -m phrike alfven1d --config configs/alfven1d.yaml --backend metal
   python -m phrike alfven1d --config configs/alfven1d.yaml --backend cuda

Metal/MPS uses single precision. On Apple Silicon, that is also the
default backend when PyTorch is installed and no backend is selected.
CUDA and Metal/MPS runs should be validated on the intended hardware.
The equations, CFL definition, and YAML keys are in :doc:`equations` and
:doc:`configuration`.
