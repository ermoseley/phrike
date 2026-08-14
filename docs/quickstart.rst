Quick start
===========

Run a CPU acoustic-wave smoke test from the repository root:

.. code-block:: bash

   python -m phrike acoustic1d --config configs/acoustic1d.yaml \
     --backend cpu --outdir /tmp/phrike-acoustic

The command writes its outputs to the chosen directory. Video generation is
disabled unless ``--video`` is supplied.

List the available problem names and options with:

.. code-block:: bash

   python -m phrike --help

For a system with a configured accelerator backend, select it explicitly:

.. code-block:: bash

   python -m phrike alfven1d --config configs/alfven1d.yaml --backend metal
   python -m phrike alfven1d --config configs/alfven1d.yaml --backend cuda

Metal/MPS uses single precision because of the PyTorch MPS precision support.
CUDA and Metal/MPS runs should be validated on the intended hardware.
