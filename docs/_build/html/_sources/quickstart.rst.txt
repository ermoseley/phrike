Quick Start Guide
=================

This guide will get you running PHRIKE simulations in minutes.

Basic Usage
-----------

The simplest way to run a simulation is using the command-line interface:

.. code-block:: bash

   # Run a 1D Sod shock tube
   phrike sod --config configs/sod.yaml
   
   # Run 2D Kelvin-Helmholtz instability
   phrike khi2d --config configs/khi2d.yaml
   
   # Run with GPU acceleration
   phrike tgv3d --backend torch --device cuda

Python API
----------

You can also use PHRIKE programmatically:

.. code-block:: python

   import phrike
   
   # Run simulation and get results
   solver, history = phrike.run_simulation(
       problem_name="sod",
       config_path="configs/sod.yaml"
   )
   
   # Access final state
   print(f"Final time: {solver.t}")
   print(f"Final density range: {solver.U[0].min():.3f} to {solver.U[0].max():.3f}")

Configuration Files
-------------------

PHRIKE uses YAML configuration files. Here's a basic example:

.. code-block:: yaml

   problem: sod
   
   grid:
     N: 1024
     Lx: 1.0
     dealias: true
   
   physics:
     gamma: 1.4
   
   integration:
     t0: 0.0
     t_end: 0.2
     cfl: 0.4
     scheme: rk4
   
   initial_conditions:
     left:
       rho: 1.0
       u: 0.0
       p: 1.0
     right:
       rho: 0.125
       u: 0.0
       p: 0.1

Available Problems
------------------

PHRIKE comes with several built-in problems:

* **1D Problems**:
  - ``sod`` - Sod shock tube
  - ``acoustic1d`` - Acoustic wave propagation
  - ``gaussian_wave1d`` - Gaussian wave packet (stationary/traveling)

* **2D Problems**:
  - ``khi2d`` - Kelvin-Helmholtz instability

* **3D Problems**:
  - ``tgv3d`` - Taylor-Green vortex
  - ``turb3d`` - 3D turbulence

Running Your First Simulation
-----------------------------

1. **Choose a problem**:

   .. code-block:: bash

      phrike sod --config configs/sod.yaml

2. **Check the output**:

   The simulation will create an ``outputs/sod/`` directory with:
   - Field snapshots (``fields_t*.png``)
   - Checkpoint files (``snapshot_t*.npz``)
   - Conservation plots (``conserved.png``)

3. **View results**:

   Open the generated PNG files to see the solution evolution.

Advanced Usage
--------------

Custom Configuration
~~~~~~~~~~~~~~~~~~~~

You can override configuration parameters:

.. code-block:: python

   config = {
       "grid": {"N": 512, "Lx": 2.0},
       "physics": {"gamma": 1.4},
       "integration": {"t_end": 1.0, "cfl": 0.3}
   }
   
   solver, history = phrike.run_simulation(
       problem_name="sod",
       config=config
   )

GPU Acceleration
~~~~~~~~~~~~~~~~

For large problems, use GPU acceleration:

.. code-block:: python

   solver, history = phrike.run_simulation(
       problem_name="khi2d",
       config_path="configs/khi2d.yaml",
       backend="torch",
       device="cuda"  # or "mps" for Apple Silicon
   )

Monitoring
~~~~~~~~~~

Enable real-time monitoring:

.. code-block:: yaml

   monitoring:
     enabled: true
     step_interval: 10
     include_conservation: true
     include_timestep: true

Restart from Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~

Restart a simulation from a checkpoint:

.. code-block:: python

   solver, history = phrike.run_simulation(
       problem_name="sod",
       config_path="configs/sod.yaml",
       restart_from="outputs/sod/snapshot_t0.100000.npz"
   )

Next Steps
----------

* Read the :doc:`user_guide` for detailed usage instructions
* Check out :doc:`examples` for more complex scenarios
* See :doc:`api_reference` for the complete API documentation
