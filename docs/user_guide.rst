User Guide
==========

This guide provides detailed information on using PHRIKE for computational fluid dynamics simulations.

Configuration
-------------

PHRIKE uses YAML configuration files to define simulation parameters. The configuration is organized into several sections:

### Grid Configuration

The ``grid`` section defines the spatial discretization:

.. code-block:: yaml

   grid:
     N: 1024              # Number of grid points (1D)
     Nx: 128              # Grid points in x (2D/3D)
     Ny: 128              # Grid points in y (2D/3D)
     Nz: 64               # Grid points in z (3D)
     Lx: 1.0              # Domain length in x
     Ly: 1.0              # Domain length in y
     Lz: 1.0              # Domain length in z
     dealias: true        # Apply 2/3-rule dealiasing
     fft_workers: 4       # Number of FFT threads

### Physics Configuration

The ``physics`` section defines the fluid properties:

.. code-block:: yaml

   physics:
     gamma: 1.4           # Adiabatic index

### Integration Configuration

The ``integration`` section controls time stepping:

.. code-block:: yaml

   integration:
     t0: 0.0              # Initial time
     t_end: 1.0           # Final time
     cfl: 0.4             # CFL number
     scheme: rk4           # Time integration scheme (rk2 or rk4)
     output_interval: 0.1  # Time between outputs
     checkpoint_interval: 0.5  # Time between checkpoints
     spectral_filter:      # Optional spectral filtering
       enabled: true
       p: 8               # Filter order (even integer >= 2)
       alpha: 36.0        # Filter strength

### Monitoring Configuration

The ``monitoring`` section controls real-time statistics:

.. code-block:: yaml

   monitoring:
     enabled: true        # Enable monitoring (default: true)
     step_interval: 10    # Steps between monitoring output
     include_conservation: true    # Track mass, momentum, energy
     include_timestep: true        # Track timestep size
     include_velocity_stats: true  # Track velocity statistics
     output_file: null    # Output file (null = stdout)

### Initial Conditions

Each problem defines its own initial conditions section. See the problem-specific documentation for details.

Command Line Interface
----------------------

PHRIKE provides a unified command-line interface:

.. code-block:: bash

   phrike <problem> [options]

### Basic Options

- ``--config PATH``: Path to configuration file
- ``--backend {numpy,torch}``: Array backend (default: numpy)
- ``--device DEVICE``: Torch device (cpu, cuda, mps)
- ``--no-video``: Skip video generation
- ``--restart-from PATH``: Restart from checkpoint

### Examples

.. code-block:: bash

   # Basic simulation
   phrike sod --config configs/sod.yaml
   
   # GPU acceleration
   phrike khi2d --backend torch --device cuda
   
   # Restart from checkpoint
   phrike sod --restart-from outputs/sod/snapshot_t0.100000.npz
   
   # Skip video generation
   phrike tgv3d --no-video

Python API
----------

PHRIKE can be used programmatically through the Python API:

### Basic Usage

.. code-block:: python

   import phrike
   
   # Run simulation
   solver, history = phrike.run_simulation(
       problem_name="sod",
       config_path="configs/sod.yaml"
   )
   
   # Access results
   print(f"Final time: {solver.t}")
   print(f"Final state shape: {solver.U.shape}")

### Advanced Usage

.. code-block:: python

   from phrike.problems import ProblemRegistry
   
   # Create problem instance
   problem = ProblemRegistry.create_problem(
       name="sod",
       config_path="configs/sod.yaml"
   )
   
   # Run with custom parameters
   solver, history = problem.run(
       backend="torch",
       device="cuda",
       generate_video=True
   )

### Custom Configuration

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

Backend Selection
-----------------

PHRIKE supports two array backends:

### NumPy Backend (Default)

- **Pros**: Stable, well-tested, no additional dependencies
- **Cons**: CPU-only, limited memory management
- **Best for**: Small to medium problems, development

### PyTorch Backend

- **Pros**: GPU acceleration, better memory management
- **Cons**: Additional dependency, device-specific
- **Best for**: Large problems, production runs

### Device Selection

For PyTorch backend, you can specify the device:

- ``cpu``: CPU computation
- ``cuda``: NVIDIA GPU (if available)
- ``mps``: Apple Silicon GPU (if available)

.. code-block:: python

   # Automatic device detection
   solver, history = phrike.run_simulation(
       problem_name="khi2d",
       backend="torch"
   )
   
   # Explicit device selection
   solver, history = phrike.run_simulation(
       problem_name="khi2d",
       backend="torch",
       device="cuda"
   )

Performance Optimization
------------------------

### Resolution Guidelines

- **1D**: Up to 8192 points (16K with sufficient memory)
- **2D**: Up to 512² points (1024² with GPU)
- **3D**: Up to 128³ points (256³ with GPU)

### Memory Requirements

Approximate memory usage (double precision):

- **1D**: ~0.1 MB per 1000 points
- **2D**: ~0.1 MB per 100² points
- **3D**: ~0.1 MB per 100³ points

### Performance Tips

1. **Use GPU for large problems**: 3D simulations benefit significantly
2. **Enable spectral filtering**: Prevents aliasing at high resolutions
3. **Optimize CFL number**: Balance stability and efficiency
4. **Use PyFFTW**: Install for better FFT performance

.. code-block:: bash

   # Install PyFFTW for better performance
   pip install pyfftw
   
   # Use multiple FFT threads
   export OMP_NUM_THREADS=4

Troubleshooting
---------------

### Common Issues

1. **ImportError: No module named 'numba'**
   - Solution: ``pip install numba``

2. **CUDA out of memory**
   - Solution: Reduce resolution or use CPU backend

3. **FFT performance issues**
   - Solution: Install PyFFTW or increase FFT workers

4. **Simulation instability**
   - Solution: Reduce CFL number or enable spectral filtering

### Debug Mode

Enable verbose output for debugging:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.DEBUG)
   
   solver, history = phrike.run_simulation(
       problem_name="sod",
       config_path="configs/sod.yaml"
   )

### Checkpoint and Restart

PHRIKE supports checkpointing for long simulations:

.. code-block:: yaml

   integration:
     checkpoint_interval: 0.5  # Save checkpoint every 0.5 time units

Restart from checkpoint:

.. code-block:: python

   solver, history = phrike.run_simulation(
       problem_name="sod",
       config_path="configs/sod.yaml",
       restart_from="outputs/sod/snapshot_t0.500000.npz"
   )

Output and Visualization
------------------------

### Output Files

PHRIKE generates several output files:

- **Field snapshots**: ``fields_t*.png`` - Solution visualization
- **Checkpoints**: ``snapshot_t*.npz`` - Restart files
- **Conservation plots**: ``conserved.png`` - Conservation tracking
- **Videos**: ``*.mp4`` - Animated solutions

### Custom Visualization

You can create custom visualizations:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   
   # Access solution data
   rho, u, p, _ = solver.equations.primitive(solver.U)
   
   # Create custom plot
   plt.figure(figsize=(10, 6))
   plt.plot(solver.grid.x, rho, label='Density')
   plt.plot(solver.grid.x, u, label='Velocity')
   plt.plot(solver.grid.x, p, label='Pressure')
   plt.legend()
   plt.xlabel('x')
   plt.title(f'Solution at t = {solver.t:.3f}')
   plt.show()

### Video Generation

Videos are generated automatically unless disabled:

.. code-block:: bash

   # Skip video generation
   phrike sod --no-video
   
   # Custom video settings
   phrike sod --config configs/sod.yaml --video-fps 60
