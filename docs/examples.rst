Examples
========

This section provides comprehensive examples of using PHRIKE for various computational fluid dynamics problems.

1D Examples
-----------

### Sod Shock Tube

The Sod shock tube is a classic Riemann problem used to validate shock-capturing schemes.

.. code-block:: python

   import phrike
   
   # Run Sod shock tube
   solver, history = phrike.run_simulation(
       problem_name="sod",
       config_path="configs/sod.yaml"
   )
   
   # Plot results
   import matplotlib.pyplot as plt
   rho, u, p, _ = solver.equations.primitive(solver.U)
   
   plt.figure(figsize=(12, 4))
   plt.subplot(1, 3, 1)
   plt.plot(solver.grid.x, rho)
   plt.title('Density')
   plt.xlabel('x')
   
   plt.subplot(1, 3, 2)
   plt.plot(solver.grid.x, u)
   plt.title('Velocity')
   plt.xlabel('x')
   
   plt.subplot(1, 3, 3)
   plt.plot(solver.grid.x, p)
   plt.title('Pressure')
   plt.xlabel('x')
   
   plt.tight_layout()
   plt.show()

### Acoustic Wave Propagation

Test linear wave propagation and spectral accuracy:

.. code-block:: python

   # Run acoustic wave test
   solver, history = phrike.run_simulation(
       problem_name="acoustic1d",
       config_path="configs/acoustic.yaml"
   )
   
   # Check conservation
   print(f"Mass conservation error: {abs(history['mass'][-1] - history['mass'][0]):.2e}")
   print(f"Momentum conservation error: {abs(history['momentum'][-1] - history['momentum'][0]):.2e}")

### Gaussian Wave Packet

Test wave packet propagation and dispersion:

.. code-block:: python

   # Stationary wave packet
   solver, history = phrike.run_simulation(
       problem_name="gaussian_wave1d",
       config_path="configs/gaussian_wave_stationary.yaml"
   )
   
   # Traveling wave packet
   solver, history = phrike.run_simulation(
       problem_name="gaussian_wave1d",
       config_path="configs/gaussian_wave_traveling.yaml"
   )

2D Examples
-----------

### Kelvin-Helmholtz Instability

Study the growth of shear layer instabilities:

.. code-block:: python

   # Run KHI simulation
   solver, history = phrike.run_simulation(
       problem_name="khi2d",
       config_path="configs/khi2d.yaml"
   )
   
   # Analyze growth rate
   import numpy as np
   t = np.array(history['time'])
   mom_y = np.array(history['momentum_y'])
   
   # Plot momentum evolution
   plt.figure(figsize=(10, 6))
   plt.plot(t, mom_y)
   plt.xlabel('Time')
   plt.ylabel('Y-momentum')
   plt.title('KHI Growth')
   plt.grid(True)
   plt.show()

### Custom 2D Problem

Create a custom 2D problem:

.. code-block:: python

   from phrike.problems import BaseProblem
   from phrike.grid import Grid2D
   from phrike.equations import EulerEquations2D
   from phrike.solver import SpectralSolver2D
   
   class Custom2DProblem(BaseProblem):
       def create_grid(self, backend="numpy", device=None):
           return Grid2D(
               Nx=128, Ny=128,
               Lx=2.0, Ly=2.0,
               dealias=True,
               backend=backend,
               torch_device=device
           )
       
       def create_equations(self):
           return EulerEquations2D(gamma=self.gamma)
       
       def create_initial_conditions(self, grid):
           # Custom initial conditions
           X, Y = grid.xy_mesh()
           rho = 1.0 + 0.1 * np.sin(2 * np.pi * X)
           ux = 0.1 * np.cos(2 * np.pi * Y)
           uy = 0.0
           p = 1.0
           
           return self.equations.conservative(rho, ux, uy, p)

3D Examples
-----------

### Taylor-Green Vortex

Validate against the decaying Taylor-Green vortex:

.. code-block:: python

   # Run TGV simulation
   solver, history = phrike.run_simulation(
       problem_name="tgv3d",
       config_path="configs/tgv3d.yaml"
   )
   
   # Compute kinetic energy
   rho, ux, uy, uz, p = solver.equations.primitive(solver.U)
   kinetic_energy = 0.5 * (ux**2 + uy**2 + uz**2)
   total_ke = kinetic_energy.sum()
   
   print(f"Total kinetic energy: {total_ke:.6f}")

### 3D Turbulence

Study forced turbulence:

.. code-block:: python

   # Run turbulence simulation
   solver, history = phrike.run_simulation(
       problem_name="turb3d",
       config_path="configs/turb3d.yaml"
   )
   
   # Analyze energy spectrum
   def compute_energy_spectrum(ux, uy, uz, grid):
       # Compute 3D FFT
       ux_k = np.fft.fftn(ux)
       uy_k = np.fft.fftn(uy)
       uz_k = np.fft.fftn(uz)
       
       # Energy spectrum
       E_k = 0.5 * (np.abs(ux_k)**2 + np.abs(uy_k)**2 + np.abs(uz_k)**2)
       
       return E_k
   
   rho, ux, uy, uz, p = solver.equations.primitive(solver.U)
   E_k = compute_energy_spectrum(ux, uy, uz, solver.grid)

Performance Examples
--------------------

### Benchmarking

Compare performance across resolutions:

.. code-block:: python

   import time
   import numpy as np
   
   resolutions = [64, 128, 256, 512]
   times = []
   
   for N in resolutions:
       config = {
           "grid": {"N": N, "Lx": 2.0},
           "physics": {"gamma": 1.4},
           "integration": {"t_end": 0.1, "cfl": 0.4}
       }
       
       start_time = time.time()
       solver, history = phrike.run_simulation(
           problem_name="sod",
           config=config
       )
       end_time = time.time()
       
       times.append(end_time - start_time)
       print(f"N={N}: {times[-1]:.2f}s")
   
   # Plot scaling
   plt.loglog(resolutions, times, 'o-')
   plt.xlabel('Resolution')
   plt.ylabel('Time (s)')
   plt.title('Performance Scaling')
   plt.grid(True)
   plt.show()

### GPU Acceleration

Use GPU acceleration for large problems:

.. code-block:: python

   # CPU version
   start_time = time.time()
   solver_cpu, _ = phrike.run_simulation(
       problem_name="khi2d",
       config_path="configs/khi2d.yaml",
       backend="numpy"
   )
   cpu_time = time.time() - start_time
   
   # GPU version
   start_time = time.time()
   solver_gpu, _ = phrike.run_simulation(
       problem_name="khi2d",
       config_path="configs/khi2d.yaml",
       backend="torch",
       device="cuda"
   )
   gpu_time = time.time() - start_time
   
   print(f"CPU time: {cpu_time:.2f}s")
   print(f"GPU time: {gpu_time:.2f}s")
   print(f"Speedup: {cpu_time/gpu_time:.2f}x")

Monitoring Examples
-------------------

### Custom Monitoring

Add custom monitoring callbacks:

.. code-block:: python

   def custom_monitor(step, dt, U):
       if step % 100 == 0:
           rho, u, p, _ = solver.equations.primitive(U)
           max_rho = rho.max()
           min_rho = rho.min()
           print(f"Step {step}: dt={dt:.2e}, rho_range=[{min_rho:.3f}, {max_rho:.3f}]")
   
   # Run with custom monitoring
   solver, history = phrike.run_simulation(
       problem_name="sod",
       config_path="configs/sod.yaml",
       on_step=custom_monitor
   )

### Conservation Analysis

Analyze conservation properties:

.. code-block:: python

   def analyze_conservation(history):
       t = np.array(history['time'])
       mass = np.array(history['mass'])
       momentum = np.array(history['momentum'])
       energy = np.array(history['energy'])
       
       # Compute relative errors
       mass_error = np.abs(mass - mass[0]) / mass[0]
       momentum_error = np.abs(momentum - momentum[0]) / momentum[0]
       energy_error = np.abs(energy - energy[0]) / energy[0]
       
       plt.figure(figsize=(12, 4))
       
       plt.subplot(1, 3, 1)
       plt.semilogy(t, mass_error)
       plt.title('Mass Conservation Error')
       plt.xlabel('Time')
       plt.ylabel('Relative Error')
       
       plt.subplot(1, 3, 2)
       plt.semilogy(t, momentum_error)
       plt.title('Momentum Conservation Error')
       plt.xlabel('Time')
       plt.ylabel('Relative Error')
       
       plt.subplot(1, 3, 3)
       plt.semilogy(t, energy_error)
       plt.title('Energy Conservation Error')
       plt.xlabel('Time')
       plt.ylabel('Relative Error')
       
       plt.tight_layout()
       plt.show()
   
   # Analyze conservation
   analyze_conservation(history)

Visualization Examples
----------------------

### Custom Plots

Create custom visualizations:

.. code-block:: python

   def plot_solution_evolution(solver, history):
       # Create subplots
       fig, axes = plt.subplots(2, 2, figsize=(12, 8))
       
       # Plot density evolution
       rho, u, p, _ = solver.equations.primitive(solver.U)
       im1 = axes[0, 0].plot(solver.grid.x, rho)
       axes[0, 0].set_title('Density')
       axes[0, 0].set_xlabel('x')
       
       # Plot velocity evolution
       im2 = axes[0, 1].plot(solver.grid.x, u)
       axes[0, 1].set_title('Velocity')
       axes[0, 1].set_xlabel('x')
       
       # Plot pressure evolution
       im3 = axes[1, 0].plot(solver.grid.x, p)
       axes[1, 0].set_title('Pressure')
       axes[1, 0].set_xlabel('x')
       
       # Plot conservation
       t = np.array(history['time'])
       mass = np.array(history['mass'])
       axes[1, 1].plot(t, mass)
       axes[1, 1].set_title('Mass Conservation')
       axes[1, 1].set_xlabel('Time')
       axes[1, 1].set_ylabel('Mass')
       
       plt.tight_layout()
       plt.show()
   
   # Create custom plot
   plot_solution_evolution(solver, history)

### 2D Visualization

Visualize 2D solutions:

.. code-block:: python

   def plot_2d_solution(solver, field_name='density'):
       rho, ux, uy, p = solver.equations.primitive(solver.U)
       
       if field_name == 'density':
           field = rho
       elif field_name == 'velocity_magnitude':
           field = np.sqrt(ux**2 + uy**2)
       elif field_name == 'pressure':
           field = p
       
       plt.figure(figsize=(10, 8))
       plt.imshow(field, extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                  origin='lower', aspect='equal')
       plt.colorbar(label=field_name)
       plt.xlabel('x')
       plt.ylabel('y')
       plt.title(f'{field_name.title()} at t = {solver.t:.3f}')
       plt.show()
   
   # Plot 2D solution
   plot_2d_solution(solver, 'density')
   plot_2d_solution(solver, 'velocity_magnitude')
