#!/usr/bin/env python3
"""
2D Kelvin-Helmholtz Instability simulation using Dedalus with phrike-compatible initial conditions.
This script uses the same initial conditions as the phrike khi2d problem and saves data every dt=0.02.

To run in parallel with 4 processes:
    $ mpiexec -n 4 python khi2d_dedalus.py
"""

import numpy as np
import dedalus.public as d3
import logging
import os
import h5py
from datetime import datetime
from mpi4py import MPI

# Set up MPI and logging
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set up logging (only on rank 0 to avoid duplicate messages)
if rank == 0:
    logging.basicConfig(level=logging.INFO)
else:
    logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

def run_khi2d_dedalus_simulation():
    """Run the 2D KHI simulation using Dedalus with phrike-compatible initial conditions."""
    
    # Parameters from phrike khi2d config
    Lx, Ly = 1.0, 1.0  # Domain size
    Nx, Ny = 128, 128  # Grid resolution
    gamma = 1.4  # Ratio of specific heats
    dealias = 3/2  # Dealiasing factor
    stop_sim_time = 5.0  # Simulation end time
    max_timestep = 1e-4  # Maximum timestep
    dt_save = 0.02  # Save data every 0.02 time units
    dtype = np.float64
    
    # Artificial viscosity for stability
    nu = 1e-5
    
    # Create output directory
    output_dir = "khi2d_dedalus_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create snapshots directory for HDF5 files
    snapshots_dir = os.path.join(output_dir, "snapshots")
    os.makedirs(snapshots_dir, exist_ok=True)
    
    # Clear existing snapshots but preserve plotting script
    if rank == 0:
        import shutil
        if os.path.exists(snapshots_dir):
            for file in os.listdir(snapshots_dir):
                if file.endswith('.h5'):
                    os.remove(os.path.join(snapshots_dir, file))
        print(f"Cleared existing snapshots in {snapshots_dir}")
    
    if rank == 0:
        logger.info(f"Setting up 2D KHI simulation with {Nx}x{Ny} Fourier modes")
        logger.info(f"Running on {size} processes")
        logger.info(f"Domain: [{Lx}, {Ly}], Save interval: {dt_save}")
        logger.info(f"Output directory: {output_dir}")

    # Create coordinate system and distributor
    coords = d3.CartesianCoordinates('x', 'y')
    dist = d3.Distributor(coords, dtype=dtype)

    # Create Fourier bases for periodic boundaries
    xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
    ybasis = d3.RealFourier(coords['y'], size=Ny, bounds=(0, Ly), dealias=dealias)

    # Create fields for primitive variables
    rho = dist.Field(name='rho', bases=(xbasis, ybasis))  # Density
    u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))  # Velocity
    p = dist.Field(name='p', bases=(xbasis, ybasis))  # Pressure
    
    # Create tau terms for pressure (incompressible constraint)
    tau_p = dist.Field(name='tau_p')

    # Substitutions
    x, y = dist.local_grids(xbasis, ybasis)
    ex, ey = coords.unit_vector_fields(dist)
    
    # Problem setup
    problem = d3.IVP([rho, u, p], namespace=locals())
    
    # 2D Compressible Euler equations in primitive form with artificial viscosity
    # LHS must be linear, RHS contains nonlinear terms
    problem.add_equation("dt(rho) - nu*lap(rho) = - div(rho*u)")
    problem.add_equation("dt(u) - nu*lap(u) = - u@grad(u) - grad(p)/rho")
    problem.add_equation("dt(p) - nu*lap(p) = - gamma*p*div(u) - u@grad(p)")

    # Build solver
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = stop_sim_time

    # Set initial conditions matching phrike khi2d
    # Parameters from phrike config
    rho_outer = 1.0
    rho_inner = 2.0
    u0 = 1.0
    shear_thickness = 0.02
    pressure_outer = 1.0
    pressure_inner = 1.0
    perturb_eps = 0.01
    perturb_sigma = 0.02
    perturb_kx = 2
    
    # Normalize coordinates to [0,1) like phrike
    yn = y / Ly  # normalized y coordinates
    
    # Velocity profile: two tanh transitions at y=0.25 and y=0.75
    U1 = -0.5 * u0  # outer regions
    U2 = +0.5 * u0  # middle region
    a = shear_thickness
    tanh_low = np.tanh((yn - 0.25) / a)
    tanh_high = np.tanh((yn - 0.75) / a)
    u['g'][0] = U1 + 0.5 * (U2 - U1) * (tanh_low - tanh_high)  # ux component
    
    # Density profile: same tanh structure
    rho['g'] = rho_outer + 0.5 * (rho_inner - rho_outer) * (tanh_low - tanh_high)
    
    # Pressure profile: same tanh structure
    p['g'] = pressure_outer + 0.5 * (pressure_inner - pressure_outer) * (tanh_low - tanh_high)
    
    # vy perturbation: Gaussian-modulated sinusoidal at both interfaces
    gauss = np.exp(-((yn - 0.25)**2) / (2.0 * perturb_sigma**2)) + np.exp(
        -((yn - 0.75)**2) / (2.0 * perturb_sigma**2)
    )
    sinus = np.sin(2.0 * np.pi * perturb_kx * x / Lx)
    u['g'][1] = perturb_eps * sinus * gauss  # uy component

    # Analysis - save data every dt_save
    snapshots = solver.evaluator.add_file_handler(
        snapshots_dir, 
        sim_dt=dt_save, 
        max_writes=int(stop_sim_time / dt_save) + 1
    )
    snapshots.add_task(rho, name='density')
    snapshots.add_task(u, name='velocity')
    snapshots.add_task(p, name='pressure')
    snapshots.add_task(-d3.div(d3.skew(u)), name='vorticity')
    
    # Add kinetic energy and other diagnostics
    snapshots.add_task(0.5 * rho * (u@u), name='kinetic_energy')
    snapshots.add_task(p / (gamma - 1), name='internal_energy')

    # CFL condition
    CFL = d3.CFL(solver, initial_dt=max_timestep, cadence=10, safety=0.2, threshold=0.1,
                 max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(u)

    # Flow properties for monitoring (only in serial mode)
    if size == 1:
        flow = d3.GlobalFlowProperty(solver, cadence=1)
        flow.add_property(rho, name='rho')
        flow.add_property(u@u, name='u2')
        flow.add_property(p, name='p')

    # Main simulation loop
    if rank == 0:
        logger.info('Starting simulation...')
    step_count = 0
    last_save_time = 0.0
    
    try:
        while solver.proceed:
            timestep = CFL.compute_timestep()
            solver.step(timestep)
            step_count += 1
            
            # Log progress (only on rank 0)
            if rank == 0 and step_count % 10 == 0:
                if size == 1:
                    max_rho = flow.max('rho')
                    max_u = np.sqrt(flow.max('u2'))
                    max_p = flow.max('p')
                    logger.info(f'Step {step_count}, Time={solver.sim_time:.6f}, dt={timestep:.2e}, '
                              f'max(rho)={max_rho:.3f}, max(u)={max_u:.3f}, max(p)={max_p:.3f}')
                else:
                    logger.info(f'Step {step_count}, Time={solver.sim_time:.6f}, dt={timestep:.2e}')
            
            # Check if it's time to save data
            if solver.sim_time - last_save_time >= dt_save:
                if rank == 0:
                    logger.info(f'Saving snapshot at t={solver.sim_time:.3f} (Step {step_count})')
                last_save_time = solver.sim_time
                
    except Exception as e:
        if rank == 0:
            logger.error(f'Exception raised: {e}')
        raise
    finally:
        solver.log_stats()

    if rank == 0:
        logger.info('Simulation completed!')
        logger.info(f"Data saved to: {snapshots_dir}")
    
    return output_dir, snapshots_dir

def create_visualization_script(snapshots_dir):
    """Create a script to visualize the saved data."""
    
    viz_script = f"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os

def plot_khi2d_snapshots(snapshots_dir):
    \"\"\"Plot snapshots from the KHI2D simulation.\"\"\"
    
    # Find all snapshot files
    snapshot_files = sorted(glob.glob(os.path.join(snapshots_dir, "snapshots_s*.h5")))
    
    if not snapshot_files:
        print("No snapshot files found!")
        return
    
    print(f"Found {{len(snapshot_files)}} snapshot files")
    
    # Create output directory for plots
    plot_dir = os.path.join(os.path.dirname(snapshots_dir), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    for snapshot_file in snapshot_files:
        print(f"Processing {{snapshot_file}}")
        
        with h5py.File(snapshot_file, 'r') as f:
            # Get all time steps
            times = f['scales']['sim_time'][:]
            n_times = len(times)
            
            print(f"Found {{n_times}} time steps from t={{times[0]:.3f}} to t={{times[-1]:.3f}}")
            
            # Plot every 10th time step to avoid too many plots
            step = max(1, n_times // 50)  # Show up to 50 frames
            
            for i in range(0, n_times, step):
                # Get data for this time step
                density = f['tasks']['density'][i, :, :]
                velocity = f['tasks']['velocity'][i, :, :, :]  # [2, Ny, Nx]
                pressure = f['tasks']['pressure'][i, :, :]
                vorticity = f['tasks']['vorticity'][i, :, :]
                
                # Get time
                time = times[i]
                
                # Create figure
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'KHI2D Simulation at t={{time:.3f}}', fontsize=16)
                
                # Density
                im1 = axes[0, 0].imshow(density, origin='lower', aspect='equal', cmap='viridis')
                axes[0, 0].set_title('Density')
                axes[0, 0].set_xlabel('x')
                axes[0, 0].set_ylabel('y')
                plt.colorbar(im1, ax=axes[0, 0])
                
                # Velocity magnitude
                vel_mag = np.sqrt(velocity[0]**2 + velocity[1]**2)
                im2 = axes[0, 1].imshow(vel_mag, origin='lower', aspect='equal', cmap='plasma')
                axes[0, 1].set_title('Velocity Magnitude')
                axes[0, 1].set_xlabel('x')
                axes[0, 1].set_ylabel('y')
                plt.colorbar(im2, ax=axes[0, 1])
                
                # Pressure
                im3 = axes[1, 0].imshow(pressure, origin='lower', aspect='equal', cmap='coolwarm')
                axes[1, 0].set_title('Pressure')
                axes[1, 0].set_xlabel('x')
                axes[1, 0].set_ylabel('y')
                plt.colorbar(im3, ax=axes[1, 0])
                
                # Vorticity
                im4 = axes[1, 1].imshow(vorticity, origin='lower', aspect='equal', cmap='RdBu_r')
                axes[1, 1].set_title('Vorticity')
                axes[1, 1].set_xlabel('x')
                axes[1, 1].set_ylabel('y')
                plt.colorbar(im4, ax=axes[1, 1])
                
                plt.tight_layout()
                
                # Save plot
                plot_file = os.path.join(plot_dir, f"khi2d_frame_{{i:04d}}_t{{time:.3f}}.png")
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved plot: {{plot_file}}")

if __name__ == "__main__":
    plot_khi2d_snapshots("{snapshots_dir}")
"""
    
    viz_file = os.path.join(os.path.dirname(snapshots_dir), "plot_khi2d_snapshots.py")
    with open(viz_file, 'w') as f:
        f.write(viz_script)
    
    logger.info(f"Created visualization script: {viz_file}")
    return viz_file

def main():
    """Main function to run the KHI2D simulation."""
    if rank == 0:
        logger.info("Starting 2D KHI simulation with Dedalus...")
    
    # Run simulation
    output_dir, snapshots_dir = run_khi2d_dedalus_simulation()
    
    # Create visualization script (only on rank 0)
    if rank == 0:
        viz_script = create_visualization_script(snapshots_dir)
        logger.info("Simulation completed successfully!")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Snapshots directory: {snapshots_dir}")
        logger.info(f"To visualize results, run: python {viz_script}")

if __name__ == "__main__":
    main()
