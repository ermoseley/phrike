#!/usr/bin/env python3
"""
Generate video of Sod shock tube simulation with comparison to analytic solution.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging
import os
from sod_analytic_solution import sod_analytic_solution

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_sod_simulation_with_snapshots():
    """Run the Sod shock tube simulation and save snapshots for video."""
    
    # Parameters
    Lx = 1.0  # Domain length [0, 1]
    Nx = 256  # Higher resolution for sharper features
    gamma = 1.4  # Ratio of specific heats
    dealias = 3/2  # Dealiasing factor
    stop_sim_time = 0.2  # Simulation end time
    timestep = 1e-6  # Timestep
    nu = 3e-4  # Moderate artificial viscosity for stability
    dtype = np.float64
    
    # Video parameters
    n_frames = 250
    fps = 30
    frame_interval = stop_sim_time / n_frames
    
    # Create output directory
    output_dir = "sod_video_frames"
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Setting up Sod shock tube simulation with {Nx} Chebyshev points")
    logger.info(f"Will generate {n_frames} frames at {fps} fps")

    # Create coordinate and distributor
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)

    # Create Chebyshev basis
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

    # Create fields for primitive variables
    rho = dist.Field(name='rho', bases=xbasis)  # Density
    u = dist.Field(name='u', bases=xbasis)      # Velocity
    p = dist.Field(name='p', bases=xbasis)      # Pressure
    
    # Create tau terms as scalar fields (first-order reductions + PDE taus)
    tau_rho1 = dist.Field(name='tau_rho1')
    tau_rho2 = dist.Field(name='tau_rho2')
    tau_u1   = dist.Field(name='tau_u1')
    tau_u2   = dist.Field(name='tau_u2')
    tau_p1   = dist.Field(name='tau_p1')
    tau_p2   = dist.Field(name='tau_p2')

    # Substitutions for derivatives with tau terms
    dx = lambda A: d3.Differentiate(A, xcoord)
    lift_basis = xbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    
    # First-order reductions (Chebyshev tau method)
    rho_x = dx(rho) + lift(tau_rho1)
    rho_xx = dx(rho_x) + lift(tau_rho2)
    u_x = dx(u) + lift(tau_u1)
    u_xx = dx(u_x) + lift(tau_u2)
    p_x = dx(p) + lift(tau_p1)
    p_xx = dx(p_x) + lift(tau_p2)

    # Problem setup with tau terms
    problem = d3.IVP([
        rho, u, p,
        tau_rho1, tau_rho2,
        tau_u1,   tau_u2,
        tau_p1,   tau_p2,
    ], namespace=locals())
    
    # Euler equations in primitive vars with artificial viscosity
    # LHS strictly linear; all nonlinear terms on RHS
    problem.add_equation("dt(rho) - nu*rho_xx + lift(tau_rho2) = -dx(rho*u)")
    problem.add_equation("dt(u)   - nu*u_xx   + lift(tau_u2)   = -u*u_x - p_x/rho")
    problem.add_equation("dt(p)   - nu*p_xx   + lift(tau_p2)   = -gamma*p*u_x - u*p_x")
    
    # Boundary conditions:
    # Reflecting walls: u=0 at both ends (Dirichlet);
    # Neumann for rho and p: zero gradient at both ends via first-order reductions
    problem.add_equation("u(x=0) = 0")
    problem.add_equation("u(x=Lx) = 0")
    problem.add_equation("rho_x(x=0) = 0")
    problem.add_equation("rho_x(x=Lx) = 0")
    problem.add_equation("p_x(x=0) = 0")
    problem.add_equation("p_x(x=Lx) = 0")

    # Set initial conditions - smooth transition to avoid discontinuities
    x = dist.local_grid(xbasis)
    x0 = 0.5  # Discontinuity location
    sigma = 0.01  # Smooth initial condition (works better with Chebyshev grid)
    
    # Sod shock tube initial conditions with smoothing
    rho_left, rho_right = 1.0, 0.125
    u_left, u_right = 0.0, 0.0
    p_left, p_right = 1.0, 0.1
    
    # Smooth step function
    rho['g'] = rho_right + (rho_left - rho_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))
    u['g'] = u_right + (u_left - u_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))
    p['g'] = p_right + (p_left - p_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))

    # Build solver
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = stop_sim_time

    # Main simulation loop with snapshot saving
    logger.info('Starting simulation...')
    step_count = 0
    frame_count = 0
    next_snapshot_time = 0.0
    
    while solver.proceed:
        solver.step(timestep)
        step_count += 1
        
        # Check if it's time for a snapshot
        if solver.sim_time >= next_snapshot_time and frame_count < n_frames:
            # Extract current solution
            x_final = x.ravel()
            rho_final = rho['g'].ravel()
            u_final = u['g'].ravel()
            p_final = p['g'].ravel()
            
            # Ensure all arrays have the same length
            min_len = min(len(x_final), len(rho_final), len(u_final), len(p_final))
            x_final = x_final[:min_len]
            rho_final = rho_final[:min_len]
            u_final = u_final[:min_len]
            p_final = p_final[:min_len]
            
            # Get analytic solution at current time
            rho_analytic, u_analytic, p_analytic = sod_analytic_solution(x_final, solver.sim_time)
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Sod Shock Tube: Dedalus vs Analytic at t={solver.sim_time:.3f}', fontsize=16)
            
            # Density comparison
            axes[0, 0].plot(x_final, rho_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
            axes[0, 0].plot(x_final, rho_final, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
            axes[0, 0].set_xlabel('x')
            axes[0, 0].set_ylabel('Density')
            axes[0, 0].set_title('Density Profile')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].legend()
            axes[0, 0].set_ylim(0, 1.1)
            
            # Velocity comparison
            axes[0, 1].plot(x_final, u_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
            axes[0, 1].plot(x_final, u_final, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
            axes[0, 1].set_xlabel('x')
            axes[0, 1].set_ylabel('Velocity')
            axes[0, 1].set_title('Velocity Profile')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].legend()
            axes[0, 1].set_ylim(-1.1, 1.1)
            
            # Pressure comparison
            axes[1, 0].plot(x_final, p_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
            axes[1, 0].plot(x_final, p_final, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
            axes[1, 0].set_xlabel('x')
            axes[1, 0].set_ylabel('Pressure')
            axes[1, 0].set_title('Pressure Profile')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].legend()
            axes[1, 0].set_ylim(0, 1.1)
            
            # Error plot
            rho_error = np.abs(rho_final - rho_analytic)
            u_error = np.abs(u_final - u_analytic)
            p_error = np.abs(p_final - p_analytic)
            
            axes[1, 1].plot(x_final, rho_error, 'b-', linewidth=2, label='Density Error', alpha=0.8)
            axes[1, 1].plot(x_final, u_error, 'g-', linewidth=2, label='Velocity Error', alpha=0.8)
            axes[1, 1].plot(x_final, p_error, 'r-', linewidth=2, label='Pressure Error', alpha=0.8)
            axes[1, 1].set_xlabel('x')
            axes[1, 1].set_ylabel('Absolute Error')
            axes[1, 1].set_title('Error Analysis')
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].legend()
            axes[1, 1].set_yscale('log')
            
            plt.tight_layout()
            
            # Save frame
            frame_filename = f"{output_dir}/frame_{frame_count:04d}.png"
            plt.savefig(frame_filename, dpi=100, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved frame {frame_count+1}/{n_frames} at t={solver.sim_time:.3f}")
            frame_count += 1
            next_snapshot_time += frame_interval
        
        if step_count % 10000 == 0:
            logger.info(f'Step {step_count}, Time={solver.sim_time:.6f}, dt={timestep:.2e}')

    logger.info('Simulation completed!')
    logger.info(f"Generated {frame_count} frames in {output_dir}/")
    
    return output_dir, frame_count

def create_video_from_frames(output_dir, frame_count, fps=30):
    """Create video from saved frames using ffmpeg."""
    import subprocess
    
    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("ffmpeg not found. Please install ffmpeg to create video.")
        return False
    
    # Create video - try different encoders
    video_filename = "sod_shock_tube_comparison.mp4"
    
    # First try with libopenh264 (available in your ffmpeg)
    cmd = [
        'ffmpeg', '-y',  # Overwrite output file
        '-framerate', str(fps),
        '-i', f'{output_dir}/frame_%04d.png',
        '-c:v', 'libopenh264',
        '-pix_fmt', 'yuv420p',
        video_filename
    ]
    
    logger.info(f"Creating video: {video_filename}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("Video created successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Video creation failed: {e.stderr}")
        return False

def main():
    """Main function to generate video."""
    logger.info("Starting Sod shock tube video generation...")
    
    # Run simulation and save frames
    output_dir, frame_count = run_sod_simulation_with_snapshots()
    
    # Create video from frames
    success = create_video_from_frames(output_dir, frame_count, fps=30)
    
    if success:
        logger.info("Video generation completed successfully!")
        logger.info("Output: sod_shock_tube_comparison.mp4")
    else:
        logger.error("Video generation failed!")

if __name__ == "__main__":
    main()
