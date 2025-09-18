#!/usr/bin/env python3
"""
Sod shock tube simulation using Dedalus with periodic boundary conditions on a long domain.
This avoids the complexity of the tau method while still being physically reasonable.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    """Run the Sod shock tube simulation with periodic BCs on a long domain."""
    
    # Parameters
    Lx = 10.0  # Very long domain to avoid boundary effects
    Nx = 512   # High resolution
    gamma = 1.4  # Ratio of specific heats
    dealias = 3/2  # Dealiasing factor
    stop_sim_time = 0.2  # Simulation end time
    timestep = 1e-5  # Time step
    nu = 1e-2  # Artificial viscosity
    dtype = np.float64

    logger.info(f"Setting up Sod shock tube simulation with {Nx} Fourier points on domain [0, {Lx}]")

    # Create coordinate and distributor
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)

    # Create Fourier basis (periodic, no boundary conditions needed)
    xbasis = d3.RealFourier(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

    # Create fields for primitive variables
    rho = dist.Field(name='rho', bases=xbasis)  # Density
    u = dist.Field(name='u', bases=xbasis)      # Velocity
    p = dist.Field(name='p', bases=xbasis)      # Pressure

    # Substitutions for derivatives
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Problem setup - following KdV-Burgers pattern
    problem = d3.IVP([rho, u, p], namespace=locals())
    
    # Euler equations with artificial viscosity (all nonlinear terms on RHS)
    problem.add_equation("dt(rho) - nu*dx(dx(rho)) = -dx(rho*u)")
    problem.add_equation("dt(u) - nu*dx(dx(u)) = -u*dx(u) - dx(p)/rho")
    problem.add_equation("dt(p) - nu*dx(dx(p)) = -gamma*p*dx(u) - u*dx(p)")

    # Set initial conditions - smooth transition to avoid discontinuities
    x = dist.local_grid(xbasis)
    x0 = Lx/2  # Discontinuity location at center of domain
    sigma = 0.1  # Smoothing parameter
    
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

    # Main simulation loop
    logger.info('Starting simulation...')
    step_count = 0
    
    while solver.proceed:
        solver.step(timestep)
        step_count += 1
        
        if step_count % 10000 == 0:
            logger.info(f'Step {step_count}, Time={solver.sim_time:.6f}, dt={timestep:.2e}')

    logger.info('Simulation completed!')
    
    # Extract final solution
    x_final = x.ravel()
    rho_final = rho['g'].ravel()
    u_final = u['g'].ravel()
    p_final = p['g'].ravel()
    
    # Ensure all arrays have the same length (dealias may cause different sizes)
    min_len = min(len(x_final), len(rho_final), len(u_final), len(p_final))
    x_final = x_final[:min_len]
    rho_final = rho_final[:min_len]
    u_final = u_final[:min_len]
    p_final = p_final[:min_len]
    
    # Focus on the central region where the shock is
    center_mask = (x_final >= 3.0) & (x_final <= 7.0)
    x_center = x_final[center_mask]
    rho_center = rho_final[center_mask]
    u_center = u_final[center_mask]
    p_center = p_final[center_mask]
    
    # Debug: Check for NaNs and print statistics
    logger.info(f"Final solution statistics:")
    logger.info(f"  x range: [{x_center.min():.3f}, {x_center.max():.3f}]")
    logger.info(f"  rho range: [{rho_center.min():.3f}, {rho_center.max():.3f}]")
    logger.info(f"  u range: [{u_center.min():.3f}, {u_center.max():.3f}]")
    logger.info(f"  p range: [{p_center.min():.3f}, {p_center.max():.3f}]")
    logger.info(f"  rho has NaNs: {np.any(np.isnan(rho_center))}")
    logger.info(f"  u has NaNs: {np.any(np.isnan(u_center))}")
    logger.info(f"  p has NaNs: {np.any(np.isnan(p_center))}")
    logger.info(f"  rho has infs: {np.any(np.isinf(rho_center))}")
    logger.info(f"  u has infs: {np.any(np.isinf(u_center))}")
    logger.info(f"  p has infs: {np.any(np.isinf(p_center))}")
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Sod Shock Tube with Fourier Basis (N={Nx}) at t={solver.sim_time:.3f}', fontsize=14)

    # Plot density
    axes[0, 0].plot(x_center, rho_center, 'b-', linewidth=2, label='Density')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Density Profile (Central Region)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot velocity
    axes[0, 1].plot(x_center, u_center, 'g-', linewidth=2, label='Velocity')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Velocity Profile (Central Region)')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot pressure
    axes[1, 0].plot(x_center, p_center, 'r-', linewidth=2, label='Pressure')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Pressure')
    axes[1, 0].set_title('Pressure Profile (Central Region)')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Combined plot
    axes[1, 1].plot(x_center, rho_center, 'b--', linewidth=1, label='Density')
    axes[1, 1].plot(x_center, u_center, 'g:', linewidth=1, label='Velocity')
    axes[1, 1].plot(x_center, p_center, 'r-.', linewidth=1, label='Pressure')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Values')
    axes[1, 1].set_title('Combined Profiles (Central Region)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('sod_periodic_long_final.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

    # Plot only density profile
    plt.figure(figsize=(8, 6))
    plt.plot(x_center, rho_center, 'b-', linewidth=2, label='Density')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title(f'Sod Shock Tube Density Profile (N={Nx}) at t={solver.sim_time:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sod_periodic_long_density.png', dpi=200, bbox_inches='tight')
    plt.close()

    logger.info('Plots saved as sod_periodic_long_final.png and sod_periodic_long_density.png')
    
    return x_center, rho_center, u_center, p_center

if __name__ == "__main__":
    x, rho, u, p = main()
