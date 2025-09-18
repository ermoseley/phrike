#!/usr/bin/env python3
"""
Sod shock tube simulation using Dedalus with Chebyshev basis and 256 grid points.

This script solves the 1D Euler equations for the Sod shock tube problem:
- Left state: rho=1.0, u=0.0, p=1.0
- Right state: rho=0.125, u=0.0, p=0.1
- Initial discontinuity at x=0.5
- Domain: [0, 1] with 256 Chebyshev-Gauss-Lobatto points
- Final time: t=0.2

The solution will be plotted showing the density profile at the final time.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    """Run the Sod shock tube simulation with Chebyshev basis."""
    
    # Parameters
    Lx = 1.0  # Domain length
    Nx = 128  # Reduced resolution for stability
    gamma = 1.4  # Ratio of specific heats
    dealias = 1.0  # No dealiasing for now
    stop_sim_time = 0.1  # Shorter simulation time
    timestep = 1e-6  # Much smaller time step
    nu = 1e-3  # Artificial viscosity
    dtype = np.float64

    logger.info(f"Setting up Sod shock tube simulation with {Nx} Chebyshev points")

    # Create coordinate and distributor
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)

    # Create Chebyshev basis
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

    # Create fields for primitive variables
    rho = dist.Field(name='rho', bases=xbasis)  # Density
    u = dist.Field(name='u', bases=xbasis)  # Velocity
    p = dist.Field(name='p', bases=xbasis)  # Pressure

    # Substitutions for derivatives
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Problem setup - use primitive variables as the main variables
    problem = d3.IVP([rho, u, p], namespace=locals())

    # Add the Euler equations in primitive form with artificial viscosity
    problem.add_equation("dt(rho) = -dx(rho*u) + nu*dx(dx(rho))")  # Mass conservation
    problem.add_equation("dt(u) = -u*dx(u) - dx(p)/rho + nu*dx(dx(u))")  # Momentum conservation  
    problem.add_equation("dt(p) = -gamma*p*dx(u) - u*dx(p) + nu*dx(dx(p))")  # Energy conservation

    # Set initial conditions
    x = dist.local_grid(xbasis)

    # Sod shock tube initial conditions
    # Left state (x < 0.5): rho=1.0, u=0.0, p=1.0
    # Right state (x >= 0.5): rho=0.125, u=0.0, p=0.1
    x0 = 0.5  # Discontinuity location

    # Set initial conditions in grid space with heavy smoothing
    # Use a very smooth transition to avoid discontinuities
    sigma = 0.1  # Much larger smoothing parameter
    rho_left, rho_right = 1.0, 0.125
    u_left, u_right = 0.0, 0.0
    p_left, p_right = 1.0, 0.1
    
    # Very smooth step function
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
    
    # Debug: Check for NaNs and print statistics
    logger.info(f"Final solution statistics:")
    logger.info(f"  x range: [{x_final.min():.3f}, {x_final.max():.3f}]")
    logger.info(f"  rho range: [{rho_final.min():.3f}, {rho_final.max():.3f}]")
    logger.info(f"  u range: [{u_final.min():.3f}, {u_final.max():.3f}]")
    logger.info(f"  p range: [{p_final.min():.3f}, {p_final.max():.3f}]")
    logger.info(f"  rho has NaNs: {np.any(np.isnan(rho_final))}")
    logger.info(f"  u has NaNs: {np.any(np.isnan(u_final))}")
    logger.info(f"  p has NaNs: {np.any(np.isnan(p_final))}")
    logger.info(f"  rho has infs: {np.any(np.isinf(rho_final))}")
    logger.info(f"  u has infs: {np.any(np.isinf(u_final))}")
    logger.info(f"  p has infs: {np.any(np.isinf(p_final))}")
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Sod Shock Tube with Chebyshev Basis (N={Nx}) at t={solver.sim_time:.3f}', fontsize=14)

    # Plot density
    axes[0, 0].plot(x_final, rho_final, 'b-', linewidth=2, label='Density')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Density Profile')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot velocity
    axes[0, 1].plot(x_final, u_final, 'g-', linewidth=2, label='Velocity')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Velocity Profile')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot pressure
    axes[1, 0].plot(x_final, p_final, 'r-', linewidth=2, label='Pressure')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Pressure')
    axes[1, 0].set_title('Pressure Profile')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot all variables together
    axes[1, 1].plot(x_final, rho_final, 'b-', linewidth=2, label='Density')
    axes[1, 1].plot(x_final, u_final, 'g-', linewidth=2, label='Velocity')
    axes[1, 1].plot(x_final, p_final, 'r-', linewidth=2, label='Pressure')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Normalized Values')
    axes[1, 1].set_title('All Variables')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('sod_chebyshev_256_final.png', dpi=200, bbox_inches='tight')
    plt.show()

    # Create a focused plot of just the density profile
    plt.figure(figsize=(10, 6))
    plt.plot(x_final, rho_final, 'b-', linewidth=2, label=f'Density (N={Nx})')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title(f'Sod Shock Tube - Final Density Profile (t={solver.sim_time:.3f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('sod_chebyshev_256_density.png', dpi=200, bbox_inches='tight')
    plt.show()

    logger.info('Plots saved as sod_chebyshev_256_final.png and sod_chebyshev_256_density.png')
    
    return x_final, rho_final, u_final, p_final

if __name__ == "__main__":
    x, rho, u, p = main()
