#!/usr/bin/env python3
"""
Simple test of Chebyshev basis with Dedalus to debug the NaN issue.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def test_simple_wave():
    """Test with a simple wave equation to verify Chebyshev basis works."""
    
    # Parameters
    Lx = 1.0
    Nx = 64  # Smaller for testing
    stop_sim_time = 0.1
    timestep = 1e-4
    dtype = np.float64

    logger.info(f"Testing simple wave with {Nx} Legendre points")

    # Create coordinate and distributor
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)

    # Create Legendre basis
    xbasis = d3.Legendre(xcoord, size=Nx, bounds=(0, Lx), dealias=1.0)  # No dealiasing for now

    # Create field
    u = dist.Field(name='u', bases=xbasis)

    # Substitutions for derivatives
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Problem setup - simple wave equation
    problem = d3.IVP([u], namespace=locals())
    problem.add_equation("dt(u) = -dx(u)")  # Simple advection equation

    # Set initial conditions - smooth function
    x = dist.local_grid(xbasis)
    u['g'] = np.exp(-((x - 0.5) / 0.1)**2)  # Gaussian bump

    # Build solver
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = stop_sim_time

    # Run simulation
    logger.info('Starting simple wave test...')
    step_count = 0
    
    while solver.proceed:
        solver.step(timestep)
        step_count += 1
        
        if step_count % 100 == 0:
            logger.info(f'Step {step_count}, Time={solver.sim_time:.6f}')

    logger.info('Simple wave test completed!')
    
    # Check results
    x_final = x.ravel()
    u_final = u['g'].ravel()
    
    logger.info(f"Final solution statistics:")
    logger.info(f"  x range: [{x_final.min():.3f}, {x_final.max():.3f}]")
    logger.info(f"  u range: [{u_final.min():.3f}, {u_final.max():.3f}]")
    logger.info(f"  u has NaNs: {np.any(np.isnan(u_final))}")
    logger.info(f"  u has infs: {np.any(np.isinf(u_final))}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x_final, u_final, 'b-', linewidth=2, label=f'u (N={Nx})')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Simple Wave Test - Final State (t={solver.sim_time:.3f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('test_chebyshev_simple.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    return x_final, u_final

if __name__ == "__main__":
    x, u = test_simple_wave()
