#!/usr/bin/env python3
"""
1D wave equation with proper boundary conditions using Dedalus.
This is a simpler test case to verify the tau method works correctly.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    """Run the 1D wave equation simulation with proper boundary conditions."""
    
    # Parameters
    Lx = 1.0  # Domain length
    Nx = 64   # Number of Chebyshev modes
    c = 1.0   # Wave speed
    stop_sim_time = 0.5  # Simulation end time
    timestep = 1e-4  # Time step
    dtype = np.float64

    logger.info(f"Setting up 1D wave equation with {Nx} Chebyshev points")

    # Create coordinate and distributor
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)

    # Create Chebyshev basis
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=1.0)

    # Create fields
    u = dist.Field(name='u', bases=xbasis)  # Wave amplitude
    v = dist.Field(name='v', bases=xbasis)  # Wave velocity
    
    # Create tau terms as scalar fields
    tau_u1 = dist.Field(name='tau_u1')
    tau_u2 = dist.Field(name='tau_u2')

    # Substitutions for derivatives with tau terms
    dx = lambda A: d3.Differentiate(A, xcoord)
    lift_basis = xbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)
    
    # First-order reduction for second derivatives
    u_x = dx(u) + lift(tau_u1)
    u_xx = dx(u_x) + lift(tau_u2)

    # Problem setup with tau terms
    problem = d3.IVP([u, v, tau_u1, tau_u2], namespace=locals())
    
    # Wave equation: dt(u) = v, dt(v) = c^2 * u_xx
    problem.add_equation("dt(u) = v")
    problem.add_equation("dt(v) = c*c*u_xx")
    
    # Boundary conditions - fixed ends
    problem.add_equation("u(x=0) = 0")  # Fixed at left boundary
    problem.add_equation("u(x=Lx) = 0")  # Fixed at right boundary

    # Set initial conditions - smooth wave packet
    x = dist.local_grid(xbasis)
    x0 = 0.3
    sigma = 0.05
    u['g'] = np.exp(-((x - x0) / sigma)**2)  # Gaussian wave packet
    v['g'] = np.zeros_like(x)  # Initially at rest

    # Build solver
    solver = problem.build_solver(d3.RK443)
    solver.stop_sim_time = stop_sim_time

    # Main simulation loop
    logger.info('Starting simulation...')
    step_count = 0
    
    while solver.proceed:
        solver.step(timestep)
        step_count += 1
        
        if step_count % 1000 == 0:
            logger.info(f'Step {step_count}, Time={solver.sim_time:.6f}, dt={timestep:.2e}')

    logger.info('Simulation completed!')
    
    # Extract final solution
    x_final = x.ravel()
    u_final = u['g'].ravel()
    v_final = v['g'].ravel()
    
    # Debug: Check for NaNs and print statistics
    logger.info(f"Final solution statistics:")
    logger.info(f"  x range: [{x_final.min():.3f}, {x_final.max():.3f}]")
    logger.info(f"  u range: [{u_final.min():.3f}, {u_final.max():.3f}]")
    logger.info(f"  v range: [{v_final.min():.3f}, {v_final.max():.3f}]")
    logger.info(f"  u has NaNs: {np.any(np.isnan(u_final))}")
    logger.info(f"  v has NaNs: {np.any(np.isnan(v_final))}")
    logger.info(f"  u has infs: {np.any(np.isinf(u_final))}")
    logger.info(f"  v has infs: {np.any(np.isinf(v_final))}")
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'1D Wave Equation with Chebyshev Basis (N={Nx}) at t={solver.sim_time:.3f}', fontsize=14)

    # Plot wave amplitude
    axes[0].plot(x_final, u_final, 'b-', linewidth=2, label='Wave Amplitude')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u')
    axes[0].set_title('Wave Amplitude')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Plot wave velocity
    axes[1].plot(x_final, v_final, 'g-', linewidth=2, label='Wave Velocity')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('v')
    axes[1].set_title('Wave Velocity')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('wave_equation_1d.png', dpi=200, bbox_inches='tight')
    plt.close()

    logger.info('Plot saved as wave_equation_1d.png')
    
    return x_final, u_final, v_final

if __name__ == "__main__":
    x, u, v = main()
