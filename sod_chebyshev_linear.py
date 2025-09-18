#!/usr/bin/env python3
"""
Linear wave equation simulation using Dedalus with Chebyshev basis.
This is a much more stable alternative to the full Euler equations.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def main():
    """Run a linear wave equation simulation with Chebyshev basis."""
    
    # Parameters
    Lx = 1.0  # Domain length
    Nx = 256  # Number of Chebyshev modes
    c = 1.0  # Wave speed
    stop_sim_time = 0.5  # Simulation end time
    timestep = 1e-4  # Time step
    dtype = np.float64

    logger.info(f"Setting up linear wave simulation with {Nx} Legendre points")

    # Create coordinate and distributor
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)

    # Create Legendre basis
    xbasis = d3.Legendre(xcoord, size=Nx, bounds=(0, Lx), dealias=1.0)

    # Create fields
    u = dist.Field(name='u', bases=xbasis)  # Wave amplitude
    v = dist.Field(name='v', bases=xbasis)  # Wave velocity
    tau_u = dist.Field(name='tau_u', bases=xbasis)  # Tau term for boundary conditions

    # Substitutions for derivatives
    dx = lambda A: d3.Differentiate(A, xcoord)
    lift_basis = xbasis.derivative_basis(1)
    lift = lambda A: d3.Lift(A, lift_basis, -1)

    # Problem setup - linear wave equation with boundary conditions
    problem = d3.IVP([u, v, tau_u], namespace=locals())
    problem.add_equation("dt(u) = v")  # Velocity equation
    problem.add_equation("dt(v) = c*c*dx(dx(u)) + lift(tau_u)")  # Wave equation with tau term
    
    # Add boundary conditions for the wave equation
    # At x=0: u = 0 (fixed end)
    problem.add_equation("u(x=0) = 0")
    # At x=Lx: u = 0 (fixed end)
    problem.add_equation("u(x=Lx) = 0")

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
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Linear Wave Equation with Legendre Basis (N={Nx}) at t={solver.sim_time:.3f}', fontsize=14)

    # Plot wave amplitude
    axes[0, 0].plot(x_final, u_final, 'b-', linewidth=2, label='Wave Amplitude')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('u')
    axes[0, 0].set_title('Wave Amplitude')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Plot wave velocity
    axes[0, 1].plot(x_final, v_final, 'g-', linewidth=2, label='Wave Velocity')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('v')
    axes[0, 1].set_title('Wave Velocity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    # Plot both together
    axes[1, 0].plot(x_final, u_final, 'b-', linewidth=2, label='Amplitude')
    axes[1, 0].plot(x_final, v_final, 'g-', linewidth=2, label='Velocity')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Values')
    axes[1, 0].set_title('Wave Solution')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Plot energy (u^2 + v^2)
    energy = u_final**2 + v_final**2
    axes[1, 1].plot(x_final, energy, 'r-', linewidth=2, label='Energy')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Energy')
    axes[1, 1].set_title('Wave Energy')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig('linear_wave_legendre_256.png', dpi=200, bbox_inches='tight')
    plt.show()

    # Create a focused plot of just the wave amplitude
    plt.figure(figsize=(10, 6))
    plt.plot(x_final, u_final, 'b-', linewidth=2, label=f'Wave Amplitude (N={Nx})')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title(f'Linear Wave Equation with Legendre Basis - Final Amplitude (t={solver.sim_time:.3f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('linear_wave_legendre_256_amplitude.png', dpi=200, bbox_inches='tight')
    plt.show()

    logger.info('Plots saved as linear_wave_legendre_256.png and linear_wave_legendre_256_amplitude.png')
    
    return x_final, u_final, v_final

if __name__ == "__main__":
    x, u, v = main()
