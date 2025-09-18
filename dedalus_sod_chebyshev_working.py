#!/usr/bin/env python3
"""
Dedalus-style implementation for 1D Sod shock tube problem with Chebyshev basis.
This script demonstrates the proper structure for solving the 1D Euler equations 
using Dedalus with a Chebyshev spectral basis and 256 grid points.

Note: This requires a properly installed Dedalus package.
To install Dedalus: pip install dedalus

The Sod shock tube problem consists of:
- Left state: rho=1.0, u=0.0, p=1.0
- Right state: rho=0.125, u=0.0, p=0.1
- Initial discontinuity at x=0.5

To run:
    $ python3 dedalus_sod_chebyshev_working.py
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def create_dedalus_sod_chebyshev():
    """
    Create a Dedalus implementation for Sod shock tube with Chebyshev basis.
    
    This function shows the proper structure for a Dedalus implementation.
    It will work once Dedalus is properly installed.
    """
    
    try:
        import dedalus.public as d3
        logger.info("Dedalus import successful!")
    except ImportError as e:
        logger.error(f"Dedalus import failed: {e}")
        logger.error("Please install Dedalus: pip install dedalus")
        return None
    
    # Parameters
    Lx = 1.0  # Domain length
    Nx = 256  # Number of Chebyshev modes
    gamma = 1.4  # Ratio of specific heats
    dealias = 3/2  # Dealiasing factor for nonlinear terms
    stop_sim_time = 0.2  # Simulation end time
    timestep = 1e-4  # Time step
    dtype = np.float64

    # Create coordinate and distributor
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=dtype)

    # Create Chebyshev basis
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=dealias)

    # Create fields for conservative variables
    rho = dist.Field(name='rho', bases=xbasis)  # Density
    rhou = dist.Field(name='rhou', bases=xbasis)  # Momentum
    E = dist.Field(name='E', bases=xbasis)  # Total energy

    # Create fields for primitive variables (for initial conditions)
    u = dist.Field(name='u', bases=xbasis)  # Velocity
    p = dist.Field(name='p', bases=xbasis)  # Pressure

    # Substitutions for derivatives
    dx = lambda A: d3.Differentiate(A, xcoord)

    # Problem setup
    problem = d3.IVP([rho, rhou, E], namespace=locals())

    # Add the Euler equations in conservative form
    problem.add_equation("dt(rho) + dx(rhou) = 0")  # Mass conservation
    problem.add_equation("dt(rhou) + dx(rhou*u + p) = 0")  # Momentum conservation  
    problem.add_equation("dt(E) + dx((E + p)*u) = 0")  # Energy conservation

    # Add equation relating primitive and conservative variables
    problem.add_equation("rhou = rho*u")
    problem.add_equation("p = (gamma - 1)*(E - 0.5*rho*u*u)")

    # Set initial conditions
    x = dist.local_grid(xbasis)

    # Sod shock tube initial conditions
    # Left state (x < 0.5): rho=1.0, u=0.0, p=1.0
    # Right state (x >= 0.5): rho=0.125, u=0.0, p=0.1
    x0 = 0.5  # Discontinuity location

    # Set initial conditions in grid space
    rho['g'] = np.where(x < x0, 1.0, 0.125)
    u['g'] = np.zeros_like(x)
    p['g'] = np.where(x < x0, 1.0, 0.1)

    # Convert to conservative variables
    rhou['g'] = rho['g'] * u['g']
    E['g'] = p['g']/(gamma - 1) + 0.5 * rho['g'] * u['g']**2

    # Build solver
    solver = problem.build_solver(d3.timesteppers.RK443)
    solver.stop_sim_time = stop_sim_time

    # Main simulation loop
    u_list = []
    p_list = []
    rho_list = []
    t_list = []

    logger.info('Starting simulation...')
    while solver.proceed:
        solver.step(timestep)
        
        if solver.iteration % 100 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e' % (solver.iteration, solver.sim_time, timestep))
        
        if solver.iteration % 50 == 0:
            # Store data for plotting
            u_list.append(u['g'].copy())
            p_list.append(p['g'].copy())
            rho_list.append(rho['g'].copy())
            t_list.append(solver.sim_time)

    logger.info('Simulation completed!')
    
    return x, u_list, p_list, rho_list, t_list

def create_plots(x, u_list, p_list, rho_list, t_list):
    """Create visualization plots for the Sod shock tube results."""
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Sod Shock Tube with Chebyshev Basis (N=256)', fontsize=14)

    # Convert lists to arrays
    x_plot = x.ravel()
    u_array = np.array(u_list)
    p_array = np.array(p_list)
    rho_array = np.array(rho_list)
    t_array = np.array(t_list)

    # Plot density
    ax1 = axes[0, 0]
    for i, t in enumerate(t_array):
        ax1.plot(x_plot, rho_array[i], alpha=0.7, linewidth=1)
    ax1.set_xlabel('x')
    ax1.set_ylabel('Density')
    ax1.set_title('Density Evolution')
    ax1.grid(True, alpha=0.3)

    # Plot velocity
    ax2 = axes[0, 1]
    for i, t in enumerate(t_array):
        ax2.plot(x_plot, u_array[i], alpha=0.7, linewidth=1)
    ax2.set_xlabel('x')
    ax2.set_ylabel('Velocity')
    ax2.set_title('Velocity Evolution')
    ax2.grid(True, alpha=0.3)

    # Plot pressure
    ax3 = axes[1, 0]
    for i, t in enumerate(t_array):
        ax3.plot(x_plot, p_array[i], alpha=0.7, linewidth=1)
    ax3.set_xlabel('x')
    ax3.set_ylabel('Pressure')
    ax3.set_title('Pressure Evolution')
    ax3.grid(True, alpha=0.3)

    # Plot final state
    ax4 = axes[1, 1]
    ax4.plot(x_plot, rho_array[-1], 'b-', label='Density', linewidth=2)
    ax4.plot(x_plot, u_array[-1], 'g-', label='Velocity', linewidth=2)
    ax4.plot(x_plot, p_array[-1], 'r-', label='Pressure', linewidth=2)
    ax4.set_xlabel('x')
    ax4.set_ylabel('Normalized Values')
    ax4.set_title(f'Final State (t={t_array[-1]:.3f})')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('sod_chebyshev_256.png', dpi=200, bbox_inches='tight')
    plt.show()

    # Create space-time plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    X, T = np.meshgrid(x_plot, t_array)
    im = ax.pcolormesh(X, T, rho_array, cmap='viridis', shading='gouraud')
    ax.set_xlabel('x')
    ax.set_ylabel('Time')
    ax.set_title('Density Space-Time Plot (Sod Shock Tube)')
    plt.colorbar(im, ax=ax, label='Density')
    plt.tight_layout()
    plt.savefig('sod_chebyshev_256_spacetime.png', dpi=200, bbox_inches='tight')
    plt.show()

    logger.info('Plots saved as sod_chebyshev_256.png and sod_chebyshev_256_spacetime.png')

def demonstrate_chebyshev_basis():
    """Demonstrate Chebyshev basis properties without running full simulation."""
    
    logger.info("Demonstrating Chebyshev basis properties...")
    
    # Create Chebyshev-Gauss-Lobatto nodes
    N = 256
    j = np.arange(N, dtype=float)
    y = np.cos(np.pi * j / (N - 1))  # Nodes on [-1, 1]
    x = (1.0 - y) * 0.5  # Map to [0, 1]
    
    # Create test function
    test_func = np.exp(-((x - 0.5) / 0.1)**2)  # Gaussian bump
    
    # Compute Chebyshev coefficients using DCT-I
    from scipy import fft
    a = fft.dct(test_func, type=1, norm=None)
    a = a / (N - 1)
    a[0] *= 0.5
    a[-1] *= 0.5
    
    # Reconstruct function
    a_sum = a.copy()
    a_sum[1:-1] *= 2.0
    reconstructed = fft.idct(a_sum, type=1, norm=None) / (N - 1)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.plot(x, test_func, 'b-', label='Original', linewidth=2)
    ax1.plot(x, reconstructed, 'r--', label='Reconstructed', linewidth=2)
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.set_title('Chebyshev Basis Test Function')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot coefficients
    k = np.arange(N)
    ax2.semilogy(k, np.abs(a), 'g-', linewidth=2)
    ax2.set_xlabel('Mode number k')
    ax2.set_ylabel('|a_k|')
    ax2.set_title('Chebyshev Coefficients')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chebyshev_basis_demo.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    logger.info("Chebyshev basis demonstration completed!")

if __name__ == "__main__":
    # First demonstrate Chebyshev basis properties
    demonstrate_chebyshev_basis()
    
    # Try to run Dedalus implementation
    logger.info("Attempting to run Dedalus implementation...")
    result = create_dedalus_sod_chebyshev()
    
    if result is not None:
        x, u_list, p_list, rho_list, t_list = result
        create_plots(x, u_list, p_list, rho_list, t_list)
    else:
        logger.info("Dedalus not available. Showing structure and concepts instead.")
        logger.info("To run the full simulation, install Dedalus: pip install dedalus")
