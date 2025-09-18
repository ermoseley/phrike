#!/usr/bin/env python3
"""
Sod shock tube simulation using Dedalus with data return for comparison.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
import logging

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def run_sod_simulation():
    """Run the Sod shock tube simulation and return the data."""
    
    # Parameters
    Lx = 1.0  # Domain length [0, 1]
    Nx = 256  # Higher resolution for sharper features
    gamma = 1.4  # Ratio of specific heats
    dealias = 3/2  # Dealiasing factor
    stop_sim_time = 0.2  # Simulation end time
    timestep = 1e-6  # Timestep
    nu = 3e-4  # Moderate artificial viscosity for stability
    dtype = np.float64

    logger.info(f"Setting up Sod shock tube simulation with {Nx} Chebyshev points on domain [0, {Lx}]")

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
    
    # Ensure all arrays have the same length
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
    
    return x_final, rho_final, u_final, p_final, solver.sim_time

if __name__ == "__main__":
    x, rho, u, p, t_final = run_sod_simulation()
