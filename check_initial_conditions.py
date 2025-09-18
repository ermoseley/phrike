#!/usr/bin/env python3
"""
Check the initial conditions for the Sod shock tube problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

def check_initial_conditions():
    """Check and visualize the initial conditions."""
    
    # Parameters (same as in video generation)
    Lx = 1.0
    Nx = 256
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)
    
    # Get the grid - Chebyshev points are not uniformly distributed
    x = dist.local_grid(xbasis)
    
    # For Chebyshev, we need to map from [-1, 1] to [0, Lx]
    # The Chebyshev points are already mapped by Dedalus, but let's verify
    print(f"Chebyshev grid points (first 10): {x[:10]}")
    print(f"Chebyshev grid points (last 10): {x[-10:]}")
    
    # Initial condition parameters
    x0 = 0.5  # Discontinuity location
    sigma = 0.01  # Less sharp initial condition to work with Chebyshev grid
    
    # Sod shock tube initial conditions
    rho_left, rho_right = 1.0, 0.125
    u_left, u_right = 0.0, 0.0
    p_left, p_right = 1.0, 0.1
    
    # Create smooth step function
    rho = rho_right + (rho_left - rho_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))
    u = u_right + (u_left - u_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))
    p = p_right + (p_left - p_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))
    
    # Plot initial conditions
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Density
    axes[0].plot(x, rho, 'b-', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Initial Density Profile')
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='x=0.5')
    axes[0].legend()
    
    # Velocity
    axes[1].plot(x, u, 'g-', linewidth=2)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Velocity')
    axes[1].set_title('Initial Velocity Profile')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='x=0.5')
    axes[1].legend()
    
    # Pressure
    axes[2].plot(x, p, 'r-', linewidth=2)
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('Pressure')
    axes[2].set_title('Initial Pressure Profile')
    axes[2].grid(True, alpha=0.3)
    axes[2].axvline(x=0.5, color='r', linestyle='--', alpha=0.7, label='x=0.5')
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig('initial_conditions_check.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("Initial Conditions Analysis:")
    print(f"  Domain: x ∈ [0, {Lx}]")
    print(f"  Discontinuity location: x = {x0}")
    print(f"  Smoothing parameter: σ = {sigma}")
    print(f"  Grid points: {len(x)}")
    print(f"  x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  Density range: [{rho.min():.3f}, {rho.max():.3f}]")
    print(f"  Velocity range: [{u.min():.3f}, {u.max():.3f}]")
    print(f"  Pressure range: [{p.min():.3f}, {p.max():.3f}]")
    
    # Check where the transition occurs
    # Find where density is halfway between left and right values
    rho_mid = (rho_left + rho_right) / 2
    transition_idx = np.argmin(np.abs(rho - rho_mid))
    transition_x = x[transition_idx]
    print(f"  Density transition occurs at x ≈ {transition_x:.3f}")
    
    # Check if the problem is properly centered
    if abs(transition_x - 0.5) < 0.01:
        print("  ✓ Initial conditions are properly centered at x = 0.5")
    else:
        print(f"  ⚠ Initial conditions are NOT centered! Transition at x = {transition_x:.3f} instead of 0.5")

if __name__ == "__main__":
    check_initial_conditions()
