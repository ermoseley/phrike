#!/usr/bin/env python3
"""
Compare Chebyshev vs Legendre grid distributions for the Sod problem.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

def compare_grids():
    """Compare Chebyshev vs Legendre grid distributions."""
    
    # Parameters
    Lx = 1.0
    Nx = 256
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.float64)
    
    # Create both bases
    xbasis_cheb = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)
    xbasis_leg = d3.Legendre(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)
    
    # Get the grids
    x_cheb = dist.local_grid(xbasis_cheb)
    x_leg = dist.local_grid(xbasis_leg)
    
    # Create uniform grid for reference
    x_uniform = np.linspace(0, 1, Nx)
    
    # Initial condition parameters
    x0 = 0.5
    sigma = 0.01
    
    # Sod shock tube initial conditions
    rho_left, rho_right = 1.0, 0.125
    
    # Create initial conditions on all grids
    rho_cheb = rho_right + (rho_left - rho_right) * 0.5 * (1 + np.tanh((x0 - x_cheb) / sigma))
    rho_leg = rho_right + (rho_left - rho_right) * 0.5 * (1 + np.tanh((x0 - x_leg) / sigma))
    rho_uniform = rho_right + (rho_left - rho_right) * 0.5 * (1 + np.tanh((x0 - x_uniform) / sigma))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Grid Distribution Comparison: Chebyshev vs Legendre', fontsize=16)
    
    # Plot 1: Grid point distribution
    axes[0, 0].plot(x_cheb, np.ones_like(x_cheb), 'ro', markersize=2, label='Chebyshev', alpha=0.7)
    axes[0, 0].plot(x_leg, np.ones_like(x_leg) * 0.5, 'bo', markersize=2, label='Legendre', alpha=0.7)
    axes[0, 0].plot(x_uniform, np.ones_like(x_uniform) * 0.25, 'go', markersize=1, label='Uniform', alpha=0.5)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Grid points')
    axes[0, 0].set_title('Grid Point Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x=0.5')
    
    # Plot 2: Density on Chebyshev grid
    axes[0, 1].plot(x_cheb, rho_cheb, 'r-', linewidth=2, label='Chebyshev')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Density on Chebyshev Grid')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x=0.5')
    axes[0, 1].legend()
    
    # Plot 3: Density on Legendre grid
    axes[1, 0].plot(x_leg, rho_leg, 'b-', linewidth=2, label='Legendre')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Density on Legendre Grid')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x=0.5')
    axes[1, 0].legend()
    
    # Plot 4: Both overlaid
    axes[1, 1].plot(x_cheb, rho_cheb, 'r-', linewidth=2, label='Chebyshev', alpha=0.8)
    axes[1, 1].plot(x_leg, rho_leg, 'b--', linewidth=2, label='Legendre', alpha=0.8)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Comparison: Chebyshev vs Legendre')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x=0.5')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('grid_comparison_chebyshev_legendre.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print detailed statistics
    print("Grid Distribution Analysis:")
    print(f"  Chebyshev grid points: {len(x_cheb)}")
    print(f"  Legendre grid points: {len(x_leg)}")
    print(f"  Uniform grid points: {len(x_uniform)}")
    
    # Count points in different regions
    regions = [(0.0, 0.3, "Left"), (0.3, 0.7, "Middle"), (0.7, 1.0, "Right")]
    
    print(f"\nPoint distribution by region:")
    for start, end, name in regions:
        cheb_count = np.sum((x_cheb >= start) & (x_cheb < end))
        leg_count = np.sum((x_leg >= start) & (x_leg < end))
        uniform_count = np.sum((x_uniform >= start) & (x_uniform < end))
        
        print(f"  {name} ({start}-{end}): Chebyshev={cheb_count}, Legendre={leg_count}, Uniform={uniform_count}")
    
    # Find where the transition visually appears
    rho_mid = (rho_left + rho_right) / 2
    transition_idx_cheb = np.argmin(np.abs(rho_cheb - rho_mid))
    transition_idx_leg = np.argmin(np.abs(rho_leg - rho_mid))
    transition_idx_uniform = np.argmin(np.abs(rho_uniform - rho_mid))
    
    print(f"\nVisual transition location:")
    print(f"  Chebyshev grid: x = {x_cheb[transition_idx_cheb]:.3f}")
    print(f"  Legendre grid:  x = {x_leg[transition_idx_leg]:.3f}")
    print(f"  Uniform grid:   x = {x_uniform[transition_idx_uniform]:.3f}")
    print(f"  Expected:       x = 0.500")
    
    # Check density values at key locations
    print(f"\nDensity values at x=0.5:")
    idx_cheb_05 = np.argmin(np.abs(x_cheb - 0.5))
    idx_leg_05 = np.argmin(np.abs(x_leg - 0.5))
    idx_uniform_05 = np.argmin(np.abs(x_uniform - 0.5))
    
    print(f"  Chebyshev: x={x_cheb[idx_cheb_05]:.3f}, rho={rho_cheb[idx_cheb_05]:.3f}")
    print(f"  Legendre:  x={x_leg[idx_leg_05]:.3f}, rho={rho_leg[idx_leg_05]:.3f}")
    print(f"  Uniform:   x={x_uniform[idx_uniform_05]:.3f}, rho={rho_uniform[idx_uniform_05]:.3f}")

if __name__ == "__main__":
    compare_grids()
