#!/usr/bin/env python3
"""
Investigate the Chebyshev grid distribution and its effect on the visual representation.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3

def investigate_grid_distribution():
    """Investigate how Chebyshev grid distribution affects the visual representation."""
    
    # Parameters
    Lx = 1.0
    Nx = 256
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)
    
    # Get the Chebyshev grid
    x_cheb = dist.local_grid(xbasis)
    
    # Create a uniform grid for comparison
    x_uniform = np.linspace(0, 1, Nx)
    
    # Initial condition parameters
    x0 = 0.5  # Discontinuity location
    sigma = 0.01  # Smoothing parameter
    
    # Sod shock tube initial conditions
    rho_left, rho_right = 1.0, 0.125
    p_left, p_right = 1.0, 0.1
    
    # Create initial conditions on both grids
    rho_cheb = rho_right + (rho_left - rho_right) * 0.5 * (1 + np.tanh((x0 - x_cheb) / sigma))
    p_cheb = p_right + (p_left - p_right) * 0.5 * (1 + np.tanh((x0 - x_cheb) / sigma))
    
    rho_uniform = rho_right + (rho_left - rho_right) * 0.5 * (1 + np.tanh((x0 - x_uniform) / sigma))
    p_uniform = p_right + (p_left - p_right) * 0.5 * (1 + np.tanh((x0 - x_uniform) / sigma))
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Grid point distribution
    axes[0, 0].plot(x_cheb, np.ones_like(x_cheb), 'ro', markersize=2, label='Chebyshev points')
    axes[0, 0].plot(x_uniform, np.ones_like(x_uniform) * 0.5, 'bo', markersize=2, label='Uniform points')
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Grid points')
    axes[0, 0].set_title('Grid Point Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x=0.5')
    
    # Plot 2: Density on Chebyshev grid
    axes[0, 1].plot(x_cheb, rho_cheb, 'r-', linewidth=2, label='Chebyshev grid')
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Density on Chebyshev Grid')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x=0.5')
    axes[0, 1].legend()
    
    # Plot 3: Density on uniform grid
    axes[1, 0].plot(x_uniform, rho_uniform, 'b-', linewidth=2, label='Uniform grid')
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Density on Uniform Grid')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x=0.5')
    axes[1, 0].legend()
    
    # Plot 4: Both overlaid
    axes[1, 1].plot(x_cheb, rho_cheb, 'r-', linewidth=2, label='Chebyshev grid')
    axes[1, 1].plot(x_uniform, rho_uniform, 'b--', linewidth=2, label='Uniform grid')
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Comparison: Chebyshev vs Uniform')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=0.5, color='g', linestyle='--', alpha=0.7, label='x=0.5')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('grid_distribution_analysis.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print statistics
    print("Grid Distribution Analysis:")
    print(f"  Chebyshev grid points: {len(x_cheb)}")
    print(f"  Uniform grid points: {len(x_uniform)}")
    print(f"  Chebyshev x range: [{x_cheb.min():.3f}, {x_cheb.max():.3f}]")
    print(f"  Uniform x range: [{x_uniform.min():.3f}, {x_uniform.max():.3f}]")
    
    # Count points in different regions
    left_region_cheb = np.sum(x_cheb < 0.3)
    middle_region_cheb = np.sum((x_cheb >= 0.3) & (x_cheb <= 0.7))
    right_region_cheb = np.sum(x_cheb > 0.7)
    
    left_region_uniform = np.sum(x_uniform < 0.3)
    middle_region_uniform = np.sum((x_uniform >= 0.3) & (x_uniform <= 0.7))
    right_region_uniform = np.sum(x_uniform > 0.7)
    
    print(f"\nPoint distribution:")
    print(f"  Chebyshev - Left (<0.3): {left_region_cheb}, Middle (0.3-0.7): {middle_region_cheb}, Right (>0.7): {right_region_cheb}")
    print(f"  Uniform   - Left (<0.3): {left_region_uniform}, Middle (0.3-0.7): {middle_region_uniform}, Right (>0.7): {right_region_uniform}")
    
    # Find where the transition visually appears
    rho_mid = (rho_left + rho_right) / 2
    transition_idx_cheb = np.argmin(np.abs(rho_cheb - rho_mid))
    transition_idx_uniform = np.argmin(np.abs(rho_uniform - rho_mid))
    
    print(f"\nVisual transition location:")
    print(f"  Chebyshev grid: x = {x_cheb[transition_idx_cheb]:.3f}")
    print(f"  Uniform grid:   x = {x_uniform[transition_idx_uniform]:.3f}")
    print(f"  Expected:       x = 0.500")

if __name__ == "__main__":
    investigate_grid_distribution()
