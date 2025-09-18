#!/usr/bin/env python3
"""
Debug the frame generation to see what's actually being plotted.
"""

import numpy as np
import matplotlib.pyplot as plt
import dedalus.public as d3
from sod_analytic_solution import sod_analytic_solution

def debug_frame_generation():
    """Debug what's being plotted in the frames."""
    
    # Parameters (same as video generation)
    Lx = 1.0
    Nx = 256
    xcoord = d3.Coordinate('x')
    dist = d3.Distributor(xcoord, dtype=np.float64)
    xbasis = d3.Chebyshev(xcoord, size=Nx, bounds=(0, Lx), dealias=3/2)
    
    # Get the grid
    x = dist.local_grid(xbasis)
    
    # Initial condition parameters
    x0 = 0.5
    sigma = 0.01
    
    # Sod shock tube initial conditions
    rho_left, rho_right = 1.0, 0.125
    u_left, u_right = 0.0, 0.0
    p_left, p_right = 1.0, 0.1
    
    # Create initial conditions (t=0)
    rho = rho_right + (rho_left - rho_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))
    u = u_right + (u_left - u_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))
    p = p_right + (p_left - p_right) * 0.5 * (1 + np.tanh((x0 - x) / sigma))
    
    # Get analytic solution at t=0
    rho_analytic, u_analytic, p_analytic = sod_analytic_solution(x, 0.0)
    
    # Create debug plot exactly like in the video generation
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Debug: Initial Conditions at t=0.000', fontsize=16)
    
    # Density comparison
    axes[0, 0].plot(x, rho_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
    axes[0, 0].plot(x, rho, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Density Profile')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].set_ylim(0, 1.1)
    axes[0, 0].axvline(x=0.5, color='g', linestyle=':', alpha=0.7, label='x=0.5')
    
    # Velocity comparison
    axes[0, 1].plot(x, u_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
    axes[0, 1].plot(x, u, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Velocity Profile')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].set_ylim(-1.1, 1.1)
    axes[0, 1].axvline(x=0.5, color='g', linestyle=':', alpha=0.7, label='x=0.5')
    
    # Pressure comparison
    axes[1, 0].plot(x, p_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
    axes[1, 0].plot(x, p, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Pressure')
    axes[1, 0].set_title('Pressure Profile')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].set_ylim(0, 1.1)
    axes[1, 0].axvline(x=0.5, color='g', linestyle=':', alpha=0.7, label='x=0.5')
    
    # Error plot
    rho_error = np.abs(rho - rho_analytic)
    u_error = np.abs(u - u_analytic)
    p_error = np.abs(p - p_analytic)
    
    axes[1, 1].plot(x, rho_error, 'b-', linewidth=2, label='Density Error', alpha=0.8)
    axes[1, 1].plot(x, u_error, 'g-', linewidth=2, label='Velocity Error', alpha=0.8)
    axes[1, 1].plot(x, p_error, 'r-', linewidth=2, label='Pressure Error', alpha=0.8)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Error Analysis')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('debug_initial_frame.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    # Print detailed information
    print("Debug Information:")
    print(f"  Grid points: {len(x)}")
    print(f"  x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"  x values near 0.5: {x[np.abs(x - 0.5) < 0.1]}")
    print(f"  x values near 0.85: {x[np.abs(x - 0.85) < 0.1]}")
    
    # Check density values at key locations
    idx_05 = np.argmin(np.abs(x - 0.5))
    idx_085 = np.argmin(np.abs(x - 0.85))
    
    print(f"\nDensity values:")
    print(f"  At x={x[idx_05]:.3f}: rho={rho[idx_05]:.3f}")
    print(f"  At x={x[idx_085]:.3f}: rho={rho[idx_085]:.3f}")
    
    # Find where density is halfway between left and right
    rho_mid = (rho_left + rho_right) / 2
    transition_idx = np.argmin(np.abs(rho - rho_mid))
    print(f"  Transition at x={x[transition_idx]:.3f} (rho={rho[transition_idx]:.3f})")

if __name__ == "__main__":
    debug_frame_generation()
