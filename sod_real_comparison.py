#!/usr/bin/env python3
"""
Create real comparison plot between Dedalus solution and analytic solution.
"""

import numpy as np
import matplotlib.pyplot as plt
from sod_analytic_solution import sod_analytic_solution
from sod_chebyshev_with_data_return import run_sod_simulation

def create_real_comparison_plot():
    """Create real comparison plot between Dedalus and analytic solutions."""
    
    print("Running Dedalus simulation...")
    # Get Dedalus solution
    x_dedalus, rho_dedalus, u_dedalus, p_dedalus, t_final = run_sod_simulation()
    
    print("Computing analytic solution...")
    # Get analytic solution on the same grid
    rho_analytic, u_analytic, p_analytic = sod_analytic_solution(x_dedalus, t_final)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Sod Shock Tube: Dedalus vs Analytic Solution at t={t_final:.3f}', fontsize=16)
    
    # Density comparison
    axes[0, 0].plot(x_dedalus, rho_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
    axes[0, 0].plot(x_dedalus, rho_dedalus, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
    axes[0, 0].set_xlabel('x')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Density Profile')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Velocity comparison
    axes[0, 1].plot(x_dedalus, u_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
    axes[0, 1].plot(x_dedalus, u_dedalus, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
    axes[0, 1].set_xlabel('x')
    axes[0, 1].set_ylabel('Velocity')
    axes[0, 1].set_title('Velocity Profile')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Pressure comparison
    axes[1, 0].plot(x_dedalus, p_analytic, 'b-', linewidth=2, label='Analytic', alpha=0.8)
    axes[1, 0].plot(x_dedalus, p_dedalus, 'r--', linewidth=2, label='Dedalus', alpha=0.8)
    axes[1, 0].set_xlabel('x')
    axes[1, 0].set_ylabel('Pressure')
    axes[1, 0].set_title('Pressure Profile')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Error plot
    rho_error = np.abs(rho_dedalus - rho_analytic)
    u_error = np.abs(u_dedalus - u_analytic)
    p_error = np.abs(p_dedalus - p_analytic)
    
    axes[1, 1].plot(x_dedalus, rho_error, 'b-', linewidth=2, label='Density Error', alpha=0.8)
    axes[1, 1].plot(x_dedalus, u_error, 'g-', linewidth=2, label='Velocity Error', alpha=0.8)
    axes[1, 1].plot(x_dedalus, p_error, 'r-', linewidth=2, label='Pressure Error', alpha=0.8)
    axes[1, 1].set_xlabel('x')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Error Analysis')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('sod_real_comparison_plot.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    print("Real comparison plot saved as sod_real_comparison_plot.png")
    
    # Print statistics
    print(f"\nAnalytic solution statistics:")
    print(f"  Density range: [{rho_analytic.min():.3f}, {rho_analytic.max():.3f}]")
    print(f"  Velocity range: [{u_analytic.min():.3f}, {u_analytic.max():.3f}]")
    print(f"  Pressure range: [{p_analytic.min():.3f}, {p_analytic.max():.3f}]")
    
    print(f"\nDedalus solution statistics:")
    print(f"  Density range: [{rho_dedalus.min():.3f}, {rho_dedalus.max():.3f}]")
    print(f"  Velocity range: [{u_dedalus.min():.3f}, {u_dedalus.max():.3f}]")
    print(f"  Pressure range: [{p_dedalus.min():.3f}, {p_dedalus.max():.3f}]")
    
    print(f"\nError statistics:")
    print(f"  Max density error: {rho_error.max():.3f}")
    print(f"  Max velocity error: {u_error.max():.3f}")
    print(f"  Max pressure error: {p_error.max():.3f}")
    print(f"  RMS density error: {np.sqrt(np.mean(rho_error**2)):.3f}")
    print(f"  RMS velocity error: {np.sqrt(np.mean(u_error**2)):.3f}")
    print(f"  RMS pressure error: {np.sqrt(np.mean(p_error**2)):.3f}")

if __name__ == "__main__":
    create_real_comparison_plot()
