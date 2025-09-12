#!/usr/bin/env python3
"""Test the corrected analytical solution for wave steepening."""

import numpy as np
import matplotlib.pyplot as plt
from phrike.problems.wave_steepening1d import WaveSteepening1DProblem

def test_correct_analytical_solution():
    """Test the corrected analytical solution at different times."""
    print("=== Testing Corrected Wave-Steepening Analytical Solution ===")
    
    # Create problem
    config = {
        "grid": {"N": 256, "Lx": 2.0 * np.pi, "dealias": True},
        "physics": {"gamma": 1.4},
        "integration": {"t0": 0.0, "t_end": 25.0, "cfl": 0.4, "scheme": "rk4"},
        "io": {"outdir": "test_outputs", "save_spectra": True}
    }
    
    problem = WaveSteepening1DProblem(config=config)
    grid = problem.create_grid()
    
    # Test at different times
    times = [0.0, 5.0, 10.0, 15.0, 20.0, 25.0]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Corrected Wave-Steepening Analytical Solution Evolution', fontsize=16)
    
    for i, t in enumerate(times):
        row = i // 3
        col = i % 3
        
        rho, u, p = problem.get_analytical_solution(grid, t)
        
        axes[row, col].plot(grid.x, rho, 'b-', label='Density', linewidth=2)
        axes[row, col].plot(grid.x, u, 'r-', label='Velocity', linewidth=2)
        axes[row, col].plot(grid.x, p, 'g-', label='Pressure', linewidth=2)
        axes[row, col].set_title(f't = {t:.1f} (Shock at t=1.0)')
        axes[row, col].set_xlabel('x')
        axes[row, col].set_ylabel('Value')
        axes[row, col].legend()
        axes[row, col].grid(True)
        
        # Add shock formation indicator
        if t >= 1.0:
            axes[row, col].axvline(x=np.pi, color='k', linestyle='--', alpha=0.7, label='Shock location')
        
        # Print statistics
        print(f"t = {t:.1f}: u range [{u.min():.4f}, {u.max():.4f}], rho range [{rho.min():.4f}, {rho.max():.4f}]")
    
    plt.tight_layout()
    plt.savefig('corrected_wave_steepening_analytical.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nCorrected analytical solution test completed. Saved plot to corrected_wave_steepening_analytical.png")

if __name__ == "__main__":
    test_correct_analytical_solution()
    print("\n=== Test completed ===")
