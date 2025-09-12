#!/usr/bin/env python3
"""Debug script for Shu-Osher problem."""

import numpy as np
import matplotlib.pyplot as plt
from phrike.problems.shu_osher1d import ShuOsher1DProblem
from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D

# Create problem
config = {
    "grid": {"N": 256, "Lx": 1.0, "dealias": True},
    "physics": {"gamma": 1.4},
    "integration": {
        "t0": 0.0,
        "t_end": 0.2,
        "cfl": 0.4,
        "scheme": "rk4",
        "output_interval": 0.02,
        "checkpoint_interval": 0.1,
        "spectral_filter": {"enabled": True, "p": 8, "alpha": 36.0}
    },
    "io": {"outdir": "outputs/debug", "save_spectra": True}
}

problem = ShuOsher1DProblem(config=config)
grid = problem.create_grid()
equations = problem.create_equations()

print(f"Grid N: {grid.N}, Lx: {grid.Lx}")
print(f"Grid x shape: {grid.x.shape}")
print(f"Grid x range: [{grid.x.min():.3f}, {grid.x.max():.3f}]")

# Create initial conditions
U = problem.create_initial_conditions(grid)
print(f"Initial U shape: {U.shape}")
print(f"Initial U min/max: [{U.min():.6f}, {U.max():.6f}]")

# Check for NaN values
print(f"Initial U has NaN: {np.any(np.isnan(U))}")
print(f"Initial U has inf: {np.any(np.isinf(U))}")

# Convert to primitive variables
rho, u, p, _ = equations.primitive(U)
print(f"Primitive variables:")
print(f"  rho: min={rho.min():.6f}, max={rho.max():.6f}, has_nan={np.any(np.isnan(rho))}")
print(f"  u: min={u.min():.6f}, max={u.max():.6f}, has_nan={np.any(np.isnan(u))}")
print(f"  p: min={p.min():.6f}, max={p.max():.6f}, has_nan={np.any(np.isnan(p))}")

# Plot initial conditions
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
axs[0, 0].plot(grid.x, rho, 'b-', linewidth=2)
axs[0, 0].set_title('Initial Density')
axs[0, 0].set_xlabel('x')
axs[0, 0].set_ylabel('ρ')
axs[0, 0].grid(True, alpha=0.3)

axs[0, 1].plot(grid.x, u, 'r-', linewidth=2)
axs[0, 1].set_title('Initial Velocity')
axs[0, 1].set_xlabel('x')
axs[0, 1].set_ylabel('u')
axs[0, 1].grid(True, alpha=0.3)

axs[1, 0].plot(grid.x, p, 'g-', linewidth=2)
axs[1, 0].set_title('Initial Pressure')
axs[1, 0].set_xlabel('x')
axs[1, 0].set_ylabel('p')
axs[1, 0].grid(True, alpha=0.3)

# Show shock interface
shock_x = 0.5
axs[1, 1].axvline(x=shock_x, color='k', linestyle='--', alpha=0.7, label='Shock interface')
axs[1, 1].plot(grid.x, rho, 'b-', linewidth=2, label='Density')
axs[1, 1].set_title('Density with Shock Interface')
axs[1, 1].set_xlabel('x')
axs[1, 1].set_ylabel('ρ')
axs[1, 1].legend()
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('debug_shu_osher_initial.png', dpi=150, bbox_inches='tight')
plt.close()

print("Debug plot saved as 'debug_shu_osher_initial.png'")
