#!/usr/bin/env python3
"""Debug script for Shu-Osher solver."""

import numpy as np
from phrike.problems.shu_osher1d import ShuOsher1DProblem
from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.solver import SpectralSolver1D

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

# Create initial conditions
U0 = problem.create_initial_conditions(grid)
print(f"Initial U0 shape: {U0.shape}")
print(f"Initial U0 min/max: [{U0.min():.6f}, {U0.max():.6f}]")

# Create solver
solver = SpectralSolver1D(grid, equations, U0)
print(f"Solver U shape: {solver.U.shape}")
print(f"Solver U min/max: [{solver.U.min():.6f}, {solver.U.max():.6f}]")

# Initialize monitoring
print("\nInitializing monitoring...")
try:
    problem.initialize_monitoring(solver, solver.U)
    print(f"Monitoring initialized: {problem.monitoring_initial_values}")
except Exception as e:
    print(f"Error in monitoring initialization: {e}")
    import traceback
    traceback.print_exc()

# Test conservation errors computation
print("\nTesting conservation errors computation...")
try:
    cons_errors = problem.compute_conservation_errors(solver, solver.U)
    print(f"Conservation errors: {cons_errors}")
except Exception as e:
    print(f"Error in conservation computation: {e}")
    import traceback
    traceback.print_exc()

# Test primitive conversion
print("\nTesting primitive conversion...")
try:
    rho, u, p, _ = equations.primitive(solver.U)
    print(f"Primitive variables:")
    print(f"  rho: min={rho.min():.6f}, max={rho.max():.6f}, has_nan={np.any(np.isnan(rho))}")
    print(f"  u: min={u.min():.6f}, max={u.max():.6f}, has_nan={np.any(np.isnan(u))}")
    print(f"  p: min={p.min():.6f}, max={p.max():.6f}, has_nan={np.any(np.isnan(p))}")
except Exception as e:
    print(f"Error in primitive conversion: {e}")
    import traceback
    traceback.print_exc()

# Test one timestep
print("\nTesting one timestep...")
try:
    dt = 0.001
    solver.U = solver.step(solver.U, dt)
    print(f"After one step:")
    print(f"  U shape: {solver.U.shape}")
    print(f"  U min/max: [{solver.U.min():.6f}, {solver.U.max():.6f}]")
    print(f"  U has NaN: {np.any(np.isnan(solver.U))}")
    print(f"  U has inf: {np.any(np.isinf(solver.U))}")
except Exception as e:
    print(f"Error in timestep: {e}")
    import traceback
    traceback.print_exc()
