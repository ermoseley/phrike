#!/usr/bin/env python3
"""Debug script to find when NaN values first appear in Shu-Osher problem."""

import numpy as np
from phrike.problems.shu_osher1d import ShuOsher1DProblem
from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.solver import SpectralSolver1D

def main():
    print("=== Debug Longer Run - Finding NaN Source ===")
    
    # Create problem
    config = {
        "problem": "shu_osher1d",
        "grid": {"N": 64, "Lx": 1.0, "dealias": True},
        "physics": {"gamma": 1.4},
        "integration": {
            "t0": 0.0,
            "t_end": 0.01,  # Longer run
            "cfl": 0.4,
            "scheme": "rk4",
            "output_interval": 0.001,
            "checkpoint_interval": 0.01,
            "spectral_filter": {"enabled": True, "p": 8, "alpha": 36.0}
        },
        "io": {"outdir": "debug_outputs", "save_spectra": True}
    }
    
    problem = ShuOsher1DProblem(config=config)
    
    # Create components
    grid = problem.create_grid()
    equations = problem.create_equations()
    U0 = problem.create_initial_conditions(grid)
    
    # Create solver
    solver = SpectralSolver1D(grid=grid, equations=equations, U0=U0)
    
    print(f"Starting simulation...")
    print(f"Initial U0 has NaN: {np.isnan(U0).any()}")
    
    step_count = 0
    dt = 1e-6
    max_steps = 10000  # Limit to prevent infinite loop
    
    while step_count < max_steps and solver.t < 0.01:
        # Check for NaN values
        if np.isnan(solver.U).any():
            print(f"\n*** NaN detected at step {step_count}, t = {solver.t:.6f} ***")
            break
            
        # Check primitive variables
        try:
            rho, u, p, _ = equations.primitive(solver.U)
            if np.isnan(rho).any() or np.isnan(u).any() or np.isnan(p).any():
                print(f"\n*** NaN in primitive variables at step {step_count}, t = {solver.t:.6f} ***")
                print(f"rho has NaN: {np.isnan(rho).any()}")
                print(f"u has NaN: {np.isnan(u).any()}")
                print(f"p has NaN: {np.isnan(p).any()}")
                break
        except Exception as e:
            print(f"\n*** Error in primitive conversion at step {step_count}, t = {solver.t:.6f} ***")
            print(f"Error: {e}")
            break
        
        # Print progress every 1000 steps
        if step_count % 1000 == 0:
            try:
                rho, u, p, _ = equations.primitive(solver.U)
                print(f"Step {step_count}, t = {solver.t:.6f}, rho range: [{rho.min():.6f}, {rho.max():.6f}], u range: [{u.min():.6f}, {u.max():.6f}], p range: [{p.min():.6f}, {p.max():.6f}]")
            except Exception as e:
                print(f"Step {step_count}, t = {solver.t:.6f}, Error in primitive conversion: {e}")
                break
        
        # Take a step
        try:
            solver.U = solver.step(solver.U, dt)
            solver.t += dt
            step_count += 1
        except Exception as e:
            print(f"\n*** Error in solver step at step {step_count}, t = {solver.t:.6f} ***")
            print(f"Error: {e}")
            break
    
    print(f"\nSimulation ended at step {step_count}, t = {solver.t:.6f}")
    
    # Final check
    if np.isnan(solver.U).any():
        print("Final U contains NaN values")
    else:
        print("Final U does not contain NaN values")
        
    try:
        rho, u, p, _ = equations.primitive(solver.U)
        print(f"Final primitive variables:")
        print(f"  rho range: [{rho.min():.6f}, {rho.max():.6f}], has NaN: {np.isnan(rho).any()}")
        print(f"  u range: [{u.min():.6f}, {u.max():.6f}], has NaN: {np.isnan(u).any()}")
        print(f"  p range: [{p.min():.6f}, {p.max():.6f}], has NaN: {np.isnan(p).any()}")
    except Exception as e:
        print(f"Error in final primitive conversion: {e}")

if __name__ == "__main__":
    main()
