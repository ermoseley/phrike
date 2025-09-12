#!/usr/bin/env python3
"""Simple usage examples for the new SpectralHydro CLI and API."""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from hydra import run_simulation


def example_programmatic():
    """Example using the programmatic API."""
    print("Running Sod shock tube programmatically...")
    
    config = {
        "problem": "sod",
        "grid": {
            "N": 512,
            "Lx": 1.0,
            "x0": 0.5
        },
        "physics": {
            "gamma": 1.4
        },
        "integration": {
            "t0": 0.0,
            "t_end": 0.2,
            "cfl": 0.4,
            "scheme": "rk4",
            "output_interval": 0.02
        },
        "initial_conditions": {
            "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
            "right": {"rho": 0.125, "u": 0.0, "p": 0.1}
        },
        "io": {
            "outdir": "outputs/simple_sod"
        }
    }
    
    solver, history = run_simulation(
        problem_name="sod",
        config=config,
        generate_video=False
    )
    
    print(f"Simulation completed at t={solver.t:.6f}")
    print(f"Results saved to: {solver.grid.outdir if hasattr(solver.grid, 'outdir') else 'outputs/simple_sod'}")


def example_config_file():
    """Example using existing config files."""
    print("Running Sod from config file...")
    
    config_path = "configs/sod.yaml"
    if not os.path.exists(config_path):
        print(f"Config file {config_path} not found, skipping this example")
        return
    
    solver, history = run_simulation(
        problem_name="sod",
        config_path=config_path,
        backend="numpy",
        generate_video=False
    )
    
    print(f"Simulation completed at t={solver.t:.6f}")


if __name__ == "__main__":
    print("Hydra Simple Usage Examples")
    print("=" * 40)
    
    try:
        example_programmatic()
        print()
        example_config_file()
        
        print("\n" + "=" * 40)
        print("Examples completed successfully!")
        print("\nYou can also use the CLI:")
        print("  python -m hydra sod --config configs/sod.yaml")
        print("  python -m hydra khi2d --backend torch --device mps")
        print("  python -m hydra tgv3d --no-video")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        import traceback
        traceback.print_exc()
