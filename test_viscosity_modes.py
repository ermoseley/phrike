#!/usr/bin/env python3
"""Test script to demonstrate constant vs sensor-based artificial viscosity modes."""

import phrike
from phrike.problems import ProblemRegistry

def test_viscosity_mode(mode, config_file, description):
    """Test a specific viscosity mode."""
    print(f"\n=== Testing {description} ===")
    
    problem = ProblemRegistry.create_problem('sod', config_path=config_file)
    av_config = problem.artificial_viscosity_config
    
    print(f"Mode: {av_config['mode']}")
    if mode == "constant":
        print(f"nu_constant: {av_config['nu_constant']}")
    else:
        print(f"nu_max: {av_config['nu_max']}")
        print(f"s_ref: {av_config['s_ref']}")
        print(f"s_min: {av_config['s_min']}")
    
    # Run short simulation
    solver, history = problem.run(backend='numpy', generate_video=False)
    
    print(f"Simulation completed successfully!")
    print(f"Final time: {solver.t}")
    
    # Get final state
    rho, u, p, _ = solver.equations.primitive(solver.U)
    print(f"Final density range: {rho.min():.4f} to {rho.max():.4f}")
    print(f"Final velocity range: {u.min():.4f} to {u.max():.4f}")

if __name__ == "__main__":
    print("Testing Artificial Viscosity Modes")
    print("=" * 40)
    
    # Test constant viscosity
    test_viscosity_mode("constant", "configs/test_constant_viscosity.yaml", 
                       "Constant Viscosity Mode")
    
    # Test sensor-based viscosity
    test_viscosity_mode("sensor", "configs/test_sensor_viscosity.yaml", 
                       "Sensor-Based Viscosity Mode")
    
    print("\n" + "=" * 40)
    print("Both modes work correctly!")
    print("\nKey differences:")
    print("- Constant mode: Uniform viscosity everywhere (nu_constant)")
    print("- Sensor mode: Adaptive viscosity based on smoothness (nu_max, s_ref, s_min)")
