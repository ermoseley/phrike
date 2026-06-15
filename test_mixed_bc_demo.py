#!/usr/bin/env python3
"""Demonstration script for mixed boundary conditions with Legendre basis."""

import phrike
from phrike.problems import ProblemRegistry

def test_bc_configuration(config_name, description):
    """Test a specific boundary condition configuration."""
    print(f"\n=== {description} ===")
    
    problem = ProblemRegistry.create_problem('sod', config_path=f'configs/{config_name}.yaml')
    grid = problem.create_grid()
    
    print(f"Configuration: {config_name}")
    print(f"BC Global: {grid._bc_global}")
    print(f"BC Map: {grid._bc_map}")
    print(f"BC Boundary Map: {grid._bc_boundary_map}")
    
    # Run short simulation
    solver, history = problem.run(backend='numpy', generate_video=False)
    print(f"Simulation completed successfully! Final time: {solver.t:.3f}")

def create_test_configs():
    """Create various test configurations to demonstrate mixed BC capabilities."""
    
    # Test 1: Simple mixed BC (left: dirichlet, right: neumann)
    config1 = """# Simple mixed BC test
problem: sod
grid:
  N: 64
  Lx: 1.0
  x0: 0.5
  basis: legendre
  bc:
    left: dirichlet
    right: neumann
  dealias: true
  precision: double
physics:
  gamma: 1.4
integration:
  t0: 0.0
  t_end: 0.05
  cfl: 0.4
  scheme: rk4
  output_interval: 0.01
  checkpoint_interval: 0.0
  spectral_filter:
    enabled: true
    p: 8
    alpha: 36.0
initial_conditions:
  left:
    rho: 1.0
    u: 0.0
    p: 1.0
  right:
    rho: 0.125
    u: 0.0
    p: 0.1
io:
  outdir: outputs/test_simple_mixed
  save_spectra: false
monitoring:
  enabled: true
  step_interval: 25
  include_conservation: true
  include_timestep: true
  include_time: true
"""
    
    # Test 2: Per-variable mixed BC
    config2 = """# Per-variable mixed BC test
problem: sod
grid:
  N: 64
  Lx: 1.0
  x0: 0.5
  basis: legendre
  bc:
    left:
      density: neumann
      momentum: dirichlet
      pressure: neumann
    right:
      density: neumann
      momentum: neumann
      pressure: neumann
  dealias: true
  precision: double
physics:
  gamma: 1.4
integration:
  t0: 0.0
  t_end: 0.05
  cfl: 0.4
  scheme: rk4
  output_interval: 0.01
  checkpoint_interval: 0.0
  spectral_filter:
    enabled: true
    p: 8
    alpha: 36.0
initial_conditions:
  left:
    rho: 1.0
    u: 0.0
    p: 1.0
  right:
    rho: 0.125
    u: 0.0
    p: 0.1
io:
  outdir: outputs/test_per_variable_mixed
  save_spectra: false
monitoring:
  enabled: true
  step_interval: 25
  include_conservation: true
  include_timestep: true
  include_time: true
"""
    
    # Test 3: Edge case with aliases
    config3 = """# Edge case with aliases
problem: sod
grid:
  N: 64
  Lx: 1.0
  x0: 0.5
  basis: legendre
  bc:
    left: wall  # Should apply to all variables
    right:
      density: open  # Alias for neumann
      momentum: dirichlet  # Should become wall for momentum
      pressure: neumann
  dealias: true
  precision: double
physics:
  gamma: 1.4
integration:
  t0: 0.0
  t_end: 0.05
  cfl: 0.4
  scheme: rk4
  output_interval: 0.01
  checkpoint_interval: 0.0
  spectral_filter:
    enabled: true
    p: 8
    alpha: 36.0
initial_conditions:
  left:
    rho: 1.0
    u: 0.0
    p: 1.0
  right:
    rho: 0.125
    u: 0.0
    p: 0.1
io:
  outdir: outputs/test_edge_cases
  save_spectra: false
monitoring:
  enabled: true
  step_interval: 25
  include_conservation: true
  include_timestep: true
  include_time: true
"""
    
    # Write test configurations
    with open('configs/test_simple_mixed.yaml', 'w') as f:
        f.write(config1)
    with open('configs/test_per_variable_mixed.yaml', 'w') as f:
        f.write(config2)
    with open('configs/test_edge_cases.yaml', 'w') as f:
        f.write(config3)

if __name__ == "__main__":
    print("Mixed Boundary Conditions Demo")
    print("=" * 50)
    
    # Create test configurations
    create_test_configs()
    
    # Test various configurations
    test_bc_configuration("test_simple_mixed", "Simple Mixed BC (left: dirichlet, right: neumann)")
    test_bc_configuration("test_per_variable_mixed", "Per-Variable Mixed BC")
    test_bc_configuration("test_edge_cases", "Edge Cases with Aliases")
    test_bc_configuration("sod_mixed_bc_example", "Comprehensive Example")
    
    print("\n" + "=" * 50)
    print("All mixed BC configurations work correctly!")
    print("\nSupported BC types:")
    print("- dirichlet/wall/reflective: u=0, zero-gradient for scalars")
    print("- neumann/open: zero-gradient for all variables")
    print("\nConfiguration formats:")
    print("1. Global: bc: 'dirichlet'")
    print("2. Per-variable: bc: {density: 'neumann', momentum: 'dirichlet'}")
    print("3. Per-boundary: bc: {left: 'dirichlet', right: 'neumann'}")
    print("4. Per-boundary per-variable: bc: {left: {density: 'neumann', momentum: 'dirichlet'}}")
    
    # Clean up test files
    import os
    test_files = ['configs/test_simple_mixed.yaml', 'configs/test_per_variable_mixed.yaml', 'configs/test_edge_cases.yaml']
    for f in test_files:
        if os.path.exists(f):
            os.remove(f)
    print("\nTest files cleaned up.")
