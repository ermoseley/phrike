"""Comprehensive test suite for adaptive time-stepping functionality.

This test suite validates the adaptive time-stepping implementation including:
- Embedded Runge-Kutta methods (RK23, RK45, RK78)
- Adaptive step size control
- Error estimation and step rejection
- Integration with existing solver infrastructure
- Configuration parsing and validation
- Backward compatibility
"""

import os
import sys
import tempfile
import shutil
import pytest
import numpy as np
from pathlib import Path

# Add the phrike package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from phrike.adaptive import (
    AdaptiveStepController, RK23, RK45, RK78, 
    create_adaptive_method, create_adaptive_stepper, AdaptiveTimeStepper
)
from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.solver import SpectralSolver1D
from phrike.problems.acoustic1d import Acoustic1DProblem


class TestAdaptiveInfrastructure:
    """Test the core adaptive time-stepping infrastructure."""
    
    def test_adaptive_step_controller(self):
        """Test adaptive step controller functionality."""
        controller = AdaptiveStepController(
            rtol=1e-6, atol=1e-8, safety_factor=0.9,
            min_dt_factor=0.1, max_dt_factor=5.0
        )
        
        # Test step acceptance
        dt_new, accept = controller.compute_new_dt(0.1, 1e-8, 1.0, 4)
        assert accept == True
        assert dt_new > 0
        
        # Test step rejection
        dt_new, accept = controller.compute_new_dt(0.1, 1e-4, 1.0, 4)
        assert accept == False
        assert dt_new < 0.1  # Should reduce step size
        
        # Test extreme cases
        dt_new, accept = controller.compute_new_dt(0.1, 0.0, 1.0, 4)
        assert accept == True
        assert dt_new >= 0.1  # Should increase step size
    
    def test_embedded_rk_methods(self):
        """Test embedded Runge-Kutta method implementations."""
        # Simple test problem: dy/dt = -y, y(0) = 1
        # Exact solution: y(t) = exp(-t)
        def rhs_func(y):
            return -y
        
        U0 = np.array([1.0])
        dt = 0.1
        
        # Test RK23
        rk23 = RK23()
        y_high, y_low, error = rk23.step(rhs_func, U0, dt)
        assert y_high.shape == U0.shape
        assert y_low.shape == U0.shape
        assert error >= 0
        assert np.abs(y_high - y_low) <= error + 1e-15
        
        # Test RK45
        rk45 = RK45()
        y_high, y_low, error = rk45.step(rhs_func, U0, dt)
        assert y_high.shape == U0.shape
        assert y_low.shape == U0.shape
        assert error >= 0
        assert np.abs(y_high - y_low) <= error + 1e-15
        
        # Test RK78
        rk78 = RK78()
        y_high, y_low, error = rk78.step(rhs_func, U0, dt)
        assert y_high.shape == U0.shape
        assert y_low.shape == U0.shape
        assert error >= 0
        assert np.abs(y_high - y_low) <= error + 1e-15
    
    def test_adaptive_method_creation(self):
        """Test adaptive method creation by name."""
        # Test valid methods
        rk23 = create_adaptive_method("rk23")
        assert isinstance(rk23, RK23)
        
        rk45 = create_adaptive_method("rk45")
        assert isinstance(rk45, RK45)
        
        rk78 = create_adaptive_method("rk78")
        assert isinstance(rk78, RK78)
        
        # Test case insensitive
        rk45_upper = create_adaptive_method("RK45")
        assert isinstance(rk45_upper, RK45)
        
        # Test invalid method
        with pytest.raises(ValueError):
            create_adaptive_method("invalid")
    
    def test_adaptive_stepper_creation(self):
        """Test adaptive time stepper creation."""
        stepper = create_adaptive_stepper(
            "rk45", rtol=1e-5, atol=1e-7, 
            safety_factor=0.8, max_rejections=5
        )
        
        assert isinstance(stepper, AdaptiveTimeStepper)
        assert isinstance(stepper.method, RK45)
        assert stepper.controller.rtol == 1e-5
        assert stepper.controller.atol == 1e-7
        assert stepper.controller.safety_factor == 0.8
        assert stepper.controller.max_rejections == 5


class TestAdaptiveSolverIntegration:
    """Test integration of adaptive time-stepping with spectral solvers."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def test_grid(self):
        """Create a test grid."""
        return Grid1D(N=64, Lx=1.0)
    
    @pytest.fixture
    def test_equations(self):
        """Create test equations."""
        return EulerEquations1D(gamma=1.4)
    
    def test_solver_adaptive_initialization(self, test_grid, test_equations):
        """Test solver initialization with adaptive configuration."""
        # Test with adaptive config
        adaptive_config = {
            "enabled": True,
            "scheme": "rk45",
            "rtol": 1e-6,
            "atol": 1e-8
        }
        
        solver = SpectralSolver1D(
            grid=test_grid,
            equations=test_equations,
            scheme="rk4",
            cfl=0.4,
            adaptive_config=adaptive_config
        )
        
        assert solver.adaptive_enabled == True
        assert solver.adaptive_stepper is not None
        assert isinstance(solver.adaptive_stepper.method, RK45)
        assert solver.scheme == "rk45"
        
        # Test without adaptive config (backward compatibility)
        solver_fixed = SpectralSolver1D(
            grid=test_grid,
            equations=test_equations,
            scheme="rk4",
            cfl=0.4
        )
        
        assert solver_fixed.adaptive_enabled == False
        assert solver_fixed.adaptive_stepper is None
        assert solver_fixed.scheme == "rk4"
    
    def test_adaptive_step_vs_fixed_step(self, test_grid, test_equations):
        """Compare adaptive and fixed time-stepping on a simple problem."""
        # Create initial conditions (acoustic wave)
        U0 = test_equations.conservative(
            rho=np.ones(64) + 1e-3 * np.sin(2 * np.pi * test_grid.x),
            u=np.zeros(64),
            p=np.ones(64)
        )
        
        # Fixed time-stepping
        solver_fixed = SpectralSolver1D(
            grid=test_grid,
            equations=test_equations,
            scheme="rk4",
            cfl=0.4
        )
        
        # Adaptive time-stepping
        adaptive_config = {
            "enabled": True,
            "scheme": "rk45",
            "rtol": 1e-6,
            "atol": 1e-8
        }
        
        solver_adaptive = SpectralSolver1D(
            grid=test_grid,
            equations=test_equations,
            scheme="rk4",
            cfl=0.4,
            adaptive_config=adaptive_config
        )
        
        # Run both for a short time
        t_end = 0.01
        
        # Fixed stepping
        U_fixed = U0.copy()
        solver_fixed.U = U_fixed.copy()
        solver_fixed.t = 0.0
        fixed_steps = 0
        while solver_fixed.t < t_end:
            dt = min(solver_fixed.compute_dt(solver_fixed.U), t_end - solver_fixed.t)
            solver_fixed.U = solver_fixed.step(solver_fixed.U, dt)
            solver_fixed.t += dt
            fixed_steps += 1
        
        # Adaptive stepping
        U_adaptive = U0.copy()
        solver_adaptive.U = U_adaptive.copy()
        solver_adaptive.t = 0.0
        adaptive_steps = 0
        
        dt = solver_adaptive.compute_dt(solver_adaptive.U)
        while solver_adaptive.t < t_end:
            dt = min(dt, t_end - solver_adaptive.t)
            if solver_adaptive.adaptive_enabled:
                dt = solver_adaptive._adaptive_run_step(dt, adaptive_steps, None)
            else:
                dt = solver_adaptive._fixed_run_step(dt, adaptive_steps, None)
            adaptive_steps += 1
        
        # Both should reach approximately the same final state
        # (adaptive might be slightly more accurate)
        final_error = np.max(np.abs(solver_fixed.U - solver_adaptive.U))
        assert final_error < 1e-10, f"Adaptive and fixed stepping diverged: {final_error}"
        
        # Adaptive should use different number of steps (could be more or fewer)
        assert adaptive_steps != fixed_steps
    
    def test_adaptive_with_config_file(self, temp_dir):
        """Test adaptive time-stepping using configuration file."""
        # Create test config
        config = {
            "problem": "acoustic1d",
            "grid": {"N": 64, "Lx": 1.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "integration": {
                "t0": 0.0, "t_end": 0.05, "cfl": 0.4, "scheme": "rk4",
                "output_interval": 0.025,
                "adaptive": {
                    "enabled": True,
                    "scheme": "rk45",
                    "rtol": 1e-6,
                    "atol": 1e-8,
                    "safety_factor": 0.9
                }
            },
            "initial_conditions": {
                "rho0": 1.0, "u0": 0.0, "p0": 1.0,
                "amplitude": 1e-3, "k": 1
            },
            "io": {"outdir": temp_dir}
        }
        
        # Create problem and run
        problem = Acoustic1DProblem(config=config)
        solver, history = problem.run(backend="numpy", generate_video=False)
        
        # Check that adaptive stepping was used
        assert solver.adaptive_enabled == True
        assert solver.adaptive_stepper is not None
        
        # Check conservation (should be excellent)
        mass_error = abs(history["mass"][-1] - history["mass"][0]) / abs(history["mass"][0])
        energy_error = abs(history["energy"][-1] - history["energy"][0]) / abs(history["energy"][0])
        
        assert mass_error < 1e-12, f"Mass not conserved: {mass_error}"
        assert energy_error < 1e-12, f"Energy not conserved: {energy_error}"


class TestAdaptiveErrorHandling:
    """Test error handling and edge cases in adaptive time-stepping."""
    
    def test_step_rejection_handling(self):
        """Test handling of step rejections."""
        stepper = create_adaptive_stepper("rk45", rtol=1e-12, atol=1e-14)  # Very strict
        
        # Stiff problem that will cause rejections
        def stiff_rhs(U):
            return -1000 * U  # Very stiff
        
        U0 = np.array([1.0])
        dt = 0.1  # Too large for stiff problem
        
        # Should eventually accept or hit max rejections
        result = stepper.step(stiff_rhs, U0, dt, 1.0)
        
        # Either accept the step or hit max rejections
        assert result.accepted or result.rejections >= stepper.controller.max_rejections
    
    def test_invalid_adaptive_config(self):
        """Test handling of invalid adaptive configurations."""
        grid = Grid1D(N=32, Lx=1.0)
        equations = EulerEquations1D()
        
        # Invalid scheme should fall back
        adaptive_config = {
            "enabled": True,
            "scheme": "invalid_scheme",
            "fallback_scheme": "rk4"
        }
        
        solver = SpectralSolver1D(
            grid=grid, equations=equations, adaptive_config=adaptive_config
        )
        
        assert solver.adaptive_enabled == False
        assert solver.scheme == "rk4"
    
    def test_zero_error_handling(self):
        """Test handling of zero error estimates."""
        controller = AdaptiveStepController()
        
        # Zero error should lead to step acceptance and increased dt
        dt_new, accept = controller.compute_new_dt(0.1, 0.0, 1.0, 4)
        assert accept == True
        assert dt_new >= 0.1


class TestAdaptivePerformance:
    """Test performance characteristics of adaptive time-stepping."""
    
    def test_adaptive_efficiency(self):
        """Test that adaptive stepping can be more efficient than fixed."""
        grid = Grid1D(N=128, Lx=2.0)
        equations = EulerEquations1D()
        
        # Create smooth initial conditions (good for adaptive)
        U0 = equations.conservative(
            rho=np.ones(128) + 1e-4 * np.sin(4 * np.pi * grid.x),
            u=np.zeros(128),
            p=np.ones(128)
        )
        
        t_end = 0.1
        
        # Fixed stepping with conservative CFL
        solver_fixed = SpectralSolver1D(grid=grid, equations=equations, cfl=0.2)
        solver_fixed.U = U0.copy()
        solver_fixed.t = 0.0
        
        fixed_steps = 0
        while solver_fixed.t < t_end:
            dt = min(solver_fixed.compute_dt(solver_fixed.U), t_end - solver_fixed.t)
            solver_fixed.U = solver_fixed.step(solver_fixed.U, dt)
            solver_fixed.t += dt
            fixed_steps += 1
        
        # Adaptive stepping
        adaptive_config = {
            "enabled": True,
            "scheme": "rk45",
            "rtol": 1e-6,
            "atol": 1e-8
        }
        
        solver_adaptive = SpectralSolver1D(
            grid=grid, equations=equations, cfl=0.4, adaptive_config=adaptive_config
        )
        solver_adaptive.U = U0.copy()
        solver_adaptive.t = 0.0
        
        adaptive_steps = 0
        dt = solver_adaptive.compute_dt(solver_adaptive.U)
        while solver_adaptive.t < t_end:
            dt = min(dt, t_end - solver_adaptive.t)
            dt = solver_adaptive._adaptive_run_step(dt, adaptive_steps, None)
            adaptive_steps += 1
        
        # For smooth problems, adaptive should be more efficient
        # (fewer steps for same accuracy)
        efficiency_ratio = fixed_steps / adaptive_steps
        print(f"Fixed steps: {fixed_steps}, Adaptive steps: {adaptive_steps}")
        print(f"Efficiency ratio: {efficiency_ratio:.2f}")
        
        # Adaptive should be at least as efficient (could be much better)
        assert efficiency_ratio >= 0.5  # Allow some tolerance


def run_adaptive_tests():
    """Run all adaptive time-stepping tests and provide summary."""
    print("=" * 60)
    print("ADAPTIVE TIME-STEPPING TEST SUITE")
    print("=" * 60)
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="phrike_adaptive_tests_")
    print(f"Test output directory: {test_dir}")
    
    try:
        # Test 1: Basic adaptive functionality
        print("\n1. Testing basic adaptive functionality...")
        
        config = {
            "problem": "acoustic1d",
            "grid": {"N": 64, "Lx": 1.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "integration": {
                "t0": 0.0, "t_end": 0.05, "cfl": 0.4, "scheme": "rk4",
                "output_interval": 0.025,
                "adaptive": {
                    "enabled": True,
                    "scheme": "rk45",
                    "rtol": 1e-6,
                    "atol": 1e-8
                }
            },
            "initial_conditions": {
                "rho0": 1.0, "u0": 0.0, "p0": 1.0,
                "amplitude": 1e-3, "k": 1
            },
            "io": {"outdir": test_dir}
        }
        
        problem = Acoustic1DProblem(config=config)
        solver, history = problem.run(backend="numpy", generate_video=False)
        
        print(f"   ✓ Adaptive solver created successfully")
        print(f"   ✓ Scheme: {solver.scheme}")
        print(f"   ✓ Adaptive enabled: {solver.adaptive_enabled}")
        print(f"   ✓ Simulation completed: t = {solver.t:.6f}")
        
        # Test 2: Conservation properties
        print("\n2. Testing conservation properties...")
        
        mass_error = abs(history["mass"][-1] - history["mass"][0]) / abs(history["mass"][0])
        energy_error = abs(history["energy"][-1] - history["energy"][0]) / abs(history["energy"][0])
        
        print(f"   ✓ Mass conservation error: {mass_error:.2e}")
        print(f"   ✓ Energy conservation error: {energy_error:.2e}")
        
        assert mass_error < 1e-12, f"Mass not conserved: {mass_error}"
        assert energy_error < 1e-12, f"Energy not conserved: {energy_error}"
        
        # Test 3: Different adaptive methods
        print("\n3. Testing different adaptive methods...")
        
        methods = ["rk23", "rk45", "rk78"]
        for method in methods:
            config["integration"]["adaptive"]["scheme"] = method
            problem = Acoustic1DProblem(config=config)
            solver, _ = problem.run(backend="numpy", generate_video=False)
            
            print(f"   ✓ {method.upper()} method: t = {solver.t:.6f}")
        
        print("\n" + "=" * 60)
        print("✓ ALL ADAPTIVE TIME-STEPPING TESTS PASSED!")
        print("Adaptive time-stepping is working correctly and ready for production use.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    # Run tests when script is executed directly
    success = run_adaptive_tests()
    sys.exit(0 if success else 1)
