"""Integration tests for complete workflows.

This test suite provides end-to-end testing of complete simulation
workflows including problem setup, solving, and output generation.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from phrike.problems.acoustic1d import Acoustic1DProblem
from phrike.problems.sod import SodProblem
from phrike.problems.khi2d import KHI2DProblem
from phrike.problems.gaussian_wave1d import GaussianWave1DProblem


class TestAcoustic1DIntegration:
    """Integration tests for 1D acoustic wave problem."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_complete_workflow(self, temp_dir):
        """Test complete acoustic wave simulation workflow."""
        config = {
            "grid": {"N": 64, "Lx": 2.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "initial_conditions": {"amplitude": 1e-3, "k": 1, "p0": 1.0, "rho0": 1.0},
            "integration": {
                "t0": 0.0, "t_end": 0.1, "cfl": 0.4, "scheme": "rk4",
                "output_interval": 0.05, "checkpoint_interval": 0.1,
                "spectral_filter": {"enabled": True, "p": 8, "alpha": 36.0}
            },
            "io": {"outdir": temp_dir, "save_spectra": True}
        }
        
        # Create and run problem
        problem = Acoustic1DProblem(config=config)
        solver, history = problem.run(backend="numpy", generate_video=False)
        
        # Check that simulation completed
        assert solver.t > 0
        assert len(history["time"]) > 0
        assert len(history["mass"]) > 0
        
        # Check conservation properties
        initial_mass = history["mass"][0]
        final_mass = history["mass"][-1]
        mass_error = abs(final_mass - initial_mass) / initial_mass
        assert mass_error < 1e-10, f"Mass conservation error: {mass_error}"
        
        # Check that output files were created
        output_files = list(Path(temp_dir).glob("*.npz"))
        assert len(output_files) > 0, "No output files were created"
        
        # Check that monitoring worked
        assert "energy" in history
        assert "momentum" in history
    
    def test_different_backends(self, temp_dir):
        """Test that both NumPy and Torch backends work."""
        config = {
            "grid": {"N": 32, "Lx": 1.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "initial_conditions": {"amplitude": 1e-3, "k": 1, "p0": 1.0, "rho0": 1.0},
            "integration": {"t0": 0.0, "t_end": 0.05, "cfl": 0.4, "scheme": "rk4"},
            "io": {"outdir": temp_dir}
        }
        
        # Test NumPy backend
        problem_numpy = Acoustic1DProblem(config=config)
        solver_numpy, history_numpy = problem_numpy.run(backend="numpy", generate_video=False)
        
        # Test Torch backend if available
        try:
            import torch
            problem_torch = Acoustic1DProblem(config=config)
            solver_torch, history_torch = problem_torch.run(backend="torch", generate_video=False)
            
            # Both should complete successfully
            assert solver_numpy.t > 0
            assert solver_torch.t > 0
            
        except ImportError:
            pytest.skip("Torch not available")
    
    def test_different_schemes(self, temp_dir):
        """Test different time integration schemes."""
        config = {
            "grid": {"N": 32, "Lx": 1.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "initial_conditions": {"amplitude": 1e-3, "k": 1, "p0": 1.0, "rho0": 1.0},
            "integration": {"t0": 0.0, "t_end": 0.05, "cfl": 0.4},
            "io": {"outdir": temp_dir}
        }
        
        for scheme in ["rk2", "rk4"]:
            config["integration"]["scheme"] = scheme
            problem = Acoustic1DProblem(config=config)
            solver, history = problem.run(backend="numpy", generate_video=False)
            
            assert solver.t > 0
            assert len(history["time"]) > 0


class TestSodIntegration:
    """Integration tests for Sod shock tube problem."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_sod_workflow(self, temp_dir):
        """Test complete Sod shock tube simulation workflow."""
        config = {
            "grid": {"N": 128, "Lx": 1.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "initial_conditions": {
                "x0": 0.5, "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "p": 0.1}
            },
            "integration": {
                "t0": 0.0, "t_end": 0.2, "cfl": 0.4, "scheme": "rk4",
                "output_interval": 0.05
            },
            "io": {"outdir": temp_dir}
        }
        
        # Create and run problem
        problem = SodProblem(config=config)
        solver, history = problem.run(backend="numpy", generate_video=False)
        
        # Check that simulation completed
        assert solver.t > 0
        assert len(history["time"]) > 0
        
        # Check that shock wave developed (density should vary significantly)
        final_density = solver.U[0]
        density_range = np.max(final_density) - np.min(final_density)
        assert density_range > 0.1, "Shock wave should create significant density variation"
        
        # Check conservation properties
        initial_mass = history["mass"][0]
        final_mass = history["mass"][-1]
        mass_error = abs(final_mass - initial_mass) / initial_mass
        assert mass_error < 1e-6, f"Mass conservation error: {mass_error}"


class TestKHI2DIntegration:
    """Integration tests for 2D Kelvin-Helmholtz instability."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_khi_workflow(self, temp_dir):
        """Test complete KHI simulation workflow."""
        config = {
            "grid": {"Nx": 64, "Ny": 64, "Lx": 1.0, "Ly": 1.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "initial_conditions": {
                "amplitude": 0.01, "k": 4, "rho0": 1.0, "p0": 1.0,
                "u0": 0.5, "v0": 0.0
            },
            "integration": {
                "t0": 0.0, "t_end": 0.1, "cfl": 0.4, "scheme": "rk4",
                "output_interval": 0.05
            },
            "io": {"outdir": temp_dir}
        }
        
        # Create and run problem
        problem = KHI2DProblem(config=config)
        solver, history = problem.run(backend="numpy", generate_video=False)
        
        # Check that simulation completed
        assert solver.t > 0
        assert len(history["time"]) > 0
        
        # Check that instability developed (velocity should vary significantly)
        final_velocity = solver.U[1]  # x-velocity
        velocity_range = np.max(final_velocity) - np.min(final_velocity)
        assert velocity_range > 0.01, "KHI should create significant velocity variation"
        
        # Check conservation properties
        initial_mass = history["mass"][0]
        final_mass = history["mass"][-1]
        mass_error = abs(final_mass - initial_mass) / initial_mass
        assert mass_error < 1e-6, f"Mass conservation error: {mass_error}"


class TestGaussianWaveIntegration:
    """Integration tests for Gaussian wave packet problem."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_stationary_wave(self, temp_dir):
        """Test stationary Gaussian wave packet."""
        config = {
            "grid": {"N": 128, "Lx": 2.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "initial_conditions": {
                "amplitude": 1e-6, "sigma": 0.1, "x0": 1.0,
                "rho0": 1.0, "p0": 1.0, "u0": 0.0
            },
            "integration": {
                "t0": 0.0, "t_end": 0.2, "cfl": 0.4, "scheme": "rk4",
                "output_interval": 0.05
            },
            "io": {"outdir": temp_dir}
        }
        
        # Create and run problem
        problem = GaussianWave1DProblem(config=config)
        solver, history = problem.run(backend="numpy", generate_video=False)
        
        # Check that simulation completed
        assert solver.t > 0
        assert len(history["time"]) > 0
        
        # Check conservation properties (should be excellent for small amplitude)
        initial_mass = history["mass"][0]
        final_mass = history["mass"][-1]
        mass_error = abs(final_mass - initial_mass) / initial_mass
        assert mass_error < 1e-12, f"Mass conservation error: {mass_error}"
    
    def test_traveling_wave(self, temp_dir):
        """Test traveling Gaussian wave packet."""
        config = {
            "grid": {"N": 128, "Lx": 2.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "initial_conditions": {
                "amplitude": 1e-6, "sigma": 0.1, "x0": 1.0,
                "rho0": 1.0, "p0": 1.0, "u0": 0.0, "traveling": True
            },
            "integration": {
                "t0": 0.0, "t_end": 0.2, "cfl": 0.4, "scheme": "rk4",
                "output_interval": 0.05
            },
            "io": {"outdir": temp_dir}
        }
        
        # Create and run problem
        problem = GaussianWave1DProblem(config=config)
        solver, history = problem.run(backend="numpy", generate_video=False)
        
        # Check that simulation completed
        assert solver.t > 0
        assert len(history["time"]) > 0
        
        # Check conservation properties
        initial_mass = history["mass"][0]
        final_mass = history["mass"][-1]
        mass_error = abs(final_mass - initial_mass) / initial_mass
        assert mass_error < 1e-12, f"Mass conservation error: {mass_error}"


class TestErrorHandling:
    """Test error handling in integration workflows."""
    
    def test_invalid_config(self):
        """Test that invalid configurations are handled gracefully."""
        # Test with invalid grid size
        config = {
            "grid": {"N": 0, "Lx": 1.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "initial_conditions": {"amplitude": 1e-3, "k": 1, "p0": 1.0, "rho0": 1.0},
            "integration": {"t0": 0.0, "t_end": 0.1, "cfl": 0.4, "scheme": "rk4"},
            "io": {"outdir": "test_output"}
        }
        
        with pytest.raises((ValueError, ZeroDivisionError)):
            problem = Acoustic1DProblem(config=config)
            problem.create_grid(backend="numpy")
    
    def test_missing_required_fields(self):
        """Test that missing required configuration fields are handled."""
        # Test with missing physics section
        config = {
            "grid": {"N": 64, "Lx": 1.0, "dealias": True},
            "initial_conditions": {"amplitude": 1e-3, "k": 1, "p0": 1.0, "rho0": 1.0},
            "integration": {"t0": 0.0, "t_end": 0.1, "cfl": 0.4, "scheme": "rk4"},
            "io": {"outdir": "test_output"}
        }
        
        with pytest.raises((KeyError, ValueError)):
            problem = Acoustic1DProblem(config=config)
            problem.run(backend="numpy", generate_video=False)
