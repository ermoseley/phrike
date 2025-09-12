"""Tests for restart functionality."""

import os
import tempfile
import numpy as np

from hydra.grid import Grid1D
from hydra.equations import EulerEquations1D
from hydra.initial_conditions import sod_shock_tube
from hydra.solver import SpectralSolver1D
from hydra.io import save_solution_snapshot, load_checkpoint
from hydra.problems.sod import SodProblem


def test_checkpoint_save_and_load():
    """Test that checkpoints can be saved and loaded correctly."""
    # Create a simple 1D setup
    N = 64
    Lx = 1.0
    grid = Grid1D(N=N, Lx=Lx, dealias=True)
    eqs = EulerEquations1D(gamma=1.4)
    
    # Create initial conditions
    left = {"rho": 1.0, "u": 0.0, "p": 1.0}
    right = {"rho": 0.125, "u": 0.0, "p": 0.1}
    U0 = sod_shock_tube(grid.x, x0=0.5, left=left, right=right, gamma=1.4)
    
    # Create solver and run briefly
    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme="rk4", cfl=0.4)
    solver.run(U0, t0=0.0, t_end=0.01, output_interval=0.01)
    
    # Save checkpoint
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = save_solution_snapshot(temp_dir, solver.t, solver.U, grid, eqs)
        
        # Load checkpoint
        restart_data = load_checkpoint(checkpoint_path)
        
        # Verify data integrity
        assert abs(restart_data["t"] - solver.t) < 1e-10
        assert np.allclose(restart_data["U"], solver.U)
        assert "meta" in restart_data
        assert restart_data["meta"]["N"] == N
        assert restart_data["meta"]["Lx"] == Lx
        assert restart_data["meta"]["gamma"] == 1.4


def test_restart_simulation():
    """Test that a simulation can be restarted from a checkpoint."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a Sod problem
        config = {
            "problem": "sod",
            "grid": {"N": 64, "Lx": 1.0, "dealias": True},
            "physics": {"gamma": 1.4},
            "integration": {
                "t0": 0.0,
                "t_end": 0.1,
                "cfl": 0.4,
                "scheme": "rk4",
                "output_interval": 0.02,
                "checkpoint_interval": 0.05
            },
            "initial_conditions": {
                "left": {"rho": 1.0, "u": 0.0, "p": 1.0},
                "right": {"rho": 0.125, "u": 0.0, "p": 0.1}
            },
            "io": {"outdir": temp_dir}
        }
        
        # Run first part of simulation
        problem1 = SodProblem(config=config)
        solver1, history1 = problem1.run(backend="numpy", generate_video=False)
        
        # Find the checkpoint file
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith("snapshot_") and f.endswith(".npz")]
        assert len(checkpoint_files) > 0, "No checkpoint files found"
        checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
        
        # Create new problem with restart
        problem2 = SodProblem(config=config, restart_from=checkpoint_path)
        solver2, history2 = problem2.run(backend="numpy", generate_video=False)
        
        # Verify restart worked
        assert problem2.t0 > problem1.t0, "Restart time should be after initial time"
        assert problem2.restart_data is not None, "Restart data should be loaded"
        assert problem2.restart_data["t"] == problem2.t0, "Restart time should match checkpoint time"


def test_restart_validation():
    """Test that restart validation catches incompatible configurations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create checkpoint with different grid size
        N1 = 64
        Lx = 1.0
        grid1 = Grid1D(N=N1, Lx=Lx, dealias=True)
        eqs1 = EulerEquations1D(gamma=1.4)
        
        left = {"rho": 1.0, "u": 0.0, "p": 1.0}
        right = {"rho": 0.125, "u": 0.0, "p": 0.1}
        U0 = sod_shock_tube(grid1.x, x0=0.5, left=left, right=right, gamma=1.4)
        
        solver1 = SpectralSolver1D(grid=grid1, equations=eqs1, scheme="rk4", cfl=0.4)
        solver1.run(U0, t0=0.0, t_end=0.01, output_interval=0.01)
        
        checkpoint_path = save_solution_snapshot(temp_dir, solver1.t, solver1.U, grid1, eqs1)
        
        # Try to restart with different grid size
        config = {
            "problem": "sod",
            "grid": {"N": 128, "Lx": 1.0, "dealias": True},  # Different N
            "physics": {"gamma": 1.4},
            "integration": {"t0": 0.0, "t_end": 0.1, "cfl": 0.4, "scheme": "rk4"},
            "initial_conditions": {"left": {"rho": 1.0, "u": 0.0, "p": 1.0}, "right": {"rho": 0.125, "u": 0.0, "p": 0.1}},
            "io": {"outdir": temp_dir}
        }
        
        problem = SodProblem(config=config, restart_from=checkpoint_path)
        
        # This should raise an error due to grid size mismatch
        try:
            problem.run(backend="numpy", generate_video=False)
            assert False, "Expected ValueError due to grid size mismatch"
        except ValueError as e:
            assert "Grid size mismatch" in str(e)


def test_restart_cli():
    """Test that restart works through CLI."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple config
        config_path = os.path.join(temp_dir, "test_config.yaml")
        config_content = """
problem: sod
grid:
  N: 32
  Lx: 1.0
  dealias: true
physics:
  gamma: 1.4
integration:
  t0: 0.0
  t_end: 0.05
  cfl: 0.4
  scheme: rk4
  output_interval: 0.01
  checkpoint_interval: 0.02
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
  outdir: {temp_dir}
""".format(temp_dir=temp_dir)
        
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        # Run first simulation
        from hydra.problems import ProblemRegistry
        problem1 = ProblemRegistry.create_problem("sod", config_path=config_path)
        solver1, _ = problem1.run(backend="numpy", generate_video=False)
        
        # Find checkpoint
        checkpoint_files = [f for f in os.listdir(temp_dir) if f.startswith("snapshot_") and f.endswith(".npz")]
        assert len(checkpoint_files) > 0
        checkpoint_path = os.path.join(temp_dir, checkpoint_files[0])
        
        # Test restart through registry
        problem2 = ProblemRegistry.create_problem("sod", config_path=config_path, restart_from=checkpoint_path)
        assert problem2.restart_data is not None
        assert problem2.t0 > 0.0
