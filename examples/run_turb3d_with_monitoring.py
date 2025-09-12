#!/usr/bin/env python3
"""Example script showing how to use monitoring functionality with turb3d simulation."""

import argparse
import os

from hydra.problems.turb3d import Turb3DProblem


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D turbulent simulation with monitoring")
    parser.add_argument("--config", type=str, default="configs/turb3d.yaml", help="Path to YAML config")
    parser.add_argument("--monitor-steps", type=int, default=10, help="Monitor every N steps")
    parser.add_argument("--log-file", type=str, default=None, help="Log file (default: stdout)")
    args = parser.parse_args()

    # Load the problem
    problem = Turb3DProblem(config_path=args.config)
    
    # Override monitoring settings if provided
    if args.monitor_steps:
        problem.monitoring_step_interval = args.monitor_steps
    if args.log_file:
        problem.monitoring_output_file = args.log_file
        problem.monitoring_enabled = True
    
    print(f"Running turbulent simulation with monitoring every {problem.monitoring_step_interval} steps")
    if problem.monitoring_output_file:
        print(f"Monitoring output will be written to: {os.path.join(problem.outdir, problem.monitoring_output_file)}")
    else:
        print("Monitoring output will be printed to stdout")
    
    # Run the simulation
    solver, history = problem.run(backend="numpy", generate_video=True)
    
    print(f"\nSimulation completed!")
    print(f"Final time: {solver.t:.6f}")
    print(f"Total output steps: {len(history['time'])}")
    
    if problem.monitoring_output_file:
        log_path = os.path.join(problem.outdir, problem.monitoring_output_file)
        if os.path.exists(log_path):
            print(f"Monitoring log saved to: {log_path}")


if __name__ == "__main__":
    main()
