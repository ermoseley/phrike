"""Throughput benchmark for any registered problem, built from its YAML config.

Builds the grid, equations, initial state and solver the way ``BaseProblem.run``
does and times the loop body of ``solver.run`` (dt capped by the stability
limit, a step, the bookkeeping after an accepted step).  Reports Mzups per
accepted step, rejected attempts and simulated time per wall-clock second.

Examples:
    python scripts/bench_problem.py khi2d --config configs/khi2d_4k.yaml --device cuda
    python scripts/bench_problem.py khi2d --config configs/khi2d_4k.yaml --device cuda --legacy
    python scripts/bench_problem.py orszag_tang2d --config configs/orszag_tang2d.yaml --set grid.Nx=2048
"""

from __future__ import annotations

import argparse
import copy
import inspect
import time

import numpy as np
import torch
import yaml

from phrike import solver as solver_mod
from phrike.problems import ProblemRegistry


def _set(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node = node.setdefault(k, {})
    node[keys[-1]] = yaml.safe_load(value)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda", help="cuda | mps | cpu | numpy")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. grid.Nx=2048")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--legacy", action="store_true", help="reference path")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    cfg = copy.deepcopy(cfg)
    for item in args.set:
        key, value = item.split("=", 1)
        _set(cfg, key, value)
    problem = ProblemRegistry.get_problem_class(args.problem)(config=cfg)
    backend = "numpy" if args.device == "numpy" else "torch"
    device = None if args.device == "numpy" else args.device
    grid = problem.create_grid(backend, device)
    equations = problem.create_equations()
    U0 = problem.create_initial_conditions(grid)
    cls = problem.get_solver_class()
    kwargs = dict(grid=grid, equations=equations, scheme=problem.scheme, cfl=problem.cfl,
                  adaptive_config=problem.adaptive_config,
                  artificial_viscosity_config=problem.artificial_viscosity_config,
                  gravity_config=problem.gravity_config)
    if "spectral_fast_path" in inspect.signature(cls).parameters:
        kwargs["spectral_fast_path"] = not args.legacy
    solver = cls(**kwargs)

    # Initial projection as in solver.run.
    U = U0.clone() if isinstance(U0, torch.Tensor) else np.array(U0, copy=True)
    if hasattr(solver, "_project_B"):
        U = solver._project_B(U)
    U = grid.project_dealiased(U)
    fast = getattr(solver, "_fast", lambda: None)()
    if fast is not None and hasattr(fast, "seed"):
        U = fast.seed(U)
    solver.U = U
    solver_mod._begin_run(solver, 0.0, 1e9)

    def one_step(dt):
        t_before = solver.t
        dt = min(dt, solver._stability_limit(solver.U))
        if solver.adaptive_enabled:
            dt = solver._adaptive_run_step(dt, 0, None)
        else:
            dt = solver._fixed_run_step(dt, 0, None)
        return dt, solver.t > t_before

    sync = torch.cuda.synchronize if args.device == "cuda" else (lambda: None)
    dt = float("inf")
    for _ in range(args.warmup):
        dt, _ = one_step(dt)
    sync()
    accepted = attempts = 0
    t_start = solver.t
    t0 = time.perf_counter()
    while accepted < args.steps:
        dt, ok = one_step(dt)
        attempts += 1
        accepted += int(ok)
    sync()
    elapsed = time.perf_counter() - t0
    ncell = int(np.prod(solver.U.shape[1:]))
    precision = cfg.get("grid", {}).get("precision", "double")
    print(
        f"{args.problem} {'x'.join(str(n) for n in solver.U.shape[1:])} {precision} "
        f"scheme={solver.scheme} path={'legacy' if args.legacy else 'fast'} "
        f"accepted={accepted} rejected={attempts - accepted} "
        f"per-accepted-step={1e3 * elapsed / accepted:.2f} ms "
        f"throughput={ncell * accepted / elapsed / 1e6:.2f} Mzups "
        f"sim_time/s={(solver.t - t_start) / elapsed:.4g}"
    )


if __name__ == "__main__":
    main()
