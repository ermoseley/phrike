#!/usr/bin/env python3
"""Benchmark tracer particle cost vs PDE step. Identifies bottlenecks."""

import time
import argparse

import numpy as np

from phrike.grid import Grid3D
from phrike.equations import EulerEquations3D
from phrike.solver import SpectralSolver3D
from phrike.tracers import FourierTracers3D


def _uniform_positions_3d(n, Lx=1.0, Ly=1.0, Lz=1.0):
    x = np.linspace(0.0, Lx, n, endpoint=False)
    y = np.linspace(0.0, Ly, n, endpoint=False)
    z = np.linspace(0.0, Lz, n, endpoint=False)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    return xx.ravel(), yy.ravel(), zz.ravel()


def run_benchmark(
    backend="numpy",
    device=None,
    grid_n=64,
    num_steps=20,
    particle_counts=(0, 512, 8**3, 16**3, 32**3),
):
    """Time PDE-only vs PDE+tracers for different particle counts."""
    Lx = Ly = Lz = 1.0
    grid = Grid3D(
        Nx=grid_n, Ny=grid_n, Nz=grid_n,
        Lx=Lx, Ly=Ly, Lz=Lz,
        dealias=True,
        backend=backend,
        torch_device=device,
    )
    equations = EulerEquations3D(gamma=1.4)

    # Build solver for PDE step
    solver = SpectralSolver3D(grid=grid, equations=equations, scheme="rk4", cfl=0.8)
    np.random.seed(42)
    U = np.random.randn(5, grid_n, grid_n, grid_n).astype(np.float64) * 0.1
    U[0] += 1.0  # density
    U[4] += 1.0  # energy ~ pressure

    if backend == "torch" and hasattr(grid, "torch_device"):
        import torch
        U = [torch.from_numpy(U[i]).to(device=grid.torch_device, dtype=grid.x.dtype) for i in range(5)]

    # U: 3D state is (5, Nz, Ny, Nx) or list of 5 arrays
    if backend == "torch":
        def copy_U():
            return [u.clone() for u in U]
    else:
        def copy_U():
            return np.asarray(U).copy()

    dt = 0.005
    results = []
    for P in particle_counts:
        tracers = None
        if P > 0:
            n = max(1, int(round(P ** (1.0 / 3.0))))
            n3 = n * n * n
            x0, y0, z0 = _uniform_positions_3d(n, Lx, Ly, Lz)
            tracers = FourierTracers3D(x0, y0, z0, mass=1.0)
            P = n3  # actual count

        solver.U = copy_U()
        solver.t = 0.0
        step_dt = 0.005

        # Warmup
        for s in range(2):
            step_dt = solver._fixed_run_step(step_dt, s, None)
            if tracers is not None:
                tracers.step(grid, equations, solver.U, step_dt)

        solver.U = copy_U()
        solver.t = 0.0
        step_dt = 0.005

        # Time: PDE step + optional tracer step
        steps = num_steps
        t0 = time.perf_counter()
        for s in range(steps):
            step_dt = solver._fixed_run_step(step_dt, s, None)
            if tracers is not None:
                tracers.step(grid, equations, solver.U, step_dt)
        elapsed = time.perf_counter() - t0

        label = f"P={P}" if P > 0 else "PDE-only (no tracers)"
        per_step_ms = 1000.0 * elapsed / steps
        results.append((P, per_step_ms, label))
        print(f"  {label}: {per_step_ms:.2f} ms/step ({elapsed:.2f}s total for {steps} steps)")

    return results


def main():
    p = argparse.ArgumentParser(description="Benchmark tracer cost")
    p.add_argument("--backend", default="numpy", choices=["numpy", "torch"])
    p.add_argument("--device", default=None, help="e.g. mps, cuda, cpu")
    p.add_argument("--grid", type=int, default=64)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--particles", type=int, nargs="+", default=[0, 512, 4096, 32768])
    args = p.parse_args()

    print(f"Backend={args.backend}, device={args.device}, grid={args.grid}^3, steps={args.steps}")
    print("Particle counts:", args.particles)
    print()

    results = run_benchmark(
        backend=args.backend,
        device=args.device,
        grid_n=args.grid,
        num_steps=args.steps,
        particle_counts=tuple(args.particles),
    )

    # Summary
    if len(results) >= 2:
        base = next(r[1] for r in results if r[0] == 0)
        print()
        print("Overhead vs PDE-only:")
        for P, ms, label in results:
            if P > 0 and base > 0:
                overhead = (ms - base) / base * 100
                print(f"  {label}: +{overhead:.0f}% ({ms - base:.2f} ms/step tracer cost)")


"""
Bottleneck summary (64^3 grid, numpy):
- PDE-only: ~375 ms/step.
- Tracer cost scales with P: ~50--350% overhead for P=512--4096.
- Dominant cost: 3x evaluate_fourier_at_points per step (3 FFTs + 3 x O(P*N^3)
  matmul for Fourier interpolation at P points). For large P, consider chunking
  the numpy path or using the batched torch path (one chunk loop, shared E).
- Batched evaluate_fourier_at_points_batched_3d is used for torch only (saves
  chunk loop and kernel launches); for numpy, three separate calls are used.
"""

if __name__ == "__main__":
    main()
