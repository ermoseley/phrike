#!/usr/bin/env python3
"""Validate periodic simplex-cell tracer density against a smooth 3-D Euler flow.

This is a periodic simplex-cell/centroid validation.  It compares the density
of deformed Lagrangian tracer cells with gas density sampled at their deformed
centroids; it is not an Eulerian voxelization diagnostic.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phrike.equations import EulerEquations3D
from phrike.grid import Grid3D
from phrike.io import save_solution_snapshot
from phrike.solver import SpectralSolver3D
from phrike.tracer_density import periodic_simplex_density_3d
from phrike.tracers import FourierTracers3D


def _as_numpy(value: Any) -> np.ndarray:
    """Return a NumPy view/copy for backend-independent diagnostics."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _uniform_lattice(n: int, length: float) -> tuple[np.ndarray, ...]:
    coordinates = np.linspace(0.0, length, n, endpoint=False)
    x, y, z = np.meshgrid(coordinates, coordinates, coordinates, indexing="ij")
    return x.ravel(), y.ravel(), z.ravel()


def _initial_state(grid: Grid3D, equations: EulerEquations3D) -> np.ndarray:
    """Return a smooth, periodic, compressive Euler state on the grid."""
    x = np.linspace(0.0, grid.Lx, grid.Nx, endpoint=False)
    y = np.linspace(0.0, grid.Ly, grid.Ny, endpoint=False)
    z = np.linspace(0.0, grid.Lz, grid.Nz, endpoint=False)
    z_mesh, y_mesh, x_mesh = np.meshgrid(z, y, x, indexing="ij")
    amplitude = 0.15
    rho = np.ones_like(x_mesh)
    ux = amplitude * np.sin(2.0 * np.pi * x_mesh / grid.Lx)
    uy = amplitude * np.sin(2.0 * np.pi * y_mesh / grid.Ly)
    uz = amplitude * np.sin(2.0 * np.pi * z_mesh / grid.Lz)
    pressure = np.ones_like(rho)
    return equations.conservative(rho, ux, uy, uz, pressure)


def _to_grid_backend(state: np.ndarray, grid: Grid3D) -> Any:
    if grid.backend != "torch":
        return state
    import torch

    return torch.as_tensor(state, dtype=grid.x.dtype, device=grid.torch_device)


def _particle_state(tracers: FourierTracers3D) -> dict[str, Any]:
    coordinate = tracers.x
    coordinate_dtype = getattr(coordinate, "dtype", None)
    if coordinate_dtype is None:
        coordinate_dtype = np.asarray(coordinate).dtype
    return {
        "count": int(_as_numpy(coordinate).size),
        "lattice_shape": [int(size) for size in tracers.lattice_shape],
        "coordinate_type": type(coordinate).__name__,
        "coordinate_dtype": str(coordinate_dtype),
        "coordinate_device": str(getattr(coordinate, "device", "cpu")),
    }


def _plot_diagnostic(
    gas_density: np.ndarray,
    tracer_density: np.ndarray,
    residual: np.ndarray,
    grid: Grid3D,
    t_final: float,
    path: Path,
) -> None:
    mid = gas_density.shape[0] // 2
    positive = np.concatenate((gas_density.ravel(), tracer_density.ravel()))
    vmin = float(np.min(positive[positive > 0.0]))
    vmax = float(np.max(positive))
    if vmin == vmax:
        vmax = vmin * 1.01
    residual_limit = float(np.max(np.abs(residual)))
    if residual_limit == 0.0:
        residual_limit = 1.0e-12

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    extent = [0.0, grid.Lx, 0.0, grid.Ly]
    density_panels = (
        (gas_density[mid], "Gas density at deformed centroids"),
        (tracer_density[mid], "Periodic simplex-cell tracer density"),
    )
    for axis, (field, title) in zip(axes[:2], density_panels):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        axis.set_ylabel("y")
        fig.colorbar(image, ax=axis, shrink=0.82, label=r"$\rho$")

    image = axes[2].imshow(
        residual[mid],
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-residual_limit,
        vmax=residual_limit,
        aspect="equal",
    )
    axes[2].set_title(r"Relative residual $(\rho_T-\rho_G)/\rho_G$")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    fig.colorbar(image, ax=axes[2], shrink=0.82)
    fig.suptitle(
        f"Periodic simplex-cell/centroid tracer validation, t={t_final:.4f}; z-cell={mid}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    if args.grid < 4:
        raise ValueError("--grid must be at least 4")
    if args.t_end <= 0.0:
        raise ValueError("--t-end must be positive")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid = Grid3D(
        Nx=args.grid,
        Ny=args.grid,
        Nz=args.grid,
        Lx=1.0,
        Ly=1.0,
        Lz=1.0,
        backend=args.backend,
        torch_device=args.device,
    )
    equations = EulerEquations3D(gamma=1.4)
    initial_state = _initial_state(grid, equations)
    initial_rho = initial_state[0]
    cell_volume = grid.dx * grid.dy * grid.dz
    gas_mass_initial = float(np.sum(initial_rho) * cell_volume)

    tracer_shape = (args.grid, args.grid, args.grid)
    tracer_count = int(np.prod(tracer_shape))
    x0, y0, z0 = _uniform_lattice(args.grid, grid.Lx)
    tracer_mass = gas_mass_initial / tracer_count
    tracers = FourierTracers3D(
        x0,
        y0,
        z0,
        mass=tracer_mass,
        lattice_shape=tracer_shape,
    )

    solver = SpectralSolver3D(grid, equations, scheme="rk4", cfl=0.25)
    accepted_steps: list[float] = []
    solver.run(
        _to_grid_backend(initial_state, grid),
        t0=0.0,
        t_end=args.t_end,
        output_interval=args.t_end,
        on_step=lambda _step, dt, _state: accepted_steps.append(float(dt)),
        tracers=tracers,
    )

    snapshot = save_solution_snapshot(
        str(outdir),
        solver.t,
        U=solver.U,
        grid=grid,
        equations=equations,
        tag="tracer_validation",
        tracers=tracers,
    )
    gas_rho = _as_numpy(equations.primitive(solver.U)[0])
    simplex_rho, centroids = periodic_simplex_density_3d(
        tracers, (grid.Lx, grid.Ly, grid.Lz), mass=tracer_mass
    )
    gas_at_centroids = _as_numpy(
        grid.interpolate_periodic_linear_at_points(
            equations.primitive(solver.U)[0],
            centroids[..., 0],
            centroids[..., 1],
            centroids[..., 2],
        )
    )
    residual = (simplex_rho - gas_at_centroids) / gas_at_centroids
    correlation = float(
        np.corrcoef(simplex_rho.ravel(), gas_at_centroids.ravel())[0, 1]
    )
    gas_mass_final = float(np.sum(gas_rho) * cell_volume)
    tracer_mass_total = float(tracer_count * tracer_mass)
    gas_mass_error = (gas_mass_final - gas_mass_initial) / gas_mass_initial
    tracer_mass_error = (tracer_mass_total - gas_mass_initial) / gas_mass_initial
    rms_residual = float(np.sqrt(np.mean(residual * residual)))
    max_residual = float(np.max(np.abs(residual)))
    particle_state = _particle_state(tracers)
    expected_device = args.device
    if args.backend in {"metal", "mps"}:
        expected_device = "mps"
    elif expected_device == "metal":
        expected_device = "mps"
    elif expected_device == "auto":
        expected_device = None
    device_matches = expected_device is None or particle_state[
        "coordinate_device"
    ].startswith(expected_device)
    checks = {
        "finite_fields": bool(
            np.all(np.isfinite(simplex_rho))
            and np.all(np.isfinite(gas_at_centroids))
            and np.all(np.isfinite(residual))
        ),
        "positive_density": bool(
            np.all(simplex_rho > 0.0) and np.all(gas_at_centroids > 0.0)
        ),
        "device_matches_request": device_matches,
        "gas_mass": abs(gas_mass_error) <= args.max_mass_relative_error,
        "tracer_mass": abs(tracer_mass_error) <= args.max_mass_relative_error,
        "correlation": correlation >= args.min_correlation,
        "rms_relative_residual": rms_residual <= args.max_rms_relative_residual,
        "max_abs_relative_residual": max_residual
        <= args.max_abs_relative_residual,
    }

    figure_path = outdir / "tracer_validation.png"
    _plot_diagnostic(
        gas_at_centroids, simplex_rho, residual, grid, solver.t, figure_path
    )
    metrics = {
        "validation": "periodic simplex-cell/centroid; not Eulerian voxelization",
        "backend": grid.backend,
        "device": grid.torch_device or "cpu",
        "precision": grid.precision,
        "mass_conservation": {
            "gas_initial": gas_mass_initial,
            "gas_final": gas_mass_final,
            "gas_relative_error": gas_mass_error,
            "tracer_total": tracer_mass_total,
            "tracer_relative_error": tracer_mass_error,
        },
        "tracer_vs_gas": {
            "mean_ratio": float(np.mean(simplex_rho) / np.mean(gas_at_centroids)),
            "correlation": correlation,
            "rms_relative_residual": rms_residual,
            "max_abs_relative_residual": max_residual,
        },
        "particle_state": particle_state,
        "acceptance": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "max_mass_relative_error": args.max_mass_relative_error,
                "min_correlation": args.min_correlation,
                "max_rms_relative_residual": args.max_rms_relative_residual,
                "max_abs_relative_residual": args.max_abs_relative_residual,
            },
        },
        "run_parameters": {
            "grid": args.grid,
            "domain": [grid.Lx, grid.Ly, grid.Lz],
            "t_end_requested": args.t_end,
            "t_final": solver.t,
            "scheme": solver.scheme,
            "cfl": solver.cfl,
            "tracer_spatial_interpolation": "dealiased Fourier evaluation",
            "tracer_time_integrator": "two-evaluation explicit midpoint",
            "tracer_position_evaluations_per_step": 2,
            "gamma": equations.gamma,
            "velocity_amplitude": 0.15,
            "tracer_mass": tracer_mass,
            "accepted_steps": len(accepted_steps),
            "accepted_step_sizes": accepted_steps,
        },
        "outputs": {
            "snapshot": str(snapshot),
            "figure": str(figure_path),
        },
    }
    metrics_path = outdir / "tracer_validation_metrics.json"
    metrics["outputs"]["metrics"] = str(metrics_path)
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        default="auto",
        help="Grid backend (auto, numpy, torch, cpu, mps, metal, or cuda)",
    )
    parser.add_argument("--device", default=None, help="Torch device override")
    parser.add_argument(
        "--grid", type=int, default=16, help="Cubic grid and tracer lattice size"
    )
    parser.add_argument("--t-end", type=float, default=0.1, help="Final simulation time")
    parser.add_argument("--max-mass-relative-error", type=float, default=1.0e-5)
    parser.add_argument("--min-correlation", type=float, default=0.99)
    parser.add_argument("--max-rms-relative-residual", type=float, default=1.0e-2)
    parser.add_argument("--max-abs-relative-residual", type=float, default=3.0e-2)
    parser.add_argument(
        "--outdir",
        default="outputs/validate_tracers",
        help="Directory for snapshot and diagnostics",
    )
    args = parser.parse_args()
    metrics = run_validation(args)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    if not metrics["acceptance"]["passed"]:
        raise SystemExit("Tracer validation failed acceptance checks")


if __name__ == "__main__":
    main()
