#!/usr/bin/env python3
"""Visualize a saved turb3d snapshot (density slice + column density)."""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from phrike.io import load_config


def _column_density(rho, axis="z", thickness=1.0):
    """Project rho along axis over central fraction of domain. rho: (Nz, Ny, Nx)."""
    nz, ny, nx = rho.shape
    if axis == "z":
        z_start = int(nz * (0.5 - thickness / 2))
        z_end = int(nz * (0.5 + thickness / 2))
        z_start, z_end = max(0, z_start), min(nz, z_end)
        n_el = z_end - z_start or 1
        return np.sum(rho[z_start:z_end, :, :], axis=0) / n_el
    if axis == "y":
        y_start = int(ny * (0.5 - thickness / 2))
        y_end = int(ny * (0.5 + thickness / 2))
        y_start, y_end = max(0, y_start), min(ny, y_end)
        n_el = y_end - y_start or 1
        return np.sum(rho[:, y_start:y_end, :], axis=1) / n_el
    if axis == "x":
        x_start = int(nx * (0.5 - thickness / 2))
        x_end = int(nx * (0.5 + thickness / 2))
        x_start, x_end = max(0, x_start), min(nx, x_end)
        n_el = x_end - x_start or 1
        return np.sum(rho[:, :, x_start:x_end], axis=2) / n_el
    raise ValueError(f"axis must be x, y, or z, got {axis}")


def _mid_slice(rho, axis="z", position=0.5):
    """Mid-plane slice. rho: (Nz, Ny, Nx)."""
    nz, ny, nx = rho.shape
    if axis == "z":
        idx = int(nz * position)
        idx = max(0, min(nz - 1, idx))
        return rho[idx, :, :], (0, 1, 0, 1)  # extent (Lx, Ly) set by caller
    if axis == "y":
        idx = int(ny * position)
        idx = max(0, min(ny - 1, idx))
        return rho[:, idx, :], (0, 1, 0, 1)
    if axis == "x":
        idx = int(nx * position)
        idx = max(0, min(nx - 1, idx))
        return rho[:, :, idx], (0, 1, 0, 1)
    raise ValueError(f"axis must be x, y, or z, got {axis}")


def main():
    p = argparse.ArgumentParser(description="Visualize turb3d snapshot (density).")
    p.add_argument(
        "snapshot",
        nargs="?",
        default="outputs/turb3d/snapshot_t2.000000.npz",
        help="Path to snapshot .npz",
    )
    p.add_argument(
        "--config",
        default="configs/turb3d.yaml",
        help="Path to problem config (for video/plot settings)",
    )
    p.add_argument(
        "-o", "--output",
        default=None,
        help="Output image path (default: same dir as snapshot, name turb3d_visualization.png)",
    )
    args = p.parse_args()

    if not os.path.isfile(args.snapshot):
        print(f"Snapshot not found: {args.snapshot}", file=sys.stderr)
        sys.exit(1)

    data = np.load(args.snapshot, allow_pickle=True)
    rho = np.asarray(data["rho"])
    t = float(data["t"])
    meta = data["meta"].item() if "meta" in data else {}
    Lx = float(meta.get("Lx", 1.0))
    Ly = float(meta.get("Ly", 1.0))
    Lz = float(meta.get("Lz", 1.0))

    video_config = {}
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
        video_config = cfg.get("video", {})

    projection_type = video_config.get("projection_type", "column_density")
    column_axis = video_config.get("column_axis", "z")
    column_thickness = float(video_config.get("column_thickness", 1.0))
    slice_axis = video_config.get("slice_axis", "z")
    slice_position = float(video_config.get("slice_position", 0.5))
    scale = float(video_config.get("scale", 2.0))
    frame_dpi = int(video_config.get("frame_dpi", 150))
    colorbar_scale = video_config.get("colorbar_scale", "log")
    colorbar_min = video_config.get("colorbar_min", 0.5)
    colorbar_max = video_config.get("colorbar_max", 2.0)
    colorbar_fixed = video_config.get("colorbar_fixed", True)

    norm = None
    if colorbar_scale == "log":
        from matplotlib.colors import LogNorm
        vmin = colorbar_min if colorbar_fixed else None
        vmax = colorbar_max if colorbar_fixed else None
        norm = LogNorm(vmin=vmin, vmax=vmax)

    base = 8
    fig, axes = plt.subplots(1, 2, figsize=(base * scale * 2, base * scale), constrained_layout=True)

    # Left: column density
    proj = _column_density(rho, axis=column_axis, thickness=column_thickness)
    if column_axis == "z":
        extent = (0, Lx, 0, Ly)
        xlabel, ylabel = "x", "y"
    elif column_axis == "y":
        extent = (0, Lx, 0, Lz)
        xlabel, ylabel = "x", "z"
    else:
        extent = (0, Ly, 0, Lz)
        xlabel, ylabel = "y", "z"
    axes[0].imshow(proj, origin="lower", extent=extent, aspect="equal", norm=norm)
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(f"Column density (∫{column_axis}) t={t:.3f}")

    # Right: mid-plane slice
    slc, _ = _mid_slice(rho, axis=slice_axis, position=slice_position)
    if slice_axis == "z":
        extent = (0, Lx, 0, Ly)
        xlabel, ylabel = "x", "y"
    elif slice_axis == "y":
        extent = (0, Lx, 0, Lz)
        xlabel, ylabel = "x", "z"
    else:
        extent = (0, Ly, 0, Lz)
        xlabel, ylabel = "y", "z"
    axes[1].imshow(slc, origin="lower", extent=extent, aspect="equal", norm=norm)
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title(f"Slice ({slice_axis}=mid) t={t:.3f}")

    for ax in axes:
        plt.colorbar(ax.images[0], ax=ax, shrink=0.8)

    if args.output is None:
        outdir = os.path.dirname(args.snapshot)
        args.output = os.path.join(outdir, "turb3d_visualization.png")
    fig.savefig(args.output, dpi=frame_dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
