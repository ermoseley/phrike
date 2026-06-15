#!/usr/bin/env python3
"""Analyze turb3d tracer snapshot with LagrangianPhaseSpaceSheetEasy (tetrahedral density).

Loads a saved snapshot, computes tracer density via the tetrahedral method
(get_voxel_volumes from LagrangianPhaseSpaceSheetEasy), and saves a 6-panel
visualization (slices + projections).

Usage:
  python analyze_turb3d_lpsse.py [snapshot.npz] [--config configs/turb3d.yaml]
"""

import argparse
import os
import sys

import numpy as np

# Ensure project root on path for LagrangianPhaseSpaceSheetEasy
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from phrike.io import load_checkpoint
from phrike.tracer_density import tracer_density_3d, plot_tracer_density_3d


def main():
    p = argparse.ArgumentParser(description="Analyze turb3d tracers with LPSSE tetrahedral density")
    p.add_argument(
        "snapshot",
        nargs="?",
        default=None,
        help="Path to snapshot .npz (default: latest in outputs/turb3d)",
    )
    p.add_argument(
        "--config",
        default="configs/turb3d.yaml",
        help="Config YAML (used only to resolve outdir if snapshot not given)",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output image path (default: <outdir>/tracer_density_LPSSE.png)",
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Colorbar minimum (log10 density). Default: 2%% percentile.",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Colorbar maximum (log10 density). Default: 98%% percentile.",
    )
    p.add_argument(
        "--list",
        action="store_true",
        help="List snapshots (with/without tracer data) and exit.",
    )
    args = p.parse_args()

    if args.list:
        from phrike.io import load_config
        cfg = load_config(args.config)
        outdir = cfg.get("io", {}).get("outdir", "outputs/turb3d")
        if not os.path.isdir(outdir):
            print(f"Output dir not found: {outdir}", file=sys.stderr)
            sys.exit(1)
        cands = [f for f in os.listdir(outdir) if f.startswith("snapshot_t") and f.endswith(".npz")]
        if not cands:
            print(f"No snapshot_t*.npz in {outdir}", file=sys.stderr)
            sys.exit(0)
        def has_tracers(path):
            try:
                with np.load(path, allow_pickle=True) as d:
                    return "tracer_x" in d
            except Exception:
                return False
        cands.sort(key=lambda f: float(f.replace("snapshot_t", "").replace(".npz", "")))
        print(f"Snapshots in {os.path.abspath(outdir)}:")
        for f in cands:
            path = os.path.join(outdir, f)
            has = has_tracers(path)
            t = f.replace("snapshot_t", "").replace(".npz", "")
            print(f"  t={t:>10s}  {f}  tracer_data={'yes' if has else 'no'}")
        with_t = [f for f in cands if has_tracers(os.path.join(outdir, f))]
        print(f"\nTotal: {len(cands)} snapshots, {len(with_t)} with tracer data.")
        if with_t:
            print(f"Latest with tracers: {os.path.join(outdir, with_t[-1])}")
        sys.exit(0)

    if args.snapshot is None:
        from phrike.io import load_config
        cfg = load_config(args.config)
        outdir = cfg.get("io", {}).get("outdir", "outputs/turb3d")
        cands = [f for f in os.listdir(outdir) if f.startswith("snapshot_t") and f.endswith(".npz")]
        if not cands:
            print(f"No snapshots in {outdir}", file=sys.stderr)
            sys.exit(1)
        # Prefer snapshots that contain tracer data; take latest by time
        def has_tracers(path):
            try:
                d = np.load(path, allow_pickle=True)
                ok = "tracer_x" in d
                d.close()
                return ok
            except Exception:
                return False
        with_tracers = [f for f in cands if has_tracers(os.path.join(outdir, f))]
        if not with_tracers:
            print(f"No snapshots with tracer data in {outdir}.", file=sys.stderr)
            print("  Run the simulation with tracers enabled (tracers.enabled: true in config).", file=sys.stderr)
            print("  Output goes to the config io.outdir (e.g. outputs/turb3d).", file=sys.stderr)
            sys.exit(1)
        use = with_tracers
        use.sort(key=lambda f: float(f.replace("snapshot_t", "").replace(".npz", "")))
        args.snapshot = os.path.join(outdir, use[-1])

    if not os.path.isfile(args.snapshot):
        print(f"Snapshot not found: {args.snapshot}", file=sys.stderr)
        sys.exit(1)

    data = load_checkpoint(args.snapshot)
    if "tracer_x" not in data:
        # Double-check raw npz in case load_checkpoint dropped keys
        with np.load(args.snapshot, allow_pickle=True) as raw:
            has_raw = "tracer_x" in raw
        print("No tracer data in snapshot.", file=sys.stderr)
        print(f"  Snapshot used: {os.path.abspath(args.snapshot)}", file=sys.stderr)
        if has_raw:
            print("  (File does contain 'tracer_x'; check phrike.io.load_checkpoint.)", file=sys.stderr)
        else:
            print("  This file was saved without tracers (e.g. tracers.enabled: false).", file=sys.stderr)
            print("  Run the simulation with tracers enabled, or pass a snapshot from a tracer run:", file=sys.stderr)
            outdir = os.path.dirname(args.snapshot)
            if os.path.isdir(outdir):
                cands = [f for f in os.listdir(outdir) if f.startswith("snapshot_t") and f.endswith(".npz")]
                def has_t(path):
                    try:
                        with np.load(path, allow_pickle=True) as d:
                            return "tracer_x" in d
                    except Exception:
                        return False
                with_t = [f for f in cands if has_t(os.path.join(outdir, f))]
                if with_t:
                    with_t.sort(key=lambda f: float(f.replace("snapshot_t", "").replace(".npz", "")))
                    print(f"  Snapshots with tracer data in {outdir}: ... {with_t[-1]} (latest)", file=sys.stderr)
                    print(f"  Example: python analyze_turb3d_lpsse.py {os.path.join(outdir, with_t[-1])}", file=sys.stderr)
        sys.exit(1)

    x = np.asarray(data["tracer_x"]).flatten()
    y = np.asarray(data["tracer_y"]).flatten()
    z = np.asarray(data["tracer_z"]).flatten()
    mass = np.asarray(data.get("tracer_mass", np.ones_like(x))).flatten()

    class Tracers:
        pass

    tracers = Tracers()
    tracers.x = x
    tracers.y = y
    tracers.z = z
    tracers.mass = mass

    grid_params = data.get("grid_params", {})
    Lx = float(grid_params.get("Lx", 1.0))
    Ly = float(grid_params.get("Ly", 1.0))
    Lz = float(grid_params.get("Lz", 1.0))
    domain = (Lx, Ly, Lz)
    t = float(data.get("t", 0.0))

    # Tetrahedral density (LagrangianPhaseSpaceSheetEasy get_voxel_volumes when uniform n³)
    rho_3d, _ = tracer_density_3d(
        tracers,
        domain,
        grid_shape=None,
        layout="uniform",
    )

    outpath = args.out
    if outpath is None:
        outdir = os.path.dirname(args.snapshot)
        outpath = os.path.join(outdir, "tracer_density_LPSSE.png")

    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    plot_tracer_density_3d(
        rho_3d,
        domain,
        outpath=outpath,
        log=True,
        title=f"Tracer density (LagrangianPhaseSpaceSheetEasy tetrahedral, t={t:.3f})",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    print(f"Saved: {outpath}")
    print(f"  Grid shape: {rho_3d.shape} (tetrahedral voxels)")
    print(f"  Density range: {np.nanmin(rho_3d[rho_3d > 0]):.3e} .. {np.nanmax(rho_3d):.3e}")


if __name__ == "__main__":
    main()
