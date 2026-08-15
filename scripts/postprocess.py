#!/usr/bin/env python3
"""Inspect and plot saved 3-D PHRIKE snapshots.

Run this module from a source checkout, for example::

    python -m scripts.postprocess list --config configs/turb3d.yaml
    python -m scripts.postprocess plot outputs/turb3d/snapshot_t0.100000.npz
    python -m scripts.postprocess tracer-density --config configs/turb3d.yaml
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from phrike.io import load_checkpoint, load_config
from phrike.tracer_density import plot_tracer_density_3d, tracer_density_3d
from phrike.visualization import (
    central_column_mean,
    central_slice,
    projection_axis_labels,
    projection_extent,
)

if TYPE_CHECKING:
    from matplotlib.colors import LogNorm


_SNAPSHOT_RE = re.compile(r"^snapshot_t(?P<time>[-+0-9.eE]+)(?:_.+)?\.npz$")


def _snapshot_time(path: Path) -> float:
    """Extract a simulation time from a standard PHRIKE snapshot name."""
    match = _SNAPSHOT_RE.match(path.name)
    if match is None:
        raise ValueError(f"not a PHRIKE snapshot name: {path.name}")
    return float(match.group("time"))


def _snapshots(outdir: Path) -> List[Path]:
    snapshots = [path for path in outdir.glob("snapshot_t*.npz") if path.is_file()]
    return sorted(snapshots, key=lambda path: (_snapshot_time(path), path.name))


def _has_tracers(snapshot: Path) -> bool:
    try:
        with np.load(snapshot, allow_pickle=True) as data:
            return all(name in data for name in ("tracer_x", "tracer_y", "tracer_z"))
    except (OSError, ValueError):
        return False


def _config(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    config = load_config(path)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must contain a mapping: {path}")
    return config


def _configured_outdir(config_path: str, outdir: Optional[str]) -> Path:
    if outdir:
        return Path(outdir)
    config = _config(config_path)
    return Path(config.get("io", {}).get("outdir", "outputs"))


def _resolve_snapshot(
    snapshot: Optional[str], config_path: str, outdir: Optional[str], require_tracers: bool
) -> Path:
    if snapshot:
        path = Path(snapshot)
        if not path.is_file():
            raise FileNotFoundError(f"snapshot not found: {path}")
        return path

    resolved_outdir = _configured_outdir(config_path, outdir)
    snapshots = _snapshots(resolved_outdir)
    if require_tracers:
        snapshots = [path for path in snapshots if _has_tracers(path)]
    if not snapshots:
        qualifier = " with tracer data" if require_tracers else ""
        raise FileNotFoundError(f"no snapshots{qualifier} in {resolved_outdir}")
    return snapshots[-1]


def _snapshot_density(snapshot: Path) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    with np.load(snapshot, allow_pickle=True) as data:
        if "rho" not in data:
            raise ValueError(f"snapshot has no density field: {snapshot}")
        rho = np.asarray(data["rho"])
        if rho.ndim != 3:
            raise ValueError(
                f"postprocess plot requires a 3-D density field, got {rho.shape}"
            )
        t = float(data["t"])
        meta = data["meta"].item() if "meta" in data else {}
    return rho, t, meta if isinstance(meta, dict) else {}


def _domain(meta: Dict[str, Any]) -> Tuple[float, float, float]:
    domain = (
        float(meta.get("Lx", 1.0)),
        float(meta.get("Ly", 1.0)),
        float(meta.get("Lz", 1.0)),
    )
    if any(length <= 0.0 for length in domain):
        raise ValueError(f"snapshot domain lengths must be positive, got {domain}")
    return domain


def _color_norm(
    fields: Sequence[np.ndarray], scale: str, vmin: Optional[float], vmax: Optional[float]
) -> Optional[LogNorm]:
    if scale == "linear":
        return None
    if scale != "log":
        raise ValueError(f"colorbar scale must be linear or log, got {scale!r}")

    positive = [field[np.isfinite(field) & (field > 0.0)] for field in fields]
    positive = [field for field in positive if field.size]
    if not positive:
        raise ValueError("logarithmic plotting requires at least one positive value")
    values = np.concatenate(positive)
    if vmin is None:
        vmin = float(np.min(values))
    if vmax is None:
        vmax = float(np.max(values))
    if vmin <= 0.0 or vmax < vmin:
        raise ValueError(f"invalid logarithmic color limits: vmin={vmin}, vmax={vmax}")
    if vmax == vmin:
        vmax = vmin * 1.01
    from matplotlib.colors import LogNorm

    return LogNorm(vmin=vmin, vmax=vmax)


def _plot_snapshot(args: argparse.Namespace) -> int:
    import matplotlib.pyplot as plt

    snapshot = Path(args.snapshot)
    if not snapshot.is_file():
        raise FileNotFoundError(f"snapshot not found: {snapshot}")
    rho, t, meta = _snapshot_density(snapshot)
    config = _config(args.config)
    video = config.get("video", {})
    domain = _domain(meta)

    column_axis = args.column_axis or str(video.get("column_axis", "z"))
    slice_axis = args.slice_axis or str(video.get("slice_axis", "z"))
    thickness = (
        args.thickness
        if args.thickness is not None
        else float(video.get("column_thickness", 1.0))
    )
    position = (
        args.position
        if args.position is not None
        else float(video.get("slice_position", 0.5))
    )
    colorbar_scale = args.colorbar_scale or str(video.get("colorbar_scale", "linear"))
    if args.vmin is None and video.get("colorbar_fixed", False):
        vmin = video.get("colorbar_min")
    else:
        vmin = args.vmin
    if args.vmax is None and video.get("colorbar_fixed", False):
        vmax = video.get("colorbar_max")
    else:
        vmax = args.vmax

    panels = []
    if args.view in ("projection", "both"):
        panels.append(
            (
                central_column_mean(rho, axis=column_axis, thickness=thickness),
                projection_extent(domain, axis=column_axis),
                projection_axis_labels(axis=column_axis),
                f"Central density projection along {column_axis} (t={t:.3f})",
            )
        )
    if args.view in ("slice", "both"):
        panels.append(
            (
                central_slice(rho, axis=slice_axis, position=position),
                projection_extent(domain, axis=slice_axis),
                projection_axis_labels(axis=slice_axis),
                f"Density slice normal to {slice_axis} (t={t:.3f})",
            )
        )

    norm = _color_norm([panel[0] for panel in panels], colorbar_scale, vmin, vmax)
    scale = float(args.scale if args.scale is not None else video.get("scale", 1.0))
    dpi = int(args.dpi if args.dpi is not None else video.get("frame_dpi", 150))
    if scale <= 0.0 or dpi <= 0:
        raise ValueError("scale and dpi must be positive")
    figure, axes = plt.subplots(
        1, len(panels), figsize=(8.0 * scale * len(panels), 8.0 * scale), constrained_layout=True
    )
    for axis, (field, extent, labels, title) in zip(np.atleast_1d(axes), panels):
        image = axis.imshow(
            field,
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap=args.cmap,
            norm=norm,
        )
        axis.set_title(title)
        axis.set_xlabel(labels[0])
        axis.set_ylabel(labels[1])
        figure.colorbar(image, ax=axis, shrink=0.8)

    output = Path(args.output) if args.output else snapshot.with_name(
        f"{snapshot.stem}_density.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved: {output}")
    return 0


def _list_snapshots(args: argparse.Namespace) -> int:
    outdir = _configured_outdir(args.config, args.outdir)
    if not outdir.is_dir():
        raise FileNotFoundError(f"output directory not found: {outdir}")
    snapshots = _snapshots(outdir)
    if not snapshots:
        raise FileNotFoundError(f"no snapshots in {outdir}")

    print(f"Snapshots in {outdir.resolve()}:")
    tracer_count = 0
    for snapshot in snapshots:
        has_tracers = _has_tracers(snapshot)
        tracer_count += has_tracers
        print(
            f"  t={_snapshot_time(snapshot):10.6f}  {snapshot.name}"
            f"  tracer_data={'yes' if has_tracers else 'no'}"
        )
    print(f"\nTotal: {len(snapshots)} snapshots, {tracer_count} with tracer data.")
    return 0


def _tracer_mass(data: Dict[str, Any]) -> float:
    values = np.asarray(data.get("tracer_mass", 1.0), dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("tracer_mass is empty")
    if not np.allclose(values, values[0]):
        raise ValueError("per-particle tracer masses are not supported")
    return float(values[0])


def _tracer_grid_shape(
    config: Dict[str, Any], requested: Optional[List[int]]
) -> Optional[Tuple[int, int, int]]:
    if requested is not None:
        shape = tuple(int(size) for size in requested)
        if any(size <= 0 for size in shape):
            raise ValueError("tracer density grid dimensions must be positive")
        return shape[0], shape[1], shape[2]
    configured = config.get("tracers", {}).get("density_grid")
    if isinstance(configured, (list, tuple)) and len(configured) == 3:
        shape = tuple(int(size) for size in configured)
        if any(size <= 0 for size in shape):
            raise ValueError("tracer density grid dimensions must be positive")
        return shape[0], shape[1], shape[2]
    return None


def _plot_tracer_density(args: argparse.Namespace) -> int:
    snapshot = _resolve_snapshot(args.snapshot, args.config, args.outdir, require_tracers=True)
    data = load_checkpoint(str(snapshot))
    missing = [name for name in ("tracer_x", "tracer_y", "tracer_z") if name not in data]
    if missing:
        raise ValueError(f"snapshot is missing tracer data: {', '.join(missing)}")
    meta = data.get("meta", {})
    if not isinstance(meta, dict):
        meta = {}
    tracers = SimpleNamespace(
        x=np.asarray(data["tracer_x"]).flatten(),
        y=np.asarray(data["tracer_y"]).flatten(),
        z=np.asarray(data["tracer_z"]).flatten(),
        mass=_tracer_mass(data),
    )
    config = _config(args.config)
    density, _ = tracer_density_3d(
        tracers,
        _domain(meta),
        grid_shape=_tracer_grid_shape(config, args.grid),
        layout=args.layout,
    )
    output = Path(args.output) if args.output else snapshot.with_name(
        f"{snapshot.stem}_tracer_density.png"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    plot_tracer_density_3d(
        density,
        _domain(meta),
        outpath=str(output),
        log=not args.linear,
        title=f"Tracer density (t={float(data['t']):.3f})",
        vmin=args.vmin,
        vmax=args.vmax,
    )
    print(f"Saved: {output}")
    if np.any(density > 0.0):
        print(
            "  Density range: "
            f"{np.nanmin(density[density > 0.0]):.3e} .. {np.nanmax(density):.3e}"
        )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="list saved snapshots")
    list_parser.add_argument("--config", default="configs/turb3d.yaml")
    list_parser.add_argument("--outdir", help="override io.outdir from the config")
    list_parser.set_defaults(func=_list_snapshots)

    plot_parser = commands.add_parser("plot", help="plot a 3-D density snapshot")
    plot_parser.add_argument("snapshot", help="path to a 3-D snapshot")
    plot_parser.add_argument("--config", help="optional config supplying video defaults")
    plot_parser.add_argument("--view", choices=("projection", "slice", "both"), default="both")
    plot_parser.add_argument("--column-axis", choices=("x", "y", "z"))
    plot_parser.add_argument("--slice-axis", choices=("x", "y", "z"))
    plot_parser.add_argument("--thickness", type=float, help="central projection fraction")
    plot_parser.add_argument("--position", type=float, help="slice position in [0, 1]")
    plot_parser.add_argument("--colorbar-scale", choices=("linear", "log"))
    plot_parser.add_argument("--vmin", type=float)
    plot_parser.add_argument("--vmax", type=float)
    plot_parser.add_argument("--cmap", default="viridis")
    plot_parser.add_argument("--scale", type=float)
    plot_parser.add_argument("--dpi", type=int)
    plot_parser.add_argument("-o", "--output")
    plot_parser.set_defaults(func=_plot_snapshot)

    tracer_parser = commands.add_parser(
        "tracer-density", help="reconstruct tracer density from a saved 3-D snapshot"
    )
    tracer_parser.add_argument("snapshot", nargs="?", help="default: latest tracer snapshot")
    tracer_parser.add_argument("--config", default="configs/turb3d.yaml")
    tracer_parser.add_argument("--outdir", help="override io.outdir from the config")
    tracer_parser.add_argument("--layout", choices=("auto", "uniform", "random"), default="auto")
    tracer_parser.add_argument("--grid", type=int, nargs=3, metavar=("NX", "NY", "NZ"))
    tracer_parser.add_argument("--linear", action="store_true", help="use a linear color scale")
    tracer_parser.add_argument("--vmin", type=float)
    tracer_parser.add_argument("--vmax", type=float)
    tracer_parser.add_argument("-o", "--output")
    tracer_parser.set_defaults(func=_plot_tracer_density)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
