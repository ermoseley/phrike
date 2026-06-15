"""Tracer particle density estimation and plotting.

Estimates density from tracer positions: for uniform 3D grids uses the
LagrangianPhaseSpaceSheetEasy tetrahedra-based voxel volumes when that repo is
available (at phrike/LagrangianPhaseSpaceSheetEasy or on PYTHONPATH); otherwise
uses histogram deposit. Plots 3D density via slices and projections.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:
    import torch  # type: ignore
    def _arr_from_tracers(a: Any) -> np.ndarray:
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy().flatten()
        return np.asarray(a, dtype=np.float64).flatten()
except Exception:
    def _arr_from_tracers(a: Any) -> np.ndarray:
        return np.asarray(a, dtype=np.float64).flatten()

# Optional: LagrangianPhaseSpaceSheetEasy (phase-space sheet voxel volumes)
_SHEET_UTILS = None
_MAKE_FIGS = None
_pkg_dir = Path(__file__).resolve().parent
_proj_root = _pkg_dir.parent
_sheet_path = _proj_root / "LagrangianPhaseSpaceSheetEasy"
if _sheet_path.is_dir():
    try:
        sys.path.insert(0, str(_proj_root))
        from LagrangianPhaseSpaceSheetEasy.SheetUtils import get_voxel_volumes
        _SHEET_UTILS = get_voxel_volumes
    except Exception:
        pass
    try:
        from LagrangianPhaseSpaceSheetEasy.helpers import makeFigs as _makeFigs_impl
        _MAKE_FIGS = _makeFigs_impl
    except Exception:
        pass


def _is_perfect_cube(P: int) -> Optional[int]:
    """Return n if P == n**3, else None."""
    if P <= 0:
        return None
    n = int(round(P ** (1.0 / 3.0)))
    if n * n * n == P:
        return n
    return None


def _build_p3d_uniform(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, n: int
) -> np.ndarray:
    """Build (n, n, n, 3) vertex grid from raveled x, y, z (C-order, indexing='ij')."""
    p3d = np.empty((n, n, n, 3), dtype=np.float64)
    p3d[:, :, :, 0] = x.reshape(n, n, n)
    p3d[:, :, :, 1] = y.reshape(n, n, n)
    p3d[:, :, :, 2] = z.reshape(n, n, n)
    return p3d


def tracer_density_3d(
    tracers: Any,
    domain: Tuple[float, float, float],
    grid_shape: Optional[Tuple[int, int, int]] = None,
    layout: str = "auto",
    mass: Optional[float] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]:
    """Estimate 3D density field from tracer positions.

    For uniform layout and n^3 particles, uses tetrahedra-based voxel volumes
    when LagrangianPhaseSpaceSheetEasy is available; otherwise histogram deposit.
    For random layout (or non-cube count), uses histogram deposit.

    Args:
        tracers: Object with .x, .y, .z (each 1D length P) and optionally .mass.
        domain: (Lx, Ly, Lz).
        grid_shape: (nx, ny, nz) for histogram output; if None, uniform uses
            (n-1)^3 voxels, random uses (32, 32, 32).
        layout: 'auto' | 'uniform' | 'random'. If 'auto', use uniform when
            P is a perfect cube.
        mass: Override mass per particle (default: tracers.mass or 1.0).

    Returns:
        rho_3d: 3D density array (nz, ny, nx) in index order (axis 0 = z, 1 = y, 2 = x).
        extent: (xmin, xmax, ymin, ymax, zmin, zmax) for plotting.
    """
    Lx, Ly, Lz = domain
    x = _arr_from_tracers(tracers.x)
    y = _arr_from_tracers(tracers.y)
    z = _arr_from_tracers(tracers.z)
    P = len(x)
    if P != len(y) or P != len(z):
        raise ValueError("tracers.x, .y, .z must have the same length")
    mass_val = float(mass) if mass is not None else getattr(tracers, "mass", 1.0)
    x = np.mod(x, Lx)
    y = np.mod(y, Ly)
    z = np.mod(z, Lz)

    use_uniform = False
    if layout == "uniform":
        n_cube = _is_perfect_cube(P)
        use_uniform = n_cube is not None
    elif layout == "auto":
        n_cube = _is_perfect_cube(P)
        use_uniform = n_cube is not None and _SHEET_UTILS is not None

    if use_uniform and n_cube is not None and n_cube >= 2 and _SHEET_UTILS is not None:
        n = n_cube
        p3d = _build_p3d_uniform(x, y, z, n)
        M = n - 1
        voxvol = _SHEET_UTILS(M, p3d)
        voxvol = np.abs(voxvol)
        num_voxels = (n - 1) ** 3
        mass_per_voxel = (P * mass_val) / num_voxels if num_voxels > 0 else mass_val
        rho_3d = np.where(voxvol > 1e-30, mass_per_voxel / voxvol, 0.0)
        rho_3d = rho_3d.astype(np.float64)
        extent = (0.0, Lx, 0.0, Ly, 0.0, Lz)
        return rho_3d, extent

    nx, ny, nz = (32, 32, 32)
    if grid_shape is not None:
        nx, ny, nz = grid_shape
    cell_vol = (Lx / nx) * (Ly / ny) * (Lz / nz)
    if cell_vol <= 0:
        cell_vol = 1.0
    counts, _ = np.histogramdd(
        [x, y, z],
        bins=(nx, ny, nz),
        range=[[0.0, Lx], [0.0, Ly], [0.0, Lz]],
    )
    rho_3d = (counts * mass_val) / cell_vol
    rho_3d = rho_3d.astype(np.float64)
    extent = (0.0, Lx, 0.0, Ly, 0.0, Lz)
    return rho_3d, extent


def _plot_tracer_density_3d_simple(
    rho_3d: np.ndarray,
    extent: Tuple[float, float, float, float, float, float],
    outpath: Optional[str] = None,
    log: bool = True,
    title: str = "Tracer density",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile: Optional[Tuple[float, float]] = (2.0, 98.0),
) -> None:
    """6-panel figure: slices (mid) and projections. extent = (xmin, xmax, ymin, ymax, zmin, zmax).

    If vmin/vmax are None, they are set from percentile of valid values (default 2–98%)
    so the color range is not dominated by a few extreme voxels.
    """
    import matplotlib.pyplot as plt

    xmin, xmax, ymin, ymax, zmin, zmax = extent
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.subplots_adjust(wspace=0.2, hspace=0.25, bottom=0.08, left=0.06, top=0.92, right=0.88)
    axcb = fig.add_axes([0.90, 0.08, 0.02, 0.84])
    data = np.maximum(rho_3d, 1e-30) if log else rho_3d
    if log:
        data = np.log10(data)
    valid = np.isfinite(data)
    if vmin is None or vmax is None:
        if valid.any() and percentile is not None:
            lo, hi = percentile
            vmin_pt = np.nanpercentile(data[valid], lo)
            vmax_pt = np.nanpercentile(data[valid], hi)
            if vmin is None:
                vmin = vmin_pt
            if vmax is None:
                vmax = vmax_pt
    if vmin is None:
        vmin = np.nanmin(data) if valid.any() else 0.0
    if vmax is None:
        vmax = np.nanmax(data) if valid.any() else vmin + 1.0
    if not np.isfinite(vmax):
        vmax = vmin + 1.0
    imargs = {"origin": "lower", "aspect": "auto", "interpolation": "nearest", "vmin": vmin, "vmax": vmax}
    # rho_3d axis 0=z, 1=y, 2=x. Slice at axis k -> 2D in the other two dims.
    labels = ["z", "y", "x"]
    for ax_idx, axis_dim in enumerate([0, 1, 2]):
        mid = rho_3d.shape[axis_dim] // 2
        slice_2d = np.take(rho_3d, mid, axis=axis_dim)
        slice_2d = np.maximum(slice_2d, 1e-30)
        if log:
            slice_2d = np.log10(slice_2d)
        proj_2d = np.mean(rho_3d, axis=axis_dim)
        proj_2d = np.maximum(proj_2d, 1e-30)
        if log:
            proj_2d = np.log10(proj_2d)
        if axis_dim == 0:
            ext = [xmin, xmax, ymin, ymax]
        elif axis_dim == 1:
            ext = [xmin, xmax, zmin, zmax]
        else:
            ext = [ymin, ymax, zmin, zmax]
        axes[0, ax_idx].imshow(slice_2d, extent=ext, **imargs)
        axes[1, ax_idx].imshow(proj_2d, extent=ext, **imargs)
        axes[0, ax_idx].set_title(f"Slice ({labels[ax_idx]}=mid)")
        axes[1, ax_idx].set_xlabel(labels[ax_idx])
    axes[0, 0].set_ylabel("Slice")
    axes[1, 0].set_ylabel("Projection")
    im = axes[0, 0].images[0] if axes[0, 0].images else None
    if im is not None:
        plt.colorbar(im, cax=axcb, label="log10(density)" if log else "density")
    fig.suptitle(title)
    if outpath:
        fig.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_tracer_density_3d(
    rho_3d: np.ndarray,
    domain: Tuple[float, float, float],
    outpath: Optional[str] = None,
    log: bool = True,
    title: str = "Tracer density",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    percentile: Optional[Tuple[float, float]] = (2.0, 98.0),
) -> None:
    """Plot 3D tracer density: slices and projections.

    If LagrangianPhaseSpaceSheetEasy.helpers.makeFigs is available, uses it;
    otherwise uses a simple 6-panel figure. When outpath is set, saves the
    figure and does not block. Color range defaults to 2–98% percentile so
    extreme voxels do not dominate; pass vmin/vmax to override.

    Args:
        rho_3d: 3D array (nz, ny, nx).
        domain: (Lx, Ly, Lz) for extent.
        outpath: If set, save figure here and close.
        log: Use log10 scale.
        title: Figure title.
        vmin: Colorbar minimum (in log10 if log=True). Default from percentile.
        vmax: Colorbar maximum (in log10 if log=True). Default from percentile.
        percentile: (low, high) percentiles for default range; None = use full range.
    """
    Lx, Ly, Lz = domain
    extent = (0.0, Lx, 0.0, Ly, 0.0, Lz)
    if _MAKE_FIGS is not None and outpath is None:
        _MAKE_FIGS(rho_3d, log=log, title=title)
        return
    _plot_tracer_density_3d_simple(
        rho_3d, extent, outpath=outpath, log=log, title=title,
        vmin=vmin, vmax=vmax, percentile=percentile,
    )
