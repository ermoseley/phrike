"""Tracer particle density estimation and plotting.

Uniform tracer lattices use a periodic six-tetrahedron decomposition of each
Lagrangian cell. Random particles use a histogram estimate. The simplex path
uses unwrapped particle coordinates when snapshots provide them, avoiding
spurious box-spanning cells at periodic boundaries.
"""

from __future__ import annotations

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

def _is_perfect_cube(P: int) -> Optional[int]:
    """Return n if P == n**3, else None."""
    if P <= 0:
        return None
    n = int(round(P ** (1.0 / 3.0)))
    if n * n * n == P:
        return n
    return None


def _build_p3d_uniform(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, shape: Tuple[int, int, int]
) -> np.ndarray:
    """Build an ``(nx, ny, nz, 3)`` vertex grid from C-order arrays."""
    p3d = np.empty((*shape, 3), dtype=np.float64)
    p3d[..., 0] = x.reshape(shape)
    p3d[..., 1] = y.reshape(shape)
    p3d[..., 2] = z.reshape(shape)
    return p3d


_TET_CONNECTIVITY = np.asarray(
    (
        (4, 0, 7, 1),
        (1, 0, 7, 3),
        (5, 1, 4, 7),
        (2, 3, 1, 7),
        (1, 5, 6, 7),
        (2, 6, 7, 1),
    ),
    dtype=np.intp,
)
_CUBE_VERTICES = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ),
    dtype=np.intp,
)


def _uniform_lattice_shape(tracers: Any, count: int) -> Optional[Tuple[int, int, int]]:
    configured = getattr(tracers, "lattice_shape", None)
    if configured is not None:
        shape = tuple(int(value) for value in configured)
        if len(shape) != 3 or np.prod(shape) != count:
            raise ValueError("tracer lattice shape does not match particle count")
        return shape
    n = _is_perfect_cube(count)
    return None if n is None else (n, n, n)


def periodic_simplex_density_3d(
    tracers: Any,
    domain: Tuple[float, float, float],
    mass: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return periodic simplex-cell density and deformed cell centroids.

    Density is returned in ``(nz, ny, nx)`` order. Centroids have shape
    ``(nz, ny, nx, 3)`` and are wrapped into the periodic domain.
    """
    wrapped = [_arr_from_tracers(getattr(tracers, axis)) for axis in "xyz"]
    unwrapped = [
        _arr_from_tracers(getattr(tracers, f"{axis}_unwrapped", values))
        for axis, values in zip("xyz", wrapped)
    ]
    count = len(unwrapped[0])
    if any(len(values) != count for values in unwrapped[1:]):
        raise ValueError("tracer coordinate arrays must have the same length")
    shape = _uniform_lattice_shape(tracers, count)
    if shape is None or any(size < 2 for size in shape):
        raise ValueError("simplex density requires a uniform 3-D tracer lattice")

    base = _build_p3d_uniform(*unwrapped, shape)
    nx, ny, nz = shape
    extended = base[
        np.ix_(
            np.arange(nx + 1) % nx,
            np.arange(ny + 1) % ny,
            np.arange(nz + 1) % nz,
        )
    ].copy()
    extended[-1, :, :, 0] += domain[0]
    extended[:, -1, :, 1] += domain[1]
    extended[:, :, -1, 2] += domain[2]

    cube = np.empty((nx, ny, nz, 8, 3), dtype=np.float64)
    for vertex, offset in enumerate(_CUBE_VERTICES):
        i, j, k = offset
        cube[..., vertex, :] = extended[
            i : i + nx, j : j + ny, k : k + nz, :
        ]

    volume = np.zeros((nx, ny, nz), dtype=np.float64)
    for connection in _TET_CONNECTIVITY:
        a, b, c, origin = (cube[..., index, :] for index in connection)
        determinant = np.einsum(
            "...i,...i->...", a - origin, np.cross(b - origin, c - origin)
        )
        volume += np.abs(determinant) / 6.0
    if not np.all(np.isfinite(volume)) or np.any(volume <= 0.0):
        raise ValueError("tracer simplex lattice contains invalid cell volumes")

    mass_value = (
        float(mass) if mass is not None else float(getattr(tracers, "mass", 1.0))
    )
    density = mass_value / volume
    centroid = np.mean(cube, axis=-2)
    for axis, length in enumerate(domain):
        centroid[..., axis] = np.mod(centroid[..., axis], length)
    return density.transpose(2, 1, 0), centroid.transpose(2, 1, 0, 3)


def tracer_density_3d(
    tracers: Any,
    domain: Tuple[float, float, float],
    grid_shape: Optional[Tuple[int, int, int]] = None,
    layout: str = "auto",
    mass: Optional[float] = None,
) -> Tuple[np.ndarray, Tuple[float, float, float, float, float, float]]:
    """Estimate 3D density field from tracer positions.

    For a uniform lattice, use the periodic six-tetrahedron volume of each
    deformed Lagrangian cell. For random particles, use histogram deposition.

    Args:
        tracers: Object with .x, .y, .z (each 1D length P) and optionally .mass.
        domain: (Lx, Ly, Lz).
        grid_shape: (nx, ny, nz) for histogram output; if None, random layouts
            use (32, 32, 32). This is ignored for uniform lattices.
        layout: 'auto' | 'uniform' | 'random'. If 'auto', use the simplex
            estimator only when lattice topology is explicitly available.
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
        use_uniform = _uniform_lattice_shape(tracers, P) is not None
    elif layout == "auto":
        use_uniform = getattr(tracers, "lattice_shape", None) is not None

    if use_uniform:
        rho_3d, _ = periodic_simplex_density_3d(
            tracers, domain, mass=mass_val
        )
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
    rho_3d = ((counts * mass_val) / cell_vol).transpose(2, 1, 0)
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

    When outpath is set, save the figure and do not block. Color range defaults
    to the 2–98% percentile so extreme cells do not dominate; pass vmin/vmax
    to override.

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
    _plot_tracer_density_3d_simple(
        rho_3d, extent, outpath=outpath, log=log, title=title,
        vmin=vmin, vmax=vmax, percentile=percentile,
    )
