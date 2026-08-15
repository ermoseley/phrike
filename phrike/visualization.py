from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _axis_index(axis: str) -> int:
    """Return the array axis for a 3-D field stored as ``(z, y, x)``."""
    try:
        return {"z": 0, "y": 1, "x": 2}[axis]
    except KeyError as exc:
        raise ValueError(f"axis must be x, y, or z, got {axis!r}") from exc


def central_column_mean(
    field: np.ndarray, axis: str = "z", thickness: float = 1.0
) -> np.ndarray:
    """Average a central fraction of a 3-D ``(z, y, x)`` field along ``axis``."""
    if field.ndim != 3:
        raise ValueError(f"expected a 3-D field, got shape {field.shape}")
    if not 0.0 < thickness <= 1.0:
        raise ValueError(f"thickness must be in (0, 1], got {thickness}")

    index = _axis_index(axis)
    size = field.shape[index]
    start = max(0, int(size * (0.5 - thickness / 2.0)))
    stop = min(size, int(size * (0.5 + thickness / 2.0)))
    if start >= stop:
        start = min(size - 1, max(0, size // 2))
        stop = start + 1
    return np.mean(np.take(field, range(start, stop), axis=index), axis=index)


def central_slice(
    field: np.ndarray, axis: str = "z", position: float = 0.5
) -> np.ndarray:
    """Return a slice through a 3-D ``(z, y, x)`` field at fractional ``position``."""
    if field.ndim != 3:
        raise ValueError(f"expected a 3-D field, got shape {field.shape}")
    if not 0.0 <= position <= 1.0:
        raise ValueError(f"position must be in [0, 1], got {position}")

    index = _axis_index(axis)
    slice_index = min(field.shape[index] - 1, int(field.shape[index] * position))
    return np.take(field, slice_index, axis=index)


def projection_extent(
    domain: Tuple[float, float, float], axis: str = "z"
) -> Tuple[float, float, float, float]:
    """Return the Matplotlib extent for a projection or slice normal to ``axis``."""
    lx, ly, lz = domain
    _axis_index(axis)
    if axis == "z":
        return (0.0, lx, 0.0, ly)
    if axis == "y":
        return (0.0, lx, 0.0, lz)
    return (0.0, ly, 0.0, lz)


def projection_axis_labels(axis: str = "z") -> Tuple[str, str]:
    """Return coordinate labels for a projection or slice normal to ``axis``."""
    _axis_index(axis)
    if axis == "z":
        return ("x", "y")
    if axis == "y":
        return ("x", "z")
    return ("y", "z")


def plot_fields(
    grid, U, equations, title: str = "", outpath: Optional[str] = None
) -> None:
    import matplotlib.pyplot as plt

    rho, u, p, _ = equations.primitive(U)
    E = U[2]
    
    # Convert Torch tensors to NumPy for plotting
    def to_numpy(x):
        if hasattr(x, 'cpu'):  # Torch tensor
            return x.cpu().numpy()
        return x
    
    x = to_numpy(grid.x)
    rho = to_numpy(rho)
    u = to_numpy(u)
    p = to_numpy(p)
    E = to_numpy(E)

    fig, axs = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    axs[0, 0].plot(x, rho)
    axs[0, 0].set_title("Density")
    axs[0, 1].plot(x, u)
    axs[0, 1].set_title("Velocity")
    axs[1, 0].plot(x, p)
    axs[1, 0].set_title("Pressure")
    axs[1, 1].plot(x, E)
    axs[1, 1].set_title("Energy density")
    for ax in axs.flat:
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)


def plot_conserved_time_series(
    history: Dict[str, List[float]], outpath: Optional[str] = None
) -> None:
    import matplotlib.pyplot as plt

    t = np.array(history["time"])  # type: ignore[index]
    mass = np.array(history["mass"])  # type: ignore[index]
    mom = np.array(history["momentum"])  # type: ignore[index]
    energy = np.array(history["energy"])  # type: ignore[index]

    fig, axs = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axs[0].plot(t, mass)
    axs[0].set_ylabel("Mass")
    axs[1].plot(t, mom)
    axs[1].set_ylabel("Momentum")
    axs[2].plot(t, energy)
    axs[2].set_ylabel("Energy")
    for ax in axs:
        ax.set_xlabel("t")
        ax.grid(True, alpha=0.3)
    if outpath:
        fig.savefig(outpath, dpi=150)
    plt.close(fig)
