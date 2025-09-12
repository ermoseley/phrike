from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_fields(
    grid, U, equations, title: str = "", outpath: Optional[str] = None
) -> None:
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
