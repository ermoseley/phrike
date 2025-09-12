import argparse
import os

import numpy as np

from phrike.io import load_config, ensure_outdir
from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.initial_conditions import sod_shock_tube
from phrike.solver import SpectralSolver1D
from phrike.sod_analytic import sod_sample
from phrike.visualization import plot_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1D Sod and compare to analytic")
    parser.add_argument("--config", type=str, default="configs/sod.yaml")
    parser.add_argument("--time", type=float, default=0.2, help="Comparison time")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = os.path.abspath(cfg["io"]["outdir"]) if "io" in cfg else os.path.abspath("outputs")
    ensure_outdir(outdir)

    N = int(cfg["grid"]["N"])
    Lx = float(cfg["grid"]["Lx"]) 
    dealias = bool(cfg["grid"].get("dealias", True))
    gamma = float(cfg["physics"]["gamma"]) 

    grid = Grid1D(N=N, Lx=Lx, dealias=dealias, filter_params=cfg["integration"].get("spectral_filter", {"enabled": False}))
    eqs = EulerEquations1D(gamma=gamma)

    x0 = float(cfg["grid"].get("x0", 0.5 * Lx))
    left = cfg["initial_conditions"]["left"]
    right = cfg["initial_conditions"]["right"]
    U0 = sod_shock_tube(grid.x, x0, left, right, gamma)

    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme=str(cfg["integration"].get("scheme", "rk4")), cfl=float(cfg["integration"]["cfl"]))
    solver.run(U0, t0=float(cfg["integration"]["t0"]), t_end=args.time, output_interval=args.time)

    # Analytic sample at the same time
    ana = sod_sample(grid.x, t=args.time, x0=x0, left=left, right=right, gamma=gamma)
    rho_a, u_a, p_a = ana["rho"], ana["u"], ana["p"]

    # Overlay plots
    rho, u, p, _ = eqs.primitive(solver.U)
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)
    axs[0].plot(grid.x, rho, label="num")
    axs[0].plot(grid.x, rho_a, "--", label="analytic")
    axs[0].set_ylabel("rho")
    axs[1].plot(grid.x, u, label="num")
    axs[1].plot(grid.x, u_a, "--", label="analytic")
    axs[1].set_ylabel("u")
    axs[2].plot(grid.x, p, label="num")
    axs[2].plot(grid.x, p_a, "--", label="analytic")
    axs[2].set_ylabel("p")
    for ax in axs:
        ax.legend()
        ax.set_xlabel("x")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"Sod at t={args.time}")
    fig.savefig(os.path.join(outdir, f"sod_compare_t{args.time:.3f}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()


