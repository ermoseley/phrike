import argparse
import os

import numpy as np

from phrike.io import load_config, ensure_outdir
from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.solver import SpectralSolver1D
from phrike.acoustic import AcousticParams, acoustic_ic, acoustic_exact
from phrike.visualization import plot_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Run periodic acoustic wave (spectral validation)")
    parser.add_argument("--config", type=str, default="configs/acoustic.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = os.path.abspath(cfg["io"]["outdir"]) if "io" in cfg else os.path.abspath("outputs/acoustic")
    ensure_outdir(outdir)

    N = int(cfg["grid"]["N"])
    Lx = float(cfg["grid"]["Lx"]) 
    dealias = bool(cfg["grid"].get("dealias", True))
    gamma = float(cfg["physics"]["gamma"]) 

    fft_workers = int(cfg["grid"].get("fft_workers", os.cpu_count() or 1))
    grid = Grid1D(N=N, Lx=Lx, dealias=dealias, filter_params=cfg["integration"].get("spectral_filter", {"enabled": False}), fft_workers=fft_workers)
    eqs = EulerEquations1D(gamma=gamma)

    acfg = cfg["initial_conditions"]["acoustic"]
    params = AcousticParams(
        rho0=float(acfg.get("rho0", 1.0)),
        p0=float(acfg.get("p0", 1.0)),
        u0=float(acfg.get("u0", 0.0)),
        amplitude=float(acfg.get("amplitude", 1e-6)),
        mode_m=int(acfg.get("mode_m", 1)),
        Lx=Lx,
        direction=str(acfg.get("direction", "right")),
        fractional=bool(acfg.get("fractional", True)),
        gamma=gamma,
    )

    U0 = acoustic_ic(grid.x, params)

    t0 = float(cfg["integration"]["t0"]) 
    t_end = float(cfg["integration"]["t_end"]) 
    cfl = float(cfg["integration"]["cfl"]) 
    scheme = str(cfg["integration"].get("scheme", "rk4"))
    output_interval = float(cfg["integration"].get("output_interval", 0.1))

    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme=scheme, cfl=cfl)
    solver.run(U0, t0=t0, t_end=t_end, output_interval=output_interval, outdir=outdir)

    # Analytic at t_end
    rho_a, u_a, p_a = acoustic_exact(grid.x, t_end, params)
    rho, u, p, _ = eqs.primitive(solver.U)

    # Errors
    def rel_l2(a, b):
        return np.linalg.norm(a - b) / np.linalg.norm(b)

    err_rho = rel_l2(rho, rho_a)
    err_u = rel_l2(u, u_a)
    err_p = rel_l2(p, p_a)
    with open(os.path.join(outdir, "errors.txt"), "w", encoding="utf-8") as f:
        f.write(f"relL2 rho: {err_rho}\n")
        f.write(f"relL2 u:   {err_u}\n")
        f.write(f"relL2 p:   {err_p}\n")

    # Plot numerical vs analytic
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
    fig.suptitle(f"Acoustic wave m={params.mode_m} at t={t_end}")
    fig.savefig(os.path.join(outdir, f"acoustic_compare_t{t_end:.3f}.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()


