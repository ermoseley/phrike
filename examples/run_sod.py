import argparse
import os

from spectralhydro.io import load_config, ensure_outdir, save_solution_snapshot
from spectralhydro.grid import Grid1D
from spectralhydro.equations import EulerEquations1D
from spectralhydro.initial_conditions import sod_shock_tube
from spectralhydro.solver import SpectralSolver1D
from spectralhydro.visualization import plot_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 1D Sod shock tube with SpectralHydro")
    parser.add_argument("--config", type=str, default="configs/sod.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = os.path.abspath(cfg["io"]["outdir"]) if "io" in cfg else os.path.abspath("outputs")
    ensure_outdir(outdir)

    N = int(cfg["grid"]["N"])
    Lx = float(cfg["grid"]["Lx"]) 
    dealias = bool(cfg["grid"].get("dealias", True))

    gamma = float(cfg["physics"]["gamma"]) 

    t0 = float(cfg["integration"]["t0"]) 
    t_end = float(cfg["integration"]["t_end"]) 
    cfl = float(cfg["integration"]["cfl"]) 
    scheme = str(cfg["integration"].get("scheme", "rk4"))
    output_interval = float(cfg["integration"].get("output_interval", 0.05))
    checkpoint_interval = float(cfg["integration"].get("checkpoint_interval", 0.0))
    filt_cfg = cfg["integration"].get("spectral_filter", {"enabled": False})

    grid = Grid1D(N=N, Lx=Lx, dealias=dealias, filter_params=filt_cfg)
    eqs = EulerEquations1D(gamma=gamma)

    x0 = float(cfg["grid"].get("x0", 0.5 * Lx))
    left = cfg["initial_conditions"]["left"]
    right = cfg["initial_conditions"]["right"]
    U0 = sod_shock_tube(grid.x, x0, left, right, gamma)

    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme=scheme, cfl=cfl)
    history = solver.run(U0, t0=t0, t_end=t_end, output_interval=output_interval, checkpoint_interval=checkpoint_interval, outdir=outdir)

    # Save final snapshot and plot
    snapshot_path = save_solution_snapshot(outdir, solver.t, U=solver.U, grid=grid, equations=eqs)
    print(f"Saved final snapshot: {snapshot_path}")
    plot_fields(grid=grid, U=solver.U, equations=eqs, title=f"Sod at t={solver.t:.3f}", outpath=os.path.join(outdir, f"fields_t{solver.t:.3f}.png"))

    # Optionally plot conserved quantities
    if history and len(history.get("time", [])) > 0:
        from spectralhydro.visualization import plot_conserved_time_series
        plot_conserved_time_series(history, outpath=os.path.join(outdir, "conserved.png"))


if __name__ == "__main__":
    main()


