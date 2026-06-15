"""Run the 2D Orszag-Tang vortex on Metal (MPS) with blow-up detection.

Usage: python run_orszag_tang2d.py [N] [t_end] [eta] [nu] [cfl]
Defaults: N=1024 t_end=0.5 eta=0 nu=0 cfl=0.25
"""

import sys
import time
import numpy as np
import torch

from phrike.grid import Grid2D
from phrike.equations import MHDEquations2D
from phrike.solver import SpectralSolverMHD2D
from phrike.problems.problem_list.orszag_tang2d import orszag_tang_2d


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    t_end = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
    eta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    nu = float(sys.argv[4]) if len(sys.argv) > 4 else 0.0
    cfl = float(sys.argv[5]) if len(sys.argv) > 5 else 0.25
    out_int = 0.05

    print(f"OT2D N={N} t_end={t_end} eta={eta} nu={nu} cfl={cfl} device=mps float32",
          flush=True)
    g = Grid2D(Nx=N, Ny=N, Lx=1.0, Ly=1.0, dealias=True, backend="torch",
               torch_device="mps", precision="single",
               filter_params={"enabled": True, "p": 8, "alpha": 36.0})
    g.mhd_config = {"resistivity": eta, "viscosity": nu,
                    "project_each_substage": True}
    eqs = MHDEquations2D(gamma=5.0 / 3.0)
    X, Y = g.xy_mesh()
    U0 = orszag_tang_2d(X, Y, gamma=5.0 / 3.0, Lx=g.Lx, Ly=g.Ly)
    solver = SpectralSolverMHD2D(g, eqs, scheme="rk4", cfl=cfl)

    blew_up = {"flag": False, "neg_p": False}
    t0 = time.time()
    tag0 = f"N{N}_eta{eta}_nu{nu}".replace(".", "p")
    last_good = {"rho": None, "t": 0.0}

    class StopSim(Exception):
        pass

    def on_output(t, U):
        rho, ux, uy, uz, p, Bx, By, Bz = eqs.primitive(U)
        finite = bool(torch.isfinite(U).all())
        rmin = float(torch.min(rho)); rmax = float(torch.max(rho))
        pmin = float(torch.min(p))
        divb = solver.max_div_b(U)
        umax = float(torch.max(torch.abs(U)))
        print(f"  t={t:.4f}  rho=[{rmin:.4f},{rmax:.4f}]  p_min={pmin:.4e}  "
              f"max|U|={umax:.3e}  divB={divb:.2e}  finite={finite}  "
              f"wall={time.time()-t0:.0f}s", flush=True)
        if pmin < 0:
            blew_up["neg_p"] = True
        if finite and rmin > 0:
            last_good["rho"] = rho.detach().cpu().numpy()
            last_good["t"] = t
        if (not finite) or umax > 1e4:
            blew_up["flag"] = True
            raise StopSim()

    try:
        hist = solver.run(U0, 0.0, t_end, output_interval=out_int, on_output=on_output)
        e0, e1 = hist["energy"][0], hist["energy"][-1]
        dE = abs(e1 - e0) / abs(e0)
        divB_final = hist["div_b_max"][-1]
    except StopSim:
        dE, divB_final = float("nan"), float("nan")
    print(f"DONE t={solver.t:.4f} nonfinite_blowup={blew_up['flag']} "
          f"negative_pressure={blew_up['neg_p']} dE/E={dE:.2e} "
          f"divB_final={divB_final:.2e}", flush=True)

    # Save raw final density and a density figure (linear colorbar 0..0.5).
    # If it blew up (non-finite), fall back to the last good (finite) snapshot.
    if blew_up["flag"] and last_good["rho"] is not None:
        rho = last_good["rho"]
        solver.t = last_good["t"]
        print(f"(using last finite snapshot at t={solver.t:.4f} for the figure)", flush=True)
    else:
        rho = eqs.primitive(solver.U)[0].detach().cpu().numpy()
    tag = f"N{N}_t{solver.t:.2f}_eta{eta}_nu{nu}".replace(".", "p")
    try:
        np.save(f"outputs/ot_density_{tag}.npy", rho)
    except Exception as e:
        print(f"npy save failed: {e}", flush=True)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6.2, 5), constrained_layout=True)
        im = ax.imshow(rho, origin="lower", extent=[0, 1, 0, 1], cmap="inferno",
                       vmin=0.0, vmax=0.5)
        ax.set_title(f"Orszag-Tang density  N={N}  t={solver.t:.3f}"
                     + (f"  eta={eta} nu={nu}" if (eta or nu) else "  (ideal)"))
        ax.set_xlabel("x"); ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, shrink=0.85, label="density")
        fig.savefig(f"outputs/orszag_tang2d_{tag}.png", dpi=140, bbox_inches="tight")
        print(f"saved figure outputs/orszag_tang2d_{tag}.png", flush=True)
    except Exception as e:
        print(f"figure failed: {e}", flush=True)


if __name__ == "__main__":
    main()
