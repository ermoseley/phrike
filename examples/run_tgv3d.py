import argparse
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from phrike.io import load_config, ensure_outdir, save_solution_snapshot
from phrike.grid import Grid3D
from phrike.equations import EulerEquations3D
from phrike.initial_conditions import taylor_green_vortex_3d
from phrike.solver import SpectralSolver3D


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D Taylor-Green vortex with SpectralHydro")
    parser.add_argument("--config", type=str, default="configs/tgv3d.yaml")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"], help="Array backend")
    parser.add_argument("--device", type=str, default=None, help="Torch device: cpu|mps|cuda (if backend=torch)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = os.path.abspath(cfg["io"]["outdir"]) if "io" in cfg else os.path.abspath("outputs/tgv3d")
    ensure_outdir(outdir)

    Nx = int(cfg["grid"]["Nx"]) 
    Ny = int(cfg["grid"]["Ny"]) 
    Nz = int(cfg["grid"]["Nz"]) 
    Lx = float(cfg["grid"]["Lx"]) 
    Ly = float(cfg["grid"]["Ly"]) 
    Lz = float(cfg["grid"]["Lz"]) 
    dealias = bool(cfg["grid"].get("dealias", True))
    fft_workers = int(cfg["grid"].get("fft_workers", os.cpu_count() or 1))

    gamma = float(cfg["physics"]["gamma"]) 

    t0 = float(cfg["integration"]["t0"]) 
    t_end = float(cfg["integration"]["t_end"]) 
    cfl = float(cfg["integration"]["cfl"]) 
    scheme = str(cfg["integration"].get("scheme", "rk4"))
    output_interval = float(cfg["integration"].get("output_interval", 0.1))
    filt_cfg = cfg["integration"].get("spectral_filter", {"enabled": True, "p": 8, "alpha": 36.0})

    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dealias=dealias, filter_params=filt_cfg, fft_workers=fft_workers, backend=args.backend, torch_device=args.device)
    eqs = EulerEquations3D(gamma=gamma)

    X, Y, Z = grid.xyz_mesh()
    icfg = cfg["initial_conditions"]
    rho0 = float(icfg.get("rho0", 1.0))
    p0 = float(icfg.get("p0", 100.0))
    U0 = float(icfg.get("U0", 1.0))
    k = int(icfg.get("k", 2))

    U0_state = taylor_green_vortex_3d(X, Y, Z, rho0=rho0, p0=p0, U0=U0, k=k, gamma=gamma)

    solver = SpectralSolver3D(grid=grid, equations=eqs, scheme=scheme, cfl=cfl)

    frames_dir = os.path.join(outdir, "frames")
    ensure_outdir(frames_dir)

    # Render mid-plane density slices during run
    fig, ax = plt.subplots(1, 1, figsize=(7, 6), constrained_layout=True)
    def render(t: float, U):
        rho, ux, uy, uz, p = eqs.primitive(U)
        try:
            import torch  # type: ignore
            if isinstance(rho, (torch.Tensor,)):
                rho = rho.detach().cpu().numpy()
        except Exception:
            pass
        # Take mid-plane at z ~ Lz/2 (index Nz//2)
        mid = rho.shape[0] // 2
        ax.cla()
        ax.imshow(rho[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal')
        ax.set_title(f"Density z=mid t={t:.3f}")
        ax.set_xlabel('x'); ax.set_ylabel('y')
        fig.canvas.draw(); fig.canvas.flush_events()
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=120)
        snapshot_path = save_solution_snapshot(outdir, t, U=U, grid=grid, equations=eqs)
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

    solver.run(U0_state, t0=t0, t_end=t_end, output_interval=output_interval, outdir=outdir, on_output=render)
    plt.close(fig)

    # Final mid-plane slice plot
    rho, ux, uy, uz, p = eqs.primitive(solver.U)
    try:
        import torch  # type: ignore
        if isinstance(rho, (torch.Tensor,)):
            rho = rho.detach().cpu().numpy()
    except Exception:
        pass
    plt.figure(figsize=(7, 6))
    mid = rho.shape[0] // 2
    plt.imshow(rho[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal')
    plt.title(f"Density z=mid t={solver.t:.3f}")
    plt.xlabel('x'); plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"tgv3d_t{solver.t:.3f}.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    main()


