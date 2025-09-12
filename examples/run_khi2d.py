import argparse
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from phrike.io import load_config, ensure_outdir, save_solution_snapshot
from phrike.grid import Grid2D
from phrike.equations import EulerEquations2D
from phrike.initial_conditions import kelvin_helmholtz_2d
from phrike.solver import SpectralSolver2D


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D Kelvin-Helmholtz instability")
    parser.add_argument("--config", type=str, default="configs/khi2d.yaml")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"], help="Array backend")
    parser.add_argument("--device", type=str, default=None, help="Torch device: cpu|mps|cuda (if backend=torch)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = os.path.abspath(cfg["io"]["outdir"]) if "io" in cfg else os.path.abspath("outputs/khi2d")
    ensure_outdir(outdir)

    Nx = int(cfg["grid"]["Nx"])
    Ny = int(cfg["grid"]["Ny"])
    Lx = float(cfg["grid"]["Lx"]) 
    Ly = float(cfg["grid"]["Ly"]) 
    dealias = bool(cfg["grid"].get("dealias", True))
    fft_workers = int(cfg["grid"].get("fft_workers", os.cpu_count() or 1))

    gamma = float(cfg["physics"]["gamma"]) 

    t0 = float(cfg["integration"]["t0"]) 
    t_end = float(cfg["integration"]["t_end"]) 
    cfl = float(cfg["integration"]["cfl"]) 
    scheme = str(cfg["integration"].get("scheme", "rk4"))
    output_interval = float(cfg["integration"].get("output_interval", 0.1))
    filt_cfg = cfg["integration"].get("spectral_filter", {"enabled": False})

    grid = Grid2D(Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly, dealias=dealias, filter_params=filt_cfg, fft_workers=fft_workers, backend=args.backend, torch_device=args.device)
    eqs = EulerEquations2D(gamma=gamma)

    X, Y = grid.xy_mesh()
    icfg = cfg["initial_conditions"]
    rho_outer = float(icfg.get("rho_outer", 1.0))
    rho_inner = float(icfg.get("rho_inner", 2.0))
    u0 = float(icfg.get("u0", 1.0))
    shear_thickness = float(icfg.get("shear_thickness", 0.02))
    pressure_outer = float(icfg.get("pressure_outer", 1.0))
    pressure_inner = float(icfg.get("pressure_inner", 1.0))
    perturb_eps = float(icfg.get("perturb_eps", 0.01))
    perturb_sigma = float(icfg.get("perturb_sigma", 0.02))
    perturb_kx = int(icfg.get("perturb_kx", 2))

    U0 = kelvin_helmholtz_2d(X, Y, rho_outer=rho_outer, rho_inner=rho_inner, 
                              u0=u0, shear_thickness=shear_thickness, 
                              pressure_outer=pressure_outer, pressure_inner=pressure_inner,
                              perturb_eps=perturb_eps, perturb_sigma=perturb_sigma, 
                              perturb_kx=perturb_kx, gamma=gamma)

    solver = SpectralSolver2D(grid=grid, equations=eqs, scheme=scheme, cfl=cfl)

    # Simple visualization callback
    frames_dir = os.path.join(outdir, "frames")
    ensure_outdir(frames_dir)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
    def render(t: float, U):
        rho, ux, uy, p = eqs.primitive(U)
        # Convert torch tensors to numpy for plotting
        try:
            import torch  # type: ignore
            if isinstance(rho, (torch.Tensor,)):
                rho = rho.detach().cpu().numpy()
        except Exception:
            pass
        
        ax.cla()
        ax.imshow(rho, origin='lower', extent=[0, Lx, 0, Ly], aspect='equal')
        ax.set_title(f"Density t={t:.3f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        fig.canvas.draw(); fig.canvas.flush_events()
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=120)
        
        # Save snapshot at every output interval
        snapshot_path = save_solution_snapshot(outdir, t, U=U, grid=grid, equations=eqs)
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

    solver.run(U0, t0=t0, t_end=t_end, output_interval=output_interval, outdir=outdir, on_output=render)
    plt.close(fig)
    
    rho, ux, uy, p = eqs.primitive(solver.U)
    # Convert torch tensors to numpy for plotting
    try:
        import torch  # type: ignore
        if isinstance(rho, (torch.Tensor,)):
            rho = rho.detach().cpu().numpy()
    except Exception:
        pass
    
    plt.figure(figsize=(8, 8))
    plt.imshow(rho, origin='lower', extent=[0, Lx, 0, Ly], aspect='equal')
    plt.title(f"Density t={solver.t:.3f}")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"khi2d_t{solver.t:.3f}.png"), dpi=150)
    plt.close()
    
    # Generate video from frames
    fps = int(cfg.get("video", {}).get("fps", 30))
    codec = str(cfg.get("video", {}).get("codec", "h264_videotoolbox"))
    crf = int(cfg.get("video", {}).get("crf", 18))
    pix_fmt = str(cfg.get("video", {}).get("pix_fmt", "yuv420p"))
    video_path = os.path.join(outdir, "khi2d.mp4")
    
    # Normalize frame indices
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    for i, fname in enumerate(frames):
        src = os.path.join(frames_dir, fname)
        dst = os.path.join(frames_dir, f"frame_{i:08d}.png")
        if src != dst:
            os.replace(src, dst)
    
    def try_ffmpeg(codec_name: str) -> bool:
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "frame_%08d.png"),
            "-c:v", codec_name, "-crf", str(crf), "-pix_fmt", pix_fmt,
            video_path,
        ]
        try:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        except FileNotFoundError:
            return False
        return proc.returncode == 0
    
    for ctry in [codec, "libopenh264", "mpeg4", "libvpx-vp9", "libaom-av1"]:
        if try_ffmpeg(ctry):
            print(f"Wrote video: {video_path} using codec {ctry}")
            break
    else:
        print("Warning: Could not create video. Frames in:", frames_dir)


if __name__ == "__main__":
    main()


