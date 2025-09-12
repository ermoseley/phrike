import argparse
import os
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from hydra.io import load_config, ensure_outdir, save_solution_snapshot
from hydra.grid import Grid3D
from hydra.equations import EulerEquations3D
from hydra.initial_conditions import turbulent_velocity_3d
from hydra.solver import SpectralSolver3D

# Import and register the custom colormap
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from colormaps import register
register('cmapkk9')


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 3D turbulent velocity field with SpectralHydro")
    parser.add_argument("--config", type=str, default="configs/turb3d.yaml")
    parser.add_argument("--backend", type=str, default="numpy", choices=["numpy", "torch"], help="Array backend")
    parser.add_argument("--device", type=str, default=None, help="Torch device: cpu|mps|cuda (if backend=torch)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = os.path.abspath(cfg["io"]["outdir"]) if "io" in cfg else os.path.abspath("outputs/turb3d")
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
    p0 = float(icfg.get("p0", 1.0))
    vrms = float(icfg.get("vrms", 0.1))
    kmin = float(icfg.get("kmin", 2.0))
    kmax = float(icfg.get("kmax", 16.0))
    alpha = float(icfg.get("alpha", 0.3333))
    spectrum_type = str(icfg.get("spectrum_type", "parabolic"))
    power_law_slope = float(icfg.get("power_law_slope", -5.0/3.0))
    seed = int(icfg.get("seed", 42))

    U0_state = turbulent_velocity_3d(X, Y, Z, rho0=rho0, p0=p0, vrms=vrms, kmin=kmin, kmax=kmax, 
                                    alpha=alpha, spectrum_type=spectrum_type, power_law_slope=power_law_slope, 
                                    seed=seed, gamma=gamma)

    solver = SpectralSolver3D(grid=grid, equations=eqs, scheme=scheme, cfl=cfl)

    frames_dir = os.path.join(outdir, "frames")
    ensure_outdir(frames_dir)

    # Render mid-plane density slices during run
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
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
        im = ax.imshow(rho[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', cmap='cmapkk9')
        ax.set_title(f"Turbulent Density (z=mid) t={t:.3f}\nvrms={vrms:.3f}, α={alpha:.3f}")
        ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.canvas.draw(); fig.canvas.flush_events()
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=120)
        snapshot_path = save_solution_snapshot(outdir, t, U=U, grid=grid, equations=eqs)
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

    solver.run(U0_state, t0=t0, t_end=t_end, output_interval=output_interval, outdir=outdir, on_output=render)
    plt.close(fig)

    # Final analysis plots
    rho, ux, uy, uz, p = eqs.primitive(solver.U)
    try:
        import torch  # type: ignore
        if isinstance(rho, (torch.Tensor,)):
            rho = rho.detach().cpu().numpy()
            ux = ux.detach().cpu().numpy()
            uy = uy.detach().cpu().numpy()
            uz = uz.detach().cpu().numpy()
            p = p.detach().cpu().numpy()
    except Exception:
        pass

    # Create comprehensive analysis plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
    
    mid = rho.shape[0] // 2
    
    # Density
    im1 = axes[0, 0].imshow(rho[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', cmap='cmapkk9')
    axes[0, 0].set_title(f'Density (z=mid)\nt={solver.t:.3f}')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    # Velocity magnitude
    v_mag = np.sqrt(ux**2 + uy**2 + uz**2)
    im2 = axes[0, 1].imshow(v_mag[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', cmap='cmapkk9')
    axes[0, 1].set_title(f'Velocity Magnitude (z=mid)\nt={solver.t:.3f}')
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Pressure
    im3 = axes[0, 2].imshow(p[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', cmap='cmapkk9')
    axes[0, 2].set_title(f'Pressure (z=mid)\nt={solver.t:.3f}')
    axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # Velocity components
    im4 = axes[1, 0].imshow(ux[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', cmap='RdBu_r')
    axes[1, 0].set_title(f'ux (z=mid)\nt={solver.t:.3f}')
    axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    
    im5 = axes[1, 1].imshow(uy[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', cmap='RdBu_r')
    axes[1, 1].set_title(f'uy (z=mid)\nt={solver.t:.3f}')
    axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    im6 = axes[1, 2].imshow(uz[mid], origin='lower', extent=[0, Lx, 0, Ly], aspect='equal', cmap='RdBu_r')
    axes[1, 2].set_title(f'uz (z=mid)\nt={solver.t:.3f}')
    axes[1, 2].set_xlabel('x'); axes[1, 2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    plt.suptitle(f'3D Turbulent Velocity Field (GPU/MPS)\nvrms={vrms:.3f}, α={alpha:.3f}, k=[{kmin:.1f},{kmax:.1f}]', 
                 fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(outdir, f"turb3d_analysis_t{solver.t:.3f}.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Generate video from frames
    fps = int(cfg.get("video", {}).get("fps", 30))
    codec = str(cfg.get("video", {}).get("codec", "h264_videotoolbox"))
    crf = int(cfg.get("video", {}).get("crf", 18))
    pix_fmt = str(cfg.get("video", {}).get("pix_fmt", "yuv420p"))
    video_path = os.path.join(outdir, "turb3d.mp4")
    
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
