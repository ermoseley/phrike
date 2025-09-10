import argparse
import os
import shutil
import subprocess

import numpy as np
import matplotlib.pyplot as plt

from spectralhydro.io import load_config, ensure_outdir
from spectralhydro.grid import Grid1D
from spectralhydro.equations import EulerEquations1D
from spectralhydro.solver import SpectralSolver1D
from spectralhydro.acoustic import AcousticParams, two_modulated_gaussian_acoustic_ic


def main() -> None:
    parser = argparse.ArgumentParser(description="Modulated Gaussian packet collision and video output")
    parser.add_argument("--config", type=str, default="configs/gaussian_modulated.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    outdir = os.path.abspath(cfg["io"]["outdir"]) if "io" in cfg else os.path.abspath("outputs/gaussian_modulated")
    ensure_outdir(outdir)
    frames_dir = os.path.join(outdir, "frames")
    ensure_outdir(frames_dir)

    N = int(cfg["grid"]["N"])
    Lx = float(cfg["grid"]["Lx"]) 
    dealias = bool(cfg["grid"].get("dealias", True))
    gamma = float(cfg["physics"]["gamma"]) 

    grid = Grid1D(N=N, Lx=Lx, dealias=dealias, filter_params=cfg["integration"].get("spectral_filter", {"enabled": False}))
    eqs = EulerEquations1D(gamma=gamma)

    acfg = cfg["initial_conditions"]["acoustic"]
    params = AcousticParams(
        rho0=float(acfg.get("rho0", 1.0)),
        p0=float(acfg.get("p0", 1.0)),
        u0=float(acfg.get("u0", 0.0)),
        amplitude=float(acfg.get("amplitude", 1e-4)),
        Lx=Lx,
        direction="right",
        fractional=bool(acfg.get("fractional", True)),
        gamma=gamma,
    )
    gcfg = cfg["initial_conditions"]["gaussians"]
    center_left = float(gcfg.get("center_left", 0.25))
    center_right = float(gcfg.get("center_right", 0.75))
    sigma = float(gcfg.get("sigma", 0.02))
    carrier_m = int(gcfg.get("carrier_m", 64))
    phase_left = float(gcfg.get("phase_left", 0.0))
    phase_right = float(gcfg.get("phase_right", 0.0))
    same_direction = gcfg.get("same_direction", None)
    amp_left_factor = float(gcfg.get("amp_left_factor", 1.0))
    amp_right_factor = float(gcfg.get("amp_right_factor", 1.0))

    U0 = two_modulated_gaussian_acoustic_ic(
        grid.x, params,
        center_left=center_left, center_right=center_right,
        sigma=sigma, carrier_m=carrier_m,
        phase_left=phase_left, phase_right=phase_right,
        amp_left_factor=amp_left_factor, amp_right_factor=amp_right_factor,
        same_direction=same_direction)

    t0 = float(cfg["integration"]["t0"]) 
    t_end = float(cfg["integration"]["t_end"]) 
    cfl = float(cfg["integration"]["cfl"]) 
    scheme = str(cfg["integration"].get("scheme", "rk4"))
    output_interval = float(cfg["integration"].get("output_interval", 0.0025))

    # Plotting bounds
    c = np.sqrt(params.gamma * params.p0 / params.rho0)
    dr = abs(params.amplitude * (params.rho0 if params.fractional else 1.0)) * 1.5
    du = c * abs(params.amplitude if params.fractional else params.amplitude / params.rho0) * 1.5
    dp = (c * c) * abs(params.amplitude * (params.rho0 if params.fractional else 1.0)) * 1.5

    fig, axs = plt.subplots(3, 1, figsize=(10, 8), constrained_layout=True)

    def render_frame(t: float, U):
        rho, u, p, _ = eqs.primitive(U)
        axs[0].cla(); axs[1].cla(); axs[2].cla()
        axs[0].plot(grid.x, rho)
        axs[0].set_ylabel("rho")
        axs[0].set_ylim(params.rho0 - dr, params.rho0 + dr)
        axs[1].plot(grid.x, u)
        axs[1].set_ylabel("u")
        axs[1].set_ylim(-du, du)
        axs[2].plot(grid.x, p)
        axs[2].set_ylabel("p")
        axs[2].set_ylim(params.p0 - dp, params.p0 + dp)
        for ax in axs:
            ax.set_xlabel("x"); ax.grid(True, alpha=0.3)
        fig.suptitle(f"Modulated Gaussian collision t={t:.3f}")
        frame_path = os.path.join(frames_dir, f"frame_{t:08.3f}.png")
        fig.savefig(frame_path, dpi=150)

    solver = SpectralSolver1D(grid=grid, equations=eqs, scheme=scheme, cfl=cfl)
    solver.run(U0, t0=t0, t_end=t_end, output_interval=output_interval, on_output=render_frame)
    plt.close(fig)

    # Build video
    fps = int(cfg["video"].get("fps", 30))
    codec = str(cfg["video"].get("codec", "h264_videotoolbox"))
    crf = int(cfg["video"].get("crf", 18))
    pix_fmt = str(cfg["video"].get("pix_fmt", "yuv420p"))
    video_path = os.path.join(outdir, "gaussian_modulated.mp4")

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


