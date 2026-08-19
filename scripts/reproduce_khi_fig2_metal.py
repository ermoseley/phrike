#!/usr/bin/env python3
"""Metal/Phrike analogue of Figure 2 in Velasco-Romero & Teyssier (2026).

The paper's FV2 implementation is not available on Metal.  This driver keeps
the published smooth KHI initial condition and Navier--Stokes/dye equations,
but advances them with Phrike's periodic Fourier representation on PyTorch MPS.
It is therefore a same-physics Metal comparison, not a reproduction of the
paper's spatial discretisation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


GAMMA = 5.0 / 3.0
NU = 2.0e-5
CHI = 2.0e-5
D_DYE = 2.0e-5
LX = 1.0
LY = 2.0
NUMERICS_VERSION = "hard-two-thirds-v1"


def initial_condition(nx: int, device: torch.device) -> torch.Tensor:
    """Return (rho, rho*u, rho*v, E, rho*c) on Fourier nodes."""
    ny = 2 * nx
    dtype = torch.float32
    x = torch.arange(nx, dtype=dtype, device=device) * (LX / nx)
    y = torch.arange(ny, dtype=dtype, device=device) * (LY / ny)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    a = 0.05
    sigma = 0.2
    amplitude = 0.01
    u_flow = 1.0
    y1, y2 = 0.5, 1.5

    rho = torch.ones_like(xx)
    ux = u_flow * (
        torch.tanh((yy - y1) / a) - torch.tanh((yy - y2) / a) - 1.0
    )
    uy = amplitude * torch.sin(2.0 * math.pi * xx) * (
        torch.exp(-((yy - y1) ** 2) / sigma**2)
        + torch.exp(-((yy - y2) ** 2) / sigma**2)
    )
    pressure = torch.full_like(xx, 10.0)
    dye = 0.5 * (
        torch.tanh((yy - y2) / a) - torch.tanh((yy - y1) / a) + 2.0
    )
    energy = pressure / (GAMMA - 1.0) + 0.5 * rho * (ux**2 + uy**2)
    return torch.stack((rho, rho * ux, rho * uy, energy, rho * dye))


class SpectralNavierStokesDye:
    def __init__(self, nx: int, device: torch.device):
        self.nx = nx
        self.ny = 2 * nx
        self.device = device
        dtype = torch.float32

        kx = 2.0 * math.pi * torch.fft.fftfreq(
            self.nx, d=LX / self.nx, device=device
        )
        ky = 2.0 * math.pi * torch.fft.fftfreq(
            self.ny, d=LY / self.ny, device=device
        )
        self.ikx = (1j * kx).reshape(1, 1, self.nx)
        self.iky = (1j * ky).reshape(1, self.ny, 1)

        mx = torch.fft.fftfreq(self.nx, device=device) * self.nx
        my = torch.fft.fftfreq(self.ny, device=device) * self.ny
        dealias_x = torch.abs(mx) <= self.nx // 3
        dealias_y = torch.abs(my) <= self.ny // 3
        dealias = dealias_y[:, None] & dealias_x[None, :]
        self.dealias_mask = dealias.to(dtype).reshape(1, self.ny, self.nx)

    def _fft(self, fields: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(fields, dim=(-2, -1)) * self.dealias_mask

    @staticmethod
    def primitives(U: torch.Tensor):
        rho = torch.clamp(U[0], min=1.0e-6)
        ux = U[1] / rho
        uy = U[2] / rho
        kinetic = 0.5 * rho * (ux**2 + uy**2)
        pressure = torch.clamp((GAMMA - 1.0) * (U[3] - kinetic), min=1.0e-6)
        dye = U[4] / rho
        return rho, ux, uy, pressure, dye

    def rhs(self, U: torch.Tensor) -> torch.Tensor:
        rho, ux, uy, pressure, dye = self.primitives(U)
        energy = U[3]

        flux_x = torch.stack(
            (
                rho * ux,
                rho * ux**2 + pressure,
                rho * ux * uy,
                (energy + pressure) * ux,
                rho * dye * ux,
            )
        )
        flux_y = torch.stack(
            (
                rho * uy,
                rho * ux * uy,
                rho * uy**2 + pressure,
                (energy + pressure) * uy,
                rho * dye * uy,
            )
        )
        adv_hat = self._fft(torch.cat((flux_x, flux_y)))
        adv_div = torch.fft.ifft2(
            self.ikx * adv_hat[:5] + self.iky * adv_hat[5:], dim=(-2, -1)
        ).real

        temperature = pressure / rho
        primitive_hat = self._fft(torch.stack((ux, uy, temperature, dye)))
        dx = torch.fft.ifft2(self.ikx * primitive_hat, dim=(-2, -1)).real
        dy = torch.fft.ifft2(self.iky * primitive_hat, dim=(-2, -1)).real
        dux_dx, duy_dx, dtemp_dx, ddye_dx = dx
        dux_dy, duy_dy, dtemp_dy, ddye_dy = dy

        div_u = dux_dx + duy_dy
        mu = rho * NU
        tau_xx = mu * (2.0 * dux_dx - (2.0 / 3.0) * div_u)
        tau_yy = mu * (2.0 * duy_dy - (2.0 / 3.0) * div_u)
        tau_xy = mu * (dux_dy + duy_dx)

        zero = torch.zeros_like(rho)
        diff_x = torch.stack(
            (
                zero,
                tau_xx,
                tau_xy,
                ux * tau_xx + uy * tau_xy + rho * CHI * dtemp_dx,
                rho * D_DYE * ddye_dx,
            )
        )
        diff_y = torch.stack(
            (
                zero,
                tau_xy,
                tau_yy,
                ux * tau_xy + uy * tau_yy + rho * CHI * dtemp_dy,
                rho * D_DYE * ddye_dy,
            )
        )
        diff_hat = self._fft(torch.cat((diff_x, diff_y)))
        diff_div = torch.fft.ifft2(
            self.ikx * diff_hat[:5] + self.iky * diff_hat[5:], dim=(-2, -1)
        ).real
        return -adv_div + diff_div

    def project(self, U: torch.Tensor) -> torch.Tensor:
        return torch.fft.ifft2(self._fft(U), dim=(-2, -1)).real

    def step_ssprk3(self, U: torch.Tensor, dt: float) -> torch.Tensor:
        U1 = U + dt * self.rhs(U)
        U2 = 0.75 * U + 0.25 * (U1 + dt * self.rhs(U1))
        return self.project(
            (1.0 / 3.0) * U + (2.0 / 3.0) * (U2 + dt * self.rhs(U2))
        )

    def timestep(self, U: torch.Tensor, cfl: float) -> float:
        rho, ux, uy, pressure, _ = self.primitives(U)
        sound = torch.sqrt(GAMMA * pressure / rho)
        speed_x = float(torch.max(torch.abs(ux) + sound).item())
        speed_y = float(torch.max(torch.abs(uy) + sound).item())
        return cfl * min((LX / self.nx) / speed_x, (LY / self.ny) / speed_y)


def save_result(
    outdir: Path,
    nx: int,
    t: float,
    U: torch.Tensor,
    elapsed: float,
    steps: int,
    cfl: float,
) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    rho, ux, uy, pressure, dye = SpectralNavierStokesDye.primitives(U)
    arrays = {
        "rho": rho.detach().cpu().numpy(),
        "ux": ux.detach().cpu().numpy(),
        "uy": uy.detach().cpu().numpy(),
        "pressure": pressure.detach().cpu().numpy(),
        "dye": dye.detach().cpu().numpy(),
    }
    np.savez_compressed(outdir / f"khi_fig2_metal_N{nx:04d}_t{t:.3f}.npz", **arrays)
    metrics = {
        "Nx": nx,
        "Ny": 2 * nx,
        "t": t,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "backend": "torch-mps",
        "precision": "float32",
        "method": "Phrike periodic Fourier pseudo-spectral, SSPRK3, hard 2/3 dealiasing",
        "numerics_version": NUMERICS_VERSION,
        "gamma": GAMMA,
        "Re": 1.0e5,
        "nu": NU,
        "chi": CHI,
        "dye_diffusivity": D_DYE,
        "cfl": cfl,
        "mass_mean": float(arrays["rho"].mean()),
        "dye_mass_mean": float((arrays["rho"] * arrays["dye"]).mean()),
        "rho_min": float(arrays["rho"].min()),
        "rho_max": float(arrays["rho"].max()),
        "pressure_min": float(arrays["pressure"].min()),
        "pressure_max": float(arrays["pressure"].max()),
        "dye_min": float(arrays["dye"].min()),
        "dye_max": float(arrays["dye"].max()),
    }
    (outdir / f"khi_fig2_metal_N{nx:04d}_t{t:.3f}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def plot_comparison(outdir: Path, resolutions: list[int], t: float) -> Path:
    npanels = len(resolutions)
    fig_width, fig_height = 3.1 * npanels, 6.1
    fig = plt.figure(figsize=(fig_width, fig_height))
    axes_height = 0.68
    axes_bottom = 0.20
    panel_width = (fig_height / fig_width) * axes_height / 2.0
    panels_width = npanels * panel_width
    panels_left = 0.5 * (1.0 - panels_width)
    axes = []
    for index in range(npanels):
        axes.append(
            fig.add_axes(
                [
                    panels_left + index * panel_width,
                    axes_bottom,
                    panel_width,
                    axes_height,
                ],
                sharey=axes[0] if axes else None,
            )
        )
    image = None
    for ax, nx in zip(axes, resolutions):
        data = np.load(outdir / f"khi_fig2_metal_N{nx:04d}_t{t:.3f}.npz")
        image = ax.imshow(
            data["dye"],
            origin="lower",
            extent=(0.0, LX, 0.0, LY),
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_title(rf"$N_{{\rm DOF,x}}={nx}$")
        ax.set_xlabel("x")
        if ax is axes[0]:
            ax.set_ylabel("y")
        else:
            ax.tick_params(axis="y", which="both", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)
            ax.set_xticks(np.linspace(0.2, 1.0, 5))
    assert image is not None
    colorbar_ax = fig.add_axes(
        [panels_left, axes_bottom - 0.105, panels_width, 0.025]
    )
    fig.colorbar(image, cax=colorbar_ax, orientation="horizontal", label="dye c")
    fig.suptitle(rf"Metal/Phrike KHI analogue, $t={t:g}$, $Re=10^5$", y=0.97)
    path = outdir / f"khi_fig2_metal_t{t:.3f}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--t-end", type=float, default=6.0)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/khi_fig2_metal"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("This validation requires an available Metal/MPS device")
    device = torch.device("mps")
    all_metrics = []
    for nx in args.resolutions:
        model = SpectralNavierStokesDye(nx, device)
        checkpoint_path = args.outdir / f"khi_fig2_metal_N{nx:04d}.checkpoint.pt"
        if args.resume and checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if checkpoint.get("numerics_version") != NUMERICS_VERSION:
                raise RuntimeError(
                    f"{checkpoint_path} predates {NUMERICS_VERSION}; restart this "
                    "resolution without --resume"
                )
            U = checkpoint["U"].to(device)
            t_now = float(checkpoint["t"])
            step = int(checkpoint["step"])
            print(f"N={nx} resumed at step={step} t={t_now:.6f}", flush=True)
        else:
            U = model.project(initial_condition(nx, device))
            t_now = 0.0
            step = 0
        next_checkpoint = (
            (math.floor(t_now / args.checkpoint_interval) + 1) * args.checkpoint_interval
            if args.checkpoint_interval > 0.0
            else math.inf
        )
        started = time.perf_counter()
        while t_now < args.t_end:
            dt = min(model.timestep(U, args.cfl), args.t_end - t_now)
            U = model.step_ssprk3(U, dt)
            t_now += dt
            step += 1
            if step % args.progress_every == 0 or t_now >= args.t_end:
                elapsed = time.perf_counter() - started
                print(
                    f"N={nx} step={step} t={t_now:.6f}/{args.t_end:g} "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                rho, _, _, pressure, _ = model.primitives(U)
                if not bool(torch.all(torch.isfinite(U)).item()):
                    raise FloatingPointError(f"non-finite state at N={nx}, t={t_now}")
                if float(torch.min(rho).item()) <= 0.0 or float(torch.min(pressure).item()) <= 0.0:
                    raise FloatingPointError(f"non-positive state at N={nx}, t={t_now}")
            if t_now + 1.0e-12 >= next_checkpoint:
                args.outdir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "U": U.detach().cpu(),
                        "t": t_now,
                        "step": step,
                        "numerics_version": NUMERICS_VERSION,
                    },
                    checkpoint_path,
                )
                next_checkpoint += args.checkpoint_interval
        elapsed = time.perf_counter() - started
        all_metrics.append(
            save_result(args.outdir, nx, t_now, U, elapsed, step, args.cfl)
        )
    plot_path = plot_comparison(args.outdir, args.resolutions, args.t_end)
    (args.outdir / f"khi_fig2_metal_t{args.t_end:.3f}.json").write_text(
        json.dumps(all_metrics, indent=2, sort_keys=True) + "\n"
    )
    print(plot_path)


if __name__ == "__main__":
    main()
