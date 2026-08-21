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
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch


GAMMA = 5.0 / 3.0
NU = 2.0e-5
CHI = 2.0e-5
D_DYE = 2.0e-5
LX = 1.0
LY = 2.0
RHO_FLOOR = 1.0e-6
PRESSURE_FLOOR = 1.0e-6
SPECTRAL_DISSIPATION_ORDER = 2
SPECTRAL_DISSIPATION_ONSET_FRACTION = 0.8
SPECTRAL_DISSIPATION_CUTOFF_CROSSING_FRACTION = 0.5
INITIAL_MAX_SIGNAL_SPEED = 1.0 + math.sqrt(GAMMA * 10.0)
NUMERICS_VERSION = "hard-two-thirds-top-band-spectral-viscosity-v4"
FIELDS = ("rho", "ux", "uy", "pressure", "dye")
DYE_COLORMAP = "RdBu"
DYE_COLOR_LIMITS = (0.0, 1.0)


def cutoff_e_folding_time(nx: int) -> float:
    """Return the cutoff damping time in physical simulation units."""
    return (
        SPECTRAL_DISSIPATION_CUTOFF_CROSSING_FRACTION
        * (LX / nx)
        / INITIAL_MAX_SIGNAL_SPEED
    )


def numerical_signature(nx: int, cfl: float) -> dict:
    """Return the numerical policy that must match when resuming a run."""
    return {
        "version": NUMERICS_VERSION,
        "Nx": nx,
        "Ny": 2 * nx,
        "Lx": LX,
        "Ly": LY,
        "cfl": float(cfl),
        "gamma": GAMMA,
        "nu": NU,
        "chi": CHI,
        "dye_diffusivity": D_DYE,
        "time_integrator": "SSPRK3",
        "dealias": "hard-two-thirds",
        "spectral_dissipation": {
            "enabled": True,
            "order": SPECTRAL_DISSIPATION_ORDER,
            "onset_fraction": SPECTRAL_DISSIPATION_ONSET_FRACTION,
            "e_folding_time_at_cutoff": cutoff_e_folding_time(nx),
            "cutoff_crossing_fraction": (
                SPECTRAL_DISSIPATION_CUTOFF_CROSSING_FRACTION
            ),
            "reference_signal_speed": INITIAL_MAX_SIGNAL_SPEED,
        },
        "rho_floor": RHO_FLOOR,
        "pressure_floor": PRESSURE_FLOOR,
        "backend": "torch-mps",
        "precision": "float32",
    }


def validate_checkpoint_signature(
    checkpoint: dict, expected: dict, checkpoint_path: Path
) -> None:
    """Reject checkpoints made with another numerical policy."""
    if checkpoint.get("numerical_signature") != expected:
        raise RuntimeError(
            f"{checkpoint_path} has a different or obsolete numerical "
            "signature; restart this resolution without --resume"
        )


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

        ix = torch.arange(self.nx, device=device)
        iy = torch.arange(self.ny, device=device)
        mx = torch.where(ix < (self.nx + 1) // 2, ix, ix - self.nx)
        my = torch.where(iy < (self.ny + 1) // 2, iy, iy - self.ny)
        dealias_x = torch.abs(mx) <= (self.nx - 1) // 3
        dealias_y = torch.abs(my) <= (self.ny - 1) // 3
        dealias = dealias_y[:, None] & dealias_x[None, :]
        self.dealias_mask = dealias.to(dtype).reshape(1, self.ny, self.nx)
        cutoff_x = (self.nx - 1) // 3
        cutoff_y = (self.ny - 1) // 3
        eta_x = torch.abs(mx).to(dtype) / max(cutoff_x, 1)
        eta_y = torch.abs(my).to(dtype) / max(cutoff_y, 1)
        ramp_x = torch.clamp(
            (eta_x - SPECTRAL_DISSIPATION_ONSET_FRACTION)
            / (1.0 - SPECTRAL_DISSIPATION_ONSET_FRACTION),
            min=0.0,
        )
        ramp_y = torch.clamp(
            (eta_y - SPECTRAL_DISSIPATION_ONSET_FRACTION)
            / (1.0 - SPECTRAL_DISSIPATION_ONSET_FRACTION),
            min=0.0,
        )
        cutoff_rate = 1.0 / cutoff_e_folding_time(nx)
        self.dissipation_rate = cutoff_rate * (
            ramp_y[:, None] ** SPECTRAL_DISSIPATION_ORDER
            + ramp_x[None, :] ** SPECTRAL_DISSIPATION_ORDER
        )
        self.dissipation_rate = self.dissipation_rate.reshape(
            1, self.ny, self.nx
        )

    def _fft(self, fields: torch.Tensor) -> torch.Tensor:
        return torch.fft.fft2(fields, dim=(-2, -1)) * self.dealias_mask

    @staticmethod
    def primitives(U: torch.Tensor):
        rho = torch.clamp(U[0], min=RHO_FLOOR)
        ux = U[1] / rho
        uy = U[2] / rho
        kinetic = 0.5 * rho * (ux**2 + uy**2)
        pressure = torch.clamp(
            (GAMMA - 1.0) * (U[3] - kinetic), min=PRESSURE_FLOOR
        )
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

    def dissipate(self, U: torch.Tensor, dt: float) -> torch.Tensor:
        """Apply exact exponential spectral viscosity for elapsed time ``dt``."""
        U_hat = self._fft(U)
        U_hat *= torch.exp(-float(dt) * self.dissipation_rate)
        return torch.fft.ifft2(U_hat, dim=(-2, -1)).real

    def step_ssprk3(self, U: torch.Tensor, dt: float) -> torch.Tensor:
        U_split = self.dissipate(U, 0.5 * dt)
        U1 = U_split + dt * self.rhs(U_split)
        U2 = 0.75 * U_split + 0.25 * (U1 + dt * self.rhs(U1))
        U_next = (1.0 / 3.0) * U_split + (2.0 / 3.0) * (
            U2 + dt * self.rhs(U2)
        )
        return self.dissipate(U_next, 0.5 * dt)

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
        "method": (
            "Phrike periodic Fourier pseudo-spectral, SSPRK3, hard 2/3 "
            "dealiasing, timestep-aware top-band spectral viscosity"
        ),
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
        "numerical_signature": numerical_signature(nx, cfl),
    }
    (outdir / f"khi_fig2_metal_N{nx:04d}_t{t:.3f}.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n"
    )
    return metrics


def restrict_to_common_modes(field: np.ndarray, target_nx: int) -> np.ndarray:
    """Evaluate a field on the target grid after a common 2/3 truncation."""
    source_ny, source_nx = field.shape
    target_ny = 2 * target_nx
    if source_nx % target_nx or source_ny % target_ny:
        raise ValueError(
            "Common-mode comparison requires integer resolution ratios; "
            f"got {source_nx}x{source_ny} and {target_nx}x{target_ny}"
        )
    mx = np.fft.fftfreq(source_nx) * source_nx
    my = np.fft.fftfreq(source_ny) * source_ny
    mask = (
        (np.abs(my) <= (target_ny - 1) // 3)[:, None]
        & (np.abs(mx) <= (target_nx - 1) // 3)[None, :]
    )
    restricted = np.fft.ifft2(np.fft.fft2(field) * mask).real
    return restricted[:: source_ny // target_ny, :: source_nx // target_nx].copy()


def common_mode_error(reference: np.ndarray, candidate: np.ndarray) -> dict:
    difference = candidate - reference
    rms = float(np.sqrt(np.mean(difference**2)))
    reference_rms = float(np.sqrt(np.mean(reference**2)))
    return {
        "rms": rms,
        "relative_rms": rms / max(reference_rms, np.finfo(float).tiny),
        "max": float(np.max(np.abs(difference))),
    }


def common_mode_report(
    outdir: Path, resolutions: list[int], t: float, target_nx: Optional[int] = None
) -> dict:
    """Compare saved solutions after restriction to one common Fourier space."""
    ordered = sorted(set(resolutions))
    if not ordered:
        raise ValueError("At least one resolution is required")
    target_nx = ordered[0] if target_nx is None else target_nx
    restricted: dict[int, dict[str, np.ndarray]] = {}
    for nx in ordered:
        path = outdir / f"khi_fig2_metal_N{nx:04d}_t{t:.3f}.npz"
        with np.load(path) as data:
            restricted[nx] = {
                field: restrict_to_common_modes(data[field], target_nx)
                for field in FIELDS
            }

    comparisons = []
    for coarse, fine in zip(ordered[:-1], ordered[1:]):
        comparisons.append(
            {
                "coarse_Nx": coarse,
                "fine_Nx": fine,
                "errors": {
                    field: common_mode_error(
                        restricted[coarse][field], restricted[fine][field]
                    )
                    for field in FIELDS
                },
            }
        )
    convergence = []
    for previous, current in zip(comparisons[:-1], comparisons[1:]):
        resolution_ratio = current["fine_Nx"] / previous["fine_Nx"]
        convergence.append(
            {
                "from": [previous["coarse_Nx"], previous["fine_Nx"]],
                "to": [current["coarse_Nx"], current["fine_Nx"]],
                "observed_orders": {
                    field: math.log(
                        max(
                            previous["errors"][field]["rms"],
                            np.finfo(float).tiny,
                        )
                        / max(
                            current["errors"][field]["rms"],
                            np.finfo(float).tiny,
                        )
                    )
                    / math.log(resolution_ratio)
                    for field in FIELDS
                },
            }
        )
    return {
        "target_Nx": target_nx,
        "target_Ny": 2 * target_nx,
        "mode_cutoff_x": (target_nx - 1) // 3,
        "mode_cutoff_y": (2 * target_nx - 1) // 3,
        "comparisons": comparisons,
        "successive_error_reduction": convergence,
    }


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
            cmap=DYE_COLORMAP,
            vmin=DYE_COLOR_LIMITS[0],
            vmax=DYE_COLOR_LIMITS[1],
            interpolation="nearest",
            aspect="equal",
        )
        ax.set_xlim(0.0, LX)
        ax.set_ylim(0.0, LY)
        ax.margins(0.0)
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
    fig.colorbar(image, cax=colorbar_ax, orientation="horizontal", label="dye C")
    fig.suptitle(rf"Metal/Phrike KHI analogue, $t={t:g}$, $Re=10^5$", y=0.97)
    path = outdir / f"khi_fig2_metal_t{t:.3f}.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return path


def run_resolution(
    outdir: Path,
    nx: int,
    t_end: float,
    cfl: float,
    progress_every: int,
    checkpoint_interval: float,
    resume: bool,
    device: torch.device,
) -> dict:
    model = SpectralNavierStokesDye(nx, device)
    signature = numerical_signature(nx, cfl)
    checkpoint_path = outdir / f"khi_fig2_metal_N{nx:04d}.checkpoint.pt"
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        validate_checkpoint_signature(checkpoint, signature, checkpoint_path)
        U = checkpoint["U"].to(device)
        t_now = float(checkpoint["t"])
        step = int(checkpoint["step"])
        if t_now > t_end + 1.0e-12:
            raise RuntimeError(
                f"{checkpoint_path} is at t={t_now:g}, beyond requested t={t_end:g}"
            )
        print(f"N={nx} resumed at step={step} t={t_now:.6f}", flush=True)
    else:
        U = model.project(initial_condition(nx, device))
        t_now = 0.0
        step = 0

    next_checkpoint = (
        (math.floor(t_now / checkpoint_interval) + 1) * checkpoint_interval
        if checkpoint_interval > 0.0
        else math.inf
    )
    started = time.perf_counter()
    while t_now < t_end - 1.0e-12:
        dt = min(model.timestep(U, cfl), t_end - t_now)
        U = model.step_ssprk3(U, dt)
        t_now += dt
        step += 1
        if step % progress_every == 0 or t_now >= t_end - 1.0e-12:
            elapsed = time.perf_counter() - started
            print(
                f"N={nx} step={step} t={t_now:.6f}/{t_end:g} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
            rho, _, _, pressure, _ = model.primitives(U)
            if not bool(torch.all(torch.isfinite(U)).item()):
                raise FloatingPointError(f"non-finite state at N={nx}, t={t_now}")
            if (
                float(torch.min(rho).item()) <= 0.0
                or float(torch.min(pressure).item()) <= 0.0
            ):
                raise FloatingPointError(f"non-positive state at N={nx}, t={t_now}")
        if t_now + 1.0e-12 >= next_checkpoint:
            outdir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "U": U.detach().cpu(),
                    "t": t_now,
                    "step": step,
                    "numerical_signature": signature,
                },
                checkpoint_path,
            )
            next_checkpoint += checkpoint_interval
    elapsed = time.perf_counter() - started
    return save_result(outdir, nx, t_now, U, elapsed, step, cfl)


def temporal_refinement_report(
    baseline_dir: Path,
    refined_dir: Path,
    nx: int,
    target_nx: int,
    t: float,
    baseline_cfl: float,
    refined_cfl: float,
) -> dict:
    baseline_path = baseline_dir / f"khi_fig2_metal_N{nx:04d}_t{t:.3f}.npz"
    refined_path = refined_dir / f"khi_fig2_metal_N{nx:04d}_t{t:.3f}.npz"
    with np.load(baseline_path) as baseline, np.load(refined_path) as refined:
        errors = {
            field: common_mode_error(
                restrict_to_common_modes(baseline[field], target_nx),
                restrict_to_common_modes(refined[field], target_nx),
            )
            for field in FIELDS
        }
    return {
        "Nx": nx,
        "target_Nx": target_nx,
        "baseline_cfl": baseline_cfl,
        "refined_cfl": refined_cfl,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resolutions", nargs="+", type=int, default=[256, 512, 1024])
    parser.add_argument("--t-end", type=float, default=6.0)
    parser.add_argument("--cfl", type=float, default=0.7)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=float, default=0.5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--temporal-refinement", action="store_true")
    parser.add_argument("--temporal-resolution", type=int, default=512)
    parser.add_argument("--temporal-cfl-factor", type=float, default=0.5)
    parser.add_argument("--common-resolution", type=int)
    parser.add_argument("--outdir", type=Path, default=Path("outputs/khi_fig2_metal"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.backends.mps.is_available():
        raise RuntimeError("This validation requires an available Metal/MPS device")
    resolutions = sorted(set(args.resolutions))
    if not resolutions or any(nx <= 0 for nx in resolutions):
        raise ValueError("resolutions must contain positive integers")
    target_nx = args.common_resolution or resolutions[0]
    if target_nx <= 0 or any(nx % target_nx for nx in resolutions):
        raise ValueError("common resolution must divide every requested resolution")
    if args.progress_every <= 0:
        raise ValueError("progress-every must be positive")
    if args.cfl <= 0.0:
        raise ValueError("cfl must be positive")
    if not 0.0 < args.temporal_cfl_factor < 1.0:
        raise ValueError("temporal-cfl-factor must lie strictly between zero and one")

    device = torch.device("mps")
    all_metrics = [
        run_resolution(
            args.outdir,
            nx,
            args.t_end,
            args.cfl,
            args.progress_every,
            args.checkpoint_interval,
            args.resume,
            device,
        )
        for nx in resolutions
    ]
    comparison = common_mode_report(
        args.outdir, resolutions, args.t_end, target_nx=target_nx
    )
    report = {"runs": all_metrics, "common_mode": comparison}

    if args.temporal_refinement:
        if args.temporal_resolution not in resolutions:
            raise ValueError("temporal-resolution must be one of --resolutions")
        refined_cfl = args.cfl * args.temporal_cfl_factor
        refined_dir = args.outdir / "temporal_refinement"
        refined_metrics = run_resolution(
            refined_dir,
            args.temporal_resolution,
            args.t_end,
            refined_cfl,
            args.progress_every,
            args.checkpoint_interval,
            args.resume,
            device,
        )
        temporal_comparison = temporal_refinement_report(
            args.outdir,
            refined_dir,
            args.temporal_resolution,
            target_nx,
            args.t_end,
            args.cfl,
            refined_cfl,
        )
        incoming_spatial = next(
            (
                item
                for item in comparison["comparisons"]
                if item["fine_Nx"] == args.temporal_resolution
            ),
            None,
        )
        if incoming_spatial is not None:
            temporal_comparison["relative_to_incoming_spatial_error"] = {
                field: {
                    "rms_ratio": temporal_comparison["errors"][field]["rms"]
                    / max(
                        incoming_spatial["errors"][field]["rms"],
                        np.finfo(float).tiny,
                    ),
                    "temporal_error_is_smaller": temporal_comparison["errors"][field][
                        "rms"
                    ]
                    < incoming_spatial["errors"][field]["rms"],
                }
                for field in FIELDS
            }
        report["temporal_refinement"] = {
            "run": refined_metrics,
            "comparison": temporal_comparison,
        }

    plot_path = plot_comparison(args.outdir, resolutions, args.t_end)
    (args.outdir / f"khi_fig2_metal_t{args.t_end:.3f}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(plot_path)


if __name__ == "__main__":
    main()
