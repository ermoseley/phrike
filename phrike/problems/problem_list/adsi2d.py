"""Single-layer ADSI: 2.5D MHD, periodic x and open transverse y."""

import json
import os
from functools import lru_cache
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
from scipy.sparse.linalg import LinearOperator, eigs

from phrike.equations import MHDEquations2D
from phrike.grid import Grid2D
from phrike.solver import SpectralSolverMHD2D
from ..base import BaseProblem


class NoGrowingModeError(RuntimeError):
    """The linear operator has no growing eigenmode to seed or compare with."""


@lru_cache(maxsize=4)
def adsi_reference(points=257, Lx=1.0, Ly=2.0, width=0.05, cs=10.0,
                   alpha=1.0, velocity=0.5, field=1.0 / np.sqrt(4.0 * np.pi)):
    """Compressible eigenmode on a padded, periodic perturbation domain.

    The background is U_z=velocity*tanh(y/width), B=(field,0,field),
    rho=1. With d/dz=0 the linear operator depends only on U', so its
    perturbations can be periodic even though the background shear is open.
    q rows are vx,vy,vz,bx,by,bz,gas pressure, normalized to max|by|=1.
    This closure approximates open boundaries when the mode tails are small.
    """
    if points < 33 or points % 2 != 1:
        raise ValueError("ADSI reference_points must be odd and at least 33")
    if min(Lx, Ly, width, cs, alpha) <= 0.0:
        raise ValueError("ADSI reference lengths, sound speed and alpha must be positive")
    y = -0.5 * Ly + np.arange(points) * Ly / points
    ky = 2.0 * np.pi * np.fft.fftfreq(points, d=Ly / points)
    I = np.eye(points)
    D = np.fft.ifft(1j * ky[:, None] * np.fft.fft(I, axis=0), axis=0).real
    Z = np.zeros_like(I)
    k = 2.0 * np.pi / Lx
    ik = 1j * k
    Up = velocity / width / np.cosh(y / width) ** 2
    div = np.hstack([ik * I, D, Z])
    grad = np.vstack([ik * I, D, Z])
    curl = np.block([[Z, Z, D], [Z, Z, -ik * I], [-D, ik * I, Z]])
    direction = np.array([1.0, 0.0, 1.0]) / np.sqrt(2.0)
    perpendicular = np.kron(np.eye(3) - np.outer(direction, direction), I)
    magnetic_pressure = np.hstack([field * I, Z, field * I])
    B = np.vstack([field * I, Z, field * I])
    nv = 3 * points
    A = np.zeros((7 * points, 7 * points), dtype=complex)
    A[2 * points:3 * points, points:2 * points] = -np.diag(Up)
    A[:nv, nv:2 * nv] = ik * field * np.eye(nv) - grad @ magnetic_pressure
    A[:nv, 2 * nv:] = -grad
    A[nv:2 * nv, :nv] = ik * field * np.eye(nv) - B @ div
    A[nv:2 * nv, nv:2 * nv] = -alpha * 2.0 * field ** 2 * (curl @ perpendicular @ curl)
    A[5 * points:6 * points, 4 * points:5 * points] += np.diag(Up)
    A[2 * nv:, :nv] = -cs ** 2 * div
    # Shift above the shear-growth bound to select the leading branch.
    shift = max(float(np.max(np.abs(Up))) / 2.0, 0.5)
    lu, piv = linalg.lu_factor(A - shift * np.eye(7 * points))
    inverse = LinearOperator(A.shape, matvec=lambda x: linalg.lu_solve((lu, piv), x),
                             dtype=complex)
    rng = np.random.default_rng(0)
    values, vectors = eigs(A, k=6, sigma=shift, OPinv=inverse, tol=1e-11,
                           v0=rng.normal(size=7 * points))
    chosen = np.argmax(values.real)
    rate = values[chosen]
    q = vectors[:, chosen].reshape(7, points)
    peak = np.argmax(np.abs(q[4]))
    q /= q[4, peak]
    Aq = A @ q.ravel()
    residual = np.linalg.norm(Aq - rate * q.ravel()) / (
        np.linalg.norm(Aq) + abs(rate) * np.linalg.norm(q))
    divergence = ik * q[3] + D @ q[4]
    div_relative = np.linalg.norm(divergence) / (
        k * np.linalg.norm(q[3]) + np.linalg.norm(D @ q[4]))
    # In the ideal limit the selected eigenvalues are round-off zeros, not a
    # growing mode, so test the rate before the (then meaningless) residual.
    if rate.real <= 1e-8:
        raise NoGrowingModeError(
            f"ADSI reference found no growing eigenmode at alpha={alpha:g} "
            f"(leading growth rate {rate.real:.3g}). The instability needs "
            "ambipolar diffusion and does not grow in ideal MHD (alpha -> 0), "
            "so there is no ideal reference eigenvalue; seed an ideal control "
            "with a positive reference_alpha such as 1.")
    if residual > 1e-8 or div_relative > 1e-6:
        raise RuntimeError("ADSI reference eigenmode failed residual/divergence checks")
    metadata = dict(growth=float(rate.real), frequency=float(rate.imag),
                    eigen_residual=float(residual), magnetic_divergence=float(div_relative),
                    points=points, Lx=Lx, Ly=Ly, width=width, cs=cs,
                    alpha=alpha, velocity=velocity, field=field,
                    boundary_closure="periodic perturbations on padded single-layer domain")
    return y, q, metadata


def _interpolate_mode(q, y, Ly):
    """Evaluate the periodic reference by Fourier interpolation."""
    points = q.shape[-1]
    ky = 2.0 * np.pi * np.fft.fftfreq(points, d=Ly / points)
    phase = np.exp(1j * ky[:, None] * (y[None, :] + 0.5 * Ly))
    return (np.fft.fft(q, axis=-1) / points) @ phase


def adsi_mode_amplitude(by):
    """Largest |By_k1|: first x Fourier coefficient of By, maximized over y."""
    by = np.asarray(by, dtype=float)
    return float(np.max(np.abs(np.fft.fft(by, axis=-1)[:, 1])) / by.shape[-1])


def adsi_growth_summary(times, amplitudes, reference_growth=None, ideal=False,
                        min_points=4, max_log_rms=0.02):
    """Amplitude envelope of the seeded mode, and a growth rate only if earned.

    ``amplitudes`` are |By_k1| samples. The envelope (relative to the first
    sample) is always reported. For ambipolar runs, a growth rate is fitted to
    ln(amplitude) over the last half of the samples (at least ``min_points``)
    and reported only if that window is a clean exponential: monotonically
    growing with an rms deviation from the straight line of at most
    ``max_log_rms`` in ln(amplitude). Ideal-MHD runs are controls: ADSI is not
    expected to grow without ambipolar diffusion, so no rate is fitted and any
    amplitude change is a transient of the seed on the discrete grid.
    """
    t = np.asarray(times, dtype=float)
    a = np.asarray(amplitudes, dtype=float)
    relative = a / a[0] if a.size and a[0] > 0.0 else np.full_like(a, np.nan)
    summary = dict(
        ideal_control=bool(ideal), samples=int(a.size),
        t_first=float(t[0]), t_last=float(t[-1]),
        relative_amplitude_min=float(np.min(relative)),
        relative_amplitude_max=float(np.max(relative)),
        relative_amplitude_final=float(relative[-1]),
        t_at_min=float(t[int(np.argmin(relative))]),
        t_at_max=float(t[int(np.argmax(relative))]),
        growth_fit=None, reference_growth=reference_growth)
    if ideal:
        summary["message"] = (
            "Ideal-MHD control: no growing mode is expected. The amplitude "
            "envelope is a transient of the seed and depends on resolution; it "
            "is not a growth rate.")
        return summary
    lo = max(a.size - max(min_points, a.size // 2), 0)
    if a.size < min_points or np.any(a[lo:] <= 0.0):
        summary["message"] = (
            f"Fewer than {min_points} usable samples: amplitude envelope only, "
            "no growth rate.")
        return summary
    tw, lw = t[lo:], np.log(a[lo:])
    slope, intercept = np.polyfit(tw, lw, 1)
    rms = float(np.sqrt(np.mean((lw - (slope * tw + intercept)) ** 2)))
    exponential = slope > 0.0 and rms <= max_log_rms and bool(np.all(np.diff(lw) > 0.0))
    fit = dict(rate=float(slope), log_rms_residual=rms,
               t_start=float(tw[0]), t_end=float(tw[-1]))
    if exponential:
        if reference_growth:
            fit["relative_difference_from_reference"] = float(
                (slope - reference_growth) / reference_growth)
        summary["growth_fit"] = fit
        summary["message"] = (
            f"Exponential phase over t=[{tw[0]:.3g}, {tw[-1]:.3g}]: fitted "
            f"growth rate {slope:.4g}.")
        if reference_growth:
            summary["message"] += (
                f" Reference eigenvalue {reference_growth:.4g} "
                f"(difference {100.0 * fit['relative_difference_from_reference']:+.1f}%; "
                "coarse grids under-resolve the mode).")
    else:
        summary["message"] = (
            "Mode is not in a clean exponential phase over the last half of the "
            f"run (log-fit rms {rms:.3g} > {max_log_rms:g}, or not monotonically "
            "growing): amplitude envelope only, no growth rate.")
        summary["rejected_fit"] = fit
    return summary


class ADSI2DProblem(BaseProblem):
    """Ambipolar diffusion destabilizes an otherwise stable out-of-plane shear.

    Without ambipolar diffusion the shear does not grow in ideal MHD, so an
    ideal run is a control that is expected to stay stable and has no reference
    eigenvalue. ``adsi_growth.json`` in the output directory records the
    |By_k1| amplitude envelope and, for ambipolar runs in an exponential phase,
    a fitted growth rate.
    """

    def __init__(self, *args, **kwargs):
        self._mode_times = []
        self._mode_amplitudes = []
        self._mode_solver = None
        super().__init__(*args, **kwargs)

    def create_grid(self, backend="numpy", device: Optional[str] = None,
                    debug=False):
        cfg = self.config["grid"]
        grid = Grid2D(Nx=int(cfg["Nx"]), Ny=int(cfg["Ny"]),
                      Lx=float(cfg.get("Lx", 1.0)), Ly=float(cfg.get("Ly", 2.0)),
                      basis_x="fourier", basis_y="sbp",
                      bc_config={"left": "periodic", "right": "periodic",
                                 "bottom": "open", "top": "open"},
                      dealias=bool(cfg.get("dealias", True)),
                      filter_params=self.filter_config, fft_workers=self.fft_workers,
                      backend=backend, torch_device=device, precision=self.precision,
                      debug=debug)
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        return grid

    def create_equations(self):
        return MHDEquations2D(gamma=self.gamma)

    def create_initial_conditions(self, grid):
        cfg = self.config.get("initial_conditions", {})
        p0 = float(cfg.get("p0", 60.0))
        width = float(cfg.get("width", 0.05))
        velocity = float(cfg.get("velocity", 0.5))
        field = float(cfg.get("field", 1.0 / np.sqrt(4.0 * np.pi)))
        amplitude = float(cfg.get("seed_amplitude", 3e-4)) * field
        cs = np.sqrt(self.gamma * p0)
        try:
            _, q, metadata = adsi_reference(
                int(cfg.get("reference_points", 257)), grid.Lx, grid.Ly, width,
                cs, float(cfg.get("reference_alpha", 1.0)), velocity, field)
        except NoGrowingModeError as error:
            raise ValueError(
                f"initial_conditions.reference_alpha={cfg.get('reference_alpha')}: "
                f"{error}") from error
        X, Y = grid.xy_mesh()
        x, y = self.convert_torch_to_numpy(grid.x, grid.y)
        mode = _interpolate_mode(q, y - 0.5 * grid.Ly, grid.Ly)
        perturbation = amplitude * np.real(
            mode[:, :, None] * np.exp(2j * np.pi * x[None, None, :] / grid.Lx))
        potential = amplitude * np.real(
            (1j * mode[4] * grid.Lx / (2.0 * np.pi))[:, None]
            * np.exp(2j * np.pi * x[None, :] / grid.Lx))
        if getattr(grid, "_use_torch", False):
            import torch
            perturbation = torch.as_tensor(perturbation, dtype=X.dtype, device=X.device)
            potential = torch.as_tensor(potential, dtype=X.dtype, device=X.device)
            shear = velocity * torch.tanh((Y - 0.5 * grid.Ly) / width)
        else:
            perturbation = perturbation.astype(X.dtype, copy=False)
            potential = potential.astype(X.dtype, copy=False)
            shear = velocity * np.tanh((Y - 0.5 * grid.Ly) / width)
        rho = 1.0 + perturbation[6] / cs ** 2
        pressure = p0 + perturbation[6]
        ux, uy = perturbation[0], perturbation[1]
        uz = shear + perturbation[2]
        Bx, By = field + grid.dy1(potential), -grid.dx1(potential)
        Bz = field + perturbation[5]
        self.reference_metadata = dict(metadata, gamma=self.gamma, p0=p0,
                                       seed_amplitude=float(amplitude / field))
        with open(os.path.join(self.outdir, "adsi_reference.json"), "w") as stream:
            json.dump(self.reference_metadata, stream, indent=2)
        return self.create_equations().conservative(rho, ux, uy, uz, pressure, Bx, By, Bz)

    def _is_ideal(self):
        """True when ambipolar diffusion is off, i.e. the ideal-MHD control."""
        ambipolar = self.config.get("physics", {}).get("mhd", {}).get("ambipolar", {})
        eta = float(ambipolar.get("eta", ambipolar.get("alpha", 1.0)))
        return not (ambipolar.get("enabled", False) and eta > 0.0)

    def _record_mode(self, solver, t, U):
        by = self.convert_torch_to_numpy(solver.equations.primitive(U)[6])[0]
        self._mode_times.append(float(t))
        self._mode_amplitudes.append(adsi_mode_amplitude(by))

    def growth_summary(self):
        """Envelope (and, if earned, a fitted rate) of the run just completed."""
        ideal = self._is_ideal()
        metadata = getattr(self, "reference_metadata", {})
        # The reference eigenvalue belongs to reference_alpha; it applies to the
        # run only when the run's eta is the same coefficient.
        ambipolar = self.config.get("physics", {}).get("mhd", {}).get("ambipolar", {})
        eta = float(ambipolar.get("eta", ambipolar.get("alpha", 1.0)))
        reference = None
        if not ideal and metadata and np.isclose(eta, metadata.get("alpha", np.nan)):
            reference = metadata["growth"]
        return adsi_growth_summary(self._mode_times, self._mode_amplitudes,
                                   reference_growth=reference, ideal=ideal)

    def _plot(self, solver, t, U, path):
        primitive = solver.equations.primitive(U)
        uy, By = self.convert_torch_to_numpy(primitive[2], primitive[6])
        x, y = self.convert_torch_to_numpy(solver.grid.x, solver.grid.y)
        fig, axes = plt.subplots(1, 2, figsize=(8, 7), constrained_layout=True)
        for ax, data, title in zip(axes, (uy, By), ("Transverse velocity", "Transverse field")):
            image = ax.pcolormesh(x, y, data, shading="auto", cmap="RdBu_r")
            ax.set(xlabel="x", ylabel="y", title=f"{title}, t={t:.3f}", aspect="equal")
            fig.colorbar(image, ax=ax, shrink=0.8)
        fig.savefig(path, dpi=int(self.config.get("video", {}).get("frame_dpi", 120)))
        plt.close(fig)

    def create_visualization(self, solver, t, U):
        self._plot(solver, t, U, os.path.join(self.outdir, "frames", f"frame_{t:08.4f}.png"))

    def create_final_visualization(self, solver):
        self._plot(solver, solver.t, solver.U,
                   os.path.join(self.outdir, f"adsi2d_t{solver.t:.3f}.png"))
        if len(self._mode_times) < 2:
            return
        summary = self.growth_summary()
        with open(os.path.join(self.outdir, "adsi_growth.json"), "w") as stream:
            json.dump(summary, stream, indent=2)
        print(f"ADSI |By_k1| relative to t={summary['t_first']:.3g}: "
              f"min {summary['relative_amplitude_min']:.3g} "
              f"(t={summary['t_at_min']:.3g}), max "
              f"{summary['relative_amplitude_max']:.3g} "
              f"(t={summary['t_at_max']:.3g}), final "
              f"{summary['relative_amplitude_final']:.3g}")
        print(summary["message"])

    def on_output(self, solver, t, U):
        # A new solver means a new run: start a fresh amplitude record.
        if self._mode_solver is not solver:
            self._mode_solver = solver
            self._mode_times.clear()
            self._mode_amplitudes.clear()
        self._record_mode(solver, t, U)

    def get_solver_class(self):
        return SpectralSolverMHD2D
