"""1D Hall-MRI: axisymmetric shearing sheet with a uniform vertical field.

Perturbations (u_r, u_phi, b_r, b_phi) ~ exp(i k z + s t) with k parallel to
B0 = B0 z_v decouple from the compressive modes. With v_A^2 = B0^2/rho0, shear
S, rotation Omega and Hall term E_H = eta_H J x B/|B|, the linearization is

  s u_r   = 2 Omega u_phi + i k (B0/rho0) b_r
  s u_phi = -(2 Omega - S) u_r + i k (B0/rho0) b_phi
  s b_r   = i k B0 u_r - h b_phi
  s b_phi = i k B0 u_phi - S b_r + h b_r,       h = eta_H sgn(B0) k^2,

(-curl E_H = (-h b_phi, +h b_r) for phi = z_v x r). The physically meaningful
sign is that of eta_H (Omega . B0). The dispersion relation is
s^4 + s^2 (2 w^2 + kappa^2 + h (h - S)) + w^4 + w^2 (4 Omega h - 2 Omega S - h S)
+ kappa^2 h (h - S) = 0 with w = k v_A, kappa^2 = 2 Omega (2 Omega - S); h = 0
recovers Balbus-Hawley. The inertia regularization replaces h by h/(1 + l^2 k^2).
"""

import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phrike.equations import MHDEquations1D
from phrike.grid import Grid1D
from phrike.solver import SpectralSolverMHD1D, shearing_sheet_axes
from ..base import BaseProblem


def hall_mri_matrix(k, omega, shear, field, rho0=1.0, eta_h=0.0, length=0.0):
    """4x4 matrix acting on (u_r, u_phi, b_r, b_phi) for wavenumber k along B0."""
    h = eta_h * np.sign(field) * k**2 / (1.0 + (length * k) ** 2)
    a, c = 1j * k * field / rho0, 1j * k * field
    return np.array([[0, 2 * omega, a, 0],
                     [-(2 * omega - shear), 0, 0, a],
                     [c, 0, 0, -h],
                     [0, c, h - shear, 0]], dtype=complex)


def hall_mri_mode(k, omega, shear, field, rho0=1.0, eta_h=0.0, length=0.0):
    """Largest-growth eigenvalue and eigenvector, normalized to |(b_r, b_phi)| = 1."""
    values, vectors = np.linalg.eig(
        hall_mri_matrix(k, omega, shear, field, rho0, eta_h, length))
    i = int(np.argmax(values.real))
    return values[i], vectors[:, i] / np.linalg.norm(vectors[2:, i])


def hall_mri_fastest(omega, shear, field, rho0=1.0, eta_h=0.0, length=0.0,
                     kmax=None, n=801):
    """Wavenumber of fastest growth on (0, kmax]; returns (k, s, v).

    The default kmax is 4 max(|Omega|, |S|)/v_A. Growth need not decay at large k
    for one sign of the Hall term, so the optimum is taken inside this range.
    """
    scale = max(abs(omega), abs(shear), 1e-12) / (abs(field) / np.sqrt(rho0))
    kmax = 4.0 * scale if kmax is None else float(kmax)
    args = (omega, shear, field, rho0, eta_h, length)

    def best(ks):
        growth = [hall_mri_mode(k, *args)[0].real for k in ks]
        return int(np.argmax(growth))

    ks = np.linspace(kmax / n, kmax, n)
    i = best(ks)
    ks = np.linspace(ks[max(i - 1, 0)], ks[min(i + 1, n - 1)], n)
    k = ks[best(ks)]
    s, v = hall_mri_mode(k, *args)
    return k, s, v


class HallMRI1DProblem(BaseProblem):
    """Fastest-growing Hall-MRI eigenmode in a 1D periodic box along the vertical axis."""

    def _params(self):
        mhd = self.config.get("physics", {}).get("mhd", {})
        shear = mhd.get("shearing_sheet") or {}
        hall = mhd.get("hall") or {}
        ic = self.config.get("initial_conditions", {})
        omega = float(shear.get("omega", 0.0))
        field, rho0 = float(ic.get("B0", 1.0)), float(ic.get("rho0", 1.0))
        eta_h = length = 0.0
        if hall.get("enabled", False):
            eta_h = float(hall.get("eta_h", 0.0))
            if str(hall.get("eta_h_form", "constant")).lower() != "constant":
                eta_h *= (abs(field) / float(hall.get("b0", 1.0))
                          * float(hall.get("rho0", 1.0)) / rho0)
            if str(hall.get("method", "reference")).lower() == "inertia":
                length = float(hall.get("length", 0.0))
        return dict(omega=omega,
                    shear=(float(shear["shear"]) if "shear" in shear
                           else float(shear.get("q", 1.5)) * omega),
                    field=field, rho0=rho0, eta_h=eta_h, length=length)

    def create_grid(self, backend: str = "numpy", device: Optional[str] = None,
                    debug: bool = False) -> Grid1D:
        cfg = self.config["grid"]
        Lx = cfg.get("Lx", "auto")
        if isinstance(Lx, str):
            # Fit the chosen box mode to the fastest-growing wavenumber.
            if Lx.lower() != "auto":
                raise ValueError("grid.Lx must be a number or 'auto'")
            k, s, _ = hall_mri_fastest(**self._params())
            if s.real <= 0.0:
                raise ValueError("No growing Hall-MRI mode in the scanned wavenumber range")
            Lx = 2.0 * np.pi * int(self.config.get("initial_conditions", {}).get("mode", 1)) / k
        grid = Grid1D(
            N=int(cfg["N"]), Lx=float(Lx), basis="fourier",
            dealias=bool(cfg.get("dealias", True)),
            filter_params=self.filter_config, fft_workers=self.fft_workers,
            backend=backend, torch_device=device, precision=self.precision,
            debug=debug,
        )
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        return grid

    def create_equations(self) -> MHDEquations1D:
        return MHDEquations1D(gamma=self.gamma)

    def _amplitude(self, U) -> float:
        """Amplitude of the fundamental box mode of (b_r, b_phi)."""
        b_r, b_phi = self.convert_torch_to_numpy(U[5 + self._radial], U[5 + self._azimuthal])
        m, n = self._mode, len(b_r)
        return float(2.0 * np.hypot(abs(np.fft.fft(b_r)[m]) / n, abs(np.fft.fft(b_phi)[m]) / n))

    def create_initial_conditions(self, grid: Grid1D):
        ic = self.config.get("initial_conditions", {})
        shear = self.config.get("physics", {}).get("mhd", {}).get("shearing_sheet") or {}
        params = self._params()
        self._mode = int(ic.get("mode", 1))
        self._radial = int(shear.get("radial_axis", 1))
        self._azimuthal, sign = shearing_sheet_axes(0, self._radial)
        k = 2.0 * np.pi * self._mode / grid.Lx
        s, v = hall_mri_mode(k, **params)
        self.linear_mode = dict(k=float(k), growth=float(s.real), frequency=float(s.imag),
                                **params)
        x = self.convert_torch_to_numpy(grid.x)[0]
        wave = (float(ic.get("amplitude", 1e-6)) * abs(params["field"])
                * np.real(v[:, None] * np.exp(1j * k * x)[None, :]))
        u, B = np.zeros((3, len(x))), np.zeros((3, len(x)))
        B[0] = params["field"]
        u[self._radial], u[self._azimuthal] = wave[0], sign * wave[1]
        B[self._radial], B[self._azimuthal] = wave[2], sign * wave[3]
        fields = [np.full_like(x, params["rho0"]), *u,
                  np.full_like(x, 0.5 * float(ic.get("beta", 400.0)) * params["field"] ** 2), *B]
        if getattr(grid, "_use_torch", False):
            import torch

            fields = [torch.as_tensor(a, dtype=grid.x.dtype, device=grid.x.device)
                      for a in fields]
        U = self.create_equations().conservative(*fields)
        self.amplitude_history = [(self.t0, self._amplitude(U))]
        return U

    def output_monitoring_info(self, solver, U, step_count, dt):
        # Requires monitoring.enabled; the history is sampled every step_interval steps.
        self.amplitude_history.append((solver.t, self._amplitude(U)))
        super().output_monitoring_info(solver, U, step_count, dt)

    def create_visualization(self, solver, t: float, U):
        pass

    def create_final_visualization(self, solver) -> None:
        t, amplitude = np.array(self.amplitude_history).T
        window = t >= t[0] + 0.4 * (t[-1] - t[0])
        if window.sum() < 3 or np.any(amplitude[window] <= 0.0):
            return
        measured = float(np.polyfit(t[window], np.log(amplitude[window]), 1)[0])
        linear = self.linear_mode["growth"]
        result = dict(self.linear_mode, measured_growth=measured, ratio=measured / linear,
                      fit_window=[float(t[window][0]), float(t[-1])])
        with open(os.path.join(self.outdir, "hall_mri_growth.json"), "w") as stream:
            json.dump(result, stream, indent=2)
        print(f"Hall-MRI growth: measured {measured:.6g}, linear {linear:.6g} "
              f"(ratio {measured / linear:.5f})")
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.semilogy(t, amplitude, "b-", label="mode amplitude")
        ax.semilogy(t, amplitude[0] * np.exp(linear * (t - t[0])), "k--",
                    label=f"linear, s={linear:.4f}")
        ax.set(xlabel="t", ylabel="|b| of fundamental mode", title="Hall-MRI growth")
        ax.legend()
        fig.savefig(os.path.join(self.outdir, f"hall_mri1d_t{solver.t:.3f}.png"), dpi=150)
        plt.close(fig)

    def get_solver_class(self):
        return SpectralSolverMHD1D
