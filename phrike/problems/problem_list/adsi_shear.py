"""Shearing-sheet ADSI: ambipolar-diffusion shear instability in an axisymmetric local box.

Geometry: grid axis x is the vertical z_v, grid axis y the radial r, and the ignorable
third axis is phi = z_v x r = +z. The background is a uniform state with B0 = B_v z_v +
B_phi phi, rest density rho0 and pressure p0, in a frame moving with v0 = -S r phi
(rotation Omega, shear S). Perturbations ~ exp(i (k_v x + k_r y) + s t) are exact
solutions of the code's equations: the shearing-sheet sources are local and nothing
depends on phi. With e1 = (-k_r, k_v, 0)/|k| and b = b_perp e1 + b_phi phi (div b = 0),
the linear system acts on q = (drho, u_v, u_r, u_phi, b_perp, b_phi, dp):

  s drho  = -rho0 i k.u                          s dp = -gamma p0 i k.u
  rho0 s u = -i k (dp + B0.b) + i (k.B0) b + rho0 (0, 2 Omega u_phi, -(2 Omega - S) u_r)
  s b      = i (k.B0) u - B0 i k.u - S b_r phi - i k x (E_AD + E_H)
  E_AD = eta (|B0|^2 J - B0 (B0.J)),   E_H = eta_H J x B0/|B0|,   J = i k x b

E_AD is exactly the electric field of MidpointAmbipolar (eta multiplies |B|^2 J_perp).
J0 = 0, so perturbing the projector and |B|^2 gives no first-order term, and the Joule
heating E.J is second order, so dp is adiabatic. E_H uses eta_H/(1 + l^2 k^2) for the
inertia method. For B_phi = 0, k_r = 0 the rows reduce to hall_mri_matrix.

Energy bound: the quadratic energy 0.5 rho0 |u|^2 + 0.5 |b|^2 + dp^2/(2 gamma p0) is
conserved by the ideal terms and the Coriolis force, the Hall term does no work and
ambipolar diffusion removes it (-eta |B0|^2 |J_perp|^2); the shear source adds
S (rho0 u_r u_phi - b_r b_phi) <= |S| times the energy density. Hence Re s <= |S|/2.

Incompressible limit (p0 -> infinity, Kunz & Balbus 2004, MNRAS 348, 355, quartic for
B_R = 0; compared term by term by hand): with w^2 = (k_v B_v)^2/rho0, f = k_v^2/k^2,
kappa^2 = 2 Omega (2 Omega - S), a = eta k^2 B_v^2, c = eta (k_v^2 B_v^2 + k^2 B_phi^2),
A2 = (eta k |B0| k_v B_v)^2 and P = eta S k_r k_v B_v B_phi,

  s^4 + (a + c) s^3 + (f kappa^2 + 2 w^2 + A2 + P) s^2 + (w^2 + f kappa^2)(a + c) s
      + w^2 (w^2 - 2 Omega S f) + f kappa^2 A2 + (w^2 + f kappa^2) P = 0.

For k_r = 0 and Omega = 0 the quartic factorizes into two damped Alfven pairs: pure shear
is stable for k parallel to the vertical axis (S cannot act: u_r, b_r form a closed block).
For Omega = 0 the Routh-Hurwitz conditions reduce to w^2 + P > 0, so the box is unstable
iff rho0 eta S k_r B_phi/(k_v B_v) < -1, i.e. S k_r k_v B_v B_phi < 0 and the ambipolar
Elsasser number Am = v_A^2/(eta |B0|^2 S) = 1/(rho0 eta S) obeys |k_r B_phi/(k_v B_v)| > Am.
"""

import json
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from phrike.equations import MHDEquations2D
from phrike.grid import Grid2D
from phrike.solver import SpectralSolverMHD2D
from ..base import BaseProblem


def adsi_shear_matrix(kv, kr, omega, shear, bv, bphi, rho0=1.0, p0=1.0, gamma=5.0 / 3.0,
                      eta=0.0, eta_h=0.0, length=0.0):
    """7x7 matrix on (drho, u_v, u_r, u_phi, b_perp, b_phi, dp) for k = (k_v, k_r) in the (v, r) plane."""
    k = np.array([kv, kr, 0.0])
    kn = np.hypot(kv, kr)
    e1, ez = np.array([-kr, kv, 0.0]) / kn, np.array([0.0, 0.0, 1.0])
    B0 = np.array([bv, 0.0, bphi])
    b2 = B0 @ B0
    hall = eta_h / (1.0 + (length * kn) ** 2) / np.sqrt(b2)
    M = np.zeros((7, 7), dtype=complex)
    for j in range(7):
        q = np.eye(7)[j]
        u, dp = q[1:4], q[6]
        b = q[4] * e1 + q[5] * ez
        div_u = 1j * (k @ u)
        J = 1j * np.cross(k, b)
        E = eta * (b2 * J - B0 * (B0 @ J)) + hall * np.cross(J, B0)
        du = (-1j * k * (dp + B0 @ b) + 1j * (k @ B0) * b) / rho0
        du[1] += 2.0 * omega * u[2]
        du[2] -= (2.0 * omega - shear) * u[1]
        db = 1j * (k @ B0) * u - B0 * div_u - 1j * np.cross(k, E)
        db[2] -= shear * b[1]
        M[:, j] = [-rho0 * div_u, *du, e1 @ db, db[2], -gamma * p0 * div_u]
    return M


def _leading(values):
    """Index of the largest real part among nonzero eigenvalues; ties go to the largest imaginary part."""
    scale = max(float(np.abs(values).max()), 1e-300)
    live = np.nonzero(np.abs(values) > 1e-9 * scale)[0]
    near = live[values.real[live] >= values.real[live].max() - 1e-9 * scale]
    return int(max(near, key=lambda i: values[i].imag))


def adsi_shear_mode(kv, kr, *args, **params):
    """Leading eigenvalue and eigenvector, normalized to |(b_perp, b_phi)| = 1."""
    values, vectors = np.linalg.eig(adsi_shear_matrix(kv, kr, *args, **params))
    i = _leading(values)
    return values[i], vectors[:, i] / np.linalg.norm(vectors[4:6, i])


def adsi_shear_left(kv, kr, *args, **params):
    """Left eigenvector of the leading eigenvalue (same ordering as adsi_shear_mode)."""
    values, vectors = np.linalg.eig(adsi_shear_matrix(kv, kr, *args, **params))
    return np.linalg.inv(vectors)[_leading(values)]


def adsi_shear_fastest(omega, shear, bv, bphi, rho0=1.0, kmin=0.1, kmax=6.0, n=31, **params):
    """Fastest-growing (k, psi) with k_v = k cos(psi), k_r = k sin(psi); returns (k, psi, s, v).

    The search covers |k| in [kmin, kmax] max(|Omega|, |S|)/v_A and |psi| <= 85 degrees on an
    n x n grid, zoomed in three times. Growth can saturate at large |k|, so the optimum is
    only taken inside this range.
    """
    scale = max(abs(omega), abs(shear), 1e-12) / (np.hypot(bv, bphi) / np.sqrt(rho0))
    args = dict(omega=omega, shear=shear, bv=bv, bphi=bphi, rho0=rho0, **params)
    ks = scale * np.geomspace(kmin, kmax, n)
    psis = np.radians(np.linspace(-85.0, 85.0, n))

    def best(ks, psis):
        growth = [[np.linalg.eigvals(adsi_shear_matrix(k * np.cos(p), k * np.sin(p), **args)).real.max()
                   for p in psis] for k in ks]
        return np.unravel_index(np.argmax(growth), (n, n))

    for _ in range(3):
        i, j = best(ks, psis)
        ks = np.geomspace(ks[max(i - 1, 0)], ks[min(i + 1, n - 1)], n)
        psis = np.linspace(psis[max(j - 1, 0)], psis[min(j + 1, n - 1)], n)
    i, j = best(ks, psis)
    k, psi = ks[i], psis[j]
    s, v = adsi_shear_mode(k * np.cos(psi), k * np.sin(psi), **args)
    return k, psi, s, v


def adsi_incompressible_roots(kv, kr, omega, shear, bv, bphi, rho0=1.0, eta=0.0):
    """Roots of the incompressible quartic of the module docstring (Kunz & Balbus 2004, B_R = 0)."""
    k2, b2 = kv * kv + kr * kr, bv * bv + bphi * bphi
    f, w2, kappa2 = kv * kv / k2, (kv * bv) ** 2 / rho0, 2.0 * omega * (2.0 * omega - shear)
    a, c = eta * k2 * bv ** 2, eta * (kv ** 2 * bv ** 2 + k2 * bphi ** 2)
    A2, P = (eta * bv * kv) ** 2 * k2 * b2, eta * shear * kr * kv * bv * bphi
    return np.roots([1.0, a + c, f * kappa2 + 2.0 * w2 + A2 + P, (w2 + f * kappa2) * (a + c),
                     w2 * (w2 - 2.0 * omega * shear * f) + f * kappa2 * A2 + (w2 + f * kappa2) * P])


def adsi_shear_box(kv, kr, mode_v=1, mode_r=1):
    """(Lx, Ly, m_v, m_r) of the box whose Fourier mode (m_v, m_r) has wavenumber (k_v, k_r); k_v > 0."""
    if kv <= 0.0:
        raise ValueError("The vertical wavenumber must be positive")
    Lx = 2.0 * np.pi * mode_v / kv
    if abs(kr) < 1e-12 * kv:
        return Lx, Lx, mode_v, 0
    return Lx, 2.0 * np.pi * mode_r / abs(kr), mode_v, int(mode_r * np.sign(kr))


def adsi_shear_fields(v, kv, kr):
    """Complex (drho, u_v, u_r, u_phi, b_v, b_r, b_phi, dp) of a mode vector v of adsi_shear_matrix."""
    e1 = np.array([-kr, kv]) / np.hypot(kv, kr)
    return np.array([v[0], v[1], v[2], v[3], v[4] * e1[0], v[4] * e1[1], v[5], v[6]])


def adsi_shear_vector(fields, kv, kr, mode_v, mode_r):
    """Complex amplitudes c of the 7 mode variables, field = Re(c exp(i (k_v x + k_r y))).

    fields holds (rho, u_v, u_r, u_phi, B_v, B_r, B_phi, p) with shape (8, Ny, Nx).
    """
    ny, nx = fields.shape[-2:]
    c = 2.0 * np.fft.fft2(fields, axes=(-2, -1))[:, mode_r % ny, mode_v] / (nx * ny)
    e1 = np.array([-kr, kv]) / np.hypot(kv, kr)
    return np.array([c[0], c[1], c[2], c[3], e1[0] * c[4] + e1[1] * c[5], c[6], c[7]])


class ADSIShearProblem(BaseProblem):
    """Fastest-growing shearing-sheet ADSI eigenmode in an axisymmetric (vertical, radial) box."""

    def _params(self):
        mhd = self.config.get("physics", {}).get("mhd", {})
        shear = mhd.get("shearing_sheet") or {}
        ad = mhd.get("ambipolar") or {}
        hall = mhd.get("hall") or {}
        ic = self.config.get("initial_conditions", {})
        omega = float(shear.get("omega", 0.0))
        bv, bphi = float(ic.get("Bv", 1.0)), float(ic.get("Bphi", 1.0))
        rho0 = float(ic.get("rho0", 1.0))
        eta = float(ad.get("eta", ad.get("alpha", 1.0))) if ad.get("enabled", False) else 0.0
        eta_h = length = 0.0
        if hall.get("enabled", False):
            eta_h = float(hall.get("eta_h", 0.0))
            if str(hall.get("eta_h_form", "constant")).lower() != "constant":
                eta_h *= (np.hypot(bv, bphi) / float(hall.get("b0", 1.0))
                          * float(hall.get("rho0", 1.0)) / rho0)
            if str(hall.get("method", "reference")).lower() == "inertia":
                length = float(hall.get("length", 0.0))
        return dict(omega=omega,
                    shear=(float(shear["shear"]) if "shear" in shear
                           else float(shear.get("q", 1.5)) * omega),
                    bv=bv, bphi=bphi, rho0=rho0,
                    p0=0.5 * float(ic.get("beta", 10.0)) * (bv ** 2 + bphi ** 2),
                    gamma=self.gamma, eta=eta, eta_h=eta_h, length=length)

    def create_grid(self, backend: str = "numpy", device: Optional[str] = None,
                    debug: bool = False) -> Grid2D:
        cfg = self.config["grid"]
        ic = self.config.get("initial_conditions", {})
        Lx, Ly = cfg.get("Lx", "auto"), cfg.get("Ly", "auto")
        self._mode_v, self._mode_r = int(ic.get("mode_v", 1)), int(ic.get("mode_r", 1))
        if isinstance(Lx, str) or isinstance(Ly, str):
            # Fit the chosen box mode to the fastest-growing wavevector (sign of k_r included).
            if str(Lx).lower() != "auto" or str(Ly).lower() != "auto":
                raise ValueError("grid.Lx and grid.Ly must both be numbers or both 'auto'")
            k, psi, s, _ = adsi_shear_fastest(**self._params())
            if s.real <= 0.0:
                raise ValueError("No growing shearing-sheet ADSI mode in the scanned range")
            Lx, Ly, self._mode_v, self._mode_r = adsi_shear_box(
                k * np.cos(psi), k * np.sin(psi), self._mode_v, abs(self._mode_r))
        grid = Grid2D(
            Nx=int(cfg["Nx"]), Ny=int(cfg["Ny"]), Lx=float(Lx), Ly=float(Ly),
            dealias=bool(cfg.get("dealias", True)),
            filter_params=self.filter_config, fft_workers=self.fft_workers,
            backend=backend, torch_device=device, precision=self.precision,
            debug=debug,
        )
        grid.mhd_config = self.config.get("physics", {}).get("mhd", {})
        self._setup_mode(grid)
        return grid

    def _setup_mode(self, grid: Grid2D) -> None:
        """Mode metadata and amplitude history; shared by fresh starts and restarts."""
        params = self._params()
        kv = 2.0 * np.pi * self._mode_v / grid.Lx
        kr = 2.0 * np.pi * self._mode_r / grid.Ly
        s, self._vector = adsi_shear_mode(kv, kr, **params)
        self.linear_mode = dict(k_v=float(kv), k_r=float(kr), growth=float(s.real),
                                frequency=float(s.imag), mode_v=self._mode_v,
                                mode_r=self._mode_r, **params)
        self.amplitude_history = []

    def create_equations(self) -> MHDEquations2D:
        return MHDEquations2D(gamma=self.gamma)

    def _amplitude(self, U) -> float:
        """Amplitude |b| of the box mode, from the magnetic components of the state."""
        B = self.convert_torch_to_numpy(U[5], U[6], U[7])
        fields = np.zeros((8,) + B[0].shape)
        fields[4:7] = B
        c = adsi_shear_vector(fields, self.linear_mode["k_v"], self.linear_mode["k_r"],
                              self._mode_v, self._mode_r)
        return float(np.linalg.norm(c[4:6]))

    def create_initial_conditions(self, grid: Grid2D):
        ic = self.config.get("initial_conditions", {})
        params, lm = self._params(), self.linear_mode
        x, y = self.convert_torch_to_numpy(grid.x, grid.y)
        phase = lm["k_v"] * x[None, :] + lm["k_r"] * y[:, None]
        wave = (float(ic.get("amplitude", 1e-6)) * np.hypot(params["bv"], params["bphi"])
                * np.real(adsi_shear_fields(self._vector, lm["k_v"], lm["k_r"])[:, None, None]
                          * np.exp(1j * phase)[None]))
        one = np.ones_like(phase)
        fields = [params["rho0"] * one + wave[0], *wave[1:4], params["p0"] * one + wave[7],
                  params["bv"] * one + wave[4], wave[5], params["bphi"] * one + wave[6]]
        if getattr(grid, "_use_torch", False):
            import torch

            fields = [torch.as_tensor(a, dtype=grid.x.dtype, device=grid.x.device)
                      for a in fields]
        U = self.create_equations().conservative(*fields)
        self.amplitude_history = [(self.t0, self._amplitude(U))]
        return U

    def initialize_monitoring(self, solver, U):
        # Also reached on restart, where the history starts from the loaded state.
        self.amplitude_history = [(self.t0, self._amplitude(U))]
        super().initialize_monitoring(solver, U)

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
        result = dict(self.linear_mode, measured_growth=measured,
                      ratio=measured / linear if linear != 0.0 else None,
                      fit_window=[float(t[window][0]), float(t[-1])])
        with open(os.path.join(self.outdir, "adsi_shear_growth.json"), "w") as stream:
            json.dump(result, stream, indent=2)
        print(f"Shearing-sheet ADSI growth: measured {measured:.6g}, linear {linear:.6g}")
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
        ax.semilogy(t, amplitude, "b-", label="mode amplitude")
        ax.semilogy(t, amplitude[0] * np.exp(linear * (t - t[0])), "k--",
                    label=f"linear, s={linear:.4f}")
        ax.set(xlabel="t", ylabel="|b| of box mode", title="Shearing-sheet ADSI growth")
        ax.legend()
        fig.savefig(os.path.join(self.outdir, f"adsi_shear_t{solver.t:.3f}.png"), dpi=150)
        plt.close(fig)

    def get_solver_class(self):
        return SpectralSolverMHD2D
