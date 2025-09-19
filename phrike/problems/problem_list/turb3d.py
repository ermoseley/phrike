"""3D turbulent velocity field problem."""

import os
import sys
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from phrike.grid import Grid3D
from phrike.equations import EulerEquations3D
# Initial condition function moved from phrike.initial_conditions
from phrike.solver import SpectralSolver3D
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def turbulent_velocity_3d(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    rho0: float = 1.0,
    p0: float = 1.0,
    vrms: float = 0.1,
    kmin: float = 2.0,
    kmax: float = 16.0,
    alpha: float = 0.3333,  # Compressive fraction (0=solenoidal, 1=compressive)
    spectrum_type: str = "parabolic",  # "parabolic" or "power_law"
    power_law_slope: float = -5.0 / 3.0,  # Kolmogorov slope
    seed: int = 42,
    gamma: float = 1.4,
) -> np.ndarray:
    """3D turbulent velocity initial condition matching RAMSES turb.py.

    Generates a divergence/rotation-controlled random velocity field using
    spectral filtering between k_min and k_max.

    Args:
        X, Y, Z: 3D coordinate arrays (shape: Nz, Ny, Nx)
        rho0: Uniform density
        p0: Uniform pressure
        vrms: Target RMS velocity magnitude
        kmin: Minimum wavenumber for band-pass
        kmax: Maximum wavenumber for band-pass
        alpha: Compressive fraction [0..1] (1=compressive, 0=solenoidal)
        spectrum_type: "parabolic" (band-limited) or "power_law"
        power_law_slope: Power law slope for spectrum='power_law'
        seed: Random seed for reproducibility
        gamma: Adiabatic index
    """
    # Torch interop if needed
    try:
        import torch  # type: ignore

        _TORCH_AVAILABLE = True
    except Exception:
        _TORCH_AVAILABLE = False
        torch = None  # type: ignore

    is_torch = _TORCH_AVAILABLE and any(
        isinstance(a, (torch.Tensor,)) for a in (X, Y, Z)
    )

    # Get grid dimensions
    if is_torch:
        nz, ny, nx = X.shape
    else:
        nz, ny, nx = X.shape

    # Set up random number generator
    rng = np.random.default_rng(seed)

    # Generate k-grid for FFT
    def generate_k_grid(n1, n2, n3):
        kx = np.fft.fftfreq(n1, d=1.0 / n1).reshape(n1, 1, 1)
        ky = np.fft.fftfreq(n2, d=1.0 / n2).reshape(1, n2, 1)
        kz = np.fft.fftfreq(n3, d=1.0 / n3).reshape(1, 1, n3)
        kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
        return kx, ky, kz, kmag

    # Generate turbulent velocity field
    def build_turbulent_velocity(
        n1, n2, n3, kmin, kmax, alpha, vrms, rng, spectrum_type, power_law_slope
    ):
        # Real-space Gaussian noise
        u0 = rng.standard_normal((n1, n2, n3)).astype(np.float64)
        v0 = rng.standard_normal((n1, n2, n3)).astype(np.float64)
        w0 = rng.standard_normal((n1, n2, n3)).astype(np.float64)

        # FFT to spectral domain
        U = np.fft.fftn(u0)
        V = np.fft.fftn(v0)
        W = np.fft.fftn(w0)

        # k-grid and spectral envelope
        kx, ky, kz, kmag = generate_k_grid(n1, n2, n3)
        with np.errstate(invalid="ignore"):
            band = (kmag >= kmin) & (kmag <= kmax)

            if spectrum_type == "parabolic":
                # Parabolic band-pass: w(k) ∝ (k - kmin)(kmax - k) within [kmin,kmax]
                envelope = (kmag - kmin) * (kmax - kmag)
                envelope = np.where(band, envelope, 0.0)
                envelope = np.clip(envelope, 0.0, None)
            elif spectrum_type == "power_law":
                # Power law: w(k) ∝ k^slope within [kmin,kmax]
                kmag_safe = np.where(kmag == 0.0, 1.0, kmag)
                envelope = np.where(band, kmag_safe**power_law_slope, 0.0)
                # Apply smooth cutoff at boundaries
                k_transition = 0.1
                dk = kmax - kmin
                k1 = kmin + k_transition * dk
                k2 = kmax - k_transition * dk
                low_transition = np.where(
                    (kmag >= kmin) & (kmag < k1),
                    0.5 * (1.0 + np.cos(np.pi * (kmag - kmin) / (k1 - kmin))),
                    1.0,
                )
                high_transition = np.where(
                    (kmag > k2) & (kmag <= kmax),
                    0.5 * (1.0 + np.cos(np.pi * (kmag - k2) / (kmax - k2))),
                    1.0,
                )
                envelope *= low_transition * high_transition
            else:
                raise ValueError(f"Unknown spectrum_type: {spectrum_type}")

            filt = np.sqrt(envelope, dtype=np.float64)

        # Avoid division by zero at k=0
        kmag_safe = np.where(kmag == 0.0, 1.0, kmag)

        # Projection to compressive (parallel) and solenoidal (perpendicular)
        dot = kx * U + ky * V + kz * W
        U_par = kx * dot / (kmag_safe * kmag_safe)
        V_par = ky * dot / (kmag_safe * kmag_safe)
        W_par = kz * dot / (kmag_safe * kmag_safe)
        U_perp = U - U_par
        V_perp = V - V_par
        W_perp = W - W_par

        # Mix
        a = float(alpha)
        U_mix = a * U_par + (1.0 - a) * U_perp
        V_mix = a * V_par + (1.0 - a) * V_perp
        W_mix = a * W_par + (1.0 - a) * W_perp

        # Apply spectral envelope
        U_mix *= filt
        V_mix *= filt
        W_mix *= filt

        # Enforce zero at k=0
        U_mix[0, 0, 0] = 0.0
        V_mix[0, 0, 0] = 0.0
        W_mix[0, 0, 0] = 0.0

        # Back to real space
        u = np.fft.ifftn(U_mix).real.astype(np.float32)
        v = np.fft.ifftn(V_mix).real.astype(np.float32)
        w = np.fft.ifftn(W_mix).real.astype(np.float32)

        # Normalize to desired vrms
        speed2 = u * u + v * v + w * w
        rms = float(np.sqrt(np.mean(speed2)))
        if rms > 0:
            s = float(vrms) / rms
            u *= s
            v *= s
            w *= s

        return u, v, w

    # Generate velocity field
    u, v, w = build_turbulent_velocity(
        nx, ny, nz, kmin, kmax, alpha, vrms, rng, spectrum_type, power_law_slope
    )

    # Convert to torch if needed
    if is_torch:
        assert torch is not None
        u = torch.from_numpy(u).to(device=X.device, dtype=X.dtype)
        v = torch.from_numpy(v).to(device=X.device, dtype=X.dtype)
        w = torch.from_numpy(w).to(device=X.device, dtype=X.dtype)
        rho = torch.full_like(X, float(rho0))
        p = torch.full_like(X, float(p0))
    else:
        rho = rho0 * np.ones_like(X)
        p = p0 * np.ones_like(X)

    eqs = EulerEquations3D(gamma=gamma)
    return eqs.conservative(rho, u, v, w, p)


class Turb3DProblem(BaseProblem):
    """3D turbulent velocity field problem."""

    def __init__(
        self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize with custom colormap registration."""
        super().__init__(config_path, config)

        # Import and register the custom colormap
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        try:
            from colormaps import register

            register("cmapkk9")
        except ImportError:
            pass  # colormap registration is optional

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid3D:
        """Create 3D grid."""
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Nz = int(self.config["grid"]["Nz"])
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        Lz = float(self.config["grid"]["Lz"])
        dealias = bool(self.config["grid"].get("dealias", True))

        return Grid3D(
            Nx=Nx,
            Ny=Ny,
            Nz=Nz,
            Lx=Lx,
            Ly=Ly,
            Lz=Lz,
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
        )

    def create_equations(self) -> EulerEquations3D:
        """Create 3D Euler equations."""
        return EulerEquations3D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid3D):
        """Create turbulent velocity initial conditions."""
        X, Y, Z = grid.xyz_mesh()
        icfg = self.config["initial_conditions"]

        rho0 = float(icfg.get("rho0", 1.0))
        p0 = float(icfg.get("p0", 1.0))
        vrms = float(icfg.get("vrms", 0.1))
        kmin = float(icfg.get("kmin", 2.0))
        kmax = float(icfg.get("kmax", 16.0))
        alpha = float(icfg.get("alpha", 0.3333))
        spectrum_type = str(icfg.get("spectrum_type", "parabolic"))
        power_law_slope = float(icfg.get("power_law_slope", -5.0 / 3.0))
        seed = int(icfg.get("seed", 42))

        return turbulent_velocity_3d(
            X,
            Y,
            Z,
            rho0=rho0,
            p0=p0,
            vrms=vrms,
            kmin=kmin,
            kmax=kmax,
            alpha=alpha,
            spectrum_type=spectrum_type,
            power_law_slope=power_law_slope,
            seed=seed,
            gamma=self.gamma,
        )

    def create_visualization(self, solver, t: float, U) -> None:
        """Create visualization for current state."""
        rho, ux, uy, uz, p = solver.equations.primitive(U)
        rho = self.convert_torch_to_numpy(rho)[0]

        # Get video configuration for frame settings
        video_config = self.config.get("video", {})
        frame_dpi = int(video_config.get("frame_dpi", 120))
        frame_scale = float(video_config.get("scale", 1.0))
        projection_type = video_config.get("projection_type", "column_density")

        # Calculate figure size based on scale
        base_size = 8
        figsize = (base_size * frame_scale, base_size * frame_scale)

        # Create frame
        fig, ax = plt.subplots(1, 1, figsize=figsize, constrained_layout=True)

        if projection_type == "column_density":
            # Create column density projection
            projection_data = self._create_column_density_projection(
                rho, solver.grid, video_config
            )
            extent = self._get_projection_extent(solver.grid, video_config)
            axis_labels = self._get_projection_axis_labels(video_config)
        else:
            # Create slice projection (original behavior)
            projection_data = self._create_slice_projection(
                rho, solver.grid, video_config
            )
            extent = self._get_slice_extent(solver.grid, video_config)
            axis_labels = self._get_slice_axis_labels(video_config)

        # Get colorbar settings
        colorbar_scale = video_config.get("colorbar_scale", "linear")
        colorbar_min = video_config.get("colorbar_min", None)
        colorbar_max = video_config.get("colorbar_max", None)
        colorbar_fixed = video_config.get("colorbar_fixed", False)

        # Set up colorbar parameters
        if colorbar_fixed and colorbar_min is not None and colorbar_max is not None:
            vmin, vmax = colorbar_min, colorbar_max
        else:
            vmin, vmax = None, None

        # Choose norm based on scale
        if colorbar_scale == "log":
            from matplotlib.colors import LogNorm

            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

        try:
            im = ax.imshow(
                projection_data,
                origin="lower",
                extent=extent,
                aspect="equal",
                cmap="cmapkk9",
                norm=norm,
            )
        except ValueError:
            # Fallback to default colormap if custom one not available
            im = ax.imshow(
                projection_data,
                origin="lower",
                extent=extent,
                aspect="equal",
                norm=norm,
            )

        icfg = self.config["initial_conditions"]
        vrms = float(icfg.get("vrms", 0.1))
        alpha = float(icfg.get("alpha", 0.3333))

        if projection_type == "column_density":
            thickness = video_config.get("column_thickness", 0.2)
            axis = video_config.get("column_axis", "z")
            ax.set_title(
                f"Turbulent Column Density (∫{axis}) t={t:.3f}\nvrms={vrms:.3f}, α={alpha:.3f}, thickness={thickness:.1f}"
            )
        else:
            slice_axis = video_config.get("slice_axis", "z")
            ax.set_title(
                f"Turbulent Density ({slice_axis}=mid) t={t:.3f}\nvrms={vrms:.3f}, α={alpha:.3f}"
            )

        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Save frame with high DPI
        frames_dir = os.path.join(self.outdir, "frames")
        fig.savefig(
            os.path.join(frames_dir, f"frame_{t:08.3f}.png"),
            dpi=frame_dpi,
            bbox_inches="tight",
        )
        plt.close(fig)

        # Save snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir, t, U=U, grid=solver.grid, equations=solver.equations
        )
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

    def _create_column_density_projection(self, rho, grid, video_config):
        """Create column density projection along specified axis."""
        axis = video_config.get("column_axis", "z")
        thickness = video_config.get("column_thickness", 0.2)

        # Calculate integration range
        if axis == "z":
            # Project along z-axis (integrate over z)
            nz = rho.shape[0]
            z_start = int(nz * (0.5 - thickness / 2))
            z_end = int(nz * (0.5 + thickness / 2))
            z_start = max(0, z_start)
            z_end = min(nz, z_end)
            n_elements = z_end - z_start
            projection = np.sum(rho[z_start:z_end, :, :], axis=0) / n_elements
        elif axis == "y":
            # Project along y-axis (integrate over y)
            ny = rho.shape[1]
            y_start = int(ny * (0.5 - thickness / 2))
            y_end = int(ny * (0.5 + thickness / 2))
            y_start = max(0, y_start)
            y_end = min(ny, y_end)
            n_elements = y_end - y_start
            projection = np.sum(rho[:, y_start:y_end, :], axis=1) / n_elements
        elif axis == "x":
            # Project along x-axis (integrate over x)
            nx = rho.shape[2]
            x_start = int(nx * (0.5 - thickness / 2))
            x_end = int(nx * (0.5 + thickness / 2))
            x_start = max(0, x_start)
            x_end = min(nx, x_end)
            n_elements = x_end - x_start
            projection = np.sum(rho[:, :, x_start:x_end], axis=2) / n_elements
        else:
            raise ValueError(f"Invalid column axis: {axis}")

        return projection

    def _create_slice_projection(self, rho, grid, video_config):
        """Create slice projection along specified axis."""
        axis = video_config.get("slice_axis", "z")
        position = video_config.get("slice_position", 0.5)

        if axis == "z":
            # Slice along z-axis
            nz = rho.shape[0]
            slice_idx = int(nz * position)
            slice_idx = max(0, min(nz - 1, slice_idx))
            projection = rho[slice_idx, :, :]
        elif axis == "y":
            # Slice along y-axis
            ny = rho.shape[1]
            slice_idx = int(ny * position)
            slice_idx = max(0, min(ny - 1, slice_idx))
            projection = rho[:, slice_idx, :]
        elif axis == "x":
            # Slice along x-axis
            nx = rho.shape[2]
            slice_idx = int(nx * position)
            slice_idx = max(0, min(nx - 1, slice_idx))
            projection = rho[:, :, slice_idx]
        else:
            raise ValueError(f"Invalid slice axis: {axis}")

        return projection

    def _get_projection_extent(self, grid, video_config):
        """Get extent for column density projection."""
        axis = video_config.get("column_axis", "z")

        if axis == "z":
            return [0, grid.Lx, 0, grid.Ly]
        elif axis == "y":
            return [0, grid.Lx, 0, grid.Lz]
        elif axis == "x":
            return [0, grid.Ly, 0, grid.Lz]
        else:
            raise ValueError(f"Invalid column axis: {axis}")

    def _get_slice_extent(self, grid, video_config):
        """Get extent for slice projection."""
        axis = video_config.get("slice_axis", "z")

        if axis == "z":
            return [0, grid.Lx, 0, grid.Ly]
        elif axis == "y":
            return [0, grid.Lx, 0, grid.Lz]
        elif axis == "x":
            return [0, grid.Ly, 0, grid.Lz]
        else:
            raise ValueError(f"Invalid slice axis: {axis}")

    def _get_projection_axis_labels(self, video_config):
        """Get axis labels for column density projection."""
        axis = video_config.get("column_axis", "z")

        if axis == "z":
            return ["x", "y"]
        elif axis == "y":
            return ["x", "z"]
        elif axis == "x":
            return ["y", "z"]
        else:
            raise ValueError(f"Invalid column axis: {axis}")

    def _get_slice_axis_labels(self, video_config):
        """Get axis labels for slice projection."""
        axis = video_config.get("slice_axis", "z")

        if axis == "z":
            return ["x", "y"]
        elif axis == "y":
            return ["x", "z"]
        elif axis == "x":
            return ["y", "z"]
        else:
            raise ValueError(f"Invalid slice axis: {axis}")

    def create_final_visualization(self, solver) -> None:
        """Create final visualization plots."""
        rho, ux, uy, uz, p = solver.equations.primitive(solver.U)
        rho, ux, uy, uz, p = self.convert_torch_to_numpy(rho, ux, uy, uz, p)

        # Create comprehensive analysis plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)

        mid = rho.shape[0] // 2
        icfg = self.config["initial_conditions"]
        vrms = float(icfg.get("vrms", 0.1))
        alpha = float(icfg.get("alpha", 0.3333))
        kmin = float(icfg.get("kmin", 2.0))
        kmax = float(icfg.get("kmax", 16.0))

        # Density
        try:
            im1 = axes[0, 0].imshow(
                rho[mid],
                origin="lower",
                extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
                aspect="equal",
                cmap="cmapkk9",
            )
        except ValueError:
            im1 = axes[0, 0].imshow(
                rho[mid],
                origin="lower",
                extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
                aspect="equal",
            )
        axes[0, 0].set_title(f"Density (z=mid)\nt={solver.t:.3f}")
        axes[0, 0].set_xlabel("x")
        axes[0, 0].set_ylabel("y")
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)

        # Velocity magnitude
        v_mag = np.sqrt(ux**2 + uy**2 + uz**2)
        try:
            im2 = axes[0, 1].imshow(
                v_mag[mid],
                origin="lower",
                extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
                aspect="equal",
                cmap="cmapkk9",
            )
        except ValueError:
            im2 = axes[0, 1].imshow(
                v_mag[mid],
                origin="lower",
                extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
                aspect="equal",
            )
        axes[0, 1].set_title(f"Velocity Magnitude (z=mid)\nt={solver.t:.3f}")
        axes[0, 1].set_xlabel("x")
        axes[0, 1].set_ylabel("y")
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)

        # Pressure
        try:
            im3 = axes[0, 2].imshow(
                p[mid],
                origin="lower",
                extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
                aspect="equal",
                cmap="cmapkk9",
            )
        except ValueError:
            im3 = axes[0, 2].imshow(
                p[mid],
                origin="lower",
                extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
                aspect="equal",
            )
        axes[0, 2].set_title(f"Pressure (z=mid)\nt={solver.t:.3f}")
        axes[0, 2].set_xlabel("x")
        axes[0, 2].set_ylabel("y")
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)

        # Velocity components
        im4 = axes[1, 0].imshow(
            ux[mid],
            origin="lower",
            extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
            aspect="equal",
            cmap="RdBu_r",
        )
        axes[1, 0].set_title(f"ux (z=mid)\nt={solver.t:.3f}")
        axes[1, 0].set_xlabel("x")
        axes[1, 0].set_ylabel("y")
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)

        im5 = axes[1, 1].imshow(
            uy[mid],
            origin="lower",
            extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
            aspect="equal",
            cmap="RdBu_r",
        )
        axes[1, 1].set_title(f"uy (z=mid)\nt={solver.t:.3f}")
        axes[1, 1].set_xlabel("x")
        axes[1, 1].set_ylabel("y")
        plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)

        im6 = axes[1, 2].imshow(
            uz[mid],
            origin="lower",
            extent=[0, solver.grid.Lx, 0, solver.grid.Ly],
            aspect="equal",
            cmap="RdBu_r",
        )
        axes[1, 2].set_title(f"uz (z=mid)\nt={solver.t:.3f}")
        axes[1, 2].set_xlabel("x")
        axes[1, 2].set_ylabel("y")
        plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)

        plt.suptitle(
            f"3D Turbulent Velocity Field (GPU/MPS)\nvrms={vrms:.3f}, α={alpha:.3f}, k=[{kmin:.1f},{kmax:.1f}]",
            fontsize=16,
            fontweight="bold",
        )
        plt.savefig(
            os.path.join(self.outdir, f"turb3d_analysis_t{solver.t:.3f}.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def get_solver_class(self):
        """Get 3D solver class."""
        return SpectralSolver3D
