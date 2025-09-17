"""1D Gaussian wave packet problem for testing wave propagation and conservation.

This problem implements two versions of Gaussian wave packet tests:

1. Stationary wave packet:
   - Gaussian perturbation in density at domain center
   - Uniform background flow to the right
   - Wave packet propagates left at equal and opposite speed
   - Tests conservation and long-time stability

2. Traveling wave packet:
   - Same Gaussian perturbation in medium at rest
   - Packet propagates leftward at sound speed
   - Returns to initial position after integer crossing times
   - Tests convergence, dispersion, and profile preservation
"""

import os
import numpy as np
from typing import Optional, Dict

import matplotlib.pyplot as plt

from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.solver import SpectralSolver1D
from phrike.visualization import plot_fields, plot_conserved_time_series
from phrike.io import save_solution_snapshot
from .base import BaseProblem


class GaussianWave1DProblem(BaseProblem):
    """1D Gaussian wave packet problem for testing wave propagation and conservation.

    This problem is designed to test:
    - Conservation properties (mass, momentum, energy)
    - Wave propagation accuracy
    - Dispersion and diffusion characteristics
    - Long-time stability
    - Convergence with resolution
    """

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid1D:
        """Create 1D periodic grid."""
        N = int(self.config["grid"]["N"])
        Lx = float(self.config["grid"]["Lx"])
        dealias = bool(self.config["grid"].get("dealias", True))
        basis = str(self.config["grid"].get("basis", "fourier")).lower()
        bc = self.config["grid"].get("bc", None)

        return Grid1D(
            N=N,
            Lx=Lx,
            basis=basis,
            bc=bc,
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device,
            precision=self.precision,
        )

    def create_equations(self) -> EulerEquations1D:
        """Create 1D Euler equations."""
        return EulerEquations1D(gamma=self.gamma)

    def create_initial_conditions(self, grid: Grid1D):
        """Create Gaussian wave packet initial conditions with proper acoustic wave relations."""
        # Get parameters from config
        ic_config = self.config["initial_conditions"]
        rho0 = float(ic_config.get("rho0", 1.0))
        u0 = float(ic_config.get("u0", 0.0))
        p0 = float(ic_config.get("p0", 1.0))
        amplitude = float(ic_config.get("amplitude", 0.1))
        sigma = float(ic_config.get("sigma", 0.1))
        x0 = float(ic_config.get("x0", grid.Lx / 2.0))
        wave_type = ic_config.get(
            "wave_type", "traveling"
        )  # "traveling" or "stationary"

        # Create Gaussian perturbation
        x = grid.x
        gaussian = amplitude * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))

        # Calculate sound speed
        c_s = np.sqrt(self.gamma * p0 / rho0)

        if wave_type == "stationary":
            # For stationary wave packet: background flows right at sound speed
            # Wave packet propagates left at sound speed, so it appears stationary
            u_background = c_s  # Background velocity to the right

            # For a stationary packet, we need the wave to propagate left
            # This means we need a leftward traveling wave
            # Linear acoustic relations: δp = c_s² δρ, δu = -c_s δρ/ρ₀
            delta_rho = gaussian
            delta_p = c_s**2 * delta_rho
            delta_u = -c_s * delta_rho / rho0

            rho = rho0 + delta_rho
            u = u_background + delta_u
            p = p0 + delta_p
        else:  # traveling
            # For traveling wave packet: medium at rest initially
            # Wave propagates leftward at sound speed
            # Linear acoustic relations: δp = c_s² δρ, δu = -c_s δρ/ρ₀
            delta_rho = gaussian
            delta_p = c_s**2 * delta_rho
            delta_u = -c_s * delta_rho / rho0

            rho = rho0 + delta_rho
            u = u0 + delta_u  # u0 is typically 0 for traveling wave
            p = p0 + delta_p

        # Convert to conservative variables
        equations = EulerEquations1D(gamma=self.gamma)
        return equations.conservative(rho, u, p)

    def get_analytical_solution(
        self, grid: Grid1D, t: float
    ) -> Optional[Dict[str, np.ndarray]]:
        """Get analytical solution at time t with proper acoustic wave relations.

        For the traveling wave packet, this returns the exact translated profile.
        For the stationary wave packet, this returns the initial profile (should be unchanged).
        """
        ic_config = self.config["initial_conditions"]
        rho0 = float(ic_config.get("rho0", 1.0))
        u0 = float(ic_config.get("u0", 0.0))
        p0 = float(ic_config.get("p0", 1.0))
        amplitude = float(ic_config.get("amplitude", 0.1))
        sigma = float(ic_config.get("sigma", 0.1))
        x0 = float(ic_config.get("x0", grid.Lx / 2.0))
        wave_type = ic_config.get("wave_type", "traveling")

        x = grid.x
        c_s = np.sqrt(self.gamma * p0 / rho0)

        if wave_type == "stationary":
            # Stationary packet should remain at center
            gaussian = amplitude * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))

            # Apply linear acoustic relations
            delta_rho = gaussian
            delta_p = c_s**2 * delta_rho
            delta_u = -c_s * delta_rho / rho0

            rho_analytical = rho0 + delta_rho
            u_analytical = c_s + delta_u  # Background + perturbation
            p_analytical = p0 + delta_p
        else:  # traveling
            # Traveling packet moves left at sound speed
            # Account for periodic boundary conditions
            x_translated = (x + c_s * t) % grid.Lx
            gaussian = amplitude * np.exp(
                -((x_translated - x0) ** 2) / (2.0 * sigma**2)
            )

            # Apply linear acoustic relations
            delta_rho = gaussian
            delta_p = c_s**2 * delta_rho
            delta_u = -c_s * delta_rho / rho0

            rho_analytical = rho0 + delta_rho
            u_analytical = u0 + delta_u  # u0 is typically 0 for traveling wave
            p_analytical = p0 + delta_p

        return {"rho": rho_analytical, "u": u_analytical, "p": p_analytical}

    def compute_wave_packet_errors(self, solver, t: float) -> Dict[str, float]:
        """Compute L2 and L∞ errors against analytical solution."""
        analytical = self.get_analytical_solution(solver.grid, t)
        if analytical is None:
            return {"l2_error": np.nan, "linf_error": np.nan}

        # Get numerical solution
        rho_num, u_num, p_num, _ = solver.equations.primitive(solver.U)

        # Compute errors
        rho_error = rho_num - analytical["rho"]
        u_error = u_num - analytical["u"]
        p_error = p_num - analytical["p"]

        # L2 errors
        l2_rho = np.sqrt(np.mean(rho_error**2))
        l2_u = np.sqrt(np.mean(u_error**2))
        l2_p = np.sqrt(np.mean(p_error**2))
        l2_total = np.sqrt(l2_rho**2 + l2_u**2 + l2_p**2)

        # L∞ errors
        linf_rho = np.max(np.abs(rho_error))
        linf_u = np.max(np.abs(u_error))
        linf_p = np.max(np.abs(p_error))
        linf_total = max(linf_rho, linf_u, linf_p)

        return {
            "l2_error": l2_total,
            "linf_error": linf_total,
            "l2_rho": l2_rho,
            "l2_u": l2_u,
            "l2_p": l2_p,
            "linf_rho": linf_rho,
            "linf_u": linf_u,
            "linf_p": linf_p,
        }

    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        # Save frame for video generation
        frames_dir = os.path.join(self.outdir, "frames")
        if os.path.exists(frames_dir):
            U_primitive = solver.equations.primitive(U)
            analytical = self.get_analytical_solution(solver.grid, t)

            # Create frame plot
            fig, ax = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Gaussian Wave Packet at t={t:.3f}", fontsize=14)

            x = solver.grid.x
            rho, u, p, a = U_primitive

            # Density
            ax[0, 0].plot(x, rho, "b-", linewidth=2, label="Numerical")
            if analytical:
                ax[0, 0].plot(
                    x,
                    analytical["rho"],
                    "r--",
                    linewidth=1,
                    alpha=0.7,
                    label="Analytical",
                )
            ax[0, 0].set_xlabel("x")
            ax[0, 0].set_ylabel("Density")
            ax[0, 0].set_title("Density Profile")
            ax[0, 0].legend()
            ax[0, 0].grid(True, alpha=0.3)

            # Velocity
            ax[0, 1].plot(x, u, "g-", linewidth=2, label="Numerical")
            if analytical:
                ax[0, 1].plot(
                    x,
                    analytical["u"],
                    "r--",
                    linewidth=1,
                    alpha=0.7,
                    label="Analytical",
                )
            ax[0, 1].set_xlabel("x")
            ax[0, 1].set_ylabel("Velocity")
            ax[0, 1].set_title("Velocity Profile")
            ax[0, 1].legend()
            ax[0, 1].grid(True, alpha=0.3)

            # Pressure
            ax[1, 0].plot(x, p, "m-", linewidth=2, label="Numerical")
            if analytical:
                ax[1, 0].plot(
                    x,
                    analytical["p"],
                    "r--",
                    linewidth=1,
                    alpha=0.7,
                    label="Analytical",
                )
            ax[1, 0].set_xlabel("x")
            ax[1, 0].set_ylabel("Pressure")
            ax[1, 0].set_title("Pressure Profile")
            ax[1, 0].legend()
            ax[1, 0].grid(True, alpha=0.3)

            # Error plot
            if analytical:
                rho_error = np.abs(rho - analytical["rho"])
                u_error = np.abs(u - analytical["u"])
                p_error = np.abs(p - analytical["p"])

                # Only use log scale if there are positive values
                if np.max(rho_error) > 0:
                    ax[1, 1].semilogy(x, rho_error, "b-", linewidth=1, label="Density")
                else:
                    ax[1, 1].plot(x, rho_error, "b-", linewidth=1, label="Density")

                if np.max(u_error) > 0:
                    ax[1, 1].semilogy(x, u_error, "g-", linewidth=1, label="Velocity")
                else:
                    ax[1, 1].plot(x, u_error, "g-", linewidth=1, label="Velocity")

                if np.max(p_error) > 0:
                    ax[1, 1].semilogy(x, p_error, "m-", linewidth=1, label="Pressure")
                else:
                    ax[1, 1].plot(x, p_error, "m-", linewidth=1, label="Pressure")

                ax[1, 1].set_xlabel("x")
                ax[1, 1].set_ylabel("Absolute Error")
                ax[1, 1].set_title("Error vs Analytical")
                ax[1, 1].legend()
                ax[1, 1].grid(True, alpha=0.3)
            else:
                ax[1, 1].text(
                    0.5,
                    0.5,
                    "No analytical solution",
                    ha="center",
                    va="center",
                    transform=ax[1, 1].transAxes,
                )
                ax[1, 1].set_title("Error vs Analytical")

            plt.tight_layout()

            # Save frame with timestamp-based naming
            frame_path = os.path.join(frames_dir, f"frame_{t:.6f}.png")
            plt.savefig(frame_path, dpi=100, bbox_inches="tight")
            plt.close()

            # Also save regular field plot at intervals (every 0.1 time units)
            if t % 0.1 < 1e-6:  # Check if t is close to a multiple of 0.1
                plot_fields(
                    grid=solver.grid,
                    U=U,
                    equations=solver.equations,
                    title=f"GaussianWave1D at t={t:.3f}",
                    outpath=os.path.join(self.outdir, f"fields_t{t:.6f}.png"),
                )

    def create_final_visualization(self, solver) -> None:
        """Create final visualization plots."""
        # Save final snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir,
            solver.t,
            U=solver.U,
            grid=solver.grid,
            equations=solver.equations,
        )
        print(f"Saved final snapshot: {snapshot_path}")

        # Plot fields
        plot_fields(
            grid=solver.grid,
            U=solver.U,
            equations=solver.equations,
            title=f"Gaussian Wave at t={solver.t:.6f}",
            outpath=os.path.join(self.outdir, f"fields_t{solver.t:.6f}.png"),
        )

        # Plot conserved quantities if history is available
        if (
            hasattr(solver, "history")
            and solver.history
            and len(solver.history.get("time", [])) > 0
        ):
            plot_conserved_time_series(
                solver.history, outpath=os.path.join(self.outdir, "conserved.png")
            )

            # Create additional analysis plots for wave packet evolution
            self._create_wave_packet_analysis_plots(solver)

    def _create_wave_packet_analysis_plots(self, solver) -> None:
        """Create additional analysis plots specific to wave packet evolution."""
        if not hasattr(solver, "history") or not solver.history:
            return

        history = solver.history
        times = np.array(history.get("time", []))

        if len(times) == 0:
            return

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Gaussian Wave Packet Analysis", fontsize=14)

        # Get current solution
        rho, u, p, a = solver.equations.primitive(solver.U)
        x = solver.grid.x

        # Plot 1: Current density profile vs analytical
        analytical = self.get_analytical_solution(solver.grid, solver.t)
        if analytical is not None:
            axes[0, 0].plot(x, rho, "b-", linewidth=2, label="Numerical")
            axes[0, 0].plot(
                x, analytical["rho"], "r--", linewidth=2, label="Analytical"
            )
            axes[0, 0].set_xlabel("x")
            axes[0, 0].set_ylabel("Density")
            axes[0, 0].set_title(f"Density Profile at t={solver.t:.3f}")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Conservation errors over time
        if "mass" in history and "momentum" in history and "energy" in history:
            mass = np.array(history["mass"])
            momentum = np.array(history["momentum"])
            energy = np.array(history["energy"])

            mass_error = np.abs(mass - mass[0]) / np.abs(mass[0])
            momentum_error = np.abs(momentum - momentum[0]) / (
                np.abs(momentum[0]) + 1e-16
            )
            energy_error = np.abs(energy - energy[0]) / np.abs(energy[0])

            axes[0, 1].semilogy(
                times, mass_error, label="Mass", marker="o", markersize=3
            )
            axes[0, 1].semilogy(
                times, momentum_error, label="Momentum", marker="s", markersize=3
            )
            axes[0, 1].semilogy(
                times, energy_error, label="Energy", marker="^", markersize=3
            )
            axes[0, 1].set_xlabel("Time")
            axes[0, 1].set_ylabel("Relative Error")
            axes[0, 1].set_title("Conservation Errors")
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Wave packet evolution (if we have multiple time snapshots)
        if len(times) > 1:
            # Plot density at different times
            time_indices = np.linspace(0, len(times) - 1, min(5, len(times)), dtype=int)
            colors = plt.cm.viridis(np.linspace(0, 1, len(time_indices)))

            for i, (idx, color) in enumerate(zip(time_indices, colors)):
                if "density_history" in history:
                    rho_t = history["density_history"][idx]
                    axes[0, 2].plot(
                        x,
                        rho_t,
                        color=color,
                        linewidth=1.5,
                        label=f"t={times[idx]:.3f}",
                    )

            axes[0, 2].set_xlabel("x")
            axes[0, 2].set_ylabel("Density")
            axes[0, 2].set_title("Wave Packet Evolution")
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)

        # Plot 4: Error evolution over time (if available)
        if "l2_error_history" in history:
            l2_errors = history["l2_error_history"]
            axes[1, 0].semilogy(times, l2_errors, "b-", linewidth=2)
            axes[1, 0].set_xlabel("Time")
            axes[1, 0].set_ylabel("L2 Error")
            axes[1, 0].set_title("L2 Error vs Time")
            axes[1, 0].grid(True, alpha=0.3)

        # Plot 5: Spectral analysis
        rho_hat = np.fft.fft(rho)
        k_modes = np.fft.fftfreq(len(rho), d=solver.grid.dx)
        positive_k = k_modes[: len(k_modes) // 2]
        rho_spectrum = np.abs(rho_hat[: len(k_modes) // 2])

        axes[1, 1].semilogy(positive_k[1:], rho_spectrum[1:], "r-", linewidth=1)
        axes[1, 1].set_xlabel("Wavenumber")
        axes[1, 1].set_ylabel("Density Spectrum")
        axes[1, 1].set_title("Final Density Spectrum")
        axes[1, 1].grid(True, alpha=0.3)

        # Plot 6: Phase space or velocity profile
        if np.std(u) > 1e-12:
            axes[1, 2].plot(x, u, "g-", linewidth=2)
            axes[1, 2].set_xlabel("x")
            axes[1, 2].set_ylabel("Velocity")
            axes[1, 2].set_title("Velocity Profile")
        else:
            axes[1, 2].plot(x, rho, "g-", linewidth=2)
            axes[1, 2].set_xlabel("x")
            axes[1, 2].set_ylabel("Density")
            axes[1, 2].set_title("Final Density Profile")

        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.outdir, "wave_packet_analysis.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()

    def get_solver_class(self):
        """Get 1D solver class."""
        return SpectralSolver1D

    def get_wave_parameters(self) -> Dict[str, float]:
        """Get wave packet parameters for analysis."""
        ic_config = self.config["initial_conditions"]
        rho0 = float(ic_config.get("rho0", 1.0))
        p0 = float(ic_config.get("p0", 1.0))
        amplitude = float(ic_config.get("amplitude", 0.1))
        sigma = float(ic_config.get("sigma", 0.1))
        x0 = float(ic_config.get("x0", 0.0))
        wave_type = ic_config.get("wave_type", "traveling")

        # Get domain length from grid config
        Lx = float(self.config["grid"]["Lx"])

        # Calculate derived parameters
        c_s = np.sqrt(self.gamma * p0 / rho0)  # Sound speed
        crossing_time = Lx / c_s  # Time for wave to cross domain

        return {
            "rho0": rho0,
            "p0": p0,
            "amplitude": amplitude,
            "sigma": sigma,
            "x0": x0,
            "wave_type": wave_type,
            "sound_speed": c_s,
            "crossing_time": crossing_time,
        }
