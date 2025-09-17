"""1D periodic acoustic wave problem for comprehensive testing.

This problem uses sinusoidal density perturbations in a periodic domain,
creating acoustic waves that are ideal for:
- Conservation testing (mass, momentum, energy should be preserved exactly)
- Spectral accuracy testing (smooth periodic functions are resolved exactly)
- Convergence testing (can vary amplitude and wavenumber)
- Monitoring system testing (predictable evolution)
- Restart functionality testing (periodic state is easy to validate)
"""

import os
import numpy as np
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt

from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.initial_conditions import sinusoidal_density
from phrike.solver import SpectralSolver1D
from phrike.visualization import plot_fields, plot_conserved_time_series
from phrike.io import save_solution_snapshot
from .base import BaseProblem


class Acoustic1DProblem(BaseProblem):
    """1D periodic acoustic wave problem with sinusoidal density perturbations.

    This problem is designed specifically for comprehensive testing of the 1D solver:
    - Uses periodic boundary conditions (natural for spectral methods)
    - Small amplitude perturbations for linear regime testing
    - Predictable evolution for validation
    - Exact conservation properties
    - Smooth initial conditions for spectral accuracy testing
    """

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid1D:
        """Create 1D periodic grid."""
        N = int(self.config["grid"]["N"])
        Lx = float(self.config["grid"]["Lx"])
        dealias = bool(self.config["grid"].get("dealias", True))

        return Grid1D(
            N=N,
            Lx=Lx,
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
        """Create sinusoidal density perturbation initial conditions."""
        # Get parameters from config
        ic_config = self.config["initial_conditions"]
        rho0 = float(ic_config.get("rho0", 1.0))
        u0 = float(ic_config.get("u0", 0.0))
        p0 = float(ic_config.get("p0", 1.0))
        amplitude = float(ic_config.get("amplitude", 1e-3))
        k = int(ic_config.get("k", 1))

        return sinusoidal_density(
            x=grid.x,
            rho0=rho0,
            u0=u0,
            p0=p0,
            amplitude=amplitude,
            k=k,
            Lx=grid.Lx,
            gamma=self.gamma,
        )

    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        # For 1D problems, we don't need frame-by-frame visualization
        # The final plot will be created in create_final_visualization
        pass

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
            title=f"1D Test at t={solver.t:.6f}",
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

            # Create additional analysis plots for testing
            self._create_test_analysis_plots(solver)

    def _create_test_analysis_plots(self, solver) -> None:
        """Create additional analysis plots specific to testing."""
        if not hasattr(solver, "history") or not solver.history:
            return

        history = solver.history
        times = np.array(history.get("time", []))

        if len(times) == 0:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle("1D Test Problem Analysis", fontsize=14)

        # Conservation errors over time
        if "mass" in history and "momentum" in history and "energy" in history:
            mass = np.array(history["mass"])
            momentum = np.array(history["momentum"])
            energy = np.array(history["energy"])

            # Relative errors
            mass_error = np.abs(mass - mass[0]) / np.abs(mass[0])
            momentum_error = np.abs(momentum - momentum[0]) / (
                np.abs(momentum[0]) + 1e-16
            )
            energy_error = np.abs(energy - energy[0]) / np.abs(energy[0])

            axes[0, 0].semilogy(
                times, mass_error, label="Mass", marker="o", markersize=3
            )
            axes[0, 0].semilogy(
                times, momentum_error, label="Momentum", marker="s", markersize=3
            )
            axes[0, 0].semilogy(
                times, energy_error, label="Energy", marker="^", markersize=3
            )
            axes[0, 0].set_xlabel("Time")
            axes[0, 0].set_ylabel("Relative Error")
            axes[0, 0].set_title("Conservation Errors")
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)

        # Time step evolution
        if "dt" in history:
            dt = np.array(history["dt"])
            axes[0, 1].plot(times, dt, "b-", linewidth=1)
            axes[0, 1].set_xlabel("Time")
            axes[0, 1].set_ylabel("Time Step")
            axes[0, 1].set_title("Adaptive Time Step")
            axes[0, 1].grid(True, alpha=0.3)

        # Spectral analysis of final state
        rho, u, p, a = solver.equations.primitive(solver.U)
        k_modes = np.fft.fftfreq(len(rho), d=solver.grid.dx)
        rho_hat = np.fft.fft(rho)

        # Only plot positive frequencies
        positive_k = k_modes[: len(k_modes) // 2]
        rho_spectrum = np.abs(rho_hat[: len(k_modes) // 2])

        axes[1, 0].semilogy(positive_k[1:], rho_spectrum[1:], "r-", linewidth=1)
        axes[1, 0].set_xlabel("Wavenumber")
        axes[1, 0].set_ylabel("Density Spectrum")
        axes[1, 0].set_title("Final Density Spectrum")
        axes[1, 0].grid(True, alpha=0.3)

        # Phase space plot (if there's motion)
        if np.std(u) > 1e-12:
            axes[1, 1].plot(rho, u, "g-", alpha=0.7, linewidth=1)
            axes[1, 1].set_xlabel("Density")
            axes[1, 1].set_ylabel("Velocity")
            axes[1, 1].set_title("Phase Space (ρ-u)")
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If no motion, plot density profile
            axes[1, 1].plot(solver.grid.x, rho, "g-", linewidth=2)
            axes[1, 1].set_xlabel("x")
            axes[1, 1].set_ylabel("Density")
            axes[1, 1].set_title("Final Density Profile")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            os.path.join(self.outdir, "test_analysis.png"), dpi=150, bbox_inches="tight"
        )
        plt.close()

    def get_solver_class(self):
        """Get 1D solver class."""
        return SpectralSolver1D

    def get_analytical_solution(self, t: float) -> Optional[Dict[str, Any]]:
        """Get analytical solution at time t (if available).

        For small amplitude sinusoidal perturbations, the solution should
        remain close to the initial conditions for short times.

        Returns:
            Dict with analytical values for validation, or None if not available.
        """
        # For very small amplitude perturbations in the linear regime,
        # we can provide some analytical expectations
        ic_config = self.config["initial_conditions"]
        amplitude = float(ic_config.get("amplitude", 1e-3))

        if amplitude < 1e-2:  # Linear regime
            # In the linear regime, small perturbations should evolve predictably
            # This is a placeholder for more sophisticated analytical solutions
            return {
                "regime": "linear",
                "expected_amplitude_growth": "bounded",
                "conservation_tolerance": 1e-14,
                "spectral_decay_rate": "exponential",
            }

        return None
