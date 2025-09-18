"""1D Sod shock tube problem."""

import os
from typing import Optional

import matplotlib.pyplot as plt

from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.initial_conditions import sod_shock_tube
from phrike.solver import SpectralSolver1D
from phrike.visualization import plot_fields, plot_conserved_time_series
from phrike.io import save_solution_snapshot
from .base import BaseProblem


class SodProblem(BaseProblem):
    """1D Sod shock tube problem."""

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid1D:
        """Create 1D grid."""
        # Debug mode: validate backend and device availability
        if debug:
            if backend == "torch":
                try:
                    import torch
                    if device == "cuda" and not torch.cuda.is_available():
                        raise RuntimeError(f"Debug mode: CUDA requested but not available. PyTorch was not compiled with CUDA support.")
                    elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                        raise RuntimeError(f"Debug mode: MPS requested but not available. MPS is only available on Apple Silicon Macs.")
                    elif device not in [None, "cpu", "cuda", "mps"]:
                        raise RuntimeError(f"Debug mode: Unknown device '{device}' requested. Valid devices are: cpu, cuda, mps")
                except ImportError:
                    raise RuntimeError(f"Debug mode: Torch backend requested but PyTorch is not installed.")
        
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
        """Create Sod shock tube initial conditions.

        Supports optional tanh-smoothed discontinuity controlled by
        initial_conditions.smoothing: { enabled: bool, sigma: float }.
        If sigma <= 0, reverts to sharp discontinuity.
        """
        import numpy as np

        x0 = float(self.config["grid"].get("x0", 0.5 * grid.Lx))
        left = self.config["initial_conditions"]["left"]
        right = self.config["initial_conditions"]["right"]

        # Tanh smoothing config (separate from Gaussian smoother)
        ic_block = self.config.get("initial_conditions", {})
        smoothing_cfg = ic_block.get("smoothing", {}) if isinstance(ic_block, dict) else {}
        tanh_enabled = bool(smoothing_cfg.get("enabled", False))
        sigma = float(smoothing_cfg.get("sigma", 0.0))

        if tanh_enabled and sigma > 0.0:
            # Build smoothed primitive profiles to mirror Dedalus script
            try:
                import torch  # type: ignore
                is_torch = isinstance(grid.x, torch.Tensor)
            except Exception:
                is_torch = False
                torch = None  # type: ignore

            if is_torch:
                xl = grid.x
                def _t(vl, vr):
                    return vr + (vl - vr) * 0.5 * (1.0 + torch.tanh((x0 - xl) / sigma))
                rho = _t(float(left["rho"]), float(right["rho"]))
                u   = _t(float(left["u"]),   float(right["u"]))
                p   = _t(float(left["p"]),   float(right["p"]))
            else:
                x = np.asarray(grid.x)
                tanh_step = 0.5 * (1.0 + np.tanh((x0 - x) / sigma))
                rho = float(right["rho"]) + (float(left["rho"]) - float(right["rho"])) * tanh_step
                u   = float(right["u"])   + (float(left["u"])   - float(right["u"]))   * tanh_step
                p   = float(right["p"])   + (float(left["p"])   - float(right["p"]))   * tanh_step

            eq = EulerEquations1D(gamma=self.gamma)
            U0 = eq.conservative(rho, u, p)
        else:
            # Sharp discontinuity (original)
            U0 = sod_shock_tube(grid.x, x0, left, right, self.gamma)

        # Optional Gaussian smoothing (existing feature)
        if getattr(self, "ic_smoothing_config", None) and self.ic_smoothing_config.get("enabled", False):
            eq = EulerEquations1D(gamma=self.gamma)
            rho, u, p, _ = eq.primitive(U0)
            rho_s, u_s, p_s = self.apply_initial_conditions_smoothing(rho, u, p, grid)
            U0 = eq.conservative(rho_s, u_s, p_s)

        # If non-periodic basis and Dirichlet BCs are specified per-var, pass values to grid
        # so that BCs like pressure Dirichlet can be enforced if requested.
        bc = self.config.get("grid", {}).get("bc", None)
        if isinstance(bc, dict):
            dirichlet_values = {}
            # Allow optional boundary_values map under grid.bc_values
            bc_vals = self.config.get("grid", {}).get("bc_values", {})
            for key in ("density", "rho", "pressure", "p", "momentum", "velocity", "u"):
                if key in bc_vals:
                    vals = bc_vals[key]
                    if isinstance(vals, (list, tuple)) and len(vals) == 2:
                        dirichlet_values[key] = (float(vals[0]), float(vals[1]))
            if dirichlet_values:
                try:
                    grid.set_dirichlet_bc_values(dirichlet_values)
                except Exception:
                    pass

        return U0

    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        # Save frame for video generation
        frames_dir = os.path.join(self.outdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Get video configuration for frame settings
        video_config = self.config.get("video", {})
        frame_dpi = int(video_config.get("frame_dpi", 150))
        frame_scale = float(video_config.get("scale", 1.0))
        
        # Calculate figure size based on scale
        base_size = 10
        figsize = (base_size * frame_scale, 6 * frame_scale)
        
        # Create frame plot
        fig, axs = plt.subplots(2, 2, figsize=figsize, constrained_layout=True)
        fig.suptitle(f"Sod Shock Tube at t={t:.3f}", fontsize=14)
        
        x = solver.grid.x
        rho, u, p, _ = solver.equations.primitive(U)
        
        # Convert to numpy if needed
        x = self.convert_torch_to_numpy(x)[0]
        rho = self.convert_torch_to_numpy(rho)[0]
        u = self.convert_torch_to_numpy(u)[0]
        p = self.convert_torch_to_numpy(p)[0]
        
        # Density
        axs[0, 0].plot(x, rho, "b-", linewidth=2)
        axs[0, 0].set_xlabel("x")
        axs[0, 0].set_ylabel("Density")
        axs[0, 0].set_title("Density Profile")
        axs[0, 0].grid(True, alpha=0.3)
        
        # Velocity
        axs[0, 1].plot(x, u, "g-", linewidth=2)
        axs[0, 1].set_xlabel("x")
        axs[0, 1].set_ylabel("Velocity")
        axs[0, 1].set_title("Velocity Profile")
        axs[0, 1].grid(True, alpha=0.3)
        
        # Pressure
        axs[1, 0].plot(x, p, "m-", linewidth=2)
        axs[1, 0].set_xlabel("x")
        axs[1, 0].set_ylabel("Pressure")
        axs[1, 0].set_title("Pressure Profile")
        axs[1, 0].grid(True, alpha=0.3)
        
        # Energy density
        E = U[2]  # Total energy density
        E = self.convert_torch_to_numpy(E)[0]
        axs[1, 1].plot(x, E, "r-", linewidth=2)
        axs[1, 1].set_xlabel("x")
        axs[1, 1].set_ylabel("Energy Density")
        axs[1, 1].set_title("Energy Density Profile")
        axs[1, 1].grid(True, alpha=0.3)
        
        # Save frame
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=frame_dpi)
        plt.close(fig)
        
        # Save snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir, t, U=U, grid=solver.grid, equations=solver.equations
        )
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

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
            title=f"Sod at t={solver.t:.3f}",
            outpath=os.path.join(self.outdir, f"fields_t{solver.t:.3f}.png"),
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

    def get_solver_class(self):
        """Get 1D solver class."""
        return SpectralSolver1D
