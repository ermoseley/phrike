"""1D periodic acoustic wave.

The default initial condition is a right-going acoustic eigenmode,

    u' = c * rho' / rho0,    p' = c^2 * rho',

which translates at the sound speed. ``initial_conditions.wave: entropy``
keeps the old stationary entropy mode (density perturbation only).
"""

import os
import numpy as np
from typing import Optional, Dict, Any

from phrike.grid import Grid1D
from phrike.equations import EulerEquations1D
from phrike.solver import SpectralSolver1D
from phrike.visualization import plot_fields
from phrike.io import save_solution_snapshot
from ..base import BaseProblem


def _wave_sign(direction: str) -> float:
    name = str(direction).lower()
    if name == "right":
        return 1.0
    if name == "left":
        return -1.0
    raise ValueError("initial_conditions.direction must be 'right' or 'left'")


def acoustic_primitives(
    x: np.ndarray,
    rho0: float,
    u0: float,
    p0: float,
    amplitude: float,
    k: int,
    Lx: float,
    gamma: float,
    wave: str = "acoustic",
    direction: str = "right",
    t: float = 0.0,
):
    """Primitive fields of a sinusoidal acoustic eigenmode or entropy mode.

    The acoustic mode is the exact linear translate. The entropy mode is
    stationary. Both are periodic on ``[0, Lx)``.
    """
    c = float(np.sqrt(gamma * p0 / rho0))
    kind = str(wave).lower()
    if kind not in ("acoustic", "entropy"):
        raise ValueError("initial_conditions.wave must be 'acoustic' or 'entropy'")
    sign = _wave_sign(direction) if kind == "acoustic" else 0.0
    phase_speed = sign * c if kind == "acoustic" else 0.0

    if hasattr(x, "cpu"):
        import torch

        phase = 2.0 * torch.pi * k * (x - phase_speed * t) / Lx
        sine = torch.sin(phase)
        ones = torch.ones_like(x)
    else:
        phase = 2.0 * np.pi * k * (x - phase_speed * t) / Lx
        sine = np.sin(phase)
        ones = np.ones_like(x)

    drho = rho0 * amplitude * sine
    rho = rho0 + drho
    if kind == "entropy":
        u = u0 * ones
        p = p0 * ones
    else:
        u = u0 + sign * c * (drho / rho0)
        p = p0 + (c * c) * drho
    return rho, u, p


def sinusoidal_density(
    x: np.ndarray,
    rho0: float = 1.0,
    u0: float = 0.0,
    p0: float = 1.0,
    amplitude: float = 1e-3,
    k: int = 1,
    Lx: float = 1.0,
    gamma: float = 1.4,
    wave: str = "acoustic",
    direction: str = "right",
) -> np.ndarray:
    rho, u, p = acoustic_primitives(
        x, rho0, u0, p0, amplitude, k, Lx, gamma, wave=wave, direction=direction
    )
    return EulerEquations1D(gamma=gamma).conservative(rho, u, p)


class Acoustic1DProblem(BaseProblem):
    """Right-going acoustic eigenmode on a periodic interval.

    ``initial_conditions.wave`` is ``acoustic`` (default) or ``entropy``.
    The entropy option is a stationary density perturbation with uniform
    pressure and velocity. It is not an acoustic wave.
    """

    def create_grid(
        self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False
    ) -> Grid1D:
        """Create 1D periodic grid."""
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

    def _wave_parameters(self) -> Dict[str, Any]:
        ic_config = self.config["initial_conditions"]
        return {
            "rho0": float(ic_config.get("rho0", 1.0)),
            "u0": float(ic_config.get("u0", 0.0)),
            "p0": float(ic_config.get("p0", 1.0)),
            "amplitude": float(ic_config.get("amplitude", 1e-3)),
            "k": int(ic_config.get("k", 1)),
            "wave": str(ic_config.get("wave", "acoustic")),
            "direction": str(ic_config.get("direction", "right")),
        }

    def create_initial_conditions(self, grid: Grid1D):
        """Create the acoustic eigenmode, or a named stationary entropy mode."""
        params = self._wave_parameters()
        return sinusoidal_density(
            x=grid.x,
            rho0=params["rho0"],
            u0=params["u0"],
            p0=params["p0"],
            amplitude=params["amplitude"],
            k=params["k"],
            Lx=grid.Lx,
            gamma=self.gamma,
            wave=params["wave"],
            direction=params["direction"],
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

    def get_solver_class(self):
        """Get 1D solver class."""
        return SpectralSolver1D

    def get_analytical_solution(self, grid: Grid1D, t: float) -> Optional[Dict[str, Any]]:
        """Exact linear translate of the periodic eigenmode.

        The translate assumes a periodic Fourier domain. Wall reflections are
        not included, so this returns None for a non-periodic basis.
        """
        basis = str(self.config["grid"].get("basis", "fourier")).lower()
        if basis != "fourier":
            return None
        params = self._wave_parameters()
        rho, u, p = acoustic_primitives(
            grid.x,
            params["rho0"],
            params["u0"],
            params["p0"],
            params["amplitude"],
            params["k"],
            grid.Lx,
            self.gamma,
            wave=params["wave"],
            direction=params["direction"],
            t=t,
        )
        return {"rho": rho, "u": u, "p": p}
