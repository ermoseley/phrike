"""3D turbulent velocity field problem."""

import os
import sys
from typing import Optional, Dict, Any

import matplotlib.pyplot as plt
import numpy as np

from hydra.grid import Grid3D
from hydra.equations import EulerEquations3D
from hydra.initial_conditions import turbulent_velocity_3d
from hydra.solver import SpectralSolver3D
from hydra.io import save_solution_snapshot
from .base import BaseProblem


class Turb3DProblem(BaseProblem):
    """3D turbulent velocity field problem."""
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize with custom colormap registration."""
        super().__init__(config_path, config)
        
        # Import and register the custom colormap
        sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        try:
            from colormaps import register
            register('cmapkk9')
        except ImportError:
            pass  # colormap registration is optional
    
    def create_grid(self, backend: str = "numpy", device: Optional[str] = None) -> Grid3D:
        """Create 3D grid."""
        Nx = int(self.config["grid"]["Nx"])
        Ny = int(self.config["grid"]["Ny"])
        Nz = int(self.config["grid"]["Nz"])
        Lx = float(self.config["grid"]["Lx"])
        Ly = float(self.config["grid"]["Ly"])
        Lz = float(self.config["grid"]["Lz"])
        dealias = bool(self.config["grid"].get("dealias", True))
        
        return Grid3D(
            Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz,
            dealias=dealias,
            filter_params=self.filter_config,
            fft_workers=self.fft_workers,
            backend=backend,
            torch_device=device
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
        power_law_slope = float(icfg.get("power_law_slope", -5.0/3.0))
        seed = int(icfg.get("seed", 42))
        
        return turbulent_velocity_3d(
            X, Y, Z,
            rho0=rho0, p0=p0, vrms=vrms, kmin=kmin, kmax=kmax,
            alpha=alpha, spectrum_type=spectrum_type, power_law_slope=power_law_slope,
            seed=seed, gamma=self.gamma
        )
    
    def create_visualization(self, solver, t: float, U) -> None:
        """Create visualization for current state."""
        rho, ux, uy, uz, p = solver.equations.primitive(U)
        rho = self.convert_torch_to_numpy(rho)[0]
        
        # Take mid-plane slice
        mid = rho.shape[0] // 2
        
        # Create frame
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        try:
            im = ax.imshow(rho[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                          aspect='equal', cmap='cmapkk9')
        except ValueError:
            # Fallback to default colormap if custom one not available
            im = ax.imshow(rho[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                          aspect='equal')
        
        icfg = self.config["initial_conditions"]
        vrms = float(icfg.get("vrms", 0.1))
        alpha = float(icfg.get("alpha", 0.3333))
        
        ax.set_title(f"Turbulent Density (z=mid) t={t:.3f}\nvrms={vrms:.3f}, α={alpha:.3f}")
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        plt.colorbar(im, ax=ax, shrink=0.8)
        fig.canvas.draw()
        fig.canvas.flush_events()
        
        # Save frame
        frames_dir = os.path.join(self.outdir, "frames")
        fig.savefig(os.path.join(frames_dir, f"frame_{t:08.3f}.png"), dpi=120)
        plt.close(fig)
        
        # Save snapshot
        snapshot_path = save_solution_snapshot(
            self.outdir, t, U=U, grid=solver.grid, equations=solver.equations
        )
        print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")
    
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
            im1 = axes[0, 0].imshow(rho[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                                   aspect='equal', cmap='cmapkk9')
        except ValueError:
            im1 = axes[0, 0].imshow(rho[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                                   aspect='equal')
        axes[0, 0].set_title(f'Density (z=mid)\nt={solver.t:.3f}')
        axes[0, 0].set_xlabel('x')
        axes[0, 0].set_ylabel('y')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        
        # Velocity magnitude
        v_mag = np.sqrt(ux**2 + uy**2 + uz**2)
        try:
            im2 = axes[0, 1].imshow(v_mag[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                                   aspect='equal', cmap='cmapkk9')
        except ValueError:
            im2 = axes[0, 1].imshow(v_mag[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                                   aspect='equal')
        axes[0, 1].set_title(f'Velocity Magnitude (z=mid)\nt={solver.t:.3f}')
        axes[0, 1].set_xlabel('x')
        axes[0, 1].set_ylabel('y')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        
        # Pressure
        try:
            im3 = axes[0, 2].imshow(p[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                                   aspect='equal', cmap='cmapkk9')
        except ValueError:
            im3 = axes[0, 2].imshow(p[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                                   aspect='equal')
        axes[0, 2].set_title(f'Pressure (z=mid)\nt={solver.t:.3f}')
        axes[0, 2].set_xlabel('x')
        axes[0, 2].set_ylabel('y')
        plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        
        # Velocity components
        im4 = axes[1, 0].imshow(ux[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                               aspect='equal', cmap='RdBu_r')
        axes[1, 0].set_title(f'ux (z=mid)\nt={solver.t:.3f}')
        axes[1, 0].set_xlabel('x')
        axes[1, 0].set_ylabel('y')
        plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        
        im5 = axes[1, 1].imshow(uy[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                               aspect='equal', cmap='RdBu_r')
        axes[1, 1].set_title(f'uy (z=mid)\nt={solver.t:.3f}')
        axes[1, 1].set_xlabel('x')
        axes[1, 1].set_ylabel('y')
        plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
        
        im6 = axes[1, 2].imshow(uz[mid], origin='lower', extent=[0, solver.grid.Lx, 0, solver.grid.Ly], 
                               aspect='equal', cmap='RdBu_r')
        axes[1, 2].set_title(f'uz (z=mid)\nt={solver.t:.3f}')
        axes[1, 2].set_xlabel('x')
        axes[1, 2].set_ylabel('y')
        plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
        
        plt.suptitle(f'3D Turbulent Velocity Field (GPU/MPS)\nvrms={vrms:.3f}, α={alpha:.3f}, k=[{kmin:.1f},{kmax:.1f}]', 
                     fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.outdir, f"turb3d_analysis_t{solver.t:.3f}.png"), dpi=150, bbox_inches='tight')
        plt.close()
    
    def get_solver_class(self):
        """Get 3D solver class."""
        return SpectralSolver3D
