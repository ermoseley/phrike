#!/usr/bin/env python3
"""
Compare different turbulence spectrum types and wavenumber ranges.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from phrike.grid import Grid3D
from phrike.initial_conditions import turbulent_velocity_3d

# Import and register the custom colormap
import sys
sys.path.append(os.path.dirname(__file__))
from colormaps import register
register('cmapkk9')

def compare_spectra():
    """Compare different turbulence spectrum configurations."""
    
    # Grid setup
    Nx, Ny, Nz = 64, 64, 64
    Lx, Ly, Lz = 1.0, 1.0, 1.0
    grid = Grid3D(Nx=Nx, Ny=Ny, Nz=Nz, Lx=Lx, Ly=Ly, Lz=Lz, dealias=True)
    X, Y, Z = grid.xyz_mesh()
    
    # Configuration 1: Parabolic spectrum, k=[2,16]
    print("Generating parabolic spectrum, k=[2,16]...")
    U1 = turbulent_velocity_3d(X, Y, Z, vrms=0.1, kmin=2.0, kmax=16.0, 
                              alpha=0.3333, spectrum_type="parabolic", seed=42)
    
    # Configuration 2: Power law spectrum, k=[1,5] 
    print("Generating power law spectrum, k=[1,5]...")
    U2 = turbulent_velocity_3d(X, Y, Z, vrms=0.1, kmin=1.0, kmax=5.0, 
                              alpha=0.3333, spectrum_type="power_law", seed=42)
    
    # Configuration 3: Power law spectrum, k=[2,16] (for comparison)
    print("Generating power law spectrum, k=[2,16]...")
    U3 = turbulent_velocity_3d(X, Y, Z, vrms=0.1, kmin=2.0, kmax=16.0, 
                              alpha=0.3333, spectrum_type="power_law", seed=42)
    
    # Extract velocity components
    def get_velocity_components(U):
        from phrike.equations import EulerEquations3D
        eqs = EulerEquations3D(gamma=1.4)
        rho, ux, uy, uz, p = eqs.primitive(U)
        return ux, uy, uz
    
    ux1, uy1, uz1 = get_velocity_components(U1)
    ux2, uy2, uz2 = get_velocity_components(U2)
    ux3, uy3, uz3 = get_velocity_components(U3)
    
    # Calculate velocity magnitudes
    v1 = np.sqrt(ux1**2 + uy1**2 + uz1**2)
    v2 = np.sqrt(ux2**2 + uy2**2 + uz2**2)
    v3 = np.sqrt(ux3**2 + uy3**2 + uz3**2)
    
    # Create comparison plots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), constrained_layout=True)
    
    mid_z = v1.shape[0] // 2
    
    # Row 1: Parabolic spectrum, k=[2,16]
    im1 = axes[0, 0].imshow(v1[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='cmapkk9')
    axes[0, 0].set_title('Parabolic Spectrum\nk=[2,16]')
    axes[0, 0].set_xlabel('x'); axes[0, 0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    
    im2 = axes[0, 1].imshow(ux1[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    axes[0, 1].set_title('ux Component\nParabolic k=[2,16]')
    axes[0, 1].set_xlabel('x'); axes[0, 1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    im3 = axes[0, 2].imshow(uy1[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    axes[0, 2].set_title('uy Component\nParabolic k=[2,16]')
    axes[0, 2].set_xlabel('x'); axes[0, 2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # Row 2: Power law spectrum, k=[1,5]
    im4 = axes[1, 0].imshow(v2[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='cmapkk9')
    axes[1, 0].set_title('Power Law Spectrum\nk=[1,5]')
    axes[1, 0].set_xlabel('x'); axes[1, 0].set_ylabel('y')
    plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    
    im5 = axes[1, 1].imshow(ux2[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    axes[1, 1].set_title('ux Component\nPower Law k=[1,5]')
    axes[1, 1].set_xlabel('x'); axes[1, 1].set_ylabel('y')
    plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    
    im6 = axes[1, 2].imshow(uy2[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    axes[1, 2].set_title('uy Component\nPower Law k=[1,5]')
    axes[1, 2].set_xlabel('x'); axes[1, 2].set_ylabel('y')
    plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    
    # Row 3: Power law spectrum, k=[2,16] (for comparison)
    im7 = axes[2, 0].imshow(v3[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='cmapkk9')
    axes[2, 0].set_title('Power Law Spectrum\nk=[2,16]')
    axes[2, 0].set_xlabel('x'); axes[2, 0].set_ylabel('y')
    plt.colorbar(im7, ax=axes[2, 0], shrink=0.8)
    
    im8 = axes[2, 1].imshow(ux3[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    axes[2, 1].set_title('ux Component\nPower Law k=[2,16]')
    axes[2, 1].set_xlabel('x'); axes[2, 1].set_ylabel('y')
    plt.colorbar(im8, ax=axes[2, 1], shrink=0.8)
    
    im9 = axes[2, 2].imshow(uy3[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    axes[2, 2].set_title('uy Component\nPower Law k=[2,16]')
    axes[2, 2].set_xlabel('x'); axes[2, 2].set_ylabel('y')
    plt.colorbar(im9, ax=axes[2, 2], shrink=0.8)
    
    plt.suptitle('Turbulence Spectrum Comparison\nAll with vrms=0.1, α=0.3333', 
                 fontsize=16, fontweight='bold')
    plt.savefig('outputs/turb3d/spectrum_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    print(f"\nTurbulence Statistics:")
    print(f"Parabolic k=[2,16]:  RMS velocity = {np.sqrt(np.mean(v1**2)):.4f}")
    print(f"Power Law k=[1,5]:   RMS velocity = {np.sqrt(np.mean(v2**2)):.4f}")
    print(f"Power Law k=[2,16]:  RMS velocity = {np.sqrt(np.mean(v3**2)):.4f}")
    
    print(f"\nVelocity magnitude ranges:")
    print(f"Parabolic k=[2,16]:  min={np.min(v1):.4f}, max={np.max(v1):.4f}")
    print(f"Power Law k=[1,5]:   min={np.min(v2):.4f}, max={np.max(v2):.4f}")
    print(f"Power Law k=[2,16]:  min={np.min(v3):.4f}, max={np.max(v3):.4f}")

if __name__ == "__main__":
    print("Comparing turbulence spectrum types...")
    compare_spectra()
    print("Comparison complete!")
