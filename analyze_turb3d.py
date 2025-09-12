#!/usr/bin/env python3
"""
Analysis script for 3D turbulent velocity field simulation results.
Creates various plots showing the evolution of the turbulent flow field.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation, PillowWriter

# Import and register the custom colormap
import sys
sys.path.append(os.path.dirname(__file__))
from colormaps import register
register('cmapkk9')

def load_snapshot(snapshot_path):
    """Load a single snapshot and return data."""
    data = np.load(snapshot_path, allow_pickle=True)
    return {
        't': data['t'],
        'x': data['x'],
        'y': data['y'], 
        'z': data['z'],
        'rho': data['rho'],
        'ux': data['ux'],
        'uy': data['uy'],
        'uz': data['uz'],
        'p': data['p'],
        'meta': data['meta'].item()
    }

def create_turbulence_analysis():
    """Create comprehensive turbulence analysis plots."""
    output_dir = "outputs/turb3d"
    
    # Find all snapshots
    snapshots = sorted(glob.glob(os.path.join(output_dir, "snapshot_t*.npz")))
    print(f"Found {len(snapshots)} snapshots")
    
    if len(snapshots) == 0:
        print("No snapshots found!")
        return
    
    # Load initial and final snapshots
    initial = load_snapshot(snapshots[0])
    final = load_snapshot(snapshots[-1])
    
    print(f"Initial time: {initial['t']:.3f}")
    print(f"Final time: {final['t']:.3f}")
    print(f"Grid size: {initial['meta']['Nx']} x {initial['meta']['Ny']} x {initial['meta']['Nz']}")
    
    # Get mid-plane indices
    mid_z = initial['rho'].shape[0] // 2
    mid_y = initial['rho'].shape[1] // 2
    mid_x = initial['rho'].shape[2] // 2
    
    # Create comprehensive analysis plots
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Density evolution at mid-planes
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(initial['rho'][mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='cmapkk9')
    ax1.set_title(f'Initial Density (z-mid)\nt={initial["t"]:.3f}')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(final['rho'][mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='cmapkk9')
    ax2.set_title(f'Final Density (z-mid)\nt={final["t"]:.3f}')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 2. Velocity magnitude evolution
    v_initial = np.sqrt(initial['ux']**2 + initial['uy']**2 + initial['uz']**2)
    v_final = np.sqrt(final['ux']**2 + final['uy']**2 + final['uz']**2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(v_initial[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='cmapkk9')
    ax3.set_title(f'Initial |V| (z-mid)\nt={initial["t"]:.3f}')
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(v_final[mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='cmapkk9')
    ax4.set_title(f'Final |V| (z-mid)\nt={final["t"]:.3f}')
    ax4.set_xlabel('x'); ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # 3. Velocity components at final time
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(final['ux'][mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    ax5.set_title(f'Final ux (z-mid)\nt={final["t"]:.3f}')
    ax5.set_xlabel('x'); ax5.set_ylabel('y')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(final['uy'][mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    ax6.set_title(f'Final uy (z-mid)\nt={final["t"]:.3f}')
    ax6.set_xlabel('x'); ax6.set_ylabel('y')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(final['uz'][mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='RdBu_r')
    ax7.set_title(f'Final uz (z-mid)\nt={final["t"]:.3f}')
    ax7.set_xlabel('x'); ax7.set_ylabel('y')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    # 4. Pressure evolution
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(final['p'][mid_z], origin='lower', extent=[0, 1, 0, 1], aspect='equal', cmap='cmapkk9')
    ax8.set_title(f'Final Pressure (z-mid)\nt={final["t"]:.3f}')
    ax8.set_xlabel('x'); ax8.set_ylabel('y')
    plt.colorbar(im8, ax=ax8, shrink=0.8)
    
    # 5. 3D isosurface visualization (projected)
    ax9 = fig.add_subplot(gs[2, :2])
    
    # Create a 3D-like visualization by showing multiple z-slices
    z_slices = [0, mid_z//2, mid_z, mid_z + mid_z//2, -1]
    colors = ['red', 'orange', 'yellow', 'green', 'blue']
    
    for i, (z_idx, color) in enumerate(zip(z_slices, colors)):
        if z_idx < final['rho'].shape[0]:
            alpha = 0.3 + 0.7 * (i / (len(z_slices) - 1))
            ax9.contour(final['x'], final['y'], final['rho'][z_idx], 
                       levels=10, colors=color, alpha=alpha, linewidths=1)
    
    ax9.set_title(f'3D Density Contours (Multiple Z-slices)\nt={final["t"]:.3f}')
    ax9.set_xlabel('x'); ax9.set_ylabel('y')
    ax9.set_aspect('equal')
    
    # 6. Velocity field with streamlines
    ax10 = fig.add_subplot(gs[2, 2:])
    
    # Create meshgrid for quiver plot
    x = final['x']
    y = final['y']
    ux_slice = final['ux'][mid_z]
    uy_slice = final['uy'][mid_z]
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Subsample for cleaner quiver plot
    step = 4
    ax10.quiver(X[::step, ::step], Y[::step, ::step], 
               ux_slice[::step, ::step], uy_slice[::step, ::step],
               alpha=0.7, scale=50)
    ax10.set_title(f'Velocity Field (z-mid)\nt={final["t"]:.3f}')
    ax10.set_xlabel('x'); ax10.set_ylabel('y')
    ax10.set_aspect('equal')
    
    # 7. Conservation analysis
    ax11 = fig.add_subplot(gs[3, :2])
    
    # Load all snapshots for conservation analysis
    times = []
    masses = []
    energies = []
    momentum_x = []
    momentum_y = []
    momentum_z = []
    
    for snapshot_path in snapshots[::2]:  # Sample every 2nd snapshot
        data = load_snapshot(snapshot_path)
        times.append(data['t'])
        
        # Calculate conserved quantities
        rho = data['rho']
        ux = data['ux']
        uy = data['uy']
        uz = data['uz']
        p = data['p']
        gamma = data['meta']['gamma']
        
        masses.append(np.sum(rho))
        energies.append(np.sum(p / (gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2 + uz**2)))
        momentum_x.append(np.sum(rho * ux))
        momentum_y.append(np.sum(rho * uy))
        momentum_z.append(np.sum(rho * uz))
    
    times = np.array(times)
    masses = np.array(masses)
    energies = np.array(energies)
    momentum_x = np.array(momentum_x)
    momentum_y = np.array(momentum_y)
    momentum_z = np.array(momentum_z)
    
    # Normalize by initial values
    masses = masses / masses[0]
    energies = energies / energies[0]
    
    # For momentum, use absolute values since initial momentum is zero
    momentum_x = np.abs(momentum_x)
    momentum_y = np.abs(momentum_y) 
    momentum_z = np.abs(momentum_z)
    
    ax11.plot(times, masses, 'b-', label='Mass', linewidth=2)
    ax11.plot(times, energies, 'r-', label='Energy', linewidth=2)
    ax11.plot(times, momentum_x, 'g--', label='Momentum X', linewidth=1)
    ax11.plot(times, momentum_y, 'g:', label='Momentum Y', linewidth=1)
    ax11.plot(times, momentum_z, 'g-.', label='Momentum Z', linewidth=1)
    ax11.set_xlabel('Time')
    ax11.set_ylabel('Conserved Quantities')
    ax11.set_title('Conservation Analysis')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim(0.95, 1.05)
    
    # 8. Power spectrum analysis
    ax12 = fig.add_subplot(gs[3, 2:])
    
    # Calculate power spectrum of velocity field
    def power_spectrum_3d(ux, uy, uz):
        # FFT of velocity components
        Ux = np.fft.fftn(ux)
        Uy = np.fft.fftn(uy)
        Uz = np.fft.fftn(uz)
        
        # Power spectrum
        P = np.abs(Ux)**2 + np.abs(Uy)**2 + np.abs(Uz)**2
        
        # Get k values
        nx, ny, nz = ux.shape
        kx = np.fft.fftfreq(nx, d=1.0/nx)
        ky = np.fft.fftfreq(ny, d=1.0/ny)
        kz = np.fft.fftfreq(nz, d=1.0/nz)
        
        # Create k magnitude array
        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(Kx**2 + Ky**2 + Kz**2)
        
        # Bin the power spectrum
        k_bins = np.logspace(0, np.log10(nx/2), 20)
        P_binned = np.zeros(len(k_bins)-1)
        k_centers = np.zeros(len(k_bins)-1)
        
        for i in range(len(k_bins)-1):
            mask = (K >= k_bins[i]) & (K < k_bins[i+1])
            P_binned[i] = np.mean(P[mask])
            k_centers[i] = np.sqrt(k_bins[i] * k_bins[i+1])
        
        return k_centers, P_binned
    
    k_centers, P_spectrum = power_spectrum_3d(final['ux'], final['uy'], final['uz'])
    
    ax12.loglog(k_centers, P_spectrum, 'b-', linewidth=2, label='Velocity Power Spectrum')
    ax12.loglog(k_centers, k_centers**(-5/3), 'r--', linewidth=2, label='Kolmogorov -5/3')
    ax12.set_xlabel('Wavenumber k')
    ax12.set_ylabel('Power Spectrum P(k)')
    ax12.set_title('Velocity Power Spectrum')
    ax12.legend()
    ax12.grid(True, alpha=0.3)
    
    plt.suptitle('3D Turbulent Velocity Field Analysis (GPU/MPS)', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'turb3d_comprehensive_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Analysis complete! Plots saved to {output_dir}/turb3d_comprehensive_analysis.png")
    
    # Print some statistics
    print(f"\nConservation Statistics:")
    print(f"Mass conservation error: {abs(masses[-1] - 1.0):.2e}")
    print(f"Energy conservation error: {abs(energies[-1] - 1.0):.2e}")
    print(f"Max momentum X: {np.max(momentum_x):.2e}")
    print(f"Max momentum Y: {np.max(momentum_y):.2e}")
    print(f"Max momentum Z: {np.max(momentum_z):.2e}")
    
    # Print turbulence statistics
    print(f"\nTurbulence Statistics:")
    print(f"Initial RMS velocity: {np.sqrt(np.mean(v_initial**2)):.4f}")
    print(f"Final RMS velocity: {np.sqrt(np.mean(v_final**2)):.4f}")
    print(f"Initial density std: {np.std(initial['rho']):.4f}")
    print(f"Final density std: {np.std(final['rho']):.4f}")

def create_turbulence_animation():
    """Create an animated GIF from the frame sequence."""
    frames_dir = "outputs/turb3d/frames"
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    
    if len(frame_files) == 0:
        print("No frame files found!")
        return
    
    print(f"Found {len(frame_files)} frames")
    
    # Load all frames
    frames = []
    for frame_file in frame_files:
        from PIL import Image
        img = Image.open(frame_file)
        frames.append(np.array(img))
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    def animate(frame):
        ax.clear()
        ax.axis('off')
        ax.imshow(frames[frame])
        # Extract time from filename
        time_str = os.path.basename(frame_files[frame]).split('_')[1].split('.')[0]
        time_val = float(time_str)
        ax.set_title(f'3D Turbulent Velocity Field Evolution (GPU/MPS)\nTime: {time_val:.3f}', 
                    fontsize=14, fontweight='bold')
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=200, repeat=True)
    
    # Save as GIF
    output_path = "outputs/turb3d/turb3d_animation.gif"
    anim.save(output_path, writer=PillowWriter(fps=5))
    print(f"Animation saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    print("Analyzing 3D turbulent velocity field simulation results...")
    create_turbulence_analysis()
    create_turbulence_animation()
    print("Analysis complete!")
