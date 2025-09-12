#!/usr/bin/env python3
"""
Analysis script for 3D Taylor-Green vortex simulation results.
Creates various plots showing the evolution of the flow field.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

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

def create_evolution_plots():
    """Create plots showing the evolution of the TGV."""
    output_dir = "outputs/tgv3d"
    
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
    fig = plt.figure(figsize=(20, 15))
    gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Density evolution at mid-planes
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(initial['rho'][mid_z], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal')
    ax1.set_title(f'Initial Density (z-mid)\nt={initial["t"]:.3f}')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    plt.colorbar(im1, ax=ax1, shrink=0.8)
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(final['rho'][mid_z], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal')
    ax2.set_title(f'Final Density (z-mid)\nt={final["t"]:.3f}')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # 2. Velocity magnitude evolution
    v_initial = np.sqrt(initial['ux']**2 + initial['uy']**2 + initial['uz']**2)
    v_final = np.sqrt(final['ux']**2 + final['uy']**2 + final['uz']**2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(v_initial[mid_z], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal')
    ax3.set_title(f'Initial |V| (z-mid)\nt={initial["t"]:.3f}')
    ax3.set_xlabel('x'); ax3.set_ylabel('y')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    ax4 = fig.add_subplot(gs[0, 3])
    im4 = ax4.imshow(v_final[mid_z], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal')
    ax4.set_title(f'Final |V| (z-mid)\nt={final["t"]:.3f}')
    ax4.set_xlabel('x'); ax4.set_ylabel('y')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    # 3. Pressure evolution
    ax5 = fig.add_subplot(gs[1, 0])
    im5 = ax5.imshow(initial['p'][mid_z], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal')
    ax5.set_title(f'Initial Pressure (z-mid)\nt={initial["t"]:.3f}')
    ax5.set_xlabel('x'); ax5.set_ylabel('y')
    plt.colorbar(im5, ax=ax5, shrink=0.8)
    
    ax6 = fig.add_subplot(gs[1, 1])
    im6 = ax6.imshow(final['p'][mid_z], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal')
    ax6.set_title(f'Final Pressure (z-mid)\nt={final["t"]:.3f}')
    ax6.set_xlabel('x'); ax6.set_ylabel('y')
    plt.colorbar(im6, ax=ax6, shrink=0.8)
    
    # 4. Velocity components at final time
    ax7 = fig.add_subplot(gs[1, 2])
    im7 = ax7.imshow(final['ux'][mid_z], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal', cmap='RdBu_r')
    ax7.set_title(f'Final ux (z-mid)\nt={final["t"]:.3f}')
    ax7.set_xlabel('x'); ax7.set_ylabel('y')
    plt.colorbar(im7, ax=ax7, shrink=0.8)
    
    ax8 = fig.add_subplot(gs[1, 3])
    im8 = ax8.imshow(final['uy'][mid_z], origin='lower', extent=[0, 2*np.pi, 0, 2*np.pi], aspect='equal', cmap='RdBu_r')
    ax8.set_title(f'Final uy (z-mid)\nt={final["t"]:.3f}')
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
    
    # 6. Conservation analysis
    ax10 = fig.add_subplot(gs[2, 2:])
    
    # Load all snapshots for conservation analysis
    times = []
    masses = []
    energies = []
    momentum_x = []
    momentum_y = []
    momentum_z = []
    
    for snapshot_path in snapshots[::5]:  # Sample every 5th snapshot
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
    
    # Normalize by initial values (handle zero initial momentum)
    masses = masses / masses[0]
    energies = energies / energies[0]
    
    # For momentum, use absolute values since initial momentum is zero
    momentum_x = np.abs(momentum_x)
    momentum_y = np.abs(momentum_y) 
    momentum_z = np.abs(momentum_z)
    
    ax10.plot(times, masses, 'b-', label='Mass', linewidth=2)
    ax10.plot(times, energies, 'r-', label='Energy', linewidth=2)
    ax10.plot(times, momentum_x, 'g--', label='Momentum X', linewidth=1)
    ax10.plot(times, momentum_y, 'g:', label='Momentum Y', linewidth=1)
    ax10.plot(times, momentum_z, 'g-.', label='Momentum Z', linewidth=1)
    ax10.set_xlabel('Time')
    ax10.set_ylabel('Conserved Quantities')
    ax10.set_title('Conservation Analysis')
    ax10.legend()
    ax10.grid(True, alpha=0.3)
    # Adjust y-limits for momentum plots
    ax10.set_ylim(0.95, 1.05)
    
    plt.suptitle('3D Taylor-Green Vortex Evolution (GPU/MPS)', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'tgv3d_analysis.png'), dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Analysis complete! Plots saved to {output_dir}/tgv3d_analysis.png")
    
    # Print some statistics
    print(f"\nConservation Statistics:")
    print(f"Mass conservation error: {abs(masses[-1] - 1.0):.2e}")
    print(f"Energy conservation error: {abs(energies[-1] - 1.0):.2e}")
    print(f"Max momentum X: {np.max(momentum_x):.2e}")
    print(f"Max momentum Y: {np.max(momentum_y):.2e}")
    print(f"Max momentum Z: {np.max(momentum_z):.2e}")

def create_velocity_field_plot():
    """Create a detailed velocity field visualization."""
    output_dir = "outputs/tgv3d"
    snapshots = sorted(glob.glob(os.path.join(output_dir, "snapshot_t*.npz")))
    
    if len(snapshots) == 0:
        return
    
    # Load final snapshot
    final = load_snapshot(snapshots[-1])
    mid_z = final['rho'].shape[0] // 2
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Velocity field with streamlines
    x = final['x']
    y = final['y']
    ux_slice = final['ux'][mid_z]
    uy_slice = final['uy'][mid_z]
    
    # Create meshgrid for quiver plot
    X, Y = np.meshgrid(x, y, indexing='xy')
    
    # Subsample for cleaner quiver plot
    step = 4
    ax1.quiver(X[::step, ::step], Y[::step, ::step], 
               ux_slice[::step, ::step], uy_slice[::step, ::step],
               alpha=0.7, scale=50)
    ax1.set_title(f'Velocity Field (z-mid)\nt={final["t"]:.3f}')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')
    ax1.set_aspect('equal')
    
    # Streamlines
    ax2.streamplot(X, Y, ux_slice, uy_slice, density=1.5, color='black')
    ax2.set_title(f'Streamlines (z-mid)\nt={final["t"]:.3f}')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tgv3d_velocity_field.png'), dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Analyzing 3D Taylor-Green Vortex simulation results...")
    create_evolution_plots()
    create_velocity_field_plot()
    print("Analysis complete!")
