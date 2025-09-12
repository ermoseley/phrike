#!/usr/bin/env python3
"""
Side-by-side comparison of RAMSES and HYDRA KHI simulations using matplotlib.
This script generates comparison frames showing RAMSES (left) vs HYDRA (right).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# Add ramses-utils to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'outputs', 'khi2d', 'ramses-utils'))

import miniramses as ram

# Import and register the custom colormap
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from colormaps import cmaps, register
register('cmapkk10')


def read_ramses_data(output_num, ramses_path, variable=0):
    """Read RAMSES data for a specific output number."""
    try:
        c = ram.rd_cell(output_num, path=ramses_path, prefix="hydro")
        # Extract 2D data (assuming z-projection)
        ii, jj = 1, 2  # x and y coordinates for z-projection
        
        # Get coordinates and data
        x = c.x[ii-1]  # x coordinates
        y = c.x[jj-1]  # y coordinates
        dx = c.dx      # cell sizes
        data = c.u[variable]  # variable data (0=density, 1=momentum_x, 2=momentum_y, 3=energy, 4=pressure)
        
        # Create a regular grid for plotting
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        # Use a reasonable resolution for the plot
        nx, ny = 512, 512
        x_grid = np.linspace(x_min, x_max, nx)
        y_grid = np.linspace(y_min, y_max, ny)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Interpolate data to regular grid (simple nearest neighbor)
        from scipy.interpolate import griddata
        points = np.column_stack([x, y])
        data_grid = griddata(points, data, (X_grid, Y_grid), method='nearest')
        
        return X_grid, Y_grid, data_grid, x_min, x_max, y_min, y_max
        
    except Exception as e:
        print(f"Error reading RAMSES output {output_num}: {e}")
        return None, None, None, None, None, None, None


def read_hydra_data(snapshot_path):
    """Read HYDRA data from snapshot."""
    try:
        data = np.load(snapshot_path, allow_pickle=True)
        U = data['U']
        
        # Get primitive variables directly from saved data
        rho = data['rho']
        ux = data['ux'] 
        uy = data['uy']
        p = data['p']
        
        # Get grid coordinates
        x = data['x']
        y = data['y']
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Get domain size from meta
        meta = data['meta'].item()
        Lx = meta['Lx']
        Ly = meta['Ly']
        
        return X, Y, rho, ux, uy, p, Lx, Ly
        
    except Exception as e:
        print(f"Error reading HYDRA snapshot: {e}")
        return None, None, None, None, None, None, None, None


def create_comparison_frame(frame_idx, ramses_path, hydra_snapshots, output_dir, 
                          variable_name="density", variable_idx=0):
    """Create a single comparison frame."""
    
    # Calculate time from frame index
    time = frame_idx * 0.02
    
    # Read RAMSES data
    ramses_output = f"output_{frame_idx + 1:05d}"
    ramses_data_path = os.path.join(ramses_path, ramses_output)
    
    if not os.path.exists(ramses_data_path):
        print(f"RAMSES output {ramses_output} not found")
        return False
    
    X_ram, Y_ram, data_ram, x_min, x_max, y_min, y_max = read_ramses_data(
        frame_idx + 1, ramses_path, variable_idx
    )
    
    if X_ram is None:
        print(f"Failed to read RAMSES data for frame {frame_idx}")
        return False
    
    # Read HYDRA data
    if frame_idx >= len(hydra_snapshots):
        print(f"HYDRA snapshot {frame_idx} not found")
        return False
    
    snapshot_path = hydra_snapshots[frame_idx]
    X_hyd, Y_hyd, rho_hyd, ux_hyd, uy_hyd, p_hyd, Lx, Ly = read_hydra_data(snapshot_path)
    
    if X_hyd is None:
        print(f"Failed to read HYDRA data for frame {frame_idx}")
        return False
    
    # Select the appropriate variable for HYDRA
    if variable_idx == 0:  # density
        data_hyd = rho_hyd
    elif variable_idx == 1:  # momentum x
        data_hyd = rho_hyd * ux_hyd
    elif variable_idx == 2:  # momentum y
        data_hyd = rho_hyd * uy_hyd
    elif variable_idx == 3:  # energy
        data_hyd = p_hyd / (1.4 - 1) + 0.5 * rho_hyd * (ux_hyd**2 + uy_hyd**2)
    elif variable_idx == 4:  # pressure
        data_hyd = p_hyd
    else:
        data_hyd = rho_hyd
    
    # Create the comparison plot
    fig = plt.figure(figsize=(16, 8))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], hspace=0.1, wspace=0.1)
    
    # RAMSES plot (left)
    ax1 = fig.add_subplot(gs[0])
    im1 = ax1.imshow(data_ram, origin='lower', extent=[x_min, x_max, y_min, y_max], 
                     aspect='equal', cmap='cmapkk10')
    ax1.set_title(f'RAMSES\n{variable_name.title()}\nt = {time:.3f}', fontsize=14, fontweight='bold')
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    
    # Add large RAMSES label
    ax1.text(0.02, 0.98, 'RAMSES', transform=ax1.transAxes, fontsize=20, 
             fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.3', 
             facecolor='black', alpha=0.7), verticalalignment='top')
    
    # Add colorbar for RAMSES
    cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
    cbar1.set_label(variable_name.title(), fontsize=12)
    
    # HYDRA plot (right)
    ax2 = fig.add_subplot(gs[1])
    im2 = ax2.imshow(data_hyd, origin='lower', extent=[0, Lx, 0, Ly], 
                     aspect='equal', cmap='cmapkk10')
    ax2.set_title(f'HYDRA\n{variable_name.title()}\nt = {time:.3f}', fontsize=14, fontweight='bold')
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    
    # Add large HYDRA label
    ax2.text(0.02, 0.98, 'HYDRA', transform=ax2.transAxes, fontsize=20, 
             fontweight='bold', color='white', bbox=dict(boxstyle='round,pad=0.3', 
             facecolor='black', alpha=0.7), verticalalignment='top')
    
    # Add colorbar for HYDRA
    cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
    cbar2.set_label(variable_name.title(), fontsize=12)
    
    # Set consistent color scales with fixed range
    vmin = 0.75
    vmax = 2.5
    im1.set_clim(vmin, vmax)
    im2.set_clim(vmin, vmax)
    
    # Overall title
    fig.suptitle(f'KHI Simulation Comparison: RAMSES vs HYDRA - {variable_name.title()}', 
                 fontsize=16, fontweight='bold', y=0.95)
    
    # Save the frame
    frame_path = os.path.join(output_dir, f'comparison_frame_{frame_idx:05d}.png')
    plt.savefig(frame_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Created comparison frame {frame_idx:05d} at t={time:.3f}")
    return True


def main():
    """Main function to generate comparison frames."""
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ramses_path = os.path.join(base_dir, 'outputs', 'khi2d', 'khi_ramses')
    hydra_snapshots_dir = os.path.join(base_dir, 'outputs', 'khi2d')
    output_dir = os.path.join(base_dir, 'outputs', 'khi2d', 'comparison_frames')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find HYDRA snapshots by scanning the directory
    hydra_snapshots = []
    snapshot_files = sorted([f for f in os.listdir(hydra_snapshots_dir) if f.startswith('snapshot_t') and f.endswith('.npz')])
    
    for snapshot_file in snapshot_files:
        snapshot_path = os.path.join(hydra_snapshots_dir, snapshot_file)
        hydra_snapshots.append(snapshot_path)
    
    print(f"Found {len(hydra_snapshots)} HYDRA snapshots")
    
    # Create comparison frames for all time steps
    if len(hydra_snapshots) > 0:
        print(f"Creating comparison frames for {len(hydra_snapshots)} time steps...")
        
        # Only generate density frames
        var_name = "density"
        var_idx = 0
        
        print(f"\nGenerating comparison frames for {var_name}...")
        
        var_output_dir = os.path.join(output_dir, var_name)
        os.makedirs(var_output_dir, exist_ok=True)
        
        success_count = 0
        for frame_idx in range(len(hydra_snapshots)):
            if create_comparison_frame(frame_idx, ramses_path, hydra_snapshots, 
                                    var_output_dir, var_name, var_idx):
                success_count += 1
        
        print(f"Generated {success_count}/{len(hydra_snapshots)} frames for {var_name}")
    
    print(f"\nComparison frames saved to: {output_dir}")
    print("You can now create videos from these frames using ffmpeg.")


if __name__ == "__main__":
    main()
