
import numpy as np
import matplotlib.pyplot as plt
import h5py
import glob
import os

def plot_khi2d_snapshots(snapshots_dir):
    """Plot snapshots from the KHI2D simulation."""
    
    # Find all snapshot files
    snapshot_files = sorted(glob.glob(os.path.join(snapshots_dir, "snapshots_s*.h5")))
    
    if not snapshot_files:
        print("No snapshot files found!")
        return
    
    print(f"Found {len(snapshot_files)} snapshot files")
    
    # Create output directory for plots
    plot_dir = os.path.join(os.path.dirname(snapshots_dir), "plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    for snapshot_file in snapshot_files:
        print(f"Processing {snapshot_file}")
        
        with h5py.File(snapshot_file, 'r') as f:
            # Get all time steps
            times = f['scales']['sim_time'][:]
            n_times = len(times)
            
            print(f"Found {n_times} time steps from t={times[0]:.3f} to t={times[-1]:.3f}")
            
            # Plot every 10th time step to avoid too many plots
            step = max(1, n_times // 50)  # Show up to 50 frames
            
            for i in range(0, n_times, step):
                       # Get data for this time step
                       # Note: Transpose x and y axes for correct plotting
                       # Dedalus stores data as [time, y, x], so transpose to get [x, y]
                       density = f['tasks']['density'][i, :, :].T
                       velocity = f['tasks']['velocity'][i, :, :, :].T  # [2, Nx, Ny] after transpose
                       pressure = f['tasks']['pressure'][i, :, :].T
                       vorticity = f['tasks']['vorticity'][i, :, :].T
                
                # Get time
                time = times[i]
                
                # Create figure
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f'KHI2D Simulation at t={time:.3f}', fontsize=16)
                
                # Density
                im1 = axes[0, 0].imshow(density, origin='lower', aspect='equal', cmap='viridis')
                axes[0, 0].set_title('Density')
                axes[0, 0].set_xlabel('x')
                axes[0, 0].set_ylabel('y')
                plt.colorbar(im1, ax=axes[0, 0])
                
                       # Velocity magnitude (after transpose, velocity is [2, Nx, Ny])
                       vel_mag = np.sqrt(velocity[0]**2 + velocity[1]**2)
                im2 = axes[0, 1].imshow(vel_mag, origin='lower', aspect='equal', cmap='plasma')
                axes[0, 1].set_title('Velocity Magnitude')
                axes[0, 1].set_xlabel('x')
                axes[0, 1].set_ylabel('y')
                plt.colorbar(im2, ax=axes[0, 1])
                
                # Pressure
                im3 = axes[1, 0].imshow(pressure, origin='lower', aspect='equal', cmap='coolwarm')
                axes[1, 0].set_title('Pressure')
                axes[1, 0].set_xlabel('x')
                axes[1, 0].set_ylabel('y')
                plt.colorbar(im3, ax=axes[1, 0])
                
                # Vorticity
                im4 = axes[1, 1].imshow(vorticity, origin='lower', aspect='equal', cmap='RdBu_r')
                axes[1, 1].set_title('Vorticity')
                axes[1, 1].set_xlabel('x')
                axes[1, 1].set_ylabel('y')
                plt.colorbar(im4, ax=axes[1, 1])
                
                plt.tight_layout()
                
                # Save plot
                plot_file = os.path.join(plot_dir, f"khi2d_frame_{i:04d}_t{time:.3f}.png")
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                plt.close()
                
                print(f"Saved plot: {plot_file}")

if __name__ == "__main__":
    plot_khi2d_snapshots("khi2d_dedalus_output/snapshots")
