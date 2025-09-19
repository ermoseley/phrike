
#!/usr/bin/env python3
"""
Create a corrected density-only video from the KHI2D Dedalus simulation.
Fixes the coordinate system and domain mapping issues.
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py
from pathlib import Path
import subprocess

# Set up matplotlib for non-interactive backend
matplotlib.use('Agg')

def plot_density_snapshots_corrected(snapshots_dir, output_dir="corrected_density_frames"):
    """Plot density snapshots with corrected coordinate system and domain mapping."""
    
    # Find all snapshot files
    snapshot_files = sorted(glob.glob(os.path.join(snapshots_dir, "snapshots_s*.h5")))
    
    if not snapshot_files:
        print("No snapshot files found!")
        return []
    
    print(f"Found {len(snapshot_files)} snapshot files")
    
    # Create output directory for plots
    os.makedirs(output_dir, exist_ok=True)
    
    frame_paths = []
    
    for snapshot_file in snapshot_files:
        print(f"Processing {snapshot_file}")
        
        with h5py.File(snapshot_file, 'r') as f:
            # Get all time steps
            times = f['scales']['sim_time'][:]
            n_times = len(times)
            
            print(f"Found {n_times} time steps from t={times[0]:.3f} to t={times[-1]:.3f}")
            
            # Plot every 5th time step for smoother video
            step = max(1, n_times // 100)  # Show up to 100 frames
            
            for i in range(0, n_times, step):
                # Get data for this time step
                density = f['tasks']['density'][i, :, :]  # Shape: (ny, nx)
                
                # Get time
                time = times[i]
                
                # Create proper coordinate arrays for the domain [0,1] x [0,1]
                ny, nx = density.shape
                x = np.linspace(0, 1, nx, endpoint=False)  # [0,1) - periodic
                y = np.linspace(0, 1, ny, endpoint=False)  # [0,1) - periodic
                X, Y = np.meshgrid(x, y)
                
                # Create figure with better styling
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                
                # Use pcolormesh with actual coordinates for proper domain mapping
                im = ax.pcolormesh(X, Y, density, cmap='viridis', 
                                 vmin=0.4, vmax=2.6, shading='nearest')
                
                # Add title with time
                ax.set_title(f'2D Kelvin-Helmholtz Instability - Density\nTime = {time:.3f}', 
                           fontsize=16, fontweight='bold', pad=20)
                
                # Add axis labels
                ax.set_xlabel('x', fontsize=14)
                ax.set_ylabel('y', fontsize=14)
                
                # Set proper domain limits
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
                ax.set_aspect('equal')
                
                # Add colorbar with proper styling
                cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
                cbar.set_label('Density', fontsize=12, fontweight='bold')
                cbar.ax.tick_params(labelsize=10)
                
                # Add grid for better visualization
                ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
                
                # Add domain boundaries
                ax.axhline(y=0.25, color='white', linestyle='-', alpha=0.5, linewidth=2)
                ax.axhline(y=0.75, color='white', linestyle='-', alpha=0.5, linewidth=2)
                
                # Tight layout
                plt.tight_layout()
                
                # Save plot with high quality
                frame_filename = f"corrected_density_frame_{i:04d}_t{time:.3f}.png"
                frame_path = os.path.join(output_dir, frame_filename)
                plt.savefig(frame_path, dpi=200, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                plt.close()
                
                frame_paths.append(frame_path)
                print(f"Saved frame: {frame_filename}")

    return frame_paths

def create_corrected_density_video(frame_dir, output_video, fps=30, quality="high"):
    """Create a high-quality video from corrected density frames using ffmpeg."""
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg to create videos.")
        print("Frames have been generated in:", frame_dir)
        return False
    
    # Find all corrected density frames
    frame_pattern = os.path.join(frame_dir, "corrected_density_frame_*.png")
    frames = sorted(glob.glob(frame_pattern))
    
    if not frames:
        print("No corrected density frames found to create video.")
        return False
    
    print(f"Found {len(frames)} corrected density frames")
    
    # Try different encoding options based on what's available
    encoding_options = [
        # Option 1: H.264 with h264_videotoolbox (macOS) - preferred
        {
            "name": "H.264 (videotoolbox)",
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-pattern_type", "glob",
                "-i", os.path.join(frame_dir, "corrected_density_frame_*.png"),
                "-c:v", "h264_videotoolbox",
                "-b:v", "15M" if quality == "high" else "8M",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=1920:1920:force_original_aspect_ratio=decrease,pad=1920:1920:(ow-iw)/2:(oh-ih)/2",
                output_video
            ]
        },
        # Option 2: H.264 with libopenh264
        {
            "name": "H.264 (libopenh264)",
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-pattern_type", "glob",
                "-i", os.path.join(frame_dir, "corrected_density_frame_*.png"),
                "-c:v", "libopenh264",
                "-b:v", "15M" if quality == "high" else "8M",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=1920:1920:force_original_aspect_ratio=decrease,pad=1920:1920:(ow-iw)/2:(oh-ih)/2",
                output_video
            ]
        },
        # Option 3: MPEG-4 (more widely supported)
        {
            "name": "MPEG-4",
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-pattern_type", "glob",
                "-i", os.path.join(frame_dir, "corrected_density_frame_*.png"),
                "-c:v", "mpeg4",
                "-q:v", "2" if quality == "high" else "5",
                "-pix_fmt", "yuv420p",
                "-vf", "scale=1920:1920:force_original_aspect_ratio=decrease,pad=1920:1920:(ow-iw)/2:(oh-ih)/2",
                output_video
            ]
        }
    ]
    
    # Try each encoding option until one works
    for option in encoding_options:
        print(f"Trying {option['name']}...")
        try:
            result = subprocess.run(option['cmd'], capture_output=True, text=True, timeout=600)
            if result.returncode == 0:
                print(f"Video created successfully using {option['name']}: {output_video}")
                return True
            else:
                print(f"Failed with {option['name']}: {result.stderr}")
                continue
        except subprocess.TimeoutExpired:
            print(f"Timeout with {option['name']}")
            continue
        except Exception as e:
            print(f"Exception with {option['name']}: {e}")
            continue
    
    print("All encoding options failed. Frames are available for manual processing.")
    return False

def main():
    # Set up paths
    snapshots_dir = "snapshots"
    frame_dir = "corrected_density_frames"
    output_video = "khi2d_density_simulation_corrected.mp4"
    
    print("Creating corrected density-only video from KHI2D Dedalus simulation...")
    print(f"Snapshots directory: {snapshots_dir}")
    print(f"Frame directory: {frame_dir}")
    print(f"Output video: {output_video}")
    
    # Generate corrected density frames
    print("\nGenerating corrected density frames...")
    frame_paths = plot_density_snapshots_corrected(snapshots_dir, frame_dir)
    
    if not frame_paths:
        print("No frames generated!")
        return
    
    print(f"Generated {len(frame_paths)} corrected density frames")
    
    # Create video
    print("\nCreating video...")
    success = create_corrected_density_video(frame_dir, output_video, fps=30, quality="high")
    
    if success:
        print(f"\n✅ Corrected video created successfully: {output_video}")
        
        # Show file size
        if os.path.exists(output_video):
            size_mb = os.path.getsize(output_video) / (1024 * 1024)
            print(f"File size: {size_mb:.1f} MB")
    else:
        print("\n❌ Failed to create video")
        print(f"Frames are available in: {frame_dir}")

if __name__ == "__main__":
    main()
