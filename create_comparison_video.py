#!/usr/bin/env python3
"""
Create a 2-panel comparison video between mini-ramses and phrike outputs.
Left panel: mini-ramses (from khi/ directory)
Right panel: phrike (from outputs/khi2d/ directory)

Usage with MPI:
    mpirun -np 4 python create_comparison_video.py --start 1 --end 100
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path
import subprocess
import argparse
from datetime import datetime

# Try to import MPI, but don't fail if not available
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Running in serial mode.")

# Add the project root to the path so we can import phrike modules
sys.path.insert(0, '/Users/moseley/hydra')

# Import phrike modules
from phrike.io import load_checkpoint
from colormaps import cmaps, register

# Register the custom colormaps with matplotlib
register('cmapkk9')

# Set up matplotlib for non-interactive backend
matplotlib.use('Agg')

def get_ramses_outputs(ramses_dir):
    """Get list of mini-ramses output directories."""
    pattern = os.path.join(ramses_dir, "output_*")
    dirs = glob.glob(pattern)
    dirs = [d for d in dirs if os.path.isdir(d)]
    
    # Extract numbers and sort
    output_numbers = []
    for d in dirs:
        try:
            dirname = os.path.basename(d)
            if dirname.startswith("output_"):
                num_str = dirname[7:]  # Remove "output_" prefix
                num = int(num_str)
                output_numbers.append((num, d))
        except (ValueError, IndexError):
            continue
    
    output_numbers.sort(key=lambda x: x[0])
    return output_numbers

def get_phrike_snapshots(phrike_dir):
    """Get list of phrike snapshot files."""
    pattern = os.path.join(phrike_dir, "snapshot_t*.npz")
    files = glob.glob(pattern)
    
    # Extract time values and sort
    snapshots = []
    for f in files:
        try:
            basename = os.path.basename(f)
            if basename.startswith("snapshot_t") and basename.endswith(".npz"):
                time_str = basename[10:-4]  # Remove "snapshot_t" prefix and ".npz" suffix
                time = float(time_str)
                snapshots.append((time, f))
        except (ValueError, IndexError):
            continue
    
    snapshots.sort(key=lambda x: x[0])
    return snapshots

def generate_ramses_frame(output_num, output_dir, frame_dir, colormap):
    """Generate a single frame from mini-ramses output using amr2img.py."""
    frame_filename = f"ramses_frame_{output_num:05d}.png"
    frame_path = os.path.join(frame_dir, frame_filename)
    
    # Create a wrapper script that registers the colormap before calling amr2img
    wrapper_script = f"""
import sys
sys.path.insert(0, '/Users/moseley/hydra')
sys.path.insert(0, '/Users/moseley/hydra/ramses-utils')
from colormaps import register
register('{colormap}')

# Now import and run amr2img
import miniramses as ram
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import os
import re

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("nout")
parser.add_argument("--path", default="./")
parser.add_argument("--log", action="store_true")
parser.add_argument("--out")
parser.add_argument("--col", default="viridis")
parser.add_argument("--var", default="0")
parser.add_argument("--no-display", action="store_true")
args = parser.parse_args()

# Set up matplotlib
if args.no_display:
    import matplotlib
    matplotlib.use('Agg')

# Configure parameters
path = args.path
ivar = int(args.var)
col = args.col
log = args.log
nout = int(args.nout)

# Set defaults
if ivar == 0:
    ivar = 0
else:
    ivar = ivar - 1

if path is None:
    path = "./"
else:
    path = path + "/"

# Read time from info.txt
info_file = os.path.join(path, f"output_{{nout:05d}}", "info.txt")
time = 0.0
if os.path.exists(info_file):
    with open(info_file, 'r') as f:
        for line in f:
            if 'time' in line.lower():
                # Extract time value using regex
                match = re.search(r'time\s*=\s*([0-9.E+-]+)', line)
                if match:
                    time = float(match.group(1))
                break

# Read data
c = ram.rd_cell(nout, path=path, prefix="hydro")

# Create figure and axis
fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Determine bounds like miniramses.visu
xmin = np.min(c.x[0] - c.dx/2)
xmax = np.max(c.x[0] + c.dx/2)
ymin = np.min(c.x[1] - c.dx/2)
ymax = np.max(c.x[1] + c.dx/2)
ax.set_xlim([xmin, xmax])
ax.set_ylim([ymin, ymax])
ax.set_aspect("equal")

# Sort indices for overdraw order (use density index 0)
sort_vals = c.u[0] if c.u is not None else None
if sort_vals is not None:
    ind = np.argsort(sort_vals)
else:
    ind = np.arange(c.u[ivar].size)

# Marker size scaled by cell size
rescale = max(xmax - xmin, ymax - ymin)
sizes = (c.dx[ind] * 800.0 / rescale) ** 2

# Scatter cells as squares with fixed vmin/vmax and requested cmap
sc = ax.scatter(
    c.x[0][ind], c.x[1][ind], c=c.u[ivar][ind], s=sizes, marker='s',
    vmin=0.75, vmax=2.25, cmap=col, edgecolor=None, linewidths=0.0
)

# Hide axes for identical look
ax.set_axis_off()

# Label and time (time below label)
ax.text(0.5, 1.05, "mini-RAMSES", transform=ax.transAxes, ha='center', va='bottom', fontsize=14, fontweight='bold')
ax.text(0.5, 0.98, f"t = {{time:.3f}}", transform=ax.transAxes, ha='center', va='top', fontsize=12)

# Full-height colorbar
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = plt.colorbar(sc, cax=cax)
cbar.set_label("Density")

# Save output
if args.out:
    plt.savefig(args.out, dpi=150, bbox_inches='tight')

plt.close()
"""
    
    # Write wrapper script to temporary file
    wrapper_path = os.path.join(frame_dir, f"ramses_wrapper_{output_num}.py")
    with open(wrapper_path, 'w') as f:
        f.write(wrapper_script)
    
    # Build the command to run the wrapper
    cmd = [
        "python", wrapper_path, str(output_num), 
        "--no-display", 
        "--out", frame_path,
        "--path", os.path.dirname(output_dir),
        "--col", colormap,
        "--var", "1",  # Density variable
        "--log"  # Use log scale
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.getcwd())
        
        # Clean up wrapper script
        try:
            os.remove(wrapper_path)
        except OSError:
            pass
        
        if result.returncode == 0:
            print(f"Generated mini-ramses frame from output {output_num:05d}")
            return frame_path
        else:
            print(f"Error generating mini-ramses frame from output {output_num:05d}: {result.stderr}")
            return None
    except Exception as e:
        print(f"Exception generating mini-ramses frame from output {output_num:05d}: {e}")
        # Clean up wrapper script on exception
        try:
            os.remove(wrapper_path)
        except OSError:
            pass
        return None

def generate_phrike_frame(snapshot_path, frame_dir, colormap):
    """Generate a single frame from phrike snapshot."""
    # Load the snapshot
    data = load_checkpoint(snapshot_path)
    
    # Extract data
    rho = data['primitive_vars']['rho']
    x = data['x']
    y = data['y']
    t = data['t']
    
    # Convert to numpy if needed
    if hasattr(rho, 'cpu'):  # Torch tensor
        rho = rho.cpu().numpy()
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    if hasattr(y, 'cpu'):
        y = y.cpu().numpy()
    
    # Create the frame
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    
    # Use the custom colormap
    cmap = cmaps(colormap)
    
    # Create the image with fixed colorbar range and no interpolation
    im = ax.imshow(
        rho,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        cmap=cmap,
        vmin=0.75,
        vmax=2.25,
        interpolation='nearest'  # No interpolation between grid points
    )
    
    # No title for phrike plot
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Density")
    
    # Save the frame
    frame_filename = f"phrike_frame_{t:08.3f}.png"
    frame_path = os.path.join(frame_dir, frame_filename)
    fig.savefig(frame_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Generated phrike frame at t={t:.3f}")
    return frame_path, t

def generate_combined_frame_from_data(
    output_num: int,
    output_dir: str,
    phrike_snapshot_path: str,
    frame_dir: str,
    colormap: str,
) -> str:
    """Render a single combined 2-panel frame directly from data with identical layout.

    Left: mini-RAMSES (scatter of AMR cells), Right: PHRIKE (imshow grid)
    Both use cmapkk9, vmin=0.75, vmax=2.25, no interpolation, full-height colorbars.
    Labels and time are placed ABOVE each panel.
    """
    # Import here to avoid global dependency if user doesn't have ramses-utils in path
    sys.path.insert(0, '/Users/moseley/hydra/ramses-utils')
    import miniramses as ram  # type: ignore

    # Load mini-RAMSES cell data
    base_path = os.path.dirname(output_dir)
    c = ram.rd_cell(output_num, path=base_path, prefix="hydro")

    # Load mini-RAMSES time from info.txt
    info_file = os.path.join(base_path, f"output_{output_num:05d}", "info.txt")
    time_r = 0.0
    if os.path.exists(info_file):
        with open(info_file, 'r') as f:
            for line in f:
                if 'time' in line.lower():
                    try:
                        time_r = float(line.split('=')[1].strip())
                    except Exception:
                        pass
                    break

    # Load PHRIKE snapshot
    phrike_data = load_checkpoint(phrike_snapshot_path)
    rho = phrike_data['primitive_vars']['rho']
    x = phrike_data['x']
    y = phrike_data['y']
    time_p = float(phrike_data['t'])
    if hasattr(rho, 'cpu'):
        rho = rho.cpu().numpy()
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    if hasattr(y, 'cpu'):
        y = y.cpu().numpy()

    # Ensure axes divider available locally (avoid NameError)
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    # Figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=False)

    # Colormap
    cmap = cmaps(colormap)

    # mini-RAMSES scatter (match miniramses.visu style)
    xmin = np.min(c.x[0] - c.dx/2)
    xmax = np.max(c.x[0] + c.dx/2)
    ymin = np.min(c.x[1] - c.dx/2)
    ymax = np.max(c.x[1] + c.dx/2)
    ax1.set_xlim([xmin, xmax])
    ax1.set_ylim([ymin, ymax])
    ax1.set_aspect("equal")

    sort_vals = c.u[0] if c.u is not None else None
    if sort_vals is not None:
        ind = np.argsort(sort_vals)
    else:
        ind = np.arange(c.u[1].size)
    rescale = max(xmax - xmin, ymax - ymin)
    sizes = (c.dx[ind] * 800.0 / rescale) ** 2
    sc1 = ax1.scatter(
        c.x[0][ind], c.x[1][ind], c=c.u[0][ind], s=sizes, marker='s',
        vmin=0.75, vmax=2.25, cmap=cmap, edgecolor=None, linewidths=0.0
    )
    ax1.set_axis_off()

    # PHRIKE imshow, no interpolation
    im2 = ax2.imshow(
        rho,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        aspect="equal",
        cmap=cmap,
        vmin=0.75,
        vmax=2.25,
        interpolation='nearest',
    )
    ax2.set_axis_off()

    # Full-height colorbars for both
    div1 = make_axes_locatable(ax1)
    cax1 = div1.append_axes("right", size="5%", pad=0.05)
    cb1 = plt.colorbar(sc1, cax=cax1)
    cb1.set_label("Density")

    div2 = make_axes_locatable(ax2)
    cax2 = div2.append_axes("right", size="5%", pad=0.05)
    cb2 = plt.colorbar(im2, cax=cax2)
    cb2.set_label("Density")

    # Labels and times ABOVE each panel (outside axes, using figure coords)
    for ax, title, tval in (
        (ax1, "mini-RAMSES", time_r),
        (ax2, "PHRIKE", time_p),
    ):
        bb = ax.get_position()
        xmid = (bb.x0 + bb.x1) * 0.5
        ytop = bb.y1
        fig.text(xmid, ytop + 0.02, f"t = {tval:.3f}", ha='center', va='bottom', fontsize=12)
        fig.text(xmid, ytop + 0.06, title, ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Save combined frame
    Path(frame_dir).mkdir(exist_ok=True)
    frame_path = os.path.join(frame_dir, f"combined_frame_{output_num:05d}.png")
    fig.savefig(frame_path, dpi=150)
    plt.close(fig)
    return frame_path

def create_combined_frame(ramses_frame, phrike_frame, output_path, ramses_time=None, phrike_time=None):
    """Create a combined 2-panel frame."""
    if not os.path.exists(ramses_frame) or not os.path.exists(phrike_frame):
        print(f"Warning: Missing frame files. Ramses: {os.path.exists(ramses_frame)}, Phrike: {os.path.exists(phrike_frame)}")
        return False
    
    # Load the images
    ramses_img = plt.imread(ramses_frame)
    phrike_img = plt.imread(phrike_frame)
    
    # Create combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display images - no titles, no axis labels, identical formatting
    ax1.imshow(ramses_img)
    ax1.axis('off')
    
    ax2.imshow(phrike_img)
    ax2.axis('off')
    
    # No overall title - panels are identical except for data source
    
    # Save combined frame
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return True

def create_movie(frame_dir, output_movie, fps=30, quality="high"):
    """Create a movie from the generated frames using ffmpeg."""
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg to create movies.")
        print("Frames have been generated in:", frame_dir)
        return False
    
    # Find all combined frames
    frame_pattern = os.path.join(frame_dir, "combined_frame_*.png")
    frames = sorted(glob.glob(frame_pattern))
    
    if not frames:
        print("No combined frames found to create movie.")
        return False
    
    print(f"Found {len(frames)} combined frames")
    
    # Try different encoding options based on what's available
    encoding_options = [
        # Option 1: H.264 with h264_videotoolbox (macOS) - preferred
        {
            "name": "H.264 (videotoolbox)",
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frame_dir, "combined_frame_%05d.png"),
                "-c:v", "h264_videotoolbox",
                "-b:v", "10M" if quality == "high" else "5M",
                "-pix_fmt", "yuv420p",
                output_movie
            ]
        },
        # Option 2: H.264 with libopenh264
        {
            "name": "H.264 (libopenh264)",
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frame_dir, "combined_frame_%05d.png"),
                "-c:v", "libopenh264",
                "-b:v", "10M" if quality == "high" else "5M",
                "-pix_fmt", "yuv420p",
                output_movie
            ]
        },
        # Option 3: MPEG-4 (more widely supported)
        {
            "name": "MPEG-4",
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frame_dir, "combined_frame_%05d.png"),
                "-c:v", "mpeg4",
                "-q:v", "2" if quality == "high" else "5",
                "-pix_fmt", "yuv420p",
                output_movie
            ]
        }
    ]
    
    # Try each encoding option until one works
    for option in encoding_options:
        print(f"Trying {option['name']}...")
        try:
            result = subprocess.run(option['cmd'], capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"Movie created successfully using {option['name']}: {output_movie}")
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
    parser = argparse.ArgumentParser(description="Create 2-panel comparison video")
    parser.add_argument("--ramses-dir", default="/Users/moseley/hydra/khi", 
                       help="Directory containing mini-ramses outputs")
    parser.add_argument("--phrike-dir", default="/Users/moseley/hydra/outputs/khi2d", 
                       help="Directory containing phrike snapshots")
    parser.add_argument("--output", default="khi2d_comparison.mp4", 
                       help="Output movie filename")
    parser.add_argument("--frame-dir", default="comparison_frames", 
                       help="Directory to store temporary frames")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="high", 
                       help="Movie quality")
    parser.add_argument("--colormap", default="cmapkk9", help="Colormap to use")
    parser.add_argument("--start", type=int, default=1, help="Starting output number")
    parser.add_argument("--end", type=int, default=100, help="Ending output number")
    parser.add_argument("--keep-frames", action="store_true", help="Keep individual frames")
    parser.add_argument("--parallel", action="store_true", help="Use MPI parallel processing")
    
    args = parser.parse_args()
    
    # Initialize MPI if available and requested
    if args.parallel and MPI_AVAILABLE:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        is_root = (rank == 0)
    else:
        comm = None
        rank = 0
        size = 1
        is_root = True
        if args.parallel and not MPI_AVAILABLE:
            print("Error: --parallel requested but mpi4py not available.")
            print("Install mpi4py with: pip install mpi4py")
            sys.exit(1)
    
    # Create frame directory (all processes need to do this)
    frame_dir = Path(args.frame_dir)
    frame_dir.mkdir(exist_ok=True)
    
    if is_root:
        print(f"Creating comparison video:")
        print(f"  Mini-RAMSESt directory: {args.ramses_dir}")
        print(f"  Phrike directory: {args.phrike_dir}")
        print(f"  Output movie: {args.output}")
        print(f"  Frame directory: {args.frame_dir}")
        print(f"  Colormap: {args.colormap}")
        print(f"  FPS: {args.fps}")
        if args.parallel:
            print(f"  Using MPI with {size} processes")
    
    # Get available outputs (only on root process)
    if is_root:
        ramses_outputs = get_ramses_outputs(args.ramses_dir)
        phrike_snapshots = get_phrike_snapshots(args.phrike_dir)
        
        print(f"Found {len(ramses_outputs)} mini-ramses outputs")
        print(f"Found {len(phrike_snapshots)} phrike snapshots")
        
        if not ramses_outputs or not phrike_snapshots:
            print("Error: No outputs found in specified directories")
            return
        
        # Filter to requested range
        ramses_outputs = [(num, dir_path) for num, dir_path in ramses_outputs 
                         if args.start <= num <= args.end]
        
        print(f"Processing {len(ramses_outputs)} mini-ramses outputs in range {args.start}-{args.end}")
    else:
        ramses_outputs = None
        phrike_snapshots = None
    
    # Broadcast data to all processes
    if args.parallel and MPI_AVAILABLE:
        if is_root:
            output_nums = [num for num, _ in ramses_outputs]
            output_dirs = [dir_path for _, dir_path in ramses_outputs]
            phrike_times = [time for time, _ in phrike_snapshots]
            phrike_files = [file_path for _, file_path in phrike_snapshots]
        else:
            output_nums = None
            output_dirs = None
            phrike_times = None
            phrike_files = None
        
        output_nums = comm.bcast(output_nums, root=0)
        output_dirs = comm.bcast(output_dirs, root=0)
        phrike_times = comm.bcast(phrike_times, root=0)
        phrike_files = comm.bcast(phrike_files, root=0)
        
        # Reconstruct the lists
        ramses_outputs = list(zip(output_nums, output_dirs))
        phrike_snapshots = list(zip(phrike_times, phrike_files))
    
    # Distribute work among processes
    if args.parallel and MPI_AVAILABLE:
        # Distribute frame indices to processes
        my_indices = [i for i in range(len(ramses_outputs)) if i % size == rank]
        print(f"Process {rank}: processing {len(my_indices)} frames")
    else:
        my_indices = list(range(len(ramses_outputs)))
    
    # Generate frames
    combined_frames = []
    
    for i in my_indices:
        output_num, output_dir = ramses_outputs[i]
        phrike_time, phrike_snapshot = phrike_snapshots[i]
        
        if not is_root:
            print(f"Process {rank}: Processing output {output_num} (frame {i+1})")
        
        # Generate combined frame directly with identical layout
        combined_frame_path = generate_combined_frame_from_data(
            output_num=output_num,
            output_dir=output_dir,
            phrike_snapshot_path=phrike_snapshot,
            frame_dir=args.frame_dir,
            colormap=args.colormap,
        )
        if combined_frame_path:
            combined_frames.append(combined_frame_path)
    
    # Gather all generated frames
    if args.parallel and MPI_AVAILABLE:
        all_frames = comm.gather(combined_frames, root=0)
        if is_root:
            combined_frames = [frame for sublist in all_frames for frame in sublist]
            combined_frames.sort()  # Sort by frame number
    else:
        combined_frames.sort()
    
    # Create movie (only on root process)
    if is_root and combined_frames:
        print(f"Generated {len(combined_frames)} combined frames")
        
        success = create_movie(args.frame_dir, args.output, args.fps, args.quality)
        
        if success:
            print(f"Video created successfully: {args.output}")
            
            # Clean up frames if requested
            if not args.keep_frames:
                print("Cleaning up frame files...")
                for frame_file in combined_frames:
                    try:
                        os.remove(frame_file)
                    except OSError:
                        pass
                try:
                    os.rmdir(args.frame_dir)
                except OSError:
                    pass
                print("Frame cleanup complete")
            else:
                print(f"Frames kept in: {args.frame_dir}")
        else:
            print("Failed to create video")
    elif is_root:
        print("No frames generated")
    
    # Finalize MPI
    if args.parallel and MPI_AVAILABLE:
        MPI.Finalize()

if __name__ == "__main__":
    main()
