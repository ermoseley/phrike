"""
PHRIKE2VID - Generate movies from phrike simulation snapshots

This script generates movies from phrike simulation data by processing .npz snapshot files.
It supports high-quality visualization with full resolution plotting and parallel processing.

USAGE:
  Basic usage (processes snapshots in specified directory):
    python phrike2vid.py <start_time> <end_time> [options]
    
  Example with custom settings:
    python phrike2vid.py 0.0 1.0 --var density --fps 60 --quality high --log

ARGUMENTS:
  start_time, end_time    Range of simulation times to process (e.g., 0.0 1.0)
  
  --snapshot-dir          Directory containing snapshot files (default: current directory)
  --var                   Variable to plot: "density", "x-velocity", "y-velocity", "pressure", "kinetic-energy"
                          (default: density)

MOVIE GENERATION OPTIONS:
  --fps                   Frames per second for output movie (default: 30)
  --quality               Movie quality: "low", "medium", "high" (default: medium)
  --output                Output movie filename (default: phrike_movie.mp4)
  --frame-dir             Directory to store temporary frames (default: frames/)
  --keep-frames           Keep individual frames after movie creation
  --ffmpeg-only           Skip frame generation, create movie from existing frames

PLOTTING OPTIONS:
  --log                   Use logarithmic scale for variable plotting
  --col                   Colormap selection (e.g., viridis, plasma, hot, coolwarm)
  --min, --max            Minimum/maximum values for colorbar scaling
  --dpi                   DPI for output frames (default: 300 for high quality)

4K RENDERING OPTIONS:
  --4k                    Enable 4K rendering mode (3840x2160) - must be explicitly enabled
  --no-decorations        Remove axes, labels, colorbar and whitespace for clean output
  --aspect-ratio          Force specific aspect ratio: "16:9", "4:3", "1:1", "auto" (default: auto)

QUALITY ENHANCEMENT OPTIONS:
  --interpolation         Interpolation method: "nearest", "bilinear", "bicubic", "spline16", "spline36", "hanning", "hamming", "hermite", "kaiser", "quadric", "catrom", "gaussian", "bessel", "mitchell", "sinc", "lanczos" (default: nearest)
  --antialias             Use smooth interpolation for antialiasing effect
  --high-quality          Enable high-quality rendering with optimized settings (higher DPI, better interpolation)

PARALLEL PROCESSING:
  --parallel              Enable MPI parallel processing for faster frame generation
                          Requires mpi4py: pip install mpi4py
                          Usage: mpirun -np <n> python phrike2vid.py <start> <end> --parallel

EXAMPLES:

1. Create movie from snapshots with times 0.0 to 1.0:
   python phrike2vid.py 0.0 1.0

2. Create movie with custom settings:
   python phrike2vid.py 0.0 1.0 --var density --fps 60 --quality high --log

3. Create movie from pressure with custom colormap:
   python phrike2vid.py 0.0 1.0 --var pressure --col coolwarm --min 0.5 --max 2.0

4. Run in parallel with MPI (4 processes):
   mpirun -np 4 python phrike2vid.py 0.0 1.0 --parallel

5. Create movie from existing frames only:
   python phrike2vid.py 0.0 1.0 --ffmpeg-only --fps 30 --quality high

6. Keep frames after movie creation:
   python phrike2vid.py 0.0 1.0 --keep-frames

7. Generate 4K movie with clean output (no axes, labels, colorbar):
   python phrike2vid.py 0.0 1.0 --4k --no-decorations --quality high

8. Generate clean 4K movie (will notify if 4K grid is detected but 4K mode not enabled):
   python phrike2vid.py 0.0 1.0 --no-decorations --quality high

9. Generate high-quality 4K movie with smooth interpolation:
   python phrike2vid.py 0.0 1.0 --4k --no-decorations --high-quality --interpolation bicubic

10. Generate smooth artistic frames with antialiasing:
    python phrike2vid.py 0.0 1.0 --4k --no-decorations --antialias --interpolation lanczos

OUTPUT:
  - Creates a frames/ directory with individual PNG frames
  - Generates a movie file (default: phrike_movie.mp4) in current directory
  - Automatically cleans up frame files unless --keep-frames is specified

REQUIREMENTS:
  - Python 3.6+
  - matplotlib, numpy
  - ffmpeg (for movie creation)
  - mpi4py (for parallel processing, optional)
"""

import os
import sys
import argparse
import subprocess
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

# Try to import MPI, but don't fail if not available
try:
    from mpi4py import MPI
    MPI_AVAILABLE = True
except ImportError:
    MPI_AVAILABLE = False
    print("Warning: mpi4py not available. Running in serial mode.")

def find_phrike_snapshots(snapshot_dir):
    """Find all phrike snapshot files matching the pattern snapshot_t*.npz."""
    if snapshot_dir is None:
        snapshot_dir = "."  # Default to current directory
    
    # Look for files matching snapshot_t*.npz pattern
    pattern = os.path.join(snapshot_dir, "snapshot_t*.npz")
    files = glob.glob(pattern)
    
    # Extract times and sort
    snapshot_times = []
    for f in files:
        try:
            # Extract time from filename (e.g., "snapshot_t0.050000.npz" -> 0.05)
            basename = os.path.basename(f)
            match = re.search(r'snapshot_t([0-9.]+)\.npz', basename)
            if match:
                time = float(match.group(1))
                snapshot_times.append((time, f))
        except (ValueError, AttributeError):
            continue
    
    # Sort by time
    snapshot_times.sort(key=lambda x: x[0])
    
    return snapshot_times

def load_phrike_snapshot(snapshot_path):
    """Load a phrike snapshot file and return the data."""
    data = np.load(snapshot_path, allow_pickle=True)
    
    # Extract basic data
    result = {
        "t": float(data["t"]),
        "U": data["U"],
        "meta": data["meta"].item() if "meta" in data else {},
    }
    
    # Extract coordinates
    if "x" in data:
        result["x"] = data["x"]
    if "y" in data:
        result["y"] = data["y"]
    if "z" in data:
        result["z"] = data["z"]
    
    # Extract primitive variables
    primitive_vars = {}
    for var in ["rho", "u", "ux", "uy", "uz", "p"]:
        if var in data:
            primitive_vars[var] = data[var]
    
    result["primitive_vars"] = primitive_vars
    
    return result

def get_variable_data(data, variable):
    """Extract the requested variable data from phrike snapshot."""
    if variable == "density":
        if "rho" in data["primitive_vars"]:
            return data["primitive_vars"]["rho"]
        else:
            raise ValueError("Density (rho) not found in snapshot")
    
    elif variable == "x-velocity":
        if "ux" in data["primitive_vars"]:
            return data["primitive_vars"]["ux"]
        elif "u" in data["primitive_vars"]:
            return data["primitive_vars"]["u"]
        else:
            raise ValueError("X-velocity not found in snapshot")
    
    elif variable == "y-velocity":
        if "uy" in data["primitive_vars"]:
            return data["primitive_vars"]["uy"]
        else:
            raise ValueError("Y-velocity not found in snapshot")
    
    elif variable == "pressure":
        if "p" in data["primitive_vars"]:
            return data["primitive_vars"]["p"]
        else:
            raise ValueError("Pressure not found in snapshot")
    
    elif variable == "kinetic-energy":
        # Calculate kinetic energy: 0.5 * rho * (ux^2 + uy^2)
        rho = data["primitive_vars"]["rho"]
        ux = data["primitive_vars"].get("ux", np.zeros_like(rho))
        uy = data["primitive_vars"].get("uy", np.zeros_like(rho))
        return 0.5 * rho * (ux**2 + uy**2)
    
    else:
        raise ValueError(f"Unknown variable: {variable}")

def get_variable_label(variable):
    """Get the label for the variable."""
    labels = {
        "density": "Density",
        "x-velocity": "X-Velocity", 
        "y-velocity": "Y-Velocity",
        "pressure": "Pressure",
        "kinetic-energy": "Kinetic Energy"
    }
    return labels.get(variable, variable)

def detect_4k_grid(var_data):
    """Detect if the grid size corresponds to 4K resolution."""
    if len(var_data.shape) == 2:
        height, width = var_data.shape
        # Check for 4K resolution (3840x2160) or close approximations
        is_4k = (width == 3840 and height == 2160)
        # Also check for common variations like 4096x2160 (DCI 4K)
        is_dci_4k = (width == 4096 and height == 2160)
        # Check if aspect ratio is approximately 16:9
        aspect_ratio = width / height
        is_16_9 = abs(aspect_ratio - 16/9) < 0.01
        
        return is_4k or is_dci_4k, is_16_9, (width, height)
    return False, False, (0, 0)

def get_4k_figure_size(grid_width, grid_height, target_dpi=1):
    """Calculate figure size for 4K rendering with 1 pixel per grid point."""
    # For 4K rendering with 1 pixel per grid point, we want the figure size
    # to be exactly the grid dimensions in inches at the specified DPI
    width_inches = grid_width / target_dpi
    height_inches = grid_height / target_dpi
    return (width_inches, height_inches)

def generate_phrike_frame(snapshot_path, args, frame_dir, frame_index):
    """Generate a single frame from phrike snapshot."""
    
    # Ensure frame directory exists
    frame_dir = Path(frame_dir)
    frame_dir.mkdir(exist_ok=True)
    
    try:
        # Load the snapshot
        data = load_phrike_snapshot(snapshot_path)
        
        # Extract coordinates
        x = data["x"]
        y = data["y"]
        
        # Convert to numpy if needed (handle torch tensors)
        if hasattr(x, 'cpu'):
            x = x.cpu().numpy()
        if hasattr(y, 'cpu'):
            y = y.cpu().numpy()
        
        # Get variable data
        var_data = get_variable_data(data, args.var)
        
        # Convert to numpy if needed
        if hasattr(var_data, 'cpu'):
            var_data = var_data.cpu().numpy()
        
        # Apply logarithmic scaling if requested
        if args.log:
            # Avoid log(0) by adding small epsilon
            var_data = np.log10(np.maximum(var_data, 1e-10))
        
        # Detect if this is a 4K grid and check if 4K mode should be used
        is_4k_grid, is_16_9, (grid_width, grid_height) = detect_4k_grid(var_data)
        # Only use 4K mode if explicitly requested with --4k flag
        # Auto-detection is disabled for backwards compatibility
        use_4k_mode = getattr(args, '4k', False)  # argparse keeps the original name
        use_clean_output = args.__dict__.get('no_decorations', False)
        
        # Print info about detected grid and 4K mode status
        if is_4k_grid and use_4k_mode:
            print(f"4K mode enabled: {grid_width}x{grid_height} grid -> 4K rendering")
        elif is_4k_grid and not use_4k_mode:
            print(f"4K grid detected ({grid_width}x{grid_height}) but 4K mode not enabled. Use --4k to enable 4K rendering.")
        
        # Set up figure parameters based on mode and quality settings
        high_quality = args.__dict__.get('high_quality', False)
        
        if use_4k_mode:
            # For 4K mode, always use 1 pixel per grid point for true 1:1 mapping
            # High quality comes from interpolation, not DPI scaling
            fig_width, fig_height = get_4k_figure_size(grid_width, grid_height, target_dpi=1)
            dpi = 1  # 1 pixel per grid point - this is the correct approach
            quality_note = " (high-quality interpolation)" if high_quality else ""
            print(f"4K mode: {grid_width}x{grid_height} grid -> {fig_width:.1f}x{fig_height:.1f} inches at {dpi} DPI{quality_note}")
        else:
            # Standard mode
            if high_quality:
                fig_width, fig_height = (12, 12)  # Larger figure for better quality
                dpi = max(args.dpi, 600)  # Higher DPI for better quality
                print(f"High-quality mode: {fig_width}x{fig_height} inches at {dpi} DPI")
            else:
                fig_width, fig_height = (10, 10)
                dpi = args.dpi
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), dpi=dpi)
        
        # Set up colormap
        colormap = args.col if args.col else 'viridis'
        
        # Determine interpolation method based on quality settings
        interpolation_method = args.__dict__.get('interpolation', 'nearest')
        if high_quality and interpolation_method == 'nearest':
            # For high-quality mode, use smoother interpolation by default
            interpolation_method = 'bilinear'
        elif antialias and interpolation_method == 'nearest':
            # For antialiasing, use smooth interpolation
            interpolation_method = 'bicubic'
        
        
        # Create the image with quality settings
        imshow_kwargs = {
            'origin': "lower",
            'extent': [x.min(), x.max(), y.min(), y.max()],
            'aspect': "equal",
            'cmap': colormap,
            'vmin': args.min,
            'vmax': args.max,
            'interpolation': interpolation_method
        }
        
        # Note: antialiasing is handled through interpolation methods
        # The 'antialiased' parameter is not available in imshow
        
        im = ax.imshow(var_data, **imshow_kwargs)
        
        # Configure plot based on mode
        if use_clean_output or use_4k_mode:
            # Clean mode: no decorations, no whitespace
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.axis('off')  # Turn off all axes
            
            # Remove all spines (borders)
            for spine in ax.spines.values():
                spine.set_visible(False)
        else:
            # Standard mode: include labels and title
            ax.set_xlabel("x", fontsize=12)
            ax.set_ylabel("y", fontsize=12)
            ax.set_title(f"{get_variable_label(args.var)} at t = {data['t']:.4f}", fontsize=14)
        
        # Add colorbar only if not in clean mode
        if not use_clean_output and not use_4k_mode:
            cbar = plt.colorbar(im, ax=ax)
            label = get_variable_label(args.var)
            if args.log:
                label = f"log₁₀({label})"
            cbar.set_label(label, fontsize=12)
        
        # Set output filename for this frame
        frame_filename = f"frame_{frame_index:05d}.png"
        frame_path = frame_dir / frame_filename
        
        # Save the frame with appropriate settings
        if use_clean_output or use_4k_mode:
            # Clean mode: no padding, tight bounding box
            fig.savefig(frame_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
        else:
            # Standard mode: small padding
            fig.savefig(frame_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        
        plt.close(fig)
        
        mode_info = "4K" if use_4k_mode else "standard"
        clean_info = " (clean)" if use_clean_output else ""
        print(f"Generated {mode_info} frame {frame_index:05d}{clean_info} from snapshot at t={data['t']:.6f}")
        return str(frame_path)
        
    except Exception as e:
        print(f"Error generating frame from {os.path.basename(snapshot_path)}: {e}")
        return None

def detect_frame_pattern(frame_dir):
    """Detect the frame numbering pattern from existing frames."""
    frame_pattern = str(Path(frame_dir) / "frame_*.png")
    frames = sorted(glob.glob(frame_pattern))
    
    if not frames:
        return None, None
    
    # Extract the numbering pattern from the first few frames
    sample_frames = frames[:min(5, len(frames))]
    patterns = []
    
    for frame in sample_frames:
        basename = os.path.basename(frame)
        if basename.startswith("frame_"):
            # Extract the number part
            number_part = basename[6:-4]  # Remove "frame_" and ".png"
            patterns.append(len(number_part))
    
    if not patterns:
        return None, None
    
    # Use the most common pattern length
    pattern_length = max(set(patterns), key=patterns.count)
    
    # Determine the format string - support any number of leading zeros
    if pattern_length == 1:
        format_str = "frame_%d.png"
    else:
        format_str = f"frame_%0{pattern_length}d.png"
    
    return format_str, len(frames)

def create_movie(frame_dir, output_movie, fps=30, quality="high", is_4k=False):
    """Create a movie from the generated frames using ffmpeg."""
    
    # Check if ffmpeg is available
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: ffmpeg not found. Please install ffmpeg to create movies.")
        print("Frames have been generated in:", frame_dir)
        return False
    
    # Detect frame pattern automatically
    format_str, num_frames = detect_frame_pattern(frame_dir)
    if format_str is None:
        print("No frames found to create movie.")
        return False
    
    print(f"Detected frame pattern: {format_str}")
    print(f"Number of frames: {num_frames}")
    if is_4k:
        print("4K video encoding enabled")
    
    # Try different encoding options based on what's available
    encoding_options = [
        # Option 1: H.264 with libx264 (4K optimized)
        {
            "name": "H.264 (libx264)" + (" 4K" if is_4k else ""),
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(Path(frame_dir) / format_str),
                "-c:v", "libx264",
                "-crf", "18" if quality == "high" else "23" if quality == "medium" else "28",
                "-pix_fmt", "yuv420p",
                "-preset", "slow" if is_4k else "medium",  # Slower preset for 4K
                "-profile:v", "high" if is_4k else "main",
                "-level", "5.1" if is_4k else "4.1",  # Higher level for 4K
                output_movie
            ]
        },
        # Option 2: H.264 with h264_videotoolbox (macOS, 4K optimized)
        {
            "name": "H.264 (videotoolbox)" + (" 4K" if is_4k else ""),
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(Path(frame_dir) / format_str),
                "-c:v", "h264_videotoolbox",
                "-b:v", "50M" if is_4k and quality == "high" else "25M" if is_4k and quality == "medium" else "15M" if is_4k else "20M" if quality == "high" else "10M" if quality == "medium" else "5M",
                "-pix_fmt", "yuv420p",
                "-profile:v", "high" if is_4k else "main",
                "-level", "5.1" if is_4k else "4.1",
                output_movie
            ]
        },
        # Option 3: H.265/HEVC (better for 4K)
        {
            "name": "H.265/HEVC" + (" 4K" if is_4k else ""),
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(Path(frame_dir) / format_str),
                "-c:v", "libx265",
                "-crf", "20" if quality == "high" else "25" if quality == "medium" else "30",
                "-pix_fmt", "yuv420p",
                "-preset", "slow" if is_4k else "medium",
                output_movie
            ]
        },
        # Option 4: MPEG-4 (more widely supported, fallback)
        {
            "name": "MPEG-4" + (" 4K" if is_4k else ""),
            "cmd": [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", str(Path(frame_dir) / format_str),
                "-c:v", "mpeg4",
                "-q:v", "1" if quality == "high" else "3" if quality == "medium" else "5",
                "-pix_fmt", "yuv420p",
                output_movie
            ]
        }
    ]
    
    print(f"Creating movie: {output_movie}")
    print(f"Using {num_frames} frames at {fps} fps")
    
    # Try each encoding option until one works
    for option in encoding_options:
        print(f"Trying {option['name']}...")
        try:
            result = subprocess.run(option['cmd'], capture_output=True, text=True, timeout=120)
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
    print("You can try creating the movie manually with:")
    print(f"ffmpeg -framerate {fps} -i {frame_dir}/frame_%05d.png -c:v mpeg4 -q:v 2 {output_movie}")
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Generate movies from phrike simulation snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate movie from snapshots with times 0.0 to 1.0
  python phrike2vid.py 0.0 1.0
  
  # Generate movie with custom settings
  python phrike2vid.py 0.0 1.0 --var density --fps 60 --quality high --log
  
  # Generate movie from pressure with custom colormap
  python phrike2vid.py 0.0 1.0 --var pressure --col coolwarm --min 0.5 --max 2.0
  
  # Run in parallel with MPI (4 processes)
  mpirun -np 4 python phrike2vid.py 0.0 1.0 --parallel
  
  # Create movie from existing frames (skip generation)
  python phrike2vid.py 0.0 1.0 --ffmpeg-only --fps 30 --quality high
  
  # Keep frames after movie creation
  python phrike2vid.py 0.0 1.0 --keep-frames
        """
    )
    
    # Range arguments
    parser.add_argument("start_time", type=float, help="starting simulation time")
    parser.add_argument("end_time", type=float, help="ending simulation time")
    
    # Snapshot directory
    parser.add_argument("--snapshot-dir", default='.', 
                       help="directory containing snapshot files (default: current directory)")
    
    # Variable selection
    parser.add_argument("--var", default="density", 
                       choices=["density", "x-velocity", "y-velocity", "pressure", "kinetic-energy"],
                       help="variable to plot (default: density)")
    
    # Movie-specific arguments
    parser.add_argument("--fps", type=int, default=30, help="frames per second (default: 30)")
    parser.add_argument("--quality", choices=["low", "medium", "high"], default="medium", 
                       help="movie quality (default: medium)")
    parser.add_argument("--output", help="output movie filename (default: phrike_movie.mp4)")
    parser.add_argument("--frame-dir", help="directory to store frames (default: frames/)")
    parser.add_argument("--parallel", action="store_true", help="use MPI parallel processing")
    parser.add_argument("--keep-frames", action="store_true", help="keep individual frames after movie creation")
    parser.add_argument("--ffmpeg-only", action="store_true", help="skip frame generation, create movie from existing frames")
    
    # Plotting arguments
    parser.add_argument("--log", help="plot log variable", action="store_true")
    parser.add_argument("--col", help="choose the color map")
    parser.add_argument("--min", type=float, help="specify a minimum variable value for colorbar")
    parser.add_argument("--max", type=float, help="specify a maximum variable value for colorbar")
    parser.add_argument("--dpi", type=int, default=300, help="DPI for output frames (default: 300)")
    
    # 4K rendering options
    parser.add_argument("--4k", action="store_true", help="enable 4K rendering mode (3840x2160)")
    parser.add_argument("--no-decorations", action="store_true", help="remove axes, labels, colorbar and whitespace for clean output")
    parser.add_argument("--aspect-ratio", choices=["16:9", "4:3", "1:1", "auto"], default="auto", help="force specific aspect ratio (default: auto)")
    
    # Quality enhancement options
    parser.add_argument("--interpolation", choices=["nearest", "bilinear", "bicubic", "spline16", "spline36", "hanning", "hamming", "hermite", "kaiser", "quadric", "catrom", "gaussian", "bessel", "mitchell", "sinc", "lanczos"], default="nearest", help="interpolation method for rendering (default: nearest)")
    parser.add_argument("--antialias", action="store_true", help="enable antialiasing for smoother rendering")
    parser.add_argument("--high-quality", action="store_true", help="enable high-quality rendering with optimized settings")
    
    args = parser.parse_args()
    
    # Check MPI availability
    if args.parallel and not MPI_AVAILABLE:
        print("Error: --parallel requested but mpi4py not available.")
        print("Install mpi4py with: pip install mpi4py")
        sys.exit(1)
    
    # Initialize MPI if using parallel mode
    if args.parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        is_root = (rank == 0)
        # Add rank and parallel info to args for use in functions
        args.rank = rank
        args.parallel = True
    else:
        comm = None
        rank = 0
        size = 1
        is_root = True
        # Add rank and parallel info to args for use in functions
        args.rank = rank
        args.parallel = False
    
    # Set default values
    if args.output is None:
        args.output = "phrike_movie.mp4"
    if args.frame_dir is None:
        args.frame_dir = "frames"
    
    # Create frame directory (all processes need to do this)
    frame_dir = Path(args.frame_dir).resolve()  # Use absolute path
    frame_dir.mkdir(exist_ok=True)
    
    if is_root:
        print(f"Frame directory: {frame_dir}")
        print(f"Output movie: {args.output}")
        print(f"Variable: {args.var}")
        print(f"Snapshot directory: {args.snapshot_dir}")
        print(f"Time range: {args.start_time} to {args.end_time}")
    
    # If ffmpeg-only mode, skip frame generation and just create movie
    if args.ffmpeg_only:
        if is_root:
            print("ffmpeg-only mode: skipping frame generation")
            # Check if frames exist
            frame_pattern = str(frame_dir / "frame_*.png")
            existing_frames = sorted(glob.glob(frame_pattern))
            if not existing_frames:
                print(f"No frames found in {frame_dir}")
                print("Use pattern: frame_00001.png, frame_00002.png, etc.")
                sys.exit(1)
            
            print(f"Found {len(existing_frames)} existing frames")
            # Detect if existing frames are 4K by checking the first frame
            is_4k_frames = False
            if existing_frames:
                try:
                    from PIL import Image
                    with Image.open(existing_frames[0]) as img:
                        width, height = img.size
                        is_4k_frames = (width >= 3840 and height >= 2160)
                        if is_4k_frames:
                            print(f"Detected 4K frames: {width}x{height}")
                except ImportError:
                    print("PIL not available for frame detection, assuming standard resolution")
                except Exception as e:
                    print(f"Could not detect frame resolution: {e}")
            
            # Create movie from existing frames
            success = create_movie(frame_dir, args.output, args.fps, args.quality, is_4k_frames)
            if success:
                print(f"Movie created successfully: {args.output}")
            else:
                print("Failed to create movie")
            sys.exit(0)
    
    # Find snapshot files
    if is_root:
        snapshots = find_phrike_snapshots(args.snapshot_dir)
        # Filter to requested time range
        snapshots = [(t, path) for t, path in snapshots 
                    if args.start_time <= t <= args.end_time]
        
        if not snapshots:
            print(f"No snapshot files found in time range {args.start_time}-{args.end_time}")
            print(f"Searched in: {args.snapshot_dir}")
            sys.exit(1)
        
        print(f"Found {len(snapshots)} snapshot files")
        for t, path in snapshots:
            print(f"  t={t:8.6f}: {os.path.basename(path)}")
        
        # Create a global frame index mapping (time -> sequential_index)
        # This ensures frames are numbered 1, 2, 3, ... regardless of gaps
        frame_index_map = {}
        for i, (t, _) in enumerate(snapshots, 1):
            frame_index_map[t] = i
        
        print(f"Frame index mapping:")
        for t, idx in frame_index_map.items():
            print(f"  t={t:8.6f} -> Frame {idx:05d}")
    
    # Broadcast snapshots to all processes
    if args.parallel:
        if is_root:
            snapshot_times = [t for t, _ in snapshots]
            snapshot_paths = [path for _, path in snapshots]
            frame_indices = [frame_index_map[t] for t in snapshot_times]
        else:
            snapshot_times = None
            snapshot_paths = None
            frame_indices = None
        
        snapshot_times = comm.bcast(snapshot_times, root=0)
        snapshot_paths = comm.bcast(snapshot_paths, root=0)
        frame_indices = comm.bcast(frame_indices, root=0)
        
        # Create frame index mapping on all processes
        frame_index_map = dict(zip(snapshot_times, frame_indices))
        snapshots = list(zip(snapshot_times, snapshot_paths))
    else:
        snapshot_times = [t for t, _ in snapshots]
        snapshot_paths = [path for _, path in snapshots]
        frame_indices = [frame_index_map[t] for t in snapshot_times]
    
    # Distribute work among processes using frame indices
    if args.parallel:
        # Distribute frame indices (not times) to avoid conflicts
        my_frame_indices = [frame_indices[i] for i in range(len(frame_indices)) if i % size == rank]
        my_snapshot_paths = [snapshot_paths[i] for i in range(len(snapshot_paths)) if i % size == rank]
        print(f"Process {rank}: processing {len(my_snapshot_paths)} snapshots")
    else:
        my_frame_indices = frame_indices
        my_snapshot_paths = snapshot_paths
    
    # Generate frames
    generated_frames = []
    
    for snapshot_path, frame_index in zip(my_snapshot_paths, my_frame_indices):
        frame_path = generate_phrike_frame(snapshot_path, args, frame_dir, frame_index)
        if frame_path:
            generated_frames.append(frame_path)
    
    # Gather all generated frames
    if args.parallel:
        all_frames = comm.gather(generated_frames, root=0)
        if is_root:
            generated_frames = [frame for sublist in all_frames for frame in sublist]
            generated_frames.sort()  # Sort by frame number
    else:
        generated_frames.sort()
    
    # Create movie (only on root process)
    if is_root and generated_frames:
        print(f"Generated {len(generated_frames)} frames")
        
        # Check if 4K mode was explicitly requested
        is_4k_generated = getattr(args, '4k', False)  # argparse keeps the original name
        
        # Create movie
        success = create_movie(frame_dir, args.output, args.fps, args.quality, is_4k_generated)
        
        # Clean up frames if requested
        if success and not args.keep_frames:
            print("Cleaning up frame files...")
            for frame_file in generated_frames:
                try:
                    os.remove(frame_file)
                except OSError:
                    pass
            try:
                os.rmdir(frame_dir)
            except OSError:
                pass
            print("Frame cleanup complete")
        elif args.keep_frames:
            print(f"Frames kept in: {frame_dir}")
    
    if args.parallel:
        MPI.Finalize()

if __name__ == "__main__":
    main()
