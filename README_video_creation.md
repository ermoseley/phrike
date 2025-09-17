# KHI2D Comparison Video Creation

This script creates a high-quality 2-panel comparison video between mini-ramses and phrike outputs for the KHI2D simulation.

## Features

- **2-panel layout**: Mini-ramses on the left, Phrike on the right
- **Custom colormap**: Uses cmapkk9 for both simulations
- **MPI parallel processing**: Significantly faster frame generation
- **High-quality output**: 30 FPS MP4 video with H.264 encoding
- **Automatic cleanup**: Removes temporary frames after video creation

## Usage

### Basic Usage (Serial)
```bash
python create_comparison_video.py --start 1 --end 100
```

### Parallel Usage (Recommended)
```bash
mpirun -np 4 python create_comparison_video.py --start 1 --end 100 --parallel
```

### Full Range (All 251 outputs)
```bash
mpirun -np 8 python create_comparison_video.py --start 1 --end 251 --parallel --output khi2d_full_comparison.mp4
```

## Command Line Options

- `--start`: Starting output number (default: 1)
- `--end`: Ending output number (default: 100)
- `--parallel`: Use MPI parallel processing
- `--output`: Output video filename (default: khi2d_comparison.mp4)
- `--fps`: Frames per second (default: 30)
- `--quality`: Video quality - low/medium/high (default: high)
- `--colormap`: Colormap to use (default: cmapkk9)
- `--keep-frames`: Keep individual frames after video creation
- `--ramses-dir`: Directory containing mini-ramses outputs (default: /Users/moseley/hydra/khi)
- `--phrike-dir`: Directory containing phrike snapshots (default: /Users/moseley/hydra/outputs/khi2d)

## Requirements

- Python 3.6+
- matplotlib, numpy, scipy
- ffmpeg (for video creation)
- mpi4py (for parallel processing)
- miniramses (for mini-ramses data reading)

## Output

The script generates:
1. Individual frames for mini-ramses and phrike (temporarily)
2. Combined 2-panel frames
3. Final MP4 video with both simulations side by side

## Performance

- **Serial**: ~1-2 seconds per frame
- **MPI (4 processes)**: ~4x faster
- **MPI (8 processes)**: ~8x faster

For 251 frames:
- Serial: ~4-8 minutes
- MPI (4 processes): ~1-2 minutes
- MPI (8 processes): ~30-60 seconds

## Example Output

The generated video shows:
- **Left panel**: Mini-ramses density field with cmapkk9 colormap
- **Right panel**: Phrike density field with cmapkk9 colormap
- **Labels**: Each panel is clearly labeled
- **Time information**: Shows simulation time for each frame
- **Smooth animation**: 30 FPS for fluid motion
