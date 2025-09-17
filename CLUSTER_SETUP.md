# PHRIKE Cluster Setup Guide

This guide helps you run PHRIKE simulations on a cluster with proper GPU and ffmpeg support.

## Quick Start

1. **Check your cluster environment:**
   ```bash
   ./check_cluster_modules.sh
   ```

2. **Use the improved submission script:**
   ```bash
   sbatch sub
   ```

3. **Or use the cluster-optimized version:**
   ```bash
   sbatch sub_cluster_optimized
   ```

## Files Overview

### Submission Scripts

- **`sub`** - Basic improved submission script with GPU and ffmpeg support
- **`sub_cluster_optimized`** - Advanced script with better error handling and resource allocation
- **`check_cluster_modules.sh`** - Utility to check available modules on your cluster

### Configuration Files

- **`configs/khi2d.yaml`** - Default configuration
- **`configs/khi2d_cluster.yaml`** - Cluster-optimized configuration with better video codec settings

## Key Improvements

### GPU Support
- ✅ Proper CUDA environment setup
- ✅ GPU availability verification
- ✅ PyTorch CUDA compatibility check
- ✅ Optimized memory allocation settings

### FFmpeg Support
- ✅ Module loading with fallback options
- ✅ FFmpeg availability verification
- ✅ Cluster-compatible video codec settings
- ✅ Multiple encoding options for compatibility

### Error Handling
- ✅ Comprehensive error checking
- ✅ Detailed logging with timestamps
- ✅ Graceful fallbacks for missing modules
- ✅ Output validation and reporting

### Resource Optimization
- ✅ Increased memory allocation (32-64GB)
- ✅ Optimized CPU allocation (8-16 cores)
- ✅ Better thread configuration
- ✅ Timestamped output directories

## Configuration Details

### SLURM Parameters
```bash
#SBATCH --mem=64G              # Increased memory for video processing
#SBATCH --cpus-per-task=16     # More CPUs for parallel processing
#SBATCH --time=01:00:00        # Extended time for video generation
```

### Environment Variables
```bash
export CUDA_LAUNCH_BLOCKING=0      # Better GPU performance
export CUDA_CACHE_DISABLE=1        # Reduce memory usage
export OMP_NUM_THREADS=16          # Thread optimization
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:512  # Memory management
```

### Video Settings (Cluster Optimized)
```yaml
video:
  codec: libx264           # Better cluster compatibility than h264_videotoolbox
  preset: fast             # Faster encoding for cluster
  crf: 20                  # High quality
  pix_fmt: yuv420p         # Widely supported format
```

## Troubleshooting

### Common Issues

1. **FFmpeg not found:**
   - Check available modules: `module avail ffmpeg`
   - Try different module names in the script
   - Install ffmpeg in your environment if needed

2. **GPU not detected:**
   - Verify CUDA modules are loaded
   - Check `nvidia-smi` output
   - Ensure PyTorch CUDA is working

3. **Video generation fails:**
   - Check frames directory for generated images
   - Try manual ffmpeg command provided in logs
   - Verify disk space in output directory

4. **Memory issues:**
   - Increase `--mem` parameter in SLURM
   - Reduce video quality settings
   - Use smaller grid resolution for testing

### Debug Commands

```bash
# Check GPU status
nvidia-smi

# Check ffmpeg
ffmpeg -version

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Check available modules
module avail

# Check job status
squeue -u $USER

# Check job output
tail -f run.out
tail -f error.out
```

## Customization

### Adjusting Resources
Modify these SLURM parameters based on your cluster:

```bash
#SBATCH --mem=32G              # Adjust based on problem size
#SBATCH --cpus-per-task=8      # Adjust based on available cores
#SBATCH --time=00:30:00        # Adjust based on expected runtime
```

### Video Quality Settings
Modify `configs/khi2d_cluster.yaml`:

```yaml
video:
  fps: 30                    # Frames per second
  quality: high              # low, medium, high
  crf: 20                    # Lower = better quality (15-28)
  preset: fast               # ultrafast, fast, medium, slow
```

### Problem-Specific Settings
For different problems, copy and modify the configuration:

```bash
cp configs/khi2d_cluster.yaml configs/sod_cluster.yaml
# Edit sod_cluster.yaml for your problem
# Update submission script to use new config
```

## Performance Tips

1. **Use cluster-optimized config** for better video encoding
2. **Increase memory allocation** for large problems or high-quality videos
3. **Use appropriate CPU count** based on your problem size
4. **Monitor disk space** in output directory
5. **Check cluster policies** for maximum job time and resource limits

## Support

If you encounter issues:

1. Run `./check_cluster_modules.sh` to diagnose environment
2. Check the logs in `run.out` and `error.out`
3. Verify all required modules are loaded
4. Test with smaller problem sizes first
5. Check cluster documentation for specific module names and policies
