#!/bin/bash
# Script to check available modules on your cluster

echo "=== Cluster Module Check ==="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo ""

echo "=== Available modules ==="
if command -v module &> /dev/null; then
    echo "Available modules:"
    module avail 2>&1 | grep -E "(ffmpeg|cuda|gcc|python)" | head -20
    echo ""
    
    echo "=== CUDA modules ==="
    module avail 2>&1 | grep -i cuda
    echo ""
    
    echo "=== FFmpeg modules ==="
    module avail 2>&1 | grep -i ffmpeg
    echo ""
    
    echo "=== Python modules ==="
    module avail 2>&1 | grep -i python
    echo ""
else
    echo "Module system not available"
fi

echo "=== System information ==="
echo "OS: $(uname -a)"
echo ""

echo "=== GPU information ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "nvidia-smi not found"
fi
echo ""

echo "=== FFmpeg check ==="
if command -v ffmpeg &> /dev/null; then
    echo "FFmpeg found:"
    ffmpeg -version | head -n 1
else
    echo "FFmpeg not found in PATH"
fi
echo ""

echo "=== Python environment ==="
if command -v python &> /dev/null; then
    echo "Python version: $(python --version)"
    echo "Python path: $(which python)"
    
    echo "Checking PyTorch CUDA:"
    python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'Device count: {torch.cuda.device_count()}')" 2>/dev/null || echo "PyTorch not available or error"
else
    echo "Python not found"
fi
echo ""

echo "=== Environment variables ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SCRATCH: $SCRATCH"
echo "HOME: $HOME"
echo ""

echo "=== SLURM information ==="
if command -v squeue &> /dev/null; then
    echo "Current jobs:"
    squeue -u $USER 2>/dev/null || echo "No jobs or squeue not available"
else
    echo "SLURM not available"
fi
