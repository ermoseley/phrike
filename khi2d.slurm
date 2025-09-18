#!/bin/bash
#SBATCH --job-name=khi2d_gpu
#SBATCH -A marlowe-m000115
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G 
#SBATCH --time=12:00:00
#SBATCH --output=khi2d.out
#SBATCH --error=khi2d.err

# Load modules
module load slurm nvhpc cudnn/cuda12/9.3.0.75

# Activate phrike environment
source /users/emoseley/phrike_env/bin/activate

# Set up scratch output directory with timestamp
SCRATCH_OUTDIR="$SCRATCH/outputs/khi2d"
echo "Output directory: $SCRATCH_OUTDIR"

# Force Python output to be unbuffered
export PYTHONUNBUFFERED=1

# Run KHI2D simulation with GPU using absolute paths
phrike khi2d \
    --backend torch \
    --device cuda \
    --precision single \
    --config "$SCRATCH/configs/khi2d.yaml" \
    --outdir "$SCRATCH_OUTDIR" \
    --no-video

echo "Simulation completed. Output saved to: $SCRATCH_OUTDIR"
