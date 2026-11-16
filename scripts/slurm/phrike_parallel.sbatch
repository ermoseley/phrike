#!/bin/bash
# Template: one MPI rank per GPU.  Edit the #SBATCH lines, the module/conda
# lines and the run line for your cluster; nothing below is site-specific.
#
#SBATCH --job-name=phrike
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8        # = GPUs per node
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=phrike_%j.out

set -euo pipefail

# module load cuda openmpi          # an MPI built with CUDA support, if available
# conda activate phrike             # torch, triton, and mpi4py built against that MPI

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Device buffers in MPI (only after `srun -n 2 python -m scripts.check_gpu_aware_mpi` passes):
# export PHRIKE_GPU_AWARE_MPI=1
# Triton's kernel cache; a node-local or per-job path avoids a slow shared home:
# export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton_$USER

CONFIG=configs/parallel/imhd_driven_turb.yaml
PROBLEM=imhd_turb

# 1. The kernels and the MPI layer on this machine (about a minute).
srun python -m phrike.parallel.selftest

# 2. The run.  Stops with a checkpoint before the time limit; to continue,
#    add:  --restart outputs/.../checkpoints/chk_NNNNNN
srun python -m phrike.parallel "$PROBLEM" --config "$CONFIG" --max-wall-minutes 225
