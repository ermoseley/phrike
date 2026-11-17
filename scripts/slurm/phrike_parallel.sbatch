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
# Fewer allocator fragmentation failures on large grids.  It may break CUDA IPC in a
# CUDA-aware MPI (unverified): leave it off if you use PHRIKE_GPU_AWARE_MPI=1.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF-expandable_segments:True}
# Exchange the FFT transposes with NCCL (torch.distributed) over NVLink/InfiniBand instead of
# MPI.  Recommended on GPU clusters: host MPI all-to-all can be as slow as tens of MB/s per
# rank, and a CUDA-aware MPI may crash.  All ranks fall back to alltoallv if NCCL cannot start.
# Uncomment after `srun python -m scripts.bench_parallel --transpose nccl` beats the default;
# set MASTER_ADDR if the node names do not resolve from each other.
# export PHRIKE_MPI_TRANSPOSE=nccl
# Device buffers in MPI instead (only after `srun -n 2 python -m scripts.check_gpu_aware_mpi`
# passes; also unset PYTORCH_CUDA_ALLOC_CONF above if it misbehaves):
# export PHRIKE_GPU_AWARE_MPI=1
# If the bench reports host alltoallv under ~1 GB/s per rank, MPI is misconfigured; on S3DF
# (one node) these restored 5.5 GB/s where the UCX defaults ran at 45 MB/s:
# export OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,vader
# Triton's kernel cache; a node-local or per-job path avoids a slow shared home:
# export TRITON_CACHE_DIR=${SLURM_TMPDIR:-/tmp}/triton_$USER

CONFIG=configs/parallel/imhd_driven_turb.yaml
PROBLEM=imhd_turb

# 1. The kernels and the MPI layer on this machine (about a minute); it also compares the
#    nccl transport with alltoallv, bit for bit, when PHRIKE_MPI_TRANSPOSE=nccl.
srun python -m phrike.parallel.selftest

# 2. The run.  Stops with a checkpoint before the time limit; to continue,
#    add:  --restart outputs/.../checkpoints/chk_NNNNNN
srun python -m phrike.parallel "$PROBLEM" --config "$CONFIG" --max-wall-minutes 225
