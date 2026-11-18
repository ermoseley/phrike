#!/bin/bash
#SBATCH --job-name=dynamo
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=3
#SBATCH --mem=350G
#SBATCH --time=16:00:00
#SBATCH --output=dynamo-%j.log

# From the checkout, copy configs/parallel/dynamo_2048_fp64.yaml to dynamo.yaml.
# Pass additional PHRIKE options after the job filename, e.g. --restart DIR.
set -eo pipefail
module load anaconda3/2024.10 openmpi/gcc/5.0.10
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMPI_MCA_coll=^hcoll,ucc OMPI_MCA_pml=ob1 OMPI_MCA_btl=self,sm,tcp
export PHRIKE_TRITON_CC=/usr/bin/gcc PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

srun --mpi=pmix --gpu-bind=none --kill-on-bad-exit=1 \
    .venv/bin/python -m phrike.parallel imhd_turb --backend cuda \
    --config dynamo.yaml --max-wall-minutes 930 "$@"
