#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/baselines_dfc2020/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101 has failing GPU memory: every job scheduled there since 2026-09-15 died
# with "CUDA error: uncorrectable ECC error encountered", while jobs on every
# other node succeeded. Slurm still lists it as healthy. Remove if repaired.
#SBATCH --exclude=kn101

# One baseline run on DFC2020 / Copernicus-Bench.
# Expected env: BASELINE_ARGS (full python arg string), RUN_TAG, DECODER

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Must match the split the teachers were trained on.

mkdir -p logs/baselines_dfc2020 res/baselines checkpoints

echo "=== baseline | ${RUN_TAG} | decoder=${DECODER} ==="
echo "    args: ${BASELINE_ARGS}"

python -u ${BASELINE_ARGS}
