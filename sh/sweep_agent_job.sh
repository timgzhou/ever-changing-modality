#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/sweep_agent/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101 has failing GPU memory: every job scheduled there since 2026-09-15 died
# with "CUDA error: uncorrectable ECC error encountered", while jobs on every
# other node succeeded. Slurm still lists it as healthy. Remove if repaired.
#SBATCH --exclude=kn101

# Run one W&B sweep agent for a bounded number of trials.
#
# Expected env: SWEEP_ID (entity/project/id), COUNT (trials this agent runs)
set -euo pipefail
source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/sweep_agent res/delulu-sweep

: "${SWEEP_ID:?}"
echo "=== sweep agent | ${SWEEP_ID} | count=${COUNT:-4} ==="
wandb agent --count "${COUNT:-4}" "${SWEEP_ID}"
