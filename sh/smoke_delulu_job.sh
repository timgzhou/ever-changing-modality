#!/bin/bash
#SBATCH --time=1:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/smoke/%j.out
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101: failing GPU memory since 2026-09-15 ("uncorrectable ECC error").
# kn159: 50/50 jobs died there in 1-3s with exit 53 on 2026-09-19.
#SBATCH --exclude=kn101,kn159

# Short (2-epoch) smoke test of the paper losswts sweep command, one dataset.
# Expected env: SMOKE_ARGS (full arg string for sweep/sweep_delulu.py), RUN_TAG
#
# A real job script, not `sbatch --wrap`: --wrap executes under sh, where
# `source sh/env.sh` is a "source: not found" no-op, so the venv never activates
# and every run dies with ModuleNotFoundError: torch.

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/smoke res/delulu-sweep

echo "=== smoke | ${RUN_TAG} ==="
echo "    args: ${SMOKE_ARGS}"

python -u sweep/sweep_delulu.py ${SMOKE_ARGS}
