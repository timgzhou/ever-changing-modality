#!/bin/bash
#SBATCH --time=8:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/delulu_seeds/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101: failing GPU memory since 2026-09-15 ("uncorrectable ECC error").
# kn159: 50/50 jobs died there in 1-3s with exit 53 and no output file on
# 2026-09-19 while concurrent jobs on 13 other nodes ran fine.
#SBATCH --exclude=kn101,kn159

# One seeded Delulu run at a sweep-winning config.
# Expected env: DELULU_ARGS (full arg string for train_delulu.py), RUN_TAG

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/delulu_seeds res/delulu

echo "=== delulu seed run | ${RUN_TAG} ==="
echo "    args: ${DELULU_ARGS}"

python -u train_delulu.py ${DELULU_ARGS}
