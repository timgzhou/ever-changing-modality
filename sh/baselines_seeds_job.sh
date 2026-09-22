#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/baselines_seeds/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101: failing GPU memory since 2026-09-15 ("uncorrectable ECC error").
# kn159: 50/50 jobs died there in 1-3s with exit 53 and no output file on
# 2026-09-19, while 10 concurrent jobs on other nodes ran fine. Slurm still
# lists both as healthy. Remove either if repaired.
#SBATCH --exclude=kn101,kn159

# One seed-replicate baseline run, any dataset/family.
# Expected env: BASELINE_ARGS (full python arg string), RUN_TAG
#
# Generic on purpose: sh/mke/mke_sweep_job.sh, sh/freematch/freematch_job.sh and
# sh/mixmatch/mixmatch_job.sh each hardcode their own positional args and build
# the python command themselves, so none of them can pass --seed through. This
# mirrors sh/baselines_dfc2020_job.sh, which already takes a raw arg string.

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/baselines_seeds res/baselines checkpoints

echo "=== baseline seed run | ${RUN_TAG} ==="
echo "    args: ${BASELINE_ARGS}"

python -u ${BASELINE_ARGS}
