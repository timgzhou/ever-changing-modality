#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/baselines_dfc2020/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One baseline run on DFC2020 / Copernicus-Bench.
# Expected env: BASELINE_ARGS (full python arg string), RUN_TAG, DECODER

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Must match the split the teachers were trained on.
export DFC2020_SPLIT=cobench

mkdir -p logs/baselines_dfc2020 res/baselines checkpoints

echo "=== baseline | ${RUN_TAG} | decoder=${DECODER} ==="
echo "    args: ${BASELINE_ARGS}"

python -u ${BASELINE_ARGS}
