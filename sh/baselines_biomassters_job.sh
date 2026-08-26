#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/baselines_biomassters/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One baseline run on BioMassters (T=12 temporal AGB regression).
# Expected env: BASELINE_ARGS (full python arg string), RUN_TAG

source sh/env.sh
export TQDM_DISABLE=1
# T=12 folds time into the batch dim, so memory is tight; bs=8 is the safe point.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/baselines_biomassters res/baselines checkpoints

echo "=== biomassters baseline | ${RUN_TAG} ==="
echo "    args: ${BASELINE_ARGS}"
python -u ${BASELINE_ARGS}
