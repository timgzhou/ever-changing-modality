#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/train_delulu/chain_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Chain job: runs after the stage-0 (train_sft) jobs finish, rebuilds the teacher
# registry from their results, then submits stage 1.
#
# The registry is NOT a plain file dependency -- artifacts/sft_teachers.json is
# derived from the results CSVs by res/train_sft/sft_best.py -- so stage 1 cannot
# simply be `--dependency=afterok` on the stage-0 jobs. This job is that missing
# step: regenerate, then launch.
#
# Submit with:
#   sbatch --dependency=afterany:<comma-separated stage-0 job ids> \
#          sh/train_delulu_after_sft.sh
# (afterANY, not afterok: one failed/pre-empted modality should not block the
#  directions whose teachers did land -- train_delulu_all.sh skips missing ones.)

set -uo pipefail
source sh/env.sh

mkdir -p logs/train_delulu

echo "=== regenerating teacher registry from stage-0 results ==="
python res/train_sft/sft_best.py || { echo "[error] sft_best.py failed"; exit 1; }

echo
echo "=== submitting stage 1 ==="
bash sh/train_delulu_all.sh
