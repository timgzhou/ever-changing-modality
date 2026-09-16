#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/projector_probe/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101 has failing GPU memory: every job scheduled there on 2026-09-15 died with
# "CUDA error: uncorrectable ECC error encountered" (9 of 9), while every job on
# every other node succeeded. Slurm still lists it as healthy, so exclude it
# explicitly. Override with EXCLUDE_NODES= if it is ever repaired.
#SBATCH --exclude=kn101

# One projector-probe arm: frozen encoder, train ONLY the cross-modal projector.
# See analysis/train_projector_probe.py for what this measures and why.
#
# Required: CKPT, SRC, TGT, DEPTH, LOSS
# Optional: COS_WEIGHT EPOCHS LR BATCH_SIZE EVAL_BATCHES SEED RESULTS_CSV TAG
#           MASK_RATIO (source patches hidden from the projector, as in real
#           training -- tuned dfc2020 uses 0.64) and SCORE_MASKED_ONLY=1
#
# 3h walltime: the encoder runs under no_grad and only a ~14-28M param projector
# gets gradients. At 8 epochs a job took ~1.2 min; 32 epochs puts it near 5.
set -euo pipefail
source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/projector_probe res/delulu

: "${CKPT:?}"; : "${SRC:?}"; : "${TGT:?}"; : "${DEPTH:?}"; : "${LOSS:?}"
[ -f "${CKPT}" ] || { echo "[error] checkpoint not found: ${CKPT}"; exit 1; }

python -u analysis/train_projector_probe.py \
    --checkpoint "${CKPT}" \
    --dataset "${DATASET:-dfc2020}" \
    --src_mod "${SRC}" \
    --tgt_mod "${TGT}" \
    --depth "${DEPTH}" \
    --loss "${LOSS}" \
    --cos_weight "${COS_WEIGHT:-1.0}" \
    --epochs "${EPOCHS:-32}" \
    --mask_ratio "${MASK_RATIO:-0.0}" \
    --lr "${LR:-3e-4}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --eval_batches "${EVAL_BATCHES:-24}" \
    --seed "${SEED:-0}" \
    --results_csv "${RESULTS_CSV:-res/delulu/projector_probe.csv}" \
    --tag "${TAG:-}" \
    ${SCORE_MASKED_ONLY:+--score_masked_only}
