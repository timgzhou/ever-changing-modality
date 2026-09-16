#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/projector_probe/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=FAIL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One projector-probe arm: frozen encoder, train ONLY the cross-modal projector.
# See analysis/train_projector_probe.py for what this measures and why.
#
# Required: CKPT, SRC, TGT, DEPTH, LOSS
# Optional: COS_WEIGHT EPOCHS LR BATCH_SIZE EVAL_BATCHES SEED RESULTS_CSV TAG
#
# 2h walltime: the encoder runs under no_grad and only a ~14-28M param projector
# gets gradients, so an epoch is minutes, not the ~3h a full delulu run takes.
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
    --epochs "${EPOCHS:-8}" \
    --lr "${LR:-3e-4}" \
    --batch_size "${BATCH_SIZE:-16}" \
    --eval_batches "${EVAL_BATCHES:-24}" \
    --seed "${SEED:-0}" \
    --results_csv "${RESULTS_CSV:-res/delulu/projector_probe.csv}" \
    --tag "${TAG:-}"
