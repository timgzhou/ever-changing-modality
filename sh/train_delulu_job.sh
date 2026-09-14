#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/train_delulu/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One train_delulu direction, any dataset. Generic counterpart to
# sh/train_delulu_dfc2020_job.sh (which hardcodes the DFC2020 specifics).
#
# Expected env vars (set by sh/train_delulu_all.sh):
#   DATASET, NEW, TEACHER            required
#   EPOCHS, BATCH_SIZE, SEED, LAMBDA_LATENT, STUDENT_INIT, RESULTS_CSV   optional
#
# The STARTING modality is not passed: train_delulu.py reads it back out of the
# teacher checkpoint's evan_config, so teacher and student can never disagree.

set -euo pipefail
source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p logs/train_delulu checkpoints res/delulu

: "${DATASET:?}"; : "${NEW:?}"; : "${TEACHER:?}"
if [ ! -f "${TEACHER}" ]; then
    echo "[error] teacher checkpoint not found: ${TEACHER}"; exit 1
fi

# Delulu hyperparameters: the biomassters s1->s2 best config, carried over as a
# first pass for every dataset (NOT tuned per dataset -- sweep afterwards).
# Kept identical to sh/train_delulu_dfc2020_job.sh so the two are comparable.
LR="${LR:-0.0001569391767106977}"
WEIGHT_DECAY="${WEIGHT_DECAY:-3.351617951860976e-05}"
MODALITY_DROPOUT="${MODALITY_DROPOUT:-0.3}"
MODALITY_DROPOUT_STARTMOD="${MODALITY_DROPOUT_STARTMOD:-0.33189226742900324}"
MODALITY_DROPOUT_NEWMOD="${MODALITY_DROPOUT_NEWMOD:-0.17068517311514753}"
LABELED_FREQUENCY="${LABELED_FREQUENCY:-0.23002477810989655}"
LABELED_START_FRACTION="${LABELED_START_FRACTION:-0}"
# The latent loss was inflated by embed_dim (768) until 2026-08-20, so this
# carried-over value was tuned at a scale where latent was ~98% of the signal;
# at the corrected scale it contributes ~5%. Sweep it from the launcher.
LAMBDA_LATENT="${LAMBDA_LATENT:-0.3613664751387723}"
LAMBDA_PREFUSION="${LAMBDA_PREFUSION:-0.6430194633931678}"
LAMBDA_DISTILL="${LAMBDA_DISTILL:-0.15374988356364516}"
TOKEN_MASK_RATIO="${TOKEN_MASK_RATIO:-0.40414477259411485}"
PROTECT_LRM="${PROTECT_LRM:-0.0}"
SEED="${SEED:-0}"
STUDENT_INIT="${STUDENT_INIT:-teacher}"
EPOCHS="${EPOCHS:-64}"
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULTS_CSV="${RESULTS_CSV:-res/delulu/${DATASET}_unimodal_pairs.csv}"

# BioMassters is temporal: pool features over this many timesteps (<=12).
EXTRA_ARGS=""
if [ "${DATASET}" = "biomassters" ]; then
    EXTRA_ARGS="--num_time_steps ${NUM_TIME_STEPS:-12}"
fi

# START is informational only (train_delulu.py reads the real starting modality
# out of the teacher checkpoint); it is echoed so the launcher's in-flight guard
# can identify which direction a queued job is running.
echo "=== ${DATASET} | ${START:-?} -> +${NEW} | teacher=${TEACHER} ==="
echo "    lr=${LR} epochs=${EPOCHS} bs=${BATCH_SIZE} lambda_latent=${LAMBDA_LATENT} seed=${SEED}"

python -u train_delulu.py \
    --dataset "${DATASET}" \
    --new_mod_group "${NEW}" \
    --stage0_checkpoint "${TEACHER}" \
    --active_losses latent prefusion distill ce \
    --wandb_project "delulu-${DATASET}-pairs" \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --modality_dropout "${MODALITY_DROPOUT}" \
    --modality_dropout_startmod "${MODALITY_DROPOUT_STARTMOD}" \
    --modality_dropout_newmod "${MODALITY_DROPOUT_NEWMOD}" \
    --labeled_frequency "${LABELED_FREQUENCY}" \
    --labeled_start_fraction "${LABELED_START_FRACTION}" \
    --lambda_latent "${LAMBDA_LATENT}" \
    --lambda_prefusion "${LAMBDA_PREFUSION}" \
    --lambda_distill "${LAMBDA_DISTILL}" \
    --token_mask_ratio "${TOKEN_MASK_RATIO}" \
    --protect_lrm "${PROTECT_LRM}" \
    --latent_masked_only \
    --student_init "${STUDENT_INIT}" \
    --seed "${SEED}" \
    --save_checkpoint \
    --results_csv "${RESULTS_CSV}" \
    ${EXTRA_ARGS}
