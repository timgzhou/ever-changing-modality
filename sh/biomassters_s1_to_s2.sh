#!/bin/bash
#SBATCH --time=7:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/biomassters_s1_to_s2/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# End-to-end: train an S1 ViT-base (EVAN-base) on BioMassters via SFT,
# then SHOT-adapt it to take S2 input.
#
# Usage:  sbatch sh/biomassters_s1_to_s2.sh
#
# Optional env overrides:
#   MODEL          (default evan_base)
#   LR / WD        stage-0 SFT lr / weight decay (default 0.0005 / 0.01)
#   NUM_TIME_STEPS temporal pooling window, <=12 (default 6)
#   SELECT_BY      SHOT hparam bucket in the sweep json (default addition)

set -euo pipefail

source sh/env.sh
export TQDM_DISABLE=1

# ---- config -------------------------------------------------------------
DATASET="biomassters"
MODEL="${MODEL:-evan_base}"
STARTING_MOD="s1"
NEW_MOD="s2"
TRAIN_MODE="fft"
LR="${LR:-0.0005}"
WD="${WD:-0.01}"
NUM_TIME_STEPS="${NUM_TIME_STEPS:-6}"
SELECT_BY="${SELECT_BY:-addition}"

mkdir -p logs/biomassters_s1_to_s2 checkpoints

# Deterministic checkpoint name so stage 1 can locate the teacher.
CKPT_NAME="sft_${MODEL}_${DATASET}_${STARTING_MOD}_${TRAIN_MODE}_${SLURM_JOB_ID}"
TEACHER="checkpoints/${CKPT_NAME}.pt"

# =========================================================================
# Stage 0 — SFT an S1-only EVAN-base on BioMassters
# =========================================================================
echo "=== [stage 0] SFT ${MODEL} on ${DATASET} (${STARTING_MOD}) ==="
python -u train_sft.py \
    --model "${MODEL}" \
    --dataset "${DATASET}" \
    --modalities "${STARTING_MOD}" \
    --train_mode "${TRAIN_MODE}" \
    --epochs 24 \
    --lr "${LR}" \
    --weight_decay "${WD}" \
    --num_time_steps "${NUM_TIME_STEPS}" \
    --use_dino_weights \
    --checkpoint_name "${CKPT_NAME}"

if [ ! -f "${TEACHER}" ]; then
    echo "[error] expected teacher checkpoint not found: ${TEACHER}"
    exit 1
fi
echo "=== [stage 0] done — teacher: ${TEACHER} ==="

# =========================================================================
# Stage 1 — SHOT-adapt the S1 teacher to S2 input (shot_ete)
# Hparams pulled from the tuned sweep json (same source as shot_ete_sweep_job.sh)
# =========================================================================
SWEEP_JSON="res/delulu-sweep/best_masking.json"
RESULTS_CSV="res/delulu/biomassters_s1_to_s2.csv"

ENTRY=$(jq -c ".\"${SELECT_BY}\".hparams" "$SWEEP_JSON")
if [ -z "$ENTRY" ] || [ "$ENTRY" = "null" ]; then
    echo "[error] no entry in ${SWEEP_JSON} for select_by=${SELECT_BY}"
    exit 1
fi

S_LR=$(echo "$ENTRY"          | jq -r '.lr')
S_WD=$(echo "$ENTRY"          | jq -r '.weight_decay')
S_EPOCHS=$(echo "$ENTRY"      | jq -r '.epochs')
MD=$(echo "$ENTRY"            | jq -r '.modality_dropout')
MD_START=$(echo "$ENTRY"      | jq -r '.modality_dropout_startmod')
MD_NEW=$(echo "$ENTRY"        | jq -r '.modality_dropout_newmod')
LF=$(echo "$ENTRY"            | jq -r '.labeled_frequency')
LS=$(echo "$ENTRY"            | jq -r '.labeled_start_fraction')
LL=$(echo "$ENTRY"            | jq -r '.lambda_latent')
LP=$(echo "$ENTRY"            | jq -r '.lambda_prefusion')
LD=$(echo "$ENTRY"            | jq -r '.lambda_distill')
MR=$(echo "$ENTRY"            | jq -r '.mae_mask_ratio')
LATENT_MASKED_ONLY=$(echo "$ENTRY" | jq -r '.latent_masked_only')
PROTECT_LRM=$(echo "$ENTRY"        | jq -r '.protect_lrm')
USE_MASK_TOKEN=$(echo "$ENTRY"     | jq -r '.use_mask_token')
UNPROTECT=$(echo "$ENTRY"          | jq -r '.unprotect_starting_mod')

LATENT_MASKED_ONLY_FLAG=""
[ "$LATENT_MASKED_ONLY" = "true" ] && LATENT_MASKED_ONLY_FLAG="--latent_masked_only"
USE_MASK_TOKEN_FLAG=""
[ "$USE_MASK_TOKEN" = "true" ] && USE_MASK_TOKEN_FLAG="--use_mask_token"
UNPROTECT_FLAG=""
[ "$UNPROTECT" = "true" ] && UNPROTECT_FLAG="--unprotect_starting_mod"

WANDB_PROJECT="delulu-${DATASET}-${STARTING_MOD}-${NEW_MOD}"

echo "=== [stage 1] SHOT ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | select_by=${SELECT_BY} ==="
echo "    lr=${S_LR} wd=${S_WD} epochs=${S_EPOCHS} md=${MD} lf=${LF} ls=${LS}"

python -u shot_ete.py \
    --dataset "${DATASET}" \
    --new_mod_group "${NEW_MOD}" \
    --stage0_checkpoint "${TEACHER}" \
    --wandb_project "${WANDB_PROJECT}" \
    --lr "${S_LR}" \
    --weight_decay "${S_WD}" \
    --epochs "${S_EPOCHS}" \
    --modality_dropout "${MD}" \
    --modality_dropout_startmod "${MD_START}" \
    --modality_dropout_newmod "${MD_NEW}" \
    --labeled_frequency "${LF}" \
    --labeled_start_fraction "${LS}" \
    --lambda_latent "${LL}" \
    --lambda_prefusion "${LP}" \
    --lambda_distill "${LD}" \
    --token_mask_ratio "${MR}" \
    --protect_lrm "${PROTECT_LRM}" \
    ${LATENT_MASKED_ONLY_FLAG} \
    ${USE_MASK_TOKEN_FLAG} \
    ${UNPROTECT_FLAG} \
    --active_losses latent prefusion distill ce \
    --select_by "${SELECT_BY}" \
    --results_csv "${RESULTS_CSV}" \
    --batch_size 32 \
    --num_workers 4 \
    --num_time_steps "${NUM_TIME_STEPS}"

echo "=== done ==="
