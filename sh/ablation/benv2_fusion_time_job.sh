#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/ablate/fusion_time/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

DATASET="$1"
MODEL="$2"
STARTING_MOD="$3"
NEW_MOD="$4"
TEACHER="$5"
SELECT_BY="${6:-addition}"
FUSION_TIME="$7"

source sh/env.sh
export TQDM_DISABLE=1

SWEEP_JSON="res/delulu-sweep/best_masking.json"
RESULTS_CSV="res/ablation/benv2_fusion_time.csv"

mkdir -p res/ablation logs/shot_ete_ablate_fusion_time

HPARAMS=$(jq -c ".\"${SELECT_BY}\".hparams" "$SWEEP_JSON")

if [ -z "$HPARAMS" ] || [ "$HPARAMS" = "null" ]; then
    echo "[error] no hparams for select_by=${SELECT_BY} in ${SWEEP_JSON}"
    exit 1
fi

hp() { echo "$HPARAMS" | jq -r ".$1"; }

LR=$(hp lr)
WD=$(hp weight_decay)
EPOCHS=$(hp epochs)
MR=$(hp mae_mask_ratio)
MD=$(hp modality_dropout)
MD_SM=$(hp modality_dropout_startmod)
MD_NM=$(hp modality_dropout_newmod)
LF=$(hp labeled_frequency)
LS=$(hp labeled_start_fraction)
PROTECT_LRM=$(hp protect_lrm)
LAMBDA_LATENT=$(hp lambda_latent)
LAMBDA_PREFUSION=$(hp lambda_prefusion)
LAMBDA_DISTILL=$(hp lambda_distill)
LATENT_MASKED_ONLY=$(hp latent_masked_only)
UNPROTECT_STARTING_MOD=$(hp unprotect_starting_mod)
USE_MASK_TOKEN=$(hp use_mask_token)

BOOL_FLAGS=""
[ "$LATENT_MASKED_ONLY" = "true" ]      && BOOL_FLAGS="$BOOL_FLAGS --latent_masked_only"
[ "$UNPROTECT_STARTING_MOD" = "true" ]  && BOOL_FLAGS="$BOOL_FLAGS --unprotect_starting_mod"
[ "$USE_MASK_TOKEN" = "true" ]          && BOOL_FLAGS="$BOOL_FLAGS --use_mask_token"

echo "  fusion_time=${FUSION_TIME} select_by=${SELECT_BY} teacher=${TEACHER}"

python -u shot_ete.py \
    --dataset "$DATASET" \
    --new_mod_group "$NEW_MOD" \
    --stage0_checkpoint "$TEACHER" \
    --lr "$LR" \
    --weight_decay "$WD" \
    --epochs "$EPOCHS" \
    --token_mask_ratio "$MR" \
    --modality_dropout "$MD" \
    --modality_dropout_startmod "$MD_SM" \
    --modality_dropout_newmod "$MD_NM" \
    --labeled_frequency "$LF" \
    --labeled_start_fraction "$LS" \
    --protect_lrm "$PROTECT_LRM" \
    --lambda_latent "$LAMBDA_LATENT" \
    --lambda_prefusion "$LAMBDA_PREFUSION" \
    --lambda_distill "$LAMBDA_DISTILL" \
    --active_losses latent prefusion distill ce \
    --tz_fusion_time "$FUSION_TIME" \
    --results_csv "$RESULTS_CSV" \
    --eval_every_n_epochs 4 \
    --batch_size 32 \
    --num_workers 4 \
    $BOOL_FLAGS
