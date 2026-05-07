#!/bin/bash
#SBATCH --time=5:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/shot_ete/%j.out
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

source sh/env.sh
export TQDM_DISABLE=1

SWEEP_JSON="res/delulu-sweep/best_masking.json"
RESULTS_CSV="res/delulu/hptuned_masking_may6.csv"

ENTRY=$(jq -c ".\"${SELECT_BY}\".hparams" "$SWEEP_JSON")

if [ -z "$ENTRY" ] || [ "$ENTRY" = "null" ]; then
    echo "[error] no entry in ${SWEEP_JSON} for select_by=${SELECT_BY}"
    exit 1
fi

LR=$(echo "$ENTRY"           | jq -r '.lr')
WD=$(echo "$ENTRY"           | jq -r '.weight_decay')
EPOCHS=$(echo "$ENTRY"       | jq -r '.epochs')
MD=$(echo "$ENTRY"           | jq -r '.modality_dropout')
MD_START=$(echo "$ENTRY"     | jq -r '.modality_dropout_startmod')
MD_NEW=$(echo "$ENTRY"       | jq -r '.modality_dropout_newmod')
LF=$(echo "$ENTRY"           | jq -r '.labeled_frequency')
LS=$(echo "$ENTRY"           | jq -r '.labeled_start_fraction')
LL=$(echo "$ENTRY"           | jq -r '.lambda_latent')
LP=$(echo "$ENTRY"           | jq -r '.lambda_prefusion')
LD=$(echo "$ENTRY"           | jq -r '.lambda_distill')
MR=$(echo "$ENTRY"           | jq -r '.mae_mask_ratio')
LATENT_MASKED_ONLY=$(echo "$ENTRY"  | jq -r '.latent_masked_only')
PROTECT_LRM=$(echo "$ENTRY"         | jq -r '.protect_lrm')
USE_MASK_TOKEN=$(echo "$ENTRY"      | jq -r '.use_mask_token')
UNPROTECT=$(echo "$ENTRY"           | jq -r '.unprotect_starting_mod')

LATENT_MASKED_ONLY_FLAG=""
[ "$LATENT_MASKED_ONLY" = "true" ] && LATENT_MASKED_ONLY_FLAG="--latent_masked_only"
USE_MASK_TOKEN_FLAG=""
[ "$USE_MASK_TOKEN" = "true" ] && USE_MASK_TOKEN_FLAG="--use_mask_token"
UNPROTECT_FLAG=""
[ "$UNPROTECT" = "true" ] && UNPROTECT_FLAG="--unprotect_starting_mod"

WANDB_PROJECT="delulu-${DATASET}-${STARTING_MOD}-${NEW_MOD}"

echo "=== ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | select_by=${SELECT_BY} ==="
echo "    lr=${LR} wd=${WD} epochs=${EPOCHS} md=${MD} lf=${LF} ls=${LS}"

python -u shot_ete.py \
    --dataset "$DATASET" \
    --new_mod_group "$NEW_MOD" \
    --stage0_checkpoint "$TEACHER" \
    --wandb_project "$WANDB_PROJECT" \
    --lr "$LR" \
    --weight_decay "$WD" \
    --epochs "$EPOCHS" \
    --modality_dropout "$MD" \
    --modality_dropout_startmod "$MD_START" \
    --modality_dropout_newmod "$MD_NEW" \
    --labeled_frequency "$LF" \
    --labeled_start_fraction "$LS" \
    --lambda_latent "$LL" \
    --lambda_prefusion "$LP" \
    --lambda_distill "$LD" \
    --token_mask_ratio "$MR" \
    --protect_lrm "$PROTECT_LRM" \
    $LATENT_MASKED_ONLY_FLAG \
    $USE_MASK_TOKEN_FLAG \
    $UNPROTECT_FLAG \
    --active_losses latent prefusion distill ce \
    --select_by "$SELECT_BY" \
    --results_csv "$RESULTS_CSV" \
    --batch_size 32 \
    --num_workers 4
