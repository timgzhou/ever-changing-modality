#!/bin/bash
# BEN-v2 S2+=S1: ablate active_losses starting from full set (no mae),
# removing latent and prefusion one at a time.
# Loss combos:
#   latent prefusion distill ce  (full)
#   prefusion distill ce         (no latent)
#   latent distill ce            (no prefusion)
# Usage: bash sh/ablation/benv2_active_losses_sweep.sh

DATASET="benv2"
STARTING_MOD="s2"
NEW_MOD="s1"
MODEL="evan_base"
TEACHERS_JSON="artifacts/sft_teachers.json"

mkdir -p logs/shot_ete res/ablation

TEACHER_KEY="${DATASET}/${STARTING_MOD}/${MODEL}"
TEACHER=$(jq -r ".\"${TEACHER_KEY}\".checkpoint // empty" "$TEACHERS_JSON")
if [ -z "$TEACHER" ]; then
    echo "[error] no teacher found for ${TEACHER_KEY}"
    exit 1
fi
if [ ! -f "$TEACHER" ]; then
    echo "[error] teacher checkpoint missing: $TEACHER"
    exit 1
fi

declare -a LOSS_COMBOS=(
    "latent prefusion distill ce"
    "prefusion distill ce"
    "latent distill ce"
)

for LOSSES in "${LOSS_COMBOS[@]}"; do
    for SELECT_BY in transfer peeking addition ens_addition; do
        LOSSES_TAG=$(echo "$LOSSES" | tr ' ' '+')
        echo "[submit] ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | ${SELECT_BY} | losses=${LOSSES_TAG}"
        sbatch sh/ablation/benv2_active_losses_job.sh \
            "$DATASET" "$MODEL" "$STARTING_MOD" "$NEW_MOD" "$TEACHER" "$SELECT_BY" "$LOSSES"
    done
done

# bash sh/ablation/benv2_active_losses_sweep.sh
