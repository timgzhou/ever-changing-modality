#!/bin/bash
# BEN-v2 S2+=S1: sweep hyperparams from sweep_best.json with use_mask_token enabled.
# active_losses: latent distill ce  (prefusion excluded — incompatible with use_mask_token).
# Usage: bash sh/ablation/benv2_mask_token_sweep.sh

DATASET="benv2"
STARTING_MOD="$1"
NEW_MOD="$2"
MODEL="evan_base"
TEACHERS_JSON="artifacts/sft_teachers.json"
RESULTS_CSV="res/ablation/benv2_mask_token.csv"

if [ -z "$STARTING_MOD" ] || [ -z "$NEW_MOD" ]; then
    echo "Usage: bash sh/ablation/benv2_mask_token_sweep.sh <starting_mod> <new_mod>"
    echo "  e.g. bash sh/ablation/benv2_mask_token_sweep.sh s2 s1"
    exit 1
fi

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

for SELECT_BY in transfer peeking addition ens_addition; do
    echo "[submit] ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | ${SELECT_BY} | use_mask_token"
    sbatch sh/ablation/benv2_mask_token_sweep_job.sh \
        "$DATASET" "$MODEL" "$STARTING_MOD" "$NEW_MOD" "$TEACHER" "$SELECT_BY"
done

# bash sh/ablation/benv2_mask_token_sweep.sh
