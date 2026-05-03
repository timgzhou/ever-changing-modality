#!/bin/bash
# BEN-v2 S2+=S1: ablate tz_fusion_time (when fusion/cross-modal attention begins).
# Uses rank=1 hyperparams from sweep_best.json for each of transfer, peeking, addition.
# Fusion times: 0..12 (checkpoint was trained with 3)
# Usage: bash sh/ablation/benv2_fusion_time_sweep.sh s2 s1

DATASET="benv2"
STARTING_MOD="$1"
NEW_MOD="$2"
MODEL="evan_base"
TEACHERS_JSON="artifacts/sft_teachers.json"

if [ -z "$STARTING_MOD" ] || [ -z "$NEW_MOD" ]; then
    echo "Usage: bash sh/ablation/benv2_fusion_time_sweep.sh <starting_mod> <new_mod>"
    echo "  e.g. bash sh/ablation/benv2_fusion_time_sweep.sh s2 s1"
    exit 1
fi

mkdir -p logs/shot_ete_ablate_fusion_time res/ablation

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

for FT in $(seq 0 12); do
    for SELECT_BY in transfer peeking addition; do
        echo "[submit] ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | ${SELECT_BY} | tz_fusion_time=${FT}"
        sbatch sh/ablation/benv2_fusion_time_job.sh \
            "$DATASET" "$MODEL" "$STARTING_MOD" "$NEW_MOD" "$TEACHER" "$SELECT_BY" "$FT"
    done
done

# bash sh/ablation/benv2_fusion_time_sweep.sh s2 s1
