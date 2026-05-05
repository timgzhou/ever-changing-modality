#!/bin/bash
# Launcher: submits one sbatch job per (dataset, model, start_mod->new_mod) pair.
# Usage: bash sh/distillation/distillation_sweep.sh

TEACHERS_JSON="artifacts/sft_teachers.json"

# Same pairs as sh/shot_ete_sweep.sh
PAIRS=(
    "benv2   s2_rgb  s2_norgb"
    "benv2   s2_rgb  s1"
    "benv2   s1      s2"
    "benv2   s2      s1"
    "dfc2020 s2_rgb  s2_norgb"
    "dfc2020 s2_rgb  s1"
    "dfc2020 s1      s2"
    "dfc2020 s2      s1"
    "eurosat rgb     vre"
)

MODELS=('evan_base')

mkdir -p logs/distillation

for PAIR in "${PAIRS[@]}"; do
    read -r DATASET STARTING_MOD NEW_MOD <<< "$PAIR"

    for MODEL in "${MODELS[@]}"; do
        TEACHER_KEY="${DATASET}/${STARTING_MOD}/${MODEL}"
        TEACHER=$(jq -r ".\"${TEACHER_KEY}\".checkpoint // empty" "$TEACHERS_JSON")
        if [ -z "$TEACHER" ]; then
            echo "[skip] no teacher for ${TEACHER_KEY}"
            continue
        fi
        if [ ! -f "$TEACHER" ]; then
            echo "[skip] checkpoint missing: $TEACHER"
            continue
        fi

        for KL_TYPE in ttm kd; do
            echo "[submit] ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | kl=${KL_TYPE}"
            sbatch sh/distillation/distillation_sweep_job.sh "$DATASET" "$MODEL" "$STARTING_MOD" "$NEW_MOD" "$TEACHER" "$KL_TYPE"
        done
    done
done

# bash sh/distillation/distillation_sweep.sh
