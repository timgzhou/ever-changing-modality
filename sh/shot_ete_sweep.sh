#!/bin/bash
# Launcher: submits 5 sbatch jobs per (dataset, model, start_mod->new_mod, select_by) combo.
# Usage: bash sh/shot_ete_sweep.sh

TEACHERS_JSON="artifacts/sft_teachers.json"

# Hardcoded pairs: "dataset start_mod new_mod"
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
    # "eurosat vre     rgb"
    # "eurosat rgb     nir"
    # "eurosat nir     rgb"
    # "eurosat nir     vre"
    # "eurosat vre     nir"
)

MODELS=('evan_base')
N_RUNS=5

mkdir -p logs/shot_ete

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

        for SELECT_BY in transfer peeking addition; do
            echo "[submit x${N_RUNS}] ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | ${SELECT_BY}"
            for _ in $(seq 1 "$N_RUNS"); do
                sbatch sh/shot_ete_sweep_job.sh "$DATASET" "$MODEL" "$STARTING_MOD" "$NEW_MOD" "$TEACHER" "$SELECT_BY"
            done
        done

    done
done

# bash sh/shot_ete_sweep.sh
