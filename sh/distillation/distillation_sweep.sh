#!/bin/bash
# Launcher: submits one sbatch job per dataset+model+starting_mod combo.
# Usage: bash sh/distillation/distillation_sweep.sh

TEACHERS_JSON="artifacts/sft_teachers.json"

declare -A STARTING_MODS
# STARTING_MODS['benv2']='s2 s2_rgb s1'
# STARTING_MODS['dfc2020']='s2 s2_rgb s1'
STARTING_MODS['benv2']='s2_rgb'
STARTING_MODS['dfc2020']='s2_rgb'
STARTING_MODS['eurosat']='vre swir s2 rgb nir'

# DATASETS=('benv2' 'dfc2020' 'eurosat')
DATASETS=('dfc2020')
MODELS=('evan_base' 'evan_large')

mkdir -p logs/distillation

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for STARTING_MOD in ${STARTING_MODS[$DATASET]}; do

            TEACHER_KEY="${DATASET}/${STARTING_MOD}/${MODEL}"
            TEACHER=$(jq -r ".\"${TEACHER_KEY}\".checkpoint // empty" "$TEACHERS_JSON")
            if [ -z "$TEACHER" ]; then
                echo "[skip] no teacher found for ${TEACHER_KEY}"
                continue
            fi
            if [ ! -f "$TEACHER" ]; then
                echo "[skip] teacher checkpoint missing: $TEACHER"
                continue
            fi

            echo "[submit] ${DATASET} | ${MODEL} | ${STARTING_MOD}"
            sbatch sh/distillation/distillation_sweep_job.sh "$DATASET" "$MODEL" "$STARTING_MOD" "$TEACHER"

        done
    done
done


#   bash sh/distillation/distillation_sweep.sh