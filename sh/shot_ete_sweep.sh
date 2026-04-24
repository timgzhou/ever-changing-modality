#!/bin/bash
# Launcher: submits one sbatch job per dataset+model+starting_mod+new_mod combo.
# Usage: bash sh/shot_ete_sweep.sh

TEACHERS_JSON="artifacts/sft_teachers.json"

declare -A STARTING_MODS
STARTING_MODS['benv2']='s2 s2_rgb s1'
STARTING_MODS['dfc2020']='s2 s2_rgb s1'
# STARTING_MODS['eurosat']='rgb vre swir'
STARTING_MODS['eurosat']='rgb'

declare -A ALL_MODS
ALL_MODS['benv2']='s2 s2_rgb s1 s2_norgb'
ALL_MODS['dfc2020']='s2 s2_rgb s1 s2_norgb'
ALL_MODS['eurosat']='swir rgb vre nir '

# DATASETS=('benv2' 'dfc2020' 'eurosat')

DATASETS=('eurosat')
MODELS=('evan_base' 'evan_large')
RESULTS_CSV="res/delulu/hptuned_apr21.csv"

mkdir -p logs/shot_ete

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

            for NEW_MOD in ${ALL_MODS[$DATASET]}; do
                [ "$NEW_MOD" = "$STARTING_MOD" ] && continue
                { [ "$STARTING_MOD" = "s2_rgb" ] && [ "$NEW_MOD" = "s2" ]; } && continue
                { [ "$STARTING_MOD" = "s2" ] && [ "$NEW_MOD" = "s2_rgb" ]; } && continue
                { [ "$STARTING_MOD" = "s2" ] && [ "$NEW_MOD" = "s2_norgb" ]; } && continue
                { [ "$STARTING_MOD" = "s2_norgb" ] && [ "$NEW_MOD" = "s2" ]; } && continue

                if [ -f "$RESULTS_CSV" ]; then
                    ROW_COUNT=$(tail -n +2 "$RESULTS_CSV" | awk -F',' -v ds="$DATASET" -v mo="$MODEL" -v sm="$STARTING_MOD" -v nm="$NEW_MOD" '$1==ds && $2==mo && $3==sm && $4==nm' | wc -l)
                    if [ "$ROW_COUNT" -ge 16 ]; then
                        echo "[skip] already ${ROW_COUNT} rows for ${DATASET}/${MODEL}/${STARTING_MOD}/${NEW_MOD}"
                        continue
                    fi
                fi

                for SELECT_BY in transfer peeking addition ens_addition; do
                    echo "[submit] ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | ${SELECT_BY}"
                    sbatch sh/shot_ete_sweep_job.sh "$DATASET" "$MODEL" "$STARTING_MOD" "$NEW_MOD" "$TEACHER" "$SELECT_BY"
                done
            done

        done
    done
done

# bash sh/shot_ete_sweep.sh
