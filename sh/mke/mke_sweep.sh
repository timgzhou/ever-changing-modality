#!/bin/bash
# Launcher: one sbatch job per dataset+model+teacher_mod+student_mods combo.
# Student modalities = teacher + all non-empty subsets of that teacher's allowed companions.
#
# Allowed companions per teacher (modalities that make sense to add):
#   s2    → s1 only          (s2_rgb/s2_norgb are sub-bands of s2, not new sensors)
#   s1    → s2 only
#   s2_rgb → s2_norgb, s1   (s2_norgb completes s2; s1 is a new sensor)
#   rgb   (eurosat) → s2_norgb only
#   vre/nir/swir (eurosat) → remaining eurosat sub-bands
#
# Usage: bash sh/mke/mke_sweep.sh

TEACHERS_JSON="artifacts/sft_teachers.json"

# Per-teacher companion modalities (space-separated).
# Student = teacher + non-empty subset of companions.
declare -A COMPANIONS
# COMPANIONS['benv2/s2']='s1'
# COMPANIONS['benv2/s1']='s2'
# COMPANIONS['benv2/s2_rgb']='s2_norgb s1'
# COMPANIONS['dfc2020/s2']='s1'
# COMPANIONS['dfc2020/s1']='s2'
# COMPANIONS['dfc2020/s2_rgb']='s2_norgb s1'
# COMPANIONS['eurosat/rgb']='s2_norgb'
COMPANIONS['eurosat/rgb']='nir vre'
# COMPANIONS['eurosat/swir']='vre rgb nir'
COMPANIONS['eurosat/vre']='nir rgb'
COMPANIONS['eurosat/swir']='nir'

declare -A TEACHER_MODS
TEACHER_MODS['benv2']='s2 s1 s2_rgb'
TEACHER_MODS['dfc2020']='s2 s1 s2_rgb'
# TEACHER_MODS['eurosat']='rgb swir vre'
TEACHER_MODS['eurosat']='swir'

# DATASETS=('benv2' 'dfc2020' 'eurosat')
DATASETS=('eurosat')
MODELS=('evan_base' 'evan_large')

mkdir -p logs/mke

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for TEACHER_MOD in ${TEACHER_MODS[$DATASET]}; do

            TEACHER_KEY="${DATASET}/${TEACHER_MOD}/${MODEL}"
            TEACHER=$(jq -r ".\"${TEACHER_KEY}\".checkpoint // empty" "$TEACHERS_JSON")
            if [ -z "$TEACHER" ]; then
                echo "[skip] no teacher found for ${TEACHER_KEY}"
                continue
            fi
            if [ ! -f "$TEACHER" ]; then
                echo "[skip] teacher checkpoint missing: $TEACHER"
                continue
            fi

            COMPANION_KEY="${DATASET}/${TEACHER_MOD}"
            read -ra OTHER_MODS <<< "${COMPANIONS[$COMPANION_KEY]}"
            N_OTHER=${#OTHER_MODS[@]}

            if [ "$N_OTHER" -eq 0 ]; then
                echo "[skip] no companions defined for ${COMPANION_KEY}"
                continue
            fi

            # Iterate over all non-empty subsets of OTHER_MODS (2^N - 1 combos)
            for ((mask=1; mask < (1 << N_OTHER); mask++)); do
                STUDENT_MODS=("$TEACHER_MOD")
                for ((bit=0; bit < N_OTHER; bit++)); do
                    if (( (mask >> bit) & 1 )); then
                        STUDENT_MODS+=("${OTHER_MODS[$bit]}")
                    fi
                done
                STUDENT_STR="${STUDENT_MODS[*]}"  # space-separated for passing to job

                echo "[submit] ${DATASET} | ${MODEL} | teacher=${TEACHER_MOD} | student=${STUDENT_STR} | ckpt=${TEACHER}"
                sbatch sh/mke/mke_sweep_job.sh \
                    "$DATASET" "$MODEL" "$TEACHER_MOD" "$TEACHER" "$STUDENT_STR"
            done

        done
    done
done

# bash sh/mke/mke_sweep.sh
