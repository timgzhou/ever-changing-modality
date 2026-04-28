#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/distillation/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

DATASET="$1"
MODEL="$2"
STARTING_MOD="$3"
TEACHER="$4"

source sh/env.sh
export TQDM_DISABLE=1
export PYTHONPATH="$PWD:$PYTHONPATH"

# Each entry is a space-separated list of modalities for one student run.
# Multi-word entries (multimodal students) are separated by commas within the entry.
declare -A STUDENT_MODS
# STUDENT_MODS['benv2']='s1 s2 s1,s2 s2_norgb'
STUDENT_MODS['benv2']='s2_norgb'
# STUDENT_MODS['dfc2020']='s1 s2 s1,s2 s2_norgb'
STUDENT_MODS['dfc2020']='s2_norgb'
STUDENT_MODS['eurosat']='swir rgb vre nir s2'



for STUDENT_ENTRY in ${STUDENT_MODS[$DATASET]}; do

    # Convert comma-separated modalities to space-separated for --modalities arg
    STUDENT_MOD_ARGS="${STUDENT_ENTRY//,/ }"
    # Label for paths/logs: use + as separator
    STUDENT_LABEL="${STUDENT_ENTRY//,/+}"

    # Skip if student is identical to teacher (single-mod only)
    [ "$STUDENT_LABEL" = "$STARTING_MOD" ] && continue

    # init_from_teacher only valid for single-modality students
    INIT_FLAG=""
    [[ "$STUDENT_ENTRY" != *","* ]] && INIT_FLAG="--init_from_teacher"

    RESULTS_CSV="res/baselines/distillation/${DATASET}/${MODEL}/${STARTING_MOD}+=${STUDENT_LABEL}.csv"
    mkdir -p "$(dirname "$RESULTS_CSV")"

    for LR in 1e-3 3e-4; do
        for KL_TYPE in ttm kd; do
            for TEMP in 0.5 1 2; do

                echo "=== ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${STUDENT_LABEL} | lr=${LR} kl=${KL_TYPE} temp=${TEMP} ==="

                # Skip if this exact combo already exists in the CSV
                # Columns: 1=model_type, 3=student_modality, 7=learning_rate, 10=temperature, 13=kl_type
                if [ -f "$RESULTS_CSV" ] && awk -F, -v m="$MODEL" -v s="$STUDENT_LABEL" -v lr="$LR" -v temp="$TEMP" -v kl="$KL_TYPE" \
                    'NR>1 && $1==m && $3==s && $7==lr && $10==temp && $13==kl {found=1; exit} END {exit !found}' "$RESULTS_CSV"; then
                    echo "  [skip] already in CSV"
                    continue
                fi

                python -u baseline/baseline_distillation.py \
                    --dataset "$DATASET" \
                    --model "$MODEL" \
                    --teacher_checkpoint "$TEACHER" \
                    --modalities $STUDENT_MOD_ARGS \
                    --epochs 16 \
                    --batch_size 64 \
                    --num_workers 8 \
                    --distillation_mode regular \
                    $INIT_FLAG \
                    --lr "$LR" \
                    --temperature "$TEMP" \
                    --kl_type "$KL_TYPE" \
                    --results_csv "$RESULTS_CSV" \
                    --init_from_teacher

            done
        done
    done

    # Sort by best_val_agreement descending (col 20), fall back gracefully if empty
    if [ -f "$RESULTS_CSV" ]; then
        (head -1 "$RESULTS_CSV" && tail -n +2 "$RESULTS_CSV" | sort -t, -k20 -rn) > "$RESULTS_CSV.tmp" \
            && mv "$RESULTS_CSV.tmp" "$RESULTS_CSV"
    fi

done
