#!/bin/bash
#SBATCH --time=05:59:00
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
NEW_MOD="$4"
TEACHER="$5"
KL_TYPE="${6:-kd}"

source sh/env.sh
export TQDM_DISABLE=1
export PYTHONPATH="$PWD:$PYTHONPATH"

RESULTS_CSV="res/baselines/distillation/${DATASET}/${MODEL}/${STARTING_MOD}+=${NEW_MOD}.csv"
mkdir -p "$(dirname "$RESULTS_CSV")"

for LR in 1e-3 3e-4; do
    for TEMP in 0.5 1 2; do

        echo "=== ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | lr=${LR} kl=${KL_TYPE} temp=${TEMP} ==="

        # Skip if this exact combo already exists in the CSV
        # Columns: 1=model_type, 3=student_modality, 7=learning_rate, 10=temperature, 13=kl_type
        if [ -f "$RESULTS_CSV" ] && awk -F, -v m="$MODEL" -v s="$NEW_MOD" -v lr="$LR" -v temp="$TEMP" -v kl="$KL_TYPE" \
            'NR>1 && $1==m && $3==s && $7==lr && $10==temp && $13==kl {found=1; exit} END {exit !found}' "$RESULTS_CSV"; then
            echo "  [skip] already in CSV"
            continue
        fi

        python -u baseline/baseline_distillation.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --teacher_checkpoint "$TEACHER" \
            --modalities "$NEW_MOD" \
            --epochs 16 \
            --batch_size 64 \
            --num_workers 8 \
            --distillation_mode regular \
            --init_from_teacher \
            --lr "$LR" \
            --temperature "$TEMP" \
            --kl_type "$KL_TYPE" \
            --results_csv "$RESULTS_CSV"

    done
done

# Sort by best_val_agreement descending (col 20), fall back gracefully if empty
if [ -f "$RESULTS_CSV" ]; then
    (head -1 "$RESULTS_CSV" && tail -n +2 "$RESULTS_CSV" | sort -t, -k20 -rn) > "$RESULTS_CSV.tmp" \
        && mv "$RESULTS_CSV.tmp" "$RESULTS_CSV"
fi
