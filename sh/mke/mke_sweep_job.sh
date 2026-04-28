#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/mke/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

DATASET="$1"
MODEL="$2"
TEACHER_MOD="$3"
TEACHER="$4"
# Remaining args are student modalities (space-separated, passed as one quoted string)
read -ra STUDENT_MODS <<< "$5"

source sh/env.sh
export TQDM_DISABLE=1

RESULTS_CSV="res/baselines/mke/${DATASET}.csv"
mkdir -p "$(dirname "$RESULTS_CSV")"

STUDENT_STR="${STUDENT_MODS[*]}"  # e.g. "s2 s1"

echo "=== MKE | ${DATASET} | ${MODEL} | teacher=${TEACHER_MOD} | student=${STUDENT_STR} ==="

STUDENT_CSV="${STUDENT_MODS[*]}"
STUDENT_CSV="${STUDENT_CSV// /+}"  # "s2 s1" -> "s2+s1"

for LR in 1e-3 3e-4 1e-4; do
    for WD in 1e-2 1e-5; do
        if [ -f "$RESULTS_CSV" ] && awk -F',' -v m="$MODEL" -v tm="$TEACHER_MOD" -v sm="$STUDENT_CSV" \
            -v lr="$LR" -v wd="$WD" \
            'NR>1 && $1==m && $2==tm && $3==sm && $7==lr && $8==wd {found=1; exit} END {exit !found}' \
            "$RESULTS_CSV"; then
            echo "[skip] already exists: model=${MODEL} teacher=${TEACHER_MOD} student=${STUDENT_CSV} lr=${LR} wd=${WD}"
            continue
        fi
        echo "--- lr=${LR} wd=${WD} ---"
        python -u baseline/baseline_mke.py \
            --dataset "$DATASET" \
            --model "$MODEL" \
            --modalities "${STUDENT_MODS[@]}" \
            --teacher_checkpoint "$TEACHER" \
            --train_mode fft \
            --use_dino_weights \
            --epochs 20 \
            --lr "$LR" \
            --weight_decay "$WD" \
            --batch_size 32 \
            --num_workers 8 \
            --warmup_epochs 1 \
            --results_csv "$RESULTS_CSV" \
            --wandb_project delulu-mke
    done
done
