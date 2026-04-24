#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/train_sft/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Expected env vars (set by train_sft_all.sh):
#   DATASET, MODEL, TRAIN_MODE, MODALITY_ENTRY, LR, WD

source sh/env.sh
export TQDM_DISABLE=1

MODALITIES="${MODALITY_ENTRY//+/ }"
MODALITY_KEY="${MODALITY_ENTRY}"
RESULTS_CSV="res/train_sft/${DATASET}.csv"

echo "Running: model=${MODEL} dataset=${DATASET} train_mode=${TRAIN_MODE} modalities=${MODALITIES} lr=${LR} wd=${WD}"

for USE_DINO in 0 1; do
    DINO_VAL="False"
    DINO_FLAG=""
    if [ "${USE_DINO}" = "1" ]; then
        DINO_VAL="True"
        DINO_FLAG="--use_dino_weights"
    fi

    if grep -qP "^${DATASET},${MODEL},${MODALITY_KEY},${TRAIN_MODE},[^,]+,[^,]+,${LR},${WD},[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,${DINO_VAL}" "${RESULTS_CSV}" 2>/dev/null; then
        echo "  → dino_init=${DINO_VAL} already in results, skipping"
        continue
    fi

    echo "--- use_dino=${USE_DINO} ---"
    python -u train_sft.py \
        --model ${MODEL} \
        --dataset ${DATASET} \
        --modalities ${MODALITIES} \
        --train_mode ${TRAIN_MODE} \
        --epochs 24 \
        --lr ${LR} \
        --weight_decay ${WD} \
        ${DINO_FLAG}
done
