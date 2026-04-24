#!/bin/bash
#SBATCH --time=11:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/rsfm_sft/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Expected env vars (set by rsfm_sft_all.sh):
#   MODEL, DATASET, TRAIN_MODE, MODALITY

source sh/env.sh
export TQDM_DISABLE=1

RESULTS_CSV="res/rsfm/rsfm_results.csv"
LRS=('0.001' '0.0005' '0.0001')
WDS=('0.01' '0.0001' '0')

for LR in "${LRS[@]}"; do
    for WD in "${WDS[@]}"; do
        echo "Running: model=${MODEL} dataset=${DATASET} train_mode=${TRAIN_MODE} modality=${MODALITY} lr=${LR} wd=${WD}"
        if grep -qP "^${MODEL},${DATASET},${MODALITY},${TRAIN_MODE},[^,]+,${LR},${WD}," "${RESULTS_CSV}" 2>/dev/null; then
            echo "  → already in results, skipping"
            continue
        fi
        python -u rsfm_sft.py \
            --model ${MODEL} \
            --dataset ${DATASET} \
            --modality ${MODALITY} \
            --train_mode ${TRAIN_MODE} \
            --epochs 20 \
            --lr ${LR} \
            --weight_decay ${WD}
    done
done
