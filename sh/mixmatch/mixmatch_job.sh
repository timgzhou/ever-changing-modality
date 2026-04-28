#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/mixmatch/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

DATASET="$1"
MODALITY="$2"

source sh/env.sh
export TQDM_DISABLE=1

RESULTS_CSV="res/baselines/mixmatch/baseline_mixmatch_${DATASET}.csv"
mkdir -p "$(dirname "$RESULTS_CSV")"

echo "=== MixMatch | ${DATASET} | modality=${MODALITY} ==="

MODELS=('evan_large' 'evan_base')
DINO_FLAGS=('--use_dino_weights')
LRS=('0.0003' '0.0004')
ALPHAS=('0.5')
LAMBDA_US=('50' '75')

for MODEL in "${MODELS[@]}"; do
    for DINO_FLAG in "${DINO_FLAGS[@]}"; do
        if [[ "${DINO_FLAG}" == "--use_dino_weights" ]]; then
            DINO_VAL="True"
        else
            DINO_VAL="False"
        fi

        for LR in "${LRS[@]}"; do
            for ALPHA in "${ALPHAS[@]}"; do
                for LAMBDA_U in "${LAMBDA_US[@]}"; do
                    echo "--- model=${MODEL} dino=${DINO_VAL} lr=${LR} alpha=${ALPHA} lambda_u=${LAMBDA_U} ---"

                    if [[ -f "${RESULTS_CSV}" ]] && grep -qP "^[^,]+,${MODALITY},fft,${LR},[^,]+,[^,]+,[^,]+,2,0\.5,${ALPHA},${LAMBDA_U},[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,${DINO_VAL}$" "${RESULTS_CSV}" 2>/dev/null; then
                        echo "  → already in results, skipping"
                        continue
                    fi

                    python -u baseline/baseline_mixmatch.py \
                        --dataset "$DATASET" \
                        --modality "$MODALITY" \
                        --train_mode fft \
                        --epochs 20 \
                        --lr "$LR" \
                        --K 2 \
                        --temperature 0.5 \
                        --alpha "$ALPHA" \
                        --lambda_u "$LAMBDA_U" \
                        --num_workers 8 \
                        --results_csv "$RESULTS_CSV" \
                        --batch_size 32 \
                        --model "$MODEL" \
                        ${DINO_FLAG}
                done
            done
        done
    done
done
