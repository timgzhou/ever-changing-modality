#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/freematch/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

DATASET="$1"
MODALITY="$2"
# Optional 3rd arg: "no_strong_aug" runs the augmentation-parity ablation.
ABLATION="${3:-}"

if [[ "${ABLATION}" == "no_strong_aug" ]]; then
    STRONG_AUG_FLAG="--no_strong_aug"
    NO_STRONG_VAL="True"
else
    STRONG_AUG_FLAG=""
    NO_STRONG_VAL="False"
fi

source sh/env.sh
export TQDM_DISABLE=1

RESULTS_CSV="res/baselines/freematch/baseline_freematch_${DATASET}.csv"
mkdir -p "$(dirname "$RESULTS_CSV")"

echo "=== FreeMatch | ${DATASET} | modality=${MODALITY} | no_strong_aug=${NO_STRONG_VAL} ==="

MODELS=('evan_base')
DINO_FLAGS=('--use_dino_weights')
LRS=('0.0003' '0.0004')
# SAF weight: paper uses w_f=0.01 for CIFAR-10/10-label, 0.05 elsewhere.
LAMBDA_ES=('0.01' '0.05')
# EMA decay for the adaptive threshold. Paper Table 11 ablates 0.9/0.99/0.999/0.9999;
# 0.999 is best, 0.99 adapts faster and is the sensible second point for short runs.
EMA_MOMENTUMS=('0.999' '0.99')

# lambda_u is fixed at the paper default of 1.0 (FreeMatch does not ramp it).
LAMBDA_U='1.0'
# FreeMatch does NOT sharpen pseudo-labels — temperature must stay at 1.0.
TEMPERATURE='1.0'

for MODEL in "${MODELS[@]}"; do
    for DINO_FLAG in "${DINO_FLAGS[@]}"; do
        if [[ "${DINO_FLAG}" == "--use_dino_weights" ]]; then
            DINO_VAL="True"
        else
            DINO_VAL="False"
        fi

        for LR in "${LRS[@]}"; do
            for LAMBDA_E in "${LAMBDA_ES[@]}"; do
                for EMA_M in "${EMA_MOMENTUMS[@]}"; do
                    echo "--- model=${MODEL} dino=${DINO_VAL} lr=${LR} lambda_e=${LAMBDA_E} ema=${EMA_M} ---"

                    # Resume guard: skip if this exact config already has a row.
                    # Columns: model_type,modality,train_mode,learning_rate,weight_decay,
                    #   trainable_params,epochs,lambda_u,lambda_e,ema_momentum,temperature,
                    #   use_quantile,clip_thresh,no_strong_aug,metric_name,test_metric,
                    #   best_val_metric,best_val_test_metric,saved_checkpoint,global_rep,
                    #   use_dino_weights,use_s2dino_weights
                    # NOTE: no_strong_aug is pinned to the ${NO_STRONG_VAL} arm so the
                    # ablation and main sweep never skip each other's rows.
                    if [[ -f "${RESULTS_CSV}" ]] && grep -qP "^${MODEL},${MODALITY},fft,${LR},[^,]+,[^,]+,[^,]+,${LAMBDA_U},${LAMBDA_E},${EMA_M},${TEMPERATURE},[^,]+,[^,]+,${NO_STRONG_VAL},[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,${DINO_VAL},[^,]+$" "${RESULTS_CSV}" 2>/dev/null; then
                        echo "  → already in results, skipping"
                        continue
                    fi

                    python -u baseline/baseline_freematch.py \
                        --dataset "$DATASET" \
                        --modality "$MODALITY" \
                        --train_mode fft \
                        --epochs 20 \
                        --lr "$LR" \
                        --lambda_u "$LAMBDA_U" \
                        --lambda_e "$LAMBDA_E" \
                        --ema_momentum "$EMA_M" \
                        --temperature "$TEMPERATURE" \
                        --num_workers 8 \
                        --results_csv "$RESULTS_CSV" \
                        --batch_size 32 \
                        --model "$MODEL" \
                        ${DINO_FLAG} ${STRONG_AUG_FLAG}
                done
            done
        done
    done
done
