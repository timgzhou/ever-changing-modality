#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/mixmatch/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

source sh/env.sh
export TQDM_DISABLE=1

# DATASET and MODALITY are passed via environment variables.
# Launch one job per modality:
#
# declare -A MODALITY_CONFIGS
# MODALITY_CONFIGS['eurosat']='rgb vre nir swir'
# MODALITY_CONFIGS['benv2']='s2 s1 s2_rgb'
# MODALITY_CONFIGS['dfc2020']='s2 s1 s2_rgb'
# for DATASET in eurosat benv2 dfc2020; do
#   for MODALITY in ${MODALITY_CONFIGS[$DATASET]}; do
#     export DATASET MODALITY; sbatch sh/mixmatch/mixmatch_all.sh
#   done
# done

LRS=('0.0003')
ALPHAS=('0.5')
LAMBDA_US=('50' '75')
DINO_FLAGS=('--use_dino_weights' '')
MODELS=('evan_base' 'evan_large')

RESULTS_CSV="res/baselines/mixmatch/baseline_mixmatch_${DATASET}.csv"
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
                    echo "Running: dataset=${DATASET} modality=${MODALITY} lr=${LR} alpha=${ALPHA} lambda_u=${LAMBDA_U} dino=${DINO_VAL}"

                    if [[ -f "${RESULTS_CSV}" ]] && grep -qP "^[^,]+,${MODALITY},fft,${LR},[^,]+,[^,]+,[^,]+,2,0\.5,${ALPHA},${LAMBDA_U},[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,${DINO_VAL}$" "${RESULTS_CSV}" 2>/dev/null; then
                        echo "  → already in results, skipping"
                        continue
                    fi

                    python -u baseline/baseline_mixmatch.py \
                        --dataset ${DATASET} \
                        --modality ${MODALITY} \
                        --train_mode fft \
                        --epochs 20 \
                        --lr ${LR} \
                        --K 2 \
                        --temperature 0.5 \
                        --alpha ${ALPHA} \
                        --lambda_u ${LAMBDA_U} \
                        --num_workers 8 \
                        --results_csv ${RESULTS_CSV} \
                        --batch_size 32 \
                        --model ${MODEL} \
                        ${DINO_FLAG}
                done
            done
        done
    done
done
