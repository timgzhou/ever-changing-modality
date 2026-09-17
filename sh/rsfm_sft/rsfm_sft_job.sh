#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/rsfm_sft/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101 has failing GPU memory: every job scheduled there since 2026-09-15 died
# with "CUDA error: uncorrectable ECC error encountered", while jobs on every
# other node succeeded. Slurm still lists it as healthy. Remove if repaired.
#SBATCH --exclude=kn101
# Expected env vars (set by rsfm_sft_all.sh):
#   MODEL, DATASET, TRAIN_MODE, MODALITY

source sh/env.sh
export TQDM_DISABLE=1

RESULTS_CSV="res/rsfm/rsfm_results.csv"
# Default 3x3 grid. BIOMASSTERS MUST OVERRIDE THIS: it is temporal (T=12) and
# larger (train1 2005 vs dfc2020's 1578), so a forward pass costs ~15x more. The
# dfc2020 jobs run the full 9-combo x 20-epoch sweep in 1:12-1:33; the same
# sweep on biomassters projects to ~20 h, past even the 11:59 partition limit,
# and a job killed at the wall writes nothing. Pass LRS/WDS to shrink the grid.
#
# Cut the GRID, not the epochs: on biomassters lr dominates (the 48-epoch SFT
# singles reach 41.9 RMSE at 5e-4 but 62-66 at 1e-4), and 24 epochs was visibly
# under-converged there, so epochs are the wrong thing to spend.
LRS=(${LRS:-'0.001' '0.0005' '0.0001'})
WDS=(${WDS:-'0.01' '0.0001' '0'})

for LR in "${LRS[@]}"; do
    for WD in "${WDS[@]}"; do
        echo "Running: model=${MODEL} dataset=${DATASET} train_mode=${TRAIN_MODE} modality=${MODALITY} lr=${LR} wd=${WD}"
        if grep -qP "^${MODEL},${DATASET},${MODALITY},${TRAIN_MODE},[^,]+,${LR},${WD}," "${RESULTS_CSV}" 2>/dev/null; then
            echo "  → already in results, skipping"
            continue
        fi
        # BATCH_SIZE: biomassters is temporal (T=12) and rsfm_sft.py folds T into
        # the batch before the frozen backbone, so the effective backbone batch is
        # BATCH_SIZE*12. The rsfm default of 32 would be 384 and OOMs; 4 keeps it
        # at 48. Matches sh/baselines_biomassters.sh, which documents 8 as the
        # safe point for a model that folds T the same way.
        python -u rsfm_sft.py \
            --model ${MODEL} \
            --dataset ${DATASET} \
            --modality ${MODALITY} \
            --train_mode ${TRAIN_MODE} \
            --epochs ${EPOCHS:-20} \
            --batch_size ${BATCH_SIZE:-32} \
            --lr ${LR} \
            --weight_decay ${WD}
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "  → python exited with code ${EXIT_CODE}, stopping remaining configs for this job"
            exit $EXIT_CODE
        fi
    done
done
