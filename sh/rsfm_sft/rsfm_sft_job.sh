#!/bin/bash
#SBATCH --time=2:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/rsfm_sft/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
# l40s, not h100: L40S nodes are far less contended here, so these queue and
# start much sooner. BATCH_SIZE stays 32: every dfc2020/benv2 oracle row in
# res/rsfm/rsfm_results.csv was run at 32, and mixing batch sizes within the
# oracle column would change BN statistics and gradient noise independently of
# the lr the 9-combo sweep selects. These are non-temporal probes, so 32 fits
# in the L40S's 46G -- the small-batch note applies only to biomassters, where
# T=12 folds into the batch.
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
# kn101 has failing GPU memory: every job scheduled there since 2026-09-15 died
# with "CUDA error: uncorrectable ECC error encountered", while jobs on every
# other node succeeded. Slurm still lists it as healthy. Remove if repaired.
#
# kn159 added 2026-09-19: 50/50 jobs landing there died in 1-3s with exit code
# 53 and no output file, while concurrent jobs on 13 other nodes ran normally.
#SBATCH --exclude=kn101,kn159
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
        # BATCH_SIZE*12. The rsfm default of 32 would be 384 and OOMs.
        #
        # Use 8, NOT 4. The heads start with BatchNorm2d, which raises
        # "Expected more than 1 value per channel when training" on a batch of 1,
        # and biomassters train1=2005 / val1=869 both leave a remainder of 1 at
        # bs=4 (2005 = 501*4 + 1). The loaders set no drop_last, so that final
        # batch is always delivered and the job dies at the END of epoch 1 --
        # after the wall-clock cost of a full epoch, and invisible to a smoke
        # test that stops early. 2005 % 8 = 5 and 869 % 8 = 5, so 8 is safe.
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
