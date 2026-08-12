#!/bin/bash
#SBATCH --time=11:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/shot_ete_best/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Run DeluluNet on BioMassters with the best sweep hyperparameters and SAVE the
# final checkpoint so predictions can be visualized later.
#
# Hyperparameters come from the top run by val_peeking in
#   res/delulu-sweep/sweep_results_biomassters_{s1s2,s2s1}.csv
# val_peeking is used for selection because it is the only criterion whose val
# and test values are the same metric (negated RMSE); val_transfer/val_addition
# are composite peek x agreement scores on a different scale.
#
# Usage:  sbatch sh/shot_ete_biomassters_best.sh              # s1 -> s2 (default)
#         DIRECTION=s2s1 sbatch sh/shot_ete_biomassters_best.sh
#         EPOCHS=64 BATCH_SIZE=16 sbatch sh/shot_ete_biomassters_best.sh

source sh/env.sh
export TQDM_DISABLE=1
# T=12 folds the time axis into the batch dim (effective batch = batch_size * T),
# so SHOT (student + frozen teacher + decoders + projectors) is memory-heavy.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/shot_ete_best checkpoints

DIRECTION="${DIRECTION:-s1s2}"

# ---- per-direction best config (by val_peeking) ---------------------------
case "${DIRECTION}" in
  s1s2)
    # wandb run 17albymd | val_peeking -45.81  test_peeking -46.94
    NEW_MOD="s2"
    TEACHER="checkpoints/sft_evan_base_biomassters_s1_fft_lr0.0005_20260725_075947.pt"
    LR="0.0001569391767106977"
    WEIGHT_DECAY="3.351617951860976e-05"
    MODALITY_DROPOUT_STARTMOD="0.33189226742900324"
    MODALITY_DROPOUT_NEWMOD="0.17068517311514753"
    LABELED_FREQUENCY="0.23002477810989655"
    LAMBDA_LATENT="0.3613664751387723"
    LAMBDA_PREFUSION="0.6430194633931678"
    LAMBDA_DISTILL="0.15374988356364516"
    TOKEN_MASK_RATIO="0.40414477259411485"
    ;;
  s2s1)
    # wandb run d3k33298 | val_peeking -42.42  test_peeking -43.40
    NEW_MOD="s1"
    TEACHER="checkpoints/sft_evan_base_biomassters_s2_fft_lr0.0005_20260725_075836.pt"
    LR="9.928740856000608e-05"
    WEIGHT_DECAY="1.0712967039939078e-05"
    MODALITY_DROPOUT_STARTMOD="0.20965117152957152"
    MODALITY_DROPOUT_NEWMOD="0.11196402369777392"
    LABELED_FREQUENCY="0.4577756907743401"
    LAMBDA_LATENT="0.4035482513964165"
    LAMBDA_PREFUSION="0.2187001470110117"
    LAMBDA_DISTILL="0.9862176806762428"
    TOKEN_MASK_RATIO="0.4051096150678868"
    ;;
  *)
    echo "[error] DIRECTION must be s1s2 or s2s1 (got '${DIRECTION}')"
    exit 1
    ;;
esac

# Shared across both best runs.
MODALITY_DROPOUT="0.3"
LABELED_START_FRACTION="0"
PROTECT_LRM="0.0"
NUM_TIME_STEPS="${NUM_TIME_STEPS:-12}"   # MUST match the teacher's T
EPOCHS="${EPOCHS:-64}"
BATCH_SIZE="${BATCH_SIZE:-16}"

if [ ! -f "${TEACHER}" ]; then
    echo "[error] teacher checkpoint not found: ${TEACHER}"
    exit 1
fi

echo "=== shot_ete BEST params | biomassters ${DIRECTION} (new_mod=${NEW_MOD}) | T=${NUM_TIME_STEPS} ==="
echo "    teacher: ${TEACHER}"
echo "    lr=${LR}  wd=${WEIGHT_DECAY}  epochs=${EPOCHS}  bs=${BATCH_SIZE}"

python -u shot_ete.py \
    --dataset biomassters \
    --new_mod_group "${NEW_MOD}" \
    --stage0_checkpoint "${TEACHER}" \
    --active_losses latent prefusion distill ce \
    --lr "${LR}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --modality_dropout "${MODALITY_DROPOUT}" \
    --modality_dropout_startmod "${MODALITY_DROPOUT_STARTMOD}" \
    --modality_dropout_newmod "${MODALITY_DROPOUT_NEWMOD}" \
    --labeled_frequency "${LABELED_FREQUENCY}" \
    --labeled_start_fraction "${LABELED_START_FRACTION}" \
    --lambda_latent "${LAMBDA_LATENT}" \
    --lambda_prefusion "${LAMBDA_PREFUSION}" \
    --lambda_distill "${LAMBDA_DISTILL}" \
    --token_mask_ratio "${TOKEN_MASK_RATIO}" \
    --protect_lrm "${PROTECT_LRM}" \
    --latent_masked_only \
    --unprotect_starting_mod \
    --num_time_steps "${NUM_TIME_STEPS}" \
    --epochs "${EPOCHS}" \
    --batch_size "${BATCH_SIZE}" \
    --num_workers 4 \
    --save_checkpoint \
    --checkpoint_dir checkpoints \
    --results_csv "res/delulu/biomassters_best_${DIRECTION}.csv" \
    --wandb_project "delulu-biomassters-best"
