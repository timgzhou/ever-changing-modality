#!/bin/bash
#SBATCH --time=14:59:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/shot_ete_best/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# One array task = one (config, seed) pair for DeluluNet on BioMassters.
#
# The 6 configs are the top-2 runs by each of val_transfer / val_peeking /
# val_addition, read directly out of the sweep CSV (not sweep_best.json, which
# gets overwritten per-DATASET by the analysis script). Each is run 3x with
# seeds 0/1/2 -> 18 array tasks per direction.
#
# Usage:
#   sbatch --array=0-17 sh/shot_ete_biomassters_best_job.sh              # s1 -> s2
#   DIRECTION=s2s1 sbatch --array=0-17 sh/shot_ete_biomassters_best_job.sh
#
# Index layout: TASK = config_idx * 3 + seed
#   config_idx 0..5 = transfer#1 transfer#2 peeking#1 peeking#2 addition#1 addition#2

source sh/env.sh
export TQDM_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs/shot_ete_best checkpoints res/delulu

DIRECTION="${DIRECTION:-s1s2}"
SEEDS_PER_CONFIG="${SEEDS_PER_CONFIG:-3}"

case "${DIRECTION}" in
  s1s2) SWEEP_CSV="res/delulu-sweep/sweep_results_biomassters_s1s2.csv"; NEW_MOD="s2" ;;
  s2s1) SWEEP_CSV="res/delulu-sweep/sweep_results_biomassters_s2s1.csv"; NEW_MOD="s1" ;;
  *) echo "[error] DIRECTION must be s1s2 or s2s1 (got '${DIRECTION}')"; exit 1 ;;
esac

TASK="${SLURM_ARRAY_TASK_ID:-0}"
CONFIG_IDX=$(( TASK / SEEDS_PER_CONFIG ))
SEED=$(( TASK % SEEDS_PER_CONFIG ))

# Pull the CONFIG_IDX-th config (top-2 by each criterion) out of the sweep CSV.
# Emits shell assignments consumed by eval below.
CONFIG_ENV=$(python3 - "$SWEEP_CSV" "$CONFIG_IDX" <<'PYEOF'
import csv, sys

csv_path, idx = sys.argv[1], int(sys.argv[2])
rows = list(csv.DictReader(open(csv_path)))

# 6 slots: top-2 by each criterion, in a fixed order.
slots = []
for crit in ("transfer", "peeking", "addition"):
    col = f"val_{crit}"
    ranked = sorted(rows, key=lambda r: float(r[col]), reverse=True)[:2]
    for rank, row in enumerate(ranked, start=1):
        slots.append((f"{crit}_rank{rank}", row))

if not 0 <= idx < len(slots):
    sys.exit(f"config index {idx} out of range (have {len(slots)} configs)")

label, r = slots[idx]
out = {
    "CONFIG_LABEL": label,
    "SRC_RUN": r["wandb_run_id"],
    "TEACHER": r["stage0_checkpoint"],
    "LR": r["lr"],
    "WEIGHT_DECAY": r["weight_decay"],
    "MODALITY_DROPOUT": r["modality_dropout"],
    "MODALITY_DROPOUT_STARTMOD": r["modality_dropout_startmod"],
    "MODALITY_DROPOUT_NEWMOD": r["modality_dropout_newmod"],
    "LABELED_FREQUENCY": r["labeled_frequency"],
    "LABELED_START_FRACTION": r["labeled_start_fraction"],
    "LAMBDA_LATENT": r["lambda_latent"],
    "LAMBDA_PREFUSION": r["lambda_prefusion"],
    "LAMBDA_DISTILL": r["lambda_distill"],
    "TOKEN_MASK_RATIO": r["mae_mask_ratio"],
    "PROTECT_LRM": r["protect_lrm"],
}
for k, v in out.items():
    print(f"{k}='{v}'")
PYEOF
)
if [ $? -ne 0 ] || [ -z "$CONFIG_ENV" ]; then
    echo "[error] failed to extract config ${CONFIG_IDX} from ${SWEEP_CSV}"
    exit 1
fi
eval "$CONFIG_ENV"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_TIME_STEPS="${NUM_TIME_STEPS:-12}"   # MUST match the teacher's T

if [ ! -f "${TEACHER}" ]; then
    echo "[error] teacher checkpoint not found: ${TEACHER}"
    exit 1
fi

RUN_TAG="${DIRECTION}_${CONFIG_LABEL}_seed${SEED}"

echo "=== biomassters ${DIRECTION} | ${CONFIG_LABEL} (from sweep run ${SRC_RUN}) | seed ${SEED} ==="
echo "    array task ${TASK} -> config ${CONFIG_IDX}, seed ${SEED}"
echo "    teacher: ${TEACHER}"
echo "    lr=${LR} wd=${WEIGHT_DECAY} epochs=${EPOCHS} bs=${BATCH_SIZE}"

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
    --seed "${SEED}" \
    --config_label "${CONFIG_LABEL}" \
    --save_checkpoint \
    --checkpoint_dir checkpoints \
    --checkpoint_name "delulunet_biomassters_${RUN_TAG}" \
    --results_csv "res/delulu/biomassters_best_${DIRECTION}.csv" \
    --wandb_project "delulu-biomassters-best"
