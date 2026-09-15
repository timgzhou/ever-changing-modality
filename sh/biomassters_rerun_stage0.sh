#!/bin/bash
# Re-train the biomassters s2_rgb / s2_norgb stage-0 teachers.
#
# WHY: their only existing runs used lr 1e-4, and across all 41 biomassters
# stage-0 runs lr 1e-4 never beat val RMSE 71.35 (median 76.41) while lr 5e-4
# reaches 42.49. The current s2_rgb / s2_norgb teachers sit at 76.10 / 77.99 --
# undertrained by learning rate, not by epochs. Both lrs are swept here so the
# cause is confirmed rather than assumed, at 48 epochs (double the previous 24).
#
# WALLTIME: biomassters stage-0 runs at ~14.4 min/epoch, so 48 epochs is ~11.5h
# for ONE init arm. The job script runs dino and random-init arms sequentially by
# default, which would be ~23h; DINO_ARMS=1 keeps one arm per job. The earlier
# 24-epoch jobs died at the 6h wall for exactly this reason (2 arms = 11.5h).
#
# Usage:  bash sh/biomassters_rerun_stage0.sh          # submit
#         DRYRUN=1 bash sh/biomassters_rerun_stage0.sh
set -uo pipefail
EPOCHS="${EPOCHS:-48}"
MODS="${MODS:-s2_rgb s2_norgb}"
LRS="${LRS:-0.0005 0.0001}"
WD="${WD:-0.01}"
mkdir -p logs/train_sft res/train_sft
ids=()
for MOD in ${MODS}; do
  for LR in ${LRS}; do
    EX="ALL,DATASET=biomassters,MODEL=evan_base,TRAIN_MODE=fft,MODALITY_ENTRY=${MOD}"
    EX="${EX},LR=${LR},WD=${WD},TRAIN_SPLIT=split1,TRAIN_AUG=none"
    EX="${EX},EPOCHS=${EPOCHS},DINO_ARMS=1,NUM_TIME_STEPS=12"
    if [ "${DRYRUN:-0}" = "1" ]; then
        echo "  [dry] ${MOD} lr=${LR} epochs=${EPOCHS} (dino arm only, --time=23:59)"
    else
        jid=$(sbatch --parsable --time=23:59:00 --export="${EX}" sh/train_sft/train_sft_job.sh)
        ids+=("${jid}"); echo "  submitted ${jid}  ${MOD} lr=${LR} epochs=${EPOCHS}"
    fi
  done
done
[ "${DRYRUN:-0}" = "1" ] && { echo "total: $(echo ${MODS} | wc -w) x $(echo ${LRS} | wc -w) jobs"; exit 0; }
printf '%s\n' "${ids[@]}" > logs/train_sft/biomassters_rerun_ids.txt
echo "stage-0 job ids -> logs/train_sft/biomassters_rerun_ids.txt"
echo
echo "Chain stage 1 after these finish:"
echo "  sbatch --dependency=afterany:$(IFS=:; echo "${ids[*]}") sh/biomassters_rerun_stage1.sh"
