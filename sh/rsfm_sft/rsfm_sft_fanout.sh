#!/bin/bash
# One SLURM job per (model, dataset, modality, lr, wd) -- i.e. ONE training run
# per job, instead of rsfm_sft_all.sh's one job that runs all 9 lr/wd combos
# sequentially.
#
# WHY: the sequential form wastes a lot of wall-clock. olmoearth-large on benv2
# took ~68 min/combo, so 9 combos needed ~10 h against a 5:59 walltime and job
# 5605171 died at combo 5 with 4 combos unrun. One config per job means:
#   - a timeout costs ONE config, not the tail of a sweep
#   - short jobs backfill far sooner than a 6 h ask
#   - the 9 configs run in parallel rather than end to end
#
# rsfm_sft.py appends to the results CSV per run and rsfm_sft_all.sh's skip
# logic is per (model,dataset,modality,lr,wd), so re-running this is safe and
# resumes: combos already in the CSV are skipped by the job script itself.
#
# Usage:
#   bash sh/rsfm_sft/rsfm_sft_fanout.sh                       # dry run
#   SUBMIT=1 bash sh/rsfm_sft/rsfm_sft_fanout.sh
#   MODELS=olmoearth-base MODALITIES='s2_rgb+s1' SUBMIT=1 bash ...
#
# NOTE olmoearth-large is deliberately NOT in the default MODELS: the paper
# tables report panopticon and olmoearth-base only.

set -u
SUBMIT="${SUBMIT:-0}"
MODELS="${MODELS:-panopticon olmoearth-base}"
DATASETS="${DATASETS:-dfc2020 benv2}"
MODALITIES="${MODALITIES:-s2_rgb+s1 s2_rgb+s2_norgb s2s1}"
TRAIN_MODE="${TRAIN_MODE:-fft}"
LRS="${LRS:-0.001 0.0005 0.0001}"
WDS="${WDS:-0.01 0.0001 0}"
# One 20-epoch run. Measured: panopticon/olmo-base ~20-27 min, olmoearth-large
# ~68 min. 2 h covers all of them with margin and backfills quickly.
WALLTIME="${WALLTIME:-2:00:00}"
GPU="${GPU:-l40s}"
# Forwarded to the job script. 32 matches every other oracle row; olmoearth
# combos that include an s1 part need 16 on dfc2020 -- adding the sentinel1
# field gives the encoder a 4th modality group at full 256x256 resolution and
# batch 32 then exceeds the L40S's 44 G (verified: inputs are correct, the
# blow-up is encoder-side).
BATCH_SIZE="${BATCH_SIZE:-32}"
RESULTS_CSV="${RESULTS_CSV:-res/rsfm/rsfm_results.csv}"

n=0; skipped=0
for MODEL in ${MODELS}; do
for DATASET in ${DATASETS}; do
for MODALITY in ${MODALITIES}; do
for LR in ${LRS}; do
for WD in ${WDS}; do
    # Skip combos already recorded, so a resubmit only fills the gaps.
    if python3 - "$RESULTS_CSV" "$MODEL" "$DATASET" "$MODALITY" "$LR" "$WD" <<'PY'
import csv, os, sys
csv_path, model, ds, mod, lr, wd = sys.argv[1:7]
if not os.path.isfile(csv_path):
    sys.exit(1)
def close(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-12
    except (TypeError, ValueError):
        return str(a) == str(b)
with open(csv_path) as f:
    for r in csv.DictReader(f):
        if (r.get('model', '').lower() == model.lower()
                and r.get('dataset') == ds
                and r.get('modality') == mod
                and close(r.get('lr'), lr)
                and close(r.get('weight_decay'), wd)):
            sys.exit(0)
sys.exit(1)
PY
    then
        skipped=$((skipped+1)); continue
    fi

    TAG="rsfm_${MODEL}_${DATASET}_${MODALITY}_lr${LR}_wd${WD}"
    if [ "$SUBMIT" = "1" ]; then
        sbatch --time="${WALLTIME}" --gres="gpu:${GPU}:1" --job-name="${TAG}" \
            --export=ALL,MODEL="${MODEL}",DATASET="${DATASET}",TRAIN_MODE="${TRAIN_MODE}",MODALITY="${MODALITY}",LRS="${LR}",WDS="${WD}",BATCH_SIZE="${BATCH_SIZE}" \
            sh/rsfm_sft/rsfm_sft_job.sh >/dev/null
    fi
    echo "  [$((++n))] ${TAG}"
done; done; done; done; done

echo
echo "total: ${n} jobs submitted, ${skipped} already in ${RESULTS_CSV} (SUBMIT=${SUBMIT}, gpu=${GPU})"
