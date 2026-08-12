#!/bin/bash
# Launcher: submits one SLURM job per dataset+model+modality+lr+wd combo.
# Usage: bash sh/train_sft/train_sft_all.sh

declare -A MODALITY_CONFIGS
# Single modalities (stage 0 oracle)
# MODALITY_CONFIGS['eurosat']='s2 rgb vre nir swir'
# MODALITY_CONFIGS['benv2']='s2 s1 s2_rgb s2_norgb'
# MODALITY_CONFIGS['dfc2020']='s2 s1 s2_rgb s2_norgb'

# Combined modalities: use '+' separator — job script splits into --modalities args
# (oracle upper-bound for Addition table: DINO-init SFT on start+new combined)
# MODALITY_CONFIGS['eurosat']='rgb+nir rgb+vre rgb+swir swir+nir swir+rgb swir+vre vre+nir vre+rgb'
# MODALITY_CONFIGS['benv2']='s2_norgb s2_rgb+s1 s2_rgb+s2_norgb s1+s2 s2+s1'
# MODALITY_CONFIGS['dfc2020']='s2_norgb s2_rgb+s1 s2_rgb+s2_norgb s1+s2 s2+s1'

MODALITY_CONFIGS['eurosat']='vre+nir'
# split1-vs-full data sweep on reBEN + DFC2020: the s2_rgb single-modality
# baseline and the s2_rgb+s1 two-modality addition upper bound.
# Other entries kept here (commented) so they are not lost:
#   MODALITY_CONFIGS['benv2']='s2_norgb s2 s2_rgb+s1 s2_rgb+s2_norgb'
#   MODALITY_CONFIGS['dfc2020']='s2_norgb s2_rgb+s1 s2_rgb+s2_norgb'
MODALITY_CONFIGS['benv2']='s2_rgb s2_rgb+s1'
MODALITY_CONFIGS['dfc2020']='s2_rgb s2_rgb+s1'
# BioMassters (temporal S1/S2 AGB regression). Stage-0 oracles: single modality.
# Combined (s2+s1 / s1+s2) is the addition upper bound. Temporal steps via --num_time_steps.
MODALITY_CONFIGS['biomassters']='s2 s1 s2+s1'


DATASETS=('dfc2020')
MODELS=('evan_base')
TRAIN_MODES=('fft')
LRS=('0.0005' '0.0001')
WDS=('0.01' '0.0')
# Labeled training data: 'split1' = train1 half only (all historical runs),
# 'full' = train1+train2 concatenated (same-distribution 2x-data upper bound).
# Validation stays on val1 in both arms, so the rows stay comparable.
TRAIN_SPLITS=('split1' 'full')
# Train-time input augmentation (classification/multilabel only): none|weak|strong.
# Uses the same pipelines as baseline_freematch.py. Override from the shell, e.g.
#   TRAIN_AUG=weak bash sh/train_sft/train_sft_all.sh
# The results-CSV key includes train_aug, so arms never suppress each other.
TRAIN_AUG="${TRAIN_AUG:-none}"

# ---------------------------------------------------------------------------
# In-flight guard.
#
# The results-CSV dedup below only sees *finished* runs. A job that is queued or
# still training has written no row yet, so re-running this launcher would
# happily submit a second copy of it. Each job echoes its config as
#   "Running: model=... dataset=... modalities=... lr=... wd=... train_split=..."
# into logs/train_sft/<jobid>.out, so we collect those lines for the user's
# currently queued/running jobs and skip any combo that matches.
#
# Set SKIP_INFLIGHT_CHECK=1 to bypass (e.g. to deliberately re-run a combo).
# ---------------------------------------------------------------------------
INFLIGHT=""
if [ -z "${SKIP_INFLIGHT_CHECK}" ] && command -v squeue >/dev/null 2>&1; then
    for jid in $(squeue -u "$USER" -h -o "%i" 2>/dev/null); do
        line=$(grep -m1 "^Running:" "logs/train_sft/${jid}.out" 2>/dev/null)
        [ -n "$line" ] && INFLIGHT="${INFLIGHT}${line}"$'\n'
    done
    n_inflight=$(printf '%s' "$INFLIGHT" | grep -c . || true)
    echo "In-flight guard: found ${n_inflight} queued/running train_sft job(s)."
fi

for DATASET in "${DATASETS[@]}"; do
    for MODEL in "${MODELS[@]}"; do
        for TRAIN_MODE in "${TRAIN_MODES[@]}"; do
            for MODALITY_ENTRY in ${MODALITY_CONFIGS[$DATASET]}; do
                for LR in "${LRS[@]}"; do
                    for WD in "${WDS[@]}"; do
                      for TRAIN_SPLIT in "${TRAIN_SPLITS[@]}"; do
                        MODALITY_KEY="${MODALITY_ENTRY}"
                        RESULTS_CSV="res/train_sft/${DATASET}.csv"
                        # dino_init is followed by num_time_steps,decoder,train_aug,train_split.
                        # Anchor on train_aug AND train_split so the augmented/unaugmented
                        # and split1/full arms are tracked separately. train_split is the
                        # critical one: a 'full' run is identical to its 'split1'
                        # counterpart in every other column, so without it the split1 row
                        # would always suppress the full run.
                        # Every row carries all four trailing columns since the migration
                        # (res/train_sft/*.csv were normalized and backfilled with
                        # train_split=split1), so the tail is required, not optional.
                        # \r? tolerates legacy CRLF line endings.
                        DINO_PAT="^${DATASET},${MODEL},${MODALITY_KEY},${TRAIN_MODE},[^,]+,[^,]+,${LR},${WD},[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+,[^,]+"
                        DINO_TAIL=",[^,]*,[^,]*,${TRAIN_AUG},${TRAIN_SPLIT}\r?$"
                        grep -qP "${DINO_PAT},True${DINO_TAIL}" "${RESULTS_CSV}" 2>/dev/null && DINO_TRUE=1 || DINO_TRUE=0
                        grep -qP "${DINO_PAT},False${DINO_TAIL}" "${RESULTS_CSV}" 2>/dev/null && DINO_FALSE=1 || DINO_FALSE=0
                        if [ "$DINO_TRUE" -ge 1 ] && [ "$DINO_FALSE" -ge 1 ]; then
                            echo "Skipping (both dino variants done): dataset=${DATASET} model=${MODEL} train_mode=${TRAIN_MODE} modality=${MODALITY_ENTRY} lr=${LR} wd=${WD} train_split=${TRAIN_SPLIT}"
                            continue
                        fi
                        # Skip combos already queued/running (see in-flight guard above).
                        # MODALITIES in the job log is '+'-separated entries split on
                        # spaces, so compare against that form.
                        INFLIGHT_MODS="${MODALITY_ENTRY//+/ }"
                        if [ -n "${INFLIGHT}" ] && printf '%s' "${INFLIGHT}" | grep -qF \
                            "model=${MODEL} dataset=${DATASET} train_mode=${TRAIN_MODE} modalities=${INFLIGHT_MODS} lr=${LR} wd=${WD} train_split=${TRAIN_SPLIT}"; then
                            echo "Skipping (already queued/running): dataset=${DATASET} model=${MODEL} train_mode=${TRAIN_MODE} modality=${MODALITY_ENTRY} lr=${LR} wd=${WD} train_split=${TRAIN_SPLIT}"
                            continue
                        fi
                        echo "Submitting: dataset=${DATASET} model=${MODEL} train_mode=${TRAIN_MODE} modality=${MODALITY_ENTRY} lr=${LR} wd=${WD} train_aug=${TRAIN_AUG} train_split=${TRAIN_SPLIT}"
                        sbatch --export=ALL,DATASET="${DATASET}",MODEL="${MODEL}",TRAIN_MODE="${TRAIN_MODE}",MODALITY_ENTRY="${MODALITY_ENTRY}",LR="${LR}",WD="${WD}",TRAIN_AUG="${TRAIN_AUG}",TRAIN_SPLIT="${TRAIN_SPLIT}" \
                            sh/train_sft/train_sft_job.sh
                      done
                    done
                done
            done
        done
    done
done


# bash sh/train_sft/train_sft_all.sh
