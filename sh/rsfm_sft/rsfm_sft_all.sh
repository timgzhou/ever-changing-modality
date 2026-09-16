#!/bin/bash
# Launcher: one SLURM job per model+dataset+modality combo. Each job sweeps 9
# lr/wd combos internally and SKIPS combos already present in
# res/rsfm/rsfm_results.csv, so re-running is safe and resumes.
#
# Usage:
#   bash sh/rsfm_sft/rsfm_sft_all.sh            # dry run, prints the plan
#   SUBMIT=1 bash sh/rsfm_sft/rsfm_sft_all.sh
#
# WALLTIME: 9 sweep combos x 20 epochs in one job. The #SBATCH default is 2h;
# WALLTIME here overrides it, and a job killed at the wall loses only the combos
# it had not yet written (the CSV is appended per combo, and the skip-check
# above makes a resubmit resume).
#
# The dfc2020 oracle rows were re-run on 2026-09-15: the previous ones predated
# the Copernicus-Bench split and were quarantined to res/ignore/. rsfm_sft.py
# loads train1/val1, i.e. split1, so these are the correct non-leaky oracles.
set -u
SUBMIT="${SUBMIT:-0}"
WALLTIME="${WALLTIME:-5:59:00}"

declare -A MODALITY_CONFIGS
# The modalities the paper tables need: the four single modalities for the
# Transfer/Peek oracle columns, plus the combined ones for Addition.
# s2s1 covers the s1<->s2 Addition rows; s2_rgb+s1 and s2_rgb+s2_norgb cover the
# two S2-RGB Addition rows. COMBINED_RSFM_ALIASES in results_BL.py maps them.
MODALITY_CONFIGS['dfc2020']="${MODALITIES:-s1 s2 s2_rgb s2_norgb s2s1 s2_rgb+s1 s2_rgb+s2_norgb}"
# biomassters is absent from rsfm_sft.py's MODALITY_CONFIGS, and neither wrapper
# can take its data as-is (checked 2026-09-15):
#   biomassters loaders yield [B, C, T=12, H, W]; both wrappers take [B, C, H, W].
#   PanopticonWrapper  - upstream is a DINOv2 ViT over (imgs, chn_ids). No time
#                        axis in the architecture at all. Not temporal.
#   OlmoEarthWrapper   - the MODEL is temporal (its tensors are [B,H,W,T,C]), but
#                        the wrapper hardcodes T=1 via .unsqueeze(3) and passes a
#                        dummy [B,1,3] timestamp. The capability is there, unused.
# So OlmoEarth could be extended to real T=12 (plus genuine timestamps), while
# Panopticon would have to mean-pool over T the way EVAN does
# (forward_modality_specific_features). Both are code changes, not launcher ones.
MODALITY_CONFIGS['biomassters']="${MODALITIES:-}"
MODALITY_CONFIGS['benv2']="${MODALITIES:-s2_norgb}"
MODALITY_CONFIGS['eurosat']="${MODALITIES:-rgb+nir rgb+vre rgb+swir}"

MODELS="${MODELS:-panopticon olmoearth-base olmoearth-large}"
DATASETS="${DATASETS:-dfc2020}"
TRAIN_MODES="${TRAIN_MODES:-fft}"

n=0
for MODEL in ${MODELS}; do
    for DATASET in ${DATASETS}; do
        for TRAIN_MODE in ${TRAIN_MODES}; do
            for MODALITY in ${MODALITY_CONFIGS[$DATASET]}; do
                n=$((n+1))
                if [ "${SUBMIT}" = "1" ]; then
                    sbatch --time="${WALLTIME}" \
                        --export=ALL,MODEL="${MODEL}",DATASET="${DATASET}",TRAIN_MODE="${TRAIN_MODE}",MODALITY="${MODALITY}" \
                        sh/rsfm_sft/rsfm_sft_job.sh >/dev/null
                    echo "  [$n] submitted ${MODEL} ${DATASET} ${MODALITY}"
                else
                    echo "  [$n] ${MODEL} ${DATASET} ${TRAIN_MODE} ${MODALITY}"
                fi
            done
        done
    done
done
echo
echo "total: ${n} jobs, --time=${WALLTIME} (SUBMIT=${SUBMIT})"
