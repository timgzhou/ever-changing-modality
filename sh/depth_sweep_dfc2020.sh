#!/bin/bash
# Does a deeper cross-modal (masking) projector help?
#
# CrossSequenceProjector is (num_layers - 1) self-attention blocks over the
# source tokens followed by 1 cross-attention block that reads the target
# modality's learned queries. So:
#   depth 1 = cross only              14.2 M params/projector
#   depth 2 = 1 self + 1 cross        28.4 M   <- current default
#   depth 3 = 2 self + 1 cross        42.5 M
#   depth 4 = 3 self + 1 cross        56.7 M
#
# Design: 3 configs x 4 depths x 2 seeds = 24 jobs, on s1->+s2_norgb (the
# direction the DFC2020 configs were swept on). Two seeds because seed-to-seed
# sd on DFC2020 transfer is ~1.09 mIoU -- a single run per depth could not
# separate a depth effect from luck.
#
# The hyperparameters were tuned at depth 2, which biases against depths 1/3/4.
# That is accepted deliberately: if extra depth genuinely helps it should show
# through even at a config it was not tuned for.
#
# Depth is NOT a results-CSV column, so it is encoded in config_label as
# "<selector>_d<depth>" -- e.g. transfer_d3.
#
# Usage:  bash sh/depth_sweep_dfc2020.sh
#         DRYRUN=1 bash sh/depth_sweep_dfc2020.sh
set -uo pipefail

CONFIG="configs/delulu_best_dfc2020.yaml"
START="${START:-s1}"; NEW="${NEW:-s2_norgb}"
DEPTHS="${DEPTHS:-1 2 3 4}"
SELECTORS="${SELECTORS:-transfer peeking addition}"
SEEDS="${SEEDS:-0 1}"
RESULTS_CSV="res/delulu/dfc2020_projector_depth.csv"
LEDGER="logs/train_delulu/submitted_depth.tsv"
mkdir -p logs/train_delulu res/delulu; touch "${LEDGER}"

TEACHER=$(jq -r ".\"dfc2020_cobench/${START}/evan_base/upernet/split1\".checkpoint // empty" artifacts/sft_teachers.json)
if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
    echo "[error] no teacher for ${START}"; exit 1
fi

ACTIVE=" $(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '\n' ' ') "
INFLIGHT=""
while read -r jid rest; do
    case "${ACTIVE}" in *" ${jid} "*) INFLIGHT="${INFLIGHT}${rest}"$'\n' ;; esac
done < "${LEDGER}"

n=0; dup=0
for SEL in ${SELECTORS}; do
  for D in ${DEPTHS}; do
    for SEED in ${SEEDS}; do
      TAG="${START} ${NEW} ${SEL} d${D} s${SEED}"
      if printf '%s' "${INFLIGHT}" | grep -qxF "${TAG}"; then
          echo "  [have] ${TAG}"; dup=$((dup+1)); continue
      fi
      n=$((n+1))
      EX="ALL,DATASET=dfc2020,START=${START},NEW=${NEW},TEACHER=${TEACHER}"
      EX="${EX},CONFIG=${CONFIG},SELECT_BY=${SEL},PROJ_LAYERS=${D},SEED=${SEED}"
      EX="${EX},BATCH_SIZE=8,RESULTS_CSV=${RESULTS_CSV}"
      EX="${EX},CONFIG_LABEL=${SEL}_d${D}"
      # metrics-only: 24 checkpoints would be ~8 G and scratch is quota-bound
      EX="${EX},SAVE_CHECKPOINT=0"
      if [ "${DRYRUN:-0}" = "1" ]; then
          echo "[$n] ${SEL} depth=${D} seed=${SEED}"
      else
          jid=$(sbatch --parsable --export="${EX}" sh/train_delulu_job.sh)
          printf '%s\t%s\n' "${jid}" "${TAG}" >> "${LEDGER}"
          echo "[$n] ${jid}  ${SEL} depth=${D} seed=${SEED}"
      fi
    done
  done
done
echo "total: ${n} submitted; ${dup} already in flight  ->  ${RESULTS_CSV}"
