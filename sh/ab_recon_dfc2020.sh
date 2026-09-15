#!/bin/bash
# A/B: do the two reconstruction-loss fixes close the gap to the distillation
# baseline on DFC2020?
#
# Four arms, 2x2 over two independent changes:
#   control      plain MSE, CLS kept          (current behaviour)
#   mse_cos      MSE + (1 - cosine), CLS kept
#   drop_cls     plain MSE, CLS dropped
#   both         MSE + cosine, CLS dropped
#
# WHY mse_cos: measured on trained checkpoints, the cross-modal projector is
# WORSE than a trivial per-sample-mean predictor on both MSE and cosine, and
# keeps only 3-27% of the target's across-patch variance -- plain MSE is largely
# satisfied by regressing to the mean. Cosine penalises that collapse.
#
# WHY drop_cls: a segmenter's decoder reads only x_norm_patchtokens, so matching
# the teacher's CLS token spends capacity on something the task head never
# consumes. On a classifier the opposite holds, which is one candidate
# explanation for delulu beating distillation on BEN-v2 but only tying on
# DFC2020.
#
# Direction s1->+s2_norgb (the swept one), tuned dfc2020 configs, 3 seeds.
# 3 configs x 4 arms x 3 seeds = 36 jobs. Metrics only (SAVE_CHECKPOINT=0).
set -uo pipefail

CONFIG="configs/delulu_best_dfc2020.yaml"
START="${START:-s1}"; NEW="${NEW:-s2_norgb}"
SELECTORS="${SELECTORS:-transfer peeking addition}"
SEEDS="${SEEDS:-0 1 2}"
RESULTS_CSV="res/delulu/dfc2020_recon_ab.csv"
LEDGER="logs/train_delulu/submitted_recon.tsv"
mkdir -p logs/train_delulu res/delulu; touch "${LEDGER}"

TEACHER=$(jq -r ".\"dfc2020_cobench/${START}/evan_base/upernet/split1\".checkpoint // empty" artifacts/sft_teachers.json)
[ -n "${TEACHER}" ] && [ -f "${TEACHER}" ] || { echo "[error] no teacher for ${START}"; exit 1; }

ACTIVE=" $(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '\n' ' ') "
INFLIGHT=""
while read -r jid rest; do
    case "${ACTIVE}" in *" ${jid} "*) INFLIGHT="${INFLIGHT}${rest}"$'\n' ;; esac
done < "${LEDGER}"

n=0; dup=0
for SEL in ${SELECTORS}; do
  for ARM in control mse_cos drop_cls both; do
    for SEED in ${SEEDS}; do
      TAG="${START} ${NEW} ${SEL} ${ARM} s${SEED}"
      if printf '%s' "${INFLIGHT}" | grep -qxF "${TAG}"; then
          echo "  [have] ${TAG}"; dup=$((dup+1)); continue
      fi
      EX="ALL,DATASET=dfc2020,START=${START},NEW=${NEW},TEACHER=${TEACHER}"
      EX="${EX},CONFIG=${CONFIG},SELECT_BY=${SEL},SEED=${SEED},BATCH_SIZE=8"
      EX="${EX},RESULTS_CSV=${RESULTS_CSV},SAVE_CHECKPOINT=0"
      EX="${EX},CONFIG_LABEL=${SEL}_${ARM}"
      case "${ARM}" in
        mse_cos)  EX="${EX},RECON_LOSS=mse_cos" ;;
        drop_cls) EX="${EX},RECON_DROP_CLS=1" ;;
        both)     EX="${EX},RECON_LOSS=mse_cos,RECON_DROP_CLS=1" ;;
      esac
      n=$((n+1))
      if [ "${DRYRUN:-0}" = "1" ]; then
          echo "[$n] ${SEL} ${ARM} seed=${SEED}"
      else
          jid=$(sbatch --parsable --export="${EX}" sh/train_delulu_job.sh)
          printf '%s\t%s\n' "${jid}" "${TAG}" >> "${LEDGER}"
          echo "[$n] ${jid}  ${SEL} ${ARM} seed=${SEED}"
      fi
    done
  done
done
echo "total: ${n} submitted; ${dup} in flight  ->  ${RESULTS_CSV}"
