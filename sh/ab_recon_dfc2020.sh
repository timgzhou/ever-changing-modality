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
# Plus three later arms isolating WHY mse_cos does or doesn't help:
#   ccos         MSE + (1 - cosine) on MEAN-CENTERED tokens, cosine weight 0.01
#                at prefusion / 1.0 at latent
#   ccos_w1      same, but cosine weight left at 1.0 everywhere
#   cos_w1       raw mse_cos at the SAME reweighting as ccos (0.01 prefusion)
#
# WHY mse_cos: measured on trained checkpoints, the cross-modal projector is
# WORSE than a trivial per-sample-mean predictor on both MSE and cosine, and
# keeps only 3-27% of the target's across-patch variance -- plain MSE is largely
# satisfied by regressing to the mean. Cosine penalises that collapse.
#
# WHY ccos: the RAW cosine is itself dominated by the sample mean, which the
# collapsed solution already reproduces -- a pure mean predictor scores 0.865
# (dfc2020) / 0.708 (benv2) cosine, so mse_cos charges collapse only ~0.14-0.29.
# Centering each side by its own mean deletes that shared component and charges
# the full 1.0 (verified: 0.105 -> 1.000 on synthetic collapsed input). What is
# left is exactly the per-patch residual structure the variance ratio says is
# missing.
#
# WHY the cosine weight: prefusion targets are raw mid-network activations with
# MSE ~0.009 (dfc2020), while the cosine term is O(1). At weight 1.0 the cosine
# is ~100x the MSE, so mse_cos/ccos at prefusion is close to a pure cosine
# objective AND a ~100x reweight of the whole prefusion term -- which confounds
# loss SHAPE with loss WEIGHT. The ccos arm sets 0.01 to put them in the same
# range; ccos_w1 and cos_w1 are the controls that separate the two effects.
# If ccos wins, check the variance ratio on a checkpoint before believing it
# fixed collapse rather than just reweighting prefusion.
#
# WHY drop_cls: a segmenter's decoder reads only x_norm_patchtokens, so matching
# the teacher's CLS token spends capacity on something the task head never
# consumes. On a classifier the opposite holds, which is one candidate
# explanation for delulu beating distillation on BEN-v2 but only tying on
# DFC2020.
#
# Direction s1->+s2_norgb (the swept one), tuned dfc2020 configs, 3 seeds.
# 3 configs x 7 arms x 3 seeds = 63 jobs. Metrics only (SAVE_CHECKPOINT=0).
# The ledger is keyed on the arm name, so re-running this script submits only
# the arms not already in flight -- the original 36 are left alone.
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
  for ARM in ${ARMS:-control mse_cos drop_cls both ccos ccos_w1 cos_w1}; do
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
        # centered cosine, cosine term scaled to match the prefusion MSE
        ccos)     EX="${EX},RECON_LOSS=mse_ccos,RECON_COS_W_PREFUSION=0.01,RECON_COS_W_LATENT=1.0" ;;
        # centered cosine at the naive weight -- isolates the reweighting effect
        ccos_w1)  EX="${EX},RECON_LOSS=mse_ccos" ;;
        # raw cosine at the ccos weighting -- isolates the centering effect
        cos_w1)   EX="${EX},RECON_LOSS=mse_cos,RECON_COS_W_PREFUSION=0.01,RECON_COS_W_LATENT=1.0" ;;
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
