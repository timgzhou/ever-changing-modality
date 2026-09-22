#!/bin/bash
# Cross-config generalisation run: do hyperparameters tuned on one dataset
# transfer to another?
#
#   6 configs   = {dfc2020, benv2} tuned x {transfer, peeking, addition}
#   2 datasets  = dfc2020, biomassters
#   4 directions= s1->s2, s2->s1, s2_rgb->s2_norgb, s2_norgb->s2_rgb
#   = 48 jobs
#
# Note biomassters is NOT one of the tuned datasets, so every biomassters job is
# an out-of-domain test of both config families; dfc2020 jobs test its own
# configs in-domain and the benv2 configs out-of-domain.
#
# EPOCHS: biomassters is capped at 64 regardless of the config's own epoch count.
# Measured throughput is ~9.9 min/epoch (10h for 64 epochs), so the 128-epoch
# dfc2020 configs would need ~20h and be killed at the 11:59 walltime. The cap
# means the 3 dfc2020-tuned configs run at a different epoch count on
# biomassters than in their source sweep -- a confound to keep in mind when
# comparing config families there, but it is the only way every job completes.
#
# Usage:  bash sh/train_delulu_crossconfig.sh
#         DRYRUN=1 bash sh/train_delulu_crossconfig.sh
#         DATASETS="dfc2020" bash sh/train_delulu_crossconfig.sh
#         WALLTIME=23:59:00 DATASETS=biomassters bash sh/train_delulu_crossconfig.sh
#
# WALLTIME overrides the job script's #SBATCH --time. A job killed at the wall
# writes NO results row -- train_delulu.py appends only after the final epoch.
#
# Throughput (measured 2026-09-18 over 15 recent biomassters jobs, bs 8 and 16
# alike): ~3.8-4.6 min/epoch, so 64 epochs is ~4.5h and 128 epochs ~9.5h, both
# inside the 11:59 default. The earlier 9.4-10.9 min/epoch figure quoted here
# was measured before the loader/prefusion work and no longer holds; keep
# WALLTIME=23:59:00 in reserve if a future change slows the step again.
set -uo pipefail

TEACHERS_JSON="artifacts/sft_teachers.json"
DATASETS="${DATASETS:-dfc2020 biomassters}"
SELECTORS="${SELECTORS:-transfer peeking addition}"
CONFIG_FAMILIES="${CONFIG_FAMILIES:-dfc2020 benv2}"
LEDGER="logs/train_delulu/submitted_crossconfig.tsv"
mkdir -p logs/train_delulu res/delulu; touch "${LEDGER}"

prefix_for () { [ "$1" = "dfc2020" ] && echo "dfc2020_cobench" || echo "$1"; }
decoder_for () { [ "$1" = "dfc2020" ] && echo "upernet" || echo "upernet+relu"; }
# BATCH_SIZE override: --temporal_prefusion keeps T folded through the
# projector, so it sees B*T rows instead of B. At the biomassters default of 16
# that is 192 rows and OOMs on a 44 GB card; 8 gives 96 and fits. Both A/B arms
# must use the SAME value or the comparison is confounded.
batch_for ()   { [ -n "${BATCH_SIZE:-}" ] && { echo "${BATCH_SIZE}"; return; }
                 [ "$1" = "dfc2020" ] && echo "8" || echo "16"; }

# Skip (dataset,direction,config,selector) combos already submitted and still
# queued/running, so re-running the launcher is safe.
ACTIVE=" $(squeue -u "$USER" -h -o '%i' 2>/dev/null | tr '\n' ' ') "
INFLIGHT=""
while read -r jid rest; do
    case "${ACTIVE}" in *" ${jid} "*) INFLIGHT="${INFLIGHT}${rest}"$'\n' ;; esac
done < "${LEDGER}"

n=0; dup=0; miss=0
for DATASET in ${DATASETS}; do
    PRE=$(prefix_for "${DATASET}"); DEC=$(decoder_for "${DATASET}"); BS=$(batch_for "${DATASET}")
    # PAIRS is a space-separated list of START:NEW, overriding the default,
    # e.g. PAIRS="s2_rgb:s1" to fill a single missing table cell. s2_rgb->s1 is
    # in the paper table but was never in the default list.
    for pair in ${PAIRS:-s1:s2 s2:s1 s2_rgb:s2_norgb s2_norgb:s2_rgb}; do
        START="${pair%%:*}"; NEW="${pair##*:}"
        # TEMPORAL_POOL=1 selects the mean-pooled biomassters teachers (the
        # '/tpooled' registry suffix) and passes the pooled window through as a
        # negative --num_time_steps. Student and teacher MUST come from the same
        # regime: a pooled student reads [C,1,H,W] while a T=12 teacher expects
        # the full stack.
        KEY="${PRE}/${START}/evan_base/${DEC}/split1${TEMPORAL_POOL:+/tpooled}"
        TEACHER=$(jq -r ".\"${KEY}\".checkpoint // empty" "${TEACHERS_JSON}")
        if [ -z "${TEACHER}" ] || [ ! -f "${TEACHER}" ]; then
            echo "  [skip] ${DATASET} ${START}->+${NEW}: no teacher (${KEY})"; miss=$((miss+1)); continue
        fi
        for FAM in ${CONFIG_FAMILIES}; do
            CONFIG="configs/delulu_best_${FAM}.yaml"
            for SEL in ${SELECTORS}; do
                # EPOCHS is part of the key: a 64- and a 128-epoch run of the
                # same cell are different experiments, and without it the
                # in-flight guard would silently skip the second one.
                TAG="${DATASET} ${START} ${NEW} ${FAM} ${SEL}${LR:+ lr${LR}}${EPOCHS:+ ep${EPOCHS}}${STUDENT_INIT:+ init${STUDENT_INIT}}${SEED:+ s${SEED}}${TEMPORAL_PREFUSION:+ tpf}${TEMPORAL_POOL:+ tpool}"
                if printf '%s' "${INFLIGHT}" | grep -qxF "${TAG}"; then
                    echo "  [have] ${TAG}: in flight"; dup=$((dup+1)); continue
                fi
                n=$((n+1))
                EX="ALL,DATASET=${DATASET},START=${START},NEW=${NEW},TEACHER=${TEACHER}"
                EX="${EX},CONFIG=${CONFIG},SELECT_BY=${SEL},BATCH_SIZE=${BS}"
                # Pooled runs go to their OWN csv. The crossconfig schema has no
                # num_time_steps column -- it lives only in the checkpoint -- so
                # a pooled and a T=12 row are indistinguishable once written,
                # and res/results_BL.py would pool them into one mean per cell.
                # The _tpool config_label alone is too fragile to rely on for
                # that separation, and a separate file also makes the T=12 set
                # trivially archivable as one unit.
                _RCSV="res/delulu/${DATASET}_crossconfig.csv"
                if [ -n "${TEMPORAL_POOL:-}" ] && [ "${TEMPORAL_POOL}" != "0" ]; then
                    _RCSV="res/delulu/${DATASET}_crossconfig_tpooled.csv"
                fi
                EX="${EX},RESULTS_CSV=${_RCSV}"
                # LR override: the source config's lr can be catastrophically
                # wrong for the target dataset. On biomassters every config with
                # lr <= 1e-4 collapsed (dfc2020_peeking 5/5, benv2_addition 4/4;
                # RMSE 70-86 against a ~42-48 teacher) while every config with
                # lr >= 3.9e-4 survived (0/14). Set LR=<value> to rescue those
                # families; the label records it so the rows stay distinguishable.
                LBL="${FAM}_${SEL}"
                if [ -n "${LR:-}" ]; then
                    EX="${EX},LR=${LR}"
                    LBL="${LBL}_lr${LR}"
                fi
                # Epoch count in the label too: _load_delulu picks the top-3
                # configs by val per cell, so a 64- and a 128-epoch run of the
                # same config would otherwise pool into one mean.
                if [ -n "${EPOCHS:-}" ] && [ "${EPOCHS}" != "64" ]; then
                    LBL="${LBL}_ep${EPOCHS}"
                fi
                # STUDENT_INIT=random re-rolls the student's weights while
                # KEEPING the frozen teacher, so supervision is unchanged and
                # only the starting point moves (train_delulu.py:255). The
                # earlier 6-pair probe at lr 1.57e-4 predates the per-selector
                # tuned configs and had no config_label, so it is not comparable
                # to the current table; this reruns it properly. The label
                # records the arm so teacher- and random-init rows never pool.
                if [ -n "${STUDENT_INIT:-}" ] && [ "${STUDENT_INIT}" != "teacher" ]; then
                    EX="${EX},STUDENT_INIT=${STUDENT_INIT}"
                    LBL="${LBL}_init${STUDENT_INIT}"
                fi
                # SEED: replicates. The teacher-init rows show within-pair sd
                # 0.77-0.99 mIoU, so a single run per cell cannot resolve the
                # ~0.5 effect this ablation is chasing.
                [ -n "${SEED:-}" ] && EX="${EX},SEED=${SEED}" && LBL="${LBL}_s${SEED}"
                # Loss-weight overrides. The config's lambdas were tuned on the
                # SOURCE dataset; on biomassters the prefusion/latent terms are
                # reconstruction losses whose natural scale differs with the
                # target (RMSE regression, not logits), so they need their own
                # values. Labelled so rows at different weights never pool.
                if [ -n "${LAMBDA_PREFUSION:-}" ]; then
                    EX="${EX},LAMBDA_PREFUSION=${LAMBDA_PREFUSION}"
                    LBL="${LBL}_lpf${LAMBDA_PREFUSION}"
                fi
                if [ -n "${LAMBDA_LATENT:-}" ]; then
                    EX="${EX},LAMBDA_LATENT=${LAMBDA_LATENT}"
                    LBL="${LBL}_llat${LAMBDA_LATENT}"
                fi
                # Mean-pooled biomassters: negative NUM_TIME_STEPS makes the
                # loader pool T at the input and read the on-disk cache, so the
                # run is fully non-temporal (~7x faster per epoch). Labelled so
                # pooled and temporal rows never pool in the table selection.
                if [ -n "${TEMPORAL_POOL:-}" ] && [ "${TEMPORAL_POOL}" != "0" ]; then
                    EX="${EX},NUM_TIME_STEPS=-${POOL_T:-12}"
                    LBL="${LBL}_tpool"
                fi
                # A/B arm: prefusion before the temporal pool (biomassters only).
                if [ -n "${TEMPORAL_PREFUSION:-}" ] && [ "${TEMPORAL_PREFUSION}" != "0" ]; then
                    EX="${EX},TEMPORAL_PREFUSION=1"; LBL="${LBL}_tpf"
                fi
                EX="${EX},CONFIG_LABEL=${LBL}"
                # biomassters: pin 64 epochs, overriding the config's value --
                # unless the caller set EPOCHS explicitly (e.g. a 24-epoch A/B,
                # which costs ~4 h/job instead of ~10 h at ~9-10 min/epoch).
                [ "${DATASET}" = "biomassters" ] && EX="${EX},EPOCHS=${EPOCHS:-64}"
                if [ "${DRYRUN:-0}" = "1" ]; then
                    echo "[$n] ${DATASET} ${START}->+${NEW}  cfg=${FAM}/${SEL}"
                else
                    TIME_ARG=""
                    [ -n "${WALLTIME:-}" ] && TIME_ARG="--time=${WALLTIME}"
                    jid=$(sbatch --parsable ${TIME_ARG} --export="${EX}" sh/train_delulu_job.sh)
                    printf '%s\t%s\n' "${jid}" "${TAG}" >> "${LEDGER}"
                    echo "[$n] ${jid}  ${DATASET} ${START}->+${NEW}  cfg=${FAM}/${SEL}"
                fi
            done
        done
    done
done
echo "total: ${n} submitted; ${dup} already in flight; ${miss} skipped (no teacher)"
