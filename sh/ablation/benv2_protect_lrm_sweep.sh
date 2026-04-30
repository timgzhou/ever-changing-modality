#!/bin/bash
# BEN-v2: ablate protect_lrm for both s2->s1 and s1->s2.
# Runs full loss set (latent prefusion distill ce) with protect_lrm=True.
# Usage: bash sh/ablation/benv2_protect_lrm_sweep.sh

DATASET="benv2"
MODEL="evan_base"
TEACHERS_JSON="artifacts/sft_teachers.json"

mkdir -p logs/shot_ete_ablate_protect_lrm res/ablation

for STARTING_MOD in s2 s1; do
    if [ "$STARTING_MOD" = "s2" ]; then NEW_MOD="s1"; else NEW_MOD="s2"; fi

    TEACHER_KEY="${DATASET}/${STARTING_MOD}/${MODEL}"
    TEACHER=$(jq -r ".\"${TEACHER_KEY}\".checkpoint // empty" "$TEACHERS_JSON")
    if [ -z "$TEACHER" ]; then
        echo "[error] no teacher found for ${TEACHER_KEY}"
        continue
    fi
    if [ ! -f "$TEACHER" ]; then
        echo "[error] teacher checkpoint missing: $TEACHER"
        continue
    fi

    for SELECT_BY in transfer peeking addition ens_addition; do
        echo "[submit] ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | ${SELECT_BY} | protect_lrm"
        sbatch sh/ablation/benv2_protect_lrm_job.sh \
            "$DATASET" "$MODEL" "$STARTING_MOD" "$NEW_MOD" "$TEACHER" "$SELECT_BY"
    done
done

# bash sh/ablation/benv2_protect_lrm_sweep.sh