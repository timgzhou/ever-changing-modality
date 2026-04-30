#!/bin/bash
#SBATCH --time=12:00:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/shot_ete_ablate_protect_lrm/%j.out
#SBATCH --mail-user=tiange.zhou@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

DATASET="$1"
MODEL="$2"
STARTING_MOD="$3"
NEW_MOD="$4"
TEACHER="$5"
SELECT_BY="${6:-ens_addition}"

source sh/env.sh
export TQDM_DISABLE=1

SWEEP_JSON="res/delulu-sweep/artifacts/sweep_best.json"
RESULTS_CSV="res/ablation/benv2_protect_lrm.csv"

mkdir -p res/ablation logs/shot_ete_ablate_protect_lrm

row_exists() {
    local csv="$1" dataset="$2" start_mod="$3" new_mod="$4"
    local lr="$5" wd="$6" epochs="$7" mask_ratio="$8"
    local mod_dropout="$9" labeled_freq="${10}" labeled_start="${11}"
    [ -f "$csv" ] || return 1
    awk -F',' -v d="$dataset" -v sm="$start_mod" -v nm="$new_mod" \
        -v lr="$lr" -v wd="$wd" -v ep="$epochs" -v mr="$mask_ratio" \
        -v md="$mod_dropout" -v lf="$labeled_freq" -v ls="$labeled_start" \
        'function near(a,b) { return (a==0 && b==0) || (a!=0 && (a-b)^2/(a*a) < 1e-6) }
         NR>1 && $1==d && $3==sm && $4==nm && near($5+0,lr+0) && near($6+0,wd+0) \
             && $7+0==ep+0 && near($8+0,mr+0) && near($9+0,md+0) \
             && near($10+0,lf+0) && near($11+0,ls+0) && $15=="True" \
             {found=1; exit} END {exit !found}' \
        "$csv"
}

N_CONFIGS=$(jq "[.[] | select(.selected_by==\"${SELECT_BY}\")] | length" "$SWEEP_JSON")

if [ "$N_CONFIGS" -eq 0 ]; then
    echo "[skip] no sweep configs for selected_by=${SELECT_BY}"
    exit 0
fi

echo "=== ${DATASET} | ${MODEL} | ${STARTING_MOD} -> ${NEW_MOD} | ${N_CONFIGS} configs | protect_lrm ==="

for IDX in $(seq 0 $((N_CONFIGS - 1))); do
    ENTRY=$(jq -c "[.[] | select(.selected_by==\"${SELECT_BY}\")] | .[$IDX]" "$SWEEP_JSON")
    RANK=$(echo "$ENTRY" | jq -r '.rank')

    ARGS=$(echo "$ENTRY" | jq -r '.args | to_entries | map("--\(.key) \(.value)") | join(" ")')

    LR=$(echo "$ENTRY"  | jq -r '.args.lr')
    WD=$(echo "$ENTRY"  | jq -r '.args.weight_decay')
    EPOCHS=120
    MR=$(echo "$ENTRY"  | jq -r '.args.mae_mask_ratio')
    MD=$(echo "$ENTRY"  | jq -r '.args.modality_dropout')
    LF=$(echo "$ENTRY"  | jq -r '.args.labeled_frequency')
    LS=$(echo "$ENTRY"  | jq -r '.args.labeled_start_fraction')

    if row_exists "$RESULTS_CSV" "$DATASET" "$STARTING_MOD" "$NEW_MOD" \
                  "$LR" "$WD" "$EPOCHS" "$MR" "$MD" "$LF" "$LS"; then
        echo "  [skip] rank=${RANK} protect_lrm already in CSV"
        continue
    fi

    echo "  rank=${RANK} teacher=${TEACHER} protect_lrm=true"

    python -u shot_ete.py \
        --dataset "$DATASET" \
        --new_mod_group "$NEW_MOD" \
        --stage0_checkpoint "$TEACHER" \
        $ARGS \
        --active_losses latent prefusion distill ce \
        --protect_lrm \
        --results_csv "$RESULTS_CSV" \
        --epochs 120 \
        --eval_every_n_epochs 4 \
        --batch_size 32 \
        --num_workers 4
done
