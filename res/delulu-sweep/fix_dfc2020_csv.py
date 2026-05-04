"""One-shot fixup for sweep_results_dfc2020_final.csv.

Old format (rows 2-62, 30 cols):
  wandb_run_id, dataset, starting_modality, ..., stage0_checkpoint, shote2e_checkpoint

New format (rows 63+, 30 cols):
  wandb_run_id, wandb_project, dataset, starting_modality, ..., stage0_checkpoint

Target schema (29 cols):
  wandb_run_id, wandb_project, dataset, ..., stage0_checkpoint

Run from repo root:
    python res/delulu-sweep/fix_dfc2020_csv.py
"""

import csv
import subprocess

TARGET_HEADER = [
    "wandb_run_id", "wandb_project", "dataset", "starting_modality", "new_modality",
    "teacher_test_metric",
    "lr", "asym_lr", "weight_decay", "epochs",
    "modality_dropout", "labeled_frequency", "labeled_start_fraction",
    "protect_lrm", "use_mask_token", "latent_masked_only",
    "lambda_latent", "lambda_prefusion", "lambda_distill",
    "trainable_params", "active_losses",
    "val_transfer", "test_transfer",
    "val_peeking", "test_peeking",
    "val_addition", "test_addition",
    "val_ens_addition", "test_ens_addition",
    "stage0_checkpoint",
]

OLD_HEADER = [
    "wandb_run_id", "dataset", "starting_modality", "new_modality",
    "teacher_test_metric",
    "lr", "asym_lr", "weight_decay", "epochs",
    "modality_dropout", "labeled_frequency", "labeled_start_fraction",
    "protect_lrm", "use_mask_token", "latent_masked_only",
    "lambda_latent", "lambda_prefusion", "lambda_distill",
    "trainable_params", "active_losses",
    "val_transfer", "test_transfer",
    "val_peeking", "test_peeking",
    "val_addition", "test_addition",
    "val_ens_addition", "test_ens_addition",
    "stage0_checkpoint",
    "shote2e_checkpoint",
]

# Read original from git to avoid operating on already-modified file
raw = subprocess.check_output(
    ["git", "show", "HEAD:res/delulu-sweep/sweep_results_dfc2020_final.csv"]
).decode()
rows = list(csv.reader(raw.splitlines()))
data = rows[1:]  # skip header


def classify_config(active_losses, use_mask_token):
    """Mirror of classify_config() in sweep_analysis_benv2.py."""
    al = str(active_losses)
    mt = str(use_mask_token).strip().lower()
    if mt == 'true':
        return 'mask-token'
    if 'prefusion' not in al and 'latent' not in al:
        return 'other'
    if 'prefusion' not in al:
        return 'no-prefusion'
    if 'latent' not in al:
        return 'no-latent'
    if 'ce' not in al:
        return 'no-batch-mixing'
    return 'delulu'


def infer_project(d):
    cfg  = classify_config(d['active_losses'], d['use_mask_token'])
    ds   = d['dataset']
    slug = f"{d['starting_modality']}{d['new_modality']}"  # e.g. 's1s2'
    return f"delulu-{ds}-final-{cfg}-{slug}"


def is_new_format(row):
    # New format: col 1 is wandb_project slug containing dashes
    return '-' in row[1]


fixed = []
for row in data:
    if is_new_format(row):
        # Already has wandb_project; drop shote2e_checkpoint if present (new rows don't have it)
        d = dict(zip(TARGET_HEADER, row[:30]))
        fixed.append([d[c] for c in TARGET_HEADER])
    else:
        # Old format: map via OLD_HEADER, backfill wandb_project
        d = dict(zip(OLD_HEADER, row))
        d['wandb_project'] = infer_project(d)
        fixed.append([d[c] for c in TARGET_HEADER])

lens = set(len(r) for r in fixed)
assert lens == {len(TARGET_HEADER)}, f"Unexpected row lengths: {lens}"

OUT = 'res/delulu-sweep/sweep_results_dfc2020_final.csv'
with open(OUT, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(TARGET_HEADER)
    writer.writerows(fixed)

print(f"Written {len(fixed)} rows to {OUT}")

# Spot-check
with open(OUT) as f:
    check = list(csv.DictReader(f))
print("Sample old-format row:", check[0]['wandb_run_id'], '|', check[0]['wandb_project'], '|', check[0]['active_losses'])
print("Sample new-format row:", check[62]['wandb_run_id'], '|', check[62]['wandb_project'], '|', check[62]['active_losses'])
