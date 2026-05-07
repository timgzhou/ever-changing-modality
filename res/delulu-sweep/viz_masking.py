import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

df = pd.read_csv('sweep_results_benv2_masking.csv')

MODES = [
    ('peeking',  'Peek'),
    ('transfer', 'Transfer'),
    ('addition', 'Addition'),
]
IDENTITY_LINE = {'peeking', 'peek'}
LRM_VALS = [0.0, 0.5, 1.0]
LRM_LABELS = {0.0: 'protect_lrm=0', 0.5: 'protect_lrm=0.5', 1.0: 'protect_lrm=1'}
COLORS = {0.0: '#4e79a7', 0.5: '#f28e2b', 1.0: '#e15759'}
# unprotect_starting_mod: False=filled circle, True=open diamond
USM_MARKER  = {False: 'o', True: 'D'}
USM_FILL    = {False: 1.0, True: 0.0}   # alpha for facecolor
USM_LABELS  = {False: 'unprotect_start=F', True: 'unprotect_start=T'}
USM_COLORS  = {False: '#555555', True: '#cc77cc'}

fig = plt.figure(figsize=(14, 12))
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.32)

# ── Row 0: val vs test scatter — colour=protect_lrm, shape=unprotect_starting_mod ──
for col_idx, (key, title) in enumerate(MODES):
    ax = fig.add_subplot(gs[0, col_idx])
    for lrm in LRM_VALS:
        for usm in [False, True]:
            sub = df[(df['protect_lrm'] == lrm) & (df['unprotect_starting_mod'] == usm)]
            if sub.empty:
                continue
            ax.scatter(
                sub[f'val_{key}'], sub[f'test_{key}'],
                c=COLORS[lrm], marker=USM_MARKER[usm],
                s=45, alpha=0.80,
                facecolors=COLORS[lrm] if not usm else 'none',
                edgecolors=COLORS[lrm],
                linewidths=1.2,
                zorder=3,
            )
    if key in IDENTITY_LINE:
        all_vals = pd.concat([df[f'val_{key}'], df[f'test_{key}']])
        lo, hi = all_vals.min(), all_vals.max()
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'k--', lw=0.8, alpha=0.4)
    ax.set_xlabel('Val mAP')
    ax.set_ylabel('Test mAP')
    ax.set_title(title)
    ax.grid(True, lw=0.4, alpha=0.5)
    if col_idx == 0:
        # legend: colour for lrm, shape for usm
        handles = []
        for lrm in LRM_VALS:
            handles.append(plt.Line2D([0], [0], marker='o', color='w',
                markerfacecolor=COLORS[lrm], markersize=7, label=LRM_LABELS[lrm]))
        for usm in [False, True]:
            handles.append(plt.Line2D([0], [0], marker=USM_MARKER[usm], color='w',
                markerfacecolor='#555' if not usm else 'none',
                markeredgecolor='#555', markersize=7, label=USM_LABELS[usm]))
        ax.legend(handles=handles, fontsize=6.5, markerscale=1.0)

# ── Row 1: test mAP by protect_lrm, shape=unprotect_starting_mod ────────────
for col_idx, (key, title) in enumerate(MODES):
    ax = fig.add_subplot(gs[1, col_idx])
    for i, lrm in enumerate(LRM_VALS):
        sub = df[df['protect_lrm'] == lrm][f'test_{key}'].dropna()
        ax.boxplot(sub, positions=[i], widths=0.45, patch_artist=True,
            boxprops=dict(facecolor=COLORS[lrm], alpha=0.5),
            medianprops=dict(color='black', lw=1.5),
            whiskerprops=dict(lw=1), capprops=dict(lw=1),
            flierprops=dict(marker='x', markersize=4, alpha=0.5))
        rng = np.random.default_rng(42)
        for usm in [False, True]:
            pts = df[(df['protect_lrm'] == lrm) & (df['unprotect_starting_mod'] == usm)][f'test_{key}'].dropna()
            if pts.empty:
                continue
            jitter = rng.uniform(-0.15, 0.15, len(pts))
            ax.scatter(i + jitter, pts,
                marker=USM_MARKER[usm], s=28,
                facecolors=COLORS[lrm] if not usm else 'none',
                edgecolors=COLORS[lrm], linewidths=1.2,
                alpha=0.85, zorder=4)
    ax.set_xticks(range(len(LRM_VALS)))
    ax.set_xticklabels(['0', '0.5', '1'], fontsize=9)
    ax.set_xlabel('protect_lrm')
    ax.set_ylabel('Test mAP')
    ax.set_title(f'{title} — by protect_lrm')
    ax.grid(True, axis='y', lw=0.4, alpha=0.5)
    if col_idx == 0:
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#555',
                markeredgecolor='#555', markersize=7, label='unprotect_start=F'),
            plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='none',
                markeredgecolor='#555', markersize=7, label='unprotect_start=T'),
        ]
        ax.legend(handles=handles, fontsize=6.5)

# ── Row 2: test mAP by unprotect_starting_mod ───────────────────────────────
USM_VALS = [False, True]
USM_BOX_COLORS = {False: '#888888', True: '#cc77cc'}

for col_idx, (key, title) in enumerate(MODES):
    ax = fig.add_subplot(gs[2, col_idx])
    for i, usm in enumerate(USM_VALS):
        sub = df[df['unprotect_starting_mod'] == usm][f'test_{key}'].dropna()
        ax.boxplot(sub, positions=[i], widths=0.45, patch_artist=True,
            boxprops=dict(facecolor=USM_BOX_COLORS[usm], alpha=0.5),
            medianprops=dict(color='black', lw=1.5),
            whiskerprops=dict(lw=1), capprops=dict(lw=1),
            flierprops=dict(marker='x', markersize=4, alpha=0.5))
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(sub))
        ax.scatter(i + jitter, sub, c=USM_BOX_COLORS[usm], s=18, alpha=0.7, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['False', 'True'], fontsize=9)
    ax.set_xlabel('unprotect_starting_mod')
    ax.set_ylabel('Test mAP')
    ax.set_title(f'{title} — by unprotect_starting_mod')
    ax.grid(True, axis='y', lw=0.4, alpha=0.5)

fig.suptitle('BEN-v2 masking sweep', fontsize=13, y=1.01)

out = 'viz_masking.png'
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f'Saved {out}')

# ── Best config per eval type (highest val score) ────────────────────────────
HPARAM_COLS = [
    "lr", "weight_decay", "epochs",
    "modality_dropout", "modality_dropout_startmod", "modality_dropout_newmod",
    "labeled_frequency", "labeled_start_fraction",
    "protect_lrm", "use_mask_token", "latent_masked_only", "unprotect_starting_mod",
    "lambda_latent", "lambda_prefusion", "lambda_distill", "mae_mask_ratio",
    "active_losses", "stage0_checkpoint",
]

VAL_FOR_EVAL = {
    "transfer": "val_transfer",
    "peeking":  "val_peeking",
    "addition": "val_addition",
}

def row_to_hparams(row):
    out = {}
    for col in HPARAM_COLS:
        if col not in row.index:
            continue
        val = row[col]
        if pd.isna(val):
            out[col] = None
        elif isinstance(val, (bool, np.bool_)):
            out[col] = bool(val)
        elif isinstance(val, float):
            out[col] = float(val)
        elif isinstance(val, (int, np.integer)):
            out[col] = int(val)
        else:
            out[col] = str(val)
    return out

sub = df[(df['unprotect_starting_mod'] == True) & (df['protect_lrm'] != 1.0)]

best = {}
for eval_type, val_col in VAL_FOR_EVAL.items():
    test_col = val_col.replace("val_", "test_")
    row = sub.nlargest(1, val_col).iloc[0]
    best[eval_type] = {
        "val_col":      val_col,
        "val_score":    float(row[val_col]),
        "test_score":   float(row[test_col]),
        "wandb_run_id": str(row["wandb_run_id"]),
        "hparams":      row_to_hparams(row),
    }
    print(f"Best {eval_type}: run {row['wandb_run_id']}  val={row[val_col]:.2f}  test={row[test_col]:.2f}")

best_path = 'best_masking.json'
with open(best_path, 'w') as f:
    json.dump(best, f, indent=2)
print(f'Saved {best_path}')
