"""Analysis of sweep_results_nomae.csv (no-MAE sweep).

Hyperparams are treated as dataset/modality-agnostic and applied to all combos.

Outputs:
  - Top-2 hyperparams per val metric (4 metrics × 2 = 8 rows each)
  - Val vs test scatter plots saved to res/delulu-sweep/artifacts/
  - JSON of best configs saved to res/delulu-sweep/artifacts/sweep_best.json

Run from repo root:
    python res/delulu-sweep/sweep_analysis.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

df = pd.read_csv('res/delulu-sweep/sweep_results_nomae.csv')

print(f"Rows: {len(df)}")

VAL_TEST_PAIRS = [
    ('val_peeking',     'test_peeking'),
    ('val_transfer',    'test_transfer'),
    ('val_addition',    'test_addition'),
    ('val_ens_addition','test_ens_addition'),
]

HYPERPARAM_COLS = ['lr', 'weight_decay', 'mask_ratio', 'modality_dropout',
                   'labeled_frequency', 'labeled_start_fraction',
                   'lambda_latent', 'lambda_prefusion', 'lambda_distill']

# ---------------------------------------------------------------------------
# Top-2 per val metric + collect for JSON
# ---------------------------------------------------------------------------

# CSV columns → shot_ete.py argument names
COL_TO_ARG = {
    'lr':                     'lr',
    'weight_decay':           'weight_decay',
    'mask_ratio':             'mae_mask_ratio',
    'modality_dropout':       'modality_dropout',
    'labeled_frequency':      'labeled_frequency',
    'labeled_start_fraction': 'labeled_start_fraction',
    'lambda_latent':          'lambda_latent',
    'lambda_prefusion':       'lambda_prefusion',
    'lambda_distill':         'lambda_distill',
}

best_configs = []

df_nodyn = df[df['dyn_teacher'].astype(str).str.lower() != 'true'].copy()

for val_col, test_col in VAL_TEST_PAIRS:
    metric_name = val_col.replace('val_', '')
    sorted_df = df_nodyn.sort_values(val_col, ascending=False).reset_index(drop=True)
    top2 = sorted_df.head(2)[HYPERPARAM_COLS + ['stage0_checkpoint', val_col, test_col]].copy()

    display = top2[HYPERPARAM_COLS + [val_col, test_col]].map(
        lambda x: float(f'{x:.3g}') if isinstance(x, float) else x
    )
    print(f'\n=== Top-2 by {val_col} ===')
    print(display.to_string(index=False))

    for rank, (_, row) in enumerate(top2.iterrows(), start=1):
        args = {}
        for c in HYPERPARAM_COLS:
            if c not in COL_TO_ARG:
                continue
            val = row[c]
            arg_name = COL_TO_ARG[c]
            args[arg_name] = float(f"{val:.3g}")
        entry = {
            'selected_by': metric_name,
            'rank':        rank,
            'val_score':   float(f"{row[val_col]:.3g}"),
            'test_score':  float(f"{row[test_col]:.3g}"),
            'stage0_checkpoint': row['stage0_checkpoint'],
            'args': args,
        }
        best_configs.append(entry)

# Save JSON
out_dir = 'res/delulu-sweep/artifacts'
os.makedirs(out_dir, exist_ok=True)
json_path = f'{out_dir}/sweep_best.json'
with open(json_path, 'w') as f:
    json.dump(best_configs, f, indent=2)
print(f'\nSaved {len(best_configs)} configs to {json_path}')  # 4 metrics × 2 = 8

# ---------------------------------------------------------------------------
# Val vs test scatter plots
# ---------------------------------------------------------------------------

PLOT_PAIRS = [
    ('val_peeking',  'test_peeking',  'Peeking',  True),
    ('val_transfer', 'test_transfer', 'Transfer', False),
    ('val_addition', 'test_addition', 'Addition', False),
]

METRIC_LABELS = {
    'peeking':  'Peeking',
    'transfer': 'Transfer',
    'addition': 'Addition',
}

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle('Validation vs. Test Performance on reBEN\nStarting Modality = S2, New Modality = S1', fontsize=12)

for ax, (val_col, test_col, metric_label, show_identity) in zip(axes, PLOT_PAIRS):
    x = df[val_col]
    y = df[test_col]
    ax.scatter(x, y, alpha=0.6, s=20)

    if show_identity:
        lo = min(x.min(), y.min()) - 1
        hi = max(x.max(), y.max()) + 1
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='Identity (val = test)')
        ax.legend(fontsize=8)

    r = x.corr(y)
    ax.set_xlabel('Validation Score', fontsize=9)
    ax.set_ylabel('Test Score', fontsize=9)
    ax.set_title(f'{metric_label}  (r = {r:.3f})', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.4)

plt.tight_layout()
out = f'{out_dir}/sweep_val_vs_test.png'
plt.savefig(out, dpi=150)
print(f'Saved plot to {out}')

# python res/delulu-sweep/sweep_analysis.py
