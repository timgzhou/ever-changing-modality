"""Analysis of sweep_results.csv for dfc2020/s2_rgb/s1 with prefusion+distill+ce+latent.

Outputs:
  - Top-3 hyperparams per val metric (4 metrics × 3 = 12 rows each)
  - Val vs test scatter plots saved to artifacts/
  - JSON of best configs saved to artifacts/sweep_best.json (readable by bash/python)

Run from repo root:
    python res/sweep_analysis.py
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------------------------
# Load and filter
# ---------------------------------------------------------------------------

df = pd.read_csv('res/sweep_results.csv')
df = df[
    (df['dataset'] == 'benv2') &
    (df['starting_modality'] == 's2') &
    (df['new_modality'] == 's1') &
    (df['active_losses'] == 'prefusion+distill+ce+mae+latent')
].copy()

print(f"Filtered rows: {len(df)}")

VAL_TEST_PAIRS = [
    ('val_transfer',    'test_transfer'),
    ('val_peeking',     'test_peeking'),
    ('val_addition',    'test_addition'),
    ('val_ens_addition','test_ens_addition'),
]

HYPERPARAM_COLS = ['lr', 'weight_decay', 'mask_ratio', 'modality_dropout',
                   'labeled_frequency', 'labeled_start_fraction',
                   'lambda_latent', 'lambda_prefusion', 'lambda_distill', 'lambda_mae', 'lambda_ce']

# ---------------------------------------------------------------------------
# Top-3 per val metric + collect for JSON
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
    'lambda_mae':             'lambda_mae',
    'lambda_ce':              'lambda_ce',
}

best_configs = []

for val_col, test_col in VAL_TEST_PAIRS:
    metric_name = val_col.replace('val_', '')
    sorted_df = df.sort_values(val_col, ascending=False).reset_index(drop=True)
    top3 = sorted_df.head(3)[HYPERPARAM_COLS + ['stage0_checkpoint', val_col, test_col]].copy()

    # 3 significant digits for display
    display = top3[HYPERPARAM_COLS + [val_col, test_col]].map(
        lambda x: float(f'{x:.3g}') if isinstance(x, float) else x
    )
    print(f'\n=== Top-3 by {val_col} ===')
    print(display.to_string(index=False))

    # Collect JSON entries (full precision)
    for rank, (_, row) in enumerate(top3.iterrows(), start=1):
        entry = {
            'selected_by': metric_name,
            'rank':        rank,
            'val_score':   round(float(row[val_col]), 4),
            'test_score':  round(float(row[test_col]), 4),
            'stage0_checkpoint': row['stage0_checkpoint'],
            'args': {COL_TO_ARG[c]: float(row[c]) for c in HYPERPARAM_COLS if c in COL_TO_ARG},
        }
        best_configs.append(entry)

# Save JSON
os.makedirs('artifacts', exist_ok=True)
json_path = 'artifacts/sweep_best.json'
with open(json_path, 'w') as f:
    json.dump(best_configs, f, indent=2)
print(f'\nSaved {len(best_configs)} configs to {json_path}')

# ---------------------------------------------------------------------------
# Val vs test scatter plots
# ---------------------------------------------------------------------------

os.makedirs('artifacts', exist_ok=True)

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('Val vs Test — dfc2020 s2_rgb→s1 (prefusion+distill+ce+latent)', fontsize=12)

for ax, (val_col, test_col) in zip(axes, VAL_TEST_PAIRS):
    x = df[val_col]
    y = df[test_col]
    ax.scatter(x, y, alpha=0.6, s=20)

    # y=x reference line
    lo = min(x.min(), y.min()) - 1
    hi = max(x.max(), y.max()) + 1
    ax.plot([lo, hi], [lo, hi], 'k--', linewidth=0.8, label='y=x')

    # Pearson r
    r = x.corr(y)
    ax.set_xlabel(val_col, fontsize=9)
    ax.set_ylabel(test_col, fontsize=9)
    metric = val_col.replace('val_', '')
    ax.set_title(f'{metric}  (r={r:.3f})', fontsize=10)
    ax.legend(fontsize=8)

plt.tight_layout()
out = 'artifacts/sweep_val_vs_test.png'
plt.savefig(out, dpi=150)
print(f'\nSaved plot to {out}')

# python res/sweep_analysis.py
