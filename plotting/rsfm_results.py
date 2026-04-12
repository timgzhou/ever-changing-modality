"""
Plot RSFM SFT results from res/rsfm_results.csv.

For each (model, dataset, modality, train_mode), picks the best LR by test_metric,
then plots one subplot per dataset. Within each subplot, x groups = modalities,
and bars within each group = model × train_mode combos (hue=model, hatch=train_mode).
"""

import re
import io
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'res', 'rsfm_results.csv')

# ---------------------------------------------------------------------------
# Load & fix concatenated rows (two rows joined without newline)
# ---------------------------------------------------------------------------
with open(CSV_PATH) as f:
    raw = f.read()
raw = re.sub(r'(\.pt)(?=[a-z])', r'\1\n', raw)
df = pd.read_csv(io.StringIO(raw))
df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
df = df.dropna(subset=['model', 'dataset', 'modality', 'train_mode', 'test_metric'])

# ---------------------------------------------------------------------------
# Best LR: keep max test_metric per (model, dataset, modality, train_mode)
# ---------------------------------------------------------------------------
group_keys = ['model', 'dataset', 'modality', 'train_mode']
best = (
    df.sort_values('test_metric', ascending=False)
      .drop_duplicates(subset=group_keys)
      .reset_index(drop=True)
)

# ---------------------------------------------------------------------------
# Layout config
# ---------------------------------------------------------------------------
MODALITY_ORDER = {
    'eurosat': ['s2', 'rgb', 'vre', 'nir', 'swir'],
    'benv2':   ['s2', 's1', 's2s1', 's2_rgb'],
    'dfc2020': ['s2', 's1', 's2s1', 's2_rgb'],
}
METRIC_LABEL = {
    'eurosat': 'Accuracy (%)',
    'benv2':   'mAP (%)',
    'dfc2020': 'mIoU (%)',
}

models      = sorted(best['model'].unique())       # e.g. ['olmoearth', 'panopticon']
train_modes = sorted(best['train_mode'].unique())  # ['fft', 'lp']
datasets    = sorted(best['dataset'].unique())

palette      = sns.color_palette('tab10', n_colors=len(models))
model_colors = {m: palette[i] for i, m in enumerate(models)}
mode_hatches = {'lp': '', 'fft': '////'}

# bars per modality group = len(models) × len(train_modes)
combos     = [(m, tm) for m in models for tm in train_modes]
n_combos   = len(combos)
bar_width  = 0.8 / n_combos
group_gap  = 1.0  # x-distance between modality groups

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, len(datasets), figsize=(6 * len(datasets), 5), sharey=False)
if len(datasets) == 1:
    axes = [axes]

for ax, dataset in zip(axes, datasets):
    sub = best[best['dataset'] == dataset]
    mod_order = [m for m in MODALITY_ORDER.get(dataset, []) if m in sub['modality'].values]
    n_mods = len(mod_order)

    group_centers = np.arange(n_mods) * group_gap
    offsets = np.linspace(-(n_combos - 1) / 2, (n_combos - 1) / 2, n_combos) * bar_width

    for idx, (model, mode) in enumerate(combos):
        vals = []
        for mod in mod_order:
            row = sub[(sub['model'] == model) & (sub['train_mode'] == mode) & (sub['modality'] == mod)]
            vals.append(row['test_metric'].values[0] if not row.empty else float('nan'))

        ax.bar(
            group_centers + offsets[idx],
            vals,
            width=bar_width,
            color=model_colors[model],
            hatch=mode_hatches[mode],
            edgecolor='white',
            linewidth=0.5,
            alpha=0.85 if mode == 'lp' else 1.0,
        )

    ax.set_title(dataset, fontsize=13, fontweight='bold')
    ax.set_xticks(group_centers)
    ax.set_xticklabels(mod_order, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Modality', fontsize=10)
    ax.set_ylabel(METRIC_LABEL.get(dataset, 'Test Metric (%)'), fontsize=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.1f}'))
    ax.grid(axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)

# ---------------------------------------------------------------------------
# Legend: model = color patch, train_mode = hatch pattern
# ---------------------------------------------------------------------------
legend_handles = []
for model in models:
    legend_handles.append(mpatches.Patch(
        facecolor=model_colors[model], edgecolor='grey', label=model,
    ))
legend_handles.append(mpatches.Patch(visible=False, label=''))  # spacer
for mode in train_modes:
    legend_handles.append(mpatches.Patch(
        facecolor='lightgrey',
        hatch=mode_hatches[mode],
        edgecolor='grey',
        label=f'train_mode={mode}',
    ))

fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=len(models) + 1 + len(train_modes),
    fontsize=9,
    frameon=False,
    bbox_to_anchor=(0.5, -0.06),
)

fig.suptitle('RSFM SFT — Best LR per (model, dataset, modality, train_mode)', fontsize=13, y=1.01)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), '..', 'res', 'rsfm_results.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
