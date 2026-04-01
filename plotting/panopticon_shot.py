"""
Plot SHOT results using Panopticon LP checkpoints as teacher (BEN-v2).

Two side-by-side subplots: self-projector and cross-projector.
Each subplot has two groups (s1->s2, s2->s1), each with 4 bars:
  valchecked_transfer, valchecked_peek, valchecked_add, valchecked_add_ens
  (mean ± std across runs with the same checkpoint mode suffix).

Horizontal baselines:
  - Panopticon LP s2 (trained on s2): 62.46
  - Panopticon LP s1 (trained on s1): 55.68
  - Panopticon zero-shot s1->s2 (s1 teacher evaluated on s2):  31.09
  - Panopticon zero-shot s2->s1 (s2 teacher evaluated on s1):  27.45
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── data ──────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), '..')
df_cross = pd.read_csv(os.path.join(ROOT, 'res/delulu/benv2/panopticon_cross.csv'))
df_self  = pd.read_csv(os.path.join(ROOT, 'res/delulu/benv2/panopticon_self.csv'))

# Panopticon LP baselines (from res/train_stage0_benv2.csv)
# s2-trained head: test_s2=62.46, test_s1=27.45
# s1-trained head: test_s2=31.09, test_s1=55.68
BASELINES = {
    's2': 62.46,  # panopticon LP trained on s2, evaluated on s2
    's1': 55.68,  # panopticon LP trained on s1, evaluated on s1
    'zs_s1_to_s2': 31.09,  # s1 teacher zero-shot on s2 (s1 head applied to s2 features)
    'zs_s2_to_s1': 27.45,  # s2 teacher zero-shot on s1
}

METRICS = ['valchecked_transfer', 'valchecked_peek', 'valchecked_add', 'valchecked_add_ens']
METRIC_LABELS = ['Transfer', 'Peeking', 'Addition', 'Ens Addition']

DIRECTIONS = [('s1', 's2'), ('s2', 's1')]  # (starting_modality, new_modality)
DIRECTION_LABELS = ['S1 → S2', 'S2 → S1']


def extract_mode(checkpoint_path):
    """Extract hparam mode suffix from checkpoint filename."""
    m = re.search(r'_(transfer|peeking|addition|ens_addition)\.pt$', checkpoint_path)
    return m.group(1) if m else None


def compute_stats(df, start_mod, new_mod):
    """Return dict of metric -> (mean, std) aggregated over all runs for this direction."""
    mask = (df['starting_modality'] == start_mod) & (df['new_modality'] == new_mod)
    sub = df[mask].copy()
    sub['mode'] = sub['shote2e_checkpoint'].apply(extract_mode)
    results = {}
    for metric in METRICS:
        vals = pd.to_numeric(sub[metric], errors='coerce').dropna().values
        results[metric] = (vals.mean(), vals.std()) if len(vals) > 0 else (0.0, 0.0)
    return results


def plot_subplot(ax, df, proj_label):
    n_metrics = len(METRICS)
    n_directions = len(DIRECTIONS)
    group_width = 0.8
    bar_width = group_width / n_metrics
    group_gap = 0.3
    group_centers = np.arange(n_directions) * (group_width + group_gap)

    colors = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

    for m_idx, (metric, label) in enumerate(zip(METRICS, METRIC_LABELS)):
        offsets = (np.arange(n_metrics) - (n_metrics - 1) / 2) * bar_width
        xs = group_centers + offsets[m_idx]
        for d_idx, (start_mod, new_mod) in enumerate(DIRECTIONS):
            stats = compute_stats(df, start_mod, new_mod)
            mean, std = stats[metric]
            ax.bar(xs[d_idx], mean, width=bar_width * 0.9,
                   color=colors[m_idx], yerr=std, capsize=3,
                   error_kw=dict(elinewidth=1, ecolor='black', alpha=0.7),
                   label=label if d_idx == 0 else None)

    # Horizontal baselines — all four span the full plot
    x_left  = group_centers[0] - group_width / 2 - group_gap / 4
    x_right = group_centers[-1] + group_width / 2 + group_gap / 4

    ax.hlines(BASELINES['s2'], x_left, x_right,
              colors='steelblue', linestyles='--', linewidth=1.5, label='LP s2 baseline')
    ax.hlines(BASELINES['zs_s1_to_s2'], x_left, x_right,
              colors='steelblue', linestyles=':', linewidth=1.5, label='Zero-shot s1→s2')
    ax.hlines(BASELINES['s1'], x_left, x_right,
              colors='coral', linestyles='--', linewidth=1.5, label='LP s1 baseline')
    ax.hlines(BASELINES['zs_s2_to_s1'], x_left, x_right,
              colors='coral', linestyles=':', linewidth=1.5, label='Zero-shot s2→s1')

    ax.set_xticks(group_centers)
    ax.set_xticklabels(DIRECTION_LABELS, fontsize=11)
    ax.set_ylabel('mAP (%)', fontsize=11)
    ax.set_title(f'Panopticon Teacher — {proj_label} Projector', fontsize=12)
    ax.set_ylim(20, 75)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)


# ── figure ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
fig.suptitle('SHOT with Panopticon Teacher (BEN-v2)', fontsize=14, fontweight='bold')

plot_subplot(axes[0], df_self,  'Self')
plot_subplot(axes[1], df_cross, 'Cross')

# Unified legend (bars + baselines)
bar_patches = [mpatches.Patch(color=c, label=l)
               for c, l in zip(['#4C72B0', '#DD8452', '#55A868', '#C44E52'], METRIC_LABELS)]
baseline_handles = [
    plt.Line2D([0], [0], color='steelblue', linestyle='--', linewidth=1.5, label='LP s2 baseline'),
    plt.Line2D([0], [0], color='steelblue', linestyle=':',  linewidth=1.5, label='Zero-shot s1→s2'),
    plt.Line2D([0], [0], color='coral',     linestyle='--', linewidth=1.5, label='LP s1 baseline'),
    plt.Line2D([0], [0], color='coral',     linestyle=':',  linewidth=1.5, label='Zero-shot s2→s1'),
]
fig.legend(handles=bar_patches + baseline_handles,
           loc='lower center', ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.08, 1, 1])

out_path = os.path.join(ROOT, 'res/delulu/benv2/panopticon_shot.png')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved to {out_path}')
plt.show()

# python plotting/panopticon_shot.py