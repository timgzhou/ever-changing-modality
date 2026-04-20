"""Visualization of sweep_results_nomae.csv.

Compares test metrics grouped by asym_lr and dyn_teacher.
Saves to res/delulu-sweep/artifacts/sweep_viz_nomae.png.

Run from repo root:
    python res/delulu-sweep/sweep_viz_nomae.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

TEST_METRICS = ['test_transfer', 'test_peeking', 'test_addition', 'test_ens_addition']

df = pd.read_csv('res/delulu-sweep/sweep_results_nomae.csv')
df['asym_lr'] = df['asym_lr'].astype(float)
df['dyn_teacher'] = df['dyn_teacher'].astype(str)

# Load old sweep (sweep_results.csv) and add as asym_lr=1, dyn_teacher=False baseline
df_old = pd.read_csv('res/delulu-sweep/sweep_results.csv')
df_old = df_old[TEST_METRICS].copy()
df_old['asym_lr'] = 1.0
df_old['dyn_teacher'] = 'False'

df = pd.concat([df_old, df], ignore_index=True)

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
fig.suptitle('Sweep (nomae): test metrics by asym_lr (top) and dyn_teacher (bottom)', fontsize=12)

rng = np.random.default_rng(0)

def strip_box(ax, groups, values_by_group, xlabel):
    positions = range(len(groups))
    data = [values_by_group[g] for g in groups]
    bp = ax.boxplot(data, positions=positions, widths=0.4, patch_artist=True,
                    medianprops=dict(color='black', linewidth=1.5),
                    boxprops=dict(facecolor='lightsteelblue', alpha=0.6),
                    whiskerprops=dict(linewidth=1), capprops=dict(linewidth=1),
                    flierprops=dict(marker=''))
    for i, (g, vals) in enumerate(zip(groups, data)):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter([i + j for j in jitter], vals, s=15, alpha=0.7, zorder=3, color='steelblue')
    ax.set_xticks(list(positions))
    ax.set_xticklabels([str(g) for g in groups])
    ax.set_xlabel(xlabel, fontsize=9)

# Row 0: grouped by asym_lr
asym_groups = sorted(df['asym_lr'].unique())
for ax, metric in zip(axes[0], TEST_METRICS):
    vals = {g: df[df['asym_lr'] == g][metric].dropna().tolist() for g in asym_groups}
    strip_box(ax, asym_groups, vals, 'asym_lr')
    ax.set_title(metric.replace('test_', ''), fontsize=10)
    ax.set_ylabel('test metric', fontsize=8)

# Row 1: grouped by dyn_teacher
dt_groups = sorted(df['dyn_teacher'].unique())
for ax, metric in zip(axes[1], TEST_METRICS):
    vals = {g: df[df['dyn_teacher'] == g][metric].dropna().tolist() for g in dt_groups}
    strip_box(ax, dt_groups, vals, 'dyn_teacher')
    ax.set_title(metric.replace('test_', ''), fontsize=10)
    ax.set_ylabel('test metric', fontsize=8)

plt.tight_layout()
os.makedirs('res/delulu-sweep/artifacts', exist_ok=True)
out = 'res/delulu-sweep/artifacts/sweep_viz_nomae.png'
plt.savefig(out, dpi=150)
print(f'Saved to {out}')
