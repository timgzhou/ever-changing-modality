"""Compare Panopticon LP / zero-shot baselines vs DeluluNet transfer mAP on BEN-v2.

Run from repo root:
    python res/baselines/panoptucon-zeroshot/plot_panopticon_vs_delulu.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

CSV = 'res/baselines/panoptucon-zeroshot/benv2_panopticon_zeroshot.csv'
df = pd.read_csv(CSV)
full = df[df['train_fraction'] == 1.0].set_index('train_modality')

DIRECTIONS = [
    {'x': 0, 'xlabel': r'$M_A$=S2, $M_B$=S1',
     'lp': full.loc['s2', 'test_s2'], 'zs': full.loc['s2', 'test_s1'], 'del': 51.0},
    {'x': 1, 'xlabel': r'$M_A$=S1, $M_B$=S2',
     'lp': full.loc['s1', 'test_s1'], 'zs': full.loc['s1', 'test_s2'], 'del': 52.8},
]

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

pal   = sns.color_palette("Set2", 3)
C_DEL, C_LP, C_ZS = pal[1], pal[0], pal[2]
BAR_W = 0.38
HSPAN = 0.44
FS    = 11   # base font size

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(4.0, 3.5))
ax.set_title("")

for d in DIRECTIONS:
    x = d['x']

    ax.bar(x, d['del'], width=BAR_W, color=C_DEL, alpha=0.88, zorder=3)
    ax.text(x, d['del'] + 0.6, f"{d['del']:.1f}",
            ha='center', va='bottom', fontsize=FS, fontweight='bold', color=C_DEL)

    ax.hlines(d['lp'], x - HSPAN, x + HSPAN, colors=C_LP, linewidths=2.2, linestyles='-', zorder=2)
    ax.text(x - HSPAN + 0.02, d['lp'] + 0.5, f"{d['lp']:.1f}",
            ha='left', va='bottom', fontsize=FS - 1, color=C_LP)

    ax.hlines(d['zs'], x - HSPAN, x + HSPAN, colors=C_ZS, linewidths=2.2, linestyles='--', zorder=5)
    ax.text(x - HSPAN + 0.02, d['zs'] + 0.5, f"{d['zs']:.1f}",
            ha='left', va='bottom', fontsize=FS - 1, color=C_ZS)

ax.set_xticks([0, 1])
ax.set_xticklabels([d['xlabel'] for d in DIRECTIONS], fontsize=FS)
ax.set_xlim(-0.7, 1.7)
ax.set_ylabel("reBEN mAP", fontsize=FS)
ax.tick_params(axis='y', labelsize=FS - 1)
ax.set_ylim(0, max(d['lp'] for d in DIRECTIONS) * 1.2)
ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5)

legend_handles = [
    mpatches.Patch(color=C_DEL, alpha=0.88, label=r'DeluluNet Transfer ($M_A$→$M_B$)'),
    plt.Line2D([0], [0], color=C_LP, linewidth=2.2, linestyle='-',  label=r'Panopticon Supervised on $M_A$'),
    plt.Line2D([0], [0], color=C_ZS, linewidth=2.2, linestyle='--', label=r'Panopticon Zero-Shot on $M_B$'),
]
ax.legend(handles=legend_handles, fontsize=FS - 1.5, framealpha=0.9,
          loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=1)

fig.tight_layout()

out = 'res/baselines/panoptucon-zeroshot/panopticon_vs_delulu.pdf'
fig.savefig(out, bbox_inches='tight')
print(f'Saved to {out}')
