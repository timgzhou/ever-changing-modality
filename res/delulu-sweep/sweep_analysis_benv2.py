"""BEN-v2 sweep analysis — 5-config comparison and val-vs-test scatter.

Run from repo root:
    python -u res/delulu-sweep/sweep_analysis_benv2.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



"""
CONFIGURING THE PLOTS
"""

STARTMOD = "s1"
NEWMOD   = "s2"
K        = 1   # fraction of top runs to keep per config (by val score); 1.0 = all runs
TRIM     = 0    # number of top+bottom points to trim per config (by test metric); 0 = no trim

df = pd.read_csv('res/delulu-sweep/sweep_results_benv2_final.csv')
print(f"Rows: {len(df)}")

for col in ['protect_lrm', 'use_mask_token', 'latent_masked_only']:
    df[col] = df[col].astype(str).str.strip().str.lower().map({'true': True, 'false': False})

out_dir = 'res/delulu-sweep/artifacts'
os.makedirs(out_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Config classification
# ---------------------------------------------------------------------------

def classify_config(row) -> str:
    al = str(row["active_losses"])
    mt = str(row["use_mask_token"]).strip().lower()
    if mt == "true":
        return "mask-token"
    if "prefusion" not in al and "latent" not in al:
        return "other"
    if "prefusion" not in al:
        return "no-prefusion"
    if "latent" not in al:
        return "no-latent"
    if "ce" not in al:
        return "no-batch-mixing"
    return "delulu"


CONFIG_ORDER   = ["delulu", "mask-token", "no-prefusion", "no-latent", "no-batch-mixing"]
CONFIG_DISPLAY = ["delulu", "mask\ntoken", "no\nprefusion", "no\nlatent", "no batch\nmixing"]
METRICS        = ["test_transfer", "test_peeking", "test_addition"]
METRIC_LABELS  = ["Transfer", "Peek", "Addition"]
COLORS         = plt.cm.tab10.colors

df["_cfg"] = df.apply(classify_config, axis=1)

sub = df[(df["starting_modality"] == STARTMOD) & (df["new_modality"] == NEWMOD)]
rng = np.random.default_rng(0)

# val score used for top-K selection: peeking for peeking, composite score for transfer/addition
VAL_SEL = "val_peeking"

def topk(group_df, k=K, val_col=VAL_SEL):
    """Return top k fraction of rows by val_col (k=1.0 returns all)."""
    if k >= 1.0:
        return group_df
    n = max(1, int(len(group_df) * k))
    return group_df.nlargest(n, val_col)

def trim(vals, t=TRIM):
    """Drop the t lowest and t highest values."""
    if t <= 0 or len(vals) <= 2 * t:
        return vals
    return np.sort(vals)[t:-t]

# ---------------------------------------------------------------------------
# Plot 1 — scatter with mean diamond  (sweep_benv2_5configs.png)
# ---------------------------------------------------------------------------

fig1, axes1 = plt.subplots(1, 3, figsize=(11, 4.2), sharey=False)
fig1.suptitle(f"BEN-v2 Sweep Config Comparison ({STARTMOD}→{NEWMOD})", fontsize=12, fontweight="bold")

for ax, metric, mlabel in zip(axes1, METRICS, METRIC_LABELS):
    for ci, cfg in enumerate(CONFIG_ORDER):
        rows = trim(topk(sub[sub["_cfg"] == cfg])[metric].dropna().values)
        if len(rows) == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=len(rows))
        ax.scatter(ci + jitter, rows, color=COLORS[ci], s=18, alpha=0.45, linewidths=0)
        mean = rows.mean()
        ax.scatter(ci, mean, color=COLORS[ci], s=80, marker="D",
                   edgecolors="black", linewidths=0.8, zorder=5)
        ax.text(ci, mean + 0.25, f"{mean:.1f}",
                ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_xticks(np.arange(len(CONFIG_ORDER)))
    ax.set_xticklabels(CONFIG_DISPLAY, fontsize=9)
    ax.set_title(mlabel, fontsize=11)
    ax.set_ylabel("mAP", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.margins(x=0.08)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

fig1.tight_layout()
out1 = f'{out_dir}/sweep_benv2_5configs.png'
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f'Saved to {out1}')

# ---------------------------------------------------------------------------
# LaTeX table for Plot 1
# ---------------------------------------------------------------------------

ROW_NAMES = ["DeluluNet (ours)", r"$+$ Masking Token", r"$-$ Prefusion MSE", r"$-$ Latent MSE", r"$-$ Batch Mixing"]

# collect means: means[cfg][metric]
means = {}
for cfg, metric in [(c, m) for c in CONFIG_ORDER for m in METRICS]:
    vals = trim(topk(sub[sub["_cfg"] == cfg])[metric].dropna().values)
    means.setdefault(cfg, {})[metric] = vals.mean() if len(vals) > 0 else float('nan')

# baseline = delulu row
baseline = means["delulu"]

def fmt_cell(cfg, metric):
    m = means[cfg][metric]
    base = baseline[metric]
    diff = m - base
    if cfg == "delulu":
        return f"{m:.1f}"
    sign = "+" if diff >= 0 else ""
    color = "\\cellcolor{green!15}" if diff >= 0 else "\\cellcolor{red!15}"
    return f"{color}{m:.1f} ({sign}{diff:.1f})"

lines = []
lines.append(r"\begin{table}[h]")
lines.append(r"\centering")
lines.append(r"\setlength{\tabcolsep}{6pt}")
lines.append(r"\begin{tabular}{lccc}")
lines.append(r"\toprule")
lines.append(r"Configuration & Transfer & Peek & Addition \\")
lines.append(r"\midrule")
for cfg, row_name in zip(CONFIG_ORDER, ROW_NAMES):
    cells = " & ".join(fmt_cell(cfg, m) for m in METRICS)
    lines.append(f"{row_name} & {cells} \\\\")
lines.append(r"\bottomrule")
lines.append(r"\end{tabular}")
lines.append(r"\caption{\textbf{Delulu Components Ablation on reBEN for Masking Transformer, MSE losses, and Batch Mixing} (start with S1, introduce S2), deltas relative to DeluluNet (ours).}")
lines.append(r"\end{table}")

tex = "\n".join(lines)
tex_path = f'{out_dir}/sweep_benv2_5configs_table.tex'
with open(tex_path, 'w') as f:
    f.write(tex)
print(f'Saved to {tex_path}')
print(tex)

# ---------------------------------------------------------------------------
# Plot 1b — boxplot with mean ± std  (sweep_benv2_5configs_box.png)
# ---------------------------------------------------------------------------

fig1b, axes1b = plt.subplots(1, 3, figsize=(11, 4.2), sharey=False)
fig1b.suptitle(f"BEN-v2 Sweep Config Comparison ({STARTMOD}→{NEWMOD})", fontsize=12, fontweight="bold")

for ax, metric, mlabel in zip(axes1b, METRICS, METRIC_LABELS):
    group_data = [trim(topk(sub[sub["_cfg"] == cfg])[metric].dropna().values)
                  for cfg in CONFIG_ORDER]

    bp = ax.boxplot(
        group_data, positions=np.arange(len(CONFIG_ORDER)), widths=0.5,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=1.5),
        whiskerprops=dict(linewidth=1),
        capprops=dict(linewidth=1),
        flierprops=dict(marker=''),
    )
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)

    ax.set_xticks(np.arange(len(CONFIG_ORDER)))
    ax.set_xticklabels(CONFIG_DISPLAY, fontsize=9)
    ax.set_title(mlabel, fontsize=11)
    ax.set_ylabel("mAP", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.margins(x=0.08)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.5)

    # annotate mean±std inside top of each box
    for ci, vals in enumerate(group_data):
        if len(vals) == 0:
            continue
        mean, std = vals.mean(), vals.std(ddof=1) if len(vals) > 1 else 0.0
        ypos = np.percentile(vals, 75)
        ax.text(ci, ypos, f"{mean:.1f}±{std:.1f}",
                ha="center", va="bottom", fontsize=7.5, fontweight="bold")

out1b = f'{out_dir}/sweep_benv2_5configs_box.png'
fig1b.savefig(out1b, dpi=150, bbox_inches="tight")
print(f'Saved to {out1b}')

# ---------------------------------------------------------------------------
# Plot 2 — val vs test scatter  (sweep_benv2_val_vs_test.png)
# ---------------------------------------------------------------------------

VAL_TEST_PAIRS = [
    ("val_transfer", "test_transfer", "Transfer", "Val score (peek×agree/100)"),
    ("val_peeking",  "test_peeking",  "Peek",     "Val mAP"),
    ("val_addition", "test_addition", "Addition", "Val score (peek×agree/100)"),
]

df5 = topk(sub[sub["_cfg"] == "delulu"]).copy()

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4.2))
fig2.suptitle(f"BEN-v2 Final Sweep — Val vs Test mAP ({STARTMOD}→{NEWMOD}, delulu config)", fontsize=12, fontweight="bold")

for ax, (val_col, test_col, mlabel, val_xlabel) in zip(axes2, VAL_TEST_PAIRS):
    for col in [val_col, test_col]:
        df5[col] = pd.to_numeric(df5[col], errors="coerce")

    rows = df5.dropna(subset=[val_col, test_col])
    ax.scatter(rows[val_col], rows[test_col], color=COLORS[0], s=22, alpha=0.75, linewidths=0)

    if val_col == "val_peeking":
        all_vals = pd.concat([df5[val_col], df5[test_col]]).dropna()
        lo, hi = all_vals.min() - 1, all_vals.max() + 1
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8)

    r = df5[val_col].corr(df5[test_col])
    ax.set_title(f"{mlabel}  (r = {r:.3f})", fontsize=10)
    ax.set_xlabel(val_xlabel, fontsize=9)
    ax.set_ylabel("Test mAP", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.45)

fig2.tight_layout()

out2 = f'{out_dir}/sweep_benv2_val_vs_test.png'
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f'Saved to {out2}')

# python res/delulu-sweep/sweep_analysis_benv2.py