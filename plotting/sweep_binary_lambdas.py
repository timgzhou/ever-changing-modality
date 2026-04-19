"""
Plot sweep results where lambda_mae, lambda_latent, lambda_prefusion are binary (0 or 1).
Groups by dataset × starting_modality × new_modality and plots four test metrics.

Usage: python plotting/sweep_binary_lambdas.py [--dataset benv2]
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default=None, help="Filter to a specific dataset")
parser.add_argument("--starting", type=str, default=None, help="Filter to a specific starting modality")
parser.add_argument("--newmod", type=str, default=None, help="Filter to a specific new modality")
args = parser.parse_args()

df = pd.read_csv("res/sweep_results.csv")

if args.dataset:
    df = df[df["dataset"] == args.dataset].copy()
    print(f"Filtered to dataset={args.dataset}: {len(df)} rows")
if args.starting:
    df = df[df["starting_modality"] == args.starting].copy()
    print(f"Filtered to starting_modality={args.starting}: {len(df)} rows")
if args.newmod:
    df = df[df["new_modality"] == args.newmod].copy()
    print(f"Filtered to new_modality={args.newmod}: {len(df)} rows")

# Filter rows where the three lambdas are strictly 0.0 or 1.0
binary_cols = ["lambda_mae", "lambda_latent", "lambda_prefusion"]
mask = df[binary_cols].isin([0.0, 1.0]).all(axis=1)
df = df[mask].copy()
print(f"Filtered to {len(df)} rows with binary lambdas")

# Shorten lambda combo label: e.g. "mae=1+lat=0+pre=1" -> "M·L̶·P"
def lambda_label(row):
    parts = []
    for col, short in [("lambda_mae", "mae"), ("lambda_latent", "lat"), ("lambda_prefusion", "pre")]:
        parts.append(f"{short}={'1' if row[col] == 1.0 else '0'}")
    return "\n".join(parts)

df["lambda_combo"] = df.apply(lambda_label, axis=1)

test_metrics = ["test_transfer", "test_peeking", "test_addition", "test_ens_addition"]
metric_labels = {
    "test_transfer":     "Transfer",
    "test_peeking":      "Peeking",
    "test_addition":     "Addition",
    "test_ens_addition": "Ens. Addition",
}

groups = df.groupby(["dataset", "starting_modality", "new_modality"])

for (dataset, start_mod, new_mod), grp in groups:
    combos = sorted(grp["lambda_combo"].unique())
    n_combos = len(combos)
    palette = sns.color_palette("tab10", n_colors=n_combos)
    color_map = {c: palette[i] for i, c in enumerate(combos)}

    # Teacher baseline: constant across runs for this group
    teacher_metric = grp["teacher_test_metric"].dropna()
    teacher_val = teacher_metric.iloc[0] if len(teacher_metric) else None

    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=False)
    fig.suptitle(
        f"{dataset}  |  {start_mod} → +{new_mod}   (n={len(grp)}, binary λ_mae/lat/pre)",
        fontsize=12,
    )

    for ax, metric in zip(axes, test_metrics):
        combo_order = list(range(n_combos))
        combo_to_x = {c: i for i, c in enumerate(combos)}

        means = []
        for combo, sub in grp.groupby("lambda_combo"):
            x = combo_to_x[combo]
            vals = sub[metric].dropna()
            if vals.empty:
                means.append((x, None))
                continue
            # Jitter x slightly for visibility
            import numpy as np
            jitter = np.random.uniform(-0.15, 0.15, size=len(vals))
            ax.scatter(
                x + jitter,
                vals,
                color=color_map[combo],
                alpha=0.55,
                s=35,
                zorder=3,
            )
            mean_val = vals.mean()
            means.append((x, mean_val))
            ax.plot([x - 0.3, x + 0.3], [mean_val, mean_val],
                    color=color_map[combo], linewidth=2.0, zorder=4)
            ax.annotate(f"{mean_val:.1f}", xy=(x, mean_val),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=7, color=color_map[combo])

        # Teacher baseline (only shown on transfer panel, or all panels)
        if teacher_val is not None:
            ax.axhline(teacher_val, color="black", linewidth=1.0,
                       linestyle=":", alpha=0.7, label=f"teacher ({teacher_val:.1f})")
            ax.annotate(f"teacher\n{teacher_val:.1f}", xy=(n_combos - 0.5, teacher_val),
                        xytext=(2, 4), textcoords="offset points",
                        fontsize=7, color="black", alpha=0.8)

        ax.set_title(metric_labels[metric], fontsize=11)
        ax.set_xlabel("")
        ax.set_ylabel("mAP (%)" if dataset == "benv2" else "metric (%)", fontsize=9)
        ax.set_xticks(combo_order)
        ax.set_xticklabels(combos, fontsize=7)
        ax.tick_params(axis='x', which='both', length=0)
        sns.despine(ax=ax)

    plt.tight_layout()
    fname = f"artifacts/sweep_binary_{dataset}_{start_mod}_{new_mod}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname}")
    plt.close()

# python plotting/sweep_binary_lambdas.py --dataset benv2 --starting s2 --newmod s1
