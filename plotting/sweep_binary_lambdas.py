"""
Plot sweep results where lambda_mae, lambda_latent, lambda_prefusion are binary (0 or 1).
Groups by dataset × starting_modality × new_modality and plots four test metrics.

Usage: python plotting/sweep_binary_lambdas.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

df = pd.read_csv("res/sweep_results.csv")

# Filter rows where the three lambdas are strictly 0.0 or 1.0
binary_cols = ["lambda_mae", "lambda_latent", "lambda_prefusion"]
mask = df[binary_cols].isin([0.0, 1.0]).all(axis=1)
df = df[mask].copy()
print(f"Filtered to {len(df)} rows with binary lambdas (from {mask.sum()} matching)")

# Build a readable label for the lambda combination
def lambda_label(row):
    parts = []
    for col, short in [("lambda_mae", "mae"), ("lambda_latent", "lat"), ("lambda_prefusion", "pre")]:
        parts.append(f"{short}={'1' if row[col] == 1.0 else '0'}")
    return "+".join(parts)

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
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    fig.suptitle(
        f"{dataset}  |  {start_mod} → +{new_mod}\n"
        f"(n={len(grp)}, binary lambda_mae/latent/prefusion only)",
        fontsize=12,
    )

    palette = sns.color_palette("tab10", n_colors=grp["lambda_combo"].nunique())
    combos = sorted(grp["lambda_combo"].unique())
    color_map = {c: palette[i] for i, c in enumerate(combos)}

    for ax, metric in zip(axes, test_metrics):
        for combo, sub in grp.groupby("lambda_combo"):
            ax.scatter(
                [combo] * len(sub),
                sub[metric],
                color=color_map[combo],
                alpha=0.75,
                s=60,
                label=combo,
            )
            ax.axhline(sub[metric].mean(), color=color_map[combo], linewidth=1.2,
                       linestyle="--", alpha=0.6)

        ax.set_title(metric_labels[metric])
        ax.set_xlabel("λ combo")
        ax.set_ylabel("metric (%)")
        ax.set_xticks(range(len(combos)))
        ax.set_xticklabels(combos, rotation=35, ha="right", fontsize=7)

    plt.tight_layout()
    fname = f"artifacts/sweep_binary_{dataset}_{start_mod}_{new_mod}.png"
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"Saved {fname}")
    plt.close()

# python plotting/sweep_binary_lambdas.py
