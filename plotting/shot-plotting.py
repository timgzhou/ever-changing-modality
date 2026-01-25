import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
date="jan25"
df = pd.read_csv(f"res/modality-transfer_{date}.csv")

# Filter to 32 epochs for consistency (exclude the 2-epoch entries)
df = df[df["num_supervised_epochs"] == 32]

# Get unique starting modalities
starting_modalities = df["starting_modality"].unique()
n_cols = len(starting_modalities)

# Create subplots - one per starting modality
fig, axes = plt.subplots(1, n_cols, figsize=(4 * n_cols, 5), sharey=True)
if n_cols == 1:
    axes = [axes]

for ax, start_mod in zip(axes, starting_modalities):
    subset = df[df["starting_modality"] == start_mod]

    sns.pointplot(
        data=subset,
        x="real_modality",
        y="test_acc",
        hue="eval_type",
        ax=ax,
        dodge=True,
        markers="o",
        linestyles="",
        errorbar=None,
    )

    ax.set_title(f"Starting: {start_mod}")
    ax.set_xlabel("Real Modality")
    ax.set_ylabel("Test Accuracy" if ax == axes[0] else "")
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')

    # Remove legend from individual subplots
    if ax.get_legend():
        ax.get_legend().remove()

plt.suptitle("Modality Transfer: Test Accuracy by Evaluation Type", y=1.02)
plt.tight_layout()

# Add shared legend below subplots
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Eval Type", loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=8)

plt.savefig(f"artifacts/shot-plotting_{date}.png", bbox_inches='tight', dpi=150)
print(f"Saved to artifacts/shot-plotting_{date}.png")

# python plotting/shot-plotting.py
