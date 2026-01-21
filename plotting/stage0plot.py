import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

stage0_res = pd.read_csv("res/legacy/train_stage0_res.csv")
stage0_res = stage0_res[22:]
# Filter out lora with rank != 32
stage0_res = stage0_res[~((stage0_res["modality_specific_layer_augmenter"] == "lora") & (stage0_res["tz_lora_rank"] != 32))]
stage0_res = stage0_res[(stage0_res["modality"] != "aw")]
# Filter out fft+lora (adaptor mode with fft augmenter)
stage0_res = stage0_res[~((stage0_res["train_mode"] == "adaptor") & (stage0_res["modality_specific_layer_augmenter"] == "fft"))]

stage0_res["capacity"] = 'placeholder'

# FFT mode
stage0_res.loc[stage0_res["train_mode"] == "fft", "capacity"] = "FFT(upperbound)"

# Adaptor mode with fft augmenter
mask_adaptor_fft = (stage0_res["train_mode"] == "adaptor") & (stage0_res["modality_specific_layer_augmenter"] == "fft")
stage0_res.loc[mask_adaptor_fft, "capacity"] = "fft+lora" + stage0_res.loc[mask_adaptor_fft, "tz_lora_rank"].astype(str)

# Adaptor mode with lora augmenter
mask_adaptor_lora = (stage0_res["train_mode"] == "adaptor") & (stage0_res["modality_specific_layer_augmenter"] == "lora")
stage0_res.loc[mask_adaptor_lora, "capacity"] = "lora" + stage0_res.loc[mask_adaptor_lora, "tz_lora_rank"].astype(str)

# Probe mode
stage0_res.loc[stage0_res["train_mode"] == "probe", "capacity"] = "probe"

# Emb+probe mode
stage0_res.loc[stage0_res["train_mode"] == "emb+probe", "capacity"] = "emb+probe"

# Define custom order: probe -> emb+probe -> lora (sorted by rank) -> fft+lora (sorted by rank) -> FFT(upperbound)
def capacity_sort_key(cap):
    if cap == "probe":
        return (0, 0)
    elif cap == "emb+probe":
        return (1, 0)
    elif cap.startswith("lora"):
        rank = int(cap.replace("lora", ""))
        return (2, rank)
    elif cap.startswith("fft+lora"):
        rank = int(cap.replace("fft+lora", ""))
        return (3, rank)
    elif cap == "FFT(upperbound)":
        return (4, 0)
    else:
        return (5, 0)

# Get unique capacities and sort them
unique_capacities = sorted(stage0_res["capacity"].unique(), key=capacity_sort_key)

# Convert to categorical with custom order
stage0_res["capacity"] = pd.Categorical(stage0_res["capacity"], categories=unique_capacities, ordered=True)

# Create hue column based on first index of capacity_sort_key (category group)
hue_map = {0: "probe", 1: "emb+probe", 2: "lora", 3: "fft+lora", 4: "FFT(upperbound)"}
stage0_res["capacity_group"] = stage0_res["capacity"].apply(lambda x: hue_map[capacity_sort_key(x)[0]])

# Get probe accuracy for each modality (for horizontal reference lines)
probe_acc_by_modality = stage0_res[stage0_res["capacity"] == "probe"].groupby("modality")["test_accuracy"].mean().to_dict()

# Get unique modalities
modalities = stage0_res["modality"].unique()
n_modalities = len(modalities)

# Create subplots - one per modality, arranged horizontally (narrower)
fig, axes = plt.subplots(1, n_modalities, figsize=(1.5 * n_modalities, 2), sharey=False)
if n_modalities == 1:
    axes = [axes]

# Get Set2 colormap colors
set2_colors = plt.colormaps['Set2'].colors

# Track handles/labels for legend (only need one set since colors are consistent)
legend_handles = {}

for ax, modality in zip(axes, modalities):
    # Filter data for this modality
    modality_data = stage0_res[stage0_res["modality"] == modality]

    # Plot points on a single vertical line (x=0), colored by capacity
    for i, (cap, group) in enumerate(modality_data.groupby("capacity", observed=True)):
        cap_group = hue_map[capacity_sort_key(cap)[0]]
        color_idx = list(hue_map.values()).index(cap_group) % len(set2_colors)
        color = set2_colors[color_idx]
        scatter = ax.scatter([0] * len(group), group["test_accuracy"], label=cap, color=color, s=60, zorder=3, alpha=0.8)
        if cap_group not in legend_handles:
            legend_handles[cap_group] = scatter

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    # Add probe reference line for non-rgb modalities
    if modality != "rgb" and modality in probe_acc_by_modality:
        probe_acc = probe_acc_by_modality[modality]
        ax.axhline(y=probe_acc, color='gray', linestyle='--', alpha=0.7)
        ax.annotate("random\ntokenizer", xy=(0.5, probe_acc+0.5),
                    xycoords=('axes fraction', 'data'), fontsize=10, color='gray', va='bottom', ha='center')

    ax.set_title(modality)
    ax.set_ylabel("Test Accuracy \u2191" if ax == axes[0] else "")
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])  # Hide x-axis ticks since it's just a single vertical line

    # Set y-limits with 10% padding, upper bounded by 100
    data_min = modality_data["test_accuracy"].min()
    data_max = modality_data["test_accuracy"].max()
    data_range = data_max - data_min
    ymin = data_min - 0.1 * data_range
    ymax = min(data_max + 0.1 * data_range, 100)
    ax.set_ylim(ymin, ymax)

    # Set y-tick spacing: use 0.5 minimum, but allow coarser if data range is large
    if data_range < 3:  # Only apply finer ticks for small ranges
        ax.yaxis.set_major_locator(MultipleLocator(0.5))

# Adjust spacing between subplots
plt.subplots_adjust(wspace=0.6)  # increase for more horizontal space

# Add legend at bottom, one row
fig.legend(legend_handles.values(), legend_handles.keys(), loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=len(legend_handles))

fig.suptitle("Modality-Specific Capacity vs Test Accuracy by Modality",y=1.15)

plt.savefig("artifacts/stage0.png", bbox_inches='tight')

# python plotting/stage0plot