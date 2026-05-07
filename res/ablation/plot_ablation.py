import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

METRICS = ["best_transfer", "best_peeking", "best_addition"]
FUSION_METRICS = ["test_transfer", "test_peeking", "test_addition"]
METRIC_LABELS = ["Transfer", "Peek", "Addition"]

def get_group(df, label):
    df = df.copy()
    for col in METRICS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.assign(group=label)

def build_groups(df_losses, df_mask, df_no_ce, df_protect_lrm, start_mod, new_mod):
    def filt(df):
        return df[(df["starting_modality"] == start_mod) & (df["new_modality"] == new_mod)]

    def filt_full(df):
        sub = filt(df)
        sub = sub[sub["active_losses"] == "latent+prefusion+distill+ce"]
        # fix tz_fusion_time to 3 to match all other ablation conditions
        sub = sub[pd.to_numeric(sub["tz_fusion_time"], errors="coerce") == 3]
        return sub

    full         = get_group(filt_full(df_losses),                                                          "full")
    no_pre       = get_group(filt(df_losses)[filt(df_losses)["active_losses"] == "latent+distill+ce"],      "-prefusion")
    no_latent    = get_group(filt(df_losses)[filt(df_losses)["active_losses"] == "prefusion+distill+ce"],   "-latent")
    no_mask      = get_group(filt(df_mask),                                                                  "-masking\ntransformer")
    no_ce        = get_group(filt(df_no_ce),                                                                 "-batch_mixing")
    protect_lrm  = get_group(filt(df_protect_lrm),                                                          "+protect_lrm")
    return [full, no_pre, no_latent, no_mask, no_ce, protect_lrm]

def plot_direction(axes, groups, title_prefix):
    group_labels = [g["group"].iloc[0] for g in groups if not g.empty]
    groups = [g for g in groups if not g.empty]

    means = {lbl: [] for lbl in group_labels}
    stds  = {lbl: [] for lbl in group_labels}
    for g in groups:
        lbl = g["group"].iloc[0]
        for col in METRICS:
            vals = g[col].dropna().values
            means[lbl].append(np.mean(vals) if len(vals) > 0 else float("nan"))
            stds[lbl].append(np.std(vals)  if len(vals) > 0 else 0.0)

    colors = plt.cm.tab10.colors
    n_groups = len(group_labels)
    x = np.arange(n_groups)
    width = 0.65

    for ax, col, metric_label in zip(axes, METRICS, METRIC_LABELS):
        y    = [means[lbl][METRICS.index(col)] for lbl in group_labels]
        yerr = [stds[lbl][METRICS.index(col)]  for lbl in group_labels]

        ax.bar(x, y, width, yerr=yerr, capsize=4,
               color=colors[:n_groups], alpha=0.8, edgecolor="black", linewidth=0.6,
               error_kw=dict(elinewidth=1.2, ecolor="black"))

        for xi, (yi, ei) in enumerate(zip(y, yerr)):
            if not np.isnan(yi):
                ax.text(xi, yi + ei + 0.15, f"{yi:.1f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_title(f"{title_prefix} — {metric_label}", fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(group_labels, fontsize=8.5, rotation=20, ha="right")
        ax.set_ylabel("mAP" if col == METRICS[0] else "", fontsize=9)
        ax.tick_params(axis="y", labelsize=8)
        ax.margins(x=0.08)
        valid_y = [v for v in y if not np.isnan(v)]
        valid_e = [e for v, e in zip(y, yerr) if not np.isnan(v)]
        if valid_y:
            ax.set_ylim(min(valid_y) - max(valid_e) - 2, max(valid_y) + max(valid_e) + 2)


MOD_DISPLAY = {"s2": "S2", "s1": "S1", "rgb": "RGB", "vre": "VRE"}

def plot_fusion_time(df_fusion, directions):
    import seaborn as sns
    palette = sns.color_palette("Set2", n_colors=len(FUSION_METRICS))

    available = []
    for start, new in directions:
        sub = df_fusion[(df_fusion["starting_modality"] == start) & (df_fusion["new_modality"] == new)].copy()
        for col in FUSION_METRICS:
            sub[col] = pd.to_numeric(sub[col], errors="coerce")
        sub["tz_fusion_time"] = pd.to_numeric(sub["tz_fusion_time"], errors="coerce")
        sub = sub.dropna(subset=["tz_fusion_time"])
        if not sub.empty:
            available.append((start, new, sub))

    if not available:
        return

    n_rows = len(available)
    fig, axes = plt.subplots(n_rows, 1, figsize=(5.5, 3.8 * n_rows), squeeze=False)

    for row_idx, (start, new, sub) in enumerate(available):
        ax = axes[row_idx][0]
        grouped = sub.groupby("tz_fusion_time")[FUSION_METRICS]
        means = grouped.mean()
        stds  = grouped.std().fillna(0)
        x = means.index.values

        start_d = MOD_DISPLAY.get(start, start.upper())
        new_d   = MOD_DISPLAY.get(new,   new.upper())

        for col, metric_label, color in zip(FUSION_METRICS, METRIC_LABELS, palette):
            y    = means[col].values
            yerr = stds[col].values
            ax.errorbar(x, y, yerr=yerr, fmt="o-", capsize=4, linewidth=1.8,
                        markersize=6, color=color, ecolor=color, elinewidth=1.2,
                        label=metric_label)

        f0 = pd.to_numeric(sub["teacher_test_metric"], errors="coerce").mean()
        if not np.isnan(f0):
            ax.axhline(f0, color="gray", linewidth=1.4, linestyle="--")
            ax.text(x[0], f0 - 0.8, f"DINOv3 SFT ({start_d} $f_0$): {f0:.1f}",
                    va="top", ha="left", fontsize=8, color="gray")

        ax.axhline(49.2, color="gray", linewidth=1.4, linestyle="--")
        ax.text(x[0], 49.2 - 0.8, f"DINOv3 SFT ({new_d} label oracle): 49.2",
                va="top", ha="left", fontsize=8, color="gray")

        ax.set_title(f"reBEN  ||  Start: {start_d}, New: {new_d}", fontsize=10)
        ax.set_xlabel("Fusion Layer", fontsize=9)
        ax.set_ylabel("mAP", fontsize=9)
        ax.set_xticks(x)
        ax.tick_params(labelsize=8)
        ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.7)
        ax.legend(fontsize=8, framealpha=0.8)

    plt.tight_layout()
    out = "res/ablation/ablation_benv2_fusion_time.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")


# --- Load data ---
df_losses      = pd.read_csv("res/ablation/benv2_active_losses.csv")
df_mask        = pd.read_csv("res/ablation/benv2_mask_token.csv")
df_no_ce       = pd.read_csv("res/ablation/benv2_no_ce.csv")
df_protect_lrm = pd.read_csv("res/ablation/benv2_protect_lrm.csv")
df_fusion_time = pd.read_csv("res/ablation/benv2_fusion_time.csv")

directions = [("s1", "s2")]

# --- Ablation bar plot ---
available = []
for start, new in directions:
    grps = build_groups(df_losses, df_mask, df_no_ce, df_protect_lrm, start, new)
    has_data = any(not g.empty for g in grps)
    if has_data:
        available.append((start, new, grps))

n_rows = len(available)
fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.8 * n_rows), squeeze=False)
fig.suptitle("BEN-v2 Ablation (mAP)", fontsize=12, fontweight="bold")

for row_idx, (start, new, grps) in enumerate(available):
    label = f"{start}→{start}+{new}"
    plot_direction(axes[row_idx], grps, label)

plt.tight_layout()
out = "res/ablation/ablation_benv2.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"saved {out}")

# --- Fusion time line plot ---
plot_fusion_time(df_fusion_time, directions)

# python res/ablation/plot_ablation.py
