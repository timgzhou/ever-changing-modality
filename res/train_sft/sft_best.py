import json
import os
import pandas as pd

# dfc2020 has two incompatible split definitions; dfc2020_cobench.csv is the
# Copernicus-Bench official split (8 classes) and dfc2020.csv is ROI-disjoint
# (10 classes). Both are read, and the split is carried into the teacher key so
# a cobench teacher can never be handed to an ROI-disjoint run.
csvs = {
    "eurosat": "res/train_sft/eurosat.csv",
    "benv2":   "res/train_sft/benv2.csv",
    "dfc2020": "res/train_sft/dfc2020.csv",
    "dfc2020_cobench": "res/train_sft/dfc2020_cobench.csv",
}

frames = []
for dataset, path in csvs.items():
    if not os.path.isfile(path):
        print(f"[warn] missing: {path}")
        continue
    f = pd.read_csv(path)
    # the CSV's own `dataset` column says "dfc2020" for both splits; override it
    # with the registry name so the two never collide in the lookup
    f["dataset"] = dataset
    if "decoder" not in f.columns:
        f["decoder"] = "linear"
    frames.append(f)

df = pd.concat(frames, ignore_index=True)

# decoder is part of the identity: an upernet teacher and a linear teacher are
# different models and are not interchangeable (different head, ~10 mIoU apart).
# train_split is part of the identity. Stage-1 methods (shot_ete, distillation,
# MKE) use train2 as their UNLABELED pool, so a `full` teacher -- supervised on
# train1+train2 -- has already seen that pool with labels and leaks. Those
# methods must take a split1 teacher; `full` is only valid for the supervised
# upper-bound row.
group_keys = ["dataset", "modality", "model_type", "train_mode", "dino_init",
              "decoder", "train_split"]

counts = df.groupby(group_keys).size().reset_index(name="n_runs")

best = (
    df.sort_values("val_metric", ascending=False)
    .groupby(group_keys, as_index=False)
    .first()[group_keys + ["learning_rate", "trainable_params", "metric_name", "val_metric", "test_metric", "saved_checkpoint"]]
    .merge(counts, on=group_keys)
    .sort_values(group_keys)
)

for dataset, group in best.groupby("dataset"):
    print(f"\n=== {dataset} ===")
    print(group.drop(columns="dataset").to_string(index=False))

# ---------------------------------------------------------------------------
# Save teacher lookup: best dino_init checkpoint per (dataset, modality, model_type)
# Key: "dataset/modality/model_type" — used by bash to select teacher checkpoint.
# ---------------------------------------------------------------------------

# Best run per (dataset, modality, model, decoder) by val, regardless of
# dino_init. Filtering to dino_init==True was right when DINO always won, but
# with the upernet decoder the best s1 and s2_norgb teachers are dino_init=False.
teachers = (
    best.sort_values("val_metric", ascending=False)
        .groupby(["dataset", "modality", "model_type", "decoder", "train_split"],
                 as_index=False)
        .first()
)
lookup = {}
for _, row in teachers.iterrows():
    key = (f"{row['dataset']}/{row['modality']}/{row['model_type']}/"
           f"{row['decoder']}/{row['train_split']}")
    lookup[key] = {
        "dataset":     row["dataset"],
        "modality":    row["modality"],
        "model_type":  row["model_type"],
        "decoder":     row["decoder"],
        "train_split": row["train_split"],
        "dino_init":   bool(row["dino_init"]),
        "val_metric":  round(float(row["val_metric"]), 4),
        "test_metric": round(float(row["test_metric"]), 4),
        "metric_name": row["metric_name"],
        "checkpoint":  row["saved_checkpoint"],
    }

os.makedirs("artifacts", exist_ok=True)
out = "artifacts/sft_teachers.json"
with open(out, "w") as f:
    json.dump(lookup, f, indent=2, sort_keys=True)

print(f"\nSaved {len(lookup)} teacher checkpoints to {out}")
for key, v in sorted(lookup.items()):
    print(f"  {key:<45}  {v['metric_name']} val={v['val_metric']:.2f} test={v['test_metric']:.2f}  {v['checkpoint']}")

# python res/train_sft/sft_best.py
