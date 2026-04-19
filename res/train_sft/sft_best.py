import json
import os
import pandas as pd

csvs = {
    "eurosat": "res/train_sft/eurosat.csv",
    "benv2":   "res/train_sft/benv2.csv",
    "dfc2020": "res/train_sft/dfc2020.csv",
}

frames = []
for dataset, path in csvs.items():
    if not os.path.isfile(path):
        print(f"[warn] missing: {path}")
        continue
    frames.append(pd.read_csv(path))

df = pd.concat(frames, ignore_index=True)

group_keys = ["dataset", "modality", "model_type", "train_mode", "dino_init"]

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

teachers = best[best["dino_init"] == True].copy()
lookup = {}
for _, row in teachers.iterrows():
    key = f"{row['dataset']}/{row['modality']}/{row['model_type']}"
    lookup[key] = {
        "dataset":     row["dataset"],
        "modality":    row["modality"],
        "model_type":  row["model_type"],
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
