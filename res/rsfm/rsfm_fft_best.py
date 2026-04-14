import pandas as pd

df = pd.read_csv("res/rsfm/rsfm_results.csv")

fft = df[df["train_mode"] == "fft"]

best = (
    fft.sort_values("test_metric", ascending=False)
    .groupby(["model", "dataset", "modality"], as_index=False)
    .first()[["model", "dataset", "modality", "trainable_params", "test_metric", "metric_name"]]
    .sort_values(["model", "dataset", "modality"])
)

for dataset, group in best.groupby("dataset"):
    print(f"\n=== {dataset} ===")
    print(group.drop(columns="dataset").to_string(index=False))

# python res/rsfm/rsfm_fft_best.py