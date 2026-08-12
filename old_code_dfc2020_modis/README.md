# Archived DFC2020 results — MODIS-label era

Everything here was produced against the **wrong labels** and should not be
reported as DFC2020.

## What was wrong

The HuggingFace `GFM-Bench/DFC2020` packaging shipped the SEN12MS MODIS-derived
`lc` product as the segmentation target instead of the DFC2020 contest ground
truth. MODIS land cover is ~500 m native, so on a 96x96 tile (960 m) it resolves
to 2-3 blobs.

Measured over 300 test tiles with the old loader:

| metric | value |
|---|---|
| connected regions / tile | 2.3 |
| median region size | 2276 px |
| mean dominant-class fraction | **0.89** |

A model predicting a single constant class scores ~89% pixel accuracy. Reported
per-tile accuracies in the old figure (37.5 / 58.4 / 25.9) are *below* that
floor, and mIoU over 2-3 blob classes is extremely high-variance.

Additionally, the old loader sent Savanna (raw IGBP 8/9) to ignore_index,
discarding ~50% of the average tile — but Savanna is a **scored** class in the
real benchmark (DFC class 3).

## What replaced it

- Data: `datasets/DFC2020_official/DFC_Public_Dataset/` — official
  `DFC_Public_Dataset.zip` from IEEE DataPort competition 17534
  (10344429770 bytes, `unzip -t` clean), 6114 paired patches at 256x256,
  10 classes, 10 m semi-manual labels.
- Loader: `dfc2020_official_data_utils.py`
- Provenance figure: `figs/dfc2020_label_provenance.png`

## Contents

- `res/` — DFC2020-exclusive result CSVs, moved wholesale.
- `shared_csv_snapshots/` — DFC2020 rows **extracted** from CSVs that also hold
  benv2/eurosat/biomassters rows. The originals were left in place and still
  contain their dfc2020 rows; these are read-only snapshots for reference, not
  authoritative copies. Dedup logic in the launchers still reads the originals,
  so old dfc2020 rows there will continue to suppress resubmission — see below.
- `figs/` — the old `dfc2020_s2rgb_to_s2norgb.{png,pdf}` prediction figure.
- `checkpoints/` — 24 SFT/SHOT checkpoints (11 GB) with **8-class heads**.
  Unusable against the 10-class official labels; kept only for provenance.

## Before rerunning DFC2020 jobs

1. `artifacts/sft_teachers.json` has 14 dfc2020 entries, all pointing at
   checkpoint paths that were **already missing** before this archival. Any
   SHOT/delulu sweep will skip dfc2020 pairs until SFT is retrained and the
   registry refreshed.
2. The launchers dedup against results CSVs. `res/train_sft/dfc2020.csv` was
   moved here, so SFT will resubmit cleanly. But the shared
   `res/delulu/hptuned_*.csv` files still hold old dfc2020 rows; either point
   new runs at a fresh `--results_csv` or strip those rows first.
3. Decide splits (ROI-disjoint default vs Copernicus-Bench official 986/986)
   and resolution (256 native vs `target_size=96`) before spending GPU time.
