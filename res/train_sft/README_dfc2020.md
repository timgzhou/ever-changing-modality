# DFC2020 results — label sources

## MODIS-label era (removed 2026-08-20)

DFC2020 results produced before 2026-08-19 used the HuggingFace
`GFM-Bench/DFC2020` packaging, which ships the SEN12MS MODIS-derived `lc`
product instead of the contest's 10 m human-annotated `dfc_` ground truth
(~16 connected regions/tile vs ~400 for the real labels; a majority-class
predictor scored 56.8% pixel accuracy). It also used an 8-class head that sent
savanna to ignore_index, discarding 37.8% of pixels, and its val split was
half-downloaded (4617 metadata rows, 2308 files on disk).

Everything from that era was **deleted on 2026-08-20**: the results CSV, the
31 checkpoints (8-class heads, Aug-12 timestamps), `dfc2020_data_utils.py`,
`old_code_dfc2020_modis/`, the extracted `datasets/GFM-Bench/` tree, and
`DFC2020.zip`. `sh/shot_ete_dfc2020_best.sh` pointed at a MODIS teacher and
went with them.

`viz_dfc2020_labels.py` still documents the bug as a figure — it reads `lc_`
and `dfc_` side by side from the *official* release, so it needs nothing from
GFM-Bench.

## Current results

- `dfc2020.csv` — ROI-disjoint split via `dfc2020_official_data_utils.py`,
  10 classes, real labels. Honest generalization, but val is one ROI
  (Chabarovsk, 63% Wetlands vs 1% of train) so val is a weak selector
  (test-vs-val Spearman rho 0.53).
- `dfc2020_cobench.csv` — Copernicus-Bench official 3156/986/986 split via
  `dfc2020_cobench_data_utils.py`, 8 classes (their cls_mapping ignores
  Background/Savanna/Snow-Ice). Comparable to published baselines
  (DFC2020-S2 mIoU: supervised ViT-B/16 66.2, random init 62.3) and val is
  reliable here (rho 0.99).

The two are **not** comparable to each other and their checkpoints have
different head sizes (10 vs 8).

## Bimodal training bug (fixed 2026-08-20)

`dfc2020_cobench_BROKEN_bimodal_pre20260820.csv` holds the 60 combined-modality
rows (s2_rgb+s1, s2_norgb+s1, s2_rgb+s2_norgb) produced before the fix.

`train_sft.py` passed `modality=primary_modality` (= `args.modalities[0]`) into
`single_modality_training_loop`, which forwarded it as `modalities=(modality,)`.
The EVAN backbone DID build components for every modality in `--modalities`
(`starting_modality=args.modalities`), but only the first was ever fed to the
model. So a `--modalities s2_rgb s1` run was a **single-modality s2_rgb model
carrying untrained s1 parameters**. The training logs show it: the header reads
`modalities=s2_rgb s1` while every epoch prints `Train (S2_RGB)`.

This is why combined entries never beat their best single modality — they were
the same model. Measured on the s2_rgb+s1 upernet checkpoint, feeding both
modalities at inference (off-distribution for it) drops test mIoU 59.18 -> 6.02.

Fixed in `train_utils.py` (`single_modality_training_loop` now accepts a str or
a sequence and forwards the full tuple) and `train_sft.py` (passes
`tuple(args.modalities)`). Single-modality behaviour is unchanged.

The 60 single-modality rows in `dfc2020_cobench.csv` are NOT affected and were
kept. Any other dataset's combined-modality runs (e.g. benv2 `s2_rgb+s1`) share
the bug and should be re-run before use.

## Teacher train_split leak (found 2026-08-21)

Stage-1 methods — `shot_ete.py` (delulu), `baseline_distillation.py`,
`baseline_mke.py` — use **train2 as the unlabeled/adaptation pool** while
train1 is the labeled set. A teacher trained with `--train_split=full` was
supervised on train1+train2, so it has already seen the "unlabeled" pool with
labels. Any transfer/peeking/addition number from such a teacher overstates
what the method achieves from unlabeled data.

Evidence: MKE logs the teacher's own metrics on both halves. For every
upernet teacher, train2 ~ train1 and both are ~7 mIoU above test:

    teacher    train1  train2   test
    s1          62.06   61.60  52.55
    s2_rgb      69.27   69.01  62.05
    s2_norgb    69.67   69.41  66.74

A teacher that had never seen train2 would score near its test value there.

Affected (all used `full` teachers):
  - res/delulu/dfc2020_cobench_upernet.csv          42 runs
  - res/baselines/dfc2020_cobench_*_{linear,upernet}.csv   72 runs
  - the three W&B sweeps registered 2026-08-21 (cancelled before completion)

Teachers for stage-1 methods must come from `--train_split=split1`.
`full` teachers remain correct for the *supervised upper bound* row, which is
allowed to use all labels. sh/train_sft/train_sft_dfc2020_upernet.sh now
defaults to split1; override with TRAIN_SPLIT=full for the upper bound.
