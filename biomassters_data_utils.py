"""
BioMassters data utilities — temporal S1/S2 above-ground-biomass (AGB) regression.

BioMassters (GeoBench-v2): Sentinel-1 SAR (4 bands: VV/VH asc+desc) + Sentinel-2
optical (10 bands) time series with a dense per-pixel AGB regression target.
Source: HuggingFace aialliance/biomassters (7 tortilla parts), loaded through
geobench_v2.datasets.biomassters.GeoBenchBioMassters.

Temporal handling: the underlying dataset returns per-modality tensors of shape
[C, T, H, W] when num_time_steps > 1. We keep the time axis explicit and stack
modalities on the channel axis, producing a single image tensor:

    image: [C_total, T, H, W]   (S2 channels then S1 channels), z-score normalized
    mask:  [H, W]               float32, z-normalized AGB (mean 0, std 289.89)

Downstream, create_multimodal_batch slices the channel axis (dim 1 after batching)
and leaves T intact; the model's temporal shim folds T into the batch dimension,
runs the non-temporal backbone per timestep, and mean-pools features over T. So the
backbone, classifier/segmenter heads, and all SHOT loss paths stay non-temporal.

The task is regression: label_key='mask', task_type='regression', num_classes=1.

Usage:
    from data_utils import get_loaders
    loaders = get_loaders('biomassters', 's2', batch_size=8, num_workers=4,
                          new_modality='s1', num_time_steps=6)
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

import torch.nn as nn

from geobench_v2.datasets.biomassters import GeoBenchBioMassters

from data_utils import TaskConfig  # noqa: F401


class IdentityNormalizer(nn.Module):
    """No-op normalizer: returns raw bands so we can apply min-max ourselves.

    Passed to GeoBenchBioMassters as a pre-initialized instance, so geobench uses
    it directly (its 'callable instance' branch) instead of constructing it with
    stats/band_order like a ZScoreNormalizer.
    """
    def forward(self, data):
        return data

    def __call__(self, data):
        return data


# ---------------------------------------------------------------------------
# Band definitions
# ---------------------------------------------------------------------------

# S2 optical (10 bands) — order matches GeoBenchBioMassters.band_default_order['s2'].
# Kept in a fixed, explicit order so modality_slices are deterministic.
BIOMASSTERS_S2_BANDS = ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12')
# S1 SAR (4 bands): ascending + descending VV/VH.
BIOMASSTERS_S1_BANDS = ('VV_asc', 'VH_asc', 'VV_desc', 'VH_desc')

# --- Min-max normalization constants (BioMassters 1st-place solution, kbrodt) ---
# We normalize with these ourselves rather than using geobench's z-score, since we
# read the raw BioMassters tortillas directly. Applied per band, in the band order
# above. S1: (x - min) / (max - min); S2: x / max.
#
# Verified against our tortillas (part 0, 60 samples): every observed per-band
# min/max falls within these bounds (S1 obs maxes 5..16 << [29,28,30,22]; S2 obs
# maxes ~15k-18k just under s2_max). The winner's raw product has an 11th S2 band
# (cloud probability, max 255) that geobench drops; confirmed our band 10 maxes at
# exactly 255, so geobench's 10 spectral bands map 1:1 to the winner's first 10
# s2_max values below.
S1_MIN = (-25.0, -62.0, -25.0, -60.0)      # VV_asc, VH_asc, VV_desc, VH_desc
S1_MAX = ( 29.0,  28.0,  30.0,  22.0)
S2_MAX = (19616.0, 18400.0, 17536.0, 17097.0, 16928.0,
          16768.0, 16593.0, 16492.0, 15401.0, 15226.0)  # B02..B12
S1_NODATA = -9999.0

# AGB target is kept in RAW t/ha (matching the winner), so reported RMSE is already
# in t/ha and regression_scale is 1.0. Geobench z-normalizes AGB internally with the
# std below; we invert that in the wrapper to recover raw t/ha.
AGB_MEAN = 0.0
AGB_STD = 289.89   # geobench's internal AGB std, used only to un-normalize the target

# BioMassters winner masks pixels with AGB >= this (t/ha) from the loss.
BIOMASSTERS_AGB_MASK_THRESHOLD = 400.0

# Native spatial size of BioMassters tiles.
BIOMASSTERS_IMG_SIZE = 256

# Total channels in the stacked image (S2 first, then S1).
_N_S2 = len(BIOMASSTERS_S2_BANDS)
_N_S1 = len(BIOMASSTERS_S1_BANDS)


# ---------------------------------------------------------------------------
# Temporal stacking dataset
# ---------------------------------------------------------------------------

class TemporalStackedDataset(Dataset):
    """
    Wraps GeoBenchBioMassters and stacks S2+S1 into one channel-major image
    tensor while preserving the time axis. Applies the BioMassters 1st-place
    min-max normalization (not geobench z-score) and keeps the AGB target in raw
    t/ha. Optionally applies month dropout as a train-time augmentation.

    The wrapped GeoBenchBioMassters is created with an IdentityNormalizer so the
    bands come out raw (un-normalized); we normalize here instead.

    GeoBenchBioMassters (with return_stacked_image=False) yields:
        image_s2: [C_s2, T, H, W]   (or [C_s2, H, W] if num_time_steps == 1)
        image_s1: [C_s1, T, H, W]
        mask:     [H, W]   (z-normalized by geobench: agb / AGB_STD)

    This wrapper outputs (s2 channels then s1):
        image: [C_s2 + C_s1, T, H, W]   min-max normalized, T dim always present
        mask:  [H, W]                    raw AGB in t/ha (geobench z-score inverted)

    Month dropout: with per-timestep probability `month_dropout`, a timestep's
    channels are zeroed (matching the winner's augmentation: "simply removing
    images"). Applied only when `training=True`.
    """

    def __init__(self, dataset, num_time_steps: int, training: bool = False,
                 month_dropout: float = 0.0):
        self.dataset = dataset
        self.num_time_steps = num_time_steps
        self.training = training
        self.month_dropout = month_dropout

        # Per-band normalization tensors, shaped [C, 1, 1, 1] to broadcast over
        # [C, T, H, W]. Order: s2 (10) then s1 (4), matching the stack below.
        s2_max = torch.tensor(S2_MAX, dtype=torch.float32)
        s1_min = torch.tensor(S1_MIN, dtype=torch.float32)
        s1_max = torch.tensor(S1_MAX, dtype=torch.float32)
        s1_range = s1_max - s1_min
        # image = cat([s2, s1]); s2 uses x/max (min 0), s1 uses (x-min)/range.
        self._sub = torch.cat([torch.zeros(_N_S2), s1_min]).view(-1, 1, 1, 1)
        self._div = torch.cat([s2_max, s1_range]).view(-1, 1, 1, 1)
        # Which channels are S1 (for -9999 nodata handling before normalization).
        self._s1_start = _N_S2

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _ensure_cthw(x: Tensor) -> Tensor:
        # Dataset returns [C, H, W] for a single step; make it [C, 1, H, W].
        if x.dim() == 3:
            return x.unsqueeze(1)
        return x  # already [C, T, H, W]

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]

        s2 = self._ensure_cthw(sample['image_s2']).float()
        s1 = self._ensure_cthw(sample['image_s1']).float()

        # Concatenate on channel axis; both share the same T, H, W.
        image = torch.cat([s2, s1], dim=0)  # [C_total, T, H, W]

        # S1 no-data (-9999) -> 0 before min-max (matches winner). S2 missing
        # months are already zero-filled by geobench.
        s1_slice = image[self._s1_start:]
        s1_slice[s1_slice == S1_NODATA] = 0.0

        # Min-max normalization: s2 -> x/max, s1 -> (x-min)/range.
        image = (image - self._sub) / self._div

        # Month dropout (train only): zero out whole timesteps with prob p.
        if self.training and self.month_dropout > 0.0:
            T = image.shape[1]
            drop = torch.rand(T) < self.month_dropout
            if drop.all():          # never drop every timestep
                drop[torch.randint(T, (1,))] = False
            image[:, drop] = 0.0

        # Recover raw AGB in t/ha (geobench applied agb / AGB_STD internally).
        mask = sample['mask'].float() * AGB_STD

        return {'image': image, 'mask': mask}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def get_biomassters_loaders(
    batch_size: int = 8,
    num_workers: int = 4,
    data_root: str = 'datasets/geoben2/biomassters',
    seed: int = 42,
    starting_modality: str = 's2',
    new_modality: str | None = None,
    num_time_steps: int = 6,
    data_normalizer=None,          # ignored; we apply min-max ourselves
    month_dropout: float = 0.3,    # train-time per-timestep drop prob (winner: 0.3)
) -> tuple:
    """
    Create the standard 5-loader + TaskConfig tuple for BioMassters.

    Mirrors get_benv2_loaders: all loaders expose the full stacked S2+S1 image;
    modality selection happens at train time via create_multimodal_batch using
    modality_slices. train1/val1 and train2/val2 are disjoint 50/50 splits of the
    train/val sets, so train1 supports starting-modality supervised training and
    train2 supports SSL/distillation on the same underlying distribution.

    Args:
        starting_modality: 's1' or 's2'. Recorded in TaskConfig; does not change
            which channels are loaded (the full stack is always loaded).
        new_modality: 's1' or 's2' (the "other" modality) or None for stage-0 only.
        num_time_steps: number of most-recent timesteps (max 12). Features are
            mean-pooled over these timesteps inside the model.

    Returns:
        train1, val1, train2, val2, test loaders + TaskConfig.
    """
    assert starting_modality in ('s1', 's2'), \
        f"starting_modality must be 's1' or 's2', got {starting_modality!r}"
    assert new_modality in (None, 's1', 's2'), \
        f"new_modality must be 's1', 's2', or None, got {new_modality!r}"

    root = Path(data_root)

    band_order = {
        's2': list(BIOMASSTERS_S2_BANDS),
        's1': list(BIOMASSTERS_S1_BANDS),
    }

    # IdentityNormalizer: geobench returns raw bands; we apply the winner's
    # min-max in TemporalStackedDataset. (data_normalizer arg is ignored.)
    common = dict(
        root=root,
        band_order=band_order,
        data_normalizer=IdentityNormalizer(),   # instance -> geobench uses it as-is (no-op)
        num_time_steps=num_time_steps,
        return_stacked_image=False,   # we stack ourselves to keep the T axis
        download=False,               # data already present; installed class lists only 3 parts
    )

    train_full = GeoBenchBioMassters(split='train', **common)
    val_full   = GeoBenchBioMassters(split='validation', **common)
    test_full  = GeoBenchBioMassters(split='test', **common)

    # Month dropout applies to training splits only.
    train_ds = TemporalStackedDataset(train_full, num_time_steps, training=True, month_dropout=month_dropout)
    val_ds   = TemporalStackedDataset(val_full, num_time_steps, training=False)
    test_ds  = TemporalStackedDataset(test_full, num_time_steps, training=False)

    # Disjoint, deterministic 50/50 splits of train and val.
    rng = random.Random(seed)

    train_indices = list(range(len(train_ds)))
    rng.shuffle(train_indices)
    half_t = len(train_indices) // 2
    train1_ds = Subset(train_ds, train_indices[:half_t])
    train2_ds = Subset(train_ds, train_indices[half_t:])

    val_indices = list(range(len(val_ds)))
    rng.shuffle(val_indices)
    half_v = len(val_indices) // 2
    val1_ds = Subset(val_ds, val_indices[:half_v])
    val2_ds = Subset(val_ds, val_indices[half_v:])

    print(f"BioMassters — Train1: {len(train1_ds)}, Train2: {len(train2_ds)}, "
          f"Val1: {len(val1_ds)}, Val2: {len(val2_ds)}, Test: {len(test_ds)} "
          f"(S2+S1, T={num_time_steps})")

    train1_loader = DataLoader(train1_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val1_loader   = DataLoader(val1_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    train2_loader = DataLoader(train2_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val2_loader   = DataLoader(val2_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # Fixed modality slices into the stacked channel axis (s2 first, then s1).
    modality_slices: dict[str, slice] = {
        's2': slice(0, _N_S2),
        's1': slice(_N_S2, _N_S2 + _N_S1),
    }
    # S2 RGB sub-group (B04, B03, B02 -> indices 2,1,0 within the s2 slice).
    modality_slices['s2_rgb'] = [2, 1, 0]

    assert starting_modality in modality_slices
    assert new_modality is None or new_modality in modality_slices

    def _bands_len(spec) -> int:
        if isinstance(spec, slice):
            return spec.stop - spec.start
        return len(spec)

    start_channels = _bands_len(modality_slices[starting_modality])
    new_channels   = _bands_len(modality_slices[new_modality]) if new_modality is not None else 0

    task_config = TaskConfig(
        dataset_name='biomassters',
        task_type='regression',
        modality_a=starting_modality,
        modality_b=new_modality,
        modality_a_channels=start_channels,
        modality_b_channels=new_channels,
        num_classes=1,              # single continuous per-pixel output
        multilabel=False,
        label_key='mask',
        modality_bands_dict=modality_slices,
        img_size=BIOMASSTERS_IMG_SIZE,
        regression_scale=1.0,   # target already in raw t/ha, so RMSE is in t/ha
        regression_loss_scale=AGB_STD,   # divide distill/CE MSE by AGB_STD^2 to match latent/prefusion scale
        regression_mask_above=BIOMASSTERS_AGB_MASK_THRESHOLD,   # exclude AGB>=400 (winner)
    )

    return train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
