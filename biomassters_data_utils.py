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
backbone, classifier/segmenter heads, and all Delulu loss paths stay non-temporal.

The task is regression: label_key='mask', task_type='regression', num_classes=1.

Usage:
    from data_utils import get_loaders
    loaders = get_loaders('biomassters', 's2', batch_size=8, num_workers=4,
                          new_modality='s1', num_time_steps=6)
"""

from __future__ import annotations

import os
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
                 month_dropout: float = 0.0, temporal_pool: bool = False):
        self.dataset = dataset
        self.num_time_steps = num_time_steps
        self.training = training
        self.month_dropout = month_dropout
        # temporal_pool: average the T timesteps into one BEFORE the model sees
        # them, so the sample is [C, 1, H, W] and every downstream path (patch
        # embed, fusion, Delulu's projector and losses) is the ordinary
        # non-temporal one. This is NOT num_time_steps=1, which keeps only the
        # most recent month and discards the other 11; here all T contribute.
        self.temporal_pool = temporal_pool

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

    def pooled_sample(self, index: int) -> tuple:
        """The [C,1,H,W] pooled image and mask, with augmentation disabled.

        Used to build the on-disk cache, so the cached tensor is the clean
        12-month mean -- deterministic and identical for every epoch.
        """
        was_training = self.training
        self.training = False          # never month-drop into a cache
        try:
            s = self[index]
        finally:
            self.training = was_training
        return s['image'], s['mask']

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

        # Collapse time AFTER normalization and month dropout, so both still act
        # per-timestep and a dropped month contributes zeros to the average
        # exactly as it would to the model's own feature pool. S2 missing months
        # are zero-filled upstream, so this is a plain mean over the T slots, the
        # same reduction the temporal shim applies to features -- moved to the
        # input, where it costs one forward pass instead of T.
        if self.temporal_pool:
            image = image.mean(dim=1, keepdim=True)  # [C, 1, H, W]

        # Recover raw AGB in t/ha (geobench applied agb / AGB_STD internally).
        mask = sample['mask'].float() * AGB_STD

        return {'image': image, 'mask': mask}


class CachedPooledDataset(Dataset):
    """Memory-mapped [N,C,1,H,W] fp16 images + [N,H,W] fp32 masks.

    Why this exists: ~70% of a BioMassters epoch is data loading, not GPU. Even
    with temporal_pool=True the loader still reads all 12 months off disk and
    normalizes them before averaging, so pooling alone made loading marginally
    SLOWER (measured 488 vs 448 ms/batch). Caching the pooled result collapses
    that to a single mmap read: 14.6 GiB fp16 for all 8526 samples, against the
    129 GB source tree.

    fp16 is safe here because the images are already min-max normalized into
    ~[0,1]; the mask stays fp32 since AGB is in raw t/ha up to ~400.

    Month dropout is NOT applied -- the cache holds the clean mean (see
    pooled_sample). Pooled runs therefore train without that augmentation, which
    is a deliberate trade for speed.
    """

    def __init__(self, img_path: Path, mask_path: Path, shape: tuple, n: int):
        self.img_path = Path(img_path)
        self.mask_path = Path(mask_path)
        self.shape = shape          # (C, 1, H, W)
        self.n = n
        self._img = None            # opened lazily, per worker
        self._mask = None

    def _ensure_open(self):
        # np.memmap is not fork-safe when opened before the worker forks, so
        # each DataLoader worker opens its own handle on first access.
        if self._img is None:
            import numpy as np
            self._img = np.memmap(self.img_path, dtype=np.float16, mode='r',
                                  shape=(self.n,) + self.shape)
            H, W = self.shape[2], self.shape[3]
            self._mask = np.memmap(self.mask_path, dtype=np.float32, mode='r',
                                   shape=(self.n, H, W))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index: int) -> dict:
        self._ensure_open()
        img = torch.from_numpy(self._img[index].astype('float32'))
        mask = torch.from_numpy(self._mask[index].copy())
        return {'image': img, 'mask': mask}


def build_pooled_cache(cache_dir, num_time_steps: int = 12,
                       data_root: str = 'datasets/geoben2/biomassters',
                       splits=('train', 'validation', 'test')) -> None:
    """Materialise the mean-pooled dataset to disk (idempotent)."""
    import numpy as np
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    band_order = {'s2': list(BIOMASSTERS_S2_BANDS), 's1': list(BIOMASSTERS_S1_BANDS)}
    for split in splits:
        img_p = cache_dir / f'{split}_img.f16'
        msk_p = cache_dir / f'{split}_mask.f32'
        meta_p = cache_dir / f'{split}.meta'
        if meta_p.exists():
            print(f'  [{split}] cache present, skipping')
            continue
        base = GeoBenchBioMassters(
            split=split, root=Path(data_root), band_order=band_order,
            data_normalizer=IdentityNormalizer(), num_time_steps=num_time_steps,
            return_stacked_image=False, download=False)
        ds = TemporalStackedDataset(base, num_time_steps, training=False,
                                    month_dropout=0.0, temporal_pool=True)
        n = len(ds)
        img0, msk0 = ds.pooled_sample(0)
        C, T, H, W = img0.shape
        assert T == 1, f'expected pooled T=1, got {T}'
        # Write to .tmp then rename, so an interrupted build never leaves a
        # half-written cache that looks complete.
        im = np.memmap(str(img_p) + '.tmp', dtype=np.float16, mode='w+', shape=(n, C, 1, H, W))
        mm = np.memmap(str(msk_p) + '.tmp', dtype=np.float32, mode='w+', shape=(n, H, W))
        for i in range(n):
            a, b = ds.pooled_sample(i)
            im[i] = a.numpy().astype(np.float16)
            mm[i] = b.numpy()
            if (i + 1) % 250 == 0:
                print(f'  [{split}] {i+1}/{n}', flush=True)
        im.flush(); mm.flush()
        del im, mm
        os.replace(str(img_p) + '.tmp', img_p)
        os.replace(str(msk_p) + '.tmp', msk_p)
        meta_p.write_text(f'{n} {C} 1 {H} {W} {num_time_steps}\n')
        print(f'  [{split}] wrote {n} samples -> {img_p.name}', flush=True)


def _load_cached_split(cache_dir, split: str):
    """Return a CachedPooledDataset, or None when the cache is absent."""
    cache_dir = Path(cache_dir)
    meta_p = cache_dir / f'{split}.meta'
    if not meta_p.exists():
        return None
    n, C, T, H, W, _ = (int(x) for x in meta_p.read_text().split())
    return CachedPooledDataset(cache_dir / f'{split}_img.f16',
                               cache_dir / f'{split}_mask.f32', (C, T, H, W), n)


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
    temporal_pool: bool = False,   # mean-pool T at the INPUT -> non-temporal
    cache_dir: str | None = None,  # pooled-cache location (default: $BIOMASSTERS_POOL_CACHE)
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
    # 's1'/'s2' are the two sensors; 's2_rgb'/'s2_norgb' are sub-groups of the
    # S2 stack (see modality_slices below). Data loading is identical for all of
    # them -- both sensors are always read and stacked -- and the modality key
    # only selects which channels a model slices out, so sub-groups are accepted
    # here as well.
    _VALID_MODS = ('s1', 's2', 's2_rgb', 's2_norgb')
    assert starting_modality in _VALID_MODS, \
        f"starting_modality must be one of {_VALID_MODS}, got {starting_modality!r}"
    assert new_modality in (None,) + _VALID_MODS, \
        f"new_modality must be one of {_VALID_MODS} or None, got {new_modality!r}"

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

    # Pooled mode prefers the on-disk cache: a single mmap read per sample
    # instead of decoding 12 months. Falls back to on-the-fly pooling when the
    # cache has not been built (see build_pooled_cache / BIOMASSTERS_POOL_CACHE).
    cached = None
    if temporal_pool:
        cdir = cache_dir or os.environ.get(
            'BIOMASSTERS_POOL_CACHE', str(Path(data_root).parent / 'biomassters_pooled'))
        c_tr, c_va, c_te = (_load_cached_split(cdir, s)
                            for s in ('train', 'validation', 'test'))
        if c_tr is not None and c_va is not None and c_te is not None:
            cached = (c_tr, c_va, c_te)
            print(f"BioMassters — using pooled cache at {cdir} "
                  f"(month_dropout disabled: the cache holds the clean mean)")

    if cached is not None:
        train_ds, val_ds, test_ds = cached
    else:
        train_full = GeoBenchBioMassters(split='train', **common)
        val_full   = GeoBenchBioMassters(split='validation', **common)
        test_full  = GeoBenchBioMassters(split='test', **common)

        # Month dropout applies to training splits only.
        train_ds = TemporalStackedDataset(train_full, num_time_steps, training=True,
                                          month_dropout=month_dropout, temporal_pool=temporal_pool)
        val_ds   = TemporalStackedDataset(val_full, num_time_steps, training=False,
                                          temporal_pool=temporal_pool)
        test_ds  = TemporalStackedDataset(test_full, num_time_steps, training=False,
                                          temporal_pool=temporal_pool)

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

    _t_desc = (f"T={num_time_steps} mean-pooled at input -> 1"
               if temporal_pool else f"T={num_time_steps}")
    print(f"BioMassters — Train1: {len(train1_ds)}, Train2: {len(train2_ds)}, "
          f"Val1: {len(val1_ds)}, Val2: {len(val2_ds)}, Test: {len(test_ds)} "
          f"(S2+S1, {_t_desc})")

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
    # S2 minus RGB: the remaining 7 bands (B05..B12), i.e. the complement of
    # s2_rgb within the 10-band S2 stack. Indices are into the full stacked
    # channel axis, which for s2 starts at 0, so they coincide with the S2
    # band positions: B05=3, B06=4, B07=5, B08=6, B8A=7, B11=8, B12=9.
    modality_slices['s2_norgb'] = [3, 4, 5, 6, 7, 8, 9]

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
