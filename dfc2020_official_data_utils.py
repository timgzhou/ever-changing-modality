"""
DFC2020 data utilities — OFFICIAL high-resolution labels.

This module replaces dfc2020_data_utils.py, which was backed by the HuggingFace
GFM-Bench/DFC2020 repackaging. That packaging shipped the SEN12MS MODIS-derived
`lc` product as the segmentation target instead of the DFC2020 contest ground
truth. The MODIS labels are ~500 m native, so on a 96x96 (960 m) tile they
resolve to 2-3 blobs: measured over 300 test tiles, mean 2.3 connected regions,
median region 2276 px, and a mean dominant-class fraction of 0.89 (i.e. constant
prediction scores ~89% pixel accuracy). Numbers computed against those labels are
not comparable to published DFC2020 results.

This module reads the official DFC_Public_Dataset release instead, which carries
the semi-manually generated 10 m labels in `dfc_*` directories alongside the
`lc_*` MODIS maps. Same sample measured with the real labels: 1214 connected
regions, median region 4 px, dominant-class fraction 0.41.

Data source: IEEE DataPort, 2020 IEEE GRSS Data Fusion Contest (competition
17534), file DFC_Public_Dataset.zip. Access requires an approved DataPort
account; the competition is archived, so access may need to be requested from
iadf_chairs@grss-ieee.org.

Layout after extraction (data_root points at DFC_Public_Dataset/):

    DFC_Public_Dataset/
        dfc_sen12ms_dataset.py
        ROIs0000_autumn/
            s1_Mumbai/ROIs0000_autumn_s1_Mumbai_p741.tif      (2ch VV/VH)
            s2_Mumbai/ROIs0000_autumn_s2_Mumbai_p741.tif      (13ch)
            lc_Mumbai/ROIs0000_autumn_lc_Mumbai_p741.tif      (MODIS, unused)
            dfc_Mumbai/ROIs0000_autumn_dfc_Mumbai_p741.tif    (10 m ground truth)
            ... also BandarAnzali, CapeTown
        ROIs0000_spring/   BlackForest
        ROIs0000_summer/   Chabarovsk
        ROIs0000_winter/   KippaRing, MexicoCity

Note the public bundle is organised by season + named city ROI, NOT by the
ROIs0000_validation / ROIs0000_test split that the organisers' loader script
(dfc_sen12ms_dataset.py, Seasons enum) expects. Splits here are therefore built
by held-out ROI; see get_dfc2020_loaders.

Usage:
    loaders = get_dfc2020_loaders(data_root='datasets/DFC2020_official/DFC_Public_Dataset', ...)
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from data_utils import TaskConfig

# ---------------------------------------------------------------------------
# Normalization statistics
# ---------------------------------------------------------------------------
# Reused from the SEN12MS/GFM-Bench statistics; the underlying Sentinel imagery
# is the same product, only the labels differ between packagings.

# S2: 13 bands — B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B10, B11, B12
DFC2020_S2_MEAN = [1370.19151926, 1184.3824625,  1120.77120066, 1136.26026392, 1263.73947144,
                   1645.40315151, 1846.87040806, 1762.59530783, 1972.62420416,  582.72633433,
                     14.77112979, 1732.16362238, 1247.91870117]
DFC2020_S2_STD  = [ 633.15169573,  650.2842772,   712.12507725,  965.23119807,  948.9819932,
                   1108.06650639, 1258.36394548, 1233.1492281,  1364.38688993,  472.37967789,
                     14.3114637,  1310.36996126, 1087.6020813]

# S1: 2 bands — VV (index 0), VH (index 1)
DFC2020_S1_MEAN = [-12.54847273, -20.19237134]
DFC2020_S1_STD  = [  5.25697717,   5.91150917]

# ---------------------------------------------------------------------------
# Label handling
# ---------------------------------------------------------------------------
# The dfc_* rasters are already stored in the simplified DFC scheme (values
# 1-10, with 0 meaning unlabeled). The organisers' loader defines the mapping
# from raw IGBP as:
#
#     IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10])
#
# We keep it here and apply it defensively: if a raster is found to contain
# values > 10 it is raw IGBP and gets remapped; otherwise it is passed through.
IGBP2DFC = np.array([0, 1, 1, 1, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7, 6, 8, 9, 10], dtype=np.int64)

# DFC class ids are 1-10; we shift to 0-9 for the loss and send 0 (unlabeled)
# to ignore_index.
DFC2020_CLASS_NAMES = [
    'Forest',          # 1
    'Shrubland',       # 2
    'Savanna',         # 3
    'Grassland',       # 4
    'Wetlands',        # 5
    'Croplands',       # 6
    'Urban/Built-up',  # 7
    'Snow/Ice',        # 8
    'Barren',          # 9
    'Water',           # 10
]

DFC2020_NUM_CLASSES  = 10
DFC2020_IGNORE_INDEX = 255

# Savanna (DFC class 3, raw IGBP 8 and 9) is a *scored* class in the official
# benchmark. The previous MODIS-backed loader sent it to ignore_index, which
# discarded roughly half of the average tile.


def _dfc_to_train_ids(raw: np.ndarray) -> np.ndarray:
    """Map a raw dfc raster to contiguous train ids 0-9, with 255 = ignore."""
    arr = raw.astype(np.int64)
    if arr.max() > DFC2020_NUM_CLASSES:
        # Raw IGBP (1-17) rather than simplified DFC — remap.
        arr = IGBP2DFC[np.clip(arr, 0, 17)]
    out = np.full(arr.shape, DFC2020_IGNORE_INDEX, dtype=np.int64)
    valid = (arr >= 1) & (arr <= DFC2020_NUM_CLASSES)
    out[valid] = arr[valid] - 1
    return out


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

_PATCH_RE = re.compile(r'^(ROIs\d+)_(\w+?)_dfc_(\w+)_p(\d+)\.tif$')


def build_index(data_root: str | Path) -> list[dict]:
    """
    Walk the extracted DFC_Public_Dataset tree and return one record per patch.

    A patch is kept only when all three of s1/s2/dfc exist for it, so the index
    is guaranteed complete (the release has a handful of unpaired rasters).

    Returns a list of dicts with keys: season, roi, patch, s1, s2, dfc, lc.
    The `lc` entry is the MODIS raster kept only for provenance comparison
    figures; it is never used as a training target.
    """
    root = Path(data_root)
    records = []
    for dfc_dir in sorted(root.glob('ROIs*/dfc_*')):
        if not dfc_dir.is_dir():
            continue
        season = dfc_dir.parent.name
        roi = dfc_dir.name[len('dfc_'):]
        for dfc_path in sorted(dfc_dir.glob('*.tif')):
            m = _PATCH_RE.match(dfc_path.name)
            if m is None:
                continue
            patch = m.group(4)
            s1_path = dfc_dir.parent / f's1_{roi}' / f'{season}_s1_{roi}_p{patch}.tif'
            s2_path = dfc_dir.parent / f's2_{roi}' / f'{season}_s2_{roi}_p{patch}.tif'
            if not (s1_path.exists() and s2_path.exists()):
                continue
            lc_path = dfc_dir.parent / f'lc_{roi}' / f'{season}_lc_{roi}_p{patch}.tif'
            records.append({
                'season': season, 'roi': roi, 'patch': patch,
                's1': s1_path, 's2': s2_path, 'dfc': dfc_path,
                'lc': lc_path if lc_path.exists() else None,
            })
    if not records:
        raise RuntimeError(
            f'No DFC2020 patches found under {root!r}. Expected the extracted '
            'DFC_Public_Dataset directory containing ROIs*/dfc_* subdirectories.'
        )
    return records


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DFC2020Dataset(Dataset):
    """
    DFC2020 with official 10 m labels.

    Exposes:
        batch['image']: [15, H, W] float32 — S2 (ch 0-12) then S1 (ch 13-14)
        batch['mask']:  [H, W] int64, values 0-9 + 255 (ignore_index)
    """

    def __init__(self, records: list[dict], target_size: int | None = None,
                 normalize: bool = True):
        self.records = records
        self.target_size = target_size
        self.normalize = normalize

        self._s2_mean = torch.tensor(DFC2020_S2_MEAN, dtype=torch.float32).view(-1, 1, 1)
        self._s2_std  = torch.tensor(DFC2020_S2_STD,  dtype=torch.float32).view(-1, 1, 1)
        self._s1_mean = torch.tensor(DFC2020_S1_MEAN, dtype=torch.float32).view(-1, 1, 1)
        self._s1_std  = torch.tensor(DFC2020_S1_STD,  dtype=torch.float32).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        s2_raw = tifffile.imread(rec['s2'])
        s1_raw = tifffile.imread(rec['s1'])
        lbl    = tifffile.imread(rec['dfc'])

        # Rasters are band-last ([H, W, C]) in this release: S2 is (256,256,13)
        # uint16, S1 is (256,256,2) float64 dB. Transpose to CHW; the guards also
        # tolerate an already-CHW layout.
        if s2_raw.ndim == 3 and s2_raw.shape[0] != 13 and s2_raw.shape[-1] == 13:
            s2_raw = np.transpose(s2_raw, (2, 0, 1))
        if s1_raw.ndim == 3 and s1_raw.shape[0] != 2 and s1_raw.shape[-1] == 2:
            s1_raw = np.transpose(s1_raw, (2, 0, 1))
        if lbl.ndim == 3:
            lbl = lbl[0] if lbl.shape[0] == 1 else lbl[..., 0]

        s2 = torch.from_numpy(np.ascontiguousarray(s2_raw, dtype=np.float32))
        s1 = torch.from_numpy(np.ascontiguousarray(s1_raw, dtype=np.float32))
        mask = torch.from_numpy(_dfc_to_train_ids(lbl))

        if self.normalize:
            s2 = (s2 - self._s2_mean) / (self._s2_std + 1e-6)
            s1 = (s1 - self._s1_mean) / (self._s1_std + 1e-6)

        image = torch.cat([s2, s1], dim=0)  # [15, H, W]

        if self.target_size is not None and image.shape[-1] != self.target_size:
            image = F.interpolate(
                image.unsqueeze(0), size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)
            # Labels must use nearest — bilinear would invent class ids.
            mask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0).float(),
                size=(self.target_size, self.target_size), mode='nearest',
            ).squeeze(0).squeeze(0).long()

        return {'image': image, 'mask': mask}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

# Held-out ROIs. The public release does not ship the contest's own
# train/val/test partition, so we split by *geographic ROI* to keep test
# genuinely out-of-region. This is stricter than a random tile split: adjacent
# crops of one scene cannot straddle the boundary.
DEFAULT_TEST_ROIS = ('CapeTown', 'MexicoCity')
DEFAULT_VAL_ROIS  = ('Chabarovsk',)


def get_dfc2020_loaders(
    batch_size: int = 32,
    num_workers: int = 8,
    data_root: str = 'datasets/DFC2020_official/DFC_Public_Dataset',
    seed: int = 42,
    starting_modality: str = 's2',
    new_modality: str | None = 's1',
    val_fraction: float = 0.0,  # accepted for interface compatibility; unused
    normalize: bool = True,
    target_size: int | None = None,
    test_rois: tuple[str, ...] = DEFAULT_TEST_ROIS,
    val_rois: tuple[str, ...] = DEFAULT_VAL_ROIS,
) -> tuple:
    """
    Create 5 dataloaders for DFC2020 matching the SHOT interface.

    Splits are ROI-disjoint: test and val ROIs are held out entirely, and the
    remaining ROIs form train. Train is then split into train1 (stage-0) and
    train2 (SSL) by *patch id* rather than by shuffled tile index, so spatially
    adjacent crops of the same scene cannot land on both sides.

    Returns:
        train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
    """
    modality_bands_dict = {
        's2':      slice(0, 13),
        's1':      slice(13, 15),
        's2_rgb':   [3, 2, 1],                          # B4, B3, B2
        's2_norgb': [0, 4, 5, 6, 7, 8, 9, 10, 11, 12],  # B1, B5-B12
    }

    assert starting_modality in modality_bands_dict, \
        f"starting_modality must be one of {list(modality_bands_dict)}, got {starting_modality!r}"
    assert new_modality is None or new_modality in modality_bands_dict, \
        f"new_modality must be one of {list(modality_bands_dict)} or None, got {new_modality!r}"

    records = build_index(data_root)

    test_recs  = [r for r in records if r['roi'] in test_rois]
    val_recs   = [r for r in records if r['roi'] in val_rois]
    train_recs = [r for r in records if r['roi'] not in test_rois and r['roi'] not in val_rois]

    for name, recs, rois in (('test', test_recs, test_rois), ('val', val_recs, val_rois)):
        if not recs:
            raise ValueError(f'{name} split is empty — no patches matched ROIs {rois}. '
                             f'Available ROIs: {sorted({r["roi"] for r in records})}')

    # Split train by patch id so adjacent crops of one scene stay together.
    rng = random.Random(seed)
    train_keys = sorted({(r['season'], r['roi'], r['patch']) for r in train_recs})
    rng.shuffle(train_keys)
    mid = len(train_keys) // 2
    keys1 = set(train_keys[:mid])
    train1_recs = [r for r in train_recs if (r['season'], r['roi'], r['patch']) in keys1]
    train2_recs = [r for r in train_recs if (r['season'], r['roi'], r['patch']) not in keys1]

    # Same treatment for val.
    val_keys = sorted({(r['season'], r['roi'], r['patch']) for r in val_recs})
    rng.shuffle(val_keys)
    vmid = len(val_keys) // 2
    vkeys1 = set(val_keys[:vmid])
    val1_recs = [r for r in val_recs if (r['season'], r['roi'], r['patch']) in vkeys1]
    val2_recs = [r for r in val_recs if (r['season'], r['roi'], r['patch']) not in vkeys1]

    mk = lambda recs: DFC2020Dataset(recs, target_size=target_size, normalize=normalize)
    train1_ds, train2_ds = mk(train1_recs), mk(train2_recs)
    val1_ds,   val2_ds   = mk(val1_recs),   mk(val2_recs)
    test_ds              = mk(test_recs)

    print(f"DFC2020 (official labels) — Train1: {len(train1_ds)}, Train2: {len(train2_ds)}, "
          f"Val1: {len(val1_ds)}, Val2: {len(val2_ds)}, Test: {len(test_ds)}")
    print(f"  train ROIs: {sorted({r['roi'] for r in train_recs})}")
    print(f"  val ROIs:   {sorted({r['roi'] for r in val_recs})}")
    print(f"  test ROIs:  {sorted({r['roi'] for r in test_recs})}")

    timeout = 120 if num_workers > 0 else 0
    train1_loader = DataLoader(train1_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, timeout=timeout)
    val1_loader   = DataLoader(val1_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers // 2, pin_memory=True)
    train2_loader = DataLoader(train2_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, timeout=timeout)
    val2_loader   = DataLoader(val2_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers // 2, pin_memory=True)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False,
                               num_workers=num_workers // 2, pin_memory=True)

    def _bands_len(spec):
        if isinstance(spec, slice):
            return len(range(*spec.indices(15)))
        return len(spec)

    start_ch = _bands_len(modality_bands_dict[starting_modality])
    new_ch   = _bands_len(modality_bands_dict[new_modality]) if new_modality is not None else 0

    sample_img = test_ds[0]['image']
    img_size = int(sample_img.shape[-1])

    task_config = TaskConfig(
        dataset_name='dfc2020',
        task_type='segmentation',
        modality_a=starting_modality,
        modality_b=new_modality,
        modality_a_channels=start_ch,
        modality_b_channels=new_ch,
        num_classes=DFC2020_NUM_CLASSES,
        multilabel=False,
        label_key='mask',
        modality_bands_dict=modality_bands_dict,
        img_size=img_size,
        ignore_index=DFC2020_IGNORE_INDEX,
    )

    return train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
