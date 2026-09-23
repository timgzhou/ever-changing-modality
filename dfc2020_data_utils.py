"""
DFC2020 data utilities — official high-resolution labels, Copernicus-Bench split.

Reads the official IEEE DataPort DFC_Public_Dataset release, which carries the
semi-manually generated 10 m `dfc_*` labels. Splitting follows Copernicus-Bench
(3156/986/986 over the same imagery, 8 classes after their cls_mapping drops
Background/Savanna/Snow-Ice), so numbers are comparable to published baselines
(DFC2020-S2 mIoU: supervised ViT-B/16 66.2, random init 62.3).

History — two superseded variants, neither reachable from this module:
  * The HuggingFace GFM-Bench/DFC2020 packaging shipped the SEN12MS MODIS-derived
    `lc` product as the target instead of the contest ground truth. MODIS is
    ~500 m native, so a 96x96 (960 m) tile resolved to 2-3 blobs and a constant
    predictor scored ~89% pixel accuracy. Deleted 2026-08-20 along with its
    results and checkpoints; numbers from it are not comparable to anything.
  * A ROI-disjoint split (held-out cities CapeTown/MexicoCity, val Chabarovsk,
    10 classes) used real labels but made val a weak model selector
    (test-vs-val Spearman rho 0.53 vs 0.99 here). Removed 2026-09-13; its
    stage-0 rows survive in res/train_sft/dfc2020.csv for reference only.

Data source: IEEE DataPort, 2020 IEEE GRSS Data Fusion Contest (competition
17534), file DFC_Public_Dataset.zip.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

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

DFC2020_S2_BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')

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


# Copernicus-Bench cls_mapping (cobench_dfc2020s12_wrapper.py):
#   {0:255, 1:0, 2:1, 3:255, 4:2, 5:3, 6:4, 7:5, 8:255, 9:6, 10:7}
# i.e. Background, Savanna and Snow/Ice are ignored; 8 valid classes remain.
COBENCH_CLS_MAPPING = {0: 255, 1: 0, 2: 1, 3: 255, 4: 2, 5: 3,
                       6: 4, 7: 5, 8: 255, 9: 6, 10: 7}
COBENCH_NUM_CLASSES = 8
COBENCH_CLASS_NAMES = ['Forest', 'Shrubland', 'Grassland', 'Wetlands',
                       'Croplands', 'Urban/Built-up', 'Barren', 'Water']

# LUT over raw dfc ids 0-10 -> train ids 0-7 (255 = ignore)
_COBENCH_LUT = np.full(18, DFC2020_IGNORE_INDEX, dtype=np.int64)
for _raw, _tid in COBENCH_CLS_MAPPING.items():
    _COBENCH_LUT[_raw] = _tid

_HERE = Path(__file__).resolve().parent
DEFAULT_MAP = _HERE / 'cb_patch_map.json'
DEFAULT_SPLITS = _HERE / 'cb_splits'


class CoBenchDFC2020Dataset(DFC2020Dataset):
    """DFC2020Dataset with the Copernicus-Bench 8-class label mapping."""

    def __getitem__(self, idx: int) -> dict:
        out = super().__getitem__(idx)
        # super() applied the 10-class mapping (raw-1, 0 -> ignore); undo it by
        # recovering the raw id, then apply the Copernicus-Bench LUT instead.
        mask = out['mask']
        raw = torch.where(mask == DFC2020_IGNORE_INDEX,
                          torch.zeros_like(mask), mask + 1)
        out['mask'] = torch.from_numpy(
            _COBENCH_LUT[raw.numpy().astype(np.int64)]
        )
        return out


def _load_split_ids(splits_dir: Path) -> dict[str, list[str]]:
    """Read the three official CSVs -> {split: [CB dfc filename, ...]}."""
    names = {'train': 'dfc-train-new.csv',
             'val': 'dfc-val-new.csv',
             'test': 'dfc-test-new.csv'}
    out = {}
    for split, fname in names.items():
        path = Path(splits_dir) / fname
        if not path.exists():
            raise FileNotFoundError(
                f'Copernicus-Bench split file missing: {path}. Fetch the three '
                'dfc-*-new.csv files from the l2_dfc2020_s1s2/dfc2020.zip on '
                'HuggingFace (they sit at the head of the archive, so an HTTP '
                'range request is enough -- see scripts/build_cobench_map.py).')
        out[split] = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    return out


def get_dfc2020_loaders(
    batch_size: int = 32,
    num_workers: int = 8,
    data_root: str = 'datasets/DFC2020_official/DFC_Public_Dataset',
    seed: int = 42,
    starting_modality: str = 's2',
    new_modality: str | None = 's1',
    val_fraction: float = 0.0,   # accepted for interface compatibility; unused
    normalize: bool = True,
    target_size: int | None = None,
    patch_map: str | Path = DEFAULT_MAP,
    splits_dir: str | Path = DEFAULT_SPLITS,
) -> tuple:
    """
    Create 5 dataloaders on the Copernicus-Bench official split.

    train (3156) is halved into train1 (stage-0) / train2 (SSL) and val (986)
    into val1 / val2, matching the Delulu interface used by the ROI-disjoint
    loader. test (986) is returned whole.

    Returns:
        train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
    """
    modality_bands_dict = {
        's2':      slice(0, 13),
        's1':      slice(13, 15),
        's2_rgb':   [3, 2, 1],
        's2_norgb': [0, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }
    assert starting_modality in modality_bands_dict, \
        f"starting_modality must be one of {list(modality_bands_dict)}, got {starting_modality!r}"
    assert new_modality is None or new_modality in modality_bands_dict, \
        f"new_modality must be one of {list(modality_bands_dict)} or None, got {new_modality!r}"

    patch_map = Path(patch_map)
    if not patch_map.exists():
        raise FileNotFoundError(
            f'Patch mapping not found: {patch_map}. Generate it with '
            'scripts/build_cobench_map.py (maps Copernicus-Bench patch ids to '
            'local filenames by S2 pixel hash).')
    cb_to_local = json.loads(patch_map.read_text())
    if 'map' in cb_to_local:           # tolerate the builder's wrapper form
        cb_to_local = cb_to_local['map']

    # local filename (s2 form) -> record
    records = build_index(data_root)
    by_s2name = {Path(r['s2']).name: r for r in records}

    splits = _load_split_ids(Path(splits_dir))
    picked: dict[str, list[dict]] = {}
    missing: dict[str, int] = {}
    for split, fnames in splits.items():
        recs, miss = [], 0
        for f in fnames:
            s2_cb = f.replace('_dfc_', '_s2_')
            local = cb_to_local.get(s2_cb)
            rec = by_s2name.get(local) if local else None
            if rec is None:
                miss += 1
                continue
            recs.append(rec)
        picked[split] = recs
        missing[split] = miss

    if any(missing.values()):
        raise RuntimeError(
            'Could not resolve every Copernicus-Bench patch to a local file '
            f'(unresolved: {missing}). The split would not match the published '
            'benchmark, so refusing to proceed. Regenerate cb_patch_map.json '
            'after confirming the local bundle is fully extracted.')

    ds = lambda recs: CoBenchDFC2020Dataset(recs, target_size=target_size,
                                            normalize=normalize)
    train_ds_full = ds(picked['train'])
    val_ds_full = ds(picked['val'])
    test_ds = ds(picked['test'])

    # halve train -> train1/train2 and val -> val1/val2 (Delulu interface)
    rng = random.Random(seed)
    tr_idx = list(range(len(train_ds_full))); rng.shuffle(tr_idx)
    va_idx = list(range(len(val_ds_full)));   rng.shuffle(va_idx)
    tmid, vmid = len(tr_idx) // 2, len(va_idx) // 2
    train1_ds = Subset(train_ds_full, tr_idx[:tmid])
    train2_ds = Subset(train_ds_full, tr_idx[tmid:])
    val1_ds   = Subset(val_ds_full, va_idx[:vmid])
    val2_ds   = Subset(val_ds_full, va_idx[vmid:])

    print(f'DFC2020 (Copernicus-Bench official split) — '
          f'Train1: {len(train1_ds)}, Train2: {len(train2_ds)}, '
          f'Val1: {len(val1_ds)}, Val2: {len(val2_ds)}, Test: {len(test_ds)}')
    print(f'  {COBENCH_NUM_CLASSES} classes (Savanna/Snow-Ice/Background ignored, '
          f'per Copernicus-Bench cls_mapping)')

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
    img_size = int(test_ds[0]['image'].shape[-1])

    task_config = TaskConfig(
        dataset_name='dfc2020',
        task_type='segmentation',
        modality_a=starting_modality,
        modality_b=new_modality,
        modality_a_channels=start_ch,
        modality_b_channels=new_ch,
        num_classes=COBENCH_NUM_CLASSES,
        multilabel=False,
        label_key='mask',
        modality_bands_dict=modality_bands_dict,
        img_size=img_size,
        ignore_index=DFC2020_IGNORE_INDEX,
    )
    return train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
