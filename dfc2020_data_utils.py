"""
DFC2020 data utilities — local GeoTIFF-backed semantic segmentation dataset.

DFC2020 (IEEE GRSS Data Fusion Contest 2020): Sentinel-1 (2ch VV/VH) + Sentinel-2
(13ch) paired imagery with 8-class semantic segmentation labels at 96×96 pixels.

Data source: HuggingFace GFM-Bench/DFC2020 (data/DFC2020.zip).
Download and extract the ZIP so that data_root points to the inner DFC2020/
directory containing metadata.csv:

    wget https://huggingface.co/datasets/GFM-Bench/DFC2020/resolve/main/data/DFC2020.zip
    unzip DFC2020.zip -d datasets/GFM-Bench/DFC2020/

This creates:
    datasets/GFM-Bench/DFC2020/DFC2020/
        metadata.csv
        ROIs0000_autumn_s2_*/...   (optical TIFFs)
        ROIs0000_autumn_s1_*/...   (radar TIFFs)
        ...

Usage:
    loaders = get_dfc2020_loaders(data_root='datasets/GFM-Bench/DFC2020', ...)
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from data_utils import TaskConfig

# ---------------------------------------------------------------------------
# Normalization statistics (from HuggingFace DFC2020.py loading script)
# ---------------------------------------------------------------------------

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

# Label remapping: raw pixel value (0-17) → class index (0-7) or 255 (ignore)
DFC2020_CLASSES = [
    255,          # 0  — unused
    0, 0, 0, 0, 0,  # 1-5 → 0
    1, 1,           # 6-7 → 1
    255, 255,       # 8-9 — masked (savanna)
    2,              # 10  → 2
    3,              # 11  → 3
    4,              # 12  → 4
    5,              # 13  → 5
    4,              # 14  → 4
    255,            # 15  — masked
    6,              # 16  → 6
    7,              # 17  → 7
]
_CLASSES_LUT = np.array(DFC2020_CLASSES, dtype=np.int32)

DFC2020_NUM_CLASSES  = 8
DFC2020_IGNORE_INDEX = 255


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DFC2020Dataset(Dataset):
    """
    Loads DFC2020 GeoTIFFs from a local extracted directory.

    Reads metadata.csv to locate optical/radar/label TIFFs per sample.
    Applies z-score normalization and stacks S2+S1 into a single image tensor.
    Output layout: S2 (channels 0–12) followed by S1 (channels 13–14).

    Exposes:
        batch['image']: [15, 96, 96] float32, z-score normalized
        batch['mask']:  [96, 96] int64, values 0–7 + 255 (ignore_index)
    """

    def __init__(self, data_root: str | Path, split: str, target_size: int | None = None):
        """
        Args:
            data_root: Path to the extracted DFC2020 directory (contains metadata.csv).
            split: One of 'train', 'val', 'test'.
            target_size: Optional spatial resize (bilinear). None = keep native 96×96.
        """
        self.root = Path(data_root)
        self.target_size = target_size

        meta = pd.read_csv(self.root / 'metadata.csv')
        self.samples = meta[meta['split'] == split].reset_index(drop=True)

        # Pre-build normalization tensors (cpu)
        self._s2_mean = torch.tensor(DFC2020_S2_MEAN, dtype=torch.float32).view(-1, 1, 1)
        self._s2_std  = torch.tensor(DFC2020_S2_STD,  dtype=torch.float32).view(-1, 1, 1)
        self._s1_mean = torch.tensor(DFC2020_S1_MEAN, dtype=torch.float32).view(-1, 1, 1)
        self._s1_std  = torch.tensor(DFC2020_S1_STD,  dtype=torch.float32).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        row = self.samples.iloc[idx]

        s2_raw = tifffile.imread(self.root / row['optical_path'])  # [H, W, 13]
        s1_raw = tifffile.imread(self.root / row['radar_path'])    # [H, W, 2]
        lbl    = tifffile.imread(self.root / row['label_path'])    # [H, W, 1] or [H, W]

        # HWC → CHW
        s2 = torch.from_numpy(np.transpose(s2_raw, (2, 0, 1)).astype(np.float32))  # [13, 96, 96]
        s1 = torch.from_numpy(np.transpose(s1_raw, (2, 0, 1)).astype(np.float32))  # [2,  96, 96]

        if lbl.ndim == 3:
            lbl = lbl[:, :, 0]
        mask = torch.from_numpy(_CLASSES_LUT[lbl.astype(np.int64)].astype(np.int64))  # [96, 96]

        # Z-score normalize
        s2 = (s2 - self._s2_mean) / (self._s2_std + 1e-6)
        s1 = (s1 - self._s1_mean) / (self._s1_std + 1e-6)

        image = torch.cat([s2, s1], dim=0)  # [15, 96, 96]

        if self.target_size is not None and image.shape[-1] != self.target_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)

        return {'image': image, 'mask': mask}


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def get_dfc2020_loaders(
    batch_size: int = 32,
    num_workers: int = 8,
    data_root: str = 'datasets/GFM-Bench/DFC2020/DFC2020',
    seed: int = 42,
    starting_modality: str = 's2',
    new_modality: str | None = 's1',
    val_fraction: float = 0.0,
) -> tuple:
    """
    Create 5 dataloaders for DFC2020 matching the SHOT interface.

    metadata.csv has train/val/test splits. val_fraction is ignored when val
    rows already exist in metadata.csv (val_fraction=0.0 default uses the
    pre-defined val split). Set val_fraction > 0 to carve val from train instead.

    Args:
        data_root: Path to extracted DFC2020 directory (contains metadata.csv).
        starting_modality: 's2' or 's1'.
        new_modality: The other modality.
        val_fraction: If > 0, carve this fraction from train as val (ignores
            metadata val split). Default 0.0 = use metadata val split as-is.

    Returns:
        train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
    """
    # Slices into the 15-channel stacked image: S2 first, S1 second
    # S2 band order: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12 (idx 0-12)
    # Sub-band groups match EuroSAT groupings: rgb/vre/nir/swir/aw
    modality_bands_dict = {
        's2':      slice(0, 13),
        's1':      slice(13, 15),
        's2_rgb':  [3, 2, 1],     # B4, B3, B2
        's2_vre':  slice(4, 7),   # B5, B6, B7
        's2_nir':  slice(7, 9),   # B8, B8A
        's2_swir': slice(10, 13), # B10, B11, B12
        's2_aw':   [0, 9],        # B1, B9
    }
    
    assert starting_modality in modality_bands_dict, \
        f"starting_modality must be one of {list(modality_bands_dict)}, got {starting_modality!r}"
    assert new_modality is None or new_modality in modality_bands_dict, \
        f"new_modality must be one of {list(modality_bands_dict)} or None, got {new_modality!r}"

    test_ds = DFC2020Dataset(data_root, split='test')

    if val_fraction > 0.0:
        # Carve val from train deterministically
        train_full = DFC2020Dataset(data_root, split='train')
        n = len(train_full)
        rng = random.Random(seed)
        indices = list(range(n))
        rng.shuffle(indices)
        n_val = int(n * val_fraction)
        val_indices   = indices[:n_val]
        train_indices = indices[n_val:]
    else:
        # Use metadata-defined splits
        train_full = DFC2020Dataset(data_root, split='train')
        val_full   = DFC2020Dataset(data_root, split='val')
        n_train = len(train_full)
        n_val   = len(val_full)
        rng = random.Random(seed)
        train_indices = list(range(n_train))
        val_indices   = list(range(n_val))
        rng.shuffle(train_indices)
        rng.shuffle(val_indices)
        # val subsets are drawn from val_full, train subsets from train_full
        val_full_for_split = val_full

    # Split train 50/50 for train1 (stage-0) vs train2 (SSL)
    mid_tr = len(train_indices) // 2
    train1_ds = Subset(train_full, train_indices[:mid_tr])
    train2_ds = Subset(train_full, train_indices[mid_tr:])

    # Split val 50/50
    mid_val = len(val_indices) // 2
    if val_fraction > 0.0:
        val1_ds = Subset(train_full, val_indices[:mid_val])
        val2_ds = Subset(train_full, val_indices[mid_val:])
    else:
        val1_ds = Subset(val_full_for_split, val_indices[:mid_val])
        val2_ds = Subset(val_full_for_split, val_indices[mid_val:])

    print(f"DFC2020 — Train1: {len(train1_ds)}, Train2: {len(train2_ds)}, "
          f"Val1: {len(val1_ds)}, Val2: {len(val2_ds)}, Test: {len(test_ds)}")

    timeout = 120 if num_workers > 0 else 0
    train1_loader = DataLoader(train1_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, timeout=timeout)
    val1_loader   = DataLoader(val1_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers//2)
    train2_loader = DataLoader(train2_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True, timeout=timeout)
    val2_loader   = DataLoader(val2_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers//2)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers//2)

    def _bands_len(spec):
        if isinstance(spec, slice):
            return len(range(*spec.indices(15)))
        return len(spec)

    start_ch = _bands_len(modality_bands_dict[starting_modality])
    new_ch   = _bands_len(modality_bands_dict[new_modality]) if new_modality is not None else 0

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
        img_size=96,
        ignore_index=DFC2020_IGNORE_INDEX,
    )

    return train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
