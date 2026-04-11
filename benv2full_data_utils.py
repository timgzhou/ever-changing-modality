"""
BigEarthNet v2 data utilities for multimodal loading and normalization.

Supports scene-level multi-label classification with Sentinel-1 (SAR) and Sentinel-2 (optical).
Designed to work with extracted BigEarthNet-S2 and BigEarthNet-S1 directories.

Data structure:
  BigEarthNet-S2/
    [TILE_NAME]/
      [PATCH_ID]/
        B01.tif, B02.tif, ..., B12.tif (12 bands)

  BigEarthNet-S1/
    [SCENE_NAME]/
      [PATCH_ID_WITH_GRID]/
        VV.tif, VH.tif (2 bands)

Patches are matched by grid coordinates (e.g., 'T29UPU_72_03').
Labels come from metadata.parquet (scene-level multi-label).
"""

import random
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

from data_utils import TaskConfig


# BigEarthNet v2 normalization statistics (from GEO-Bench-2)
# Computed on the full dataset: z-score normalization
BEN_NORMALIZATION_STATS = {
    "means": {
        "B01": 355.96197509765625,
        "B02": 414.3730773925781,
        "B03": 594.096435546875,
        "B04": 559.0433959960938,
        "B05": 919.4099731445312,
        "B06": 1794.6605224609375,
        "B07": 2091.45947265625,
        "B08": 2241.517822265625,
        "B8A": 2288.0302734375,
        "B09": 2289.5380859375,
        "B11": 1556.958740234375,
        "B12": 973.8273315429688,
        "VV": -18.96333885192871,
        "VH": -12.091922760009766,
    },
    "stds": {
        "B01": 512.3419799804688,
        "B02": 541.94921875,
        "B03": 532.579833984375,
        "B04": 607.0200805664062,
        "B05": 646.341064453125,
        "B06": 1041.35009765625,
        "B07": 1231.787841796875,
        "B08": 1340.4661865234375,
        "B8A": 1316.02880859375,
        "B09": 1267.3955078125,
        "B11": 984.2933349609375,
        "B12": 753.2081909179688,
        "VV": 5.396073818206787,
        "VH": 4.574888229370117,
    },
}

# BigEarthNet v2 class names (19 CORINE Land Cover classes)
BEN_CLASS_NAMES = (
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
)

# Band definitions for S2 sub-modalities (matching EuroSAT groupings)
S2_BAND_NAMES = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
S2_RGB_BANDS = ("B04", "B03", "B02")
S2_VRE_BANDS = ("B05", "B06", "B07")
S2_NIR_BANDS = ("B08", "B8A")
S2_SWIR_BANDS = ("B11", "B12")
S2_AW_BANDS = ("B01", "B09")


class BigEarthNetDataset(Dataset):
    """
    Loads BigEarthNet v2 patches with S2+S1 imagery and multi-label scene classification labels.

    Reads GeoTIFF bands from extracted directories, applies z-score normalization,
    and stacks into a single image tensor [C_total, H, W].

    Exposes:
        batch['image']:    [C_total, H, W] float32, z-score normalized
        batch['label']:    [19] binary float (multi-label one-hot)
    """

    def __init__(
        self,
        s2_root: str | Path,
        s1_root: str | Path,
        metadata_path: str | Path,
        split: Literal["train", "val", "test"],
        target_size: Optional[int] = None,
    ):
        """
        Args:
            s2_root: Path to BigEarthNet-S2 directory
            s1_root: Path to BigEarthNet-S1 directory
            metadata_path: Path to metadata.parquet with labels
            split: Dataset split ('train', 'val', 'test')
            target_size: Optional spatial resize. None = keep native 120×120.
        """
        self.s2_root = Path(s2_root)
        self.s1_root = Path(s1_root)
        self.target_size = target_size

        # Load metadata
        meta_df = pd.read_parquet(metadata_path, engine='fastparquet')
        self.metadata = meta_df[meta_df["split"] == split].reset_index(drop=True)

        print(f"BigEarthNet {split}: {len(self.metadata)} samples")

        # Build class index
        self.class2idx = {c: i for i, c in enumerate(BEN_CLASS_NAMES)}

        # Build normalization tensors
        self._s2_mean = torch.tensor(
            [BEN_NORMALIZATION_STATS["means"][b] for b in S2_BAND_NAMES],
            dtype=torch.float32,
        ).view(-1, 1, 1)
        self._s2_std = torch.tensor(
            [BEN_NORMALIZATION_STATS["stds"][b] for b in S2_BAND_NAMES],
            dtype=torch.float32,
        ).view(-1, 1, 1)
        self._s1_mean = torch.tensor(
            [BEN_NORMALIZATION_STATS["means"]["VV"], BEN_NORMALIZATION_STATS["means"]["VH"]],
            dtype=torch.float32,
        ).view(-1, 1, 1)
        self._s1_std = torch.tensor(
            [BEN_NORMALIZATION_STATS["stds"]["VV"], BEN_NORMALIZATION_STATS["stds"]["VH"]],
            dtype=torch.float32,
        ).view(-1, 1, 1)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        patch_id = row["patch_id"]  # e.g., 'S2A_MSIL2A_20170613T101031_..._T33UUP_26_57'

        # Extract grid coordinates from patch_id (last 3 parts joined, e.g., 'T33UUP_26_57')
        parts = patch_id.split('_')
        grid_coords = '_'.join(parts[-3:])  # Last 3 parts: tile + row + col

        # Find matching S2 patch directory (glob by grid coords)
        s2_patches = list(self.s2_root.glob(f"*/*{grid_coords}"))
        if not s2_patches:
            raise FileNotFoundError(f"S2 patch not found for grid coords {grid_coords}")
        s2_patch_dir = s2_patches[0]

        # Load S2 bands (with resampling to common size)
        s2_bands = []
        target_h, target_w = None, None

        # First pass: find target size (10m resolution bands)
        for band in S2_BAND_NAMES:
            band_path = s2_patch_dir / f"{s2_patch_dir.name}_{band}.tif"
            if not band_path.exists():
                raise FileNotFoundError(f"Band {band} not found at {band_path}")
            with rasterio.open(band_path) as src:
                if target_h is None:
                    target_h, target_w = src.height, src.width
                    break

        # Second pass: load and resample all bands to target size
        for band in S2_BAND_NAMES:
            band_path = s2_patch_dir / f"{s2_patch_dir.name}_{band}.tif"
            with rasterio.open(band_path) as src:
                band_data = src.read(1).astype(np.float32)  # [H, W]

            # Resample if needed
            if band_data.shape != (target_h, target_w):
                band_tensor = torch.from_numpy(band_data).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
                band_tensor = F.interpolate(
                    band_tensor, size=(target_h, target_w), mode='bilinear', align_corners=False
                )
                band_data = band_tensor.squeeze(0).squeeze(0).numpy()

            s2_bands.append(torch.from_numpy(band_data.astype(np.float32)))

        s2 = torch.stack(s2_bands, dim=0)  # [12, H, W]

        # Find matching S1 patch directory (glob by grid coords)
        # If S1 extraction is incomplete, use zeros as placeholder
        s1_patches = list(self.s1_root.glob(f"*/*{grid_coords}"))
        if s1_patches:
            s1_patch_dir = s1_patches[0]
            # Load S1 bands (VV, VH)
            s1_bands = []
            for band in ["VV", "VH"]:
                band_path = s1_patch_dir / f"{s1_patch_dir.name}_{band}.tif"
                if not band_path.exists():
                    raise FileNotFoundError(f"Band {band} not found at {band_path}")
                with rasterio.open(band_path) as src:
                    band_data = src.read(1)
                s1_bands.append(torch.from_numpy(band_data.astype(np.float32)))
            s1 = torch.stack(s1_bands, dim=0)  # [2, H, W]
        else:
            # S1 extraction incomplete — use zeros as placeholder
            # (Only safe for stage 0 when you're training on S2 only)
            s1 = torch.zeros(2, s2.shape[1], s2.shape[2], dtype=torch.float32)

        # Z-score normalize
        s2 = (s2 - self._s2_mean) / (self._s2_std + 1e-6)
        s1 = (s1 - self._s1_mean) / (self._s1_std + 1e-6)

        # Stack into single image
        image = torch.cat([s2, s1], dim=0)  # [14, H, W]

        # Optionally resize
        if self.target_size is not None and image.shape[-1] != self.target_size:
            image = F.interpolate(
                image.unsqueeze(0),
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        # Load label (multi-label one-hot)
        label_names = row["labels"]  # List of class names
        label = torch.zeros(len(BEN_CLASS_NAMES), dtype=torch.float32)
        for class_name in label_names:
            if class_name in self.class2idx:
                label[self.class2idx[class_name]] = 1.0

        return {"image": image, "label": label}


def get_bigearthnet_loaders(
    batch_size: int = 32,
    num_workers: int = 8,
    s2_root: str = "datasets/BigEarthNet-S2",
    s1_root: str = "datasets/BigEarthNet-S1",
    metadata_path: str = "datasets/reBEN/metadata.parquet",
    seed: int = 42,
    starting_modality: str = "s2",
    new_modality: Optional[str] = "s1",
) -> tuple:
    """
    Create 5 dataloaders for BigEarthNet v2 matching the SHOT interface.

    All loaders expose the full S2+S1 stacked image tensor. Modality selection
    is done at training time via create_multimodal_batch() using modality_slices.

    train1/val1 are drawn from train split (50/50).
    train2/val2 are drawn from val split (50/50).
    test is the test split.

    Args:
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
        s2_root: Path to BigEarthNet-S2 directory
        s1_root: Path to BigEarthNet-S1 directory
        metadata_path: Path to metadata.parquet
        seed: Random seed for reproducible splits
        starting_modality: Initial modality ('s2', 's2_rgb', 's2_vre', etc.)
        new_modality: Modality to add in stage 2 ('s1', etc.)

    Returns:
        train1_loader: S2+S1, labeled (train split, for stage 0)
        val1_loader:   S2+S1, labeled (first half of val split)
        train2_loader: S2+S1, labeled (train split, for SHOT SSL)
        val2_loader:   S2+S1, labeled (second half of val split)
        test_loader:   S2+S1, labeled (test split)
        task_config:   TaskConfig describing this dataset/task
    """
    # Load metadata to understand structure
    meta_df = pd.read_parquet(metadata_path, engine='fastparquet')

    # Create full datasets
    train_full = BigEarthNetDataset(s2_root, s1_root, metadata_path, split="train", target_size=120)
    val_full = BigEarthNetDataset(s2_root, s1_root, metadata_path, split="val", target_size=120)
    test_full = BigEarthNetDataset(s2_root, s1_root, metadata_path, split="test", target_size=120)

    # Split train 50/50 (no overlap, deterministic)
    rng = random.Random(seed)
    train_indices = list(range(len(train_full)))
    rng.shuffle(train_indices)
    train1_indices = train_indices[: len(train_indices) // 2]
    train2_indices = train_indices[len(train_indices) // 2 :]

    # Split val 50/50
    val_indices = list(range(len(val_full)))
    rng.shuffle(val_indices)
    val1_indices = val_indices[: len(val_indices) // 2]
    val2_indices = val_indices[len(val_indices) // 2 :]

    train1_ds = Subset(train_full, train1_indices)
    train2_ds = Subset(train_full, train2_indices)
    val1_ds = Subset(val_full, val1_indices)
    val2_ds = Subset(val_full, val2_indices)

    print(
        f"BigEarthNet v2 — Train1: {len(train1_ds)}, Train2: {len(train2_ds)}, "
        f"Val1: {len(val1_ds)}, Val2: {len(val2_ds)}, Test: {len(test_full)}"
    )

    timeout = 120 if num_workers > 0 else 0
    train1_loader = DataLoader(
        train1_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, timeout=timeout
    )
    val1_loader = DataLoader(val1_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    train2_loader = DataLoader(
        train2_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, timeout=timeout
    )
    val2_loader = DataLoader(val2_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_full, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build modality_slices (S2 starts at 0, S1 starts at 12)
    modality_slices = {
        "s2": slice(0, 12),
        "s1": slice(12, 14),
        "s2_rgb": [3, 2, 1],       # B04, B03, B02
        "s2_vre": slice(4, 7),     # B05, B06, B07
        "s2_nir": slice(7, 9),     # B08, B8A
        "s2_swir": slice(10, 12),  # B11, B12
        "s2_aw": [0, 9],           # B01, B09
    }

    assert starting_modality in modality_slices, \
        f"starting_modality must be one of {list(modality_slices)}, got {starting_modality!r}"
    assert new_modality is None or new_modality in modality_slices, \
        f"new_modality must be one of {list(modality_slices)} or None, got {new_modality!r}"

    def _bands_len(spec, total_ch=14):
        if isinstance(spec, slice):
            return len(range(*spec.indices(total_ch)))
        return len(spec)

    start_ch = _bands_len(modality_slices[starting_modality])
    new_ch = _bands_len(modality_slices[new_modality]) if new_modality is not None else 0

    task_config = TaskConfig(
        dataset_name="bigearthnet_v2",
        task_type="classification",
        modality_a=starting_modality,
        modality_b=new_modality,
        modality_a_channels=start_ch,
        modality_b_channels=new_ch,
        num_classes=len(BEN_CLASS_NAMES),
        multilabel=True,
        label_key="label",
        modality_bands_dict=modality_slices,
        img_size=120,
    )

    return train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
