"""
GeoBench-v2 data utilities for multimodal loading.

Provides dataloaders for BEN-v2 and PASTIS that match the 5-loader interface
expected by shot_ete.py:
    train1_loader, val1_loader, train2_loader, val2_loader, test_loader

Batches are normalized within the dataset and stacked into a single 'image'
tensor with a '_modality_slices' key so create_multimodal_batch() in
eurosat_data_utils.py can slice modalities without band-name lookups.
"""

import random
import sys
import os
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, Subset

# Add GEO-Bench-2 to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'GEO-Bench-2'))

from geobench_v2.datasets.benv2 import GeoBenchBENV2
from geobench_v2.datasets.pastis import GeoBenchPASTIS
from geobench_v2.datasets.normalization import ZScoreNormalizer


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Dataset/task descriptor passed through the training pipeline."""
    dataset_name: str       # 'eurosat', 'benv2', 'pastis'
    task_type: str          # 'classification', 'multilabel', 'segmentation'
    modality_a: str         # starting modality key, e.g. 's2'
    modality_b: str         # new modality key, e.g. 's1'
    modality_a_channels: int
    modality_b_channels: int
    num_classes: int
    multilabel: bool
    label_key: str          # 'label' or 'mask'
    modality_slices: dict   # {modality_name: slice} into the stacked image tensor
    img_size: int           # spatial size after any resizing


# ---------------------------------------------------------------------------
# StackedModalityDataset
# ---------------------------------------------------------------------------

class StackedModalityDataset(Dataset):
    """
    Wraps a GeoBench dataset and stacks per-modality image tensors into a
    single 'image' tensor along the channel dimension.

    The dataset exposes:
        batch['image']             : [C_total, H, W]  (already z-score normalized)
        batch['label']             : [num_classes] binary float (multi-label)
                                  OR [1] int (single-label)
        batch['mask']              : [H, W] int (segmentation only)
        batch['_modality_slices']  : dict[str, slice]  -- slices into image

    Args:
        dataset: Instantiated GeoBench dataset.
        modality_stack_order: Ordered list of modality keys whose image tensors
            will be concatenated, e.g. ['s2', 's1'] or ['s2', 's1_asc', 's1_desc'].
        merge_modalities: Optional dict mapping output modality name -> list of
            source keys to concatenate. E.g. {'s1': ['s1_asc', 's1_desc']} merges
            the two SAR passes into one 's1' modality.
    """

    def __init__(
        self,
        dataset,
        modality_stack_order: list[str],
        merge_modalities: dict[str, list[str]] | None = None,
    ):
        self.dataset = dataset
        self.modality_stack_order = modality_stack_order
        self.merge_modalities = merge_modalities or {}

        # Determine output modality names and their slices.
        # After merging, the effective modality list may differ from stack_order.
        self.modality_slices: dict[str, slice] = {}
        offset = 0

        # Build a reverse map: source_key -> output_modality_name
        source_to_output: dict[str, str] = {}
        for out_name, sources in self.merge_modalities.items():
            for src in sources:
                source_to_output[src] = out_name

        # Walk through stack order, accumulate slices for each output modality
        output_channels: dict[str, int] = {}
        for src_key in modality_stack_order:
            out_key = source_to_output.get(src_key, src_key)
            # We can't know channel count at init time without a sample,
            # so we defer slice computation to _build_slices() called lazily.

        # We'll build slices on first __getitem__ call.
        self._slices_built = False
        self._n_channels: dict[str, int] = {}

    def _build_slices(self, sample: dict):
        """Compute modality_slices from the first sample."""
        offset = 0
        # We need to handle merging: collect channel counts per source key
        src_channels: dict[str, int] = {}
        for src_key in self.modality_stack_order:
            img_key = f'image_{src_key}'
            if img_key in sample:
                c = sample[img_key].shape[0]
                src_channels[src_key] = c

        # Build reverse map: source_key -> output_modality_name
        source_to_output: dict[str, str] = {}
        for out_name, sources in self.merge_modalities.items():
            for src in sources:
                source_to_output[src] = out_name

        # Walk through stack order and accumulate
        output_start: dict[str, int] = {}
        output_end: dict[str, int] = {}
        for src_key in self.modality_stack_order:
            if src_key not in src_channels:
                continue
            out_key = source_to_output.get(src_key, src_key)
            c = src_channels[src_key]
            if out_key not in output_start:
                output_start[out_key] = offset
            output_end[out_key] = offset + c
            offset += c

        self.modality_slices = {
            k: slice(output_start[k], output_end[k])
            for k in output_start
        }
        self._slices_built = True

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index]

        if not self._slices_built:
            self._build_slices(sample)

        # Build reverse map for merging
        source_to_output: dict[str, str] = {}
        for out_name, sources in self.merge_modalities.items():
            for src in sources:
                source_to_output[src] = out_name

        # Collect tensors in stack order, grouped by output modality
        output_tensors: dict[str, list[Tensor]] = {}
        for src_key in self.modality_stack_order:
            img_key = f'image_{src_key}'
            if img_key not in sample:
                continue
            out_key = source_to_output.get(src_key, src_key)
            if out_key not in output_tensors:
                output_tensors[out_key] = []
            output_tensors[out_key].append(sample[img_key])

        # Concatenate and stack into a single image tensor
        parts = []
        for src_key in self.modality_stack_order:
            out_key = source_to_output.get(src_key, src_key)
            if out_key in output_tensors and output_tensors[out_key]:
                # Add the first time we encounter this output key
                if output_tensors[out_key] is not None:
                    parts.append(torch.cat(output_tensors[out_key], dim=0))
                    output_tensors[out_key] = None  # mark as consumed

        # Filter None placeholders
        parts = [p for p in parts if p is not None]
        image = torch.cat(parts, dim=0)  # [C_total, H, W]

        result = {'image': image}

        if 'label' in sample:
            result['label'] = sample['label']
        if 'mask' in sample:
            result['mask'] = sample['mask']

        return result


# ---------------------------------------------------------------------------
# create_multimodal_batch_geobench
# ---------------------------------------------------------------------------

def create_multimodal_batch_geobench(
    batch: dict,
    modality_slices: dict[str, slice],
    modalities: tuple[str, ...],
) -> dict[str, Tensor]:
    """
    Slice a pre-normalized stacked image tensor into per-modality tensors.

    Args:
        batch: Batch dict with 'image' key [B, C_total, H, W] (already normalized).
        modality_slices: Maps modality name -> slice into channel dim.
        modalities: Which modalities to extract.

    Returns:
        Dict {modality: [B, C_mod, H, W]}.
    """
    image = batch['image']  # [B, C_total, H, W]
    return {
        mod: image[:, modality_slices[mod], :, :]
        for mod in modalities
    }


# ---------------------------------------------------------------------------
# BEN-v2 loaders
# ---------------------------------------------------------------------------

# BEN-v2 S2 bands (12 bands, same order as GeoBenchBENV2.band_default_order)
BENV2_S2_BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')
BENV2_S1_BANDS = ('VV', 'VH')


def get_benv2_loaders(
    batch_size: int = 64,
    num_workers: int = 8,
    data_root: str = 'datasets/geoben2/benv2',
    seed: int = 42,
) -> tuple:
    """
    Create 5 dataloaders for BEN-v2 matching the SHOT interface.

    Returns:
        train1_loader: S2-only, labeled (for CE loss / stage0)
        val1_loader:   S2-only, labeled (for peeking accuracy validation)
        train2_loader: S2+S1, labeled (labels not used in SHOT; for distill/MAE)
        val2_loader:   S2+S1, labeled (for teacher-agreement validation)
        test_loader:   S2+S1, labeled (for evaluation)
        task_config:   TaskConfig describing this dataset/task
    """
    root = Path(data_root)

    # --- S2-only datasets (train1, val1) ---
    train_s2 = GeoBenchBENV2(
        root=root,
        split='train',
        band_order={'s2': list(BENV2_S2_BANDS)},
        data_normalizer=ZScoreNormalizer,
    )
    val_full_s2 = GeoBenchBENV2(
        root=root,
        split='val',
        band_order={'s2': list(BENV2_S2_BANDS)},
        data_normalizer=ZScoreNormalizer,
    )

    # --- S2+S1 datasets (train2, val2, test) ---
    train_s2s1 = GeoBenchBENV2(
        root=root,
        split='train',
        band_order={'s2': list(BENV2_S2_BANDS), 's1': list(BENV2_S1_BANDS)},
        data_normalizer=ZScoreNormalizer,
    )
    val_full_s2s1 = GeoBenchBENV2(
        root=root,
        split='val',
        band_order={'s2': list(BENV2_S2_BANDS), 's1': list(BENV2_S1_BANDS)},
        data_normalizer=ZScoreNormalizer,
    )
    test_s2s1 = GeoBenchBENV2(
        root=root,
        split='test',
        band_order={'s2': list(BENV2_S2_BANDS), 's1': list(BENV2_S1_BANDS)},
        data_normalizer=ZScoreNormalizer,
    )

    # Wrap in StackedModalityDataset
    train1_ds = StackedModalityDataset(train_s2, modality_stack_order=['s2'])
    train2_ds = StackedModalityDataset(train_s2s1, modality_stack_order=['s2', 's1'])
    test_ds = StackedModalityDataset(test_s2s1, modality_stack_order=['s2', 's1'])

    # Split validation set 50/50 into val1 (S2-only) and val2 (S2+S1)
    rng = random.Random(seed)
    val_indices = list(range(len(val_full_s2)))
    rng.shuffle(val_indices)
    val1_indices = val_indices[:len(val_indices) // 2]
    val2_indices = val_indices[len(val_indices) // 2:]

    val1_ds = StackedModalityDataset(
        Subset(val_full_s2, val1_indices), modality_stack_order=['s2']
    )
    val2_ds = StackedModalityDataset(
        Subset(val_full_s2s1, val2_indices), modality_stack_order=['s2', 's1']
    )

    print(f"BEN-v2 — Train: {len(train1_ds)} (S2-only), {len(train2_ds)} (S2+S1)")
    print(f"BEN-v2 — Val1: {len(val1_ds)} (S2-only), Val2: {len(val2_ds)} (S2+S1)")
    print(f"BEN-v2 — Test: {len(test_ds)} (S2+S1)")

    train1_loader = DataLoader(train1_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val1_loader   = DataLoader(val1_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train2_loader = DataLoader(train2_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val2_loader   = DataLoader(val2_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build modality_slices from a sample (trigger lazy init)
    _ = train2_ds[0]
    modality_slices = train2_ds.modality_slices  # {'s2': slice(0,12), 's1': slice(12,14)}

    task_config = TaskConfig(
        dataset_name='benv2',
        task_type='multilabel',
        modality_a='s2',
        modality_b='s1',
        modality_a_channels=len(BENV2_S2_BANDS),
        modality_b_channels=len(BENV2_S1_BANDS),
        num_classes=GeoBenchBENV2.num_classes,
        multilabel=True,
        label_key='label',
        modality_slices=modality_slices,
        img_size=120,
    )

    return train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config


# ---------------------------------------------------------------------------
# PASTIS loaders
# ---------------------------------------------------------------------------

# PASTIS S2 bands (10 bands)
PASTIS_S2_BANDS = ('B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B11', 'B12')
# PASTIS S1: asc + desc, each 3 channels — merged into single 's1' (6ch)
PASTIS_S1_ASC_BANDS = ('VV_asc', 'VH_asc', 'VV/VH_asc')
PASTIS_S1_DESC_BANDS = ('VV_desc', 'VH_desc', 'VV/VH_desc')


def get_pastis_loaders(
    batch_size: int = 32,
    num_workers: int = 8,
    data_root: str = 'datasets/geoben2/pastis',
    seed: int = 42,
    temporal_aggregation: str = 'mean',
) -> tuple:
    """
    Create 5 dataloaders for PASTIS (semantic segmentation) matching the SHOT interface.

    S1 ascending + descending passes are concatenated into a single 's1' modality
    (6 channels: VV_asc, VH_asc, VV/VH_asc, VV_desc, VH_desc, VV/VH_desc).

    Args:
        temporal_aggregation: How to collapse time dimension. 'mean' or 'median'.
            Using 'mean' keeps things simple; revisit for temporal modeling later.

    Returns:
        train1_loader: S2-only, with semantic segmentation masks
        val1_loader:   S2-only, with masks
        train2_loader: S2+S1, with masks
        val2_loader:   S2+S1, with masks
        test_loader:   S2+S1, with masks
        task_config:   TaskConfig describing this dataset/task
    """
    root = Path(data_root)

    common_kwargs = dict(
        temporal_aggregation=temporal_aggregation,
        label_type='semantic_seg',
        data_normalizer=ZScoreNormalizer,
    )

    # --- S2-only datasets (train1, val1) ---
    train_s2 = GeoBenchPASTIS(
        root=root, split='train',
        band_order={'s2': list(PASTIS_S2_BANDS)},
        **common_kwargs,
    )
    val_full_s2 = GeoBenchPASTIS(
        root=root, split='val',
        band_order={'s2': list(PASTIS_S2_BANDS)},
        **common_kwargs,
    )

    # --- S2+S1 datasets (train2, val2, test) ---
    s2s1_band_order = {
        's2':     list(PASTIS_S2_BANDS),
        's1_asc': list(PASTIS_S1_ASC_BANDS),
        's1_desc': list(PASTIS_S1_DESC_BANDS),
    }
    train_s2s1 = GeoBenchPASTIS(
        root=root, split='train',
        band_order=s2s1_band_order,
        **common_kwargs,
    )
    val_full_s2s1 = GeoBenchPASTIS(
        root=root, split='val',
        band_order=s2s1_band_order,
        **common_kwargs,
    )
    test_s2s1 = GeoBenchPASTIS(
        root=root, split='test',
        band_order=s2s1_band_order,
        **common_kwargs,
    )

    # Merge s1_asc + s1_desc → 's1' in StackedModalityDataset
    s1_merge = {'s1': ['s1_asc', 's1_desc']}

    train1_ds = StackedModalityDataset(train_s2, modality_stack_order=['s2'])
    train2_ds = StackedModalityDataset(
        train_s2s1,
        modality_stack_order=['s2', 's1_asc', 's1_desc'],
        merge_modalities=s1_merge,
    )
    test_ds = StackedModalityDataset(
        test_s2s1,
        modality_stack_order=['s2', 's1_asc', 's1_desc'],
        merge_modalities=s1_merge,
    )

    # Split validation 50/50
    rng = random.Random(seed)
    val_indices = list(range(len(val_full_s2)))
    rng.shuffle(val_indices)
    val1_indices = val_indices[:len(val_indices) // 2]
    val2_indices = val_indices[len(val_indices) // 2:]

    val1_ds = StackedModalityDataset(
        Subset(val_full_s2, val1_indices),
        modality_stack_order=['s2'],
    )
    val2_ds = StackedModalityDataset(
        Subset(val_full_s2s1, val2_indices),
        modality_stack_order=['s2', 's1_asc', 's1_desc'],
        merge_modalities=s1_merge,
    )

    print(f"PASTIS — Train: {len(train1_ds)} (S2-only), {len(train2_ds)} (S2+S1)")
    print(f"PASTIS — Val1: {len(val1_ds)} (S2-only), Val2: {len(val2_ds)} (S2+S1)")
    print(f"PASTIS — Test: {len(test_ds)} (S2+S1)")

    train1_loader = DataLoader(train1_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val1_loader   = DataLoader(val1_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    train2_loader = DataLoader(train2_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val2_loader   = DataLoader(val2_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader   = DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Build modality_slices from a sample
    _ = train2_ds[0]
    modality_slices = train2_ds.modality_slices  # {'s2': slice(0,10), 's1': slice(10,16)}

    n_s1 = len(PASTIS_S1_ASC_BANDS) + len(PASTIS_S1_DESC_BANDS)
    task_config = TaskConfig(
        dataset_name='pastis',
        task_type='segmentation',
        modality_a='s2',
        modality_b='s1',
        modality_a_channels=len(PASTIS_S2_BANDS),
        modality_b_channels=n_s1,
        num_classes=GeoBenchPASTIS.num_classes,
        multilabel=False,
        label_key='mask',
        modality_slices=modality_slices,
        img_size=128,
    )

    return train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config
