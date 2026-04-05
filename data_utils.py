"""
Unified data utilities: TaskConfig, create_multimodal_batch, and get_loaders.

This module is the single entry point for dataset loading across train_stage0.py,
shot_ete.py, and baseline scripts. The individual dataset files
(eurosat_data_utils.py, geobench_data_utils.py) handle dataset-specific logic;
this module provides the dispatch layer and shared types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from torch.utils.data import DataLoader
import torch


# ---------------------------------------------------------------------------
# TaskConfig
# ---------------------------------------------------------------------------

@dataclass
class TaskConfig:
    """Dataset/task descriptor passed through the training pipeline."""
    dataset_name: str           # 'eurosat', 'benv2', 'pastis'
    task_type: str              # 'classification', 'multilabel', 'segmentation'
    modality_a: str             # starting modality key, e.g. 's2'
    modality_b: str             # new modality key, e.g. 's1'
    modality_a_channels: int
    modality_b_channels: int
    num_classes: int
    multilabel: bool
    label_key: str              # 'label' or 'mask'
    modality_bands_dict: dict   # {modality: slice} for GeoBench, {modality: tuple[str]} for EuroSAT
    img_size: int
    ignore_index: int = -100    # label value to ignore in loss/metric


# ---------------------------------------------------------------------------
# create_multimodal_batch
# ---------------------------------------------------------------------------

def create_multimodal_batch(
    batch: dict,
    modality_bands_dict: dict,
    modalities: tuple[str, ...],
) -> dict[str, torch.Tensor]:
    """
    Extract per-modality tensors from a batch.

    Handles two cases transparently:
    - GeoBench: modality_bands_dict values are slice or list objects.
      batch['image'] is [B, C_total, H, W], already z-score normalized.
    - EuroSAT: modality_bands_dict values are tuples of band name strings.
      batch['image'] is [B, 13, H, W], raw DN values requiring normalization.

    Args:
        batch: Batch dict with 'image' key.
        modality_bands_dict: Maps modality name -> slice/list (GeoBench) or
            tuple of band name strings (EuroSAT).
        modalities: Which modalities to extract.

    Returns:
        Dict {modality: [B, C_mod, H, W]}.
    """
    first_val = next(iter(modality_bands_dict.values()))
    if isinstance(first_val, (slice, list)):
        # GeoBench path: slice pre-normalized stacked image
        image = batch['image']  # [B, C_total, H, W]
        result = {}
        for mod in modalities:
            x = image[:, modality_bands_dict[mod], :, :]
            if mod == 'rgb':
                # S2 RGB: clip to [0, 0.2] then rescale to [0,1] for DINOv2 compatibility
                from eurosat_data_utils import apply_imagenet_normalization, _geobench_rgb_stats
                S2_RGB_CLIP = 0.2
                x = x.clamp(0.0, S2_RGB_CLIP) / S2_RGB_CLIP
                _geobench_rgb_stats['min'] = min(_geobench_rgb_stats['min'], x.min().item())
                _geobench_rgb_stats['max'] = max(_geobench_rgb_stats['max'], x.max().item())
                _geobench_rgb_stats['sum'] += x.float().sum().item()
                _geobench_rgb_stats['count'] += x.numel()
                _geobench_rgb_stats['batches'] += 1
                x = apply_imagenet_normalization(x)
            result[mod] = x
        return result
    else:
        # EuroSAT path: band-name lookup + min-max normalization
        from eurosat_data_utils import (
            get_band_indices, normalize_bands, apply_imagenet_normalization,
            BAND_MINS, BAND_MAXS,
        )
        image = batch['image']  # [B, 13, H, W]
        result = {}
        for mod in modalities:
            bands = modality_bands_dict[mod]
            indices = get_band_indices(bands)
            normalized = normalize_bands(image, indices, BAND_MINS, BAND_MAXS)
            if mod == 'rgb':
                normalized = apply_imagenet_normalization(normalized)
            result[mod] = normalized
        return result


# ---------------------------------------------------------------------------
# get_loaders
# ---------------------------------------------------------------------------

def get_loaders(
    dataset: str,
    starting_modality: str,
    batch_size: int,
    num_workers: int,
    data_normalizer=None,
    num_time_steps: int = 10,
    new_modality: str = None,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, DataLoader, TaskConfig]:
    """
    Return the standard 5-loader tuple plus TaskConfig for a given dataset.

    Args:
        dataset: One of 'eurosat', 'benv2', 'pastis'.
        starting_modality: Modality available at stage 0.
            EuroSAT: 'rgb' | 'vre' | 'nir' | 'swir' | 'aw'
            BEN-v2:  's2' | 's1'
            PASTIS:  's2' | 's1' | 'rgb'
        batch_size: DataLoader batch size.
        num_workers: DataLoader worker count.
        data_normalizer: Optional normalizer override (e.g. div10000 for PASTIS+DINO).
        num_time_steps: PASTIS only — timestamps to sample before temporal aggregation.
        new_modality: Optional override for the new modality. If None, inferred as the
            "other" modality for two-modality datasets (BEN-v2, PASTIS) or must be
            provided for EuroSAT.

    Returns:
        train1_loader: Starting modality, labeled (stage 0 / CE loss)
        val1_loader:   Starting modality, labeled (validation)
        train2_loader: Both modalities, labeled (SSL / distillation)
        val2_loader:   Both modalities, labeled (validation)
        test_loader:   Both modalities, labeled (evaluation)
        task_config:   TaskConfig describing this dataset/task.
    """
    if dataset == 'eurosat':
        return _get_eurosat_loaders(starting_modality, new_modality, batch_size, num_workers)
    elif dataset == 'benv2':
        from geobench_data_utils import get_benv2_loaders
        return get_benv2_loaders(
            batch_size=batch_size, num_workers=num_workers,
            starting_modality=starting_modality,
        )
    elif dataset == 'pastis':
        from geobench_data_utils import get_pastis_loaders
        _new = new_modality or ('s1' if starting_modality == 's2' else 's2')
        return get_pastis_loaders(
            batch_size=batch_size, num_workers=num_workers,
            starting_modality=starting_modality, new_modality=_new,
            data_normalizer=data_normalizer, num_time_steps=num_time_steps,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Valid: 'eurosat', 'benv2', 'pastis'")


def _get_eurosat_loaders(starting_modality, new_modality, batch_size, num_workers):
    from eurosat_data_utils import get_loaders_with_val, get_modality_bands_dict, MODALITY_BANDS

    train1, val1, train2, val2, test = get_loaders_with_val(batch_size, num_workers)

    mods = (starting_modality,) if new_modality is None else (starting_modality, new_modality)
    modality_bands_dict = get_modality_bands_dict(*mods)

    task_config = TaskConfig(
        dataset_name='eurosat',
        task_type='classification',
        modality_a=starting_modality,
        modality_b=new_modality or '',
        modality_a_channels=len(MODALITY_BANDS[starting_modality]),
        modality_b_channels=len(MODALITY_BANDS[new_modality]) if new_modality else 0,
        num_classes=10,
        multilabel=False,
        label_key='label',
        modality_bands_dict=modality_bands_dict,
        img_size=224,
    )
    return train1, val1, train2, val2, test, task_config


