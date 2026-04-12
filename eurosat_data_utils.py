"""
EuroSAT data utilities for multimodal loading and normalization.
Ensures consistent transformations across train and test sets.
"""

import torch
from torchvision import transforms
import os
from torch.utils.data import DataLoader, Subset


# TODO refactor load_split_indices here with load_split_indices in train_utils.py
def load_split_indices(split_file, dataset):
    """
    Load sample names from split file and return indices in the full dataset.

    Args:
        split_file: Path to split file (e.g., 'datasets/eurosat-train1.txt')
        dataset: EuroSAT dataset object

    Returns:
        List of indices corresponding to samples in the split file
    """
    # Load sample names from split file (they are .jpg names)
    with open(split_file, 'r') as f:
        split_samples = set(line.strip().replace('.jpg', '.tif') for line in f)

    # Find indices of these samples in the full dataset
    # dataset.samples is a list of (path, class_idx) tuples from ImageFolder
    indices = []
    for idx, (sample_path, _) in enumerate(dataset.samples):
        # Extract filename from full path (e.g., 'path/to/Forest_123.tif' -> 'Forest_123.tif')
        sample_name = os.path.basename(sample_path)
        if sample_name in split_samples:
            indices.append(idx)

    return indices


# EuroSAT band names and statistics
ALL_BAND_NAMES = [
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
    'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
]

RGB_BAND_NAMES = [
    'B04', 'B03', 'B02'
]

VRE_BAND_NAMES = [
    'B05', 'B06', 'B07'
]
NIR_BAND_NAMES = [
    'B08', 'B8A'
]
SWIR_BAND_NAMES = [
    'B10', 'B11', 'B12'
]
AW_BAND_NAMES = [
    'B01', 'B09'
]

# Modality group key to band names mapping
MODALITY_BANDS = {
    's2': ALL_BAND_NAMES,
    'rgb': RGB_BAND_NAMES,
    'vre': VRE_BAND_NAMES,
    'nir': NIR_BAND_NAMES,
    'swir': SWIR_BAND_NAMES,
    'aw': AW_BAND_NAMES
}

def get_modality_bands_dict(*modality_keys):
    """
    Get a dictionary of modality keys to their band tuples.

    Args:
        *modality_keys: Variable number of modality keys (e.g., 'rgb', 'vre', 'nir', 'swir')

    Returns:
        Dict mapping modality keys to tuples of band names

    Example:
        >>> get_modality_bands_dict('rgb', 'vre')
        {'rgb': ('B04', 'B03', 'B02'), 'vre': ('B05', 'B06', 'B07')}
    """
    return {key: tuple(MODALITY_BANDS[key]) for key in modality_keys}


BAND_DESCRIPTIONS = {
    'B01': 'Coastal Aerosol',
    'B02': 'Blue',
    'B03': 'Green',
    'B04': 'Red',
    'B05': 'Vegetation Red Edge 1',
    'B06': 'Vegetation Red Edge 2',
    'B07': 'Vegetation Red Edge 3',
    'B08': 'NIR 1',
    'B8A': 'NIR 2',
    'B09': 'Water Vapour',
    'B10': 'SWIR 1',
    'B11': 'SWIR 2',
    'B12': 'SWIR 3',
}

# Sentinel-2 band centre wavelengths in nm (for Panopticon compatibility)
# Used as channel IDs in models like Panopticon that do spectral fusion
BAND_WAVELENGTHS = {
    'B01': 442,    # Coastal Aerosol
    'B02': 492,    # Blue
    'B03': 559,    # Green
    'B04': 664,    # Red
    'B05': 704,    # VRE 1
    'B06': 740,    # VRE 2
    'B07': 782,    # VRE 3
    'B08': 827,    # NIR
    'B8A': 864,    # NIR (narrow)
    'B09': 945,    # Water Vapour
    'B10': 1613,   # SWIR 1
    'B11': 2203,   # SWIR 2
    'B12': None,   # SWIR 3 (not in Panopticon's standard list, use 2400 as approximation)
}

# Per-band min/max statistics for normalization
BAND_MINS = torch.tensor([
    1013.0, 676.0, 448.0, 247.0, 269.0, 253.0, 243.0,
    189.0, 61.0, 4.0, 33.0, 11.0, 186.0
])

BAND_MAXS = torch.tensor([
    2309.0, 4543.05, 4720.2, 5293.05, 3902.05, 4473.0, 5447.0,
    5948.05, 1829.0, 23.0, 4894.05, 4076.05, 5846.0
])

# ImageNet statistics for RGB normalization (used by DINO)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


class DictTransform:
    """Wrapper to apply torchvision transforms to dict samples."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        sample['image'] = self.transform(sample['image'])
        return sample


class NormalizerWrapper:
    """Wrapper to apply normalizer to dict samples (image key)."""
    def __init__(self, normalizer):
        self.normalizer = normalizer

    def __call__(self, sample):
        sample['image'] = self.normalizer(sample['image'])
        return sample


def get_band_indices(band_names):
    """
    Get indices for specified bands.

    Args:
        band_names: Tuple or list of band names

    Returns:
        List of indices
    """
    return [ALL_BAND_NAMES.index(b) for b in band_names]


def get_band_wavelengths(band_names):
    """
    Get wavelengths (in nm) for specified bands (for Panopticon channel IDs).

    Args:
        band_names: Tuple or list of band names (e.g., ['B04', 'B03', 'B02'])

    Returns:
        List of wavelengths in nm
    """
    wavelengths = []
    for band in band_names:
        wl = BAND_WAVELENGTHS.get(band)
        if wl is None:
            if band == 'B12':
                wl = 2400  # Approximation for SWIR 3
            else:
                raise ValueError(f"Unknown band: {band}")
        wavelengths.append(wl)
    return wavelengths


def compute_eurosat_zscore_stats():
    """
    Compute z-score statistics (mean, std) from EuroSAT training data.

    Caches results to avoid recomputation. Returns dict of {band_name: (mean, std)}.
    """
    import json
    cache_file = os.path.join(os.path.dirname(__file__), '.eurosat_zscore_stats_cache.json')

    # Check cache first
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cached = json.load(f)
            return {b: tuple(stats) for b, stats in cached.items()}

    # Compute from training data by reading .tif files from disk
    print("Computing EuroSAT z-score statistics from training data (train1 split)...")
    import json
    from rasterio.io import MemoryFile
    import rasterio
    import glob

    # Read split indices
    with open('datasets/eurosat-train1.txt', 'r') as f:
        train1_samples = set(line.strip().replace('.jpg', '.tif') for line in f)

    # Accumulate statistics per band
    sums = torch.zeros(13)
    sq_sums = torch.zeros(13)
    count = 0
    processed = 0

    # Find all .tif files in EuroSAT directory structure
    tif_files = sorted(glob.glob('ds_ers/ds/images/remote_sensing/otherDatasets/sentinel_2/tif/*/*.tif'))
    if not tif_files:
        raise FileNotFoundError(
            "No .tif files found in EuroSAT structure. "
            "Please ensure EuroSAT is downloaded to ds_ers/"
        )

    for tif_path in tif_files:
        filename = os.path.basename(tif_path)
        if filename not in train1_samples:
            continue

        try:
            with rasterio.open(tif_path) as src:
                image = torch.tensor(src.read(), dtype=torch.float32)  # [13, H, W]
                sums += image.sum(dim=(1, 2))
                sq_sums += (image ** 2).sum(dim=(1, 2))
                count += image.shape[1] * image.shape[2]
                processed += 1
        except Exception as e:
            print(f"Warning: Failed to read {tif_path}: {e}")
            continue

    if processed == 0:
        raise RuntimeError(
            f"No training samples processed. Found {len(train1_samples)} samples in split, "
            f"but couldn't read any from {len(tif_files)} .tif files."
        )

    print(f"Processed {processed} training images.")
    means = sums / count
    variances = (sq_sums / count) - (means ** 2)
    stds = torch.sqrt(torch.clamp(variances, min=1e-8))

    # Build result dict and cache
    result = {}
    for i, band in enumerate(ALL_BAND_NAMES):
        result[band] = (means[i].item(), stds[i].item())

    with open(cache_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"Cached z-score stats to {cache_file}")
    return result


class ZScoreNormalizer:
    """Z-score normalizer for EuroSAT data (with min-max clipping first)."""

    def __init__(self, stats_dict=None, mins=BAND_MINS, maxs=BAND_MAXS):
        """
        Args:
            stats_dict: Dict of {band_name: (mean, std)} for z-score stats.
                       If None, computes from training data.
            mins: Min values for min-max clipping (per band)
            maxs: Max values for min-max clipping (per band)
        """
        if stats_dict is None:
            stats_dict = compute_eurosat_zscore_stats()
        self.stats = stats_dict
        self.band_names = ALL_BAND_NAMES
        self.mins = mins
        self.maxs = maxs

    def __call__(self, image):
        """
        Apply min-max clipping followed by z-score normalization.

        Pipeline:
        1. Clip to [min, max] per band
        2. Z-score normalize: (value - mean) / std

        Args:
            image: Tensor of shape [13, H, W] or [B, 13, H, W]

        Returns:
            Normalized tensor
        """
        is_batched = image.ndim == 4
        if not is_batched:
            image = image.unsqueeze(0)

        normalized = image.clone().float()

        for i, band in enumerate(self.band_names):
            # Step 1: Min-max clipping
            min_val = self.mins[i].item() if self.mins.ndim > 0 else self.mins
            max_val = self.maxs[i].item() if self.maxs.ndim > 0 else self.maxs
            normalized[:, i] = torch.clamp(normalized[:, i], min_val, max_val)

            # Step 2: Z-score normalization
            mean, std = self.stats[band]
            normalized[:, i] = (normalized[:, i] - mean) / (std + 1e-8)

        if not is_batched:
            normalized = normalized.squeeze(0)

        return normalized


def normalize_bands(image, band_indices, mins=BAND_MINS, maxs=BAND_MAXS):
    """
    Normalize bands to [0, 1] using min-max normalization (DEPRECATED - use z-score instead).

    Args:
        image: Tensor of shape [B, C, H, W] or [C, H, W]
        band_indices: List of band indices to extract and normalize
        mins: Tensor of min values for all bands
        maxs: Tensor of max values for all bands

    Returns:
        Normalized tensor for selected bands
    """
    # Extract selected bands
    if len(image.shape) == 4:  # Batched
        selected_image = image[:, band_indices, :, :]
        mins_expanded = mins[band_indices].view(1, -1, 1, 1).to(image.device)
        maxs_expanded = maxs[band_indices].view(1, -1, 1, 1).to(image.device)
    else:  # Single image
        selected_image = image[band_indices, :, :]
        mins_expanded = mins[band_indices].view(-1, 1, 1).to(image.device)
        maxs_expanded = maxs[band_indices].view(-1, 1, 1).to(image.device)

    # Min-max normalize to [0, 1]
    normalized = (selected_image - mins_expanded) / (maxs_expanded - mins_expanded + 1e-8)
    normalized = torch.clamp(normalized, 0.0, 1.0)

    return normalized


def apply_imagenet_normalization(rgb_image):
    """
    Apply ImageNet normalization (DINO expects this for RGB).

    Args:
        rgb_image: Tensor of shape [B, 3, H, W] or [3, H, W] normalized to [0, 1]

    Returns:
        ImageNet-normalized tensor
    """
    if len(rgb_image.shape) == 4:  # Batched
        mean = IMAGENET_MEAN.view(1, 3, 1, 1).to(rgb_image.device)
        std = IMAGENET_STD.view(1, 3, 1, 1).to(rgb_image.device)
    else:  # Single image
        mean = IMAGENET_MEAN.view(3, 1, 1).to(rgb_image.device)
        std = IMAGENET_STD.view(3, 1, 1).to(rgb_image.device)

    return (rgb_image - mean) / std


class MultiModalTransform:
    """
    Transform that creates multimodal batch from EuroSAT data.
    Ensures consistent normalization across train and test sets.
    """

    def __init__(
        self,
        bands_rgb=('B04', 'B03', 'B02'),
        bands_infrared=('B08', 'B8A', 'B09', 'B10'),
        modalities=('rgb', 'infrared'),
        mins=BAND_MINS,
        maxs=BAND_MAXS
    ):
        """
        Args:
            bands_rgb: Tuple of RGB band names
            bands_infrared: Tuple of infrared band names
            modalities: Tuple of modality keys to include
            mins: Min values for normalization
            maxs: Max values for normalization
        """
        self.bands_rgb = bands_rgb
        self.bands_infrared = bands_infrared
        self.modalities = modalities
        self.mins = mins
        self.maxs = maxs

        # Pre-compute band indices
        self.rgb_indices = get_band_indices(bands_rgb) if 'rgb' in modalities else None
        self.infrared_indices = get_band_indices(bands_infrared) if 'infrared' in modalities else None

    def __call__(self, sample):
        """
        Transform a sample into multimodal format.

        Args:
            sample: Dict with 'image' key containing tensor [13, H, W]

        Returns:
            Dict with modality keys (e.g., {'rgb': tensor, 'infrared': tensor})
        """
        image = sample['image']  # [13, H, W]
        result = {}

        if 'rgb' in self.modalities:
            # Normalize RGB to [0, 1]
            rgb_normalized = normalize_bands(image, self.rgb_indices, self.mins, self.maxs)
            # Apply ImageNet normalization
            rgb_final = apply_imagenet_normalization(rgb_normalized)
            result['rgb'] = rgb_final

        if 'infrared' in self.modalities:
            # Normalize infrared to [0, 1]
            infrared_normalized = normalize_bands(image, self.infrared_indices, self.mins, self.maxs)
            result['infrared'] = infrared_normalized

        # Keep label if present
        if 'label' in sample:
            result['label'] = sample['label']

        return result


_geobench_rgb_stats = {'min': float('inf'), 'max': float('-inf'), 'sum': 0.0, 'count': 0, 'batches': 0}

def print_and_reset_rgb_stats():
    s = _geobench_rgb_stats
    if s['batches'] == 0:
        return
    mean = s['sum'] / s['count'] if s['count'] > 0 else float('nan')
    print(f"  [RGB post-clip stats over {s['batches']} batches] "
          f"min={s['min']:.3f}  max={s['max']:.3f}  mean={mean:.3f}")
    s['min'] = float('inf'); s['max'] = float('-inf')
    s['sum'] = 0.0; s['count'] = 0; s['batches'] = 0


def create_multimodal_batch(
    batch,
    bands_rgb=None,
    bands_newmod=None,
    mins=BAND_MINS,
    maxs=BAND_MAXS,
    modalities=('rgb',),
    modality_bands_dict=None
):
    """
    Create a multimodal batch from EuroSAT batch.
    This is a functional API for on-the-fly batch transformation.

    Args:
        batch: EuroSAT batch dict with 'image' key [B, 13, H, W]
        bands_rgb: (Deprecated) Tuple of RGB band names - use modality_bands_dict instead
        bands_newmod: (Deprecated) Tuple of new modality band names - use modality_bands_dict instead
        mins: Min values for normalization
        maxs: Max values for normalization
        modalities: Tuple of modality keys to include (e.g., ('rgb', 'vre'))
        modality_bands_dict: Dict mapping modality names to their band tuples
                            e.g., {'rgb': ('B04', 'B03', 'B02'), 'vre': ('B05', 'B06', 'B07')}
                            If provided, overrides bands_rgb and bands_newmod

    Returns:
        Dict with requested modality keys
    """
    # GeoBench datasets pre-normalize and stack all channels into a single 'image'
    # tensor. The caller passes modality_bands_dict with slice values instead of
    # band-name tuples (e.g. {'s2': slice(0, 10), 's1': slice(10, 16)}).
    # Detect this case and bypass the band-index lookup path.
    if modality_bands_dict is not None and any(
        isinstance(v, (slice, list)) for v in modality_bands_dict.values()
    ):
        image = batch['image']  # [B, C_total, H, W] — already normalized
        result = {}
        for mod in modalities:
            x = image[:, modality_bands_dict[mod], :, :]
            if mod == 'rgb':
                # Clip to [0, S2_RGB_CLIP] and rescale to [0, 1] before ImageNet norm.
                # S2 surface reflectance (/10000) is physically in [0,1] but concentrates
                # below 0.2; clipping fills the full range for DINOv2 compatibility.
                S2_RGB_CLIP = 0.2
                x = (x.clamp(0.0, S2_RGB_CLIP) / S2_RGB_CLIP)
                _geobench_rgb_stats['min'] = min(_geobench_rgb_stats['min'], x.min().item())
                _geobench_rgb_stats['max'] = max(_geobench_rgb_stats['max'], x.max().item())
                _geobench_rgb_stats['sum'] += x.float().sum().item()
                _geobench_rgb_stats['count'] += x.numel()
                _geobench_rgb_stats['batches'] += 1
                x = apply_imagenet_normalization(x)
            result[mod] = x
        return result

    image = batch['image']  # [B, 13, H, W]
    result = {}

    # Use modality_bands_dict if provided, otherwise fall back to old parameters for backwards compatibility
    if modality_bands_dict is not None:
        bands_dict = modality_bands_dict
    else:
        # Backwards compatibility: construct dict from old parameters
        bands_dict = {}
        if bands_rgb is not None:
            bands_dict['rgb'] = bands_rgb
        if bands_newmod is not None:
            # For backwards compatibility, assume the first non-rgb modality uses bands_newmod
            non_rgb_mods = [m for m in modalities if m != 'rgb']
            if non_rgb_mods:
                bands_dict[non_rgb_mods[0]] = bands_newmod

    for modality in modalities:
        if modality not in bands_dict:
            raise ValueError(f"Modality {modality} not found in bands dictionary. Available: {list(bands_dict.keys())}")

        bands = bands_dict[modality]
        indices = get_band_indices(bands)
        normalized = normalize_bands(image, indices, mins, maxs)

        # Apply ImageNet normalization only for RGB
        if modality == 'rgb':
            normalized = apply_imagenet_normalization(normalized)

        result[modality] = normalized

    return result


def get_default_transform(target_size=224):
    """
    Get default transform for EuroSAT dataset (resize only).

    Args:
        target_size: Target image size (default: 224 for DINO)

    Returns:
        DictTransform that resizes images
    """
    return DictTransform(
        transforms.Resize(
            target_size,
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=True
        )
    )


def get_loaders(bs=32,nw=4):
    bands_full = tuple(ALL_BAND_NAMES)
    resize_transform = DictTransform(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))

    train_dataset_full = EuroSAT(
        root='datasets',
        split='train',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    train1_indices = load_split_indices('datasets/eurosat-train1.txt', train_dataset_full)
    train1_dataset = Subset(train_dataset_full, train1_indices)
    train2_indices = load_split_indices('datasets/eurosat-train2.txt', train_dataset_full)
    train2_dataset = Subset(train_dataset_full, train2_indices)

    test_dataset_full = EuroSAT(
        root='datasets',
        split='test',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    print(f"Loaded {len(train1_indices)} and {len(train2_indices)} samples from train1 and train2 splits.")
    print(f"Test samples: {len(test_dataset_full)}")

    # Create dataloaders
    train1_loader = DataLoader(train1_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    train2_loader = DataLoader(train2_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    test_loader = DataLoader(test_dataset_full, batch_size=bs, shuffle=False, num_workers=nw)
    return train1_loader, train2_loader, test_loader


def get_loaders_with_val(bs=32, nw=4, seed=42, raw_pixels=False):
    """
    Get dataloaders with validation splits.

    Uses the default EuroSAT val split, divided into val1 and val2.
    Stage 1 (self-supervised on multimodal): uses train2, validated on val2
    Stage 2 (pseudo-supervised on monomodal): uses train1, validated on val1

    Data is z-score normalized by default. Pass raw_pixels=True to skip normalization
    and return raw DN values (e.g. for OlmoEarth which applies its own normalization).

    Args:
        bs: Batch size
        nw: Number of workers
        seed: Random seed for reproducible val split
        raw_pixels: If True, skip z-score normalization (return raw DN values)

    Returns:
        train1_loader: Labeled training data (monomodal, for stage 2)
        val1_loader: Validation for stage 2 (half of default val split)
        train2_loader: Unlabeled training data (multimodal, for stage 1)
        val2_loader: Validation for stage 1 (half of default val split)
        test_loader: Test data
    """
    import random
    from torchgeo.datasets import EuroSAT

    bands_full = tuple(ALL_BAND_NAMES)
    # Apply resize + optional z-score normalization transforms
    resize_transform = DictTransform(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))
    if raw_pixels:
        combined_transform = resize_transform
    else:
        zscore_normalizer = ZScoreNormalizer()
        normalizer_transform = NormalizerWrapper(zscore_normalizer)
        combined_transform = lambda sample: normalizer_transform(resize_transform(sample))

    train_dataset_full = EuroSAT(
        root='ds_ers',
        split='train',
        bands=bands_full,
        transforms=combined_transform,
        download=True,
        checksum=False
    )

    train1_indices = load_split_indices('datasets/eurosat-train1.txt', train_dataset_full)
    train2_indices = load_split_indices('datasets/eurosat-train2.txt', train_dataset_full)

    # Load the default validation split
    val_dataset_full = EuroSAT(
        root='ds_ers',
        split='val',
        bands=bands_full,
        transforms=combined_transform,
        download=True,
        checksum=False
    )

    # Split validation set into val1 and val2 (half each)
    rng = random.Random(seed)
    val_indices = list(range(len(val_dataset_full)))
    rng.shuffle(val_indices)
    val1_indices = val_indices[:len(val_indices) // 2]
    val2_indices = val_indices[len(val_indices) // 2:]

    train1_dataset = Subset(train_dataset_full, train1_indices)
    train2_dataset = Subset(train_dataset_full, train2_indices)
    val1_dataset = Subset(val_dataset_full, val1_indices)
    val2_dataset = Subset(val_dataset_full, val2_indices)

    test_dataset_full = EuroSAT(
        root='ds_ers',
        split='test',
        bands=bands_full,
        transforms=combined_transform,
        download=True,
        checksum=False
    )

    print(f"Train1 samples: {len(train1_indices)}, Train2 samples: {len(train2_indices)}")
    print(f"Val1 samples: {len(val1_indices)}, Val2 samples: {len(val2_indices)} (from default val split)")
    print(f"Test samples: {len(test_dataset_full)}")

    # Create dataloaders
    train1_loader = DataLoader(train1_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    val1_loader = DataLoader(val1_dataset, batch_size=bs, shuffle=False, num_workers=nw)
    train2_loader = DataLoader(train2_dataset, batch_size=bs, shuffle=True, num_workers=nw)
    val2_loader = DataLoader(val2_dataset, batch_size=bs, shuffle=False, num_workers=nw)
    test_loader = DataLoader(test_dataset_full, batch_size=bs, shuffle=False, num_workers=nw)
    return train1_loader, val1_loader, train2_loader, val2_loader, test_loader