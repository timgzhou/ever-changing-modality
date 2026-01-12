"""
EuroSAT data utilities for multimodal loading and normalization.
Ensures consistent transformations across train and test sets.
"""

import torch
from torchvision import transforms


# EuroSAT band names and statistics
ALL_BAND_NAMES = [
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
    'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
]

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


def get_band_indices(band_names):
    """
    Get indices for specified bands.

    Args:
        band_names: Tuple or list of band names

    Returns:
        List of indices
    """
    return [ALL_BAND_NAMES.index(b) for b in band_names]


def normalize_bands(image, band_indices, mins=BAND_MINS, maxs=BAND_MAXS):
    """
    Normalize bands to [0, 1] using min-max normalization.

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


def create_multimodal_batch(
    batch,
    bands_rgb=('B04', 'B03', 'B02'),
    bands_infrared=('B08', 'B8A', 'B09', 'B10'),
    mins=BAND_MINS,
    maxs=BAND_MAXS,
    modalities=('rgb', 'infrared')
):
    """
    Create a multimodal batch from EuroSAT batch.
    This is a functional API for on-the-fly batch transformation.

    Args:
        batch: EuroSAT batch dict with 'image' key [B, 13, H, W]
        bands_rgb: Tuple of RGB band names
        bands_infrared: Tuple of infrared band names
        mins: Min values for normalization
        maxs: Max values for normalization
        modalities: Tuple of modality keys to include

    Returns:
        Dict with requested modality keys
    """
    image = batch['image']  # [B, 13, H, W]
    result = {}

    if 'rgb' in modalities:
        # Get RGB indices and normalize
        rgb_indices = get_band_indices(bands_rgb)
        rgb_normalized = normalize_bands(image, rgb_indices, mins, maxs)
        # Apply ImageNet normalization
        rgb_final = apply_imagenet_normalization(rgb_normalized)
        result['rgb'] = rgb_final

    if 'infrared' in modalities:
        # Get infrared indices and normalize
        infrared_indices = get_band_indices(bands_infrared)
        infrared_normalized = normalize_bands(image, infrared_indices, mins, maxs)
        result['infrared'] = infrared_normalized

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
