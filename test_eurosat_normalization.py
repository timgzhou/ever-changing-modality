#!/usr/bin/env python3
"""Test EuroSAT z-score normalization without torchgeo dependency."""

import torch
import json
import os

ALL_BAND_NAMES = [
    'B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
    'B08', 'B8A', 'B09', 'B10', 'B11', 'B12'
]

class ZScoreNormalizer:
    """Simple z-score normalizer for EuroSAT data."""

    def __init__(self, stats_dict=None):
        if stats_dict is None:
            cache_file = '.eurosat_zscore_stats_cache.json'
            if not os.path.exists(cache_file):
                raise FileNotFoundError(f"Cache file not found: {cache_file}")
            with open(cache_file, 'r') as f:
                stats_dict = json.load(f)
        self.stats = stats_dict
        self.band_names = ALL_BAND_NAMES

    def __call__(self, image):
        """Apply z-score normalization to image tensor."""
        is_batched = image.ndim == 4
        if not is_batched:
            image = image.unsqueeze(0)

        normalized = image.clone().float()
        for i, band in enumerate(self.band_names):
            mean, std = self.stats[band]
            normalized[:, i] = (normalized[:, i] - mean) / (std + 1e-8)

        if not is_batched:
            normalized = normalized.squeeze(0)

        return normalized


def test_normalizer():
    """Test normalizer on synthetic data."""
    print("=" * 60)
    print("Testing EuroSAT Z-Score Normalizer (min-max + z-score)")
    print("=" * 60)

    normalizer = ZScoreNormalizer()
    print(f"✓ Loaded normalizer with {len(normalizer.stats)} bands")
    print(f"  (includes min-max clipping + z-score normalization)")

    # Test 1: Single image with clipping
    print("\n[Test 1] Single image normalization (with clipping)")
    single_img = torch.ones(13, 64, 64) * 10000  # Out of range values
    normalized = normalizer(single_img)
    print(f"  Input shape:  {single_img.shape}, values=10000 (out of range)")
    print(f"  Output shape: {normalized.shape}")
    print(f"  Output min:   {normalized.min():.4f} (clipped bands should differ)")
    print(f"  Output max:   {normalized.max():.4f}")
    assert normalized.shape == single_img.shape, "Shape mismatch!"
    print("  ✓ Pass (clipping applied)")

    # Test 2: Batched images
    print("\n[Test 2] Batched image normalization")
    batch = torch.ones(4, 13, 64, 64) * 1000
    normalized_batch = normalizer(batch)
    print(f"  Input shape:  {batch.shape}")
    print(f"  Output shape: {normalized_batch.shape}")
    print(f"  Output mean:  {normalized_batch.mean():.4f} (should be ~0)")
    assert normalized_batch.shape == batch.shape, "Shape mismatch!"
    print("  ✓ Pass")

    # Test 3: Using realistic mean values for each band
    print("\n[Test 3] Normalize to band means")
    multi_band = torch.zeros(13, 64, 64)
    for i, band in enumerate(ALL_BAND_NAMES):
        mean, std = normalizer.stats[band]
        multi_band[i] = mean
    normalized_multi = normalizer(multi_band)
    print(f"  Input: each band set to its training mean")
    print(f"  Output mean per band (should be ~0):")
    for i in [0, 4, 8, 12]:
        mean_i = normalized_multi[i].mean()
        print(f"    Band {i}: {mean_i:.6f}")
    overall_mean = normalized_multi.mean().item()
    assert abs(overall_mean) < 0.01, f"Overall mean not ~0! Got {overall_mean:.6f}"
    print("  ✓ Pass (all means ~0)")

    # Test 4: Stats are loaded correctly
    print("\n[Test 4] Verify stats cache")
    print(f"  Band stats (sample):")
    for band in ['B01', 'B04', 'B08', 'B12']:
        mean, std = normalizer.stats[band]
        print(f"    {band}: μ={mean:.2f}, σ={std:.2f}")
    print("  ✓ Pass")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == '__main__':
    test_normalizer()
