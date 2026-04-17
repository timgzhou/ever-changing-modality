"""
Fine-tune Remote Sensing Foundation Models (OlmoEarth, etc.) on split 1 via LP or FFT.

Loads a pretrained model from HuggingFace, trains on split 1 of any supported dataset,
evaluates on split 2 test.

IMPORTANT: Model Input Format & Wrappers

  Data Flow:
  - Loaders provide batch['image'] as a stacked tensor [B, C_total, H, W].
  - For single-modality training (e.g., --modality s2), code slices via modality_bands_dict.
  - Result: imgs=[B, C_modality, H, W] ready for the model.

  Models & Wrappers:
  - OlmoEarth (HF): Standard transformer, wrapped in OlmoEarthWrapper
    → expects [B, C, H, W]
    → returns output.last_hidden_state (CLS token [B, 768] or feature map [B, D, H, W])

  - Panopticon (torch.hub): Expects special dict interface, wrapped in PanopticonWrapper
    → PanopticonWrapper automatically builds chn_ids from modality (s2, s1, s2s1)
    → internally calls model(dict(imgs=[B,C,H,W], chn_ids=[B,C]))
    → returns [B, 768] L2-normalized CLS token

  Custom models: Add a new ModelWrapper subclass in load_foundation_model().

Supports:
  - Classification (EuroSAT, BEN-v2)
  - Multilabel (BEN-v2)
  - Segmentation (PASTIS)

Architecture:
  - create_classification_head(): BatchNorm1d + Linear (follows EVANClassifier)
  - create_segmentation_head(): BatchNorm2d + Conv2d 1×1 (follows EvanSegmenter)

Metrics:
  - Classification: Accuracy
  - Multilabel: mAP (average precision per class)
  - Segmentation: mIoU (Jaccard score)

Usage:
  # OlmoEarth LP on BEN-v2 S2
  python rsfm_sft.py --model allenai/OlmoEarth-v1-Base --dataset benv2 --modality s2 --train_mode lp --epochs 50

  # Panopticon LP on BEN-v2 S2 (to verify against lp_panopticon_benv2.py baseline)
  python rsfm_sft.py --model panopticon --dataset benv2 --modality s2 --train_mode lp --epochs 50

  # Panopticon LP on BEN-v2 S1
  python rsfm_sft.py --model panopticon --dataset benv2 --modality s1 --train_mode lp --epochs 50

  # Panopticon LP on BEN-v2 S2+S1
  python rsfm_sft.py --model panopticon --dataset benv2 --modality s2s1 --train_mode lp --epochs 50
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import argparse
from datetime import datetime
import wandb
import csv
from typing import Optional
import warnings
warnings.filterwarnings('ignore', message='Use of index_put_')

from train_utils import _compute_map, compute_miou

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODALITY_CONFIGS = {
    'eurosat': ['s2', 'rgb', 'vre', 'nir', 'swir', 'aw'],  # s2 = all 13 bands
    'benv2': ['s2', 's1', 's2s1', 's2_rgb', 's2_vre', 's2_nir', 's2_swir', 's2_aw'],
    'benv2full': ['s2', 's1', 's2s1', 's2_rgb', 's2_vre', 's2_nir', 's2_swir', 's2_aw'],
    'pastis': ['s2', 's1', 's2s1', 'rgb', 's2_rgb', 's2_vre', 's2_nir', 's2_swir'],
    'dfc2020': ['s2', 's1', 's2s1', 's2_rgb', 's2_vre', 's2_nir', 's2_swir', 's2_aw'],
}

# Short model names → HuggingFace repo IDs for DINOv3 ViT models
DINO_MODELS = {
    'dinov3-vitb-lvd': 'facebook/dinov3-vitb16-pretrain-lvd1689m',
    'dinov3-vitl-lvd': 'facebook/dinov3-vitl16-pretrain-lvd1689m',
    'dinov3-vitl-sat': 'facebook/dinov3-vitl16-pretrain-sat493m',
}
# Per-model normalization stats (applied after min-max [0,1] → z-score)
# LVD models: standard ImageNet stats; SAT model: satellite-specific stats
DINO_NORM_STATS = {
    'dinov3-vitb-lvd': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'dinov3-vitl-lvd': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'dinov3-vitl-sat': ((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)),
}
DINO_RGB_MODALITIES = {'rgb', 's2_rgb'}


def _n_chans(entry) -> int:
    """Return channel count for a modality_bands_dict entry (slice or list)."""
    if isinstance(entry, slice):
        return entry.stop - entry.start
    return len(entry)


class ModelOutput:
    """Simple output container to mimic HF transformers output format."""

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class ModelWrapper(nn.Module):
    """Base wrapper to provide consistent [B, C, H, W] → features interface for all models."""

    def forward(self, imgs):
        raise NotImplementedError


class OlmoEarthWrapper(ModelWrapper):
    """
    Wraps OlmoEarth encoder to accept raw-DN [B, C, H, W] tensors.

    Applies OlmoEarth per-band clip normalization internally, then builds the
    MaskedOlmoEarthSample the encoder expects (single timestep, no missing tokens).

    forward() returns ModelOutput with:
      - classification path (output_hidden_states=False): last_hidden_state [B, 768]
        (global avg pool of spatial feature map)
      - segmentation path  (output_hidden_states=True):  last_hidden_state [B, 768, H/p, W/p]
        (spatial feature map, ready for segmentation head + bilinear upsample)
    """

    # OlmoEarth band order for sentinel2_l2a (must match normalization config keys)
    S2_BAND_ORDER_12 = ['B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09']
    # DFC2020 has 13 S2 bands (includes B10 at index 10 in DFC2020's layout)
    # DFC2020 S2 layout: B1,B2,B3,B4,B5,B6,B7,B8,B8A,B9,B10,B11,B12
    S2_BAND_ORDER_13 = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    S1_BAND_ORDER    = ['vv', 'vh']  # OlmoEarth sentinel1 band names

    # For each sub-band modality: which bands are present and their indices in S2_BAND_ORDER_12
    # S2_BAND_ORDER_12 = [B02,B03,B04,B08,B05,B06,B07,B8A,B11,B12,B01,B09]
    #                      idx: 0    1    2    3    4    5    6    7    8    9   10   11
    _S2_SUBBAND_INDICES = {
        'rgb':    ([2, 1, 0],  ['B04', 'B03', 'B02']),  # input order: B04,B03,B02
        's2_rgb': ([2, 1, 0],  ['B04', 'B03', 'B02']),
        'vre':    ([4, 5, 6],  ['B05', 'B06', 'B07']),
        's2_vre': ([4, 5, 6],  ['B05', 'B06', 'B07']),
        'nir':    ([3, 7],     ['B08', 'B8A']),
        's2_nir': ([3, 7],     ['B08', 'B8A']),
        'swir':   ([8, 9],     ['B11', 'B12']),          # B10 not in OlmoEarth; B11,B12 at 8,9
        's2_swir':([8, 9],     ['B11', 'B12']),
        'aw':     ([10, 11],   ['B01', 'B09']),
        's2_aw':  ([10, 11],   ['B01', 'B09']),
    }

    def __init__(self, model, patch_size: int = 8, dataset: str = 'benv2', modality: str = 's2'):
        super().__init__()
        self.encoder = model
        self.patch_size = patch_size
        self.dataset = dataset
        self.modality = modality

        # Load OlmoEarth normalization stats
        from olmoearth_pretrain.data.normalize import load_computed_config
        from olmoearth_pretrain.datatypes import MaskValue
        norm_cfg = load_computed_config()
        self._maskvalue_online = MaskValue.ONLINE_ENCODER.value

        std_mult = 2.0
        s2_cfg = norm_cfg['sentinel2_l2a']
        s1_cfg = norm_cfg['sentinel1']

        # Build per-channel [lo, hi] for clip normalization: (x - lo) / (hi - lo)
        # s2 — 12-band order (benv2/pastis)
        lo12 = [s2_cfg[b]['mean'] - std_mult * s2_cfg[b]['std'] for b in self.S2_BAND_ORDER_12]
        hi12 = [s2_cfg[b]['mean'] + std_mult * s2_cfg[b]['std'] for b in self.S2_BAND_ORDER_12]
        self.register_buffer('s2_lo12', torch.tensor(lo12, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('s2_hi12', torch.tensor(hi12, dtype=torch.float32).view(1, -1, 1, 1))

        # s2 — 13-band order (dfc2020)
        # B10 has no OlmoEarth stats (not in training data); use B09 stats as fallback
        s2_13_bands = self.S2_BAND_ORDER_13
        lo13, hi13 = [], []
        for b in s2_13_bands:
            if b in s2_cfg:
                lo13.append(s2_cfg[b]['mean'] - std_mult * s2_cfg[b]['std'])
                hi13.append(s2_cfg[b]['mean'] + std_mult * s2_cfg[b]['std'])
            else:
                # B10 fallback: use B09 range
                lo13.append(s2_cfg['B09']['mean'] - std_mult * s2_cfg['B09']['std'])
                hi13.append(s2_cfg['B09']['mean'] + std_mult * s2_cfg['B09']['std'])
                logger.warning(f"OlmoEarth norm stats missing for band {b}, using B09 as fallback")
        self.register_buffer('s2_lo13', torch.tensor(lo13, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('s2_hi13', torch.tensor(hi13, dtype=torch.float32).view(1, -1, 1, 1))

        # s1 — OlmoEarth expects dB values; DFC2020 already stores dB, BEN-v2 also stores dB
        lo_s1 = [s1_cfg[b]['mean'] - std_mult * s1_cfg[b]['std'] for b in self.S1_BAND_ORDER]
        hi_s1 = [s1_cfg[b]['mean'] + std_mult * s1_cfg[b]['std'] for b in self.S1_BAND_ORDER]
        self.register_buffer('s1_lo', torch.tensor(lo_s1, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer('s1_hi', torch.tensor(hi_s1, dtype=torch.float32).view(1, -1, 1, 1))

    def _normalize_clip(self, x, lo, hi):
        """Clip-normalize: (x - lo) / (hi - lo), clamped to [0, 1]."""
        return ((x - lo) / (hi - lo + 1e-6)).clamp(0.0, 1.0)

    def _build_sample(self, imgs):
        """
        Build MaskedOlmoEarthSample from [B, C, H, W] raw-DN tensor.

        Uses self.modality to know exactly which bands are present:
          - 's2':    12-band S2 (benv2/pastis layout)
          - 's2' on dfc2020: 13-band S2
          - 's1':    2-band S1 only (VV, VH)
          - 's2s1':  S2 + S1 concatenated
          - sub-band modalities (rgb, vre, nir, swir, aw, s2_rgb, ...):
            bands placed into the correct positions of a 12-band S2 tensor;
            missing bands masked as MISSING so OlmoEarth ignores them
        """
        from olmoearth_pretrain.datatypes import MaskedOlmoEarthSample, MaskValue
        B, C, H, W = imgs.shape
        device = imgs.device
        modality = self.modality
        n_band_sets = 3  # S2 has 3 band sets in OlmoEarth's Modality definition

        kwargs = {}

        if modality == 's1':
            # S1-only: no S2 input
            s1_norm = self._normalize_clip(imgs, self.s1_lo, self.s1_hi)
            kwargs['sentinel1'] = s1_norm.permute(0, 2, 3, 1).unsqueeze(3)  # [B,H,W,1,2]
            kwargs['sentinel1_mask'] = torch.full(
                (B, H, W, 1, 1), fill_value=MaskValue.ONLINE_ENCODER.value,
                dtype=torch.int32, device=device,
            )
            # Still need sentinel2_l2a (required field) — mark it all MISSING
            kwargs['sentinel2_l2a'] = torch.zeros((B, H, W, 1, 12), dtype=torch.float32, device=device)
            kwargs['sentinel2_l2a_mask'] = torch.full(
                (B, H, W, 1, n_band_sets), fill_value=MaskValue.MISSING.value,
                dtype=torch.int32, device=device,
            )

        elif modality in ('s2', 's2s1'):
            # Full S2 (12 or 13 bands), optionally followed by S1
            if self.dataset == 'dfc2020':
                s2_norm = self._normalize_clip(imgs[:, :13], self.s2_lo13, self.s2_hi13)
            else:
                s2_norm = self._normalize_clip(imgs[:, :12], self.s2_lo12, self.s2_hi12)
            kwargs['sentinel2_l2a'] = s2_norm.permute(0, 2, 3, 1).unsqueeze(3)
            kwargs['sentinel2_l2a_mask'] = torch.full(
                (B, H, W, 1, n_band_sets), fill_value=MaskValue.ONLINE_ENCODER.value,
                dtype=torch.int32, device=device,
            )
            if modality == 's2s1':
                s1_norm = self._normalize_clip(imgs[:, -2:], self.s1_lo, self.s1_hi)
                kwargs['sentinel1'] = s1_norm.permute(0, 2, 3, 1).unsqueeze(3)
                kwargs['sentinel1_mask'] = torch.full(
                    (B, H, W, 1, 1), fill_value=MaskValue.ONLINE_ENCODER.value,
                    dtype=torch.int32, device=device,
                )

        elif modality in self._S2_SUBBAND_INDICES:
            # Sub-band S2 modality: place bands into correct positions, mask the rest as MISSING
            positions, band_names = self._S2_SUBBAND_INDICES[modality]
            # Normalize each input channel with its own per-band stats from s2_lo12/s2_hi12
            s2_full = torch.zeros((B, 12, H, W), dtype=imgs.dtype, device=device)
            for in_ch, (out_pos, band_name) in enumerate(zip(positions, band_names)):
                lo = self.s2_lo12[:, out_pos:out_pos+1]
                hi = self.s2_hi12[:, out_pos:out_pos+1]
                s2_full[:, out_pos] = self._normalize_clip(imgs[:, in_ch:in_ch+1], lo, hi).squeeze(1)
            kwargs['sentinel2_l2a'] = s2_full.permute(0, 2, 3, 1).unsqueeze(3)
            # Mark all band-sets as MISSING, then flip the ones we have to ONLINE
            s2_mask = torch.full(
                (B, H, W, 1, n_band_sets), fill_value=MaskValue.MISSING.value,
                dtype=torch.int32, device=device,
            )
            # OlmoEarth band-set assignment for S2_BAND_ORDER_12:
            # set 0 (10m): B02=0, B03=1, B04=2, B08=3
            # set 1 (20m): B05=4, B06=5, B07=6, B8A=7, B11=8, B12=9
            # set 2 (60m): B01=10, B09=11
            _band_set = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2]
            active_sets = {_band_set[p] for p in positions}
            for s in active_sets:
                s2_mask[:, :, :, :, s] = MaskValue.ONLINE_ENCODER.value
            kwargs['sentinel2_l2a_mask'] = s2_mask

        else:
            raise ValueError(f"OlmoEarthWrapper: unsupported modality '{modality}'")

        # Timestamps: dummy [B, T=1, 3] — day=1, month=0 (Jan), year=2024
        timestamps = torch.zeros((B, 1, 3), dtype=torch.int32, device=device)
        timestamps[:, 0, 0] = 1
        timestamps[:, 0, 2] = 2024
        kwargs['timestamps'] = timestamps

        return MaskedOlmoEarthSample(**kwargs)

    def forward(self, imgs, output_hidden_states=False):
        """
        [B, C, H, W] raw DN → ModelOutput.

        last_hidden_state:
          - [B, 768]         when output_hidden_states=False (classification: global avg pool)
          - [B, 768, H/p, W/p] when output_hidden_states=True  (segmentation: spatial map)
        """
        sample = self._build_sample(imgs)

        with torch.amp.autocast(device_type=imgs.device.type, dtype=torch.bfloat16):
            result = self.encoder(sample, fast_pass=True, patch_size=self.patch_size)

        tokens_and_masks = result['tokens_and_masks']
        if self.modality == 's1':
            # Pool from sentinel1 tokens: [B, H/p, W/p, T, S, C] → [B, C, H/p, W/p]
            feat = tokens_and_masks.sentinel1.mean(dim=[3, 4])
        else:
            # Pool from sentinel2_l2a tokens: [B, H/p, W/p, T, S, C] → [B, C, H/p, W/p]
            feat = tokens_and_masks.sentinel2_l2a.mean(dim=[3, 4])
        feat = feat.permute(0, 3, 1, 2).float()                 # [B, C, H/p, W/p]

        if output_hidden_states:
            return ModelOutput(feat)  # [B, 768, H/p, W/p]
        else:
            return ModelOutput(feat.mean(dim=[2, 3]))  # [B, 768] global avg pool


class DinoWrapper(ModelWrapper):
    """
    Wraps a HuggingFace DINOv3 ViT to accept [B, 3, H, W] tensors.

    Two modes depending on raw_pixels:
      raw_pixels=True  (EuroSAT): applies min-max → [0,1] → ImageNet/SAT z-score internally.
      raw_pixels=False (GeoBench datasets): input is already z-score normalized by the loader;
                        internal normalization is skipped (consistent with train_sft).

    forward() returns ModelOutput with:
      - output_hidden_states=False: CLS token [B, D]
      - output_hidden_states=True:  patch tokens [B, N, D]
    """

    def __init__(self, model, model_key: str, modality: str, dataset: str, raw_pixels: bool = True):
        super().__init__()
        self.model = model
        self.modality = modality
        self.dataset = dataset
        self.raw_pixels = raw_pixels

        if raw_pixels:
            # EuroSAT path: min-max bounds for B04, B03, B02
            from eurosat_data_utils import BAND_MINS, BAND_MAXS
            rgb_indices = [3, 2, 1]  # B04=3, B03=2, B02=1 in ALL_BAND_NAMES
            self.register_buffer('band_mins', BAND_MINS[rgb_indices].view(1, 3, 1, 1))
            self.register_buffer('band_maxs', BAND_MAXS[rgb_indices].view(1, 3, 1, 1))
            mean, std = DINO_NORM_STATS[model_key]
            self.register_buffer('norm_mean', torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1))
            self.register_buffer('norm_std',  torch.tensor(std,  dtype=torch.float32).view(1, 3, 1, 1))

        # Number of register tokens (DINOv3 prepends [CLS, reg_1, ..., reg_R, patch_1, ...])
        self.num_register_tokens = getattr(model.config, 'num_register_tokens', 0)

    def forward(self, imgs, output_hidden_states=False):
        """
        [B, 3, H, W] → ModelOutput.

        last_hidden_state:
          - [B, D]    when output_hidden_states=False (CLS token, classification)
          - [B, N, D] when output_hidden_states=True  (patch tokens, segmentation)
        """
        if self.raw_pixels:
            # EuroSAT: min-max to [0, 1] then ImageNet/SAT z-score
            x = (imgs - self.band_mins) / (self.band_maxs - self.band_mins + 1e-6)
            x = x.clamp(0.0, 1.0)
            x = (x - self.norm_mean) / self.norm_std
        else:
            # GeoBench: already z-score normalized by loader — pass through as-is
            x = imgs

        output = self.model(pixel_values=x)
        # HF DINOv3: last_hidden_state is [B, 1+R+N, D]
        #   index 0:       CLS token
        #   indices 1..R:  register tokens (R = num_register_tokens, typically 4)
        #   indices R+1..: patch tokens
        tokens = output.last_hidden_state
        patch_start = 1 + self.num_register_tokens
        if output_hidden_states:
            return ModelOutput(tokens[:, patch_start:, :])   # patch tokens [B, N, D]
        else:
            return ModelOutput(tokens[:, 0, :])              # CLS token [B, D]


class PanopticonWrapper(ModelWrapper):
    """
    Wraps Panopticon to accept [B, C, H, W] and automatically build chn_ids.

    Channel IDs are Sentinel-2 band centre wavelengths (nm) or synthetic IDs for other datasets.
    """

    def __init__(self, model, modality='s2', dataset=None):
        super().__init__()
        self.model = model
        self.modality = modality
        self.dataset = dataset

        # Sentinel-2 Channel IDs (band centre wavelengths in nm)
        # 12-band S2 (BEN-v2 / BEN-v2full / PASTIS — no B10)
        self.S2_CHN_IDS = torch.tensor([442, 492, 559, 664, 704, 740, 782, 827, 864, 945, 1613, 2203],
                                       dtype=torch.int16)
        # 13-band S2 (DFC2020 — includes B10 at ~1375 nm)
        self.S2_13_CHN_IDS = torch.tensor([442, 492, 559, 664, 704, 740, 782, 827, 864, 945, 1375, 1613, 2203],
                                          dtype=torch.int16)
        self.S1_CHN_IDS = torch.tensor([-1, -2], dtype=torch.int16)
        self.S2S1_CHN_IDS = torch.cat([self.S2_CHN_IDS, self.S1_CHN_IDS])        # 14ch
        self.S2S1_13_CHN_IDS = torch.cat([self.S2_13_CHN_IDS, self.S1_CHN_IDS])  # 15ch (DFC2020)

        # Pre-compute channel IDs for EuroSAT if applicable
        self.eurosat_chn_ids = None
        if dataset == 'eurosat':
            from eurosat_data_utils import get_band_wavelengths, MODALITY_BANDS, ALL_BAND_NAMES
            if modality == 's2':
                wavelengths = get_band_wavelengths(ALL_BAND_NAMES)
            elif modality in MODALITY_BANDS:
                band_names = MODALITY_BANDS[modality]
                wavelengths = get_band_wavelengths(band_names)
            else:
                raise ValueError(f"Unknown EuroSAT modality: {modality}")
            self.eurosat_chn_ids = torch.tensor(wavelengths, dtype=torch.int16)

    def forward(self, imgs, output_hidden_states=False):
        """
        [B, C, H, W] → ModelOutput.

        last_hidden_state is:
          - [B, 768] (CLS token) when output_hidden_states=False (classification)
          - [B, N, 768] (patch tokens) when output_hidden_states=True (segmentation)

        Panopticon's channel fusion layer handles arbitrary channel counts.
        We provide chn_ids (band wavelengths in nm) for each channel.
        """
        B, C, H, W = imgs.shape
        device = imgs.device

        # Determine channel IDs based on channel count / dataset
        if C == 15:
            # DFC2020 S2(13ch)+S1(2ch)
            chn_ids = self.S2S1_13_CHN_IDS.to(device)
        elif C == 14:
            # BEN-v2 / PASTIS S2(12ch)+S1(2ch)
            chn_ids = self.S2S1_CHN_IDS.to(device)
        elif C == 13:
            # DFC2020 S2-only (13 bands including B10)
            chn_ids = self.S2_13_CHN_IDS.to(device)
        elif C == 12:
            # BEN-v2 / PASTIS S2-only (12 bands, no B10)
            chn_ids = self.S2_CHN_IDS.to(device)
        elif C == 2:
            # S1 only (VV, VH)
            chn_ids = self.S1_CHN_IDS.to(device)
        elif self.dataset == 'eurosat':
            # EuroSAT: pre-computed in __init__
            chn_ids = self.eurosat_chn_ids.to(device)
        else:
            # Unknown: use synthetic wavelength IDs
            chn_ids = torch.linspace(400, 2400, C, dtype=torch.int16).to(device)
            logger.warning(f"Unknown {C}-channel input; using synthetic wavelength IDs: {chn_ids.tolist()}")

        # Expand to batch
        chn_ids = chn_ids.unsqueeze(0).expand(B, -1)

        x_dict = dict(imgs=imgs, chn_ids=chn_ids)

        if output_hidden_states:
            # Segmentation path: return patch tokens [B, N, 768]
            feat_dict = self.model.forward_features(x_dict)
            patch_tokens = feat_dict['x_norm_patchtokens']  # [B, N, 768]
            return ModelOutput(patch_tokens)
        else:
            # Classification path: CLS token [B, 768]
            features = self.model(x_dict)
            return ModelOutput(features)


def load_foundation_model(model_name: str, device, modality='s2', dataset=None, raw_pixels=True):
    """
    Load a foundation model from HuggingFace or torch.hub.

    Supports:
      - OlmoEarth (HF): 'allenai/OlmoEarth-v1-Base'
      - Panopticon (torch.hub): 'panopticon'

    Returns (wrapped_model, feature_dim).
    """
    print(f"\n=== Loading foundation model: {model_name} ===")

    # Panopticon detection
    if model_name.lower() == 'panopticon':
        import sys
        PANOPTICON_HUB = os.path.expanduser('~/.cache/torch/hub/Panopticon-FM_panopticon_main')
        if PANOPTICON_HUB not in sys.path:
            sys.path.insert(0, PANOPTICON_HUB)

        model = torch.hub.load('Panopticon-FM/panopticon', 'panopticon_vitb14', trust_repo=True)
        model = model.to(device)
        model.eval()
        feature_dim = 768  # ViT-B/14
        wrapped = PanopticonWrapper(model, modality=modality, dataset=dataset)
        wrapped = wrapped.to(device)
        print(f"Feature dimension (Panopticon ViT-B/14): {feature_dim}")
        return wrapped, feature_dim

    # OlmoEarth detection
    if model_name.lower() in ('olmoearth', 'olmoearth-base') or 'olmoearth' in model_name.lower():
        from olmoearth_pretrain.model_loader import ModelID, load_model_from_id
        # Map short names to ModelID
        _id_map = {
            'olmoearth': ModelID.OLMOEARTH_V1_BASE,
            'olmoearth-nano': ModelID.OLMOEARTH_V1_NANO,
            'olmoearth-tiny': ModelID.OLMOEARTH_V1_TINY,
            'olmoearth-base': ModelID.OLMOEARTH_V1_BASE,
            'olmoearth-large': ModelID.OLMOEARTH_V1_LARGE,
        }
        model_id = _id_map.get(model_name.lower(), ModelID.OLMOEARTH_V1_BASE)
        print(f"Loading OlmoEarth model: {model_id}")
        raw_model = load_model_from_id(model_id)
        # Select encoder sub-module only (matches rslearn's selector=["encoder"])
        encoder = raw_model.encoder
        encoder = encoder.to(device)
        encoder.eval()
        _OLMOEARTH_EMBEDDING_SIZES = {
            'OlmoEarth-v1-Nano': 128, 'OlmoEarth-v1-Tiny': 192,
            'OlmoEarth-v1-Base': 768, 'OlmoEarth-v1-Large': 1024,
        }
        feature_dim = _OLMOEARTH_EMBEDDING_SIZES.get(str(model_id), 768)
        patch_size = 8  # good balance of spatial resolution vs speed
        wrapped = OlmoEarthWrapper(encoder, patch_size=patch_size, dataset=dataset, modality=modality)
        wrapped = wrapped.to(device)
        print(f"Feature dimension (OlmoEarth {model_id}, patch_size={patch_size}): {feature_dim}")
        return wrapped, feature_dim

    # DINOv3 detection
    if model_name in DINO_MODELS:
        if modality not in DINO_RGB_MODALITIES:
            raise ValueError(
                f"DINOv3 only supports RGB modalities {DINO_RGB_MODALITIES}, got {modality!r}"
            )
        from transformers import AutoModel
        repo_id = DINO_MODELS[model_name]
        print(f"Loading DINOv3 from {repo_id}")
        model = AutoModel.from_pretrained(repo_id)
        model = model.to(device).eval()
        feature_dim = model.config.hidden_size  # 768 for vitb, 1024 for vitl
        wrapped = DinoWrapper(model, model_key=model_name, modality=modality, dataset=dataset, raw_pixels=raw_pixels)
        wrapped = wrapped.to(device)
        print(f"Feature dimension (DINOv3 {model_name}): {feature_dim}")
        return wrapped, feature_dim

    raise ValueError(f"Unknown model: {model_name!r}. Supported: 'panopticon', 'olmoearth', {list(DINO_MODELS)}")


def create_classification_head(feature_dim: int, num_classes: int, device):
    """
    Create a classification head with BatchNorm + Linear, following EVANClassifier pattern.

    Returns nn.Sequential(BatchNorm1d, Linear)
    """
    head = nn.Sequential(
        nn.BatchNorm1d(feature_dim, affine=False),
        nn.Linear(feature_dim, num_classes)
    )
    return head.to(device)


def create_segmentation_head(feature_dim: int, num_classes: int, device):
    """
    Create a segmentation head with BatchNorm2d + Conv2d, following EvanSegmenter pattern.

    Returns nn.Sequential(BatchNorm2d, Conv2d(1×1)) for per-patch classification.
    """
    head = nn.Sequential(
        nn.BatchNorm2d(feature_dim, affine=False),
        nn.Conv2d(feature_dim, num_classes, kernel_size=1)
    )
    return head.to(device)


@torch.no_grad()
def extract_features(model, loader, device, modality_slices: dict, modality: str,
                     cache_path: Optional[str] = None, segmentation: bool = False):
    """
    Run frozen foundation model over loader, extract features and labels.

    For classification: features [N, feature_dim], labels [N] or [N, num_classes]
    For segmentation: features [N, feature_dim, H, W], labels [N, H, W]

    If cache_path exists, load from disk instead of recomputing.

    NOTE: This function assumes the model accepts [B, C, H, W] tensors directly.
    For models like Panopticon that need special input format (e.g., chn_ids), you'll
    need to wrap the model call or add model-specific handling here.
    """
    if cache_path is not None and os.path.isfile(cache_path):
        print(f'    loading cached features from {cache_path}')
        saved = torch.load(cache_path, map_location='cpu')
        return saved['feats'], saved['labels']

    all_feats = []
    all_labels = []
    model.eval()

    # Get modality slice if needed
    mod_slice = modality_slices.get(modality) if modality_slices else None

    from tqdm import tqdm
    for batch in tqdm(loader, desc=f'Extracting [{modality}]', leave=False):
        imgs = batch['image'].to(device)  # [B, C, H, W]

        # Slice to specific modality if needed
        if mod_slice is not None:
            imgs = imgs[:, mod_slice, :, :]

        # Forward pass through frozen backbone
        with torch.no_grad():
            output = model(imgs, output_hidden_states=True)

            # Handle different output formats
            if hasattr(output, 'last_hidden_state'):
                feat = output.last_hidden_state
            elif isinstance(output, tuple) and len(output) > 0:
                feat = output[0]
            else:
                feat = output

            # Extract features based on dimensionality
            if feat.dim() == 3:
                # [B, num_tokens, feature_dim] → extract CLS token [B, feature_dim]
                features = feat[:, 0, :]
            elif feat.dim() == 2:
                # [B, feature_dim] → already pooled
                features = feat
            else:
                # [B, feature_dim, H, W] → global average pool [B, feature_dim]
                features = feat.mean(dim=(2, 3))

        all_feats.append(features.cpu())

        # Handle labels
        if segmentation:
            labels = batch.get('mask', batch.get('label'))
        else:
            labels = batch['label']
        all_labels.append(labels)

    feats = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({'feats': feats, 'labels': labels}, cache_path)
        print(f'    saved features to {cache_path}')

    return feats, labels


def get_task_config_and_loaders(dataset, modality, batch_size, num_workers,
                                data_root=None, raw_pixels=False):
    """Return (train1_loader, val1_loader, test_loader, task_config, modality_bands_dict, preprocess_fn).

    preprocess_fn(batch) -> imgs [B, C, H, W]: extracts and normalizes the requested
    modality from a raw batch. For EuroSAT this applies per-band min-max normalization
    (and ImageNet norm for rgb). For GeoBench/DFC2020 datasets the batch is already
    z-score normalized by the loader, so preprocess_fn just slices the channel dim.
    For OlmoEarth/DINO (raw_pixels=True) normalization is handled inside the model wrapper,
    so preprocess_fn only slices.

    Args:
        raw_pixels: If True, skip dataset-level normalization (for models like OlmoEarth
                    that apply their own normalization internally).
    """
    from data_utils import get_loaders, create_multimodal_batch

    # s2s1 = concatenated all-bands; load with both modalities so the stacked image is available
    starting_modality = 's2' if modality == 's2s1' else modality
    new_modality = 's1' if modality == 's2s1' else None

    # data_normalizer=False signals loaders to skip normalization (OlmoEarth path)
    data_normalizer = False if raw_pixels else None

    train1, val1, _, _, test, task_config = get_loaders(
        dataset, starting_modality, batch_size, num_workers,
        data_normalizer=data_normalizer, num_time_steps=10,
        new_modality=new_modality, data_root=data_root,
    )

    if modality == 's2s1':
        # Build a slice spanning the full stacked image (s2 then s1, already concatenated)
        mbd = task_config.modality_bands_dict
        s2_sl = mbd['s2']
        s1_sl = mbd['s1']
        s2s1_slice = slice(s2_sl.start, s1_sl.stop)
        modality_bands_dict = {'s2s1': s2s1_slice}
    else:
        modality_bands_dict = {modality: task_config.modality_bands_dict[modality]}

    # Build preprocess_fn: for EuroSAT the dict values are band-name tuples, so we must
    # go through create_multimodal_batch which does min-max + optional ImageNet norm.
    # For all other datasets (GeoBench, DFC2020) dict values are slices/int-lists so
    # a plain channel slice suffices (normalization already done in the loader).
    #
    # Exception: raw_pixels=True (OlmoEarth, DINO-on-EuroSAT) means the model wrapper
    # applies its own normalization internally. For EuroSAT we must NOT call
    # create_multimodal_batch (which would normalize to [0,1]), and instead just slice
    # the raw DN channels. OlmoEarth's SWIR only uses B11+B12 (not B10).
    _EUROSAT_RAW_SLICES = {
        'rgb':  [3, 2, 1],   # B04, B03, B02
        'vre':  [4, 5, 6],   # B05, B06, B07
        'nir':  [7, 8],      # B08, B8A
        'swir': [11, 12],    # B11, B12 (drop B10 — not in OlmoEarth's band set)
    }
    first_val = next(iter(modality_bands_dict.values()))
    if isinstance(first_val, (slice, list)):
        # GeoBench / DFC2020: batch already normalized; just slice the channel dim.
        _slice = first_val  # the single modality's slice or list
        def preprocess_fn(batch):
            return batch['image'][:, _slice]
    elif raw_pixels and modality in _EUROSAT_RAW_SLICES:
        # EuroSAT + raw_pixels model (OlmoEarth, DINO): slice raw DN channels, skip normalization.
        _ch = _EUROSAT_RAW_SLICES[modality]
        def preprocess_fn(batch):
            return batch['image'][:, _ch]
    else:
        # EuroSAT: band-name tuples — delegate to create_multimodal_batch for normalization.
        _mbd = modality_bands_dict
        _mod = modality
        def preprocess_fn(batch):
            result = create_multimodal_batch(batch, modality_bands_dict=_mbd, modalities=(_mod,))
            return result[_mod]

    return train1, val1, test, task_config, modality_bands_dict, preprocess_fn


def compute_metrics(logits, labels, task_type='classification', multilabel=False,
                   num_classes=None, ignore_index=-100):
    """
    Compute metrics based on task type, using train_utils functions for consistency.

    Args:
        logits: Model outputs [B, C] or [B, C, H, W] for segmentation
        labels: Ground truth [B] or [B, C] for multilabel, [B, H, W] for segmentation
        task_type: 'classification', 'segmentation'
        multilabel: If True, compute mAP
        num_classes: Required for segmentation
        ignore_index: Label to ignore in segmentation (default -100)
    """
    if multilabel:
        # mAP for multilabel classification (uses train_utils._compute_map)
        return _compute_map(logits, labels)
    elif task_type == 'segmentation':
        # mIoU for segmentation (uses train_utils.compute_miou)
        preds = logits.argmax(dim=1)  # [B, H, W]
        return compute_miou(preds, labels, num_classes, ignore_index=ignore_index)
    else:
        # Accuracy for classification
        preds = logits.argmax(dim=1)
        acc = (preds == labels).float().mean().item()
        return acc * 100.0


def eval_head(model, head, loader, criterion, device, segmentation, multilabel,
              num_classes, ignore_index, desc='Eval', modality_slice=None,
              preprocess_fn=None):
    """Run model+head over loader, return (metric, loss). Used for val and test."""
    from tqdm import tqdm
    model.eval()
    head.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for batch in tqdm(loader, desc=desc, leave=False):
        if preprocess_fn is not None:
            imgs = preprocess_fn(batch).to(device)
        else:
            imgs = batch['image'].to(device)
            if modality_slice is not None:
                imgs = imgs[:, modality_slice]
        if segmentation:
            labels = batch.get('mask', batch.get('label')).to(device).long()
        else:
            labels = batch['label'].to(device)
            if multilabel:
                labels = labels.float()

        with torch.no_grad():
            output = model(imgs, output_hidden_states=segmentation)
            feat = output.last_hidden_state if hasattr(output, 'last_hidden_state') \
                else (output[0] if isinstance(output, tuple) else output)

            if segmentation:
                features = _patch_tokens_to_spatial(feat)
            else:
                if feat.dim() == 3:
                    features = feat[:, 0, :]
                elif feat.dim() == 2:
                    features = feat
                else:
                    features = feat.mean(dim=(2, 3))

            logits = head(features)
            if segmentation and logits.shape[-1] != labels.shape[-1]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            total_loss += criterion(logits, labels).item()
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    metric = compute_metrics(
        torch.cat(all_logits), torch.cat(all_labels),
        task_type='segmentation' if segmentation else 'classification',
        multilabel=multilabel, num_classes=num_classes, ignore_index=ignore_index,
    )
    return metric, total_loss / len(loader)


def _patch_tokens_to_spatial(feat):
    """
    Convert transformer token sequence to spatial feature map [B, D, H, W].

    Handles two cases:
      - Pure patch tokens [B, N, D]: N is a perfect square (e.g. Panopticon x_norm_patchtokens)
      - CLS + patch tokens [B, N+1, D]: skip index 0, then reshape (e.g. OlmoEarth)
    Already-spatial [B, D, H, W] tensors are passed through unchanged.
    """
    if feat.dim() == 4:
        return feat  # already [B, D, H, W]
    if feat.dim() == 3:
        B, N, D = feat.shape
        h = int(N ** 0.5)
        if h * h == N:
            # Pure patch tokens — no CLS to strip
            return feat.permute(0, 2, 1).reshape(B, D, h, h)
        else:
            # CLS token prepended — skip index 0
            n_patches = N - 1
            h = int(n_patches ** 0.5)
            return feat[:, 1:, :].permute(0, 2, 1).reshape(B, D, h, h)
    raise ValueError(f"Cannot convert feat with shape {feat.shape} to spatial")


def train_head(model, head, train_loader, val_loader, criterion, device,
               epochs, lr, weight_decay, train_mode='lp', wandb_log=False,
               segmentation=False, multilabel=False, num_classes=None, ignore_index=-100,
               modality_slice=None, preprocess_fn=None, warmup_epochs=1):
    """
    Train the head (+ optionally backbone layers) on train_loader.

    train_mode:
      'lp'  — freeze backbone, train head only
      'fft' — train backbone + head with full fine-tuning

    NOTE: This function assumes the model accepts [B, C, H, W] tensors directly.
    If using a model with special input requirements (e.g., Panopticon with chn_ids),
    you'll need to wrap the model call or add model-specific preprocessing.
    """

    # Set up training parameters based on mode
    if train_mode == 'lp':
        # Freeze backbone
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze head
        for param in head.parameters():
            param.requires_grad = True
        print(f"Mode=lp: training head only.")
    elif train_mode == 'fft':
        # Full fine-tuning
        for param in model.parameters():
            param.requires_grad = True
        for param in head.parameters():
            param.requires_grad = True
        print(f"Mode=fft: training full backbone + head.")
    else:
        raise ValueError(f"Unknown train_mode: {train_mode}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
                       sum(p.numel() for p in head.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    backbone_lr = lr / 10 if train_mode == 'fft' else lr
    optimizer = torch.optim.AdamW(
        [
            {'params': list(filter(lambda p: p.requires_grad, model.parameters())), 'lr': backbone_lr},
            {'params': list(filter(lambda p: p.requires_grad, head.parameters())),  'lr': lr},
        ],
        weight_decay=weight_decay
    )
    from train_utils import make_scheduler
    scheduler = make_scheduler(optimizer, epochs, warmup_epochs=warmup_epochs)

    best_val_metric = 0.0
    best_state = None
    best_model_state = None
    metric_name = "mIoU" if segmentation else ("mAP" if multilabel else "Acc")

    from tqdm import tqdm

    for epoch in range(epochs):
        model.train() if train_mode == 'fft' else model.eval()
        head.train()

        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [train]', leave=False):
            if preprocess_fn is not None:
                imgs = preprocess_fn(batch).to(device)
            else:
                imgs = batch['image'].to(device)
                if modality_slice is not None:
                    imgs = imgs[:, modality_slice]
            if segmentation:
                labels = batch.get('mask', batch.get('label')).to(device).long()
            else:
                labels = batch['label'].to(device)
                if multilabel:
                    labels = labels.float()

            optimizer.zero_grad()

            # Forward pass
            # output_hidden_states=True for segmentation (need patch tokens);
            # output_hidden_states=False for classification (CLS token directly).
            need_spatial = segmentation
            if train_mode == 'fft':
                output = model(imgs, output_hidden_states=need_spatial)
                if hasattr(output, 'last_hidden_state'):
                    feat = output.last_hidden_state
                else:
                    feat = output[0] if isinstance(output, tuple) else output

                if segmentation:
                    features = _patch_tokens_to_spatial(feat)
                else:
                    if feat.dim() == 3:
                        features = feat[:, 0, :]  # CLS token from [B, N, D]
                    elif feat.dim() == 2:
                        features = feat  # Already pooled [B, D]
                    else:
                        features = feat.mean(dim=(2, 3))
            else:
                with torch.no_grad():
                    output = model(imgs, output_hidden_states=need_spatial)
                    if hasattr(output, 'last_hidden_state'):
                        feat = output.last_hidden_state
                    else:
                        feat = output[0] if isinstance(output, tuple) else output

                    if segmentation:
                        features = _patch_tokens_to_spatial(feat)
                    else:
                        if feat.dim() == 3:
                            features = feat[:, 0, :]
                        elif feat.dim() == 2:
                            features = feat
                        else:
                            features = feat.mean(dim=(2, 3))

            logits = head(features)
            if segmentation and logits.shape[-1] != labels.shape[-1]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(head.parameters()), max_norm=1.0
            )
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation
        val_metric, val_loss = eval_head(
            model, head, val_loader, criterion, device,
            segmentation=segmentation, multilabel=multilabel,
            num_classes=num_classes, ignore_index=ignore_index,
            desc=f'Epoch {epoch+1}/{epochs} [val]',
            modality_slice=modality_slice,
            preprocess_fn=preprocess_fn,
        )

        if val_metric > best_val_metric:
            best_val_metric = val_metric
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
            if train_mode == 'fft':
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}

        print(f'  Epoch {epoch+1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_{metric_name}={val_metric:.2f}%  (best={best_val_metric:.2f}%)')

        if wandb_log:
            wandb.log({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                f'val_{metric_name}': val_metric,
            })

    if best_state is not None:
        head.load_state_dict(best_state)
    if train_mode == 'fft' and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return head, best_val_metric, trainable_params


def main():
    parser = argparse.ArgumentParser(description='Fine-tune RS foundation model on split 1')
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name (e.g., allenai/OlmoEarth-v1-Base)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eurosat', 'benv2', 'benv2full', 'pastis', 'dfc2020'],
                        help='Dataset to train on')
    parser.add_argument('--modality', type=str, required=True,
                        help='Modality to use (e.g., s2, s1, rgb)')
    parser.add_argument('--train_mode', type=str, default='lp', choices=['lp', 'fft'],
                        help='Training mode: lp (linear probe) or fft (full fine-tune)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--results_csv', type=str, default='res/rsfm/rsfm_results.csv')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for benv2full dataset.')
    args = parser.parse_args()

    # Validate modality
    if args.modality not in MODALITY_CONFIGS.get(args.dataset, []):
        parser.error(f"--modality {args.modality!r} is not valid for --dataset {args.dataset}. "
                     f"Valid choices: {MODALITY_CONFIGS[args.dataset]}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {args.dataset}, Modality: {args.modality}")
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    is_olmoearth = 'olmoearth' in args.model.lower()
    is_dino = args.model in DINO_MODELS
    # DINO on GeoBench datasets (benv2, dfc2020, pastis): use loader z-score normalization,
    # consistent with train_sft. DINO on EuroSAT: raw pixels, normalize inside DinoWrapper.
    dino_raw_pixels = is_dino and args.dataset == 'eurosat'

    # Data
    print("\n=== Creating datasets ===")
    train1_loader, val1_loader, test_loader, task_config, modality_bands_dict, preprocess_fn = \
        get_task_config_and_loaders(
            args.dataset, args.modality, args.batch_size, args.num_workers,
            data_root=args.data_root, raw_pixels=is_olmoearth or dino_raw_pixels,
        )

    # Model
    print("\n=== Loading foundation model ===")
    fm, feature_dim = load_foundation_model(args.model, device, modality=args.modality,
                                            dataset=args.dataset, raw_pixels=dino_raw_pixels)

    # Head
    print("\n=== Creating task-specific head ===")
    is_segmentation = (task_config.task_type == 'segmentation')

    if is_segmentation:
        head = create_segmentation_head(feature_dim, task_config.num_classes, device)
    else:
        head = create_classification_head(feature_dim, task_config.num_classes, device)

    # Loss
    ignore_index = getattr(task_config, 'ignore_index', -100)
    if task_config.multilabel:
        criterion = nn.BCEWithLogitsLoss()
        metric_name = "mAP"
    elif is_segmentation:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        metric_name = "mIoU"
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        metric_name = "Acc"
    print(f"Loss: {criterion.__class__.__name__}")

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"rsfm_{args.dataset}_{args.modality}_{args.train_mode}",
        )

    # Training
    print(f"\n=== Training for {args.epochs} epochs ===")
    ignore_index = getattr(task_config, 'ignore_index', -100)
    head, best_val_metric, trainable_params = train_head(
        fm, head, train1_loader, val1_loader, criterion, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        train_mode=args.train_mode, wandb_log=bool(args.wandb_project),
        segmentation=is_segmentation, multilabel=task_config.multilabel,
        num_classes=task_config.num_classes, ignore_index=ignore_index,
        preprocess_fn=preprocess_fn, warmup_epochs=args.warmup_epochs,
    )

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(args.checkpoint_dir,
                                   f'rsfm_{args.dataset}_{args.modality}_{args.train_mode}_{timestamp}.pt')
    torch.save(head.state_dict(), checkpoint_path)
    print(f"\n=== Training complete ===")
    print(f"Head checkpoint: {checkpoint_path}")

    # Test evaluation
    print(f"\n=== Evaluating best checkpoint on test set ===")
    test_metric, _ = eval_head(
        fm, head, test_loader, criterion, device,
        segmentation=is_segmentation, multilabel=task_config.multilabel,
        num_classes=task_config.num_classes, ignore_index=ignore_index,
        desc='Test', preprocess_fn=preprocess_fn,
    )
    print(f"Test {metric_name}: {test_metric:.2f}%")

    if args.wandb_project:
        wandb.log({f'test_{metric_name}': test_metric})

    # CSV logging
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    fieldnames = [
        'model', 'dataset', 'modality', 'train_mode', 'epochs', 'lr', 'weight_decay',
        'batch_size', 'metric_name', 'val_metric', 'test_metric', 'trainable_params', 'saved_checkpoint'
    ]
    file_exists = os.path.isfile(args.results_csv)
    with open(args.results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'model': args.model,
            'dataset': args.dataset,
            'modality': args.modality,
            'train_mode': args.train_mode,
            'epochs': args.epochs,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'batch_size': args.batch_size,
            'metric_name': metric_name,
            'val_metric': f'{best_val_metric:.4f}',
            'test_metric': f'{test_metric:.4f}',
            'trainable_params': trainable_params,
            'saved_checkpoint': checkpoint_path,
        })

    if args.wandb_project:
        wandb.finish()


if __name__ == '__main__':
    main()
