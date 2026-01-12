import argparse
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from einops import rearrange
from transformers import AutoImageProcessor
from torchgeo.datasets import EuroSAT
from tqdm import tqdm
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union
import logging
from evan.layers import LayerScale, LoRALayer, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from functools import partial

logger = logging.getLogger("evan")
ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()

sys.path.append(str(Path(__file__).resolve().parent))

from evan.models.vision_transformer import vit_small, vit_base, vit_large, vit_huge2

class MinMaxNormalize:
    """
    Min-max normalization for EuroSAT bands.
    Normalizes each band to [0, 1] range based on dataset statistics.
    """
    def __init__(self, selected_bands, all_mins, all_maxs):
        """
        Args:
            selected_bands: Tuple of band names (e.g., ('B04', 'B03', 'B02'))
            all_mins: Tensor of min values for all 13 bands
            all_maxs: Tensor of max values for all 13 bands
        """
        # EuroSAT band order
        all_band_names = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                          'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']

        # Get indices of selected bands
        band_indices = [all_band_names.index(band) for band in selected_bands]

        # Select corresponding min/max values
        self.mins = all_mins[band_indices].view(-1, 1, 1)  # Shape: (C, 1, 1)
        self.maxs = all_maxs[band_indices].view(-1, 1, 1)  # Shape: (C, 1, 1)

    def __call__(self, sample):
        """
        Apply min-max normalization to the image.

        Args:
            sample: Dict with 'image' key containing tensor of shape (C, H, W)

        Returns:
            Dict with normalized image
        """
        image = sample['image'].float()

        # Move min/max to same device as image
        mins = self.mins.to(image.device)
        maxs = self.maxs.to(image.device)

        # Min-max normalization: (x - min) / (max - min)
        # Clamp to handle any values outside the expected range
        normalized = (image - mins) / (maxs - mins + 1e-8)
        normalized = torch.clamp(normalized, 0.0, 1.0)

        sample['image'] = normalized
        return sample




class DINOv3Wrapper(nn.Module):
    """
    Wrapper for DINOv3 models using local evan implementation with HuggingFace weights.
    Handles image preprocessing and feature extraction.
    """
    def __init__(self, model_name="facebook/dinov3-vitl16-pretrain-lvd1689m", device="cuda"):
        super().__init__()
        self.model_name = model_name
        self.device = device
        print(f"Loading DINOv3 model: {model_name}...")

        # Map HuggingFace model name to our local architecture
        # Model format: facebook/dinov3-{size}{patch_size}-pretrain-{dataset}
        if 'vits' in model_name:
            model_fn = vit_small
        elif 'vitb' in model_name:
            model_fn = vit_base
        elif 'vitl' in model_name:
            model_fn = vit_large
        elif 'vith' in model_name:
            model_fn = vit_huge2
        elif 'vit7b' in model_name:
            raise NotImplementedError("vit_7b not imported yet")
        else:
            raise ValueError(f"Unknown model size in {model_name}")

        # DINOv3 uses patch size 16 and 4 register tokens
        patch_size = 16
        self.num_register_tokens = 4

        # Instantiate local model
        print(f"Instantiating local DINOv3 model...")
        self.model = model_fn(
            patch_size=patch_size,
            img_size=224,
            n_storage_tokens=self.num_register_tokens,
            layerscale_init=1e-5,  # Enable LayerScale with standard init value
            device=device
        )

        # Load checkpoint from HuggingFace
        print("Loading pretrained weights from HuggingFace...")
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")

        from safetensors.torch import load_file
        hf_checkpoint = load_file(checkpoint_path)

        # Convert HuggingFace keys to DINOv3 keys
        print("Converting HuggingFace checkpoint keys to DINOv3 format...")

        # Count original parameters
        original_params = sum(p.numel() for p in hf_checkpoint.values())
        print(f"  Original checkpoint parameters: {original_params:,}")

        checkpoint = {}
        for key, value in hf_checkpoint.items():
            new_key = key

            # Embeddings mapping
            new_key = new_key.replace('embeddings.cls_token', 'cls_token')
            new_key = new_key.replace('embeddings.mask_token', 'mask_token')
            new_key = new_key.replace('embeddings.register_tokens', 'storage_tokens')
            new_key = new_key.replace('embeddings.patch_embeddings.weight', 'patch_embed.proj.weight')
            new_key = new_key.replace('embeddings.patch_embeddings.bias', 'patch_embed.proj.bias')

            # Fix shape mismatches
            # HF mask_token: [1, 1, D] -> DINOv3: [1, D]
            if 'mask_token' in new_key and value.ndim == 3:
                value = value.squeeze(1)
            # HF cls_token: [1, D] -> DINOv3: [1, 1, D]
            elif 'cls_token' in new_key and value.ndim == 2:
                value = value.unsqueeze(1)

            # Layer mapping: layer.X -> blocks.X
            new_key = new_key.replace('layer.', 'blocks.')

            # Attention mapping: separate q,k,v to combined qkv
            # We'll handle this separately below
            if '.attention.q_proj.' in new_key or '.attention.k_proj.' in new_key or '.attention.v_proj.' in new_key:
                continue  # Skip for now, handle qkv merging below

            # Attention output projection
            new_key = new_key.replace('.attention.o_proj.', '.attn.proj.')

            # MLP mapping
            new_key = new_key.replace('.mlp.up_proj.', '.mlp.fc1.')
            new_key = new_key.replace('.mlp.down_proj.', '.mlp.fc2.')

            # LayerScale mapping
            new_key = new_key.replace('.layer_scale1.lambda1', '.ls1.gamma')
            new_key = new_key.replace('.layer_scale2.lambda1', '.ls2.gamma')

            checkpoint[new_key] = value

        # Handle QKV weight merging (HF has separate q,k,v; DINOv3 has combined qkv)
        # Find all layer indices
        layer_indices = set()
        for key in hf_checkpoint.keys():
            if key.startswith('layer.') and '.attention.q_proj.' in key:
                layer_idx = int(key.split('.')[1])
                layer_indices.add(layer_idx)

        # Track parameters created (not from checkpoint)
        created_params = 0

        for layer_idx in sorted(layer_indices):
            q_key = f'layer.{layer_idx}.attention.q_proj.weight'
            k_key = f'layer.{layer_idx}.attention.k_proj.weight'
            v_key = f'layer.{layer_idx}.attention.v_proj.weight'

            # Merge weights
            q_weight = hf_checkpoint[q_key]
            k_weight = hf_checkpoint[k_key]
            v_weight = hf_checkpoint[v_key]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            checkpoint[f'blocks.{layer_idx}.attn.qkv.weight'] = qkv_weight

            # Merge biases (only q and v have bias, k does not)
            # NOTE: HuggingFace checkpoint doesn't include k_proj.bias (it's implicitly zero)
            # but DINOv3's combined qkv.bias expects [q_bias, k_bias, v_bias] concatenated.
            # We create a zero tensor for k_bias to maintain proper dimensions.
            q_bias_key = f'layer.{layer_idx}.attention.q_proj.bias'
            v_bias_key = f'layer.{layer_idx}.attention.v_proj.bias'

            if q_bias_key in hf_checkpoint and v_bias_key in hf_checkpoint:
                q_bias = hf_checkpoint[q_bias_key]
                v_bias = hf_checkpoint[v_bias_key]
                # Create zero bias for k to match dimensions (not in checkpoint)
                k_bias = torch.zeros_like(q_bias)
                created_params += k_bias.numel()  # Track created parameters
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                checkpoint[f'blocks.{layer_idx}.attn.qkv.bias'] = qkv_bias

        # Count transferred parameters
        transferred_params = sum(p.numel() for p in checkpoint.values())
        checkpoint_params = transferred_params - created_params  # Actual params from checkpoint
        print(f"  Transferred parameters: {transferred_params:,}")
        print(f"    - From checkpoint: {checkpoint_params:,}")
        print(f"    - Created (K bias zeros): {created_params:,}")
        print(f"  Transfer ratio: {100 * checkpoint_params / original_params:.2f}%")

        result = self.model.load_state_dict(checkpoint, strict=False)

        # Filter out expected missing keys (EVAN-specific multi-modality components)
        def is_expected_missing(key):
            # EVAN multi-modality components (new, not in DINO)
            if any(pattern in key for pattern in ['modality_specific_lora_adaptors', 'modality_encoders', 'modality_fusion_lora_adaptors']):
                return True
            return False

        unexpected_missing = [k for k in result.missing_keys if not is_expected_missing(k)]

        if len(unexpected_missing) > 0:
            print(f"  ⚠️  Unexpected missing keys in model: {len(unexpected_missing)}")
            print(f"  Missing keys: {unexpected_missing}")
            missing_params = sum(
                self.model.state_dict()[k].numel()
                for k in unexpected_missing
                if k in self.model.state_dict()
            )
            print(f"  Missing parameters: {missing_params:,}")

        if len(result.unexpected_keys) > 0:
            print(f"  ⚠️  Unexpected keys in checkpoint: {len(result.unexpected_keys)}")
            print(f"  First 10 unexpected keys: {result.unexpected_keys[:10]}")
            unexpected_params = sum(
                checkpoint[k].numel()
                for k in result.unexpected_keys
                if k in checkpoint
            )
            print(f"  Unexpected parameters: {unexpected_params:,}")

        print("\nWeights loaded successfully!")

        # Move to device and set eval mode
        self.model = self.model.to(device)
        self.model.eval()

        # Load processor for image preprocessing
        self.processor = AutoImageProcessor.from_pretrained(model_name, do_rescale=False)

        print(f"Loading complete.")

        # Store model metadata
        self.dim = self.model.embed_dim
        self.image_size = 224
        self.patch_size = patch_size
        self.grid_size = self.image_size // self.patch_size

        print(f"  Hidden size: {self.dim}")
        print(f"  Image size: {self.image_size}")
        print(f"  Patch size: {self.patch_size}")
        print(f"  Grid size: {self.grid_size}x{self.grid_size}")
        print(f"  Register tokens: {self.num_register_tokens}")
        print(f"  Expected patch tokens: {self.grid_size * self.grid_size}")

    def forward(self, images, do_pool=False):
        """
        Forward pass through DINOv3.

        Args:
            images: Tensor of shape (B, C, H, W) in range [0, 1]
            do_pool: If True, return pooled (CLS token) features; if False, return patch features

        Returns:
            If do_pool=True: (B, dim) - pooled features (CLS token)
            If do_pool=False: (B, num_patches, dim) - patch features
        """
        # Process images using HuggingFace processor
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)

        # Forward pass through our local model
        if self.training:
            outputs = self.model.forward_features(pixel_values)
        else:
            with torch.no_grad():
                outputs = self.model.forward_features(pixel_values)

        # outputs is a dict with keys like 'x_norm_clstoken', 'x_norm_patchtokens', etc.
        if do_pool:
            # Use CLS token: (B, dim)
            features = outputs['x_norm_clstoken']
        else:
            # Use patch features: (B, num_patches, dim)
            features = outputs['x_norm_patchtokens']

        return features


class EVAN(nn.Module):
    "The EVer-Adapting Network (EVAN) is a framework designed to handle unseen modalities at test time."
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        tz_fusion_time: int = 3,
        tz_lora_rank: int=32,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size
        self.device = device
        self.tz_fusion_time = tz_fusion_time
        self.tz_lora_rank = tz_lora_rank

        # Multi-modality support: Initialize patch embedders dict with 'rgb' for DINO compatibility
        self.patch_embedders = nn.ModuleDict()
        rgb_embedder = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )
        self.patch_embedders['rgb'] = rgb_embedder
        # Note: DINO weight loading compatibility handled by key remapping in load_pretrained_dino()


        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        # Initialize modality-specific LoRA adaptors for first tz_fusion_time blocks
        self.modality_specific_lora_adaptors = nn.ModuleDict()
        self.modality_specific_lora_adaptors['rgb'] = nn.ModuleList([
            LoRALayer(embed_dim, rank=self.tz_lora_rank, device=device)
            for _ in range(tz_fusion_time)
        ])

        # Initialize modality encodings (per-token embeddings added after modality-specific processing)
        self.modality_encoders = nn.ParameterDict()
        self.modality_encoders['rgb'] = nn.Parameter(
            torch.zeros(1, 1, embed_dim, device=device)
        )

        # Initialize fusion LoRA adaptors for remaining blocks (after tz_fusion_time)
        self.modality_fusion_lora_adaptors = nn.ModuleDict()
        num_fusion_blocks = depth - tz_fusion_time
        self.modality_fusion_lora_adaptors['rgb'] = nn.ModuleList([
            LoRALayer(embed_dim, rank=self.tz_lora_rank, device=device)
            for _ in range(num_fusion_blocks)
        ])

    def load_pretrained_dino(self, model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m"):
        """
        Load pretrained DINO weights from HuggingFace into EVAN.

        Args:
            model_name: HuggingFace model name (default: facebook/dinov3-vitl16-pretrain-lvd1689m)
        """
        print(f"\n=== Loading pretrained DINO weights ===")
        print(f"Model: {model_name}")
        print("Loading pretrained weights from HuggingFace...")

        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        checkpoint_path = hf_hub_download(repo_id=model_name, filename="model.safetensors")
        hf_checkpoint = load_file(checkpoint_path)

        # Convert HuggingFace keys to DINOv3 keys
        print("Converting HuggingFace checkpoint keys to EVAN format...")

        # Count original parameters
        original_params = sum(p.numel() for p in hf_checkpoint.values())
        print(f"  Original checkpoint parameters: {original_params:,}")

        checkpoint = {}
        for key, value in hf_checkpoint.items():
            new_key = key

            # Embeddings mapping
            new_key = new_key.replace('embeddings.cls_token', 'cls_token')
            new_key = new_key.replace('embeddings.mask_token', 'mask_token')
            new_key = new_key.replace('embeddings.register_tokens', 'storage_tokens')
            new_key = new_key.replace('embeddings.patch_embeddings.weight', 'patch_embed.proj.weight')
            new_key = new_key.replace('embeddings.patch_embeddings.bias', 'patch_embed.proj.bias')

            # Fix shape mismatches
            # HF mask_token: [1, 1, D] -> DINOv3: [1, D]
            if 'mask_token' in new_key and value.ndim == 3:
                value = value.squeeze(1)
            # HF cls_token: [1, D] -> DINOv3: [1, 1, D]
            elif 'cls_token' in new_key and value.ndim == 2:
                value = value.unsqueeze(1)

            # Layer mapping: layer.X -> blocks.X
            new_key = new_key.replace('layer.', 'blocks.')

            # Attention mapping: separate q,k,v to combined qkv
            # We'll handle this separately below
            if '.attention.q_proj.' in new_key or '.attention.k_proj.' in new_key or '.attention.v_proj.' in new_key:
                continue  # Skip for now, handle qkv merging below

            # Attention output projection
            new_key = new_key.replace('.attention.o_proj.', '.attn.proj.')

            # MLP mapping
            new_key = new_key.replace('.mlp.up_proj.', '.mlp.fc1.')
            new_key = new_key.replace('.mlp.down_proj.', '.mlp.fc2.')

            # LayerScale mapping
            new_key = new_key.replace('.layer_scale1.lambda1', '.ls1.gamma')
            new_key = new_key.replace('.layer_scale2.lambda1', '.ls2.gamma')

            checkpoint[new_key] = value

        # Handle QKV weight merging (HF has separate q,k,v; DINOv3 has combined qkv)
        # Find all layer indices
        layer_indices = set()
        for key in hf_checkpoint.keys():
            if key.startswith('layer.') and '.attention.q_proj.' in key:
                layer_idx = int(key.split('.')[1])
                layer_indices.add(layer_idx)

        # Track parameters created (not from checkpoint)
        created_params = 0

        for layer_idx in sorted(layer_indices):
            q_key = f'layer.{layer_idx}.attention.q_proj.weight'
            k_key = f'layer.{layer_idx}.attention.k_proj.weight'
            v_key = f'layer.{layer_idx}.attention.v_proj.weight'

            # Merge weights
            q_weight = hf_checkpoint[q_key]
            k_weight = hf_checkpoint[k_key]
            v_weight = hf_checkpoint[v_key]
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            checkpoint[f'blocks.{layer_idx}.attn.qkv.weight'] = qkv_weight

            # Merge biases (only q and v have bias, k does not)
            # NOTE: HuggingFace checkpoint doesn't include k_proj.bias (it's implicitly zero)
            # but DINOv3's combined qkv.bias expects [q_bias, k_bias, v_bias] concatenated.
            # We create a zero tensor for k_bias to maintain proper dimensions.
            q_bias_key = f'layer.{layer_idx}.attention.q_proj.bias'
            v_bias_key = f'layer.{layer_idx}.attention.v_proj.bias'

            if q_bias_key in hf_checkpoint and v_bias_key in hf_checkpoint:
                q_bias = hf_checkpoint[q_bias_key]
                v_bias = hf_checkpoint[v_bias_key]
                # Create zero bias for k to match dimensions (not in checkpoint)
                k_bias = torch.zeros_like(q_bias)
                created_params += k_bias.numel()  # Track created parameters
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                checkpoint[f'blocks.{layer_idx}.attn.qkv.bias'] = qkv_bias

        # Count transferred parameters
        transferred_params = sum(p.numel() for p in checkpoint.values())
        checkpoint_params = transferred_params - created_params  # Actual params from checkpoint
        print(f"  Transferred parameters: {transferred_params:,}")
        print(f"    - From checkpoint: {checkpoint_params:,}")
        print(f"    - Created (K bias zeros): {created_params:,}")
        print(f"  Transfer ratio: {100 * checkpoint_params / original_params:.2f}%")

        # EVAN multi-modality compatibility: Remap patch_embed.* to patch_embedders.rgb.*
        # This allows EVAN's multi-modality architecture to load DINO weights correctly
        checkpoint_remapped = {}
        for key, value in checkpoint.items():
            if key.startswith('patch_embed.'):
                # Remap: patch_embed.* -> patch_embedders.rgb.*
                new_key = key.replace('patch_embed.', 'patch_embedders.rgb.')
                checkpoint_remapped[new_key] = value
            else:
                checkpoint_remapped[key] = value
        checkpoint = checkpoint_remapped

        result = self.load_state_dict(checkpoint, strict=False)

        # Filter out expected missing keys (EVAN-specific multi-modality components)
        def is_expected_missing(key):
            # RoPE periods buffer: computed deterministically from hyperparameters, not in HF checkpoint
            if key == 'rope_embed.periods':
                return True
            # EVAN multi-modality components (new, not in DINO)
            if any(pattern in key for pattern in ['modality_specific_lora_adaptors', 'modality_encoders', 'modality_fusion_lora_adaptors']):
                return True
            return False

        # Report missing keys (in EVAN but not in DINO checkpoint)
        expected_missing = [k for k in result.missing_keys if is_expected_missing(k)]
        unexpected_missing = [k for k in result.missing_keys if not is_expected_missing(k)]

        if len(expected_missing) > 0:
            expected_missing_params = sum(
                self.state_dict()[k].numel()
                for k in expected_missing
                if k in self.state_dict()
            )
            print(f"  Expected missing keys (in EVAN but not in DINO): {len(expected_missing)}")
            print(f"    Keys: {expected_missing}")
            print(f"    Parameters: {expected_missing_params:,}")

        if len(unexpected_missing) > 0:
            unexpected_missing_params = sum(
                self.state_dict()[k].numel()
                for k in unexpected_missing
                if k in self.state_dict()
            )
            print(f"  ⚠️  Unexpected missing keys (in EVAN but not in DINO): {len(unexpected_missing)}")
            print(f"    Keys: {unexpected_missing}")
            print(f"    Parameters: {unexpected_missing_params:,}")

        # Report unexpected keys (in DINO checkpoint but not in EVAN)
        if len(result.unexpected_keys) > 0:
            untransferred_params = sum(
                checkpoint[k].numel()
                for k in result.unexpected_keys
                if k in checkpoint
            )
            print(f"  ⚠️  Unexpected keys (in DINO but not loaded into EVAN): {len(result.unexpected_keys)}")
            print(f"    First 10 keys: {result.unexpected_keys[:10]}")
            print(f"    Untransferred parameters: {untransferred_params:,}")

        print("\nWeights loaded successfully!")
        print("=== DINO weight loading complete ===\n")

    def _create_modality_components(self, modality_key: str, in_chans: int):
        """
        Create embedder, LoRAs, and encoding for a new modality.

        Args:
            modality_key: Name/identifier for the new modality
            in_chans: Number of input channels for this modality
        """
        # Track parameter count
        params_before = sum(p.numel() for p in self.parameters())

        # 1. Create patch embedder
        embedder = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            flatten_embedding=False,
        )
        if self.device is not None:
            embedder = embedder.to(self.device)
        self.patch_embedders[modality_key] = embedder

        # 2. Create modality-specific LoRAs (for first tz_fusion_time blocks)
        lora_list = nn.ModuleList([
            LoRALayer(self.embed_dim, rank=self.tz_lora_rank, device=self.device)
            for _ in range(self.tz_fusion_time)
        ])
        self.modality_specific_lora_adaptors[modality_key] = lora_list

        # 3. Create modality encoding (per-token embedding)
        modality_encoding = nn.Parameter(
            torch.zeros(1, 1, self.embed_dim, device=self.device)
        )
        self.modality_encoders[modality_key] = modality_encoding

        # 4. Create fusion LoRAs (for remaining blocks after tz_fusion_time)
        num_fusion_blocks = len(self.blocks) - self.tz_fusion_time
        fusion_lora_list = nn.ModuleList([
            LoRALayer(self.embed_dim, rank=self.tz_lora_rank, device=self.device)
            for _ in range(num_fusion_blocks)
        ])
        self.modality_fusion_lora_adaptors[modality_key] = fusion_lora_list

        # 5. Freeze shared components, allow new components to train
        for param in self.blocks.parameters():
            param.requires_grad = False

        # Count new parameters
        params_after = sum(p.numel() for p in self.parameters())
        new_params = params_after - params_before

        # Verbose logging
        embedder_params = sum(p.numel() for p in embedder.parameters())
        lora_params = sum(p.numel() for p in lora_list.parameters())
        encoding_params = modality_encoding.numel()
        fusion_lora_params = sum(p.numel() for p in fusion_lora_list.parameters())

        logger.info(f"✨ Initialized new modality: '{modality_key}'")
        logger.info(f"   - Input channels: {in_chans}")
        logger.info(f"   - Components created:")
        logger.info(f"     • Patch embedder: {embedder_params:,} params")
        logger.info(f"     • Modality-specific LoRAs ({self.tz_fusion_time} blocks): {lora_params:,} params")
        logger.info(f"     • Modality encoding: {encoding_params:,} params")
        logger.info(f"     • Fusion LoRAs ({num_fusion_blocks} blocks): {fusion_lora_params:,} params")
        logger.info(f"   - Total new parameters: {new_params:,}")
        logger.info(f"   - Shared blocks frozen: ✓")

    def prepare_tokens_with_masks(self, x: Tensor, embedder: nn.Module = None, masks=None) -> Tuple[Tensor, Tuple[int]]:
        """
        Prepare tokens with optional masks and modality-specific embedder.

        Args:
            x: Input tensor
            embedder: Optional patch embedder (defaults to self.patch_embed for backward compatibility)
            masks: Optional mask tensor

        Returns:
            Tuple of (tokens with CLS and storage prepended, (H, W) spatial dimensions)
        """
        if embedder is None:
            embedder = self.patch_embed

        x = embedder(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        x = []
        rope = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )
        return output

    def forward_features(self, x: Tensor | Dict[str, Tensor] | List[Tensor], masks: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Forward features with multi-modality support.

        Args:
            x: Input tensor, dict of tensors (modality: tensor), or list of tensors
            masks: Optional mask tensor

        Returns:
            Dictionary with normalized features
        """
        # Handle backward compatibility: single tensor input
        if isinstance(x, torch.Tensor):
            x = {'rgb': x}

        # Handle list input (legacy)
        if isinstance(x, list):
            return self.forward_features_list(x, masks)[0]

        # Now x is a dict of modalities
        if not isinstance(x, dict) or len(x) == 0:
            raise ValueError("Input must be a non-empty dict of modalities or a Tensor")

        # Validate batch sizes match across modalities
        batch_sizes = [v.shape[0] for v in x.values()]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"Batch sizes must match across modalities, got {batch_sizes}")

        # Step 1: Embed each modality and prepare tokens
        embedded_modalities = {}
        hw_tuples = {}

        for modality_key, modality_tensor in x.items():
            # Check if embedder exists, if not, create it
            if modality_key not in self.patch_embedders:
                in_chans = modality_tensor.shape[1]
                self._create_modality_components(modality_key, in_chans)

            embedder = self.patch_embedders[modality_key]
            x_mod, (H, W) = self.prepare_tokens_with_masks(modality_tensor, embedder=embedder, masks=masks)
            embedded_modalities[modality_key] = x_mod
            hw_tuples[modality_key] = (H, W)

        # Step 2: Process each modality through first tz_fusion_time blocks with modality-specific LoRA
        for modality_key, x_mod in embedded_modalities.items():
            H, W = hw_tuples[modality_key]

            # Apply first tz_fusion_time blocks with modality-specific LoRA
            for i in range(self.tz_fusion_time):
                if self.rope_embed is not None:
                    rope_sincos = self.rope_embed(H=H, W=W)
                else:
                    rope_sincos = None

                # Shared block forward
                x_mod = self.blocks[i](x_mod, rope_sincos)

                # Add modality-specific LoRA adaptation
                lora = self.modality_specific_lora_adaptors[modality_key][i]
                x_mod = x_mod + lora(x_mod)

            # Add modality encoding (per-token embedding)
            modality_encoding = self.modality_encoders[modality_key]
            x_mod = x_mod + modality_encoding

            embedded_modalities[modality_key] = x_mod

        # Step 3: Concatenate patches for fusion (allow cross-modal attention)
        # Store metadata to split back later
        modality_info = {}
        all_cls_storage = {}
        all_patches = []
        current_idx = 0

        for modality_key in sorted(embedded_modalities.keys()):
            x_mod = embedded_modalities[modality_key]

            # Extract CLS and storage tokens (keep separate per modality)
            all_cls_storage[modality_key] = x_mod[:, :self.n_storage_tokens + 1, :]

            # Extract patch tokens
            patches = x_mod[:, self.n_storage_tokens + 1:, :]
            num_patches = patches.shape[1]

            # Store metadata for later splitting
            modality_info[modality_key] = {
                'num_patches': num_patches,
                'start_idx': current_idx,
                'end_idx': current_idx + num_patches
            }

            all_patches.append(patches)
            current_idx += num_patches

        # Concatenate all patches along sequence dimension
        # This allows cross-modal attention during fusion!
        x_patches_concat = torch.cat(all_patches, dim=1)  # [B, sum(num_patches), embed_dim]

        # Use CLS/storage from first modality for the concatenated sequence
        first_modality = sorted(embedded_modalities.keys())[0]
        x_cls_storage = all_cls_storage[first_modality]
        H, W = hw_tuples[first_modality]

        # Combine for fusion processing
        x_fused = torch.cat([x_cls_storage, x_patches_concat], dim=1)

        # Step 4: Process through fusion blocks with cross-modal attention
        # All patches from all modalities attend to each other here!
        for i in range(self.tz_fusion_time, len(self.blocks)):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None

            # Shared block forward on concatenated representation
            x_fused = self.blocks[i](x_fused, rope_sincos)

            # Add fusion LoRA: sum LoRAs from all present modalities
            lora_idx = i - self.tz_fusion_time
            lora_output = 0
            for modality_key in embedded_modalities.keys():
                lora = self.modality_fusion_lora_adaptors[modality_key][lora_idx]
                lora_output = lora_output + lora(x_fused)

            x_fused = x_fused + lora_output

        # Step 5: Split fused patches back into modality-specific outputs
        # Extract fused CLS/storage tokens (these went through fusion blocks)
        x_cls_storage_fused = x_fused[:, :self.n_storage_tokens + 1, :]

        # Extract fused patches (skip CLS/storage tokens)
        x_patches_fused = x_fused[:, self.n_storage_tokens + 1:, :]

        # Split back into modality-specific patches using stored metadata
        output_dict = {}
        for modality_key in sorted(embedded_modalities.keys()):
            info = modality_info[modality_key]
            start, end = info['start_idx'], info['end_idx']

            # Extract this modality's patches from the fused representation
            patches = x_patches_fused[:, start:end, :]

            # Recombine with fused CLS/storage tokens (use the same CLS/storage for all modalities)
            # Note: In single-modality case, this is just the RGB CLS/storage after all blocks
            # In multi-modality case, all modalities share the same fused CLS/storage
            x_mod = torch.cat([x_cls_storage_fused, patches], dim=1)

            # Apply normalization
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x_mod[:, :self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x_mod[:, :self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x_mod[:, self.n_storage_tokens + 1:])
            else:
                x_norm = self.norm(x_mod)
                x_norm_cls_reg = x_norm[:, :self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1:]

            output_dict[modality_key] = {
                "x_norm_clstoken": x_norm_cls_reg[:, 0],
                "x_storage_tokens": x_norm_cls_reg[:, 1:],
                "x_norm_patchtokens": x_norm_patch,
                "x_prenorm": x_mod,
                "masks": masks,
            }

        return output_dict

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> Dict[str, Dict[str, Tensor]] | Dict[str, Tensor] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            # For inference, return pooled CLS tokens
            # If input was a single tensor (converted to {'rgb': tensor}), return just the rgb output
            if isinstance(ret, dict) and len(ret) == 1 and 'rgb' in ret:
                # Backward compatibility: single RGB input
                return self.head(ret['rgb']["x_norm_clstoken"])
            elif isinstance(ret, dict):
                # Multi-modality: return dict of pooled features
                return {modality: self.head(output["x_norm_clstoken"])
                        for modality, output in ret.items()}
            else:
                # Legacy path
                return self.head(ret["x_norm_clstoken"])

# EVAN preset functions (similar to DINOv3)

def evan_small(pretrained: str = "facebook/dinov3-vits16-pretrain-lvd1689m", **kwargs):
    """
    Create EVAN-Small model (384 dim, 12 blocks, 6 heads) with pretrained DINO weights.

    Args:
        pretrained: HuggingFace model name for pretrained weights (default: facebook/dinov3-vits16-pretrain-lvd1689m)
        **kwargs: Additional arguments passed to EVAN (e.g., device, tz_fusion_time, n_storage_tokens)

    Returns:
        EVAN model with small architecture and loaded DINO weights
    """
    model = EVAN(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        layerscale_init=1e-5,  # DINOv3 uses LayerScale
        **kwargs,
    )
    model.load_pretrained_dino(model_name=pretrained)
    return model


def evan_base(pretrained: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", **kwargs):
    """
    Create EVAN-Base model (768 dim, 12 blocks, 12 heads) with pretrained DINO weights.

    Args:
        pretrained: HuggingFace model name for pretrained weights (default: facebook/dinov3-vitb16-pretrain-lvd1689m)
        **kwargs: Additional arguments passed to EVAN (e.g., device, tz_fusion_time, n_storage_tokens)

    Returns:
        EVAN model with base architecture and loaded DINO weights
    """
    model = EVAN(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        layerscale_init=1e-5,  # DINOv3 uses LayerScale
        **kwargs,
    )
    model.load_pretrained_dino(model_name=pretrained)
    return model


def evan_large(pretrained: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", **kwargs):
    """
    Create EVAN-Large model (1024 dim, 24 blocks, 16 heads) with pretrained DINO weights.

    Args:
        pretrained: HuggingFace model name for pretrained weights (default: facebook/dinov3-vitl16-pretrain-lvd1689m)
        **kwargs: Additional arguments passed to EVAN (e.g., device, tz_fusion_time, n_storage_tokens)

    Returns:
        EVAN model with large architecture and loaded DINO weights
    """
    model = EVAN(
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        layerscale_init=1e-5,  # DINOv3 uses LayerScale
        **kwargs,
    )
    model.load_pretrained_dino(model_name=pretrained)
    return model


class EuroSATClassifier(nn.Module):
    """Simple classifier head on top of DINOv3 for EuroSAT."""
    def __init__(self, dinov3_wrapper, num_classes=10):
        super().__init__()
        self.dinov3 = dinov3_wrapper
        hidden_dim = self.dinov3.dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(self.dinov3.dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, images):
        # Get pooled features (CLS token)
        features = self.dinov3(images, do_pool=True)
        logits = self.classifier(features)
        return logits


if __name__ == '__main__':
    # Define min/max values for all 13 EuroSAT bands (in order: B01-B12, B8A)
    mins = torch.tensor(
        [
            1013.0,   # B01
            676.0,    # B02
            448.0,    # B03
            247.0,    # B04
            269.0,    # B05
            253.0,    # B06
            243.0,    # B07
            189.0,    # B08
            61.0,     # B8A
            4.0,      # B09
            33.0,     # B10
            11.0,     # B11
            186.0,    # B12
        ]
    )
    maxs = torch.tensor(
        [
            2309.0,    # B01
            4543.05,   # B02
            4720.2,    # B03
            5293.05,   # B04
            3902.05,   # B05
            4473.0,    # B06
            5447.0,    # B07
            5948.05,   # B08
            1829.0,    # B8A
            23.0,      # B09
            4894.05,   # B10
            4076.05,   # B11
            5846.0,    # B12
        ]
    )

    # Select bands to use
    selected_bands = ('B04', 'B03', 'B02')  # RGB
    # Alternative: all bands
    # selected_bands = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B10', 'B11', 'B12')

    # Create normalization transform for selected bands
    normalize_transform = MinMaxNormalize(selected_bands, mins, maxs)

    # Create EuroSAT datasets with normalization transform - use official splits
    train_dataset = EuroSAT(root='datasets',
                            split='train',
                            bands=selected_bands,
                            transforms=normalize_transform,
                            download=True,
                            checksum=False)

    test_dataset = EuroSAT(root='datasets',
                           split='test',
                           bands=selected_bands,
                           transforms=normalize_transform,
                           download=True,
                           checksum=False)

    # Band descriptions (for reference)
    band_descriptions = {
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

    # Print selected bands info
    print(f"Selected bands: {selected_bands}")
    print(f"Number of bands: {len(selected_bands)}")
    print("Band descriptions:")
    for band in selected_bands:
        print(f"  {band}: {band_descriptions[band]}")

    print(f"\nDataset sizes:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    
    model_name = "facebook/dinov3-vitl16-pretrain-lvd1689m" # sat493m or lvd1689m
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create model
    print("\n=== Creating model ===")
    dinov3 = DINOv3Wrapper(model_name, device=device)

    # Freeze DINOv3 backbone - only train the classifier head
    for param in dinov3.parameters():
        param.requires_grad = False

    model = EuroSATClassifier(dinov3, num_classes=10)
    model = model.to(device)

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    # Only optimize parameters that require gradients (classifier head)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    num_epochs = 10

    print(f"\n=== Training for {num_epochs} epochs ===")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images = batch['image'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%'
            })

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Evaluation
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total

        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

    print("\n=== Training complete ===")


# python -u evan_main.py