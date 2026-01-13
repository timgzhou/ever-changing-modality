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
        tz_modality_specific_layer_augmenter: Literal["lora", "fft"] = "lora",
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
        self.tz_modality_specific_layer_augmenter = tz_modality_specific_layer_augmenter

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
        self.modality_specific_layer_adaptors = nn.ModuleDict()
        if tz_modality_specific_layer_augmenter=="lora":
            self.modality_specific_layer_adaptors['rgb'] = nn.ModuleList([
                LoRALayer(embed_dim, rank=self.tz_lora_rank, device=device)
                for _ in range(tz_fusion_time)
            ])
        elif tz_modality_specific_layer_augmenter=="fft":    
            self.modality_specific_layer_adaptors['rgb'] = nn.ModuleList([
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
                for i in range(tz_fusion_time)
            ])
        else: raise RuntimeError(f"unrecognized {tz_modality_specific_layer_augmenter=}")

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

        # For FFT mode: Copy first tz_fusion_time blocks to RGB modality-specific layers
        if self.tz_modality_specific_layer_augmenter == "fft":
            print(f"\n  FFT mode: Copying first {self.tz_fusion_time} DINO blocks to RGB modality-specific layers...")
            fft_params_copied = 0
            for i in range(self.tz_fusion_time):
                # Copy all weights from blocks[i] to modality_specific_layer_adaptors.rgb[i]
                for key, value in list(checkpoint.items()):
                    if key.startswith(f'blocks.{i}.'):
                        # Create corresponding key for RGB modality-specific layer
                        rgb_key = key.replace(f'blocks.{i}.', f'modality_specific_layer_adaptors.rgb.{i}.')
                        checkpoint[rgb_key] = value.clone()
                        fft_params_copied += value.numel()
            print(f"    Copied {fft_params_copied:,} parameters for RGB FFT blocks")

        result = self.load_state_dict(checkpoint, strict=False)

        # Filter out expected missing keys (EVAN-specific multi-modality components)
        def is_expected_missing(key):
            # RoPE periods buffer: computed deterministically from hyperparameters, not in HF checkpoint
            if key == 'rope_embed.periods':
                return True
            # EVAN multi-modality components (new, not in DINO)
            # Note: In FFT mode, modality_specific_layer_adaptors should be loaded (not missing)
            if self.tz_modality_specific_layer_augmenter == "lora":
                if 'modality_specific_layer_adaptors' in key:
                    return True
            if any(pattern in key for pattern in ['modality_encoders', 'modality_fusion_lora_adaptors']):
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

        # 2. Create modality-specific adaptors (for first tz_fusion_time blocks)
        if self.tz_modality_specific_layer_augmenter == "lora":
            # LoRA mode: create lightweight LoRA adaptors
            adaptor_list = nn.ModuleList([
                LoRALayer(self.embed_dim, rank=self.tz_lora_rank, device=self.device)
                for _ in range(self.tz_fusion_time)
            ])
        elif self.tz_modality_specific_layer_augmenter == "fft":
            # FFT mode: copy first tz_fusion_time transformer blocks from DINO
            adaptor_list = nn.ModuleList()
            for i in range(self.tz_fusion_time):
                # Deep copy the block to create an independent copy
                import copy
                block_copy = copy.deepcopy(self.blocks[i])
                if self.device is not None:
                    block_copy = block_copy.to(self.device)
                adaptor_list.append(block_copy)
        else:
            raise ValueError(f"Unknown augmenter mode: {self.tz_modality_specific_layer_augmenter}")

        self.modality_specific_layer_adaptors[modality_key] = adaptor_list

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
        adaptor_params = sum(p.numel() for p in adaptor_list.parameters())
        encoding_params = modality_encoding.numel()
        fusion_lora_params = sum(p.numel() for p in fusion_lora_list.parameters())

        adaptor_type = "LoRAs" if self.tz_modality_specific_layer_augmenter == "lora" else "FFT blocks"
        logger.info(f"✨ Initialized new modality: '{modality_key}'")
        logger.info(f"   - Input channels: {in_chans}")
        logger.info(f"   - Augmenter mode: {self.tz_modality_specific_layer_augmenter}")
        logger.info(f"   - Components created:")
        logger.info(f"     • Patch embedder: {embedder_params:,} params")
        logger.info(f"     • Modality-specific {adaptor_type} ({self.tz_fusion_time} blocks): {adaptor_params:,} params")
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

        # Step 1 & 2: Process through modality-specific layers
        embedded_modalities = self.forward_modality_specific_features(x, masks)

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
        # Compute H, W from image size and patch size
        H = W = self.img_size // self.patch_size

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

    def forward_modality_specific_features(self, x: Dict[str, Tensor], masks: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Extract features after modality-specific layers (first tz_fusion_time blocks).
        This is useful for MAE training where you want features before fusion.

        Args:
            x: Dictionary of modality tensors
            masks: Optional mask tensor

        Returns:
            Dictionary mapping modality_key -> features after modality-specific processing
        """
        if not isinstance(x, dict) or len(x) == 0:
            raise ValueError("Input must be a non-empty dict of modalities")

        # Step 1: Embed each modality and prepare tokens
        embedded_modalities = {}
        hw_tuples = {}

        for modality_key, modality_tensor in x.items():
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

            # Apply first tz_fusion_time blocks with modality-specific adaptations
            for i in range(self.tz_fusion_time):
                if self.rope_embed is not None:
                    rope_sincos = self.rope_embed(H=H, W=W)
                else:
                    rope_sincos = None

                # Apply modality-specific adaptation based on mode
                if self.tz_modality_specific_layer_augmenter == "lora":
                    # LoRA mode: shared block + additive LoRA adaptation
                    x_mod = self.blocks[i](x_mod, rope_sincos)
                    lora = self.modality_specific_layer_adaptors[modality_key][i]
                    x_mod = x_mod + lora(x_mod)
                elif self.tz_modality_specific_layer_augmenter == "fft":
                    # FFT mode: replace with modality-specific full transformer block
                    adaptor = self.modality_specific_layer_adaptors[modality_key][i]
                    x_mod = adaptor(x_mod, rope_sincos)
                else:
                    raise ValueError(f"Unknown augmenter mode: {self.tz_modality_specific_layer_augmenter}")

            # Add modality encoding (per-token embedding)
            modality_encoding = self.modality_encoders[modality_key]
            x_mod = x_mod + modality_encoding

            embedded_modalities[modality_key] = x_mod

        return embedded_modalities

    def forward(self, *args, is_training: bool = False, **kwargs) -> Dict[str, Dict[str, Tensor]] | Dict[str, Tensor] | Tensor:
        raise NotImplementedError("Why are you calling me?")

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


class EVANClassifier(nn.Module):
    """Classifier head on top of EVAN for EuroSAT."""

    def __init__(self, evan_model, num_classes=10, fusion_strategy='mean', factor=4, device = "cuda"):
        super().__init__()
        self.evan = evan_model
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        self.factor = factor
        embed_dim = self.evan.embed_dim
        hidden_dim = embed_dim * factor
        self.hidden_dim = hidden_dim
        self.device = device

        if fusion_strategy == 'mean':
            # Average CLS tokens from all modalities, then classify
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            self.modality_classifiers = None
        elif fusion_strategy == 'ensemble':
            # Per-modality classifiers that get ensembled
            self.classifier = None
            self.modality_classifiers = nn.ModuleDict()
        else:
            raise ValueError(f"Unknown fusion strategy: {fusion_strategy}")

    def _instantiate_modality_classifier(self, modality_key: str):
        """
        Create a new classifier for a specific modality.

        Args:
            modality_key: Name of the modality (e.g., 'rgb', 'vre', 'nir', 'swir')
        """
        embed_dim = self.evan.embed_dim

        classifier = nn.Sequential(
            nn.Linear(embed_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_classes)
        )

        classifier = classifier.to(self.device)
        self.modality_classifiers[modality_key] = classifier
        print(f"  Created new classifier for modality: {modality_key}")

    def forward(self, x):
        """
        Forward pass supporting both single tensor and dict inputs.

        Args:
            x: Either a tensor [B, C, H, W] or dict {modality: tensor}
            train_modality: Optional str specifying which modality to use for loss during training
                           (only applies in ensemble mode). If None, uses all modalities.
                           During eval, always uses all modalities for ensemble.

        Returns:
            logits: [B, num_classes]
        """
        # Get features from EVAN
        features_dict = self.evan.forward_features(x)

        if self.fusion_strategy == 'mean':
            # Extract CLS tokens from each modality
            cls_tokens = []
            for modality in sorted(features_dict.keys()):
                if modality=="rgb":
                    cls_tokens.append(features_dict[modality]['x_norm_clstoken'])
                else:
                    cls_tokens.append(features_dict[modality]['x_norm_patchtokens'].mean(1))

            # Average CLS tokens
            fused = torch.stack(cls_tokens).mean(dim=0)

            # Classify
            logits = self.classifier(fused)

        elif self.fusion_strategy == 'ensemble':
            all_logits = []
            for modality in sorted(features_dict.keys()):
                # Create classifier for this modality if it doesn't exist
                if modality not in self.modality_classifiers:
                    self._instantiate_modality_classifier(modality)

                # Get CLS token for this modality
                cls_token = features_dict[modality]['x_norm_clstoken']

                # Get logits from modality-specific classifier
                modality_logits = self.modality_classifiers[modality](cls_token)
                all_logits.append(modality_logits)

            # Ensemble by averaging logits
            logits = torch.stack(all_logits).mean(dim=0)

        return logits


if __name__ == '__main__':
    print("why are you calling me?")

# python -u evan_main.py