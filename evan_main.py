import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging
from evan.layers import LoRALayer, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from functools import partial

logger = logging.getLogger("evan")

ffn_layer_dict = {"mlp": Mlp}
norm_layer_dict = {"layernorm": partial(nn.LayerNorm, eps=1e-6),}
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
        n_storage_tokens: int = 4,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        tz_modality_specific_layer_augmenter: Literal["lora", "fft"] = "lora",
        tz_modality_fusion_layer_augmenter: Literal["lora", "fft"] = "lora",
        tz_fusion_time: int = 3,
        tz_lora_rank: int=32,
        starting_modality: str='rgb',
        starting_n_chans: int = 3,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs
        # Save args to Evan
        norm_layer_cls = norm_layer_dict[norm_layer]
        self.norm_layer=norm_layer
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.img_size = img_size
        self.device = device
        self.tz_fusion_time = tz_fusion_time
        self.tz_lora_rank = tz_lora_rank
        self.tz_modality_specific_layer_augmenter = tz_modality_specific_layer_augmenter
        self.tz_modality_fusion_layer_augmenter = tz_modality_fusion_layer_augmenter
        self.starting_modality=starting_modality
        self.n_storage_tokens = n_storage_tokens
        self.pos_embed_rope_base=pos_embed_rope_base
        self.pos_embed_rope_min_period=pos_embed_rope_min_period
        self.pos_embed_rope_max_period=pos_embed_rope_max_period
        self.pos_embed_rope_normalize_coords=pos_embed_rope_normalize_coords
        self.pos_embed_rope_shift_coords=pos_embed_rope_shift_coords
        self.pos_embed_rope_jitter_coords=pos_embed_rope_jitter_coords
        self.pos_embed_rope_rescale_coords=pos_embed_rope_rescale_coords
        self.pos_embed_rope_dtype=pos_embed_rope_dtype
        self.ffn_ratio=ffn_ratio
        self.ffn_layer=ffn_layer
        self.qkv_bias=qkv_bias
        self.proj_bias=proj_bias
        self.ffn_bias=ffn_bias
        self.drop_path_rate=drop_path_rate
        self.layerscale_init=layerscale_init
        self.mask_k_bias=mask_k_bias
        # Usual DINO goodies
        self.initialize_rope_embed()
        self.initialize_blocks()
        self.norm = norm_layer_cls(embed_dim) # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms: # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        # ============ EVAN-Specific Components ============
        # Initialize multimodal components
        self.supported_modalities=[starting_modality]
        self.supported_modalities_in_chans=[starting_n_chans]
        self.patch_embedders = nn.ModuleDict()
        self.cls_tokens = nn.ParameterDict()
        self.storage_tokens = nn.ParameterDict()
        self.modality_specific_layer_adaptors = nn.ModuleDict() # shortened as "msla" as followed
        self.modality_encoders = nn.ParameterDict()
        self.modality_fusion_lora_adaptors = nn.ModuleDict()
        self.modality_specific_mask_tokens = nn.ParameterDict()
        # Initialize modality-specific components for starting_modality (starting_n_chans)
        self.add_new_patch_embedders(starting_modality,starting_n_chans)
        self.add_new_cls_token(starting_modality)
        if self.n_storage_tokens > 0: self.add_new_storage_tokens(starting_modality)
        self.add_new_msla(starting_modality)
        self.add_modality_encoder(starting_modality)
        self.add_new_mfla(starting_modality)
        self.add_new_mask_token(starting_modality)

    # Helper Functions to initialize and add new component as modality comes in.
    def initialize_rope_embed(self):
        self.rope_embed = RopePositionEmbedding(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            base=self.pos_embed_rope_base,
            min_period=self.pos_embed_rope_min_period,
            max_period=self.pos_embed_rope_max_period,
            normalize_coords=self.pos_embed_rope_normalize_coords,
            shift_coords=self.pos_embed_rope_shift_coords,
            jitter_coords=self.pos_embed_rope_jitter_coords,
            rescale_coords=self.pos_embed_rope_rescale_coords,
            dtype=dtype_dict[self.pos_embed_rope_dtype],
            device=self.device,
        )
    def initialize_blocks(self):
        logger.info(f"using {self.ffn_layer} layer as FFN")
        blocks_list = self.create_fft_list(self.n_blocks)
        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)
    def add_new_patch_embedders(self,modality_name,in_chans):
        assert modality_name not in self.patch_embedders, f"{modality_name} already in patch_embedders"
        self.patch_embedders[modality_name]=PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=in_chans,
            embed_dim=self.embed_dim,
            flatten_embedding=False,
        )
    def add_new_cls_token(self,modality_name,init_modality=None):
        assert modality_name not in self.cls_tokens, f"{modality_name} already in cls_tokens"
        if init_modality in self.cls_tokens:
            print(f"initializing {modality_name} clstoken from {init_modality} clstoken.")
            self.cls_tokens[modality_name] = nn.Parameter(self.cls_tokens[self.starting_modality].data.clone())
        else:
            self.cls_tokens[modality_name] = nn.Parameter(torch.empty(1, 1, self.embed_dim, device=self.device))
    def add_new_storage_tokens(self,modality_name):
        assert modality_name not in self.storage_tokens, f"{modality_name} already in storage_tokens"
        self.storage_tokens[modality_name] = nn.Parameter(torch.empty(1, self.n_storage_tokens, self.embed_dim, device=self.device))
    def add_new_msla(self, modality, init="backbone"):
        """
        Add modality-specific layer adaptors.

        Args:
            modality: Name of the modality
            init: Weight initialization. "backbone" copies from self.blocks,
                  a modality name copies from that modality, None for random init.
        """
        if self.tz_modality_specific_layer_augmenter == "lora":
            self.modality_specific_layer_adaptors[modality] = self.create_lora_list(self.tz_fusion_time)
        elif self.tz_modality_specific_layer_augmenter == "fft":
            self.modality_specific_layer_adaptors[modality] = self.create_fft_list(self.tz_fusion_time)
            if init is not None:
                self.copy_weights_to_adaptor(self.modality_specific_layer_adaptors[modality], source=init, block_offset=0)
        else:
            raise RuntimeError(f"unrecognized {self.tz_modality_specific_layer_augmenter=}")
    def add_modality_encoder(self,modality_name):
        self.modality_encoders[modality_name]=nn.Parameter(
            torch.zeros(1, 1, self.embed_dim, device=self.device)
        )
    def add_new_mfla(self, modality, init="backbone"):
        """
        Add modality fusion layer adaptors.

        Args:
            modality: Name of the modality
            init: Weight initialization. "backbone" copies from self.blocks,
                  a modality name copies from that modality, None for random init.
        """
        num_fusion_blocks = self.n_blocks - self.tz_fusion_time
        if self.tz_modality_fusion_layer_augmenter == "lora":
            self.modality_fusion_lora_adaptors[modality] = self.create_lora_list(num_fusion_blocks)
        elif self.tz_modality_fusion_layer_augmenter == "fft":
            self.modality_fusion_lora_adaptors[modality] = self.create_fft_list(num_fusion_blocks)
            if init is not None:
                self.copy_weights_to_adaptor(self.modality_fusion_lora_adaptors[modality], source=init, block_offset=self.tz_fusion_time)
    def add_new_mask_token(self,modality):
        self.modality_specific_mask_tokens[modality]=nn.Parameter(torch.randn(self.embed_dim, device=self.device))
    def create_lora_list(self,length):
        return nn.ModuleList([
            LoRALayer(self.embed_dim, rank=self.tz_lora_rank, device=self.device) for _ in range(length)
        ])
    def create_fft_list(self,length):
        ffn_layer_cls = ffn_layer_dict[self.ffn_layer]
        return nn.ModuleList([
                SelfAttentionBlock(
                    dim=self.embed_dim,
                    num_heads=self.num_heads,
                    ffn_ratio=self.ffn_ratio,
                    qkv_bias=self.qkv_bias,
                    proj_bias=self.proj_bias,
                    ffn_bias=self.ffn_bias,
                    drop_path=self.drop_path_rate,
                    norm_layer=norm_layer_dict[self.norm_layer],
                    act_layer=nn.GELU,
                    ffn_layer=ffn_layer_cls,
                    init_values=self.layerscale_init,
                    mask_k_bias=self.mask_k_bias,
                    device=self.device,
                )
                for i in range(length)
            ])

    def copy_weights_to_adaptor(self, adaptor_list, source="backbone", block_offset=0):
        """
        Copy weights from a source to an adaptor list.

        Args:
            adaptor_list: Target nn.ModuleList to copy weights into
            source: Weight source. Options:
                - "backbone": Copy from self.blocks
                - <modality_name>: Copy from that modality's msla/mfla
            block_offset: Starting block index when copying from backbone
        """
        for i, adaptor in enumerate(adaptor_list):
            if source == "backbone":
                src_block = self.blocks[block_offset + i]
            else:
                block_idx = i + block_offset
                if block_idx < self.tz_fusion_time:
                    src_block = self.modality_specific_layer_adaptors[source][block_idx]
                else:
                    src_block = self.modality_fusion_lora_adaptors[source][block_idx - self.tz_fusion_time]
            adaptor.load_state_dict(src_block.state_dict())

    def load_pretrained_dino(self, model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", load_weights:bool=True):
        """
        Load pretrained DINO weights from HuggingFace into EVAN.

        Args:
            model_name: HuggingFace model name (default: facebook/dinov3-vitl16-pretrain-lvd1689m)
        """
        if not load_weights:
            print("pretrained=False, not loading weight and returning directly.")
            return
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
            new_key = new_key.replace('embeddings.cls_token', f'cls_tokens.{self.starting_modality}')
            new_key = new_key.replace('embeddings.mask_token', 'mask_token')
            new_key = new_key.replace('embeddings.register_tokens', f'storage_tokens.{self.starting_modality}')
            new_key = new_key.replace('embeddings.patch_embeddings.weight', 'patch_embed.proj.weight')
            new_key = new_key.replace('embeddings.patch_embeddings.bias', 'patch_embed.proj.bias')
            new_key = new_key.replace('layer.', 'blocks.')
            if 'mask_token' in new_key and value.ndim == 3:
                value = value.squeeze(1)
            elif 'cls_tokens' in new_key and value.ndim == 2:
                value = value.unsqueeze(1)
            # Attention mapping: separate q,k,v to combined qkv
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
            # EVAN multi-modality compatibility: Remap patch_embed.* to patch_embedders.<starting_modality>.*
            # Only copy patch embedder weights if starting_modality is 'rgb' (DINO was trained on RGB)
            # For other modalities, skip patch embedder weights (random init instead)
            if 'patch_embed.' in new_key:
                if self.starting_modality == 'rgb':
                    new_key = new_key.replace('patch_embed.', f'patch_embedders.{self.starting_modality}.')
                else:
                    continue  # Skip patch embedder weights for non-rgb starting modalities
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
        print(f"    - From HuggingFace checkpoint: {checkpoint_params:,}")
        print(f"    - Created (K bias zeros): {created_params:,}")

        # For FFT mode: Copy first tz_fusion_time blocks to RGB modality-specific layers
        if self.tz_modality_specific_layer_augmenter == "fft":
            print(f"\n  FFT mode: Copying first {self.tz_fusion_time} DINO blocks to {self.starting_modality} modality-specific layers...")
            fft_params_copied = 0
            for i in range(self.tz_fusion_time):
                # Copy all weights from blocks[i] to modality_specific_layer_adaptors.rgb[i]
                for key, value in list(checkpoint.items()):
                    if key.startswith(f'blocks.{i}.'):
                        # Create corresponding key for RGB modality-specific layer
                        rgb_key = key.replace(f'blocks.{i}.', f'modality_specific_layer_adaptors.{self.starting_modality}.{i}.')
                        checkpoint[rgb_key] = value.clone()
                        fft_params_copied += value.numel()
            print(f"    Copied {fft_params_copied:,} parameters for {self.starting_modality} FFT blocks")
        
        if self.tz_modality_fusion_layer_augmenter == "fft":
            print(f"\n  FFT mode: Copying last {self.n_blocks} - {self.tz_fusion_time} DINO blocks to {self.starting_modality} modality-fusion layers...")
            fft_params_copied = 0
            for i in range(self.tz_fusion_time,self.n_blocks,1):
                # Copy all weights from blocks[i] to modality_fusion_lora_adaptors.<starting_modality>[i]
                for key, value in list(checkpoint.items()):
                    if key.startswith(f'blocks.{i}.'):
                        # Create corresponding key for starting_modality fusion layer
                        mod_key = key.replace(f'blocks.{i}.', f'modality_fusion_lora_adaptors.{self.starting_modality}.{i - self.tz_fusion_time}.')
                        checkpoint[mod_key] = value.clone()
                        fft_params_copied += value.numel()
            print(f"    Copied {fft_params_copied:,} parameters for {self.starting_modality} FFT blocks")

        result = self.load_state_dict(checkpoint, strict=False)

        # Filter out expected missing keys (EVAN-specific multi-modality components)
        def is_expected_missing(key):
            # RoPE periods buffer: computed deterministically from hyperparameters, not in HF checkpoint
            if key == 'rope_embed.periods':
                return True
            if self.tz_modality_specific_layer_augmenter == "lora":
                if 'modality_specific_layer_adaptors' in key:
                    return True
            if any(pattern in key for pattern in ['modality_encoders', 'modality_fusion_lora_adaptors']):
                return True
            return False

        unexpected_missing = [k for k in result.missing_keys if not is_expected_missing(k)]

        if len(unexpected_missing) > 0:
            unexpected_missing_params = sum(
                self.state_dict()[k].numel()
                for k in unexpected_missing
                if k in self.state_dict()
            )
            print(f"  !!  Unexpected missing keys (in EVAN but not in DINO): {len(unexpected_missing)}")
            print(f"    Keys: {unexpected_missing}")
            print(f"    Parameters: {unexpected_missing_params:,}")

        # Report unexpected keys (in DINO checkpoint but not in EVAN)
        if len(result.unexpected_keys) > 0:
            untransferred_params = sum(
                checkpoint[k].numel()
                for k in result.unexpected_keys
                if k in checkpoint
            )
            print(f"  !!  Unexpected keys (in DINO but not loaded into EVAN): {len(result.unexpected_keys)}")
            print(f"    First 10 keys: {result.unexpected_keys[:10]}")
            print(f"    Untransferred parameters: {untransferred_params:,}")

        print("\nWeights loaded successfully!")
        print("=== DINO weight loading complete ===\n")

    def create_modality_components(self, modality_key: str, in_chans: int):
        """
        Add all needed components to EVAN.

        Args:
            modality_key: Name/identifier for the new modality
            in_chans: Number of input channels for this modality
        """
        params_before = sum(p.numel() for p in self.parameters())
        self.add_new_patch_embedders(modality_key, in_chans)
        self.add_new_msla(modality_key, init="backbone")
        self.add_new_mfla(modality_key, init="backbone")
        self.add_new_cls_token(modality_key, init_modality=self.starting_modality)
        self.add_new_storage_tokens(modality_key)
        self.add_modality_encoder(modality_key)
        self.add_new_mask_token(modality_key)
        params_after = sum(p.numel() for p in self.parameters())
        new_params = params_after - params_before
        self.supported_modalities.append(modality_key)
        self.supported_modalities_in_chans.append(in_chans)

        num_fusion_blocks = self.n_blocks - self.tz_fusion_time
        embedder_params = sum(p.numel() for p in self.patch_embedders[modality_key].parameters())
        msla_params = sum(p.numel() for p in self.modality_specific_layer_adaptors[modality_key].parameters())
        mfla_params = sum(p.numel() for p in self.modality_fusion_lora_adaptors[modality_key].parameters())
        cls_token_params = self.cls_tokens[modality_key].numel()
        storage_token_params = self.storage_tokens[modality_key].numel() if self.n_storage_tokens > 0 else 0
        encoding_params = self.modality_encoders[modality_key].numel()

        logger.info(f"Initialized new modality: '{modality_key}'")
        logger.info(f"   - Input channels: {in_chans}")
        logger.info(f"   - Mod-spec Augmenter mode: {self.tz_modality_specific_layer_augmenter}")
        logger.info(f"   - Mod-fuse Augmenter mode: {self.tz_modality_fusion_layer_augmenter}")
        logger.info(f"   - Components created:")
        logger.info(f"     • Patch embedder: {embedder_params:,} params")
        logger.info(f"     • Modality-specific {self.tz_modality_specific_layer_augmenter} ({self.tz_fusion_time} blocks): {msla_params:,} params")
        logger.info(f"     • Modality-fusion {self.tz_modality_fusion_layer_augmenter} ({num_fusion_blocks} blocks): {mfla_params:,} params")
        logger.info(f"     • CLS token: {cls_token_params:,} params")
        logger.info(f"     • Storage tokens: {storage_token_params:,} params")
        logger.info(f"     • Modality encoding: {encoding_params:,} params")
        logger.info(f"   - Total new parameters: {new_params:,}")
        logger.info(f"   - Currently supported modalities: {self.supported_modalities}")

    def prepare_tokens_with_masks(self, x: Tensor, modality_key: str, masks=None) -> Tuple[Tensor, Tuple[int]]:
        """
        Prepare tokens with optional masks and modality-specific embedder.

        Args:
            x: Input tensor
            modality_key: Key identifying which modality's embedder and CLS/storage tokens to use
            masks: Optional mask tensor

        Returns:
            Tuple of (tokens with CLS and storage prepended, (H, W) spatial dimensions)
        """
        embedder = self.patch_embedders[modality_key]
        x = embedder(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_tokens[modality_key]
        else:
            cls_token = self.cls_tokens[modality_key] + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens[modality_key]
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

    def forward_features(self, x: Dict[str, Tensor] | List[Tensor], masks: Optional[Tensor] = None) -> Dict[str, Tensor]:
        """
        Forward features with multi-modality support.

        Args:
            x: Input tensor, dict of tensors (modality: tensor), or list of tensors
            masks: Optional mask tensor

        Returns:
            Dictionary with normalized features
        """
        # Now x is a dict of modalities
        if isinstance(x, torch.Tensor) or not isinstance(x, dict) or len(x) == 0:
            raise ValueError("Input must be a non-empty dict of modalities.")

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
        # This allows cross-modal attention during fusion
        x_patches_concat = torch.cat(all_patches, dim=1)  # [B, sum(num_patches), embed_dim]

        # Concatenate all CLS/storage tokens from all modalities
        all_cls_storage_list = [all_cls_storage[k] for k in sorted(all_cls_storage.keys())]
        x_cls_storage_concat = torch.cat(all_cls_storage_list, dim=1)  # [B, n_modalities * (1 + n_storage), embed_dim]
        n_cls_storage_per_modality = self.n_storage_tokens + 1
        n_modalities = len(all_cls_storage)

        # Compute H, W from image size and patch size
        H = W = self.img_size // self.patch_size

        # Combine for fusion processing
        x_fused = torch.cat([x_cls_storage_concat, x_patches_concat], dim=1)

        # Step 4: Process through fusion blocks with cross-modal attention
        # All patches from all modalities attend to each other here!
        for i in range(self.tz_fusion_time, len(self.blocks)):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None

            if self.tz_modality_fusion_layer_augmenter=="lora":
                x_fused = self.blocks[i](x_fused, rope_sincos)
                lora_idx = i - self.tz_fusion_time
                lora_output = 0
                for modality_key in embedded_modalities.keys():
                    lora = self.modality_fusion_lora_adaptors[modality_key][lora_idx]
                    lora_output = lora_output + lora(x_fused)
                x_fused = x_fused + lora_output
            elif self.tz_modality_fusion_layer_augmenter=="fft":
                x_fused = self.modality_fusion_lora_adaptors[i](x_fused, rope_sincos)

        # Step 5: Split fused representation back into modality-specific outputs
        # Total CLS/storage tokens = n_modalities * (1 + n_storage_tokens)
        total_cls_storage = n_modalities * n_cls_storage_per_modality
        x_cls_storage_fused = x_fused[:, :total_cls_storage, :]
        x_patches_fused = x_fused[:, total_cls_storage:, :]
        output_dict = {}
        sorted_modalities = sorted(embedded_modalities.keys())
        for mod_idx, modality_key in enumerate(sorted_modalities):
            # Extract this modality's CLS/storage tokens
            cls_start = mod_idx * n_cls_storage_per_modality
            cls_end = cls_start + n_cls_storage_per_modality
            x_cls_storage_mod = x_cls_storage_fused[:, cls_start:cls_end, :]

            # Extract this modality's patches
            info = modality_info[modality_key]
            patch_start, patch_end = info['start_idx'], info['end_idx']
            patches = x_patches_fused[:, patch_start:patch_end, :]

            # Recombine this modality's CLS/storage with its patches
            x_mod = torch.cat([x_cls_storage_mod, patches], dim=1)

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
            x_mod, (H, W) = self.prepare_tokens_with_masks(modality_tensor, modality_key, masks=masks)
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

            embedded_modalities[modality_key] = x_mod

        return embedded_modalities

    def forward_fusion_from_modality_features(
        self,
        embedded_modalities: Dict[str, Tensor],
        hallucinated_modalities: Optional[set] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Forward through fusion blocks only, starting from pre-computed modality-specific features.

        Useful for MAE training where masking is applied between modality-specific and fusion stages.
        This method extracts the fusion logic from forward_features (steps 3-5).

        Args:
            embedded_modalities: Dict mapping modality_key -> tensor [B, 1+n_storage+num_patches, embed_dim]
                                Output from forward_modality_specific_features (possibly with masking applied)
            hallucinated_modalities: Optional set of modality names that are hallucinated (not real).
                                    When provided, only MFLAs for these modalities are applied.
                                    When None, no MFLAs are applied (backward compatible behavior).

        Returns:
            Dictionary with normalized features per modality (same format as forward_features):
            {modality: {'x_norm_clstoken', 'x_norm_patchtokens', 'x_storage_tokens', 'x_prenorm'}}
        """
        # Step 1: Add modality encoding to each modality's features
        for modality_key in embedded_modalities.keys():
            modality_encoding = self.modality_encoders[modality_key]
            embedded_modalities[modality_key] = embedded_modalities[modality_key] + modality_encoding

        # Step 2: Concatenate patches for fusion (allow cross-modal attention)
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

        # Concatenate all patches along sequence dimension for cross-modal attention
        x_patches_concat = torch.cat(all_patches, dim=1)  # [B, sum(num_patches), embed_dim]

        # Concatenate all CLS/storage tokens from all modalities
        all_cls_storage_list = [all_cls_storage[k] for k in sorted(all_cls_storage.keys())]
        x_cls_storage_concat = torch.cat(all_cls_storage_list, dim=1)
        n_cls_storage_per_modality = self.n_storage_tokens + 1
        n_modalities = len(all_cls_storage)

        # Compute H, W from image size and patch size
        H = W = self.img_size // self.patch_size

        # Combine for fusion processing
        x_fused = torch.cat([x_cls_storage_concat, x_patches_concat], dim=1)

        # Step 2: Process through fusion blocks with cross-modal attention
        for i in range(self.tz_fusion_time, len(self.blocks)):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None

            # Shared block forward on concatenated representation
            x_fused = self.blocks[i](x_fused, rope_sincos)

            # Add fusion LoRA only for hallucinated modalities (if specified)
            if hallucinated_modalities:
                lora_idx = i - self.tz_fusion_time
                lora_output = 0
                for modality_key in hallucinated_modalities:
                    if modality_key in self.modality_fusion_lora_adaptors:
                        lora = self.modality_fusion_lora_adaptors[modality_key][lora_idx]
                        lora_output = lora_output + lora(x_fused)
                x_fused = x_fused + lora_output
            # else: no MFLA applied when hallucinated_modalities is None (all real)

        # Step 3: Split fused representation back into modality-specific outputs
        total_cls_storage = n_modalities * n_cls_storage_per_modality
        x_cls_storage_fused = x_fused[:, :total_cls_storage, :]
        x_patches_fused = x_fused[:, total_cls_storage:, :]

        # Split back into modality-specific outputs
        output_dict = {}
        sorted_modalities = sorted(embedded_modalities.keys())
        for mod_idx, modality_key in enumerate(sorted_modalities):
            # Extract this modality's CLS/storage tokens
            cls_start = mod_idx * n_cls_storage_per_modality
            cls_end = cls_start + n_cls_storage_per_modality
            x_cls_storage_mod = x_cls_storage_fused[:, cls_start:cls_end, :]

            # Extract this modality's patches
            info = modality_info[modality_key]
            patch_start, patch_end = info['start_idx'], info['end_idx']
            patches = x_patches_fused[:, patch_start:patch_end, :]

            # Recombine this modality's CLS/storage with its patches
            x_mod = torch.cat([x_cls_storage_mod, patches], dim=1)

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
            }

        return output_dict

    def forward_features_with_pseudo_modality(
        self,
        x: Dict[str, Tensor],
        pseudo_modalities: List[str],
        intermediate_projectors: nn.ModuleDict,
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Forward pass with pseudo-modalities using full sequence projection.

        When a modality is missing at inference time, this method creates pseudo-features
        by projecting the full sequence (CLS + storage + patches) from available modalities.

        Args:
            x: Dictionary of available modality tensors
            pseudo_modalities: List of modality names to hallucinate (e.g., ['vre'])
            intermediate_projectors: Trained sequence projectors with keys like 'rgb_to_vre', 'vre_to_rgb'

        Returns:
            Dictionary with normalized features per modality (same format as forward_features)
        """
        # Step 1: Get real modality features
        embedded = self.forward_modality_specific_features(x)
        available_modalities = list(embedded.keys())

        # Step 2: Create pseudo-features for missing modalities using full sequence projection
        for mod in pseudo_modalities:
            # Project full sequence from all available modalities and take mean
            projected_seqs = []
            for avail_mod in available_modalities:
                avail_seq = embedded[avail_mod]  # [B, seq_len, embed_dim]
                avail_seq_norm = F.layer_norm(avail_seq, [avail_seq.shape[-1]])
                key = f"{avail_mod}_to_{mod}"
                projected = intermediate_projectors[key](avail_seq_norm)  # [B, seq_len, embed_dim]
                projected_seqs.append(projected)
            # Mean of all projected sequences
            embedded[mod] = torch.stack(projected_seqs).mean(dim=0)

        # Step 3: Forward through fusion (modality encoding added here)
        return self.forward_fusion_from_modality_features(embedded)

    def set_requires_grad(
        self,
        modality: str,
        *,
        patch_embedders: bool = False,
        clsreg: bool = False,
        msla: bool = False,
        modality_encoders: bool = False,
        mfla: bool = False,
        blocks: bool = False,
        norm: bool = False,
        mask_token: bool = False,
    ):
        """
        Set requires_grad for specific components of the model.

        Args:
            modality: Either a modality key (e.g., 'rgb', 'vre') or 'backbone' for shared weights.
                      Use 'all' to apply to all modalities.
            patch_embedders: Unfreeze patch embedders for this modality
            clsreg: Unfreeze CLS and storage (register) tokens for this modality
            msla: Unfreeze modality-specific layer adaptors (LoRAs or FFT blocks)
            modality_encoders: Unfreeze modality encodings
            mfla: Unfreeze modality fusion LoRA adaptors
            blocks: Unfreeze shared transformer blocks (only when modality='backbone')
            norm: Unfreeze shared norm layers (only when modality='backbone')
            mask_token: Unfreeze mask token (only when modality='backbone')
        """
        if modality == 'backbone':
            # Shared backbone components
            if blocks:
                for param in self.blocks.parameters():
                    param.requires_grad = True
            if norm:
                for param in self.norm.parameters():
                    param.requires_grad = True
                if self.cls_norm is not None:
                    for param in self.cls_norm.parameters():
                        param.requires_grad = True
                if self.local_cls_norm is not None:
                    for param in self.local_cls_norm.parameters():
                        param.requires_grad = True
            if mask_token:
                self.mask_token.requires_grad = True
        elif modality == 'all':
            # Apply to all modalities
            for mod_key in self.patch_embedders.keys():
                self.set_requires_grad(
                    mod_key,
                    patch_embedders=patch_embedders,
                    clsreg=clsreg,
                    msla=msla,
                    modality_encoders=modality_encoders,
                    mfla=mfla,
                )
        else:
            # Modality-specific components
            if patch_embedders and modality in self.patch_embedders:
                for param in self.patch_embedders[modality].parameters():
                    param.requires_grad = True
            if clsreg:
                if modality in self.cls_tokens:
                    self.cls_tokens[modality].requires_grad = True
                if self.n_storage_tokens > 0 and modality in self.storage_tokens:
                    self.storage_tokens[modality].requires_grad = True
            if msla and modality in self.modality_specific_layer_adaptors:
                for param in self.modality_specific_layer_adaptors[modality].parameters():
                    param.requires_grad = True
            if modality_encoders and modality in self.modality_encoders:
                self.modality_encoders[modality].requires_grad = True
            if mfla and modality in self.modality_fusion_lora_adaptors:
                for param in self.modality_fusion_lora_adaptors[modality].parameters():
                    param.requires_grad = True

    def freeze_all(self):
        """Freeze all parameters in the model."""
        for param in self.parameters():
            param.requires_grad = False

    def get_config(self) -> Dict[str, Any]:
        """Return config dict needed to reconstruct this model architecture."""
        return {
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'pos_embed_rope_base': self.pos_embed_rope_base,
            'pos_embed_rope_min_period': self.pos_embed_rope_min_period,
            'pos_embed_rope_max_period': self.pos_embed_rope_max_period,
            'pos_embed_rope_normalize_coords': self.pos_embed_rope_normalize_coords,
            'pos_embed_rope_shift_coords': self.pos_embed_rope_shift_coords,
            'pos_embed_rope_jitter_coords': self.pos_embed_rope_jitter_coords,
            'pos_embed_rope_rescale_coords': self.pos_embed_rope_rescale_coords,
            'pos_embed_rope_dtype': self.pos_embed_rope_dtype,
            'embed_dim': self.embed_dim,
            'depth': self.n_blocks,
            'num_heads': self.num_heads,
            'ffn_ratio': self.ffn_ratio,
            'qkv_bias': self.qkv_bias,
            'drop_path_rate': self.drop_path_rate,
            'layerscale_init': self.layerscale_init,
            'norm_layer': self.norm_layer,
            'ffn_layer': self.ffn_layer,
            'ffn_bias': self.ffn_bias,
            'proj_bias': self.proj_bias,
            'n_storage_tokens': self.n_storage_tokens,
            'mask_k_bias': self.mask_k_bias,
            'untie_cls_and_patch_norms': self.untie_cls_and_patch_norms,
            'untie_global_and_local_cls_norm': self.untie_global_and_local_cls_norm,
            'tz_modality_specific_layer_augmenter': self.tz_modality_specific_layer_augmenter,
            'tz_modality_fusion_layer_augmenter': self.tz_modality_fusion_layer_augmenter,
            'tz_fusion_time': self.tz_fusion_time,
            'tz_lora_rank': self.tz_lora_rank,
            'starting_modality': self.starting_modality,
            'starting_n_chans': self.supported_modalities_in_chans[0],
            'supported_modalities': self.supported_modalities.copy(),
            'supported_modalities_in_chans': self.supported_modalities_in_chans.copy(),
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint with config and state dict."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
        }
        torch.save(checkpoint, path)
        print(f"EVAN checkpoint saved to: {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: Any | None = None) -> "EVAN":
        """
        Load EVAN model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model to

        Returns:
            EVAN model with loaded weights
        """
        checkpoint = torch.load(path, map_location=device or 'cpu')
        config = checkpoint['config']

        # Extract modality info before creating model
        supported_modalities = config.pop('supported_modalities', None)
        supported_modalities_in_chans = config.pop('supported_modalities_in_chans', None)
        starting_modality = config.get('starting_modality', 'rgb')

        # Backward compatibility: if starting_n_chans not in config, get it from supported_modalities_in_chans
        if 'starting_n_chans' not in config and supported_modalities_in_chans:
            config['starting_n_chans'] = supported_modalities_in_chans[0]

        # Create model with base config (starting modality only)
        model = cls(**config, device=device)

        # Add any additional modalities that were in the checkpoint
        if supported_modalities and supported_modalities_in_chans:
            for mod, n_chans in zip(supported_modalities, supported_modalities_in_chans):
                if mod != starting_modality and mod not in model.patch_embedders:
                    model.create_modality_components(mod, n_chans)

        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        print(f"EVAN loaded from checkpoint: {path}")
        return model

    def forward(self, *args, is_training: bool = False, **kwargs) -> Dict[str, Dict[str, Tensor]] | Dict[str, Tensor] | Tensor:
        raise NotImplementedError("Why are you calling me?")

# EVAN preset functions (similar to DINOv3)

def evan_small(pretrained: str = "facebook/dinov3-vits16-pretrain-lvd1689m", load_weights:bool=True, **kwargs):
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
    model.load_pretrained_dino(model_name=pretrained, load_weights=load_weights)
    return model


def evan_base(pretrained: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", load_weights:bool=True, **kwargs):
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
    model.load_pretrained_dino(model_name=pretrained,load_weights=load_weights)
    return model


def evan_large(pretrained: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", load_weights:bool=True, **kwargs):
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
    model.load_pretrained_dino(model_name=pretrained,load_weights=load_weights)
    return model


class EVANClassifier(nn.Module):
    """Classifier head on top of EVAN for EuroSAT."""

    def __init__(self, evan_model, num_classes=10, classifier_strategy='mean', factor=4, global_rep="clstoken", device = "cuda"):
        super().__init__()
        self.evan = evan_model
        self.classifier_strategy = classifier_strategy
        self.num_classes = num_classes
        self.factor = factor
        self.global_rep=global_rep
        embed_dim = self.evan.embed_dim
        hidden_dim = embed_dim * factor
        self.hidden_dim = hidden_dim
        self.device = device

        if classifier_strategy == 'mean':
            # Average CLS tokens from all modalities, then classify
            self.classifier = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, num_classes)
            )
            self.modality_classifiers = None
        elif classifier_strategy == 'ensemble':
            # Per-modality classifiers that get ensembled
            self.classifier = None
            self.modality_classifiers = nn.ModuleDict()
            self.instantiate_modality_classifier(evan_model.starting_modality)
        else:
            raise ValueError(f"Unknown fusion strategy: {classifier_strategy}")

    def instantiate_modality_classifier(self, modality_key: str):
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

    def classify_from_features(self, features_dict):
        """
        Classify from pre-computed features dict.

        Args:
            features_dict: Output from evan.forward_features or forward_features_with_pseudo_modality

        Returns:
            logits: [B, num_classes]
        """
        if self.classifier_strategy == 'mean':
            cls_tokens = []
            for modality in sorted(features_dict.keys()):
                if self.global_rep == "clstoken":
                    cls_tokens.append(features_dict[modality]['x_norm_clstoken'])
                elif self.global_rep == "mean_patch":
                    cls_tokens.append(features_dict[modality]['x_norm_patchtokens'].mean(1))
            fused = torch.stack(cls_tokens).mean(dim=0)
            return self.classifier(fused)

        elif self.classifier_strategy == 'ensemble':
            all_logits = []
            for modality in sorted(features_dict.keys()):
                if modality not in self.modality_classifiers:
                    raise RuntimeError(f"{modality} doesn't have its own classifier.")
                if self.global_rep == "clstoken":
                    cls_token = features_dict[modality]['x_norm_clstoken']
                elif self.global_rep == "mean_patch":
                    cls_token = features_dict[modality]['x_norm_patchtokens'].mean(1)
                else:
                    raise ValueError(f"unrecognized global_rep arg, choices are clstoken or mean_patch, received {self.global_rep}")
                modality_logits = self.modality_classifiers[modality](cls_token)
                all_logits.append(modality_logits)
            return torch.stack(all_logits).mean(dim=0)

        else:
            raise ValueError(f"Unknown classifier strategy: {self.classifier_strategy}")

    def forward(self, x, pseudo_modalities=None, intermediate_projectors=None):
        """
        Forward pass supporting both single tensor and dict inputs.
        Args:
            x: Either a tensor [B, C, H, W] or dict {modality: tensor}
            pseudo_modalities: Optional list of modalities to hallucinate using sequence projection
            intermediate_projectors: Required if pseudo_modalities is provided; trained sequence projectors
        Returns:
            logits: [B, num_classes]
        """
        if pseudo_modalities is not None:
            features_dict = self.evan.forward_features_with_pseudo_modality(
                x, pseudo_modalities, intermediate_projectors
            )
        else:
            features_dict = self.evan.forward_features(x)
        return self.classify_from_features(features_dict)
    
    def switch_strategy(self,target_strategy,key=None):
        if self.classifier_strategy == 'strategy': 
            print(f"Already using {target_strategy} head")
        elif target_strategy=="mean":
            self.ensemble_to_mean(key)
        elif target_strategy=="ensemble":
            self.mean_to_ensemble()
            for mod in self.evan.patch_embedders.keys():
                self.instantiate_modality_classifier(mod)
        return
            
    def mean_to_ensemble(self):
        """Convert from 'mean' to 'ensemble' strategy, the existing classifier becomes new_key classifier"""
        if self.classifier_strategy == 'ensemble':
            print("!!!!  mean_to_ensemble was called on classifier but it already is ensemble. No changes made.")
            return()
        assert self.classifier_strategy == 'mean'
        self.modality_classifiers = nn.ModuleDict()
        for mod in self.evan.supported_modalities:
            self.modality_classifiers[mod]=copy.deepcopy(self.classifier)
        self.classifier=None
        self.classifier_strategy='ensemble'
        print("!! Evan Classifier has switched strategy from mean to ensemble")
        
    def ensemble_to_mean(self, key_to_keep:str='rgb'):
        """Convert from 'ensemble' to 'mean' strategy, keeping only key_to_keep classifier"""
        if self.classifier_strategy == 'mean':
            print("!!!!  mean_to_ensemble was called on classifier but it already is mean. No changes made.")
            return()
        assert self.classifier_strategy == 'ensemble'
        self.classifier=self.modality_classifiers[key_to_keep]
        self.modality_classifiers=None
        self.classifier_strategy = 'mean'
        print("!! Evan Classifier has switched strategy from ensemble to mean")
        
    def set_requires_grad(
        self,
        modality: str,
        *,
        patch_embedders: bool = False,
        clsreg: bool = False,
        msla: bool = False,
        modality_encoders: bool = False,
        mfla: bool = False,
        blocks: bool = False,
        norm: bool = False,
        mask_token: bool = False,
        classifier: bool = False,
    ):
        """
        Set requires_grad for specific components of the model (EVAN + classifier).

        Args:
            modality: Either a modality key (e.g., 'rgb', 'vre'), 'backbone' for shared weights,
                      or 'all' to apply to all modalities.
            patch_embedders: Unfreeze patch embedders for this modality
            clsreg: Unfreeze CLS and storage (register) tokens for this modality
            msla: Unfreeze modality-specific layer adaptors (LoRAs or FFT blocks)
            modality_encoders: Unfreeze modality encodings
            mfla: Unfreeze modality fusion LoRA adaptors
            blocks: Unfreeze shared transformer blocks (only when modality='backbone')
            norm: Unfreeze shared norm layers (only when modality='backbone')
            mask_token: Unfreeze mask token (only when modality='backbone')
            classifier: Unfreeze classifier head(s)
        """
        # Delegate to EVAN's set_requires_grad for backbone components
        self.evan.set_requires_grad(
            modality,
            patch_embedders=patch_embedders,
            clsreg=clsreg,
            msla=msla,
            modality_encoders=modality_encoders,
            mfla=mfla,
            blocks=blocks,
            norm=norm,
            mask_token=mask_token,
        )

        # Handle classifier
        if classifier:
            if self.classifier is not None:
                for param in self.classifier.parameters():
                    param.requires_grad = True
            if self.modality_classifiers is not None:
                if modality == 'all':
                    for mod_classifier in self.modality_classifiers.values():
                        for param in mod_classifier.parameters():
                            param.requires_grad = True
                elif modality in self.modality_classifiers:
                    for param in self.modality_classifiers[modality].parameters():
                        param.requires_grad = True
                else:
                    raise RuntimeError(f"{modality} not in modality_classifiers")

    def freeze_all(self):
        """Freeze all parameters in the model (EVAN + classifier)."""
        for param in self.parameters():
            param.requires_grad = False

    def get_config(self) -> Dict[str, Any]:
        """Return config dict needed to reconstruct this model architecture."""
        return {
            'evan_config': self.evan.get_config(),
            'num_classes': self.num_classes,
            'classifier_strategy': self.classifier_strategy,
            'factor': self.factor,
            'global_rep': self.global_rep,
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint with config and state dict."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
        }
        torch.save(checkpoint, path)
        print(f"EVANClassifier checkpoint saved to: {path}")

    @classmethod
    def from_checkpoint(cls, path: str, device: Any | None = None) -> "EVANClassifier":
        """
        Load EVANClassifier model from checkpoint.

        Args:
            path: Path to checkpoint file
            device: Device to load model to

        Returns:
            EVANClassifier model with loaded weights
        """
        checkpoint = torch.load(path, map_location=device or 'cpu')
        config = checkpoint['config']
        evan_config = config['evan_config']

        # Extract modality info before creating EVAN
        supported_modalities = evan_config.pop('supported_modalities', None)
        supported_modalities_in_chans = evan_config.pop('supported_modalities_in_chans', None)
        starting_modality = evan_config.get('starting_modality', 'rgb')

        # Backward compatibility: if starting_n_chans not in config, get it from supported_modalities_in_chans
        if 'starting_n_chans' not in evan_config and supported_modalities_in_chans:
            evan_config['starting_n_chans'] = supported_modalities_in_chans[0]

        # Create EVAN model
        evan = EVAN(**evan_config, device=device)

        # Add any additional modalities
        if supported_modalities and supported_modalities_in_chans:
            for mod, n_chans in zip(supported_modalities, supported_modalities_in_chans):
                if mod != starting_modality and mod not in evan.patch_embedders:
                    evan.create_modality_components(mod, n_chans)

        # Create classifier
        model = cls(
            evan_model=evan,
            num_classes=config['num_classes'],
            classifier_strategy=config['classifier_strategy'],
            factor=config['factor'],
            global_rep=config['global_rep'],
            device=device,
        )

        # For ensemble strategy, instantiate classifiers for all modalities before loading state dict
        if config['classifier_strategy'] == 'ensemble' and supported_modalities:
            for mod in supported_modalities:
                if mod not in model.modality_classifiers:
                    model.instantiate_modality_classifier(mod)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"EVANClassifier loaded from checkpoint: {path}")
        return model


if __name__ == '__main__':
    print("why are you calling me?")

# python -u evan_main.py