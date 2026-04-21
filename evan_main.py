import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any, Dict, List, Literal, Optional, Tuple
import logging
from evan.layers import CrossAttentionBlock, LoRALayer, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from functools import partial

logger = logging.getLogger("evan")

ffn_layer_dict = {"mlp": Mlp}
norm_layer_dict = {"layernorm": partial(nn.LayerNorm, eps=1e-6),}
dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class CrossSequenceProjector(nn.Module):
    """
    Cross-attention based sequence projector for hallucinating missing modality features.

    Architecture:
    - (num_layers - 1) SelfAttentionBlock layers on source tokens (with RoPE)
    - 1 CrossAttentionBlock: tgt=queries with RoPE on patch portion, memory=source tokens with RoPE

    Query shape per target modality: [1, 2, embed_dim]
    - 1 CLS query + 1 patch prototype query (no storage queries)
    The patch prototype is broadcast to n_patches and given per-position RoPE before cross-attn.
    Output: [B, 1 + n_patches, embed_dim] (fixed shape regardless of visible input length)
    """

    def __init__(self, embed_dim, n_storage_tokens, img_size, patch_size,
                 num_heads=8, ffn_factor=4, num_layers=2, device=None):
        super().__init__()
        self.n_storage_tokens = n_storage_tokens
        self.n_patches = (img_size // patch_size) ** 2

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        if num_layers > 1:
            self.self_attn_layers = nn.ModuleList([
                SelfAttentionBlock(
                    dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_factor,
                    qkv_bias=True, proj_bias=True, ffn_bias=True,
                    drop_path=0.0, norm_layer=norm_layer, act_layer=nn.GELU,
                    ffn_layer=Mlp, init_values=None, mask_k_bias=False, device=device,
                )
                for _ in range(num_layers - 1)
            ])
        else:
            self.self_attn_layers = None

        self.cross_attn_block = CrossAttentionBlock(
            dim=embed_dim, num_heads=num_heads, ffn_ratio=ffn_factor,
            qkv_bias=True, proj_bias=True, ffn_bias=True,
            norm_layer=norm_layer, act_layer=nn.GELU, ffn_layer=Mlp, device=device,
        )

    def forward(self, x, queries, rope_embed, src_patch_mask=None):
        """
        Args:
            x: [B, 1+n_storage+n_patches, embed_dim] — full source token sequence
            queries: [1, 2, embed_dim] — target modality's learned queries (CLS + patch prototype)
            rope_embed: EVAN's RopePositionEmbedding module
            src_patch_mask: [B, n_patches] bool tensor, True=masked (will be blocked in cross-attn).
                            If None, all source patches are attended to.
        Returns:
            [B, 1 + n_patches, embed_dim] — no storage tokens in output
        """
        B = x.shape[0]
        src_n_prefix = self.n_storage_tokens + 1  # CLS + storage tokens in source
        H = W = int(self.n_patches ** 0.5)

        # Compute RoPE (full grid for both source and query patches)
        if rope_embed is not None:
            sin, cos = rope_embed(H=H, W=W)  # [n_patches, D]
            rope_memory = (sin, cos)
        else:
            rope_memory = None

        # Self-attention on source tokens with RoPE
        if self.self_attn_layers is not None:
            for blk in self.self_attn_layers:
                x = blk(x, rope_memory)

        # Prepare queries: 1 CLS query + patch prototype broadcast to n_patches
        q_cls = queries[:, :1, :].expand(B, -1, -1)                      # [B, 1, D]
        q_patch_proto = queries[:, 1:, :].expand(B, -1, -1)              # [B, 1, D]
        q_patches = q_patch_proto.expand(-1, self.n_patches, -1).clone() # [B, n_patches, D]
        q = torch.cat([q_cls, q_patches], dim=1)                         # [B, 1+n_patches, D]

        rope_tgt = (sin, cos) if rope_embed is not None else None  # full grid for Q patches

        # Build attention mask to block masked source patches from cross-attention
        # [B, 1, Nq, Nkv]: prefix memory tokens (CLS/storage) are always visible
        attn_mask = None
        if src_patch_mask is not None:
            Nq = q.shape[1]
            Nkv = x.shape[1]
            mask = torch.zeros(B, 1, Nq, Nkv, device=x.device, dtype=x.dtype)
            patch_block = src_patch_mask[:, None, None, :].expand(B, 1, Nq, -1).to(x.dtype)
            mask[:, :, :, src_n_prefix:] = patch_block.masked_fill(patch_block.bool(), float('-inf'))
            attn_mask = mask

        # Cross-attention: tgt=queries, memory=source tokens
        out = self.cross_attn_block(
            tgt=q, memory=x,
            rope_tgt=rope_tgt, rope_memory=rope_memory,
            prefix_len_tgt=1, prefix_len_memory=src_n_prefix,
            attn_mask=attn_mask,
        )
        return out  # [B, 1+n_patches, D]


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
        device: Any | None = None,
        tz_modality_specific_layer_augmenter: Literal["lora", "fft"] = "lora",
        tz_modality_fusion_layer_augmenter: Literal["lora","none"] = "none",
        tz_fusion_time: int = 3,
        tz_lora_rank: int=0,
        starting_modality: 'str | list[str]' = 'rgb',
        starting_n_chans: 'int | list[int]' = 3,
        intermediate_projector_type: Literal["self", "cross"] = "self",
        intermediate_projector_num_layers: int = 2,
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
        # Normalize starting_modality/starting_n_chans to lists; first element is primary.
        if isinstance(starting_modality, str):
            starting_modality = [starting_modality]
            starting_n_chans = [starting_n_chans]
        self.starting_modality = starting_modality[0]
        self.n_storage_tokens = n_storage_tokens
        self.intermediate_projector_type = intermediate_projector_type
        self.intermediate_projector_num_layers = intermediate_projector_num_layers
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

        self.head = nn.Identity()


        # ============ EVAN-Specific Components ============
        # Initialize multimodal components
        self.supported_modalities=[starting_modality[0]]
        self.supported_modalities_in_chans=[starting_n_chans[0]]
        self.patch_embedders = nn.ModuleDict()
        self.cls_tokens = nn.ParameterDict()
        self.storage_tokens = nn.ParameterDict()
        self.modality_specific_layer_adaptors = nn.ModuleDict() # shortened as "msla" as followed
        self.modality_encodings = nn.ParameterDict()
        if self.tz_modality_fusion_layer_augmenter!="none":
            self.modality_fusion_lora_adaptors = nn.ModuleDict()
        self.intermediate_projectors = nn.ModuleDict()
        if self.intermediate_projector_type == "cross":
            self.projector_queries = nn.ParameterDict()

        # Initialize modality-specific components for primary modality
        self.add_new_patch_embedders(starting_modality[0], starting_n_chans[0])
        self.add_new_cls_token(starting_modality[0])
        if self.n_storage_tokens > 0: self.add_new_storage_tokens(starting_modality[0])
        self.add_new_msla(starting_modality[0])
        self.add_modality_encoding(starting_modality[0])
        if self.tz_modality_fusion_layer_augmenter!="none":
            self.add_new_mfla(starting_modality[0])
        self.add_new_intermediate_projectors(starting_modality[0])
        print(f"Initialized primary modality: '{starting_modality[0]}' ({starting_n_chans[0]} channels)")

        # Register any additional modalities passed at construction time
        for mod, n_ch in zip(starting_modality[1:], starting_n_chans[1:]):
            self.create_modality_components(mod, n_ch)

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
        # In FFT-MSLA mode the first tz_fusion_time blocks are fully replaced by
        # per-modality adaptors and are never called in the forward pass.
        if self.tz_modality_specific_layer_augmenter == "fft":
            blocks_list = self.create_transformer_blocks(self.n_blocks - self.tz_fusion_time)
        else:
            blocks_list = self.create_transformer_blocks(self.n_blocks)
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
    def add_new_cls_token(self,modality_name):
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
            self.modality_specific_layer_adaptors[modality] = self.create_transformer_blocks(self.tz_fusion_time)
            # backbone blocks 0..tz_fusion_time-1 are not allocated (initialize_blocks skips them),
            # so weight init from backbone is deferred to the pretrained weight loaders.
        else:
            raise RuntimeError(f"unrecognized {self.tz_modality_specific_layer_augmenter=}")
    def add_modality_encoding(self,modality_name):
        self.modality_encodings[modality_name]=nn.Parameter(
            torch.zeros(1, 1, self.embed_dim, device=self.device)
        )
    def add_new_mfla(self, modality):
        """
        Add modality fusion layer adaptors.

        Args:
            modality: Name of the modality
        """
        num_fusion_blocks = self.n_blocks - self.tz_fusion_time
        if self.tz_modality_fusion_layer_augmenter == "lora":
            self.modality_fusion_lora_adaptors[modality] = self.create_lora_list(num_fusion_blocks)
    def _make_intermediate_projector(self, tgt_mod: str) -> nn.Module:
        if self.intermediate_projector_type == "self":
            from train_utils import SequenceProjector
            return SequenceProjector(
                embed_dim=self.embed_dim, num_heads=8, ffn_factor=4,
                num_layers=self.intermediate_projector_num_layers,
            ).to(self.device)
        elif self.intermediate_projector_type == "cross":
            return CrossSequenceProjector(
                embed_dim=self.embed_dim, n_storage_tokens=self.n_storage_tokens,
                img_size=self.img_size, patch_size=self.patch_size,
                num_heads=self.num_heads, ffn_factor=4,
                num_layers=self.intermediate_projector_num_layers,
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown intermediate_projector_type: {self.intermediate_projector_type}")

    def add_new_intermediate_projectors(self, new_modality_key: str):
        """
        Create intermediate projectors for all existing→new and new→existing modality pairs.
        For 'cross' type, also initializes projector_queries for any target modality missing them.
        Called by create_modality_components and __init__ (no-op on first modality since no pairs exist).
        """
        existing_modalities = [m for m in self.supported_modalities if m != new_modality_key]
        for src_mod in existing_modalities:
            key_fwd = f"{src_mod}_to_{new_modality_key}"
            key_rev = f"{new_modality_key}_to_{src_mod}"
            self.intermediate_projectors[key_fwd] = self._make_intermediate_projector(new_modality_key)
            self.intermediate_projectors[key_rev] = self._make_intermediate_projector(src_mod)
        if self.intermediate_projector_type == "cross" and existing_modalities:
            # 2: 1 CLS query + 1 patch prototype query (no storage queries)
            query_shape = (1, 2, self.embed_dim)
            for mod in [new_modality_key] + existing_modalities:
                if mod not in self.projector_queries:
                    param = nn.Parameter(torch.empty(*query_shape, device=self.device))
                    nn.init.trunc_normal_(param, std=0.02)
                    self.projector_queries[mod] = param

    def create_lora_list(self,length):
        return nn.ModuleList([
            LoRALayer(self.embed_dim, rank=self.tz_lora_rank, device=self.device) for _ in range(length)
        ])
    def create_transformer_blocks(self,length):
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

    def load_pretrained_dino(self, model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m", load_weights:bool=True,
                             rgb_in_s2_indices: 'list[int] | None' = None):
        """
        Load pretrained DINO weights from HuggingFace into EVAN.

        Args:
            model_name: HuggingFace model name (default: facebook/dinov3-vitl16-pretrain-lvd1689m)
            rgb_in_s2_indices: When starting_modality is 's2', a list of 3 channel indices
                within the s2 patch embedder that correspond to the RGB bands (B04, B03, B02).
                The DINO RGB patch embedder weights are copied into those positions; all other
                s2 channels remain randomly initialised.
                Example values:
                  EuroSAT s2 (13ch): [3, 2, 1]   # B04, B03, B02 at indices 3,2,1
                  BEN-v2  s2 (12ch): [3, 2, 1]   # same (B10 dropped, positions unchanged)
                  PASTIS  s2 (10ch): [2, 1, 0]   # B04=idx2, B03=idx1, B02=idx0
        """
        if not load_weights:
            print("pretrained=False, not loading weight and returning directly.")
            return
        print(f"\n=== Loading pretrained DINO weights from HuggingFace... ===")
        print(f"Model: {model_name}")

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
            # DINO patch embedder is 3-channel RGB; copy weights only when starting modality
            # is also 3-channel RGB (rgb, s2_rgb, or any *_rgb variant).
            # For s2, the RGB bands are a subset — stash the DINO weights for partial copy later.
            # For other modalities (s1, etc.) the patch embedder must be randomly init'd.
            if 'patch_embed.' in new_key:
                is_rgb_modality = (self.starting_modality == 'rgb' or
                                   self.starting_modality.endswith('_rgb'))
                is_s2_modality = (self.starting_modality == 's2')
                if is_rgb_modality:
                    new_key = new_key.replace('patch_embed.', f'patch_embedders.{self.starting_modality}.')
                elif is_s2_modality and new_key == 'patch_embed.proj.weight':
                    _dino_rgb_patch_weight = value  # [D, 3, kH, kW] — stash for post-load copy
                    continue
                else:
                    continue  # Skip patch embedder weights for non-RGB modalities
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

        # For FFT mode: Copy first tz_fusion_time blocks to modality-specific layers
        if self.tz_modality_specific_layer_augmenter == "fft":
            print(f"\n  FFT mode: Copying first {self.tz_fusion_time} DINO blocks to {self.starting_modality} modality-specific layers...")
            fft_params_copied = 0
            for i in range(self.tz_fusion_time):
                for key, value in list(checkpoint.items()):
                    if key.startswith(f'blocks.{i}.'):
                        msla_key = key.replace(f'blocks.{i}.', f'modality_specific_layer_adaptors.{self.starting_modality}.{i}.')
                        checkpoint[msla_key] = value.clone()
                        fft_params_copied += value.numel()
            print(f"    Copied {fft_params_copied:,} parameters for {self.starting_modality} FFT blocks")
            # Remap remaining blocks.{tz_fusion_time+j} → blocks.{j} so they land in the
            # trimmed self.blocks (which starts at the first fusion block).
            for j in range(self.n_blocks - self.tz_fusion_time):
                src_idx = self.tz_fusion_time + j
                for key in list(checkpoint.keys()):
                    if key.startswith(f'blocks.{src_idx}.'):
                        new_key = key.replace(f'blocks.{src_idx}.', f'blocks.{j}.')
                        checkpoint[new_key] = checkpoint.pop(key)
        
        result = self.load_state_dict(checkpoint, strict=False)

        # For s2 modality: copy DINO RGB patch weights into the rgb band positions of the
        # s2 patch embedder (all other channels remain randomly initialised).
        if (self.starting_modality == 's2' and
                '_dino_rgb_patch_weight' in dir() and
                rgb_in_s2_indices is not None):
            with torch.no_grad():
                proj_weight = self.patch_embedders['s2'].proj.weight  # [D, C_s2, kH, kW]
                for dst_ch, src_ch in enumerate(range(3)):
                    proj_weight[:, rgb_in_s2_indices[dst_ch], :, :] = _dino_rgb_patch_weight[:, src_ch, :, :]
            n_rgb_params = _dino_rgb_patch_weight.numel()
            print(f"  s2 patch embedder: copied DINO RGB weights into channels {rgb_in_s2_indices} "
                  f"({n_rgb_params:,} params); remaining {proj_weight.shape[1] - 3} channels randomly init'd.")
        elif self.starting_modality == 's2' and rgb_in_s2_indices is None:
            print("  s2 patch embedder: randomly init'd (pass rgb_in_s2_indices to copy DINO RGB weights).")

        actually_transferred_params = sum(
            v.numel() for k, v in checkpoint.items()
            if k not in result.unexpected_keys
        )
        print(f"    - Actually transferred: {actually_transferred_params:,}")

        # Filter out expected missing keys (EVAN-specific multi-modality components)
        def is_expected_missing(key):
            # RoPE periods buffer: computed deterministically from hyperparameters, not in HF checkpoint
            if key == 'rope_embed.periods':
                return True
            # Storage tokens: DINO has no register tokens
            if key.startswith('storage_tokens.'):
                return True
            # Components of extra modalities (not the primary): no pretrained weights for these
            primary = self.starting_modality
            for prefix in ('patch_embedders.', 'cls_tokens.', 'modality_specific_layer_adaptors.'):
                if key.startswith(prefix):
                    mod = key[len(prefix):].split('.')[0]
                    if mod != primary:
                        return True
            # s2 patch embedder is kept randomly init'd (or partially overwritten post-load)
            # so it won't appear in the checkpoint — expected missing.
            if primary == 's2' and key.startswith('patch_embedders.s2.'):
                return True
            if self.tz_modality_specific_layer_augmenter == "lora":
                if 'modality_specific_layer_adaptors' in key:
                    return True
            # FFT mode: extra modality MSLA blocks don't get DINO copies
            if self.tz_modality_specific_layer_augmenter == "fft":
                if key.startswith('modality_specific_layer_adaptors.'):
                    mod = key[len('modality_specific_layer_adaptors.'):].split('.')[0]
                    if mod != primary:
                        return True
            if any(pattern in key for pattern in ['modality_encodings', 'modality_fusion_lora_adaptors']):
                return True
            if key.startswith('intermediate_projectors.') or key.startswith('projector_queries.'):
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
        # mask_token was removed from EVAN — expected to be missing from the model side
        expected_missing = {'mask_token'}
        unexpected_missing = [k for k in result.unexpected_keys if k not in expected_missing]
        if len(unexpected_missing) > 0:
            untransferred_params = sum(
                checkpoint[k].numel()
                for k in unexpected_missing
                if k in checkpoint
            )
            print(f"  !!  Unexpected keys (in DINO but not loaded into EVAN): {len(unexpected_missing)}")
            print(f"    First 10 keys: {unexpected_missing[:10]}")
            print(f"    Untransferred parameters: {untransferred_params:,}")

        print("\nWeights loaded successfully!")
        print("=== DINO weight loading complete ===\n")

    def load_pretrained_torchgeo(self, weights, load_weights: bool = True, band_indices: list | None = None):
        """
        Load pretrained weights from a torchgeo ViT (timm-style) into EVAN.

        The torchgeo checkpoint uses timm key naming (blocks.*, patch_embed.proj.*,
        cls_token, mask_token) and has pre-merged qkv. pos_embed is dropped since
        EVAN uses RoPE. storage_tokens are left randomly initialized (torchgeo ViTs
        have no register tokens).

        Args:
            weights: torchgeo Weights enum (e.g. ViTSmall16_Weights.SENTINEL2_ALL_DINO),
                     or a state_dict directly, or None.
            load_weights: If False, skip loading and return immediately.
            band_indices: Optional list of channel indices to select from the teacher's
                          patch_embed.proj.weight (shape [D, 13, 16, 16]). Use this when
                          your dataset has fewer bands than the teacher (e.g. BEN-v2 drops
                          B10 → [0,1,2,3,4,5,6,7,8,9,11,12], PASTIS drops B1/B9/B10 →
                          [1,2,3,4,5,6,7,8,11,12]). Must match starting_n_chans.
        """
        if not load_weights or weights is None:
            print("pretrained=False or weights=None, not loading weights and returning directly.")
            return

        print(f"\n=== Loading pretrained torchgeo weights ===")

        # Accept either a Weights enum or a raw state_dict
        if isinstance(weights, dict):
            tg_checkpoint = weights
        else:
            tg_checkpoint = weights.get_state_dict(progress=True)

        original_params = sum(p.numel() for p in tg_checkpoint.values())
        print(f"  Original checkpoint parameters: {original_params:,}")

        # Keys to silently drop (not applicable to EVAN)
        drop_prefixes = ('pos_embed', 'head.', 'fc_norm.')

        checkpoint = {}
        for key, value in tg_checkpoint.items():
            if any(key.startswith(p) for p in drop_prefixes):
                continue

            new_key = key
            # cls_token → cls_tokens.<starting_modality>  (unsqueeze if needed)
            if new_key == 'cls_token':
                new_key = f'cls_tokens.{self.starting_modality}'
                if value.ndim == 2:
                    value = value.unsqueeze(1)
            # mask_token: shape is (1, 1, D) in timm, EVAN wants (1, D)
            elif new_key == 'mask_token':
                if value.ndim == 3:
                    value = value.squeeze(1)
            # patch_embed.proj.* → patch_embedders.<starting_modality>.proj.*
            elif new_key.startswith('patch_embed.'):
                new_key = new_key.replace('patch_embed.', f'patch_embedders.{self.starting_modality}.')

            checkpoint[new_key] = value

        # Slice patch embedder to match dataset band count
        if band_indices is not None:
            patch_embed_key = f'patch_embedders.{self.starting_modality}.proj.weight'
            if patch_embed_key in checkpoint:
                checkpoint[patch_embed_key] = checkpoint[patch_embed_key][:, band_indices, :, :]
                print(f"  Sliced patch_embed channels: {len(band_indices)}/13 bands (indices {band_indices})")

        transferred_params = sum(p.numel() for p in checkpoint.values())
        dropped_params = original_params - transferred_params
        print(f"  Transferred parameters: {transferred_params:,}")
        print(f"  Dropped (pos_embed, head): {dropped_params:,}")

        # FFT mode: copy block weights into modality-specific / fusion adaptors
        if self.tz_modality_specific_layer_augmenter == "fft":
            print(f"\n  FFT mode: Copying first {self.tz_fusion_time} blocks to {self.starting_modality} modality-specific layers...")
            for i in range(self.tz_fusion_time):
                for key, value in list(checkpoint.items()):
                    if key.startswith(f'blocks.{i}.'):
                        mod_key = key.replace(f'blocks.{i}.', f'modality_specific_layer_adaptors.{self.starting_modality}.{i}.')
                        checkpoint[mod_key] = value.clone()
            # Remap remaining blocks.{tz_fusion_time+j} → blocks.{j} for trimmed self.blocks.
            for j in range(self.n_blocks - self.tz_fusion_time):
                src_idx = self.tz_fusion_time + j
                for key in list(checkpoint.keys()):
                    if key.startswith(f'blocks.{src_idx}.'):
                        new_key = key.replace(f'blocks.{src_idx}.', f'blocks.{j}.')
                        checkpoint[new_key] = checkpoint.pop(key)

        result = self.load_state_dict(checkpoint, strict=False)

        def is_expected_missing(key):
            if key == 'rope_embed.periods':
                return True
            if key.startswith('storage_tokens.'):  # no reg tokens in torchgeo ViT
                return True
            # Components of extra modalities: no pretrained weights for these
            primary = self.starting_modality
            for prefix in ('patch_embedders.', 'cls_tokens.', 'modality_specific_layer_adaptors.'):
                if key.startswith(prefix):
                    mod = key[len(prefix):].split('.')[0]
                    if mod != primary:
                        return True
            if self.tz_modality_specific_layer_augmenter == "lora":
                if 'modality_specific_layer_adaptors' in key:
                    return True
            if self.tz_modality_specific_layer_augmenter == "fft":
                if key.startswith('modality_specific_layer_adaptors.'):
                    mod = key[len('modality_specific_layer_adaptors.'):].split('.')[0]
                    if mod != primary:
                        return True
            if any(p in key for p in ['modality_encodings', 'modality_fusion_lora_adaptors']):
                return True
            if key.startswith('intermediate_projectors.') or key.startswith('projector_queries.'):
                return True
            return False

        unexpected_missing = [k for k in result.missing_keys if not is_expected_missing(k)]
        if unexpected_missing:
            unexpected_missing_params = sum(
                self.state_dict()[k].numel() for k in unexpected_missing if k in self.state_dict()
            )
            print(f"  !!  Unexpected missing keys: {len(unexpected_missing)}")
            print(f"    Keys: {unexpected_missing}")
            print(f"    Parameters: {unexpected_missing_params:,}")

        expected_missing = {'mask_token'}
        unexpected_missing = [k for k in result.unexpected_keys if k not in expected_missing]
        if unexpected_missing:
            untransferred_params = sum(
                checkpoint[k].numel() for k in unexpected_missing if k in checkpoint
            )
            print(f"  !!  Unexpected keys (in checkpoint but not in EVAN): {len(unexpected_missing)}")
            print(f"    First 10 keys: {unexpected_missing[:10]}")
            print(f"    Untransferred parameters: {untransferred_params:,}")

        print("\nWeights loaded successfully!")
        print("=== torchgeo weight loading complete ===\n")

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
        if self.tz_modality_fusion_layer_augmenter!="none":
            self.add_new_mfla(modality_key)
        self.add_new_cls_token(modality_key)
        if self.n_storage_tokens > 0:
            self.add_new_storage_tokens(modality_key)
        self.add_modality_encoding(modality_key)
        self.supported_modalities.append(modality_key)
        self.supported_modalities_in_chans.append(in_chans)
        self.add_new_intermediate_projectors(modality_key)

        params_after = sum(p.numel() for p in self.parameters())
        new_params = params_after - params_before

        num_fusion_blocks = self.n_blocks - self.tz_fusion_time
        embedder_params = sum(p.numel() for p in self.patch_embedders[modality_key].parameters())
        msla_params = sum(p.numel() for p in self.modality_specific_layer_adaptors[modality_key].parameters())
        if self.tz_modality_fusion_layer_augmenter!="none": 
            mfla_params = sum(p.numel() for p in self.modality_fusion_lora_adaptors[modality_key].parameters())
        else: mfla_params=0
        cls_token_params = self.cls_tokens[modality_key].numel()
        storage_token_params = self.storage_tokens[modality_key].numel() if self.n_storage_tokens > 0 else 0
        encoding_params = self.modality_encodings[modality_key].numel()

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

    def prepare_tokens_with_masks(self, x: Tensor, modality_key: str) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Prepare tokens with modality-specific embedder.

        Args:
            x: Input tensor
            modality_key: Key identifying which modality's embedder and CLS/storage tokens to use

        Returns:
            Tuple of (tokens with CLS and storage prepended, (H, W) spatial dimensions)
        """
        embedder = self.patch_embedders[modality_key]
        x = embedder(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        cls_token = self.cls_tokens[modality_key]
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens[modality_key]
        else: # create an empty tensor with dim for torch.cat to work, no-op
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

    def forward_features(self, x: Dict[str, Tensor] | List[Tensor]) -> Dict[str, Tensor]:
        """
        Forward features with multi-modality support.

        Args:
            x: Dict of modality tensors {modality: [B, C, H, W]}

        Returns:
            Dictionary with normalized features
        """
        if isinstance(x, torch.Tensor) or not isinstance(x, dict) or len(x) == 0:
            raise ValueError("Input must be a non-empty dict of modalities.")

        batch_sizes = [v.shape[0] for v in x.values()]
        if len(set(batch_sizes)) > 1:
            raise ValueError(f"Batch sizes must match across modalities, got {batch_sizes}")

        embedded_modalities = self.forward_modality_specific_features(x)
        return self.forward_fusion_from_modality_features(embedded_modalities)

    def forward_modality_specific_features(self, x: Dict[str, Tensor]) -> Dict[str, Tensor]:
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
            x_mod, (H, W) = self.prepare_tokens_with_masks(modality_tensor, modality_key)
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
            hallucinated_modalities: Optional set of modality names whose sequences have shape
                                    [B, 1+n_patches, embed_dim] (no storage tokens, cross projector only).
                                    Used for variable-prefix fusion splitting.

        Returns:
            Dictionary with normalized features per modality (same format as forward_features):
            {modality: {'x_norm_clstoken', 'x_norm_patchtokens', 'x_storage_tokens', 'x_prenorm'}}
        """
        # Step 1: Add modality encoding to each modality's features
        for modality_key in embedded_modalities.keys():
            modality_encoding = self.modality_encodings[modality_key]
            embedded_modalities[modality_key] = embedded_modalities[modality_key] + modality_encoding

        # Step 2: Concatenate patches for fusion (allow cross-modal attention)
        modality_info = {}
        all_cls_storage = {}
        all_patches = []
        current_idx = 0

        hallucinated_modalities = hallucinated_modalities or set()
        prefix_sizes = {}  # per-modality CLS+storage count
        for modality_key in sorted(embedded_modalities.keys()):
            x_mod = embedded_modalities[modality_key]

            # Hallucinated modalities have shape [B, 1+n_patches, D] (no storage tokens)
            if modality_key in hallucinated_modalities:
                n_prefix = 1
            else:
                n_prefix = self.n_storage_tokens + 1
            prefix_sizes[modality_key] = n_prefix

            # Extract CLS and storage tokens (keep separate per modality)
            all_cls_storage[modality_key] = x_mod[:, :n_prefix, :]

            # Extract patch tokens
            patches = x_mod[:, n_prefix:, :]
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
        n_modalities = len(all_cls_storage)

        # Compute H, W from image size and patch size
        H = W = self.img_size // self.patch_size

        # Combine for fusion processing
        x_fused = torch.cat([x_cls_storage_concat, x_patches_concat], dim=1)

        # Step 2: Process through fusion blocks with cross-modal attention
        if self.rope_embed is not None:
            # multi-modal rope, same spatial location patch of different modalities get rotated by the same amount
            sin, cos = self.rope_embed(H=H, W=W)  # [HW, D]
            sin = sin.repeat(n_modalities, 1)       # [n_modalities*HW, D]
            cos = cos.repeat(n_modalities, 1)
            rope_sincos = (sin, cos)
        else:
            rope_sincos = None
        for fusion_idx in range(len(self.blocks)):
            # Shared block forward on concatenated representation
            x_fused = self.blocks[fusion_idx](x_fused, rope_sincos)


        # Step 3: Split fused representation back into modality-specific outputs
        total_cls_storage = sum(prefix_sizes.values())

        # Split back into modality-specific outputs
        output_dict = {}
        sorted_modalities = sorted(embedded_modalities.keys())
        cls_offset = 0
        for modality_key in sorted_modalities:
            n_prefix = prefix_sizes[modality_key]

            # Extract this modality's CLS/storage tokens
            x_cls_storage_mod = x_fused[:, cls_offset:cls_offset + n_prefix, :]
            cls_offset += n_prefix

            # Extract this modality's patches
            info = modality_info[modality_key]
            patch_start = total_cls_storage + info['start_idx']
            patch_end = total_cls_storage + info['end_idx']
            patches = x_fused[:, patch_start:patch_end, :]

            # Recombine this modality's CLS/storage with its patches
            x_mod = torch.cat([x_cls_storage_mod, patches], dim=1)

            # Apply normalization — for hallucinated modalities n_prefix=1 (CLS only, no storage)
            if self.untie_cls_and_patch_norms:
                x_norm_cls_reg = self.cls_norm(x_mod[:, :n_prefix])
                x_norm_patch = self.norm(x_mod[:, n_prefix:])
            else:
                x_norm = self.norm(x_mod)
                x_norm_cls_reg = x_norm[:, :n_prefix]
                x_norm_patch = x_norm[:, n_prefix:]

            output_dict[modality_key] = {
                "x_norm_clstoken": x_norm_cls_reg[:, 0],
                "x_storage_tokens": x_norm_cls_reg[:, 1:],
                "x_norm_patchtokens": x_norm_patch,
                "x_prenorm": x_mod,
            }

        return output_dict

    def _project_sequence(self, src_seq_norm: Tensor, key: str, tgt_mod: str, src_patch_mask=None) -> Tensor:
        """Call the intermediate projector for (src→tgt), dispatching by projector type."""
        projector = self.intermediate_projectors[key]
        if self.intermediate_projector_type == "cross":
            return projector(
                src_seq_norm,
                queries=self.projector_queries[tgt_mod],
                rope_embed=self.rope_embed,
                src_patch_mask=src_patch_mask,
            )
        else:
            return projector(src_seq_norm)

    def forward_features_with_pseudo_modality(
        self,
        x: Dict[str, Tensor],
        pseudo_modalities: List[str],
    ) -> Dict[str, Dict[str, Tensor]]:
        """
        Forward pass with pseudo-modalities using full sequence projection.

        When a modality is missing at inference time, this method creates pseudo-features
        by projecting the full sequence (CLS + storage + patches) from available modalities.

        Args:
            x: Dictionary of available modality tensors
            pseudo_modalities: List of modality names to hallucinate (e.g., ['vre'])

        Returns:
            Dictionary with normalized features per modality (same format as forward_features)
        """
        # Step 1: Get real modality features
        embedded = self.forward_modality_specific_features(x)
        available_modalities = list(embedded.keys())

        # Step 2: Create pseudo-features for missing modalities using full sequence projection
        for mod in pseudo_modalities:
            projected_seqs = []
            for avail_mod in available_modalities:
                avail_seq = embedded[avail_mod]
                avail_seq_norm = F.layer_norm(avail_seq, [avail_seq.shape[-1]])
                key = f"{avail_mod}_to_{mod}"
                projected_seqs.append(self._project_sequence(avail_seq_norm, key, mod))
            embedded[mod] = torch.stack(projected_seqs).mean(dim=0)

        # Step 3: Forward through fusion (modality encoding added here)
        # cross projector outputs [B, 1+n_patches, D] — fusion needs variable prefix handling
        hal = set(pseudo_modalities) if self.intermediate_projector_type == "cross" else None
        return self.forward_fusion_from_modality_features(embedded, hallucinated_modalities=hal)

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
        intermediate_projectors: bool = False,
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
            if intermediate_projectors:
                for param in self.intermediate_projectors.parameters():
                    param.requires_grad = True
                if hasattr(self, 'projector_queries'):
                    for param in self.projector_queries.values():
                        param.requires_grad = True
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
            if modality_encoders and modality in self.modality_encodings:
                self.modality_encodings[modality].requires_grad = True
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
            'tz_modality_specific_layer_augmenter': self.tz_modality_specific_layer_augmenter,
            'tz_modality_fusion_layer_augmenter': self.tz_modality_fusion_layer_augmenter,
            'tz_fusion_time': self.tz_fusion_time,
            'tz_lora_rank': self.tz_lora_rank,
            'starting_modality': self.starting_modality,
            'starting_n_chans': self.supported_modalities_in_chans[0],
            'supported_modalities': self.supported_modalities.copy(),
            'supported_modalities_in_chans': self.supported_modalities_in_chans.copy(),
            'intermediate_projector_type': self.intermediate_projector_type,
            'intermediate_projector_num_layers': self.intermediate_projector_num_layers,
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

def evan_small(pretrained: str = "facebook/dinov3-vits16-pretrain-lvd1689m", load_weights:bool=True,
               rgb_in_s2_indices: 'list[int] | None' = None, **kwargs):
    """
    Create EVAN-Small model (384 dim, 12 blocks, 6 heads) with pretrained DINO weights.

    Args:
        pretrained: HuggingFace model name for pretrained weights (default: facebook/dinov3-vits16-pretrain-lvd1689m)
        rgb_in_s2_indices: See EVAN.load_pretrained_dino for details.
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
    model.load_pretrained_dino(model_name=pretrained, load_weights=load_weights, rgb_in_s2_indices=rgb_in_s2_indices)
    return model


def evan_base(pretrained: str = "facebook/dinov3-vitb16-pretrain-lvd1689m", load_weights:bool=True,
              rgb_in_s2_indices: 'list[int] | None' = None, **kwargs):
    """
    Create EVAN-Base model (768 dim, 12 blocks, 12 heads) with pretrained DINO weights.

    Args:
        pretrained: HuggingFace model name for pretrained weights (default: facebook/dinov3-vitb16-pretrain-lvd1689m)
        rgb_in_s2_indices: See EVAN.load_pretrained_dino for details.
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
    model.load_pretrained_dino(model_name=pretrained, load_weights=load_weights, rgb_in_s2_indices=rgb_in_s2_indices)
    return model


def evan_large(pretrained: str = "facebook/dinov3-vitl16-pretrain-sat493m", load_weights:bool=True,
               rgb_in_s2_indices: 'list[int] | None' = None, **kwargs):
    """
    Create EVAN-Large model (1024 dim, 24 blocks, 16 heads) with pretrained DINO weights.

    Args:
        pretrained: HuggingFace model name for pretrained weights (default: facebook/dinov3-vitl16-pretrain-sat493m)
        rgb_in_s2_indices: See EVAN.load_pretrained_dino for details.
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
    model.load_pretrained_dino(model_name=pretrained, load_weights=load_weights, rgb_in_s2_indices=rgb_in_s2_indices)
    return model


def evan_small_s2(weights=None, load_weights: bool = True, band_indices: list | None = None, **kwargs):
    """
    Create EVAN-Small initialised from a torchgeo Sentinel-2 ViT-Small checkpoint.

    Uses RoPE (EVAN default) rather than the teacher's absolute pos_embed — the
    positional structure is recovered through distillation. No LayerScale (the
    torchgeo DINO v1/v2 checkpoint has none).

    Example:
        from torchgeo.models import ViTSmall16_Weights
        # Full 13-band
        evan = evan_small_s2(weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO)
        # BEN-v2 (12 bands, drop B10)
        evan = evan_small_s2(weights=..., band_indices=BENV2_BAND_INDICES, starting_n_chans=12)
        # PASTIS (10 bands, drop B1/B9/B10)
        evan = evan_small_s2(weights=..., band_indices=PASTIS_BAND_INDICES, starting_n_chans=10)

    Args:
        weights: torchgeo Weights enum or raw state_dict. None = random init.
        load_weights: If False, skip weight loading.
        band_indices: Channel indices into the teacher's 13-band patch embedder.
            Must match starting_n_chans if both are provided.
            BENV2_BAND_INDICES  = [0,1,2,3,4,5,6,7,8,9,11,12]   # drop B10
            PASTIS_BAND_INDICES = [1,2,3,4,5,6,7,8,11,12]        # drop B1,B9,B10
        **kwargs: Forwarded to EVAN (device, tz_fusion_time, tz_lora_rank,
                  starting_n_chans, etc.)
    """
    starting_n_chans = kwargs.pop('starting_n_chans', len(band_indices) if band_indices is not None else 13)
    kwargs.setdefault('starting_modality', 's2')
    model = EVAN(
        patch_size=16,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        layerscale_init=None,   # torchgeo ViT has no LayerScale
        starting_n_chans=starting_n_chans,
        **kwargs,
    )
    model.load_pretrained_torchgeo(weights, load_weights=load_weights, band_indices=band_indices)
    return model


# Convenience band index constants for torchgeo 13-band S2 checkpoints
# torchgeo order: B1,B2,B3,B4,B5,B6,B7,B8,B8a,B9,B10,B11,B12
BENV2_BAND_INDICES  = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12]   # drop B10 (index 10)
PASTIS_BAND_INDICES = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]          # drop B1 (0), B9 (9), B10 (10)


class EvanPredictor(nn.Module):
    """
    Base class shared by EVANClassifier and EvanSegmenter.

    Provides common __init__ storage, freeze_all, set_requires_grad,
    save_checkpoint, and the _reconstruct_evan static helper used by
    subclass from_checkpoint implementations.
    """

    def __init__(self, evan_model, num_classes, strategy, device):
        super().__init__()
        self.evan = evan_model
        self.num_classes = num_classes
        self.strategy = strategy   # 'mean' or 'ensemble'
        self.device = device
        self.head = None             # shared mean head (set by subclass)
        self.modality_heads = None   # shared ensemble heads (set by subclass)

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

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
        head: bool = False,
        intermediate_projectors: bool = False,
    ):
        """
        Set requires_grad for specific components of the model (EVAN + head).

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
            head: Unfreeze prediction head(s) (shared mean head or per-modality ensemble heads)
        """
        self.evan.set_requires_grad(
            modality,
            patch_embedders=patch_embedders,
            clsreg=clsreg,
            msla=msla,
            modality_encoders=modality_encoders,
            mfla=mfla,
            blocks=blocks,
            norm=norm,
            intermediate_projectors=intermediate_projectors,
        )
        if head:
            if self.head is not None:
                for param in self.head.parameters():
                    param.requires_grad = True
            if self.modality_heads is not None:
                if modality == 'all':
                    for h in self.modality_heads.values():
                        for param in h.parameters():
                            param.requires_grad = True
                elif modality in self.modality_heads:
                    for param in self.modality_heads[modality].parameters():
                        param.requires_grad = True
                else:
                    raise RuntimeError(f"'{modality}' not in modality_heads")

    def save_checkpoint(self, path: str):
        torch.save({'model_state_dict': self.state_dict(), 'config': self.get_config()}, path)
        print(f"{self.__class__.__name__} checkpoint saved to: {path}")

    @staticmethod
    def _reconstruct_evan(evan_config, device):
        """Reconstruct an EVAN model from a (possibly mutated) evan_config dict."""
        supported_modalities = evan_config.pop('supported_modalities', None)
        supported_modalities_in_chans = evan_config.pop('supported_modalities_in_chans', None)
        starting_modality = evan_config.get('starting_modality', 'rgb')
        if 'starting_n_chans' not in evan_config and supported_modalities_in_chans:
            evan_config['starting_n_chans'] = supported_modalities_in_chans[0]
        evan = EVAN(**evan_config, device=device)
        if supported_modalities and supported_modalities_in_chans:
            for mod, n_chans in zip(supported_modalities, supported_modalities_in_chans):
                if mod != starting_modality and mod not in evan.patch_embedders:
                    evan.create_modality_components(mod, n_chans)
        return evan, supported_modalities


class EVANClassifier(EvanPredictor):
    """Classifier head on top of EVAN for EuroSAT."""

    def __init__(self, evan_model, num_classes=10, classifier_strategy='mean', factor=4, global_rep="clstoken", device = "cuda"):
        super().__init__(evan_model, num_classes, classifier_strategy, device)
        self.factor = factor
        self.global_rep = global_rep
        embed_dim = self.evan.embed_dim
        hidden_dim = embed_dim * factor
        self.hidden_dim = hidden_dim

        if classifier_strategy == 'mean':
            # Average CLS tokens from all modalities, then classify
            self.head = nn.Sequential(
                nn.Linear(embed_dim, num_classes)
            )
            self.modality_heads = None
        elif classifier_strategy == 'ensemble':
            # Per-modality classifiers that get ensembled
            self.head = None
            self.modality_heads = nn.ModuleDict()
            self.instantiate_modality_head(evan_model.starting_modality)
        else:
            raise ValueError(f"Unknown fusion strategy: {classifier_strategy}")

    # --------------- public aliases for backward compatibility ---------------
    @property
    def classifier_strategy(self):
        return self.strategy

    @classifier_strategy.setter
    def classifier_strategy(self, value):
        self.strategy = value

    @property
    def classifier(self):
        return self.head

    @classifier.setter
    def classifier(self, value):
        self.head = value

    @property
    def modality_classifiers(self):
        return self.modality_heads

    @modality_classifiers.setter
    def modality_classifiers(self, value):
        self.modality_heads = value

    def instantiate_modality_head(self, modality_key: str):
        """
        Create a new classifier head for a specific modality.

        Args:
            modality_key: Name of the modality (e.g., 'rgb', 'vre', 'nir', 'swir')
        """
        embed_dim = self.evan.embed_dim

        classifier = nn.Sequential(
            nn.BatchNorm1d(embed_dim, affine=False),
            nn.Linear(embed_dim, self.num_classes)
        )

        classifier = classifier.to(self.device)
        self.modality_heads[modality_key] = classifier
        print(f"  Created new classifier for modality: {modality_key}")

    # Keep old name as alias so external callers (shot.py) continue to work
    def instantiate_modality_classifier(self, modality_key: str):
        return self.instantiate_modality_head(modality_key)

    def classify_from_features(self, features_dict):
        """
        Classify from pre-computed features dict.

        Args:
            features_dict: Output from evan.forward_features or forward_features_with_pseudo_modality

        Returns:
            logits: [B, num_classes]
        """
        if self.strategy == 'mean':
            cls_tokens = []
            for modality in sorted(features_dict.keys()):
                if self.global_rep == "clstoken":
                    cls_tokens.append(features_dict[modality]['x_norm_clstoken'])
                elif self.global_rep == "mean_patch":
                    cls_tokens.append(features_dict[modality]['x_norm_patchtokens'].mean(1))
            fused = torch.stack(cls_tokens).mean(dim=0)
            return self.head(fused)

        elif self.strategy == 'ensemble':
            all_logits = []
            for modality in sorted(features_dict.keys()):
                if modality not in self.modality_heads:
                    raise RuntimeError(f"{modality} doesn't have its own classifier.")
                if self.global_rep == "clstoken":
                    cls_token = features_dict[modality]['x_norm_clstoken']
                elif self.global_rep == "mean_patch":
                    cls_token = features_dict[modality]['x_norm_patchtokens'].mean(1)
                else:
                    raise ValueError(f"unrecognized global_rep arg, choices are clstoken or mean_patch, received {self.global_rep}")
                modality_logits = self.modality_heads[modality](cls_token)
                all_logits.append(modality_logits)
            return torch.stack(all_logits).mean(dim=0)

        else:
            raise ValueError(f"Unknown classifier strategy: {self.strategy}")

    def forward(self, x, pseudo_modalities=None):
        """
        Forward pass supporting both single tensor and dict inputs.
        Args:
            x: Either a tensor [B, C, H, W] or dict {modality: tensor}
            pseudo_modalities: Optional list of modalities to hallucinate using sequence projection
            intermediate_projectors: Deprecated, ignored. Use evan.intermediate_projectors.
        Returns:
            logits: [B, num_classes]
        """
        if pseudo_modalities is not None:
            features_dict = self.evan.forward_features_with_pseudo_modality(
                x, pseudo_modalities
            )
        else:
            features_dict = self.evan.forward_features(x)
        return self.classify_from_features(features_dict)

    def switch_strategy(self, target_strategy, key=None):
        if self.strategy == target_strategy:
            print(f"Already using {target_strategy} head")
        elif target_strategy == "mean":
            self.ensemble_to_mean(key)
        elif target_strategy == "ensemble":
            self.mean_to_ensemble()
            for mod in self.evan.patch_embedders.keys():
                if mod not in self.modality_heads:
                    self.instantiate_modality_head(mod)
        return

    def mean_to_ensemble(self):
        """Convert from 'mean' to 'ensemble' strategy, the existing classifier becomes new_key classifier"""
        if self.strategy == 'ensemble':
            print("!!!!  mean_to_ensemble was called on classifier but it already is ensemble. No changes made.")
            return()
        assert self.strategy == 'mean'
        self.modality_heads = nn.ModuleDict()
        for mod in self.evan.supported_modalities:
            self.modality_heads[mod] = copy.deepcopy(self.head)
        self.head = None
        self.strategy = 'ensemble'
        print("!! Evan Classifier has switched strategy from mean to ensemble")

    def ensemble_to_mean(self, key_to_keep: str = 'rgb'):
        """Convert from 'ensemble' to 'mean' strategy, keeping only key_to_keep classifier"""
        if self.strategy == 'mean':
            print("!!!!  mean_to_ensemble was called on classifier but it already is mean. No changes made.")
            return()
        assert self.strategy == 'ensemble'
        self.head = self.modality_heads[key_to_keep]
        self.modality_heads = None
        self.strategy = 'mean'
        print("!! Evan Classifier has switched strategy from ensemble to mean")

    def get_config(self) -> Dict[str, Any]:
        """Return config dict needed to reconstruct this model architecture."""
        return {
            'evan_config': self.evan.get_config(),
            'num_classes': self.num_classes,
            'classifier_strategy': self.strategy,  # keep old key name for checkpoint compat
            'factor': self.factor,
            'global_rep': self.global_rep,
        }

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

        evan, supported_modalities = EvanPredictor._reconstruct_evan(evan_config, device)

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
                if mod not in model.modality_heads:
                    model.instantiate_modality_head(mod)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if device is not None:
            model.to(device)
        print(f"EVANClassifier loaded from checkpoint: {path}")
        return model


class EvanSegmenter(EvanPredictor):
    """
    Segmentation head on top of EVAN.

    Mirrors EVANClassifier's interface so it can be used as a drop-in replacement
    in shot_ete.py, shot.py, and train_stage0.py when task_type == 'segmentation'.

    decoder_type='linear' (default): 1×1 Conv2d on patch tokens + bilinear upsample.
    """

    def __init__(
        self,
        evan_model,
        num_classes: int,
        decoder_strategy: str = 'mean',
        decoder_type: str = 'linear',
        device: str = 'cuda',
    ):
        """
        Args:
            evan_model: Pretrained EVAN backbone.
            num_classes: Number of segmentation classes (e.g. 19 for PASTIS).
            decoder_strategy: 'mean' (shared decoder on averaged patch tokens) or
                              'ensemble' (per-modality decoders, average logits).
            decoder_type: 'linear' (1×1 Conv2d + bilinear upsample)
            device: Target device.
        """
        super().__init__(evan_model, num_classes, decoder_strategy, device)
        self.decoder_type = decoder_type

        embed_dim = evan_model.embed_dim
        self._patch_hw = evan_model.img_size // evan_model.patch_size
        self._img_size = evan_model.img_size

        if decoder_strategy == 'mean':
            self.head = self._make_decoder(embed_dim)
            self.modality_heads = None
        elif decoder_strategy == 'ensemble':
            self.head = None
            self.modality_heads = nn.ModuleDict()
            self.instantiate_modality_head(evan_model.starting_modality)
        else:
            raise ValueError(f"Unknown decoder_strategy: {decoder_strategy!r}")

    def switch_strategy(self, target_strategy, key=None):
        if self.strategy == target_strategy:
            print(f"Already using {target_strategy} decoder strategy")
        elif target_strategy == 'ensemble':
            self.mean_to_ensemble()
            for mod in self.evan.patch_embedders.keys():
                self.instantiate_modality_head(mod)
        elif target_strategy == 'mean':
            self.ensemble_to_mean(key)

    def mean_to_ensemble(self):
        if self.strategy == 'ensemble':
            print("!!!!  mean_to_ensemble called on segmenter but already ensemble. No changes made.")
            return
        assert self.strategy == 'mean'
        self.modality_heads = nn.ModuleDict()
        for mod in self.evan.supported_modalities:
            self.modality_heads[mod] = copy.deepcopy(self.head)
        self.head = None
        self.strategy = 'ensemble'
        print("!! EvanSegmenter switched strategy from mean to ensemble")

    def ensemble_to_mean(self, key_to_keep: str = None):
        if self.strategy == 'mean':
            print("!!!!  ensemble_to_mean called on segmenter but already mean. No changes made.")
            return
        assert self.strategy == 'ensemble'
        key_to_keep = key_to_keep or self.evan.starting_modality
        self.head = self.modality_heads[key_to_keep]
        self.modality_heads = None
        self.strategy = 'mean'
        print("!! EvanSegmenter switched strategy from ensemble to mean")

    # --------------- public aliases for backward compatibility ---------------
    @property
    def decoder_strategy(self):
        return self.strategy

    @decoder_strategy.setter
    def decoder_strategy(self, value):
        self.strategy = value

    @property
    def decoder(self):
        return self.head

    @decoder.setter
    def decoder(self, value):
        self.head = value

    @property
    def modality_decoders(self):
        return self.modality_heads

    @modality_decoders.setter
    def modality_decoders(self, value):
        self.modality_heads = value

    def _make_decoder(self, embed_dim: int) -> nn.Module:
        """Return a decoder head according to decoder_type."""
        if self.decoder_type == 'unet':
            raise NotImplementedError("UNET decoder not supported for ViT")
        else:  # 'linear'
            return nn.Sequential(
                nn.BatchNorm2d(embed_dim, affine=False),
                nn.Conv2d(embed_dim, self.num_classes, kernel_size=1))

    def _apply_decoder(self, dec: nn.Module, patch_tokens: torch.Tensor) -> torch.Tensor:
        """Run patch_tokens [B, N, D] through dec and return [B, C, H, W]."""
        if self.decoder_type == 'unet':
            raise NotImplementedError("UNET decoder not supported for ViT")
        # linear: reshape to spatial map, apply 1×1 conv, bilinear upsample
        B, N, D = patch_tokens.shape
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, D, self._patch_hw, self._patch_hw)
        logits_small = dec(feat_map)
        return F.interpolate(logits_small, size=(self._img_size, self._img_size),
                             mode='bilinear', align_corners=False)

    def instantiate_modality_head(self, modality_key: str):
        """Create and register a decoder head for a new modality (ensemble strategy)."""
        embed_dim = self.evan.embed_dim
        self.modality_heads[modality_key] = self._make_decoder(embed_dim).to(self.device)

    # Keep old name as alias so external callers (shot.py) continue to work
    def instantiate_modality_decoder(self, modality_key: str):
        return self.instantiate_modality_head(modality_key)

    def segment_from_features(self, features_dict: dict) -> torch.Tensor:
        """
        Produce segmentation logits from pre-computed EVAN features.

        Args:
            features_dict: Output from evan.forward_features() or
                           evan.forward_fusion_from_modality_features().
                           Each value has 'x_norm_patchtokens': [B, N, D].
        Returns:
            logits: [B, num_classes, H, W]
        """
        if self.strategy == 'mean':
            patch_maps = [
                features_dict[mod]['x_norm_patchtokens']
                for mod in sorted(features_dict.keys())
            ]
            avg_patches = torch.stack(patch_maps).mean(dim=0)  # [B, N, D]
            return self._apply_decoder(self.head, avg_patches)

        elif self.strategy == 'ensemble':
            all_logits = []
            for mod in sorted(features_dict.keys()):
                if mod not in self.modality_heads:
                    raise RuntimeError(f"No decoder for modality '{mod}'.")
                patch_tokens = features_dict[mod]['x_norm_patchtokens']
                all_logits.append(self._apply_decoder(self.modality_heads[mod], patch_tokens))
            return torch.stack(all_logits).mean(dim=0)

    def forward(self, x: dict) -> torch.Tensor:
        """
        Args:
            x: Dict {modality: [B, C, H, W]}.
        Returns:
            logits: [B, num_classes, H, W]
        """
        features_dict = self.evan.forward_features(x)
        return self.segment_from_features(features_dict)

    def get_config(self) -> dict:
        return {
            'evan_config': self.evan.get_config(),
            'num_classes': self.num_classes,
            'decoder_strategy': self.strategy,  # keep old key name for checkpoint compat
        }

    @classmethod
    def from_checkpoint(cls, path: str, device=None) -> "EvanSegmenter":
        checkpoint = torch.load(path, map_location=device or 'cpu')
        config = checkpoint['config']
        evan_config = config['evan_config']

        evan, supported_modalities = EvanPredictor._reconstruct_evan(evan_config, device)

        model = cls(
            evan_model=evan,
            num_classes=config['num_classes'],
            decoder_strategy=config['decoder_strategy'],
            device=device,
        )

        if config['decoder_strategy'] == 'ensemble' and supported_modalities:
            for mod in supported_modalities:
                if mod not in model.modality_heads:
                    model.instantiate_modality_head(mod)

        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if device is not None:
            model.to(device)
        print(f"EvanSegmenter loaded from checkpoint: {path}")
        return model


if __name__ == '__main__':
    print("why are you calling me?")

# python -u evan_main.py