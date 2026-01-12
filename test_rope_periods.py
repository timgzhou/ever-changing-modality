"""Test if RoPE periods match between EVAN and DINO."""

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

# Create EVAN model
from evan_main import evan_large
evan_model = evan_large(
    tz_fusion_time=3,
    n_storage_tokens=4,
    device='cpu',
    load_pretrained=True  # Load DINO weights
)

# Create DINOv3 model directly
from evan.models.vision_transformer import vit_large
dino_model = vit_large(
    patch_size=16,
    img_size=224,
    n_storage_tokens=4,
    layerscale_init=1e-5,
    device='cpu'
)

# Load DINO weights into dino_model
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

checkpoint_path = hf_hub_download(
    repo_id="facebook/dinov2-large",
    filename="model.safetensors"
)

hf_checkpoint = load_file(checkpoint_path)

# Check if rope periods are in the checkpoint
rope_keys = [k for k in hf_checkpoint.keys() if 'rope' in k.lower() or 'pos' in k.lower()]
print(f"RoPE-related keys in HuggingFace checkpoint: {rope_keys}")
print()

# Compare EVAN and DINO rope periods
print("="*80)
print("EVAN RoPE periods:")
print(f"  Shape: {evan_model.rope_embed.periods.shape}")
print(f"  Values: {evan_model.rope_embed.periods}")
print(f"  Base: {evan_model.rope_embed.base}")
print()

print("="*80)
print("DINO RoPE periods:")
print(f"  Shape: {dino_model.rope_embed.periods.shape}")
print(f"  Values: {dino_model.rope_embed.periods}")
print(f"  Base: {dino_model.rope_embed.base}")
print()

print("="*80)
print("Periods match?", torch.allclose(evan_model.rope_embed.periods, dino_model.rope_embed.periods))
print("Difference:", (evan_model.rope_embed.periods - dino_model.rope_embed.periods).abs().max().item())
