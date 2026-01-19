"""Training utilities for EVAN on EuroSAT."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from einops import rearrange
from eurosat_data_utils import (
    create_multimodal_batch,
    normalize_bands,
    get_band_indices,
    BAND_MINS,
    BAND_MAXS,
)
import os
import wandb


class FullSequenceMAEDecoder(nn.Module):
    """
    MAE decoder that takes the full sequence including mask token positions.

    Unlike SimpleMAEDecoder which expects only unmasked tokens + ids_restore,
    this decoder takes the full sequence where masked positions already contain
    the mask token. Simpler interface for fusion MAE training.
    """

    def __init__(self, embed_dim, num_channels, patch_size, decoder_depth=1, decoder_heads=8, ffn_factor=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Transformer decoder
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        decoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=decoder_heads,
            dim_feedforward=embed_dim * ffn_factor,
            batch_first=True,
            norm_first=True
        )
        self.decoder = TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Linear projection to reconstruct pixels
        self.pred = nn.Linear(embed_dim, patch_size * patch_size * num_channels)

    def forward(self, x):
        """
        Args:
            x: [B, num_patches, embed_dim] - Full sequence with mask tokens at masked positions

        Returns:
            reconstructed patches: [B, num_patches, patch_size^2 * channels]
        """
        x = self.decoder(x)
        return self.pred(x)


def evaluate(model, dataloader, criterion, device, modality_bands_dict,
             modalities_to_use=('rgb',)):
    """
    Evaluate model on a dataloader.

    Args:
        model: EVAN classifier
        dataloader: DataLoader (RGB only or full bands)
        criterion: Loss function
        device: torch device
        modality_bands_dict: Dict mapping modality names to their band tuples
                            e.g., {'rgb': ('B04', 'B03', 'B02'), 'infrared': ('B08', 'B8A', 'B09', 'B10')}
        modalities_to_use: Tuple of modality names to use for evaluation (e.g., ('rgb',) or ('rgb', 'infrared'))
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)

            # Create multi-modal input with specified modalities
            modal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict, modalities=modalities_to_use
            )
            modal_input = {k: v.to(device) for k, v in modal_input.items()}
            outputs = model(modal_input)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ==================== MAE Helper Functions ====================

def random_mask_patches(x, mask_ratio=0.75):
    """
    Randomly mask patches for MAE training.

    Args:
        x: Patch embeddings [B, num_patches, embed_dim]
        mask_ratio: Fraction of patches to mask (default 0.75)

    Returns:
        x_masked: Unmasked patches only [B, num_unmasked, embed_dim]
        mask: Boolean mask [B, num_patches] where True = masked (hidden from encoder)
        ids_restore: Indices to restore original order [B, num_patches]
    """
    B, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    # Random shuffle
    noise = torch.rand(B, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # Keep first len_keep patches
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

    # Create mask: 0 is keep, 1 is masked
    mask = torch.ones([B, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask.bool(), ids_restore


def mae_reconstruction_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # Mean over pixels in patch

    # Compute loss only on masked patches
    loss = (loss * mask).sum() / mask.sum()
    return loss


def patchify(imgs, patch_size):
    """
    Convert image to patches for MAE target.

    Args:
        imgs: [B, C, H, W]
        patch_size: Size of each patch

    Returns:
        patches: [B, num_patches, num_pixel_per_patch * C]
    """
    patches = rearrange(imgs, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_size, pw=patch_size)
    return patches
 
def train_mae_fusion_phase(
    model, train_loader, test_loader, device, args,
    bands_target: dict,
    mae_modalities: list[str],
    latent_reconstruct_modalities: list[str] = ["rgb"],
    modality_bands_dict: dict = None
):
    """
    Phase 2: Hybrid MAE training for fusion components.

    Uses a hybrid loss combining:
    - MAE reconstruction loss for mae_modalities (reconstruct raw pixels)
    - Latent reconstruction loss for latent_reconstruct_modalities (match frozen teacher features)

    Trains:
    - modality_encoders (all modalities)
    - modality_fusion_lora_adaptors (all modalities)
    - mask_token (shared across modalities)

    Frozen:
    - Backbone blocks, patch_embedders, modality_specific_layer_adaptors, cls/storage tokens
    """
    print("\n" + "="*70)
    print(f"=== PHASE 2: Hybrid MAE Training for Fusion Blocks ===")
    print("="*70)

    # Validate no overlap between modality lists
    overlap = set(mae_modalities) & set(latent_reconstruct_modalities)
    if overlap:
        raise ValueError(
            f"Modality cannot be in both mae_modalities and latent_reconstruct_modalities: {overlap}"
        )

    all_modalities = list(set(mae_modalities + latent_reconstruct_modalities))
    print(f"  MAE modalities (pixel reconstruction): {mae_modalities}")
    print(f"  Latent modalities (feature matching): {latent_reconstruct_modalities}")

    evan = model.evan
    patch_size = evan.patch_size
    num_patches = (evan.img_size // patch_size) ** 2

    # Create MAE decoders for mae_modalities (pixel reconstruction)
    mae_decoders = nn.ModuleDict()
    for mod in mae_modalities:
        num_channels = len(bands_target[mod])
        mae_decoders[mod] = FullSequenceMAEDecoder(
            embed_dim=evan.embed_dim,
            num_channels=num_channels,
            patch_size=patch_size,
            decoder_depth=1,
            ffn_factor=2
        ).to(device)
        print(f"  Initialized FullSequenceMAEDecoder for {mod}, num_channels={num_channels}")

    # Create latent projectors for latent_reconstruct_modalities (feature matching)
    latent_projectors = nn.ModuleDict()
    for mod in latent_reconstruct_modalities:
        latent_projectors[mod] = nn.Sequential(
            nn.Linear(evan.embed_dim, evan.embed_dim),
            nn.GELU(),
            nn.Linear(evan.embed_dim, evan.embed_dim),
        ).to(device)
        print(f"  Initialized Latent Projector for {mod}")

    # Create frozen teacher for latent targets
    teacher_evan = copy.deepcopy(evan)
    for p in teacher_evan.parameters():
        p.requires_grad = False
    teacher_evan.eval()
    print("  Created frozen teacher EVAN for latent targets")

    # Freeze everything, then unfreeze fusion components + mask_token
    model.freeze_all()
    model.set_requires_grad("all", modality_encoders=True, mfla=True, msla=True, patch_embedders=True) # NOTE !! This used to be msla=False
    # model.set_requires_grad("backbone", mask_token=True)
    model.set_requires_grad("backbone", mask_token=True, blocks=True)
    print("  Unfroze: modality_encoders, modality_fusion_lora_adaptors, mask_token")

    # Count parameters
    trainable_in_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    total_in_evan = sum(p.numel() for p in evan.parameters())
    trainable_decoder = sum(p.numel() for p in mae_decoders.parameters())
    trainable_projector = sum(p.numel() for p in latent_projectors.parameters())
    trainable_total = trainable_in_evan + trainable_decoder + trainable_projector
    print(f"\nTrainable parameters: {trainable_total}")
    print(f"    EVAN: {trainable_in_evan} ({100*trainable_in_evan/total_in_evan:.2f}%)")
    print(f"    MAE decoders: {trainable_decoder}")
    print(f"    Latent projectors: {trainable_projector}")

    # Optimizer
    params = (
        list(filter(lambda p: p.requires_grad, evan.parameters())) +
        list(mae_decoders.parameters()) +
        list(latent_projectors.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.ssl_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    # Use modality_bands_dict if provided, otherwise use bands_target
    if modality_bands_dict is None:
        modality_bands_dict = bands_target

    # Loss function for latent reconstruction
    mse_fn = nn.MSELoss()

    # Training loop
    global_step = 0
    for epoch in range(args.stage2_fusion_epochs):
        evan.train()
        mae_decoders.train()
        latent_projectors.train()
        train_loss = 0.0
        train_mae_loss = 0.0
        train_latent_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Fusion MAE Epoch {epoch+1}/{args.stage2_fusion_epochs}")
        for batch in pbar:
            # Step 1: Create input tensors for all modalities
            evan_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=tuple(all_modalities)
            )
            evan_input = {k: v.to(device) for k, v in evan_input.items()}
            B = next(iter(evan_input.values())).shape[0]

            # Step 2: Get modality-specific features (student)
            mod_specific = evan.forward_modality_specific_features(evan_input)
            # {mod: [B, 1+n_storage+num_patches, embed_dim]}

            # Step 3: Get teacher targets (unmasked, frozen)
            with torch.no_grad():
                teacher_input={teacher_mod:evan_input[teacher_mod] for teacher_mod in latent_reconstruct_modalities}
                teacher_out = teacher_evan.forward_features(teacher_input)
                # {mod: {'x_norm_patchtokens': [B, num_patches, embed_dim], ...}}

            # Step 4: Generate independent random masks per modality
            # This forces cross-modal learning: when RGB position is masked but newmod is visible,
            # the model must use newmod features to help reconstruct RGB latents
            len_keep = int(num_patches * (1 - args.mae_mask_ratio))
            modality_masks = {}  # {mod: [B, num_patches] bool tensor, True=masked}
            for mod in all_modalities:
                noise = torch.rand(B, num_patches, device=device)
                ids_shuffle = torch.argsort(noise, dim=1)
                mask = torch.ones(B, num_patches, device=device, dtype=torch.bool)
                mask.scatter_(1, ids_shuffle[:, :len_keep], False)
                modality_masks[mod] = mask

            # Step 5: Apply per-modality mask to patch tokens (replace masked with mask_token)
            masked_mod_features = {}
            for mod, features in mod_specific.items():
                # features: [B, 1+n_storage+num_patches, embed_dim]
                n_prefix = evan.n_storage_tokens + 1
                cls_storage = features[:, :n_prefix, :]  # [B, 1+n_storage, embed_dim]
                patches = features[:, n_prefix:, :]  # [B, num_patches, embed_dim]

                # Replace masked positions with mask_token (using this modality's mask)
                mask_expanded = modality_masks[mod].unsqueeze(-1)  # [B, num_patches, 1]
                mask_token_expanded = evan.mask_token.unsqueeze(0).expand(B, num_patches, -1)
                masked_patches = torch.where(mask_expanded, mask_token_expanded, patches)

                # Reconstruct full sequence with masked patches
                masked_mod_features[mod] = torch.cat([cls_storage, masked_patches], dim=1)

            # Step 6: Forward through fusion blocks
            student_fused = evan.forward_fusion_from_modality_features(masked_mod_features)
            # {mod: {'x_norm_patchtokens': [B, num_patches, embed_dim], ...}}

            # Step 7: Compute losses (each modality uses its own mask)
            total_loss = 0.0
            batch_mae_loss = 0.0
            batch_latent_loss = 0.0

            # MAE loss: decode to pixels, loss on masked patches only
            for mod in mae_modalities:
                student_patches = student_fused[mod]['x_norm_patchtokens']  # [B, num_patches, embed_dim]
                pred_pixels = mae_decoders[mod](student_patches)  # [B, num_patches, patch_size^2 * C]

                target_img = evan_input[mod]  # [B, C, H, W]
                target_patches = patchify(target_img, patch_size)  # [B, num_patches, patch_size^2 * C]

                mask_float = modality_masks[mod].float()
                mae_loss = mae_reconstruction_loss(pred_pixels, target_patches, mask_float)
                total_loss = total_loss + mae_loss
                batch_mae_loss += mae_loss.item()

            # Latent loss: project and match teacher, loss on masked patches + CLS token
            for mod in latent_reconstruct_modalities:
                student_patches = student_fused[mod]['x_norm_patchtokens']  # [B, num_patches, embed_dim]
                teacher_patches = teacher_out[mod]['x_norm_patchtokens'].detach()  # [B, num_patches, embed_dim]
                student_cls = student_fused[mod]['x_norm_clstoken']  # [B, embed_dim]
                teacher_cls = teacher_out[mod]['x_norm_clstoken'].detach()  # [B, embed_dim]

                # Concatenate CLS with patches: treat CLS as an extra "patch" for projection
                student_all = torch.cat([student_cls.unsqueeze(1), student_patches], dim=1)  # [B, 1+num_patches, embed_dim]
                teacher_all = torch.cat([teacher_cls.unsqueeze(1), teacher_patches], dim=1)  # [B, 1+num_patches, embed_dim]

                projected = latent_projectors[mod](student_all)  # [B, 1+num_patches, embed_dim]

                latent_loss = mse_fn(projected, teacher_all)
                total_loss = total_loss + latent_loss
                batch_latent_loss += latent_loss.item()

            # Step 8: Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))
            optimizer.step()

            train_loss += total_loss.item()
            train_mae_loss += batch_mae_loss
            train_latent_loss += batch_latent_loss
            train_count += 1
            global_step += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'mae': f'{batch_mae_loss:.4f}',
                'latent': f'{batch_latent_loss:.4f}',
                'grad': f'{grad_norm:.4f}'
            })

            # Log to wandb every step
            wandb.log({
                'phase2_fusion/train_loss': total_loss.item(),
                'phase2_fusion/mae_loss': batch_mae_loss,
                'phase2_fusion/latent_loss': batch_latent_loss,
                'phase2_fusion/grad_norm': grad_norm.item(),
                'phase2_fusion/epoch': epoch + 1,
                'phase2_fusion/step': global_step,
                'phase2_fusion/lr': optimizer.param_groups[0]['lr'],
            })

        # Epoch summary
        train_loss /= train_count
        train_mae_loss /= train_count
        train_latent_loss /= train_count

        scheduler.step(train_loss)
        print(f"\nFusion MAE Epoch {epoch+1}/{args.stage2_fusion_epochs}:")
        print(f"  Train - Total: {train_loss:.4f}, MAE: {train_mae_loss:.4f}, Latent: {train_latent_loss:.4f}")
        
    print("\n=== Phase 2 (Fusion MAE Training) complete ===")
    return mae_decoders, latent_projectors
