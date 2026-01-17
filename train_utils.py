"""Training utilities for EVAN on EuroSAT."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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


class SimpleMAEDecoder(nn.Module):
    """Lightweight decoder for MAE reconstruction of specified modality patches."""

    def __init__(self, embed_dim, num_channels, patch_size, decoder_depth=2, decoder_heads=8, ffn_factor=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Learnable mask token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Simple transformer decoder (2 layers by default)
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
        self.decoder_pred = nn.Linear(embed_dim, patch_size * patch_size * num_channels)

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, x_unmasked, ids_restore):
        """
        Args:
            x_unmasked: [B, num_unmasked, embed_dim] - Unmasked patch embeddings
            ids_restore: [B, num_patches] - Indices to restore original order

        Returns:
            reconstructed patches: [B, num_patches, patch_size^2 * channels]
        """
        B, L_unmasked, D = x_unmasked.shape
        L_total = ids_restore.shape[1]

        # Create full sequence with mask tokens at masked positions
        mask_tokens = self.mask_token.expand(B, L_total - L_unmasked, -1)
        x_full = torch.cat([x_unmasked, mask_tokens], dim=1)  # [B, num_patches, embed_dim]

        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, D))

        # Decode and predict pixels for all patches
        x = self.decoder(x_full)
        x = self.decoder_pred(x)
        return x


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


def evaluate_mae_reconstruction(model, evan, mae_decoder, dataloader, device,
                                bands_target, patch_size, mask_ratio, target_modality):
    """Evaluate MAE reconstruction loss on test set."""
    model.eval()
    mae_decoder.eval()
    total_loss = 0.0
    count = 0

    target_modality_indices = get_band_indices(bands_target)

    with torch.no_grad():
        for batch in dataloader:
            # Extract and normalize target modality
            images = batch['image']
            target_modality_normalized = normalize_bands(images, target_modality_indices, BAND_MINS, BAND_MAXS).to(device)

            target_patches = patchify(target_modality_normalized, patch_size)

            target_modality_features = evan.forward_modality_specific_features({target_modality: target_modality_normalized})[target_modality]
            # Extract patch tokens (skip CLS and storage tokens)
            patch_embeddings = target_modality_features[:, evan.n_storage_tokens + 1:, :]  # [B, num_patches, embed_dim]

            # Random masking
            x_masked, mask, ids_restore = random_mask_patches(patch_embeddings, mask_ratio)

            # Decoder predicts for all patches (unmasked + mask tokens)
            pred_full = mae_decoder(x_masked, ids_restore)  # [B, num_patches, patch_size^2 * C]
            # Compute loss only on masked patches
            loss = mae_reconstruction_loss(pred_full, target_patches, mask)
            total_loss += loss.item()
            count += 1

    return total_loss / count


def single_modality_training_loop(model, train_loader, test_loader, device,
                                   modality_bands_dict, criterion, optimizer, num_epochs,
                                   modality, phase_name="Training",
                                   use_wandb=False, wandb_prefix=None, clip_norm=float('inf')):
    """
    Simple training loop for single-modality EVAN training (Stage 0).

    Args:
        model: EVAN classifier
        train_loader: Training dataloader
        test_loader: Test dataloader
        device: torch device
        modality_bands_dict: Dict mapping modality name to band tuple
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        modality: Single modality name to train and evaluate on
        phase_name: Name of this training phase for logging
        use_wandb: Whether to log to wandb
        wandb_prefix: Prefix for wandb metrics

    Returns:
        Tuple of (train_acc, test_acc)
    """
    mod_str = modality.upper()
    global_step = 0
    best_test_acc = 0
    best_epoch = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch+1}/{num_epochs} [{mod_str}]")
        for batch in pbar:
            labels = batch['label'].to(device)

            modal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict, modalities=(modality,)
            )
            modal_input = {k: v.to(device) for k, v in modal_input.items()}

            optimizer.zero_grad()
            outputs = model(modal_input)
            loss = criterion(outputs, labels)
            loss.backward()

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%',
                'grad_norm': f'{grad_norm:.4f}'
            })

            if use_wandb and wandb_prefix:
                wandb.log({
                    f'{wandb_prefix}/train_loss': loss.item(),
                    f'{wandb_prefix}/grad_norm': grad_norm.item(),
                    f'{wandb_prefix}/step': global_step,
                })

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Evaluate on same modality
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device,
            modality_bands_dict, modalities_to_use=(modality,)
        )
        if (epoch%8==1) or (epoch+1==num_epochs):
            print(f"  Train ({mod_str}): Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Test ({mod_str}):  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% (epoch {epoch+1}/{num_epochs})")
        if test_acc > best_test_acc:
            print(f"    New record: {test_acc:.2f}> previous {best_test_acc:.2f} at epoch {epoch+1}")
            best_test_acc=test_acc
            best_epoch=epoch+1
        if use_wandb and wandb_prefix:
            wandb.log({
                f'{wandb_prefix}/train_loss_epoch': train_loss,
                f'{wandb_prefix}/train_acc': train_acc,
                f'{wandb_prefix}/eval_loss': test_loss,
                f'{wandb_prefix}/eval_acc': test_acc,
                f'{wandb_prefix}/epoch': epoch + 1,
            })

    return train_acc, test_acc, best_test_acc, best_epoch


def supervised_training_loop(model, train_loader, test_loader_full, device,
                             modality_bands_dict, criterion, optimizer, num_epochs,
                             train_modalities, newmod=None, phase_name="Training",
                             modality_masking=None,
                             use_wandb=False, wandb_prefix=None, freeze_rgb=False):
    """
    General supervised training loop for EVAN with multi-modal support.

    Args:
        model: EVAN classifier
        train_loader: Training dataloader
        test_loader_full: Test dataloader
        device: torch device
        modality_bands_dict: Dict mapping modality names to band tuples
        criterion: Loss function
        optimizer: Optimizer (already configured with trainable parameters)
        num_epochs: Number of epochs to train
        train_modalities: Tuple of modality names to use for training (e.g., ('rgb',) or ('rgb', 'vre'))
        newmod: Name of the new modality (for logging)
        phase_name: Name of this training phase for logging
        modality_masking: Optional dict mapping modality name to masking probability (e.g., {'rgb': 0.3, 'vre': 0.2})
                         During training, one modality is randomly masked out per batch based on these probabilities.

    Returns:
        Tuple of (train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi)
        (test_acc_newmod is None if eval_newmod_only=False)
    """

    # Determine training modality string for logging
    train_mod_str = "+".join(m.upper() for m in train_modalities)

    # Setup modality masking if specified
    if modality_masking:
        import random
        masking_info = " (with modality masking: " + ", ".join(f"{k}={v:.1%}" for k, v in modality_masking.items()) + ")"
        print(f"Modality masking enabled{masking_info}")
    else:
        modality_masking = {}

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch+1}/{num_epochs} [Train {train_mod_str}]")
        for batch in pbar:
            labels = batch['label'].to(device)

            # Create modal input
            modal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict, modalities=train_modalities
            )
            modal_input = {k: v.to(device) for k, v in modal_input.items()}

            # Apply modality masking (mask one modality at a time per batch)
            # Note: we never mask so aggressively that no modalities remain
            # Also: if freeze_rgb=True, never mask the newmod (would leave only frozen RGB)
            if modality_masking and len(modal_input) > 1:
                import random
                for modality_name, mask_prob in modality_masking.items():
                    # Skip masking newmod if RGB is frozen (would leave no trainable params)
                    if freeze_rgb and modality_name == newmod:
                        continue
                    if modality_name in modal_input and len(modal_input) > 1 and random.random() < mask_prob:
                        # Mask this modality by removing it from the input
                        del modal_input[modality_name]
                        break  # Only mask one modality at a time

            optimizer.zero_grad()
            outputs = model(modal_input)
            loss = criterion(outputs, labels)
            loss.backward()

            # Compute gradient norm
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=float('inf'))

            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            global_step += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*train_correct/train_total:.2f}%',
                'grad_norm': f'{grad_norm:.4f}'
            })

            # Log to wandb every step
            if use_wandb and wandb_prefix:
                wandb.log({
                    f'{wandb_prefix}/train_loss': loss.item(),
                    f'{wandb_prefix}/grad_norm': grad_norm.item(),
                    f'{wandb_prefix}/step': global_step,
                })

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total

        # Evaluation on training modalities
        test_loss_train, test_acc_train = evaluate(
            model, test_loader_full, criterion, device,
            modality_bands_dict, modalities_to_use=train_modalities
        )

        # Evaluation on RGB-only
        test_loss_rgb, test_acc_rgb = evaluate(
            model, test_loader_full, criterion, device,
            modality_bands_dict, modalities_to_use=('rgb',)
        )

        test_acc_newmod = None
        if newmod:
            test_loss_newmod, test_acc_newmod = evaluate(
                model, test_loader_full, criterion, device,
                modality_bands_dict, modalities_to_use=(newmod,)
            )

        # Evaluation on RGB+newmod
        if newmod:
            test_loss_multi, test_acc_multi = evaluate(
                model, test_loader_full, criterion, device,
                modality_bands_dict, modalities_to_use=('rgb', newmod)
            )

        # Print epoch results
        print(f"\n{phase_name} Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train ({train_mod_str}):      Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test ({train_mod_str}):       Loss: {test_loss_train:.4f}, Acc: {test_acc_train:.2f}%")
        print(f"  Test (RGB only):       Loss: {test_loss_rgb:.4f}, Acc: {test_acc_rgb:.2f}%")
        if newmod:
            print(f"  Test ({newmod.upper()} only):        Loss: {test_loss_newmod:.4f}, Acc: {test_acc_newmod:.2f}%")
        print(f"  Test (RGB+{newmod.upper()}):         Loss: {test_loss_multi:.4f}, Acc: {test_acc_multi:.2f}%")
        if newmod:
            print(f"  Multi-modal gain:      {test_acc_multi - test_acc_rgb:+.2f}%")
        print()

        # Log epoch-level metrics to wandb
        if use_wandb and wandb_prefix:
            log_dict = {
                f'{wandb_prefix}/train_loss_epoch': train_loss,
                f'{wandb_prefix}/train_acc': train_acc,
                f'{wandb_prefix}/eval_loss_multi': test_loss_multi,
                f'{wandb_prefix}/eval_acc_rgb': test_acc_rgb,
                f'{wandb_prefix}/eval_acc_multi': test_acc_multi,
                f'{wandb_prefix}/epoch': epoch + 1,
            }
            if newmod:
                log_dict[f'{wandb_prefix}/eval_loss_newmod'] = test_loss_newmod
                log_dict[f'{wandb_prefix}/eval_acc_newmod'] = test_acc_newmod
            wandb.log(log_dict)

    return train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi


def supervised_finetune_phase(model, evan, train_loader, test_loader_full, device, args,
                               newmod, modality_bands_dict, criterion, phase_name="Stage 2",
                               modality_masking=None, freeze_rgb=True, unfreeze_modality_specific=False,
                               use_wandb=False, wandb_prefix=None):
    """
    Supervised fine-tuning for new modality components.

    Trains:
    - New modality fusion LoRAs
    - New modality encoding
    - Classifier(s) (depends on fusion strategy)

    Optionally trains (if train_modality_specific=True):
    - New modality patch embedder
    - New modality modality-specific LoRA

    Frozen:
    - RGB components and shared DINO backbone

    Args:
        modality_masking: Optional dict mapping modality name to masking probability
                         (e.g., {'rgb': 0.3, 'vre': 0.2}) for robustness training
    """
    print("\n" + "="*70)
    print(f"=== {phase_name}: Supervised Fine-tuning ===")
    print("="*70)

    bands_rgb = modality_bands_dict['rgb']
    bands_newmod = modality_bands_dict[newmod]

    # Ensure new modality EVAN components exist (patch embedder, LoRAs, encoders)
    if newmod not in evan.patch_embedders:
        num_newmod_channels = len(bands_newmod)
        print(f"  Creating {newmod} modality components (embedder, LoRAs, encoders)...")
        evan._create_modality_components(newmod, num_newmod_channels)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier(s)
    if model.fusion_strategy == 'mean':
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("  Unfroze: Classifier (mean fusion)")
    elif model.fusion_strategy == 'ensemble':
        # Ensure new modality classifier exists
        if newmod not in model.modality_classifiers:
            print(f"  Creating {newmod} classifier")
            model._instantiate_modality_classifier(newmod)

        # Unfreeze new modality classifier
        for param in model.modality_classifiers[newmod].parameters():
            param.requires_grad = True
        print(f"  Unfroze: {newmod.capitalize()} classifier (ensemble mode)")

        # RGB classifier stays frozen or unfrozen based on flag
        if freeze_rgb:
            print("WARNING: RGB classifier head is frozen.")
            for param in model.modality_classifiers['rgb'].parameters():
                param.requires_grad = False
        else:
            print("Unfreezing RGB classifier head.")
            for param in model.modality_classifiers['rgb'].parameters():
                param.requires_grad = True

    # Unfreeze new modality components (if training all components, not just fusion)
    if unfreeze_modality_specific:
        # Patch embedder
        if newmod in evan.patch_embedders:
            for param in evan.patch_embedders[newmod].parameters():
                param.requires_grad = True
            print(f"  Unfroze: {newmod.capitalize()} patch embedder")

        # Modality-specific layers
        if newmod in evan.modality_specific_layer_adaptors:
            for param in evan.modality_specific_layer_adaptors[newmod].parameters():
                param.requires_grad = True
            print(f"  Unfroze: {newmod.capitalize()} modality-specific layers")
    else:
        # Explicitly confirm these stay frozen
        print(f"  Frozen: {newmod.capitalize()} patch embedder (not training)")
        print(f"  Frozen: {newmod.capitalize()} modality-specific layers (not training)")

    # Modality encoding
    evan.modality_encoders[newmod].requires_grad_(True)
    print(f"  Unfroze: {newmod.capitalize()} modality encoding")

    # Fusion LoRAs
    evan.modality_fusion_lora_adaptors[newmod].requires_grad_(True)
    print(f"  Unfroze: {newmod.capitalize()} fusion LoRAs")

    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable parameters for {phase_name}: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    if trainable_params == 0:
        raise ValueError("No trainable parameters found! Check that components are being unfrozen correctly.")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.stage2_lr
    )
    num_epochs = args.stage2_ft_epochs

    print(f"\n=== {phase_name} Training for {num_epochs} epochs ===")
    print(f"Learning rate: {args.stage2_lr}")
    print(f"Modalities: RGB {bands_rgb} + {newmod.upper()} {bands_newmod}")

    if unfreeze_modality_specific:
        print(f"Training: {newmod.capitalize()} patch embedder, modality-specific LoRAs, modality encoding, fusion LoRAs, classifier")
        print("Frozen: RGB components and shared DINO backbone")
    else:
        print(f"Training: {newmod.capitalize()} modality encoding, fusion LoRAs, classifier")
        print(f"Frozen: {newmod.upper()} patch embedder, {newmod.upper()} modality-specific LoRAs, RGB components, DINO backbone")


    # Run training loop
    train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi = supervised_training_loop(
        model, train_loader, test_loader_full, device,
        modality_bands_dict, criterion, optimizer, num_epochs,
        train_modalities=('rgb', newmod), newmod=newmod, phase_name=phase_name,
        eval_newmod_only=True, modality_masking=modality_masking,
        use_wandb=use_wandb, wandb_prefix=wandb_prefix, freeze_rgb=freeze_rgb
    )

    print(f"\n=== {phase_name} complete ===")

    return optimizer, train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi


def train_mae_phase(model, evan, train_loader, test_loader_full, device, args,
                    bands_target, target_modality, use_wandb=False):
    """
    Phase 2a: MAE SSL training for target_modality components.

    Trains:
    - Target modality patch embedder
    - Target modality modality-specific LoRA

    Frozen:
    - Everything else (DINO-initialized backbone, RGB components, fusion LoRA, classifier)
    """
    print("\n" + "="*70)
    print(f"=== PHASE 2a: MAE SSL Training for {target_modality} ===")
    print("="*70)

    # Create MAE decoder
    patch_size = evan.patch_size  # EVAN patch size
    num_target_channels = len(bands_target)
    mae_decoder = SimpleMAEDecoder(
        embed_dim=evan.embed_dim,
        num_channels=num_target_channels,
        patch_size=patch_size,
        decoder_depth=2,
        decoder_heads=8
    ).to(device)

    # Ensure target modality components exist (create if needed)
    if target_modality not in evan.patch_embedders:
        print(f"  Creating {target_modality} modality components...")
        evan._create_modality_components(target_modality, num_target_channels)
        print(f"  Created: {target_modality} patch embedder, modality-specific LoRAs, modality encoding, fusion LoRAs")

    # Freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze target modality patch embedder and modality-specific layers for MAE training
    for param in evan.patch_embedders[target_modality].parameters():
        param.requires_grad = True
    for param in evan.modality_specific_layer_adaptors[target_modality].parameters():
        param.requires_grad = True
    print(f"  Unfroze: {target_modality} patch embedder")
    print(f"  Unfroze: {target_modality} modality-specific layers")

    trainable_params_in_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    trainable_params_decoder = sum(p.numel() for p in mae_decoder.parameters())
    trainable_total = trainable_params_in_evan+trainable_params_decoder
    print(f"\nTrainable parameters for MAE: {trainable_total}\n    {trainable_params_in_evan=} and {trainable_params_decoder=}")

    # Optimizer for MAE phase - collect trainable parameters from model + all decoder parameters
    mae_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(mae_decoder.parameters())

    optimizer_mae = torch.optim.AdamW(mae_params, lr=args.mae_lr)

    # Pre-compute target modality indices
    target_modality_indices = get_band_indices(bands_target)

    # Training loop
    global_step = 0
    for epoch in range(args.stage2_mae_epochs):
        model.train()
        mae_decoder.train()
        train_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"MAE Epoch {epoch+1}/{args.stage2_mae_epochs}")
        for batch in pbar:
            # Extract and normalize target modality bands
            images = batch['image']  # [B, 13, H, W]
            target_modality_normalized = normalize_bands(images, target_modality_indices, BAND_MINS, BAND_MAXS).to(device)

            # Patchify for target
            target_patches = patchify(target_modality_normalized, patch_size)  # [B, num_patches, patch_size^2 * C]

            # Forward through EVAN modality-specific layers only (first tz_fusion_time blocks)
            target_modality_features = evan.forward_modality_specific_features({target_modality: target_modality_normalized})[target_modality]
            # Extract patch tokens (skip CLS and storage tokens)
            patch_embeddings = target_modality_features[:, evan.n_storage_tokens + 1:, :]  # [B, num_patches, embed_dim]

            # Random masking
            x_masked, mask, ids_restore = random_mask_patches(patch_embeddings, args.mae_mask_ratio)

            # Decoder predicts for all patches (unmasked + mask tokens)
            pred_full = mae_decoder(x_masked, ids_restore)  # [B, num_patches, patch_size^2 * C]
            # Compute loss only on masked patches
            loss = mae_reconstruction_loss(pred_full, target_patches, mask)

            optimizer_mae.zero_grad()
            loss.backward()

            # Compute gradient norm before stepping
            grad_norm = torch.nn.utils.clip_grad_norm_(mae_params, max_norm=float('inf'))

            optimizer_mae.step()

            train_loss += loss.item()
            train_count += 1
            global_step += 1
            pbar.set_postfix({'mae_loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'})

            # Log to wandb every step
            if use_wandb:
                wandb.log({
                    'phase2a_mae/train_loss': loss.item(),
                    'phase2a_mae/grad_norm': grad_norm.item(),
                    'phase2a_mae/epoch': epoch + 1,
                    'phase2a_mae/step': global_step,
                })

        train_loss /= train_count

        # Evaluation: Show reconstruction quality on test set
        eval_loss = evaluate_mae_reconstruction(
            model, evan, mae_decoder, test_loader_full, device,
            bands_target, patch_size, args.mae_mask_ratio, target_modality
        )

        # Log epoch-level metrics
        if use_wandb:
            wandb.log({
                'phase2a_mae/train_loss_epoch': train_loss,
                'phase2a_mae/eval_loss': eval_loss,
                'phase2a_mae/epoch': epoch + 1,
            })

        print(f"\nMAE Epoch {epoch+1}/{args.stage2_mae_epochs}:")
        print(f"  Train reconstruction loss: {train_loss:.4f}")
        print(f"  Test reconstruction loss:  {eval_loss:.4f}\n")

    print("\n=== Phase 2a (MAE SSL) complete ===")
    return mae_decoder  # Return for potential analysis


# ==================== Multi-Modal Fusion MAE (Stage 3) ====================

class MultiModalMaskingStrategy:
    """
    Asymmetric masking strategy for multi-modal fusion MAE.

    - RGB: 50% modality-level drop + 25% token masking when present
    - NewMod: Always present, 75% token masking
    """

    def __init__(
        self,
        rgb_modality_drop_prob: float = 0.5,
        rgb_token_mask_ratio: float = 0.25,
        newmod_token_mask_ratio: float = 0.75
    ):
        self.rgb_modality_drop_prob = rgb_modality_drop_prob
        self.rgb_token_mask_ratio = rgb_token_mask_ratio
        self.newmod_token_mask_ratio = newmod_token_mask_ratio

    def _generate_mask(self, batch_size: int, num_patches: int, mask_ratio: float, device):
        """Generate random mask for patches."""
        len_keep = int(num_patches * (1 - mask_ratio))

        # Random shuffle
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first len_keep patches
        ids_keep = ids_shuffle[:, :len_keep]

        # Create mask: 0 is keep, 1 is masked
        mask = torch.ones([batch_size, num_patches], device=device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return ids_keep, mask.bool(), ids_restore

    def __call__(self, batch_size: int, num_patches: int, device, newmod_key: str):
        """
        Generate masking info for a batch.

        Returns:
            Dict[modality] -> {
                'present': bool,
                'ids_keep': Tensor [B, num_keep],
                'mask': Tensor [B, num_patches] (True=masked),
                'ids_restore': Tensor [B, num_patches]
            }
        """
        import random
        result = {}

        # RGB: potentially dropped entirely
        rgb_present = random.random() > self.rgb_modality_drop_prob
        if rgb_present:
            ids_keep, mask, ids_restore = self._generate_mask(
                batch_size, num_patches, self.rgb_token_mask_ratio, device
            )
            result['rgb'] = {
                'present': True,
                'ids_keep': ids_keep,
                'mask': mask,
                'ids_restore': ids_restore
            }
        else:
            result['rgb'] = {'present': False}

        # NewMod: always present, always masked at token level
        ids_keep, mask, ids_restore = self._generate_mask(
            batch_size, num_patches, self.newmod_token_mask_ratio, device
        )
        result[newmod_key] = {
            'present': True,
            'ids_keep': ids_keep,
            'mask': mask,
            'ids_restore': ids_restore
        }

        return result


class MultiModalMAEDecoder(nn.Module):
    """
    Shared decoder backbone with per-modality prediction heads for multi-modal MAE.
    """

    def __init__(self, embed_dim, patch_size, decoder_depth=2, decoder_heads=8, ffn_factor=4):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Shared mask token for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # Shared transformer decoder
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        decoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=decoder_heads,
            dim_feedforward=embed_dim * ffn_factor,
            batch_first=True,
            norm_first=True
        )
        self.decoder = TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Per-modality prediction heads (populated dynamically)
        self.modality_pred_heads = nn.ModuleDict()

        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)

    def register_modality(self, modality_name: str, num_channels: int):
        """Register a modality with its channel count."""
        self.modality_pred_heads[modality_name] = nn.Linear(
            self.embed_dim,
            self.patch_size * self.patch_size * num_channels
        )

    def forward(self, x_full, modality_name: str):
        """
        Decode and predict for a specific modality.

        Args:
            x_full: [B, num_patches, embed_dim] - Full sequence with mask tokens already inserted
            modality_name: Which modality's prediction head to use

        Returns:
            pred: [B, num_patches, patch_size^2 * num_channels]
        """
        # Process through decoder
        x = self.decoder(x_full)

        # Use modality-specific prediction head
        if modality_name not in self.modality_pred_heads:
            raise ValueError(f"Modality '{modality_name}' not registered. Call register_modality first.")

        pred = self.modality_pred_heads[modality_name](x)
        return pred


def train_multimodal_fusion_mae_phase(
    model, evan, train_loader, test_loader_full, device, args,
    newmod, modality_bands_dict, use_wandb=False
):
    """
    Phase A of Stage 3: Train modality encoding + fusion LoRAs via multi-modal MAE.

    Trains:
    - modality_encoders[newmod]
    - modality_fusion_lora_adaptors[newmod]
    - MultiModalMAEDecoder

    Frozen:
    - patch_embedders (both RGB and newmod)
    - modality_specific_layer_adaptors (both)
    - classifiers
    - shared DINO backbone

    Returns:
        MultiModalMAEDecoder (for potential analysis)
    """
    print("\n" + "="*70)
    print(f"=== Stage 3 Phase A: Multi-modal Fusion MAE ===")
    print("="*70)

    bands_rgb = modality_bands_dict['rgb']
    bands_newmod = modality_bands_dict[newmod]

    # Create shared decoder with per-modality prediction heads
    mae_decoder = MultiModalMAEDecoder(
        embed_dim=evan.embed_dim,
        patch_size=evan.patch_size,
        decoder_depth=2,
        decoder_heads=8
    ).to(device)

    # Register modalities
    mae_decoder.register_modality('rgb', num_channels=len(bands_rgb))
    mae_decoder.register_modality(newmod, num_channels=len(bands_newmod))

    # Ensure newmod components exist
    if newmod not in evan.patch_embedders:
        num_newmod_channels = len(bands_newmod)
        print(f"  Creating {newmod} modality components...")
        evan._create_modality_components(newmod, num_newmod_channels)

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze: modality encoding (newmod) and fusion LoRAs (newmod)
    evan.modality_encoders[newmod].requires_grad_(True)
    print(f"  Unfroze: {newmod.capitalize()} modality encoding")

    evan.modality_fusion_lora_adaptors[newmod].requires_grad_(True)
    print(f"  Unfroze: {newmod.capitalize()} fusion LoRAs")

    print(f"  Frozen: Patch embedders, modality-specific LoRAs, classifiers, DINO backbone")

    # Count parameters
    trainable_params_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    trainable_params_decoder = sum(p.numel() for p in mae_decoder.parameters())
    print(f"\nTrainable parameters: {trainable_params_evan + trainable_params_decoder:,}")
    print(f"  EVAN (encoding + fusion LoRAs): {trainable_params_evan:,}")
    print(f"  MAE Decoder: {trainable_params_decoder:,}")

    # Create masking strategy
    masking_strategy = MultiModalMaskingStrategy(
        rgb_modality_drop_prob=getattr(args, 'rgb_modality_drop_prob', 0.5),
        rgb_token_mask_ratio=getattr(args, 'rgb_token_mask_ratio', 0.25),
        newmod_token_mask_ratio=getattr(args, 'newmod_token_mask_ratio', 0.75)
    )
    print(f"\nMasking strategy:")
    print(f"  RGB: {masking_strategy.rgb_modality_drop_prob:.0%} modality drop, {masking_strategy.rgb_token_mask_ratio:.0%} token mask")
    print(f"  {newmod.upper()}: {masking_strategy.newmod_token_mask_ratio:.0%} token mask (always present)")

    # Create optimizer
    mae_params = (
        list(filter(lambda p: p.requires_grad, model.parameters())) +
        list(mae_decoder.parameters())
    )
    optimizer = torch.optim.AdamW(mae_params, lr=getattr(args, 'mae_fusion_lr', 1e-4))

    num_epochs = getattr(args, 'mae_fusion_epochs', 4)
    print(f"\n=== Training for {num_epochs} epochs ===")
    print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

    # Pre-compute band indices
    rgb_indices = get_band_indices(bands_rgb)
    newmod_indices = get_band_indices(bands_newmod)
    patch_size = evan.patch_size
    num_patches = (224 // patch_size) ** 2

    # Training loop
    global_step = 0
    for epoch in range(num_epochs):
        model.train()
        mae_decoder.train()
        train_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Fusion MAE Epoch {epoch+1}/{num_epochs}")
        for batch in pbar:
            images = batch['image']
            B = images.shape[0]

            # Normalize both modalities
            rgb_normalized = normalize_bands(images, rgb_indices, BAND_MINS, BAND_MAXS).to(device)
            newmod_normalized = normalize_bands(images, newmod_indices, BAND_MINS, BAND_MAXS).to(device)

            # Generate masks
            mask_info = masking_strategy(B, num_patches, device, newmod)

            # Build input dict (RGB may be dropped)
            modal_input = {}
            if mask_info['rgb']['present']:
                modal_input['rgb'] = rgb_normalized
            modal_input[newmod] = newmod_normalized

            # Forward through fusion with masking
            features, recon_info = evan.forward_features_masked_multimodal(modal_input, mask_info)

            # Compute reconstruction loss for each present modality
            total_loss = 0.0
            loss_count = 0

            for mod_name in modal_input.keys():
                # Get target patches
                if mod_name == 'rgb':
                    target_img = rgb_normalized
                else:
                    target_img = newmod_normalized
                target_patches = patchify(target_img, patch_size)

                # Extract patch features (skip CLS/storage)
                patch_features = features[mod_name][:, evan.n_storage_tokens + 1:, :]

                # Get mask
                mask = recon_info[mod_name]['mask']

                # Decode with modality-specific head
                pred_patches = mae_decoder(patch_features, mod_name)

                # Loss only on masked patches
                loss = mae_reconstruction_loss(pred_patches, target_patches, mask)
                total_loss += loss
                loss_count += 1

            # Average loss across modalities
            total_loss = total_loss / loss_count

            optimizer.zero_grad()
            total_loss.backward()

            # Compute gradient norm
            grad_norm = torch.nn.utils.clip_grad_norm_(mae_params, max_norm=float('inf'))

            optimizer.step()

            train_loss += total_loss.item()
            train_count += 1
            global_step += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'grad_norm': f'{grad_norm:.4f}',
                'mods': '+'.join(modal_input.keys())
            })

            # Log to wandb
            if use_wandb:
                wandb.log({
                    'stage3_mae/train_loss': total_loss.item(),
                    'stage3_mae/grad_norm': grad_norm.item(),
                    'stage3_mae/step': global_step,
                    'stage3_mae/rgb_present': 1.0 if mask_info['rgb']['present'] else 0.0,
                })

        train_loss /= train_count

        # Log epoch-level metrics
        if use_wandb:
            wandb.log({
                'stage3_mae/train_loss_epoch': train_loss,
                'stage3_mae/epoch': epoch + 1,
            })

        print(f"\nFusion MAE Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train reconstruction loss: {train_loss:.4f}")

    print("\n=== Stage 3 Phase A (Fusion MAE) complete ===")
    return mae_decoder
