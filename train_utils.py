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
             modalities_to_use=('rgb',), pseudo_modalities=None, cls_projectors=None):
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
        pseudo_modalities: Optional list of modalities to hallucinate using mask tokens
        cls_projectors: Required if pseudo_modalities is provided; trained CLS projectors
    """
    model.eval()
    if cls_projectors is not None:
        cls_projectors.eval()
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

            if pseudo_modalities is not None:
                outputs = model(modal_input, pseudo_modalities=pseudo_modalities, cls_projectors=cls_projectors)
            else:
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


def evaluate_mae_reconstruction(evan, mae_decoder, dataloader, device,
                                bands_target, patch_size, mask_ratio, target_modality):
    """Evaluate MAE reconstruction loss on test set."""
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
                                   use_wandb=False, wandb_prefix=None, clip_norm=10,
                                   hallucinate_modality=False, pseudo_modalities=None,
                                   cls_projectors=None):
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
        clip_norm: Max gradient norm for clipping
        hallucinate_modality: If True, use pseudo-modality inference
        pseudo_modalities: List of modalities to hallucinate (required if hallucinate_modality=True)
        cls_projectors: Trained CLS projectors for pseudo-modality (required if hallucinate_modality=True)

    Returns:
        Tuple of (train_acc, test_acc, best_test_acc, best_epoch)
    """
    mod_str = modality.upper()
    global_step = 0
    best_test_acc = 0
    best_epoch = 0
    scheduler = ReduceLROnPlateau(optimizer,factor=0.5,patience=0,min_lr=1e-6)

    # Validate hallucinate_modality requirements
    if hallucinate_modality:
        assert pseudo_modalities is not None, "pseudo_modalities required when hallucinate_modality=True"
        assert cls_projectors is not None, "cls_projectors required when hallucinate_modality=True"
        print(f"  Using pseudo-modality inference: {modality} + hallucinated {pseudo_modalities}")

    for epoch in range(num_epochs):
        model.train()
        if cls_projectors is not None:
            cls_projectors.train()
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

            if hallucinate_modality:
                outputs = model(modal_input, pseudo_modalities=pseudo_modalities, cls_projectors=cls_projectors)
            else:
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

        # Evaluate on same modality (with pseudo-modality if enabled)
        if hallucinate_modality:
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device,
                modality_bands_dict, modalities_to_use=(modality,),
                pseudo_modalities=pseudo_modalities, cls_projectors=cls_projectors
            )
        else:
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, device,
                modality_bands_dict, modalities_to_use=(modality,)
            )
        scheduler.step(train_loss)
        if (epoch%8==1) or (epoch+1==num_epochs):
            print(f"  Train ({mod_str}): Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Test ({mod_str}):  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% (epoch {epoch+1}/{num_epochs})")
        if test_acc > best_test_acc:
            print(f"    New record: {test_acc:.2f} > previous {best_test_acc:.2f} at epoch {epoch+1}")
            best_test_acc=test_acc
            best_epoch=epoch+1
        if use_wandb and wandb_prefix:
            wandb.log({
                f'{wandb_prefix}/train_loss_epoch': train_loss,
                f'{wandb_prefix}/train_acc': train_acc,
                f'{wandb_prefix}/eval_loss': test_loss,
                f'{wandb_prefix}/eval_acc': test_acc,
                f'{wandb_prefix}/epoch': epoch + 1,
                f'{wandb_prefix}/lr': optimizer.param_groups[0]['lr'],
            })

    return train_acc, test_acc, best_test_acc, best_epoch


def supervised_training_loop(model, train_loader, test_loader_full, device,
                             modality_bands_dict, criterion, optimizer, num_epochs,
                             train_modalities, newmod=None, phase_name="Training",
                             modality_masking=None,
                             use_wandb=False, wandb_prefix=None, freeze_rgb=False,
                             eval_single_modalities=False):
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
        eval_single_modalities: If True, evaluate on RGB-only, newmod-only, and RGB+newmod separately.
                               If False (default), only evaluate on train_modalities.

    Returns:
        Tuple of (train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi)
        When eval_single_modalities=False, test_acc_rgb/newmod/multi are None (only test_acc_train is meaningful)
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
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)
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
        scheduler.step(train_loss)

        # Evaluation on training modalities
        test_loss_train, test_acc_train = evaluate(
            model, test_loader_full, criterion, device,
            modality_bands_dict, modalities_to_use=train_modalities
        )

        # Initialize optional metrics
        test_acc_rgb = None
        test_acc_newmod = None
        test_acc_multi = None
        test_loss_rgb = None
        test_loss_newmod = None
        test_loss_multi = None

        if eval_single_modalities:
            # Evaluation on RGB-only
            test_loss_rgb, test_acc_rgb = evaluate(
                model, test_loader_full, criterion, device,
                modality_bands_dict, modalities_to_use=('rgb',)
            )

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
        if eval_single_modalities:
            print(f"  Test (RGB only):       Loss: {test_loss_rgb:.4f}, Acc: {test_acc_rgb:.2f}%")
            if newmod:
                print(f"  Test ({newmod.upper()} only):        Loss: {test_loss_newmod:.4f}, Acc: {test_acc_newmod:.2f}%")
            if newmod:
                print(f"  Test (RGB+{newmod.upper()}):         Loss: {test_loss_multi:.4f}, Acc: {test_acc_multi:.2f}%")
                print(f"  Multi-modal gain:      {test_acc_multi - test_acc_rgb:+.2f}%")
        print()

        # Log epoch-level metrics to wandb
        if use_wandb and wandb_prefix:
            log_dict = {
                f'{wandb_prefix}/train_loss_epoch': train_loss,
                f'{wandb_prefix}/train_acc': train_acc,
                f'{wandb_prefix}/eval_loss_train': test_loss_train,
                f'{wandb_prefix}/eval_acc_train': test_acc_train,
                f'{wandb_prefix}/epoch': epoch + 1,
                f'{wandb_prefix}/lr': optimizer.param_groups[0]['lr'],
            }
            if eval_single_modalities:
                log_dict[f'{wandb_prefix}/eval_acc_rgb'] = test_acc_rgb
                if newmod:
                    log_dict[f'{wandb_prefix}/eval_loss_newmod'] = test_loss_newmod
                    log_dict[f'{wandb_prefix}/eval_acc_newmod'] = test_acc_newmod
                    log_dict[f'{wandb_prefix}/eval_loss_multi'] = test_loss_multi
                    log_dict[f'{wandb_prefix}/eval_acc_multi'] = test_acc_multi
            wandb.log(log_dict)

    return train_acc, test_acc_rgb, test_acc_newmod, test_acc_train


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
        evan.create_modality_components(newmod, num_newmod_channels)
        model.to(device)  # Move newly created components to device

    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze classifier(s)
    if model.classifier_strategy == 'mean':
        for param in model.classifier.parameters():
            param.requires_grad = True
        print("  Unfroze: Classifier (mean fusion)")
    elif model.classifier_strategy == 'ensemble':
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
        modality_masking=modality_masking,
        use_wandb=use_wandb, wandb_prefix=wandb_prefix, freeze_rgb=freeze_rgb
    )

    print(f"\n=== {phase_name} complete ===")

    return optimizer, train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi


def train_mae_phase(model, train_loader, test_loader_full, device, args, bands_target, target_modality):
    """
    Phase 1: MAE SSL training for target_modality components.

    Trains:
    - Target modality patch embedder
    - Target modality modality-specific LoRA

    Frozen:
    - Everything else (DINO-initialized backbone, RGB components, fusion LoRA, classifier)
    """
    print("\n" + "="*70)
    print(f"=== PHASE 1a: MAE SSL Training for {target_modality} ===")
    print("="*70)
    evan=model.evan
    # Create MAE decoder
    patch_size = evan.patch_size  # EVAN patch size
    num_target_channels = len(bands_target)
    mae_decoder = SimpleMAEDecoder(
        embed_dim=evan.embed_dim,
        num_channels=num_target_channels,
        patch_size=patch_size,
        decoder_depth=1,
        decoder_heads=8
    ).to(device)

    model.freeze_all() # Freeze everything
    model.set_requires_grad(target_modality, patch_embedders=True, clsreg=True, msla=True, mfla=False, classifier=False)
    print(f"  Unfroze: patch embedder, clsreg tokens, modality-specific layers adaptors")

    trainable_in_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    total_in_evan = sum(p.numel() for p in evan.parameters())
    trainable_decoder = sum(p.numel() for p in mae_decoder.parameters())
    trainable_total = trainable_in_evan+trainable_decoder
    assert trainable_decoder!=0 and trainable_in_evan!=0
    print(f"\nTrainable parameters for MAE: {trainable_total}\n    {trainable_in_evan=} ({100*trainable_in_evan/total_in_evan:.2f}% of EVAN) and {trainable_decoder=}")

    # Optimizer for MAE phase - collect trainable parameters from model + all decoder parameters
    mae_params = list(filter(lambda p: p.requires_grad, evan.parameters())) + list(mae_decoder.parameters())

    optimizer_mae = torch.optim.AdamW(mae_params, lr=args.ssl_lr)
    scheduler_mae = ReduceLROnPlateau(optimizer_mae,factor=0.5,patience=1,min_lr=1e-6)

    # Pre-compute target modality indices
    target_modality_indices = get_band_indices(bands_target)

    # Training loop
    global_step = 0
    for epoch in range(args.stage1_ssl_epochs):
        evan.train()
        mae_decoder.train()
        train_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"MAE Epoch {epoch+1}/{args.stage1_ssl_epochs}")
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
            wandb.log({
                'phase1_mae/train_loss': loss.item(),
                'phase1_mae/grad_norm': grad_norm.item(),
                'phase1_mae/epoch': epoch + 1,
                'phase1_mae/step': global_step,
                'phase1_mae/lr': optimizer_mae.param_groups[0]['lr'],
            })

        train_loss /= train_count

        # Evaluation: Show reconstruction quality on test set
        eval_loss = evaluate_mae_reconstruction(
            evan, mae_decoder, test_loader_full, device,
            bands_target, patch_size, args.mae_mask_ratio, target_modality
        )
        scheduler_mae.step(train_loss)

        # Log epoch-level metrics
        wandb.log({
            'phase2a_mae/train_loss_epoch': train_loss,
            'phase2a_mae/eval_loss': eval_loss,
            'phase2a_mae/epoch': epoch + 1,
        })

        print(f"\nMAE Epoch {epoch+1}/{args.stage1_ssl_epochs}:")
        print(f"  Train reconstruction loss: {train_loss:.4f}")
        print(f"  Test reconstruction loss:  {eval_loss:.4f}\n")

    print("\n=== Phase 2a (MAE SSL) complete ===")
    return mae_decoder  # Return for potential analysis

def train_pseudo_supervised(model,unlabeled_train_loader,labeled_train_loader,test_loader,device,args,modality_bands_dict,student_mod,teacher_mod="rgb"):
    """
    Train classifiers using pseudo-supervision from hallucinated CLS tokens.

    Monomodal: Learn rgb_cls_mono -> student_cls_mono, train student classifier
    Multimodal: Learn rgb_cls_mono -> rgb_cls_mm and rgb_cls_mono -> student_cls_mm,
                train ensemble classifier on hallucinated multimodal CLS tokens
    """
    model.freeze_all()
    model.eval()
    evan = model.evan
    is_multimodal = args.objective == "multimodal"

    # Create projector(s)
    def make_projector():
        return nn.Sequential(
            nn.Linear(evan.embed_dim, evan.embed_dim * 4),
            nn.GELU(),
            nn.Linear(evan.embed_dim * 4, evan.embed_dim)
        ).to(device)

    if is_multimodal:
        # Two projectors: rgb_cls_mono -> rgb_cls_mm, rgb_cls_mono -> student_cls_mm
        projector_rgb = make_projector()
        projector_student = make_projector()
        projectors = nn.ModuleDict({teacher_mod: projector_rgb, student_mod: projector_student})
        params = list(projectors.parameters())
    else:
        # Single projector: rgb_cls_mono -> student_cls_mono
        projector_student = make_projector()
        params = list(projector_student.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.stage3_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)
    criteria = nn.MSELoss()

    # Stage 1: Train projector(s) on unlabeled data
    for epoch in range(args.stage3_epochs):
        train_loss = 0.0
        pbar = tqdm(unlabeled_train_loader, desc=f"ClsToken Projector Epoch {epoch+1}/{args.stage3_epochs}")
        for batch in pbar:
            multimodal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=(student_mod, teacher_mod)
            )
            multimodal_input = {k: v.to(device) for k, v in multimodal_input.items()}

            # Get monomodal rgb CLS token (source for projection)
            teacher_mono_input = {teacher_mod: multimodal_input[teacher_mod]}
            rgb_cls_mono = evan.forward_features(teacher_mono_input)[teacher_mod]["x_norm_clstoken"]

            if is_multimodal:
                # Get multimodal CLS tokens (targets)
                mm_features = evan.forward_features(multimodal_input)
                rgb_cls_mm = mm_features[teacher_mod]["x_norm_clstoken"]
                student_cls_mm = mm_features[student_mod]["x_norm_clstoken"]

                # Project and compute loss for both
                projected_rgb = projectors[teacher_mod](rgb_cls_mono)
                projected_student = projectors[student_mod](rgb_cls_mono)
                loss = criteria(projected_rgb, rgb_cls_mm) + criteria(projected_student, student_cls_mm)
            else:
                # Get monomodal student CLS token (target)
                student_mono_input = {student_mod: multimodal_input[student_mod]}
                student_cls_mono = evan.forward_features(student_mono_input)[student_mod]["x_norm_clstoken"]

                projected_student = projector_student(rgb_cls_mono)
                loss = criteria(projected_student, student_cls_mono)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=5)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm:.4f}'})

        avg_train_loss = train_loss / len(unlabeled_train_loader)
        scheduler.step(avg_train_loss)
        print(f"  Projector Epoch {epoch+1}: avg loss = {avg_train_loss:.4f}")

    # Freeze projector(s)
    if is_multimodal:
        projectors.eval()
        for p in projectors.parameters():
            p.requires_grad = False
    else:
        projector_student.eval()
        for p in projector_student.parameters():
            p.requires_grad = False

    # Stage 2: Train classifier(s) on labeled data using hallucinated CLS tokens
    ce_criteria = nn.CrossEntropyLoss()
    use_ensemble = args.classifier_strategy == "ensemble"

    if is_multimodal:
        if use_ensemble:
            # Train ensemble classifiers for both modalities
            classifier_rgb = copy.deepcopy(model.modality_classifiers[teacher_mod]).to(device)
            classifier_student = copy.deepcopy(model.modality_classifiers[student_mod]).to(device)
            classifiers = nn.ModuleDict({teacher_mod: classifier_rgb, student_mod: classifier_student})

            # Setup eval model with new classifiers
            model_eval = copy.deepcopy(model)
            model_eval.modality_classifiers[teacher_mod] = classifier_rgb
            model_eval.modality_classifiers[student_mod] = classifier_student
            model_eval.eval()
            model_eval.freeze_all()

            for clf in classifiers.values():
                clf.train()
                for p in clf.parameters():
                    p.requires_grad = True

            params = list(classifiers.parameters())
        else:
            # Mean strategy: train a single shared classifier on averaged CLS tokens
            classifier_shared = copy.deepcopy(model.classifier).to(device)

            # Setup eval model with shared classifier
            model_eval = copy.deepcopy(model)
            model_eval.classifier = classifier_shared
            model_eval.eval()
            model_eval.freeze_all()

            classifier_shared.train()
            for p in classifier_shared.parameters():
                p.requires_grad = True

            params = list(classifier_shared.parameters())
    else:
        classifier_student = copy.deepcopy(model.modality_classifiers['rgb']).to(device)
        model_eval = copy.deepcopy(model)
        model_eval.modality_classifiers[student_mod] = classifier_student
        model_eval.eval()
        model_eval.freeze_all()
        classifier_student.train()
        for p in classifier_student.parameters():
            p.requires_grad = True
        params = list(classifier_student.parameters())

    optimizer = torch.optim.AdamW(params, lr=args.stage3_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    for epoch in range(args.stage3_epochs):
        train_loss = 0.0
        pbar = tqdm(labeled_train_loader, desc=f"Pseudo Supervision Epoch {epoch+1}/{args.stage3_epochs}")
        for batch in pbar:
            labels = batch["label"].to(device)

            # Only have RGB in labeled data
            teacher_mod_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=(teacher_mod,)
            )
            teacher_mod_input = {k: v.to(device) for k, v in teacher_mod_input.items()}

            # Get monomodal rgb CLS token
            rgb_cls_mono = evan.forward_features(teacher_mod_input)[teacher_mod]["x_norm_clstoken"]

            if is_multimodal:
                # Hallucinate multimodal CLS tokens
                projected_rgb_cls = projectors[teacher_mod](rgb_cls_mono).detach()
                projected_student_cls = projectors[student_mod](rgb_cls_mono).detach()

                if use_ensemble:
                    # Train both classifiers (ensemble)
                    pred_rgb = classifiers[teacher_mod](projected_rgb_cls)
                    pred_student = classifiers[student_mod](projected_student_cls)
                    loss = ce_criteria(pred_rgb, labels) + ce_criteria(pred_student, labels)
                else:
                    # Train shared classifier on mean of CLS tokens
                    mean_cls = (projected_rgb_cls + projected_student_cls) / 2
                    pred = classifier_shared(mean_cls)
                    loss = ce_criteria(pred, labels)
            else:
                projected_student_cls = projector_student(rgb_cls_mono).detach()
                pred_student = classifier_student(projected_student_cls)
                loss = ce_criteria(pred_student, labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=5)
            optimizer.step()
            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm:.4f}'})

        avg_train_loss = train_loss / len(labeled_train_loader)
        scheduler.step(avg_train_loss)

        # Evaluate on test set (with actual multimodal data)
        if is_multimodal:
            _, test_acc = evaluate(
                model_eval, test_loader, ce_criteria, device,
                modality_bands_dict, modalities_to_use=(student_mod, teacher_mod)
            )
            print(f"  Test Acc (multimodal {teacher_mod}+{student_mod}, {args.classifier_strategy}): {test_acc:.2f}%")
        else:
            _, test_acc = evaluate(
                model_eval, test_loader, ce_criteria, device,
                modality_bands_dict, modalities_to_use=(student_mod,)
            )
            print(f"  Test Acc ({student_mod}): {test_acc:.2f}%")

    # Update model with trained classifiers
    if is_multimodal:
        if use_ensemble:
            model.modality_classifiers[teacher_mod] = classifier_rgb
            model.modality_classifiers[student_mod] = classifier_student
        else:
            model.classifier = classifier_shared
    else:
        model.modality_classifiers[student_mod] = classifier_student

    return test_acc

def train_self_distillation(model,train_loader,test_loader,device,args,modality_bands_dict,student_mod,teacher_mod="rgb",teacher_model=None):
    # Use provided teacher model or create from student
    if teacher_model is None:
        teacher_model = copy.deepcopy(model)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    model.freeze_all()
    model.set_requires_grad("all", classifier=True)
    trainable_in_evan = sum(p.numel() for p in model.evan.parameters() if p.requires_grad)
    trainable_in_classifier = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert trainable_in_evan==0
    print(f"\nTrainable parameters: {trainable_in_classifier}")

    # Optimizer
    params = (list(filter(lambda p: p.requires_grad, model.parameters())))
    optimizer = torch.optim.AdamW(params, lr=args.stage3_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    # Loss function for knowledge distillation
    criteria = nn.KLDivLoss(reduction='batchmean')
    ce_criteria = nn.CrossEntropyLoss()

    for epoch in range(args.stage3_epochs):
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Classifier Distillation Epoch {epoch+1}/{args.stage3_epochs}")
        for batch in pbar:
            # Step 1: Create input tensors for all modalities
            multimodal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=(student_mod,teacher_mod)
            )
            multimodal_input = {k: v.to(device) for k, v in multimodal_input.items()}

            teacher_input = {teacher_mod: multimodal_input[teacher_mod]}
            with torch.no_grad():
                teacher_output = teacher_model(teacher_input) / args.temperature
            if args.objective=="multimodal":
                student_input=multimodal_input
            else:
                student_input={student_mod: multimodal_input[student_mod]}
            student_output = model(student_input) / args.temperature
            # KLDivLoss expects log-probabilities as input, probabilities as target
            loss = criteria(
                F.log_softmax(student_output, dim=-1),
                F.softmax(teacher_output, dim=-1)
            )
            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))
            optimizer.step()

            train_loss += loss.item()

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad': f'{grad_norm:.4f}'
            })

        avg_train_loss = train_loss / len(train_loader)
        scheduler.step(avg_train_loss)

        # Evaluate test accuracy on student modality
        if args.objective=="multimodal":
            test_loss, test_acc = evaluate(
                model, test_loader, ce_criteria, device,
                modality_bands_dict, modalities_to_use=(student_mod,teacher_mod)
            )
        else:
            test_loss, test_acc = evaluate(
                model, test_loader, ce_criteria, device,
                modality_bands_dict, modalities_to_use=(student_mod,)
            )

        # Evaluate distillation loss on test set
        model.eval()
        test_distill_loss = 0.0
        with torch.no_grad():
            for batch in test_loader:
                multimodal_input = create_multimodal_batch(
                    batch, modality_bands_dict=modality_bands_dict,
                    modalities=(student_mod, teacher_mod)
                )
                multimodal_input = {k: v.to(device) for k, v in multimodal_input.items()}

                teacher_input = {teacher_mod: multimodal_input[teacher_mod]}
                teacher_output = teacher_model(teacher_input) / args.temperature
                if args.objective=="multimodal":
                    student_input=multimodal_input
                else:
                    student_input={student_mod: multimodal_input[student_mod]}
                student_output = model(student_input) / args.temperature

                loss = criteria(
                    F.log_softmax(student_output, dim=-1),
                    F.softmax(teacher_output, dim=-1)
                )
                test_distill_loss += loss.item()
        test_distill_loss /= len(test_loader)

        print(f"\nDistillation Epoch {epoch+1}/{args.stage3_epochs}:")
        print(f"  Train KL loss: {avg_train_loss:.4f}, Test KL loss: {test_distill_loss:.4f}")
        print(f"  Test Acc: {test_acc:.2f}%")

        wandb.log({
            'phase3_classifier/train_loss': avg_train_loss,
            'phase3_classifier/test_distill_loss': test_distill_loss,
            'phase3_classifier/test_loss': test_loss,
            'phase3_classifier/test_acc': test_acc,
            'phase3_classifier/grad_norm': grad_norm,
            'phase3_classifier/epoch': epoch + 1,
            'phase3_classifier/lr': optimizer.param_groups[0]['lr'],
        })
    return test_acc

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


def train_hybrid_phase(model, train_loader, test_loader_full, device, args, bands_target, target_modality, modality_bands_dict):
    """
    Phase 1: Hybrid training combining MAE reconstruction + RGB latent distillation.

    Uses a hybrid loss combining:
    - MAE reconstruction loss for target_modality (reconstruct raw pixels)
    - Latent distillation loss from RGB (match frozen RGB features)

    Trains:
    - Target modality patch embedder
    - Target modality cls/reg tokens
    - Target modality modality-specific LoRA

    Frozen:
    - Everything else (DINO-initialized backbone, RGB components, fusion LoRA, classifier)
    """
    print("\n" + "="*70)
    print(f"=== PHASE 1: Hybrid Training (MAE + RGB Distillation) for {target_modality} ===")
    print("="*70)

    evan = model.evan
    patch_size = evan.patch_size
    num_target_channels = len(bands_target)
    num_patches = (evan.img_size // patch_size) ** 2

    # Create MAE decoder for pixel reconstruction
    mae_decoder = FullSequenceMAEDecoder(
        embed_dim=evan.embed_dim,
        num_channels=num_target_channels,
        patch_size=patch_size,
        decoder_depth=1,
        ffn_factor=2
    ).to(device)
    print(f"  Initialized FullSequenceMAEDecoder for {target_modality}, num_channels={num_target_channels}")

    # Create latent projector for RGB feature matching
    latent_projector = nn.Sequential(
        nn.Linear(evan.embed_dim, evan.embed_dim),
        nn.GELU(),
        nn.Linear(evan.embed_dim, evan.embed_dim),
    ).to(device)
    print(f"  Initialized Latent Projector for RGB distillation")

    # Freeze everything, then unfreeze target modality components
    model.freeze_all()
    model.set_requires_grad(target_modality, patch_embedders=True, clsreg=True, msla=True, mfla=False, classifier=False)
    print(f"  Unfroze: {target_modality} patch embedder, cls/reg tokens, modality-specific layer adaptors")

    # Count parameters
    trainable_in_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    total_in_evan = sum(p.numel() for p in evan.parameters())
    trainable_decoder = sum(p.numel() for p in mae_decoder.parameters())
    trainable_projector = sum(p.numel() for p in latent_projector.parameters())
    trainable_total = trainable_in_evan + trainable_decoder + trainable_projector
    print(f"\nTrainable parameters for Hybrid Training: {trainable_total}")
    print(f"    EVAN: {trainable_in_evan} ({100*trainable_in_evan/total_in_evan:.2f}%)")
    print(f"    MAE decoder: {trainable_decoder}")
    print(f"    Latent projector: {trainable_projector}")

    # Optimizer
    params = (
        list(filter(lambda p: p.requires_grad, evan.parameters())) +
        list(mae_decoder.parameters()) +
        list(latent_projector.parameters())
    )
    optimizer = torch.optim.AdamW(params, lr=args.ssl_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    # Loss function
    mse_fn = nn.MSELoss(reduction='none')

    # Pre-compute band indices for target modality
    target_modality_indices = get_band_indices(bands_target)

    # Training loop
    global_step = 0
    for epoch in range(args.stage1_ssl_epochs):
        evan.train()
        mae_decoder.train()
        latent_projector.train()
        train_loss = 0.0
        train_mae_loss = 0.0
        train_latent_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Hybrid Epoch {epoch+1}/{args.stage1_ssl_epochs}")
        for batch in pbar:
            # Create input for both RGB and target modality
            evan_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=('rgb', target_modality)
            )
            evan_input = {k: v.to(device) for k, v in evan_input.items()}
            B = evan_input[target_modality].shape[0]

            # Get modality-specific features for both modalities
            mod_specific = evan.forward_modality_specific_features(evan_input)
            # {mod: [B, 1+n_storage+num_patches, embed_dim]}

            # RGB features are frozen teacher targets
            rgb_features = mod_specific['rgb'].detach()  # [B, 1+n_storage+num_patches, embed_dim]
            target_features = mod_specific[target_modality]  # [B, 1+n_storage+num_patches, embed_dim]

            # Extract patch tokens for MAE
            n_prefix = evan.n_storage_tokens + 1
            target_patches = target_features[:, n_prefix:, :]  # [B, num_patches, embed_dim]

            # Random masking for MAE loss (loss computed only on masked patches)
            len_keep = int(num_patches * (1 - args.mae_mask_ratio))
            noise = torch.rand(B, num_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            mask = torch.ones(B, num_patches, device=device, dtype=torch.bool)
            mask.scatter_(1, ids_shuffle[:, :len_keep], False)

            # MAE Loss: Predict pixels from features
            pred_pixels = mae_decoder(target_patches)  # [B, num_patches, patch_size^2 * C]

            # Get target patches (raw pixels)
            target_img = evan_input[target_modality]  # [B, C, H, W]
            target_pixel_patches = patchify(target_img, patch_size)  # [B, num_patches, patch_size^2 * C]

            # MAE loss on masked patches only
            mask_float = mask.float()
            mae_loss = mae_reconstruction_loss(pred_pixels, target_pixel_patches, mask_float)

            # Latent distillation loss: Project target features to match RGB features (CLS + patches only, no storage tokens)
            # Extract CLS token (index 0) and patch tokens (after storage tokens)
            target_cls = target_features[:, 0:1, :]  # [B, 1, embed_dim]
            target_patch_tokens = target_features[:, n_prefix:, :]  # [B, num_patches, embed_dim]
            target_for_distill = torch.cat([target_cls, target_patch_tokens], dim=1)  # [B, 1+num_patches, embed_dim]

            rgb_cls = rgb_features[:, 0:1, :]  # [B, 1, embed_dim]
            rgb_patch_tokens = rgb_features[:, n_prefix:, :]  # [B, num_patches, embed_dim]
            rgb_for_distill = torch.cat([rgb_cls, rgb_patch_tokens], dim=1)  # [B, 1+num_patches, embed_dim]

            projected_features = latent_projector(target_for_distill)  # [B, 1+num_patches, embed_dim]
            latent_loss = mse_fn(projected_features, rgb_for_distill).mean()

            # Combined loss
            total_loss = mae_loss + latent_loss

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=float('inf'))
            optimizer.step()

            train_loss += total_loss.item()
            train_mae_loss += mae_loss.item()
            train_latent_loss += latent_loss.item()
            train_count += 1
            global_step += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'mae': f'{mae_loss.item():.4f}',
                'latent': f'{latent_loss.item():.4f}',
                'grad': f'{grad_norm:.4f}'
            })

            # Log to wandb every step
            wandb.log({
                'phase1_hybrid/train_loss': total_loss.item(),
                'phase1_hybrid/mae_loss': mae_loss.item(),
                'phase1_hybrid/latent_loss': latent_loss.item(),
                'phase1_hybrid/grad_norm': grad_norm.item(),
                'phase1_hybrid/epoch': epoch + 1,
                'phase1_hybrid/step': global_step,
                'phase1_hybrid/lr': optimizer.param_groups[0]['lr'],
            })

        # Epoch summary
        train_loss /= train_count
        train_mae_loss /= train_count
        train_latent_loss /= train_count

        scheduler.step(train_loss)

        # Evaluation on test set
        evan.eval()
        mae_decoder.eval()
        latent_projector.eval()
        eval_loss = 0.0
        eval_mae_loss = 0.0
        eval_latent_loss = 0.0
        eval_count = 0

        with torch.no_grad():
            for batch in test_loader_full:
                evan_input = create_multimodal_batch(
                    batch, modality_bands_dict=modality_bands_dict,
                    modalities=('rgb', target_modality)
                )
                evan_input = {k: v.to(device) for k, v in evan_input.items()}
                B = evan_input[target_modality].shape[0]

                mod_specific = evan.forward_modality_specific_features(evan_input)
                rgb_features = mod_specific['rgb']
                target_features = mod_specific[target_modality]

                n_prefix = evan.n_storage_tokens + 1
                target_patches = target_features[:, n_prefix:, :]

                # Random masking
                len_keep = int(num_patches * (1 - args.mae_mask_ratio))
                noise = torch.rand(B, num_patches, device=device)
                ids_shuffle = torch.argsort(noise, dim=1)
                mask = torch.ones(B, num_patches, device=device, dtype=torch.bool)
                mask.scatter_(1, ids_shuffle[:, :len_keep], False)

                # MAE loss
                pred_pixels = mae_decoder(target_patches)
                target_img = evan_input[target_modality]
                target_pixel_patches = patchify(target_img, patch_size)
                mask_float = mask.float()
                mae_loss = mae_reconstruction_loss(pred_pixels, target_pixel_patches, mask_float)

                # Latent loss (CLS + patches only, no storage tokens)
                target_cls = target_features[:, 0:1, :]
                target_patch_tokens = target_features[:, n_prefix:, :]
                target_for_distill = torch.cat([target_cls, target_patch_tokens], dim=1)

                rgb_cls = rgb_features[:, 0:1, :]
                rgb_patch_tokens = rgb_features[:, n_prefix:, :]
                rgb_for_distill = torch.cat([rgb_cls, rgb_patch_tokens], dim=1)

                projected_features = latent_projector(target_for_distill)
                latent_loss = mse_fn(projected_features, rgb_for_distill).mean()

                eval_loss += (mae_loss.item() + latent_loss.item())
                eval_mae_loss += mae_loss.item()
                eval_latent_loss += latent_loss.item()
                eval_count += 1

        eval_loss /= eval_count
        eval_mae_loss /= eval_count
        eval_latent_loss /= eval_count

        # Log epoch-level metrics
        wandb.log({
            'phase1_hybrid/train_loss_epoch': train_loss,
            'phase1_hybrid/mae_loss_epoch': train_mae_loss,
            'phase1_hybrid/latent_loss_epoch': train_latent_loss,
            'phase1_hybrid/eval_loss_epoch': eval_loss,
            'phase1_hybrid/eval_mae_loss_epoch': eval_mae_loss,
            'phase1_hybrid/eval_latent_loss_epoch': eval_latent_loss,
            'phase1_hybrid/epoch': epoch + 1,
        })

        print(f"\nHybrid Epoch {epoch+1}/{args.stage1_ssl_epochs}:")
        print(f"  Train - Total: {train_loss:.4f}, MAE: {train_mae_loss:.4f}, Latent: {train_latent_loss:.4f}")
        print(f"  Eval  - Total: {eval_loss:.4f}, MAE: {eval_mae_loss:.4f}, Latent: {eval_latent_loss:.4f}")

    print("\n=== Phase 1 (Hybrid Training) complete ===")
    return mae_decoder, latent_projector


def evaluate_rgb_distillation(evan, projector, dataloader, device, target_modality, modality_bands_dict):
    """Evaluate RGB distillation loss on test set."""
    evan.eval()
    projector.eval()
    total_loss = 0.0
    count = 0
    mse_loss_fn = nn.MSELoss()

    with torch.no_grad():
        for batch in dataloader:
            evan_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict, modalities=('rgb', target_modality)
            )
            evan_input = {k: v.to(device) for k, v in evan_input.items()}

            outputs = evan.forward_modality_specific_features(evan_input)

            rgb_features = outputs['rgb']
            target_features = outputs[target_modality]
            projected_features = projector(target_features)

            loss = mse_loss_fn(projected_features, rgb_features)
            total_loss += loss.item()
            count += 1

    return total_loss / count


def train_distillrgb_phase(model, train_loader, test_loader_full, device, args, bands_target, target_modality, modality_bands_dict):
    """
    Phase 1: distill rgb training for target_modality components.

    Trains:
    - Target modality patch embedder
    - Target modality modality-specific LoRA

    Frozen:
    - Everything else (DINO-initialized backbone, RGB components, fusion LoRA, classifier)
    """
    print("\n" + "="*70)
    print(f"=== PHASE 1a: Feature Distillation Training for {target_modality} ===")
    print("="*70)
    evan = model.evan

    # Create projector: newmod features -> RGB feature space (nonlinear)
    projector = nn.Sequential(
        nn.Linear(evan.embed_dim, evan.embed_dim),
        nn.GELU(),
        nn.Linear(evan.embed_dim, evan.embed_dim),
    ).to(device)

    model.freeze_all()  # Freeze everything
    model.set_requires_grad(target_modality, patch_embedders=True, clsreg=True, msla=True, mfla=False, classifier=False)
    print(f"  Unfroze: patch embedder, clsreg tokens, modality-specific layers adaptors")

    trainable_in_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    total_in_evan = sum(p.numel() for p in evan.parameters())
    trainable_projector = sum(p.numel() for p in projector.parameters())
    trainable_total = trainable_in_evan + trainable_projector
    print(f"\nTrainable parameters for Feature Distillation: {trainable_total}\n    {trainable_in_evan=} ({100*trainable_in_evan/total_in_evan:.2f}% of EVAN) and {trainable_projector=}")

    # Optimizer - trainable EVAN params + projector
    feadist_params = list(filter(lambda p: p.requires_grad, evan.parameters())) + list(projector.parameters())
    optimizer = torch.optim.AdamW(feadist_params, lr=args.ssl_lr)
    scheduler = ReduceLROnPlateau(optimizer,factor=0.5,patience=1,min_lr=1e-6)
    # Training loop
    global_step = 0
    mse_loss_fn = nn.MSELoss()
    for epoch in range(args.stage1_ssl_epochs):
        evan.train()
        projector.train()
        train_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"RGB Distiller Epoch {epoch+1}/{args.stage1_ssl_epochs}")
        for batch in pbar:
            evan_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict, modalities=('rgb', target_modality)
            )
            evan_input = {k: v.to(device) for k, v in evan_input.items()}

            optimizer.zero_grad()
            outputs = evan.forward_modality_specific_features(evan_input)

            # Project target modality features to RGB feature space
            rgb_features = outputs['rgb'].detach()  # Teacher (frozen)
            target_features = outputs[target_modality]
            projected_features = projector(target_features)

            loss = mse_loss_fn(projected_features, rgb_features)

            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(feadist_params, max_norm=float('inf'))
            optimizer.step()
            train_loss += loss.item()
            train_count += 1
            global_step += 1
            pbar.set_postfix({'l2 loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'})

            # Log to wandb every step
            wandb.log({
                'phase1_distillrgb/train_loss': loss.item(),
                'phase1_distillrgb/grad_norm': grad_norm.item(),
                'phase1_distillrgb/epoch': epoch + 1,
                'phase1_distillrgb/step': global_step,
                'phase1_distillrgb/lr': optimizer.param_groups[0]['lr'],
            })

        train_loss /= train_count

        # Evaluation: Show distillation quality on test set
        eval_loss = evaluate_rgb_distillation(
            evan, projector, test_loader_full, device,
            target_modality, modality_bands_dict
        )
        scheduler.step(train_loss)
        # Log epoch-level metrics
        wandb.log({
            'phase1_distillrgb/train_loss_epoch': train_loss,
            'phase1_distillrgb/eval_loss': eval_loss,
            'phase1_distillrgb/epoch': epoch + 1,
        })

        print(f"RGB Distillation Epoch {epoch+1}/{args.stage1_ssl_epochs}:")
        print(f"  Train l2 loss: {train_loss:.4f}")
        print(f"  Test l2 loss:  {eval_loss:.4f}\n")

    print("\n=== Phase 1a (RGB Distillation) complete ===")
    return projector  # Return for potential analysis

