"""Train EVAN on EuroSAT RGB, test on RGB and RGB+NewMod."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import EuroSAT
from tqdm import tqdm
from einops import rearrange
from evan_main import evan_small, evan_base, evan_large
from eurosat_data_utils import (
    create_multimodal_batch,
    get_default_transform,
    normalize_bands,
    get_band_indices,
    DictTransform,
    BAND_MINS,
    BAND_MAXS,
    ALL_BAND_NAMES,
    get_modality_bands_dict
)
from torchvision import transforms
import logging
import os
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


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

    def forward(self, x):
        """
        Args:
            x: [B, num_patches, embed_dim]

        Returns:
            reconstructed patches: [B, num_patches, patch_size^2 * channels]
        """
        x = self.decoder(x)
        x = self.decoder_pred(x)
        return x


class EVANClassifier(nn.Module):
    """Classifier head on top of EVAN for EuroSAT."""

    def __init__(self, evan_model, num_classes=10, fusion_strategy='mean', factor=4):
        super().__init__()
        self.evan = evan_model
        self.fusion_strategy = fusion_strategy
        self.num_classes = num_classes
        self.factor = factor
        embed_dim = self.evan.embed_dim
        hidden_dim = embed_dim * factor

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
        hidden_dim = embed_dim * self.factor

        classifier = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_classes)
        )

        # Move to same device as model
        if next(self.parameters(), None) is not None:
            device = next(self.parameters()).device
            classifier = classifier.to(device)

        self.modality_classifiers[modality_key] = classifier
        print(f"  Created new classifier for modality: {modality_key}")

    def forward(self, x, train_modality=None):
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
                cls_tokens.append(features_dict[modality]['x_norm_clstoken'])

            # Average CLS tokens
            fused = torch.stack(cls_tokens).mean(dim=0)

            # Classify
            logits = self.classifier(fused)

        elif self.fusion_strategy == 'ensemble':
            # In training mode with train_modality specified, only use that modality
            if self.training and train_modality is not None:
                # Only compute logits for the training modality
                if train_modality not in self.modality_classifiers:
                    self._instantiate_modality_classifier(train_modality)

                cls_token = features_dict[train_modality]['x_norm_clstoken']
                logits = self.modality_classifiers[train_modality](cls_token)
            else:
                # Inference mode or no train_modality specified: ensemble all modalities
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


# Note: create_multimodal_batch is now imported from eurosat_data_utils


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

    # Extract bands_rgb and bands_newmod for create_multimodal_batch compatibility
    bands_rgb = modality_bands_dict.get('rgb')
    # Get the first non-rgb modality for bands_newmod (for backwards compatibility with create_multimodal_batch)
    non_rgb_keys = [k for k in modality_bands_dict.keys() if k != 'rgb']
    bands_newmod = modality_bands_dict[non_rgb_keys[0]] if non_rgb_keys else ()

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)

            # Create multi-modal input with specified modalities
            modal_input = create_multimodal_batch(
                batch, bands_rgb, bands_newmod, BAND_MINS, BAND_MAXS, modalities=modalities_to_use
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
    Based on MAESTRO's mask_seq (mae.py:228-264).

    Args:
        x: Patch embeddings [B, num_patches, embed_dim]
        mask_ratio: Fraction of patches to mask (default 0.75)

    Returns:
        x_masked: Unmasked patches only [B, num_unmasked, embed_dim]
        mask: Boolean mask [B, num_patches] where True = masked
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
    """
    MSE loss on masked patches only.
    Based on MAESTRO's compute_loss_rec (model.py:195-247).

    Args:
        pred: Predicted pixels [B, num_patches, patch_size^2 * channels]
        target: Target pixels [B, num_patches, patch_size^2 * channels]
        mask: Boolean mask [B, num_patches] where True = compute loss

    Returns:
        loss: Mean squared error on masked patches
    """
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


def unpatchify(patches, patch_size, channels, img_size):
    """
    Convert patches back to image for visualization.

    Args:
        patches: [B, num_patches, patch_size^2 * C]
        patch_size: Size of each patch
        channels: Number of channels
        img_size: Image size (H = W)

    Returns:
        imgs: [B, C, H, W]
    """
    B = patches.shape[0]
    num_patches_h = num_patches_w = img_size // patch_size

    patches = patches.reshape(B, num_patches_h, num_patches_w, patch_size, patch_size, channels)
    imgs = patches.permute(0, 5, 1, 3, 2, 4).reshape(B, channels, img_size, img_size)
    return imgs


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

            x_masked, mask, ids_restore = random_mask_patches(patch_embeddings, mask_ratio)
            pred_masked = mae_decoder(x_masked)

            # Restore full predictions
            B, L_unmasked, D_pred = pred_masked.shape
            L_total = patch_embeddings.shape[1]
            pred_full = torch.zeros(B, L_total, D_pred, device=device)
            ids_keep = ids_restore.gather(1, torch.arange(L_unmasked, device=device).unsqueeze(0).expand(B, -1))
            pred_full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D_pred), pred_masked)

            loss = mae_reconstruction_loss(pred_full, target_patches, mask)
            total_loss += loss.item()
            count += 1

    return total_loss / count


def train_mae_phase(model, evan, train_loader, test_loader_full, device, args,
                    bands_target, target_modality):
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

    # Unfreeze target modality patch embedder and modality-specific LoRAs for MAE training
    for param in evan.patch_embedders[target_modality].parameters():
        param.requires_grad = True
    for param in evan.modality_specific_lora_adaptors[target_modality].parameters():
        param.requires_grad = True
    print(f"  Unfroze: {target_modality} patch embedder")
    print(f"  Unfroze: {target_modality} modality-specific LoRAs")

    trainable_params_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    trainable_params_decoder = sum(p.numel() for p in mae_decoder.parameters())
    trainable_total = trainable_params_evan+trainable_params_decoder
    print(f"\nTrainable parameters for MAE: {trainable_total}\n    {trainable_params_evan=} and {trainable_params_decoder=}")

    # Optimizer for MAE phase - collect all trainable parameters
    mae_params = [p for p in model.parameters() if p.requires_grad]

    optimizer_mae = torch.optim.AdamW(mae_params, lr=args.mae_lr)

    # Pre-compute target modality indices
    target_modality_indices = get_band_indices(bands_target)

    # Training loop
    for epoch in range(args.stage2_epochs):  # MAE epochs = stage2_epochs
        model.train()
        mae_decoder.train()
        train_loss = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"MAE Epoch {epoch+1}/{args.stage2_epochs}")
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

            # Decoder processes only unmasked patches
            pred_masked = mae_decoder(x_masked)  # [B, num_unmasked, patch_size^2 * C]

            # Restore full sequence with mask tokens
            B, L_unmasked, D_pred = pred_masked.shape
            L_total = patch_embeddings.shape[1]
            pred_full = torch.zeros(B, L_total, D_pred, device=device)

            # Scatter unmasked predictions back to original positions
            ids_keep = ids_restore.gather(1, torch.arange(L_unmasked, device=device).unsqueeze(0).expand(B, -1))
            pred_full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, D_pred), pred_masked)

            # Compute loss on masked patches only
            loss = mae_reconstruction_loss(pred_full, target_patches, mask)

            optimizer_mae.zero_grad()
            loss.backward()
            optimizer_mae.step()

            train_loss += loss.item()
            train_count += 1
            pbar.set_postfix({'mae_loss': f'{loss.item():.4f}'})

        train_loss /= train_count

        # Evaluation: Show reconstruction quality on test set
        eval_loss = evaluate_mae_reconstruction(
            model, evan, mae_decoder, test_loader_full, device,
            bands_target, patch_size, args.mae_mask_ratio, target_modality
        )

        print(f"\nMAE Epoch {epoch+1}/{args.stage2_epochs}:")
        print(f"  Train reconstruction loss: {train_loss:.4f}")
        print(f"  Test reconstruction loss:  {eval_loss:.4f}\n")

    print("\n=== Phase 2a (MAE SSL) complete ===")
    return mae_decoder  # Return for potential analysis


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train EVAN on EuroSAT RGB, test on RGB and RGB+NewModality')
    parser.add_argument('--model', type=str, default='evan_small', choices=['evan_small', 'evan_base', 'evan_large'],
                        help='EVAN model size (default: evan_small)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--stage2', action='store_true',
                        help='Run stage 2 training after stage 1 (train on train2 with RGB+NewModality)')
    parser.add_argument('--stage2_epochs', type=int, default=None,
                        help='Number of epochs for stage 2 (default: same as --epochs)')
    parser.add_argument('--stage2_lr', type=float, default=None,
                        help='Learning rate for stage 2 (default: same as --lr)')
    parser.add_argument('--tz_fusion_time', type=int, default=3,
                        help='n modality-independent layers before fusion')
    parser.add_argument('--tz_lora_rank', type=int, default=64,
                        help='rank of lora adaptors')
    parser.add_argument('--fusion_strategy', type=str, default='mean', choices=['mean', 'ensemble'],
                        help='Fusion strategy: mean (avg CLS tokens) or ensemble (per-modality classifiers)')
    parser.add_argument('--stage2_train_method', type=str, default='supervised',
                        choices=['supervised', 'mae+supervised'],
                        help='Stage 2 training method: supervised or mae+supervised')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.75,
                        help='Mask ratio for MAE training (default: 0.75)')
    parser.add_argument('--mae_lr', type=float, default=None,
                        help='Learning rate for MAE phase (default: same as stage2_lr)')
    parser.add_argument('--new_mod_group', type=str, default='vre', choices=['vre','nir','swir'],
                        help='Learning rate for MAE phase (default: same as stage2_lr)')
    args = parser.parse_args()

    # Set stage 2 defaults
    if args.stage2_epochs is None:
        args.stage2_epochs = args.epochs
    if args.stage2_lr is None:
        args.stage2_lr = args.lr
    if args.mae_lr is None:
        args.mae_lr = args.stage2_lr

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # Band configuration (using constants from eurosat_data_utils)
    newmod = args.new_mod_group
    modality_bands_dict = get_modality_bands_dict('rgb', newmod)
    bands_rgb = modality_bands_dict['rgb']
    bands_newmod = modality_bands_dict[newmod]
    bands_full = tuple(ALL_BAND_NAMES)

    # Create datasets
    print("\n=== Creating datasets ===")

    # Resize transform (applied to all datasets)
    resize_transform = DictTransform(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))

    # Load all datasets with all bands and resize only (no normalization in dataset)
    # We'll normalize manually in the training/evaluation loops
    train_dataset_full = EuroSAT(
        root='datasets',
        split='train',
        bands=bands_full,  # Load all 13 bands
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    # Get indices for train1 split (first half of training data)
    train1_indices = load_split_indices('datasets/eurosat-train1.txt', train_dataset_full)
    train_dataset = Subset(train_dataset_full, train1_indices)
    print(f"Loaded {len(train1_indices)} samples from train1 split")

    # Test with all bands (for both single-modal and multi-modal evaluation)
    test_dataset_full = EuroSAT(
        root='datasets',
        split='test',
        bands=bands_full,  # Load all 13 bands
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    print(f"Train samples (all bands, split=train1): {len(train_dataset)}")
    print(f"Test samples (all bands): {len(test_dataset_full)}")
    print(f"Note: train2 split is reserved for future use (not used in this training)")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader_full = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create EVAN model with pretrained DINO weights
    print("\n=== Creating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
    evan = model_fn(
        # img_size=64,  # EuroSAT images are 64x64
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        n_storage_tokens=4,  # DINOv3 uses 4 register tokens
        device=device
    )
    # Note: evan_preset models automatically loads pretrained weights from facebook/dinov3 pretrained models

    # Create classifier
    model = EVANClassifier(evan, num_classes=10, fusion_strategy=args.fusion_strategy)
    model = model.to(device)

    # Freeze EVAN backbone, train only classifier(s)
    for param in evan.parameters():
        param.requires_grad = False

    # Unfreeze classifier(s) - depends on fusion strategy
    if args.fusion_strategy == 'mean':
        for param in model.classifier.parameters():
            param.requires_grad = True
        print(f"Using mean fusion: single classifier on averaged CLS tokens")
    elif args.fusion_strategy == 'ensemble':
        # Pre-instantiate RGB classifier so optimizer has parameters
        # (other modality classifiers will be created dynamically on first forward pass)
        model._instantiate_modality_classifier('rgb')
        for param in model.modality_classifiers['rgb'].parameters():
            param.requires_grad = True
        print(f"Using ensemble fusion: RGB classifier pre-instantiated, other classifiers will be created on first forward pass")

    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    num_epochs = args.epochs

    print(f"\n=== Training for {num_epochs} epochs ===")
    print(f"Strategy: Train on RGB (train1 split), test on both RGB and RGB+{args.new_mod_group}")
    print("Note: train2 split reserved for future experiments\n")

    # Training loop
    for epoch in range(num_epochs):
        # Training on RGB
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train RGB]")
        for batch in pbar:
            labels = batch['label'].to(device)

            # Normalize RGB data
            modal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict, modalities=('rgb',)
            )
            modal_input = {k: v.to(device) for k, v in modal_input.items()}

            optimizer.zero_grad()
            outputs = model(modal_input)
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

        # Evaluation on RGB only
        test_loss_rgb, test_acc_rgb = evaluate(
            model, test_loader_full, criterion, device,
            modality_bands_dict, modalities_to_use=('rgb',)
        )

        # Evaluation on RGB + new modality
        test_loss_multi, test_acc_multi = evaluate(
            model, test_loader_full, criterion, device,
            modality_bands_dict, modalities_to_use=('rgb', newmod)
        )

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}:")
        print(f"  Train (RGB):           Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"  Test (RGB only):       Loss: {test_loss_rgb:.4f}, Acc: {test_acc_rgb:.2f}%")
        print(f"  Test (RGB+{newmod}):   Loss: {test_loss_multi:.4f}, Acc: {test_acc_multi:.2f}%")
        print(f"  Multi-modal gain:      {test_acc_multi - test_acc_rgb:+.2f}%\n")

    print("\n=== Training complete ===")

    # Save checkpoint
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(checkpoint_dir, f'evan_eurosat_train1_{timestamp}.pt')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'train_acc': train_acc,
        'test_acc_rgb': test_acc_rgb,
        'test_acc_multi': test_acc_multi,
        'config': {
            'model_type': args.model,
            'num_classes': 10,
            'train_split': 'train1',
            'fusion_strategy': args.fusion_strategy,
            'bands_rgb': bands_rgb,
            'bands_full': bands_full,
            'num_epochs': num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"\n=== Stage 1 checkpoint saved to: {checkpoint_path} ===")
    print(f"Stage 1 Final metrics:")
    print(f"  Train accuracy (RGB, split=train1): {train_acc:.2f}%")
    print(f"  Test accuracy (RGB only): {test_acc_rgb:.2f}%")
    print(f"  Test accuracy (RGB+{newmod}): {test_acc_multi:.2f}%")

    # ==================== STAGE 2 TRAINING ====================
    if args.stage2:
        print("\n" + "="*70)
        print("=== STAGE 2: Training new modality components on train2 ===")
        print("="*70)

        # Prepare train2 dataset from the same full training dataset
        print("\n=== Preparing train2 dataset ===")
        train2_indices = load_split_indices('datasets/eurosat-train2.txt', train_dataset_full)
        train2_dataset = Subset(train_dataset_full, train2_indices)
        print(f"Loaded {len(train2_indices)} samples from train2 split")

        train2_loader = DataLoader(train2_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        if args.stage2_train_method == 'supervised':
            # ========== SUPERVISED TRAINING (Current Method) ==========
            print(f"\nNote: {newmod.capitalize()} components were already created during stage 1 evaluation")
            print("      Now unfreezing them for training in stage 2")

            # Freeze everything first
            for param in model.parameters():
                param.requires_grad = False

            # Unfreeze classifier(s) depending on fusion strategy
            if model.fusion_strategy == 'mean':
                for param in model.classifier.parameters():
                    param.requires_grad = True
                print("  Unfroze: Classifier (mean fusion)")
            elif model.fusion_strategy == 'ensemble':
                # In ensemble mode, only unfreeze the new modality classifier
                # RGB classifier should remain frozen from stage 1

                # Ensure new modality classifier exists (should have been created in stage 1 validation)
                if newmod not in model.modality_classifiers:
                    print(f"  Creating {newmod} classifier (was not created during stage 1 validation)")
                    model._instantiate_modality_classifier(newmod)

                # Unfreeze new modality classifier
                for param in model.modality_classifiers[newmod].parameters():
                    param.requires_grad = True
                print(f"  Unfroze: {newmod.capitalize()} classifier (ensemble mode)")

                # RGB classifier stays frozen
                if 'rgb' in model.modality_classifiers:
                    for param in model.modality_classifiers['rgb'].parameters():
                        param.requires_grad = False
                    print("  Kept frozen: RGB classifier (ensemble mode)")

            # Unfreeze new modality components
            # Patch embedder
            if newmod in evan.patch_embedders:
                for param in evan.patch_embedders[newmod].parameters():
                    param.requires_grad = True
                print(f"  Unfroze: {newmod.capitalize()} patch embedder")

            # Modality-specific LoRAs
            if newmod in evan.modality_specific_lora_adaptors:
                for param in evan.modality_specific_lora_adaptors[newmod].parameters():
                    param.requires_grad = True
                print(f"  Unfroze: {newmod.capitalize()} modality-specific LoRAs")

            # Modality encoding
            if newmod in evan.modality_encoders:
                evan.modality_encoders[newmod].requires_grad = True
                print(f"  Unfroze: {newmod.capitalize()} modality encoding")

            # Fusion LoRAs
            if newmod in evan.modality_fusion_lora_adaptors:
                for param in evan.modality_fusion_lora_adaptors[newmod].parameters():
                    param.requires_grad = True
                print(f"  Unfroze: {newmod.capitalize()} fusion LoRAs")

            # Print parameter info
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nTrainable parameters for stage 2: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

            # Create new optimizer for stage 2
            optimizer_stage2 = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.stage2_lr
            )
            num_epochs_stage2 = args.stage2_epochs

            print(f"\n=== Stage 2 Training for {num_epochs_stage2} epochs ===")
            print(f"Learning rate: {args.stage2_lr}")
            print(f"Modalities: RGB {bands_rgb} + {newmod.upper()} {bands_newmod}")
            print(f"Training: {newmod.capitalize()} patch embedder, modality-specific LoRAs, modality encoding, fusion LoRAs, classifier")
            print("Frozen: RGB components and shared DINO backbone")
            if model.fusion_strategy == 'ensemble':
                print(f"Note: In ensemble mode, training loss uses only {newmod} classifier (RGB frozen)")
                print(f"      Evaluation will ensemble both RGB and {newmod} predictions\n")
            else:
                print()

            # Stage 2 training loop
            for epoch in range(num_epochs_stage2):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                pbar = tqdm(train2_loader, desc=f"Stage 2 Epoch {epoch+1}/{num_epochs_stage2} [Train RGB+{newmod.upper()}]")
                for batch_idx, batch in enumerate(pbar):
                    labels = batch['label'].to(device)

                    # Create multi-modal input (RGB + new modality)
                    multimodal_input = create_multimodal_batch(
                        batch, modality_bands_dict=modality_bands_dict, modalities=('rgb', newmod)
                    )
                    multimodal_input = {k: v.to(device) for k, v in multimodal_input.items()}

                    optimizer_stage2.zero_grad()
                    # In ensemble mode, train only on new modality predictions
                    if model.fusion_strategy == 'ensemble':
                        outputs = model(multimodal_input, train_modality=newmod)
                    else:
                        outputs = model(multimodal_input)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer_stage2.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*train_correct/train_total:.2f}%'
                    })

                train_loss /= len(train2_loader)
                train_acc = 100. * train_correct / train_total

                # Evaluation on RGB-only, newmod-only, and RGB+newmod
                test_loss_rgb, test_acc_rgb = evaluate(
                    model, test_loader_full, criterion, device,
                    modality_bands_dict, modalities_to_use=('rgb',)
                )
                test_loss_newmod, test_acc_newmod = evaluate(
                    model, test_loader_full, criterion, device,
                    modality_bands_dict, modalities_to_use=(newmod,)
                )
                test_loss_multi, test_acc_multi = evaluate(
                    model, test_loader_full, criterion, device,
                    modality_bands_dict, modalities_to_use=('rgb', newmod)
                )

                # Print epoch results
                print(f"\nStage 2 Epoch {epoch+1}/{num_epochs_stage2}:")
                print(f"  Train (RGB+{newmod.upper()}, train2):  Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                print(f"  Test (RGB only):         Loss: {test_loss_rgb:.4f}, Acc: {test_acc_rgb:.2f}%")
                print(f"  Test ({newmod.upper()} only):          Loss: {test_loss_newmod:.4f}, Acc: {test_acc_newmod:.2f}%")
                print(f"  Test (RGB+{newmod.upper()}):           Loss: {test_loss_multi:.4f}, Acc: {test_acc_multi:.2f}%\n")

            print("\n=== Stage 2 Training complete ===")

            # Save stage 2 checkpoint
            timestamp_stage2 = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_path_stage2 = os.path.join(checkpoint_dir, f'evan_eurosat_stage2_{timestamp_stage2}.pt')

            checkpoint_stage2 = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_stage2.state_dict(),
                'epoch': num_epochs_stage2,
                'train_acc': train_acc,
                'test_acc_rgb': test_acc_rgb,
                'test_acc_newmod': test_acc_newmod,
                'test_acc_multi': test_acc_multi,
                'stage1_checkpoint': checkpoint_path,
                'config': {
                    'model_type': args.model,
                    'num_classes': 10,
                    'train_split': 'train2',
                    'fusion_strategy': args.fusion_strategy,
                    'stage2_train_method': args.stage2_train_method,
                    'bands_rgb': bands_rgb,
                    'bands_newmod': bands_newmod,
                    'newmod': newmod,
                    'num_epochs': num_epochs_stage2,
                    'batch_size': args.batch_size,
                    'learning_rate': args.stage2_lr,
                }
            }

            torch.save(checkpoint_stage2, checkpoint_path_stage2)
            print(f"\n=== Stage 2 checkpoint saved to: {checkpoint_path_stage2} ===")
            print(f"Stage 2 Final metrics:")
            print(f"  Train accuracy (RGB+{newmod.upper()}, train2): {train_acc:.2f}%")
            print(f"  Test accuracy (RGB only): {test_acc_rgb:.2f}%")
            print(f"  Test accuracy ({newmod.capitalize()} only): {test_acc_newmod:.2f}%")
            print(f"  Test accuracy (RGB+{newmod.upper()}): {test_acc_multi:.2f}%")

        elif args.stage2_train_method == 'mae+supervised':
            # ========== MAE + SUPERVISED TRAINING (Two-Phase Method) ==========

            # Phase 2a: MAE SSL training
            train_mae_phase(
                model, evan, train2_loader, test_loader_full, device, args,
                bands_newmod, newmod
            )

            # Phase 2b: Supervised fine-tuning
            print("\n" + "="*70)
            print("=== PHASE 2b: Supervised Fine-tuning ===")
            print("="*70)

            # Freeze everything first
            for param in model.parameters():
                param.requires_grad = False

            # Freeze new modality patch embedder and modality-specific LoRA (just trained in MAE)
            if newmod in evan.patch_embedders:
                for param in evan.patch_embedders[newmod].parameters():
                    param.requires_grad = False
                print(f"  Kept frozen: {newmod.capitalize()} patch embedder (trained in MAE)")

            if newmod in evan.modality_specific_lora_adaptors:
                for param in evan.modality_specific_lora_adaptors[newmod].parameters():
                    param.requires_grad = False
                print(f"  Kept frozen: {newmod.capitalize()} modality-specific LoRAs (trained in MAE)")

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

                # RGB classifier stays frozen
                if 'rgb' in model.modality_classifiers:
                    for param in model.modality_classifiers['rgb'].parameters():
                        param.requires_grad = False
                    print("  Kept frozen: RGB classifier (ensemble mode)")

            # Unfreeze new modality fusion LoRA and modality encoding
            if newmod in evan.modality_encoders:
                evan.modality_encoders[newmod].requires_grad = True
                print(f"  Unfroze: {newmod.capitalize()} modality encoding")

            if newmod in evan.modality_fusion_lora_adaptors:
                for param in evan.modality_fusion_lora_adaptors[newmod].parameters():
                    param.requires_grad = True
                print(f"  Unfroze: {newmod.capitalize()} fusion LoRAs")

            # Print parameter info
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"\nTrainable parameters for Phase 2b: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

            if trainable_params == 0:
                raise ValueError("No trainable parameters found! Check that components are being unfrozen correctly.")

            # Create optimizer for Phase 2b
            optimizer_phase2b = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.stage2_lr
            )
            num_epochs_phase2b = args.stage2_epochs

            print(f"\n=== Phase 2b Training for {num_epochs_phase2b} epochs ===")
            print(f"Learning rate: {args.stage2_lr}")
            print(f"Modalities: RGB {bands_rgb} + {newmod.upper()} {bands_newmod}")
            print(f"Training: {newmod.capitalize()} modality encoding, fusion LoRAs, classifier")
            print(f"Frozen: {newmod.upper()} patch embedder, {newmod.upper()} modality-specific LoRAs, RGB components, DINO backbone")
            print(f"Model processes multi-modal input by {model.fusion_strategy}")

            # Verify modality components exist before training
            assert newmod in evan.patch_embedders, f"{newmod} patch embedder not found!"
            assert newmod in evan.modality_encoders, f"{newmod} modality encoding not found!"
            assert newmod in evan.modality_fusion_lora_adaptors, f"{newmod} fusion LoRAs not found!"
            if model.fusion_strategy == 'ensemble':
                assert newmod in model.modality_classifiers, f"{newmod} classifier not found!"
            print(f"  Verified: All {newmod} components exist")

            # Phase 2b training loop
            for epoch in range(num_epochs_phase2b):
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                pbar = tqdm(train2_loader, desc=f"Phase 2b Epoch {epoch+1}/{num_epochs_phase2b} [Train RGB+{newmod.upper()}]")
                for batch in pbar:
                    labels = batch['label'].to(device)

                    # Create multi-modal input (RGB + new modality)
                    multimodal_input = create_multimodal_batch(
                        batch, modality_bands_dict=modality_bands_dict, modalities=('rgb', newmod)
                    )
                    multimodal_input = {k: v.to(device) for k, v in multimodal_input.items()}

                    optimizer_phase2b.zero_grad()
                    # In ensemble mode, train only on new modality predictions
                    if model.fusion_strategy == 'ensemble':
                        outputs = model(multimodal_input)  # Ensemble both in phase 2b
                    else:
                        outputs = model(multimodal_input)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer_phase2b.step()

                    train_loss += loss.item()
                    _, predicted = outputs.max(1)
                    train_total += labels.size(0)
                    train_correct += predicted.eq(labels).sum().item()

                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{100.*train_correct/train_total:.2f}%'
                    })

                train_loss /= len(train2_loader)
                train_acc = 100. * train_correct / train_total

                # Evaluation on RGB-only, newmod-only, and RGB+newmod
                test_loss_rgb, test_acc_rgb = evaluate(
                    model, test_loader_full, criterion, device,
                    modality_bands_dict, modalities_to_use=('rgb',)
                )
                test_loss_newmod, test_acc_newmod = evaluate(
                    model, test_loader_full, criterion, device,
                    modality_bands_dict, modalities_to_use=(newmod,)
                )
                test_loss_multi, test_acc_multi = evaluate(
                    model, test_loader_full, criterion, device,
                    modality_bands_dict, modalities_to_use=('rgb', newmod)
                )

                # Print epoch results
                print(f"\nPhase 2b Epoch {epoch+1}/{num_epochs_phase2b}:")
                print(f"  Train (RGB+{newmod.upper()}, train2):  Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
                print(f"  Test (RGB only):         Loss: {test_loss_rgb:.4f}, Acc: {test_acc_rgb:.2f}%")
                print(f"  Test ({newmod.upper()} only):          Loss: {test_loss_newmod:.4f}, Acc: {test_acc_newmod:.2f}%")
                print(f"  Test (RGB+{newmod.upper()}):           Loss: {test_loss_multi:.4f}, Acc: {test_acc_multi:.2f}%\n")

            print("\n=== Phase 2b (Supervised Fine-tuning) complete ===")

            # Save stage 2 checkpoint
            timestamp_stage2 = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_path_stage2 = os.path.join(checkpoint_dir, f'evan_eurosat_stage2_mae_{timestamp_stage2}.pt')

            checkpoint_stage2 = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer_phase2b.state_dict(),
                'epoch': num_epochs_phase2b,
                'train_acc': train_acc,
                'test_acc_rgb': test_acc_rgb,
                'test_acc_newmod': test_acc_newmod,
                'test_acc_multi': test_acc_multi,
                'stage1_checkpoint': checkpoint_path,
                'config': {
                    'model_type': args.model,
                    'num_classes': 10,
                    'train_split': 'train2',
                    'fusion_strategy': args.fusion_strategy,
                    'stage2_train_method': args.stage2_train_method,
                    'mae_mask_ratio': args.mae_mask_ratio,
                    'mae_lr': args.mae_lr,
                    'bands_rgb': bands_rgb,
                    'bands_newmod': bands_newmod,
                    'newmod': newmod,
                    'num_epochs': num_epochs_phase2b,
                    'batch_size': args.batch_size,
                    'learning_rate': args.stage2_lr,
                }
            }

            torch.save(checkpoint_stage2, checkpoint_path_stage2)
            print(f"\n=== Stage 2 checkpoint saved to: {checkpoint_path_stage2} ===")
            print(f"Stage 2 Final metrics (after MAE+Supervised):")
            print(f"  Train accuracy (RGB+{newmod.upper()}, train2): {train_acc:.2f}%")
            print(f"  Test accuracy (RGB only): {test_acc_rgb:.2f}%")
            print(f"  Test accuracy ({newmod.capitalize()} only): {test_acc_newmod:.2f}%")
            print(f"  Test accuracy (RGB+{newmod.upper()}): {test_acc_multi:.2f}%")

        print("\n" + "="*70)
        print("=== TWO-STAGE TRAINING COMPLETE ===")
        print("="*70)
        print(f"Stage 1 checkpoint: {checkpoint_path}")
        print(f"Stage 2 checkpoint: {checkpoint_path_stage2}")

# Examples on how to call this script:

# python -u train_evan_eurosat.py
# python -u train_evan_eurosat.py --stage2 --stage2_epochs 10 # Run both stages
# python -u train_evan_eurosat.py --stage2 --stage2_epochs 4 --stage2_train_method 'supervised' --fusion_strategy ensemble # Run fully supervised
# python -u train_evan_eurosat.py --stage2 --stage2_epochs 4 --stage2_train_method 'mae+supervised' --fusion_strategy ensemble # Run mae + supervised