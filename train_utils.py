"""Training utilities for EVAN on EuroSAT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
import wandb


def hallucinate_intermediate_features(
    source_intermediate: dict,
    source_modalities: tuple,
    target_modalities: tuple,
    intermediate_projectors: nn.ModuleDict,
) -> dict:
    """
    Hallucinate intermediate features for target modalities from source modalities.

    Uses full sequence projection (CLS + storage + patches) via transformer-based projectors.
    For each target modality, projects from all available source modalities and takes the mean.

    Args:
        source_intermediate: Dict of source modality features {mod: [B, seq_len, embed_dim]}
        source_modalities: Tuple of source modality names
        target_modalities: Tuple of target modality names to hallucinate
        intermediate_projectors: Trained sequence projectors with keys like 'rgb_to_vre'

    Returns:
        Dict of hallucinated features {mod: [B, seq_len, embed_dim]}
    """
    hallucinated = {}
    for tar_mod in target_modalities:
        projected_seqs = []
        for src_mod in source_modalities:
            if src_mod == tar_mod:
                continue
            proj_key = f"{src_mod}_to_{tar_mod}"
            src_seq = source_intermediate[src_mod]  # [B, seq_len, embed_dim]
            src_seq_norm = F.layer_norm(src_seq, [src_seq.shape[-1]])
            projected_seqs.append(intermediate_projectors[proj_key](src_seq_norm))
        # Mean of all projections
        hallucinated[tar_mod] = torch.stack(projected_seqs).mean(dim=0)
    return hallucinated


def merge_intermediate_features(real_features, hallucinated_features, real_modalities, hallucinated_modalities):
    """Merge real and hallucinated intermediate features into a single dict."""
    return {
        **{m: real_features[m] for m in real_modalities},
        **{m: hallucinated_features[m] for m in hallucinated_modalities}
    }


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
             modalities_to_use=('rgb',), pseudo_modalities=None, intermediate_projectors=None):
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
        pseudo_modalities: Optional list of modalities to hallucinate using sequence projection
        intermediate_projectors: Required if pseudo_modalities is provided; trained sequence projectors
    """
    model.eval()
    if intermediate_projectors is not None:
        intermediate_projectors.eval()
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
                outputs = model(modal_input, pseudo_modalities=pseudo_modalities, intermediate_projectors=intermediate_projectors)
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
                                   intermediate_projectors=None):
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
        intermediate_projectors: Trained sequence projectors for pseudo-modality (required if hallucinate_modality=True)

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
        assert intermediate_projectors is not None, "intermediate_projectors required when hallucinate_modality=True"
        print(f"  Using pseudo-modality inference: {modality} + hallucinated {pseudo_modalities}")

    for epoch in range(num_epochs):
        model.train()
        if intermediate_projectors is not None:
            intermediate_projectors.train()
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
                outputs = model(modal_input, pseudo_modalities=pseudo_modalities, intermediate_projectors=intermediate_projectors)
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
                pseudo_modalities=pseudo_modalities, intermediate_projectors=intermediate_projectors
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
                             train_modalities, phase_name="Training"):
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
        eval_single_modalities: If True, evaluate on RGB-only, newmod-only, and RGB+newmod separately.
                               If False (default), only evaluate on train_modalities.

    Returns:
        Tuple of (train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi)
        When eval_single_modalities=False, test_acc_rgb/newmod/multi are None (only test_acc_train is meaningful)
    """

    # Determine training modality string for logging
    train_mod_str = "+".join(m.upper() for m in train_modalities)

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

            optimizer.zero_grad()
            outputs = model(modal_input)
            loss = criterion(outputs, labels)
            loss.backward()

            # Compute gradient norm
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=4)

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

        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        scheduler.step(train_loss)

        # Evaluation on training modalities
        _, test_acc = evaluate(
            model, test_loader_full, criterion, device,
            modality_bands_dict, modalities_to_use=train_modalities
        )
        print(f"epoch {epoch+1} / {num_epochs}: {test_acc=}")

    return train_acc, test_acc


class CrossModalFusedProjector(nn.Module):
    """
    Cross-modal transformer projector operating on concatenated full sequences.

    Concatenates x_prenorm sequences from all modalities, processes through transformer
    with cross-modal attention, then splits back. No positional encoding added since
    spatial structure is already encoded in token representations from EVAN's fusion.
    """

    def __init__(self, embed_dim, modalities, num_heads=8, ffn_factor=4, num_layers=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.modalities = list(modalities)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_factor,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, fusion_output):
        """
        Args:
            fusion_output: Dict from forward_fusion_from_modality_features
                          {mod: {'x_prenorm': [B, seq_len, embed_dim], ...}}

        Returns:
            Dict of projected CLS tokens {mod: [B, embed_dim]}
        """
        seq_lens = {mod: fusion_output[mod]['x_prenorm'].shape[1] for mod in self.modalities}

        # Concatenate all modality sequences
        all_seqs = [fusion_output[mod]['x_prenorm'] for mod in self.modalities]
        concat_seq = torch.cat(all_seqs, dim=1)  # [B, total_seq_len, embed_dim]

        # Process through transformer with cross-modal attention
        projected_concat = self.transformer(concat_seq)

        # Split back and extract CLS tokens (first token of each modality's sequence)
        projected_cls = {}
        start_idx = 0
        for mod in self.modalities:
            end_idx = start_idx + seq_lens[mod]
            cls_token = projected_concat[:, start_idx, :]  # [B, embed_dim]
            projected_cls[mod] = F.layer_norm(cls_token, [self.embed_dim])
            start_idx = end_idx

        return projected_cls


def _delulu_stage1_compute_loss(
    evan, batch, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors, fused_projector, objective
):
    """Compute stage 1 loss for a single batch (used for both train and val)."""
    mse_criteria = nn.MSELoss()

    multimodal_input = create_multimodal_batch(
        batch, modality_bands_dict=modality_bands_dict,
        modalities=(*labeled_modalities, *unlabeled_modalities)
    )
    multimodal_input = {k: v.to(device) for k, v in multimodal_input.items()}

    with torch.no_grad():
        real_intermediate = evan.forward_modality_specific_features(multimodal_input)

    # Hallucinate intermediate features using full sequence projection
    hallucinated_intermediate = hallucinate_intermediate_features(
        real_intermediate, tuple(all_modalities), tuple(all_modalities),
        intermediate_projectors
    )

    # Source is always real_lab + hallucinated_unlab
    real_lab_hal_unlab = merge_intermediate_features(
        real_intermediate, hallucinated_intermediate,
        labeled_modalities, unlabeled_modalities
    )
    real_lab_hal_unlab_fusion = evan.forward_fusion_from_modality_features(real_lab_hal_unlab)

    # Target depends on objective
    if objective == "transfer":
        # Target: real_unlab + hallucinated_lab
        real_unlab_hal_lab = merge_intermediate_features(
            real_intermediate, hallucinated_intermediate,
            unlabeled_modalities, labeled_modalities
        )
        target_fusion = evan.forward_fusion_from_modality_features(real_unlab_hal_lab)
    elif objective == "addition":
        # Target: real_lab + real_unlab (both modalities real)
        real_lab_real_unlab = {m: real_intermediate[m] for m in all_modalities}
        target_fusion = evan.forward_fusion_from_modality_features(real_lab_real_unlab)

    # Project source fusion through cross-modal transformer
    predicted_cls = fused_projector(real_lab_hal_unlab_fusion)

    # Compute MSE loss on CLS tokens only
    loss = 0.0
    for mod in all_modalities:
        target = target_fusion[mod]['x_norm_clstoken']
        predicted = predicted_cls[mod]
        loss += mse_criteria(target, predicted)

    return loss


def _delulu_stage1_train_fused_projectors(
    evan, unlabeled_train_loader, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors, lr, epochs, objective="transfer",
    val_loader=None
):
    """Stage 1: Learn fused projector on unlabeled multimodal data.

    For objective="transfer": project from real_lab+hal_unlab → real_unlab+hal_lab
    For objective="addition": project from real_lab+hal_unlab → real_lab+real_unlab
    For objective="peeking": skip training entirely (return None)

    Uses a cross-modal transformer that processes concatenated full sequences,
    allowing cross-modal attention. Loss computed on CLS tokens only.

    If val_loader is provided, tracks best projector based on validation loss.
    """
    import copy

    # Peeking skips stage 1 entirely - no projector needed
    if objective == "peeking":
        print(f"\n--- Stage 1: Skipped (peeking objective) ---")
        return None

    print(f"\n--- Stage 1: Learning cross-modal fused projector ({objective}) ---")
    if val_loader is not None:
        print(f"  Using validation set for model selection")

    fused_projector = CrossModalFusedProjector(
        embed_dim=evan.embed_dim,
        modalities=all_modalities,
        num_heads=8,
        ffn_factor=4,
        num_layers=2
    ).to(device)

    optimizer = torch.optim.AdamW(fused_projector.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    best_val_loss = float('inf')
    best_projector_state = None

    for epoch in range(epochs):
        fused_projector.train()
        train_loss = 0.0
        pbar = tqdm(unlabeled_train_loader, desc=f"Fused Projector Epoch {epoch+1}/{epochs}")

        for batch in pbar:
            loss = _delulu_stage1_compute_loss(
                evan, batch, device, modality_bands_dict,
                unlabeled_modalities, labeled_modalities, all_modalities,
                intermediate_projectors, fused_projector, objective
            )

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(fused_projector.parameters(), max_norm=5)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm:.4f}'})

        avg_train_loss = train_loss / len(unlabeled_train_loader)

        # Validation
        if val_loader is not None:
            fused_projector.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    loss = _delulu_stage1_compute_loss(
                        evan, batch, device, modality_bands_dict,
                        unlabeled_modalities, labeled_modalities, all_modalities,
                        intermediate_projectors, fused_projector, objective
                    )
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_projector_state = copy.deepcopy(fused_projector.state_dict())
                print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f} (new best)")
            else:
                print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")
        else:
            print(f"  Fused Projector Epoch {epoch+1}: avg loss = {avg_train_loss:.4f}")
            scheduler.step(avg_train_loss)
            
    # Load best projector if validation was used
    if val_loader is not None and best_projector_state is not None:
        fused_projector.load_state_dict(best_projector_state)
        print(f"  Loaded best projector with val_loss={best_val_loss:.4f}")

    fused_projector.eval()
    for p in fused_projector.parameters():
        p.requires_grad = False

    return fused_projector


def _delulu_stage2_compute_loss(
    model, evan, batch, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors, fused_projector
):
    """Compute stage 2 pseudo-supervision loss for a single batch."""
    ce_criteria = nn.CrossEntropyLoss()

    labels = batch["label"].to(device)
    labmod_input = create_multimodal_batch(
        batch, modality_bands_dict=modality_bands_dict,
        modalities=(*labeled_modalities,)
    )
    labmod_input = {k: v.to(device) for k, v in labmod_input.items()}

    with torch.no_grad():
        labmod_intermediate = evan.forward_modality_specific_features(labmod_input)

    hallucinated_intermediate = hallucinate_intermediate_features(
        labmod_intermediate, labeled_modalities, unlabeled_modalities,
        intermediate_projectors
    )

    real_lab_hal_unlab = merge_intermediate_features(
        labmod_intermediate, hallucinated_intermediate,
        labeled_modalities, unlabeled_modalities
    )

    with torch.no_grad():
        hallucinated_fused = evan.forward_fusion_from_modality_features(real_lab_hal_unlab)

    # Apply cross-modal projector if available (not peeking)
    if fused_projector is not None:
        projected_cls = fused_projector(hallucinated_fused)
    else:
        projected_cls = None

    all_logits = []
    for mod in all_modalities:
        if projected_cls is not None:
            fused_cls = projected_cls[mod]
        else:
            fused_cls = hallucinated_fused[mod]['x_norm_clstoken']
        prediction = model.modality_classifiers[mod](fused_cls)
        all_logits.append(prediction)

    prediction = torch.stack(all_logits).mean(dim=0)
    loss = ce_criteria(prediction, labels)
    return loss


def _delulu_stage2_compute_distillation_loss(
    model, evan, teacher_model, batch, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors, objective, temperature=2.0
):
    """Compute distillation loss on unlabeled multimodal data (train2).

    Student input varies by objective:
    - transfer: real_unlabeled + hallucinated_labeled
    - addition: real_labeled + real_unlabeled (both real)
    - peeking: real_labeled + hallucinated_unlabeled

    Teacher: always monomodal on labeled_mod (frozen)

    Args:
        model: Student model with modality_classifiers
        evan: Student EVAN backbone
        teacher_model: Frozen monomodal teacher on labeled modality
        batch: Batch from train2 (has both modalities)
        device: torch device
        modality_bands_dict: Mapping of modality names to band indices
        unlabeled_modalities: Tuple of unlabeled modality names
        labeled_modalities: Tuple of labeled modality names
        all_modalities: Set of all modality names
        intermediate_projectors: Pre-trained sequence projectors
        objective: One of "transfer", "addition", "peeking"
        temperature: Softmax temperature for KL divergence

    Returns:
        Distillation loss (KL divergence scaled by temperature²)
    """
    # Get multimodal input (both modalities available in train2)
    multimodal_input = create_multimodal_batch(
        batch, modality_bands_dict=modality_bands_dict,
        modalities=(*labeled_modalities, *unlabeled_modalities)
    )
    multimodal_input = {k: v.to(device) for k, v in multimodal_input.items()}

    # Teacher forward (monomodal on labeled)
    labeled_input = {m: multimodal_input[m] for m in labeled_modalities}
    with torch.no_grad():
        teacher_logits = teacher_model(labeled_input)

    # Student forward (depends on objective)
    with torch.no_grad():
        real_intermediate = evan.forward_modality_specific_features(multimodal_input)

    if objective == "transfer":
        # Real unlabeled + hallucinated labeled
        hal_intermediate = hallucinate_intermediate_features(
            real_intermediate, unlabeled_modalities, labeled_modalities,
            intermediate_projectors
        )
        fusion_input = merge_intermediate_features(
            real_intermediate, hal_intermediate,
            unlabeled_modalities, labeled_modalities
        )
    elif objective == "addition":
        # Both real, no hallucination
        fusion_input = real_intermediate
    elif objective == "peeking":
        # Real labeled + hallucinated unlabeled
        hal_intermediate = hallucinate_intermediate_features(
            real_intermediate, labeled_modalities, unlabeled_modalities,
            intermediate_projectors
        )
        fusion_input = merge_intermediate_features(
            real_intermediate, hal_intermediate,
            labeled_modalities, unlabeled_modalities
        )
    else:
        raise ValueError(f"Unknown objective: {objective}")

    with torch.no_grad():
        fused_output = evan.forward_fusion_from_modality_features(fusion_input)

    # Classify with student heads and compute distillation loss
    all_logits = []
    for mod in all_modalities:
        fused_cls = fused_output[mod]['x_norm_clstoken']
        logits = model.modality_classifiers[mod](fused_cls)
        all_logits.append(logits)
    student_logits = torch.stack(all_logits).mean(dim=0)

    # KL divergence loss
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    distill_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

    return distill_loss


def _delulu_stage2_train_classifier(
    model, evan, labeled_train_loader, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors, fused_projector, lr, epochs,
    test_loader=None, eval_every_n_epochs=4, objective="transfer",
    val_loader=None,
    teacher_model=None,
    unlabeled_multimodal_loader=None,
    temperature=2.0,
    distill_only=False
):
    """Stage 2: Train classifier on labeled monomodal data with hallucinated features.

    For objective="transfer"/"addition": apply fused_projector before classifier
    For objective="peeking": skip projector (None), use fused CLS directly

    If val_loader is provided, tracks best classifier based on validation loss.
    If test_loader is provided, runs evaluation every eval_every_n_epochs epochs (oracle metric).

    Distillation modes:
        - teacher_model=None: pseudo-supervision only (CE loss on labeled monomodal)
        - teacher_model + distill_only=False: alternating epochs (CE + distillation)
        - teacher_model + distill_only=True: distillation only (KL loss on unlabeled multimodal)
    """
    import copy

    print(f"\n--- Stage 2: Training classifier on labeled monomodal data ({objective}) ---")
    if val_loader is not None:
        print(f"  Using validation set for model selection")

    # Check if distillation is enabled
    use_distillation = teacher_model is not None and unlabeled_multimodal_loader is not None
    if use_distillation:
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        mode = "distill_only" if distill_only else "alternating"
        print(f"  Distillation enabled: {mode}, temperature={temperature}")

    best_acc = 0
    best_val_loss = float('inf')
    best_classifier_state = None

    ce_criteria = nn.CrossEntropyLoss()

    # For distill_only mode: tune all EVAN parameters (not just classifiers)
    # For other modes: tune classifiers only
    if distill_only and use_distillation:
        print(f"  Distill-only mode: tuning ALL model parameters (EVAN + classifiers)")
        model.freeze_all()
        model.set_requires_grad('backbone', blocks=True, norm=True)
        model.set_requires_grad('all', classifier=True)
        trainable_params = list(model.parameters())
        best_classifier_state = None  # Will save full model state instead
    else:
        model.freeze_all()
        model.set_requires_grad("all", classifier=True)
        trainable_params = list(model.modality_classifiers.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Determine epoch type: distill_only runs all distillation, otherwise alternate
        if distill_only and use_distillation:
            is_distill_epoch = True
        else:
            is_distill_epoch = use_distillation and (epoch % 2 == 1)

        if is_distill_epoch:
            # Distillation epoch: train on unlabeled multimodal data
            pbar = tqdm(unlabeled_multimodal_loader, desc=f"Distill Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                optimizer.zero_grad()

                distill_loss = _delulu_stage2_compute_distillation_loss(
                    model, evan, teacher_model, batch, device,
                    modality_bands_dict, unlabeled_modalities, labeled_modalities,
                    all_modalities, intermediate_projectors, objective, temperature
                )

                distill_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5)
                optimizer.step()

                loss_val = distill_loss.item()
                train_loss += loss_val
                pbar.set_postfix({'distill': f'{loss_val:.4f}', 'grad': f'{grad_norm:.4f}'})

            avg_train_loss = train_loss / len(unlabeled_multimodal_loader)
        else:
            # Pseudo-supervision epoch: train on labeled monomodal data
            pbar = tqdm(labeled_train_loader, desc=f"CE Epoch {epoch+1}/{epochs}")
            for batch in pbar:
                optimizer.zero_grad()

                ce_loss = _delulu_stage2_compute_loss(
                    model, evan, batch, device, modality_bands_dict,
                    unlabeled_modalities, labeled_modalities, all_modalities,
                    intermediate_projectors, fused_projector
                )

                ce_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5)
                optimizer.step()

                loss_val = ce_loss.item()
                train_loss += loss_val
                pbar.set_postfix({'ce': f'{loss_val:.4f}', 'grad': f'{grad_norm:.4f}'})

            avg_train_loss = train_loss / len(labeled_train_loader)

        scheduler.step(avg_train_loss)

        # Validation-based model selection (pseudo-supervision loss on val set)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    loss = _delulu_stage2_compute_loss(
                        model, evan, batch, device, modality_bands_dict,
                        unlabeled_modalities, labeled_modalities, all_modalities,
                        intermediate_projectors, fused_projector
                    )
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save full model state for distill_only, classifiers only otherwise
                if distill_only and use_distillation:
                    best_classifier_state = copy.deepcopy(model.state_dict())
                else:
                    best_classifier_state = copy.deepcopy(model.modality_classifiers.state_dict())
                print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f} (new best)")
            else:
                print(f"  Epoch {epoch+1}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

        # Run test evaluation every n epochs (oracle metric, not used for model selection)
        if test_loader is not None and (epoch + 1) % eval_every_n_epochs == 0:
            curr_acc = _delulu_stage3_test(
                model, evan, test_loader, device, modality_bands_dict,
                unlabeled_modalities, labeled_modalities, all_modalities,
                intermediate_projectors, objective=objective
            )
            if curr_acc > best_acc:
                best_acc = curr_acc

    # Load best model state if validation was used
    if val_loader is not None and best_classifier_state is not None:
        if distill_only and use_distillation:
            model.load_state_dict(best_classifier_state)
            print(f"  Loaded best model (full state) with val_loss={best_val_loss:.4f}")
        else:
            model.modality_classifiers.load_state_dict(best_classifier_state)
            print(f"  Loaded best classifier with val_loss={best_val_loss:.4f}")

    return best_acc


def _delulu_stage3_test(
    model, evan, test_loader, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors, objective="transfer"
):
    """Stage 3: Test with modalities based on objective.

    For objective="transfer": test on unlabeled only, hallucinate labeled
    For objective="addition": test on both modalities (all real, no hallucination)
    For objective="peeking": test on labeled only, hallucinate unlabeled
    """
    # Determine test configuration based on objective
    if objective == "transfer":
        test_modalities = (*unlabeled_modalities,)
        desc = f"Testing on {unlabeled_modalities} only (hallucinate {labeled_modalities})"
    elif objective == "addition":
        test_modalities = (*labeled_modalities, *unlabeled_modalities)
        desc = f"Testing on both {labeled_modalities} and {unlabeled_modalities}"
    elif objective == "peeking":
        test_modalities = (*labeled_modalities,)
        desc = f"Testing on {labeled_modalities} only (hallucinate {unlabeled_modalities})"
    else:
        raise ValueError(f"Unknown objective: {objective}")

    print(f"\n--- Stage 3: {desc} ---")

    model.eval()
    softvote_correct = 0
    total = 0
    per_mod_correct = {mod: 0 for mod in all_modalities}
    # Track pairwise disagreement for diversity measurement
    all_mods_list = list(all_modalities)
    pairwise_disagreement = {}
    for i, mod_i in enumerate(all_mods_list):
        for j, mod_j in enumerate(all_mods_list):
            if i < j:
                pairwise_disagreement[f"{mod_i}_{mod_j}"] = 0

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")

        for batch in pbar:
            labels = batch["label"].to(device)
            test_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=test_modalities
            )
            test_input = {k: v.to(device) for k, v in test_input.items()}

            test_intermediate = evan.forward_modality_specific_features(test_input)

            # Build fusion input based on objective
            if objective == "transfer":
                # Real unlabeled + hallucinated labeled
                hallucinated_intermediate = hallucinate_intermediate_features(
                    test_intermediate, unlabeled_modalities, labeled_modalities,
                    intermediate_projectors
                )
                fusion_input = merge_intermediate_features(
                    test_intermediate, hallucinated_intermediate,
                    unlabeled_modalities, labeled_modalities
                )
            elif objective == "addition":
                # Both modalities are real, no hallucination needed
                fusion_input = test_intermediate
            elif objective == "peeking":
                # Real labeled + hallucinated unlabeled
                hallucinated_intermediate = hallucinate_intermediate_features(
                    test_intermediate, labeled_modalities, unlabeled_modalities,
                    intermediate_projectors
                )
                fusion_input = merge_intermediate_features(
                    test_intermediate, hallucinated_intermediate,
                    labeled_modalities, unlabeled_modalities
                )

            fused_output = evan.forward_fusion_from_modality_features(fusion_input)

            total += labels.size(0)
            all_logits = []
            for mod in all_mods_list:
                fused_cls = fused_output[mod]['x_norm_clstoken']
                prediction = model.modality_classifiers[mod](fused_cls)
                all_logits.append(prediction)

            # Get predictions for each modality
            all_preds = []
            for i, mod in enumerate(all_mods_list):
                _, predicted_mod = torch.max(all_logits[i], 1)
                all_preds.append(predicted_mod)
                per_mod_correct[mod] += (predicted_mod == labels).sum().item()

            # Compute pairwise disagreement for diversity
            all_preds_stack = torch.stack(all_preds)  # [num_mods, B]
            for i, mod_i in enumerate(all_mods_list):
                for j, mod_j in enumerate(all_mods_list):
                    if i < j:
                        pair_key = f"{mod_i}_{mod_j}"
                        disagreement = (all_preds_stack[i] != all_preds_stack[j]).sum().item()
                        pairwise_disagreement[pair_key] += disagreement

            softvote_logits = torch.stack(all_logits).mean(dim=0)
            _, softvote_predicted = torch.max(softvote_logits, 1)
            softvote_correct += (softvote_predicted == labels).sum().item()
            pbar.set_postfix({'acc': f'{100 * softvote_correct / total:.2f}%'})

    softvote_test_acc = 100 * softvote_correct / total
    print(f"  Test Accuracy (softvote): {softvote_test_acc:.2f}%")
    for mod in all_mods_list:
        mod_test_acc = 100 * per_mod_correct[mod] / total
        print(f"      Test Accuracy from {mod} classifier: {mod_test_acc:.2f}%")

    # Print predictive diversity (pairwise disagreement rates)
    print(f"  Predictive Diversity (pairwise disagreement %):")
    for pair_key, disagreement_count in pairwise_disagreement.items():
        disagreement_pct = 100 * disagreement_count / total
        print(f"      {pair_key}: {disagreement_pct:.2f}%")

    return softvote_test_acc


def delulu_supervision(
    model, unlabeled_train_loader, labeled_train_loader, test_loader,
    device, modality_bands_dict, unlabeled_modalities, labeled_modalities,
    intermediate_projectors, lr, epochs, stage2epochs=8, eval_every_n_epochs=4, objective="transfer",
    val1_loader=None, val2_loader=None,
    teacher_model=None, temperature=2.0, distill_only=False
):
    """
    Hallucination-based supervision with different objectives.

    Objectives:
        - "transfer": Learn to predict with unlabeled modality only at test time.
            Stage 1: Project real_lab+hal_unlab → real_unlab+hal_lab
            Stage 2: Train classifier with projector
            Stage 3: Test on unlabeled only (hallucinate labeled)

        - "addition": Learn to predict with both modalities at test time.
            Stage 1: Project real_lab+hal_unlab → real_lab+real_unlab
            Stage 2: Train classifier with projector
            Stage 3: Test on both modalities (no hallucination)

        - "peeking": Skip projector, test with labeled modality only.
            Stage 1: Skipped (returns None)
            Stage 2: Train classifier directly on fused CLS (no projector)
            Stage 3: Test on labeled only (hallucinate unlabeled)

    Args:
        model: Model with modality_classifiers
        unlabeled_train_loader: DataLoader for unlabeled multimodal data (Stage 1, also used for distillation in Stage 2)
        labeled_train_loader: DataLoader for labeled monomodal data (Stage 2)
        test_loader: DataLoader for test data (Stage 3)
        device: torch device
        modality_bands_dict: Mapping of modality names to band indices
        unlabeled_modalities: Tuple of unlabeled modality names
        labeled_modalities: Tuple of labeled modality names
        intermediate_projectors: Pre-trained sequence projectors
        lr: Learning rate
        epochs: Number of training epochs for Stage 1
        stage2epochs: Number of training epochs for Stage 2
        eval_every_n_epochs: Evaluation frequency in Stage 2
        objective: One of "transfer", "addition", "peeking"
        val1_loader: Validation loader from train1 (monomodal, for stage 2)
        val2_loader: Validation loader from train2 (multimodal, for stage 1)
        teacher_model: Optional monomodal teacher model for distillation (frozen, operates on labeled modality)
        temperature: Softmax temperature for KL divergence (default 2.0)
        distill_only: If True and teacher_model provided, run only distillation (no pseudo-supervision)

    Returns:
        Tuple of (best_acc, softvote_test_acc)
    """
    model.freeze_all()
    model.eval()
    evan = model.evan
    all_modalities = set(labeled_modalities + unlabeled_modalities)

    print(f"\n=== Hallucination Supervised Training ({objective}) ===")
    print(f"  Objective: {objective}")
    print(f"  Labeled modality: {labeled_modalities}")
    print(f"  Unlabeled modality: {unlabeled_modalities}")
    print(f"  Available intermediate projectors: {list(intermediate_projectors.keys())}")
    if val1_loader is not None or val2_loader is not None:
        print(f"  Using validation sets for model selection (val1={val1_loader is not None}, val2={val2_loader is not None})")

    intermediate_projectors.eval()
    for p in intermediate_projectors.parameters():
        p.requires_grad = False

    # Stage 1: trains on unlabeled multimodal (train2), validated on val2
    # Skip Stage 1 for distill_only mode (works directly with real features, no projector needed)
    if distill_only and teacher_model is not None:
        print(f"\n--- Stage 1: Skipped (distill_only mode) ---")
        fused_projector = None
    else:
        fused_projector = _delulu_stage1_train_fused_projectors(
            evan, unlabeled_train_loader, device, modality_bands_dict,
            unlabeled_modalities, labeled_modalities, all_modalities,
            intermediate_projectors, lr, epochs, objective=objective,
            val_loader=val2_loader
        )

    # Stage 2: trains on labeled monomodal (train1), validated on val1
    # Optionally uses distillation from teacher on unlabeled multimodal data (train2)
    best_acc = _delulu_stage2_train_classifier(
        model, evan, labeled_train_loader, device, modality_bands_dict,
        unlabeled_modalities, labeled_modalities, all_modalities,
        intermediate_projectors, fused_projector, lr, stage2epochs,
        test_loader=test_loader, eval_every_n_epochs=eval_every_n_epochs, objective=objective,
        val_loader=val1_loader,
        teacher_model=teacher_model,
        unlabeled_multimodal_loader=unlabeled_train_loader,
        temperature=temperature,
        distill_only=distill_only
    )

    # Stage 3
    softvote_test_acc = _delulu_stage3_test(
        model, evan, test_loader, device, modality_bands_dict,
        unlabeled_modalities, labeled_modalities, all_modalities,
        intermediate_projectors, objective=objective
    )

    return best_acc,softvote_test_acc
