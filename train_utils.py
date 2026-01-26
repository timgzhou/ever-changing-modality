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


def _delulu_stage1_train_fused_projectors(
    evan, unlabeled_train_loader, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors, lr, epochs
):
    """Stage 1: Learn fused_cls_projectors on unlabeled multimodal data."""
    print(f"\n--- Stage 1: Learning fused_cls_projectors on unlabeled multimodal data ---")

    def make_fused_cls_projector(factor=4):
        return nn.Sequential(
            nn.Linear(evan.embed_dim, evan.embed_dim * factor),
            nn.GELU(),
            nn.Linear(evan.embed_dim * factor, evan.embed_dim)
        ).to(device)

    fused_cls_projectors = nn.ModuleDict({mod: make_fused_cls_projector() for mod in all_modalities})
    fused_projector_params = list(fused_cls_projectors.parameters())
    optimizer = torch.optim.AdamW(fused_projector_params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)
    mse_criteria = nn.MSELoss()

    for epoch in range(epochs):
        fused_cls_projectors.train()
        train_loss = 0.0
        pbar = tqdm(unlabeled_train_loader, desc=f"Fused Projector Epoch {epoch+1}/{epochs}")

        for batch in pbar:
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

            real_lab_hal_unlab = merge_intermediate_features(real_intermediate, hallucinated_intermediate, labeled_modalities, unlabeled_modalities)
            real_unlab_hal_lab = merge_intermediate_features(real_intermediate, hallucinated_intermediate, unlabeled_modalities, labeled_modalities)

            real_lab_hal_unlab_fusion = evan.forward_fusion_from_modality_features(real_lab_hal_unlab)
            real_unlab_hal_lab_fusion = evan.forward_fusion_from_modality_features(real_unlab_hal_lab)

            loss = 0.0
            for mod in all_modalities:
                target = real_unlab_hal_lab_fusion[mod]['x_norm_clstoken']
                source = real_lab_hal_unlab_fusion[mod]['x_norm_clstoken']
                predicted = fused_cls_projectors[mod](source)
                loss += mse_criteria(target, predicted)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(fused_projector_params, max_norm=5)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm:.4f}'})

        avg_loss = train_loss / len(unlabeled_train_loader)
        scheduler.step(avg_loss)
        print(f"  Fused Projector Epoch {epoch+1}: avg loss = {avg_loss:.4f}")

    fused_cls_projectors.eval()
    for p in fused_cls_projectors.parameters():
        p.requires_grad = False

    return fused_cls_projectors


def _delulu_stage2_train_classifier(
    model, evan, labeled_train_loader, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors, fused_cls_projectors, lr, epochs,
    test_loader=None, eval_every_n_epochs=4
):
    """Stage 2: Train classifier on labeled monomodal data with hallucinated features.

    If test_loader is provided, runs evaluation every eval_every_n_epochs epochs.
    """
    print(f"\n--- Stage 2: Training classifier on labeled monomodal data with hallucinated features ---")
    best_acc=0
    ce_criteria = nn.CrossEntropyLoss()
    model.freeze_all()
    model.set_requires_grad("all", classifier=True)

    classifier_params = list(model.modality_classifiers.parameters())
    optimizer = torch.optim.AdamW(classifier_params, lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        pbar = tqdm(labeled_train_loader, desc=f"Classifier Epoch {epoch+1}/{epochs}")

        for batch in pbar:
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

            all_logits = []
            for mod in all_modalities:
                predicted_fused_cls = fused_cls_projectors[mod](hallucinated_fused[mod]['x_norm_clstoken'])
                prediction = model.modality_classifiers[mod](predicted_fused_cls)
                all_logits.append(prediction)

            prediction = torch.stack(all_logits).mean(dim=0)
            loss = ce_criteria(prediction, labels)

            optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(classifier_params, max_norm=5)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad': f'{grad_norm:.4f}'})

        avg_loss = train_loss / len(labeled_train_loader)
        scheduler.step(avg_loss)

        # Run evaluation every n epochs
        if test_loader is not None and (epoch + 1) % eval_every_n_epochs == 0:
            curr_acc = _delulu_stage3_test(
                model, evan, test_loader, device, modality_bands_dict,
                unlabeled_modalities, labeled_modalities, all_modalities,
                intermediate_projectors
            )
            if curr_acc > best_acc:
                best_acc=curr_acc
    return best_acc


def _delulu_stage3_test(
    model, evan, test_loader, device, modality_bands_dict,
    unlabeled_modalities, labeled_modalities, all_modalities,
    intermediate_projectors
):
    """Stage 3: Test on unlabeled modalities only."""
    print(f"\n--- Stage 3: Testing on unlabeled modalities only ---")

    model.eval()
    softvote_correct = 0
    total = 0
    per_mod_correct = {mod: 0 for mod in all_modalities}

    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")

        for batch in pbar:
            labels = batch["label"].to(device)
            unlabmod_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=(*unlabeled_modalities,)
            )
            unlabmod_input = {k: v.to(device) for k, v in unlabmod_input.items()}

            unlabmod_intermediate = evan.forward_modality_specific_features(unlabmod_input)

            hallucinated_intermediate = hallucinate_intermediate_features(
                unlabmod_intermediate, unlabeled_modalities, labeled_modalities,
                intermediate_projectors
            )

            real_unlab_hal_lab = merge_intermediate_features(
                unlabmod_intermediate, hallucinated_intermediate,
                unlabeled_modalities, labeled_modalities
            )

            fused_output = evan.forward_fusion_from_modality_features(real_unlab_hal_lab)

            total += labels.size(0)
            all_logits = []
            for mod in all_modalities:
                fused_cls = fused_output[mod]['x_norm_clstoken']
                prediction = model.modality_classifiers[mod](fused_cls)
                all_logits.append(prediction)

            for i, mod in enumerate(all_modalities):
                _, predicted_mod = torch.max(all_logits[i], 1)
                per_mod_correct[mod] += (predicted_mod == labels).sum().item()

            softvote_logits = torch.stack(all_logits).mean(dim=0)
            _, softvote_predicted = torch.max(softvote_logits, 1)
            softvote_correct += (softvote_predicted == labels).sum().item()
            pbar.set_postfix({'acc': f'{100 * softvote_correct / total:.2f}%'})

    softvote_test_acc = 100 * softvote_correct / total
    print(f"  Test Accuracy using only {unlabeled_modalities} with hallucinated softvote: {softvote_test_acc:.2f}%")
    for mod in all_modalities:
        mod_test_acc = 100 * per_mod_correct[mod] / total
        print(f"      Test Accuracy from {mod} classifier: {mod_test_acc:.2f}%")

    return softvote_test_acc


def delulu_supervision(
    model, unlabeled_train_loader, labeled_train_loader, test_loader,
    device, modality_bands_dict, unlabeled_modalities, labeled_modalities,
    intermediate_projectors, lr, epochs, eval_every_n_epochs=4
):
    """
    Modality transfer - learning a predictor for the unlabeled_modalities alone.

    Stage 1: Learn fused_cls_projectors on unlabeled multimodal data
    Stage 2: Train classifier on labeled monomodal data with hallucinated features
             (runs evaluation every eval_every_n_epochs epochs)
    Stage 3: Final test on unlabeled modalities only
    """
    model.freeze_all()
    model.eval()
    evan = model.evan
    all_modalities = set(labeled_modalities + unlabeled_modalities)

    print(f"\n=== Hallucination Supervised Training ===")
    print(f"  Labeled modality: {labeled_modalities}")
    print(f"  Unlabeled modality: {unlabeled_modalities}")
    print(f"  Available intermediate projectors: {list(intermediate_projectors.keys())}")

    intermediate_projectors.eval()
    for p in intermediate_projectors.parameters():
        p.requires_grad = False

    # Stage 1
    fused_cls_projectors = _delulu_stage1_train_fused_projectors(
        evan, unlabeled_train_loader, device, modality_bands_dict,
        unlabeled_modalities, labeled_modalities, all_modalities,
        intermediate_projectors, lr, epochs
    )

    # Stage 2 (with periodic evaluation)
    best_acc = _delulu_stage2_train_classifier(
        model, evan, labeled_train_loader, device, modality_bands_dict,
        unlabeled_modalities, labeled_modalities, all_modalities,
        intermediate_projectors, fused_cls_projectors, lr, epochs,
        test_loader=test_loader, eval_every_n_epochs=eval_every_n_epochs
    )

    # Stage 3
    softvote_test_acc = _delulu_stage3_test(
        model, evan, test_loader, device, modality_bands_dict,
        unlabeled_modalities, labeled_modalities, all_modalities,
        intermediate_projectors
    )

    return best_acc,softvote_test_acc
