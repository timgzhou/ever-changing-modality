"""Training utilities for EVAN on EuroSAT."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from einops import rearrange
from eurosat_data_utils import (
    create_multimodal_batch,
    print_and_reset_rgb_stats,
)
import wandb


class SequenceProjector(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ffn_factor=4, num_layers=1, output_dim=None):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_factor,
            batch_first=True,
            norm_first=True
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Linear(embed_dim, output_dim) if output_dim is not None else None

    def forward(self, x):
        x = self.layers(x)
        if self.proj_out is not None:
            x = self.proj_out(x)
        return x


def hallucinate_intermediate_features(
    source_intermediate: dict,
    source_modalities: tuple,
    target_modalities: tuple,
    evan,
) -> dict:
    """
    Hallucinate intermediate features for target modalities from source modalities.

    Uses full sequence projection (CLS + storage + patches) via transformer-based projectors.
    For each target modality, projects from all available source modalities and takes the mean.

    Args:
        source_intermediate: Dict of source modality features {mod: [B, seq_len, embed_dim]}
        source_modalities: Tuple of source modality names
        target_modalities: Tuple of target modality names to hallucinate
        evan: EVAN model (projectors accessed via evan.intermediate_projectors and evan._project_sequence)

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
            projected_seqs.append(evan._project_sequence(src_seq_norm, proj_key, tar_mod))
        # Mean of all projections
        hallucinated[tar_mod] = torch.stack(projected_seqs).mean(dim=0)
    return hallucinated


def merge_intermediate_features(real_features, hallucinated_features, real_modalities, hallucinated_modalities):
    """Merge real and hallucinated intermediate features into a single dict."""
    return {
        **{m: real_features[m] for m in real_modalities},
        **{m: hallucinated_features[m] for m in hallucinated_modalities}
    }


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


def _compute_map(all_outputs, all_labels):
    """Compute mean Average Precision for multilabel classification.

    Args:
        all_outputs: [N, C] float logits
        all_labels:  [N, C] float binary targets

    Returns:
        mAP as a percentage (0-100)
    """
    scores = torch.sigmoid(all_outputs)  # [N, C]
    num_classes = scores.shape[1]
    aps = []
    for c in range(num_classes):
        gt = all_labels[:, c]
        if gt.sum() == 0:
            continue  # skip classes with no positives in this split
        sc = scores[:, c]
        # Sort by descending score
        order = sc.argsort(descending=True)
        gt_sorted = gt[order]
        tp_cumsum = gt_sorted.cumsum(0)
        precision_at_k = tp_cumsum / torch.arange(1, len(gt_sorted) + 1, dtype=torch.float32, device=gt.device)
        ap = (precision_at_k * gt_sorted).sum() / gt.sum()
        aps.append(ap.item())
    return 100.0 * (sum(aps) / len(aps)) if aps else 0.0


def evaluate(model, dataloader, criterion, device, modality_bands_dict,
             modalities_to_use=('rgb',), pseudo_modalities=None, intermediate_projectors=None,
             multilabel=False, label_key='label', segmentation=False, num_classes=None,
             ignore_index=-100):
    """
    Evaluate model on a dataloader.

    Args:
        model: EVAN classifier or EvanSegmenter
        dataloader: DataLoader
        criterion: Loss function
        device: torch device
        modality_bands_dict: Dict mapping modality names to their band tuples or slices
        modalities_to_use: Tuple of modality names to use for evaluation
        pseudo_modalities: Optional list of modalities to hallucinate using sequence projection
        intermediate_projectors: Required if pseudo_modalities is provided
        multilabel: If True, report mAP instead of top-1 accuracy (for BEN-v2 etc.)
        label_key: Key for labels in batch dict ('label' or 'mask')
        segmentation: If True, compute mIoU over [B,H,W] predictions (for PASTIS etc.)
        num_classes: Required when segmentation=True.
        ignore_index: Label value excluded from mIoU (e.g. 19 for PASTIS void_label).
    """
    model.eval()
    if intermediate_projectors is not None:
        intermediate_projectors.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_outputs_list = []
    all_labels_list = []
    all_seg_preds = []
    all_seg_labels = []

    with torch.no_grad():
        for batch in dataloader:
            labels = batch[label_key].to(device)
            if multilabel:
                labels = labels.float()

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

            if segmentation:
                # outputs: [B, C, H, W]; labels: [B, H, W]
                all_seg_preds.append(outputs.argmax(dim=1).cpu())
                all_seg_labels.append(labels.cpu())
            elif multilabel:
                all_outputs_list.append(outputs.cpu())
                all_labels_list.append(labels.cpu())
            else:
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    if segmentation:
        metric = compute_miou(torch.cat(all_seg_preds), torch.cat(all_seg_labels), num_classes, ignore_index=ignore_index)
    elif multilabel:
        metric = _compute_map(torch.cat(all_outputs_list), torch.cat(all_labels_list))
    else:
        metric = 100. * correct / total

    return avg_loss, metric


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

def single_modality_training_loop(model, train_loader, test_loader, device,
                                   modality_bands_dict, criterion, optimizer, num_epochs,
                                   modality, phase_name="Training",
                                   use_wandb=False, wandb_prefix=None, clip_norm=10,
                                   multilabel=False, label_key='label',
                                   segmentation=False, num_classes=None,
                                   ignore_index=-100,
                                   val_loader=None, best_checkpoint_path=None,
                                   val_per_epoch=1):
    """
    Simple training loop for single-modality EVAN training (Stage 0).

    Args:
        model: EVANClassifier or EvanSegmenter
        train_loader: Training dataloader
        test_loader: Test dataloader
        device: torch device
        modality_bands_dict: Dict mapping modality name to band tuple or slice
        criterion: Loss function
        optimizer: Optimizer
        num_epochs: Number of epochs to train
        modality: Single modality name to train and evaluate on
        phase_name: Name of this training phase for logging
        use_wandb: Whether to log to wandb
        wandb_prefix: Prefix for wandb metrics
        clip_norm: Max gradient norm for clipping
        multilabel: If True, report mAP instead of top-1 accuracy and accumulate train outputs
        label_key: Key for labels in batch dict ('label' or 'mask')
        segmentation: If True, report mIoU; model outputs [B, C, H, W], labels are [B, H, W]
        num_classes: Required when segmentation=True.
        val_loader: Optional val1 dataloader; if provided, best checkpoint is kept by val metric.
        best_checkpoint_path: Path to save the best checkpoint (required when val_loader is set).

    Returns:
        Tuple of (train_metric, test_metric, best_val_metric, best_val_test_metric)
        where metric is Acc (%), mAP (%), or mIoU (%) depending on task.
        best_val_metric and best_val_test_metric are None when val_loader is not provided.
    """
    mod_str = modality.upper()
    if segmentation:
        metric_name = "mIoU"
    elif multilabel:
        metric_name = "mAP"
    else:
        metric_name = "Acc"
    global_step = 0
    best_val_metric = None
    best_val_test_metric = None
    if val_loader is not None:
        best_val_metric = 0
        best_val_test_metric = 0
        assert best_checkpoint_path is not None, "best_checkpoint_path required when val_loader is provided"
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=0, min_lr=1e-6)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_outputs_list = []
        train_labels_list = []
        train_seg_preds = []
        train_seg_labels = []

        pbar = tqdm(train_loader, desc=f"{phase_name} Epoch {epoch+1}/{num_epochs} [{mod_str}]")
        for batch in pbar:
            labels = batch[label_key].to(device)
            if multilabel:
                labels = labels.float()

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
            if segmentation:
                train_seg_preds.append(outputs.detach().argmax(dim=1).cpu())
                train_seg_labels.append(labels.detach().cpu())
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'})
            elif multilabel:
                train_outputs_list.append(outputs.detach().cpu())
                train_labels_list.append(labels.detach().cpu())
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'grad_norm': f'{grad_norm:.4f}'})
            else:
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100.*train_correct/train_total:.2f}%',
                    'grad_norm': f'{grad_norm:.4f}'
                })
            global_step += 1

            if use_wandb and wandb_prefix:
                wandb.log({
                    f'{wandb_prefix}/train_loss': loss.item(),
                    f'{wandb_prefix}/grad_norm': grad_norm.item(),
                    f'{wandb_prefix}/step': global_step,
                })

        train_loss /= len(train_loader)
        print_and_reset_rgb_stats()
        if segmentation:
            train_metric = compute_miou(
                torch.cat(train_seg_preds), torch.cat(train_seg_labels), num_classes,
                ignore_index=ignore_index
            )
        elif multilabel:
            train_metric = _compute_map(torch.cat(train_outputs_list), torch.cat(train_labels_list))
        else:
            train_metric = 100. * train_correct / train_total

        scheduler.step(train_loss)

        do_eval = ((epoch + 1) % val_per_epoch == 0) or (epoch + 1 == num_epochs)

        test_loss, test_metric = None, None
        val_loss, val_metric = None, None
        if do_eval:
            eval_kwargs = dict(
                modality_bands_dict=modality_bands_dict,
                modalities_to_use=(modality,),
                multilabel=multilabel,
                label_key=label_key,
                segmentation=segmentation,
                num_classes=num_classes,
                ignore_index=ignore_index,
            )
            test_loss, test_metric = evaluate(model, test_loader, criterion, device, **eval_kwargs)

            if val_loader is not None:
                val_loss, val_metric = evaluate(model, val_loader, criterion, device, **eval_kwargs)

            print(f"  Train ({mod_str}): Loss: {train_loss:.4f}, {metric_name}: {train_metric:.2f}%")
            print(f"  Test ({mod_str}):  Loss: {test_loss:.4f}, {metric_name}: {test_metric:.2f}% (epoch {epoch+1}/{num_epochs})")
            if val_metric is not None:
                print(f"  Val  ({mod_str}):  Loss: {val_loss:.4f}, {metric_name}: {val_metric:.2f}%")
            if val_metric is not None and val_metric > best_val_metric:
                print(f"    New val record: {val_metric:.2f} > previous {best_val_metric:.2f} at epoch {epoch+1} — saving checkpoint")
                best_val_metric = val_metric
                best_val_test_metric = test_metric
                model.save_checkpoint(best_checkpoint_path)

        if use_wandb and wandb_prefix:
            log_dict = {
                f'{wandb_prefix}/train_loss_epoch': train_loss,
                f'{wandb_prefix}/train_{metric_name.lower()}': train_metric,
                f'{wandb_prefix}/epoch': epoch + 1,
                f'{wandb_prefix}/lr': optimizer.param_groups[0]['lr'],
            }
            if test_metric is not None:
                log_dict[f'{wandb_prefix}/eval_loss'] = test_loss
                log_dict[f'{wandb_prefix}/eval_{metric_name.lower()}'] = test_metric
            if val_metric is not None:
                log_dict[f'{wandb_prefix}/val_loss'] = val_loss
                log_dict[f'{wandb_prefix}/val_{metric_name.lower()}'] = val_metric
            wandb.log(log_dict)

    return train_metric, test_metric, best_val_metric, best_val_test_metric

def compute_miou(preds: torch.Tensor, labels: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """
    Compute mean Intersection-over-Union over present classes.

    Args:
        preds:  [B, H, W] long — argmax predictions.
        labels: [B, H, W] long — ground truth class indices.
        num_classes: Number of classes.
        ignore_index: Label value to exclude (default 255).

    Returns:
        mIoU as a percentage (0–100).
    """
    valid = labels != ignore_index
    preds = preds[valid]
    labels = labels[valid]

    ious = []
    for cls in range(num_classes):
        pred_mask = preds == cls
        gt_mask = labels == cls
        intersection = (pred_mask & gt_mask).sum().item()
        union = (pred_mask | gt_mask).sum().item()
        if union > 0:
            ious.append(intersection / union)

    return 100.0 * (sum(ious) / len(ious)) if ious else 0.0

