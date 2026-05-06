"""Training utilities for DeluluNet."""

import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from data_utils import create_multimodal_batch
from train_utils import SequenceProjector, _compute_map, compute_miou, make_scheduler, evaluate as evaluate_unimodal
import wandb
import numpy as np

def _make_batch(batch, modality_bands_dict, modalities, device):
    out = create_multimodal_batch(batch, modality_bands_dict=modality_bands_dict, modalities=modalities)
    return {k: v.to(device) for k, v in out.items()}

def distillation_loss(student_logits, teacher_logits, temperature, task_type):
    """Distillation loss between student and teacher logits.

    task_type:
        'classification' — softmax KL divergence [B, C]
        'multilabel'     — per-class BCE against sigmoid teacher probs [B, C]
        'segmentation'   — softmax KL divergence on [B, C, img_H, img_W]
    """
    if task_type == "segmentation" and student_logits.dim() == 4:
        C = student_logits.shape[1] # number of classes
        student_logits = student_logits.permute(0, 2, 3, 1).reshape(-1, C) # (batch*n_pixels),classes (per-patch class prediction)
        teacher_logits = teacher_logits.permute(0, 2, 3, 1).reshape(-1, C)
    if task_type == "multilabel":
        teacher_probs = torch.sigmoid(teacher_logits / temperature)
        return F.binary_cross_entropy_with_logits(student_logits / temperature, teacher_probs)
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

# ==================== SHOT TRAINING COMPONENT ====================

def create_latent_decoders(hidden_dim, latent_reconstruct_modalities, device, num_heads=8, ffn_factor=4):
    """Create transformer-based projectors for latent matching (CLS + patches jointly)."""
    projectors = nn.ModuleDict()
    for mod in latent_reconstruct_modalities:
        projectors[mod] = SequenceProjector(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            ffn_factor=ffn_factor,
            num_layers=1,
        ).to(device)
        n_params = sum(p.numel() for p in projectors[mod].parameters() if p.requires_grad)
        print(f"  Initialized Latent Projector (Transformer) for {mod} ({n_params:,} params)")
    return projectors


# MASKING HELPER FUNCTION
def mask_input(evan, batch_size, num_patches, token_mask_ratio, all_modalities, prefusion_features, modality_dropout, device,
               protected_modalities=None, active_losses=None, latent_reconstruct_modalities=None, protect_lrm=False, use_mask_token=False):
    """
    Apply masking using projected sequences from other modalities, and compute prefusion loss.

    For partially masked modalities: replace masked token positions with projected tokens.
    For fully dropped modalities: replace entire sequence with mean of projected sequences.

    Args:
        protected_modalities: List of modalities that should never be fully dropped.
        active_losses: Set of active loss names; prefusion loss computed only if 'prefusion' in active_losses.
        latent_reconstruct_modalities: Modalities whose encoder gradients should be blocked
                                       (src detached before projection; tgt detached for loss).
        use_mask_token: If True, replace projector outputs with a broadcast learned mask token
                        (projector_queries). Prefusion loss is always 0 in this mode.
    Returns:
        (modality_masks, masked_mod_features, modality_dropped, prefusion_loss)
    """
    modality_masks = {}  # {mod: [B, num_patches] bool tensor, True=masked}
    modality_dropped = {}

    drop_candidates = [
        mod for mod in all_modalities
        if np.random.rand() < (modality_dropout[mod] if isinstance(modality_dropout, dict) else modality_dropout)
        and (protected_modalities is None or mod not in protected_modalities)
    ]
    if len(drop_candidates) == len(all_modalities):
        drop_candidates.pop(np.random.randint(len(drop_candidates)))

    available_modalities = [mod for mod in all_modalities if mod not in drop_candidates]

    for mod in all_modalities:
        modality_dropped[mod] = mod in drop_candidates

        if len(drop_candidates) > 0:
            # Modality dropout mode: dropped=all masked, surviving=no token masking
            mask_val = mod in drop_candidates # True means masked, False means visible
            modality_masks[mod] = torch.full((batch_size, num_patches), mask_val, device=device, dtype=torch.bool)
        else:
            # token dropping mode: randomly mask token_mask_ratio of tokens per modality
            len_keep = int(num_patches * (1 - token_mask_ratio))
            noise = torch.rand(batch_size, num_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            mask = torch.ones(batch_size, num_patches, device=device, dtype=torch.bool)
            mask.scatter_(1, ids_shuffle[:, :len_keep], False)
            modality_masks[mod] = mask

    prefusion_loss = torch.tensor(0.0, device=device)

    if not use_mask_token:
        # Compute projected sequences from all available modalities to all target modalities.
        # Sources in latent_reconstruct_modalities are detached to block encoder gradients.
        # For cross projector: mask out source patches that are also masked (no leakage).
        projected_sequences = {}  # {(src_mod, tgt_mod): [B, seq_len, embed_dim]}
        lrm = set(latent_reconstruct_modalities) if latent_reconstruct_modalities else set()
        for src_mod in available_modalities:
            src_seq = prefusion_features[src_mod]
            if protect_lrm and (src_mod in lrm):
                src_seq = src_seq.detach()
            src_patch_mask = modality_masks[src_mod]
            for tgt_mod in all_modalities:
                if src_mod != tgt_mod:
                    key = f"{src_mod}_to_{tgt_mod}"
                    projected_sequences[(src_mod, tgt_mod)] = evan._project_sequence(src_seq, key, tgt_mod, src_patch_mask=src_patch_mask)

        # cross projector outputs [B, 1+n_patches, D] (no storage tokens)
        # Prefusion loss: MSE between projected and real target features.
        # Strip storage tokens from tgt to match proj_seq shape.
        if active_losses and 'prefusion' in active_losses:
            for (src_mod, tgt_mod), proj_seq in projected_sequences.items():
                tgt_seq = prefusion_features[tgt_mod]  # [B, 1+n_storage+n_patches, D]
                if protect_lrm and (tgt_mod in lrm):
                    tgt_seq = tgt_seq.detach()
                tgt_seq = torch.cat([tgt_seq[:, :1], tgt_seq[:, evan.n_storage_tokens + 1:]], dim=1)
                prefusion_loss = prefusion_loss + F.mse_loss(proj_seq, tgt_seq)
            if projected_sequences:
                prefusion_loss = prefusion_loss / len(projected_sequences)

    masked_mod_features = {}
    n_prefix = evan.n_storage_tokens + 1

    for mod in all_modalities:
        features = prefusion_features[mod]  # [B, 1+n_storage+num_patches, embed_dim]

        if modality_dropped[mod]:
            # Fully dropped: replace with mask token or mean of projected sequences.
            if use_mask_token:
                masked_mod_features[mod] = evan._hallucinate_with_mask_token(mod, batch_size, device)
            else:
                # proj_seq is [B, 1+n_patches, D] — fusion handles variable prefix size.
                projected_list = [projected_sequences[(avail_mod, mod)] for avail_mod in available_modalities]
                masked_mod_features[mod] = torch.stack(projected_list).mean(dim=0)
        else:
            # Partially masked: replace masked patch positions with mask token or projected tokens.
            if use_mask_token:
                mask_seq = evan._hallucinate_with_mask_token(mod, batch_size, device)
                # mask_seq is [B, 1+n_patches, D]; pad storage slots with zeros to match features shape
                zeros = torch.zeros(batch_size, evan.n_storage_tokens, mask_seq.shape[-1], device=device)
                projected_seq = torch.cat([mask_seq[:, :1], zeros, mask_seq[:, 1:]], dim=1)
            else:
                other_available = [m for m in available_modalities if m != mod]
                if other_available:
                    projected_list = [projected_sequences[(avail_mod, mod)] for avail_mod in other_available]
                    proj_seq = torch.stack(projected_list).mean(dim=0)
                    zeros = torch.zeros(batch_size, evan.n_storage_tokens, proj_seq.shape[-1], device=device)
                    projected_seq = torch.cat([proj_seq[:, :1], zeros, proj_seq[:, 1:]], dim=1)
                else:
                    projected_seq = torch.zeros_like(features)

            # CLS and storage tokens are never masked for partial masking
            prefix_mask = torch.zeros(batch_size, n_prefix, device=device, dtype=torch.bool)
            full_mask = torch.cat([prefix_mask, modality_masks[mod]], dim=1)  # [B, seq_len]
            full_mask_expanded = full_mask.unsqueeze(-1)  # [B, seq_len, 1]

            # Replace masked positions with projected tokens (zeros at storage slots are never selected)
            masked_mod_features[mod] = torch.where(full_mask_expanded, projected_seq, features)

    return modality_masks, masked_mod_features, modality_dropped, prefusion_loss


def mixed_batch_iterator(unlabeled_loader, labeled_loader, labeled_freq):
    """Yield (batch, is_labeled) from a mix of unlabeled and labeled loaders.

    labeled_freq fraction of total_batches = min(len(unlabeled), len(labeled)) come from labeled_loader.
    """
    if labeled_loader is None or labeled_freq == 0:
        for batch in unlabeled_loader:
            yield batch, False
        return

    total_batches = min(len(unlabeled_loader), len(labeled_loader))
    n_labeled = int(total_batches * labeled_freq)
    n_unlabeled = total_batches - n_labeled

    unlabeled_iter = iter(unlabeled_loader)
    labeled_iter = iter(labeled_loader)

    # Create schedule: True for labeled, False for unlabeled, then shuffle
    schedule = [True] * n_labeled + [False] * n_unlabeled
    np.random.shuffle(schedule)

    for is_labeled in schedule:
        if is_labeled:
            yield next(labeled_iter), True
        else:
            yield next(unlabeled_iter), False


def evaluate_multimodal(
    model, loader, device, modality_bands_dict,
    starting_modality, newmod_modalities, all_modalities,
    task_type='classification', label_key='label',
    with_labels=False, desc="Evaluating",
    num_classes=None, ignore_index=-100,
    use_mask_token=False,
    teacher_classifier=None, agree_ref='teacher',
):
    """
    Single-pass multimodal evaluation (peeking / transfer / addition paths).

    with_labels=True: returns transfer_acc, peeking_acc, addition_acc, ens_acc.
    Always returns transfer_agree, addition_agree (agreement with peeking path, as percentages).
    """
    model.eval()

    newmod_modalities = list(newmod_modalities)
    all_mods = list(all_modalities)

    # Label-based accumulators
    total = 0
    peeking_correct = 0
    transfer_correct = 0
    addition_correct = 0
    ens_correct = 0
    peeking_logits_list, transfer_logits_list, addition_logits_list, ens_logits_list, labels_list = [], [], [], [], []
    # Segmentation accumulators
    peeking_preds_list, transfer_preds_list, addition_preds_list, ens_preds_list, seg_labels_list = [], [], [], [], []

    # Agreement accumulators
    transfer_agree = 0
    addition_agree = 0
    total_pixels = 0


    with torch.no_grad():
        for batch in tqdm(loader, desc=desc):
            mm_batch = _make_batch(batch, modality_bands_dict, tuple(all_mods), device)

            # Single modality-specific forward pass
            intermediate = model.evan.forward_modality_specific_features(mm_batch)

            # Labels (if available)
            labels = None
            if with_labels:
                labels = batch[label_key].to(device)
                if task_type == "multilabel":
                    labels = labels.float()
                total += labels.shape[0]

            # --- Peeking path: real starting_mod + hallucinated newmod ---
            peeking_sv = model.predict_from_real_modalities(
                intermediate, (starting_modality,), tuple(all_mods), use_mask_token=use_mask_token
            )
            # --- Transfer path: real newmod + hallucinated starting_mod ---
            transfer_sv = model.predict_from_real_modalities(
                intermediate, tuple(newmod_modalities), tuple(all_mods), use_mask_token=use_mask_token
            )
            # --- Addition path: both modalities real ---
            addition_sv = model.predict_from_real_modalities(
                intermediate, tuple(all_mods), tuple(all_mods)
            )

            # --- Ensemble path: average peeking + transfer ---
            ens_sv = (peeking_sv + transfer_sv) / 2

            # Accumulate label-based metrics
            if with_labels:
                if task_type == "segmentation":
                    peeking_preds_list.append(peeking_sv.argmax(1).cpu())
                    transfer_preds_list.append(transfer_sv.argmax(1).cpu())
                    addition_preds_list.append(addition_sv.argmax(1).cpu())
                    ens_preds_list.append(ens_sv.argmax(1).cpu())
                    seg_labels_list.append(labels.cpu())
                elif task_type == "multilabel":
                    peeking_logits_list.append(peeking_sv.cpu())
                    transfer_logits_list.append(transfer_sv.cpu())
                    addition_logits_list.append(addition_sv.cpu())
                    ens_logits_list.append(ens_sv.cpu())
                    labels_list.append(labels.cpu())
                else:
                    peeking_correct += (peeking_sv.argmax(1) == labels).sum().item()
                    transfer_correct += (transfer_sv.argmax(1) == labels).sum().item()
                    addition_correct += (addition_sv.argmax(1) == labels).sum().item()
                    ens_correct += (ens_sv.argmax(1) == labels).sum().item()

            # Agreement reference: teacher unimodal predictions (default) or peeking path
            if not with_labels:
                total += peeking_sv.shape[0]
            if agree_ref == 'teacher' and teacher_classifier is not None:
                ref_sv = teacher_classifier({starting_modality: mm_batch[starting_modality]})
            else:
                ref_sv = peeking_sv
            ref_preds = ref_sv.argmax(1)
            if task_type == "segmentation":
                transfer_agree += (transfer_sv.argmax(1) == ref_preds).sum().item()
                addition_agree += (addition_sv.argmax(1) == ref_preds).sum().item()
                total_pixels += ref_preds.numel()
            elif task_type == "multilabel":
                ref_preds = ref_sv.argmax(1)
                transfer_agree += (transfer_sv.argmax(1) == ref_preds).sum().item()
                addition_agree += (addition_sv.argmax(1) == ref_preds).sum().item()
            else:
                transfer_agree += (transfer_sv.argmax(1) == ref_preds).sum().item()
                addition_agree += (addition_sv.argmax(1) == ref_preds).sum().item()

    results = {}
    if with_labels:
        if task_type == "segmentation":
            all_seg_preds = torch.cat(peeking_preds_list)
            all_seg_labels = torch.cat(seg_labels_list)
            results['peeking_acc'] = compute_miou(all_seg_preds, all_seg_labels, num_classes, ignore_index)
            results['transfer_acc'] = compute_miou(torch.cat(transfer_preds_list), all_seg_labels, num_classes, ignore_index)
            results['addition_acc'] = compute_miou(torch.cat(addition_preds_list), all_seg_labels, num_classes, ignore_index)
            results['ens_acc'] = compute_miou(torch.cat(ens_preds_list), all_seg_labels, num_classes, ignore_index)
        elif task_type == "multilabel":
            all_lbls = torch.cat(labels_list)
            results['peeking_acc'] = _compute_map(torch.cat(peeking_logits_list), all_lbls)
            results['transfer_acc'] = _compute_map(torch.cat(transfer_logits_list), all_lbls)
            results['addition_acc'] = _compute_map(torch.cat(addition_logits_list), all_lbls)
            results['ens_acc'] = _compute_map(torch.cat(ens_logits_list), all_lbls)
        else:
            results['peeking_acc'] = 100.0 * peeking_correct / total
            results['transfer_acc'] = 100.0 * transfer_correct / total
            results['addition_acc'] = 100.0 * addition_correct / total
            results['ens_acc'] = 100.0 * ens_correct / total
    denom = total_pixels if task_type == "segmentation" else total
    results['transfer_agree'] = 100.0 * transfer_agree / denom if denom > 0 else 0.0
    results['addition_agree'] = 100.0 * addition_agree / denom if denom > 0 else 0.0
    return results


def _compute_latent_loss(student_fused, teacher_out, latent_projectors, latent_reconstruct_modalities, device,
                         modality_masks=None, latent_masked_only=False):
    """Compute latent reconstruction loss: project student CLS+patch tokens to match teacher."""
    latent_loss = torch.tensor(0.0, device=device)
    for mod in latent_reconstruct_modalities:
        student_patches = student_fused[mod]['x_norm_patchtokens']
        teacher_patches = teacher_out[mod]['x_norm_patchtokens'].detach()
        student_cls = student_fused[mod]['x_norm_clstoken']
        teacher_cls = teacher_out[mod]['x_norm_clstoken'].detach()
        student_seq = torch.cat([student_cls.unsqueeze(1), student_patches], dim=1)
        projected_seq = latent_projectors[mod](student_seq)   # [B, 1+N, D]
        projected_cls = projected_seq[:, 0, :]
        projected_patches = projected_seq[:, 1:, :]
        if latent_masked_only and modality_masks is not None and mod in modality_masks:
            mask = modality_masks[mod].unsqueeze(-1).float()   # [B, N, 1], 1=masked
            n_masked = mask.sum().clamp(min=1)
            patch_loss = ((projected_patches - teacher_patches) ** 2 * mask).sum() / n_masked
        else:
            patch_loss = F.mse_loss(projected_patches, teacher_patches)
        latent_loss = latent_loss + F.mse_loss(projected_cls, teacher_cls) + patch_loss
    return latent_loss


def _compute_ce_loss(student_fused, labels, model, ce_fn, device):
    """Compute cross-entropy (or BCE) loss over all modality heads."""
    ce_loss = torch.tensor(0.0, device=device)
    mods = list(student_fused.keys())
    for mod in mods:
        ce_loss = ce_loss + ce_fn(model.get_modality_logits(student_fused, mod), labels)
    return ce_loss / len(mods)


def _labeled_batch_step(
    batch, model,
    active_losses, loss_weights, starting_modality, newmod_list,
    modality_bands_dict, ce_fn,
    task_type, label_key, device,
    use_mask_token=False,
):
    """Process one labeled (monomodal) batch. Returns (total_loss, loss_dict)."""
    evan = model.evan
    labels = batch[label_key].to(device)
    if task_type == "multilabel":
        labels = labels.float()
    unimodal_input = _make_batch(batch, modality_bands_dict, (starting_modality,), device)
    with torch.no_grad():
        prefusion_features = evan.forward_modality_specific_features(unimodal_input)

    for newmod in newmod_list:
        if use_mask_token:
            B = prefusion_features[starting_modality].shape[0]
            prefusion_features[newmod] = evan._hallucinate_with_mask_token(newmod, B, device)
        else:
            with torch.no_grad():
                src_seq = prefusion_features[starting_modality]
                key = f"{starting_modality}_to_{newmod}"
                prefusion_features[newmod] = evan._project_sequence(src_seq, key, newmod)

    # Detach all prefusion features so labeled CE only trains fusion blocks + classifier
    # head. Encoder, intermediate projectors, and masking transformers are trained by
    # unlabeled batches and should not be disrupted here.
    prefusion_features = {k: v.detach() for k, v in prefusion_features.items()}

    # Cross projector outputs [B, 1+n_patches, D] (no storage tokens) — fusion must
    # know these are hallucinated so it uses n_prefix=1 instead of 1+n_storage.
    hallucinated_mods = set(newmod_list)

    student_fused = evan.forward_fusion_from_modality_features(
        prefusion_features,
        hallucinated_modalities=hallucinated_mods,
    )

    total_loss = torch.tensor(0.0, device=device)
    ce_loss_val = 0.0

    if 'ce' in active_losses:
        ce_loss = _compute_ce_loss(student_fused, labels, model, ce_fn, device)
        total_loss = total_loss + loss_weights['ce'] * ce_loss
        ce_loss_val = ce_loss.item()

    return total_loss, {'latent': 0.0, 'prefusion': 0.0, 'distill': 0.0, 'ce': ce_loss_val}

def _unlabeled_batch_step(
    batch, model, teacher_classifier, latent_projectors,
    active_losses, loss_weights, starting_modality, newmod_list, all_modalities,
    latent_reconstruct_modalities, modality_bands_dict,
    token_mask_ratio, modality_dropout,
    distillation_temperature,
    effective_labeled_freq, task_type, device,
    dyn_teacher: bool = False,
    use_mask_token: bool = False,
    protect_lrm: bool = False,
    latent_masked_only: bool = False,
    unprotect_starting_mod: bool = False,
):
    """Process one unlabeled (multimodal) batch. Returns (total_loss, loss_dict)."""
    evan = model.evan
    full_multimodal_input = _make_batch(batch, modality_bands_dict, tuple(all_modalities), device)
    batch_size = next(iter(full_multimodal_input.values())).shape[0]
    num_patches = (evan.img_size // evan.patch_size) ** 2

    prefusion_features = evan.forward_modality_specific_features(full_multimodal_input)

    # Teacher targets + logits (one no_grad block)
    with torch.no_grad():
        teacher_input = {m: full_multimodal_input[m] for m in latent_reconstruct_modalities}
        teacher_out = teacher_classifier.evan.forward_features(teacher_input)
        teacher_logits = teacher_classifier({starting_modality: full_multimodal_input[starting_modality]})

    # Masking + prefusion loss (computed together to share projections and avoid leakage)
    modality_masks, masked_mod_features, modality_dropped, prefusion_loss = mask_input(
        evan, batch_size, num_patches,
        token_mask_ratio, all_modalities, prefusion_features, modality_dropout, device,
        protected_modalities=newmod_list if (effective_labeled_freq > 0 and not unprotect_starting_mod) else None,
        active_losses=active_losses,
        latent_reconstruct_modalities=latent_reconstruct_modalities,
        use_mask_token=use_mask_token,
        protect_lrm=protect_lrm,
    )

    # cross projector produces [B, 1+n_patches, D] for dropped mods — fusion needs to know prefix is smaller
    dropped_mods = {mod for mod, dropped in modality_dropped.items() if dropped}
    student_fused = evan.forward_fusion_from_modality_features(
        masked_mod_features,
        hallucinated_modalities=dropped_mods
    )

    total_loss = loss_weights['prefusion'] * prefusion_loss
    latent_loss_val = 0.0
    distill_loss_val = 0.0

    if 'latent' in active_losses:
        latent_loss = _compute_latent_loss(
            student_fused, teacher_out, latent_projectors,
            latent_reconstruct_modalities=latent_reconstruct_modalities,
            device=device,
            modality_masks=modality_masks,
            latent_masked_only=latent_masked_only,
        )
        total_loss = total_loss + loss_weights['latent'] * latent_loss
        latent_loss_val = latent_loss.item()

    if 'distill' in active_losses:
        distill_loss = torch.tensor(0.0, device=device)
        distill_count = 0

        if dyn_teacher:
            # Peeking uses unmasked prefusion_features — grounded in real starting_mod even when
            # starting_mod was dropped in this batch.
            with torch.no_grad():
                peeking_logits = model.predict_from_real_modalities(
                    prefusion_features, (starting_modality,), tuple(all_modalities),
                    use_mask_token=use_mask_token,
                )

            for mod in student_fused.keys():
                # starting_modality head → frozen unimodal teacher; newmod heads → peeking
                target = teacher_logits if mod == starting_modality else peeking_logits
                distill_loss = distill_loss + distillation_loss(
                    model.get_modality_logits(student_fused, mod), target, distillation_temperature, task_type=task_type,
                )
                distill_count += 1
        else:
            for mod in student_fused.keys():
                distill_loss = distill_loss + distillation_loss(
                    model.get_modality_logits(student_fused, mod), teacher_logits, distillation_temperature, task_type=task_type,
                )
                distill_count += 1

        if distill_count > 0:
            distill_loss = distill_loss / distill_count
        total_loss = total_loss + loss_weights['distill'] * distill_loss
        distill_loss_val = distill_loss.item()

    return (
        total_loss,
        {'latent': latent_loss_val, 'prefusion': prefusion_loss.item(), 'distill': distill_loss_val, 'ce': 0.0},
    )

def _run_periodic_eval(
    epoch, model, teacher_classifier,
    test_loader, val_unlabeled_loader, val_labeled_loader,
    modality_bands_dict, starting_modality, newmod_list, all_modalities,
    task_type, label_key, num_classes, ignore_index, device,
    checkpoint_selection, val_weights, best_val_metric, best_checkpoint_state,
    best_checkpoints, latent_projectors,
    use_mask_token=False, agree_ref='teacher',
):
    """Run periodic eval, update checkpoints. Returns updated state."""
    metric_label = "mIoU" if task_type == "segmentation" else ("mAP" if task_type == "multilabel" else "accuracy")

    test_results = evaluate_multimodal(
        model=model, loader=test_loader, device=device,
        modality_bands_dict=modality_bands_dict,
        starting_modality=starting_modality,
        newmod_modalities=newmod_list,
        all_modalities=all_modalities,
        task_type=task_type, label_key=label_key,
        with_labels=True, desc="Testing",
        num_classes=num_classes, ignore_index=ignore_index,
        use_mask_token=use_mask_token,
    )
    periodic_test_accs = {
        'transfer': test_results['transfer_acc'],
        'peeking':  test_results['peeking_acc'],
        'addition': test_results['addition_acc'],
        'ens':      test_results['ens_acc'],
    }
    print(f"  Transfer {metric_label}: {test_results['transfer_acc']:.2f}%")
    print(f"  Peeking {metric_label}:  {test_results['peeking_acc']:.2f}%")
    print(f"  Addition {metric_label}: {test_results['addition_acc']:.2f}%")
    print(f"  Ens {metric_label}:      {test_results['ens_acc']:.2f}%")
    wandb.log({
        f"test/transfer_{metric_label}": test_results['transfer_acc'],
        f"test/peeking_{metric_label}":  test_results['peeking_acc'],
        f"test/addition_{metric_label}": test_results['addition_acc'],
        f"test/ens_{metric_label}":      test_results['ens_acc'],
        "epoch": epoch + 1,
    })

    val_metrics = {}

    if val_unlabeled_loader is not None:
        val_mm_results = evaluate_multimodal(
            model=model, loader=val_unlabeled_loader, device=device,
            modality_bands_dict=modality_bands_dict,
            starting_modality=starting_modality,
            newmod_modalities=newmod_list,
            all_modalities=all_modalities,
            with_labels=False, desc="Val (multimodal)",
            task_type=task_type, num_classes=num_classes, ignore_index=ignore_index,
            use_mask_token=use_mask_token,
            teacher_classifier=teacher_classifier, agree_ref=agree_ref,
        )
        val_metrics['transfer_agreement'] = val_mm_results['transfer_agree']
        val_metrics['addition'] = val_mm_results['addition_agree']
        ref_label = "teacher" if agree_ref == 'teacher' else "peeking"
        print(f"  Val transfer agreement (with {ref_label}): {val_metrics['transfer_agreement']:.2f}%")
        print(f"  Val addition agreement (with {ref_label}): {val_metrics['addition']:.2f}%")

    if val_labeled_loader is not None:
        peeking_acc = evaluate_multimodal(
            model=model, loader=val_labeled_loader, device=device,
            modality_bands_dict=modality_bands_dict,
            starting_modality=starting_modality,
            newmod_modalities=newmod_list,
            all_modalities=all_modalities,
            with_labels=True, task_type=task_type, label_key=label_key,
            num_classes=num_classes, ignore_index=ignore_index,
            use_mask_token=use_mask_token,
        )['peeking_acc']
        val_metrics['peeking'] = peeking_acc
        print(f"  Val peeking {metric_label}: {peeking_acc:.2f}%")

    if agree_ref == 'peeking':
        # transfer_score = val_peek(val1) * transfer_agree_with_peeking(val2) / 100
        # Lower-bounds transfer/addition: if peeking is accurate AND agreement is high, both paths work.
        if 'peeking' in val_metrics and 'transfer_agreement' in val_metrics:
            val_metrics['transfer_score'] = val_metrics['peeking'] * val_metrics['transfer_agreement'] / 100.0
            print(f"  Val transfer_score (peek * transfer_agree / 100): {val_metrics['transfer_score']:.2f}%")
        if 'peeking' in val_metrics and 'addition' in val_metrics:
            val_metrics['addition_score'] = val_metrics['peeking'] * val_metrics['addition'] / 100.0
            print(f"  Val addition_score (peek * addition_agree / 100): {val_metrics['addition_score']:.2f}%")
    else:
        # Teacher agreement is a direct quality signal — no peeking weighting needed.
        if 'transfer_agreement' in val_metrics:
            val_metrics['transfer_score'] = val_metrics['transfer_agreement']
        if 'addition' in val_metrics:
            val_metrics['addition_score'] = val_metrics['addition']

    if 'transfer_score' in val_metrics and 'addition_score' in val_metrics:
        ens_addition = (val_metrics['transfer_score'] + val_metrics['addition_score']) / 2
        val_metrics['ens_addition'] = ens_addition
        print(f"  Val ens_addition (transfer_score + addition_score)/2: {ens_addition:.2f}%")

    if checkpoint_selection == 'combined':
        current_metric = sum(val_weights.get(k, 0) * val_metrics.get(k, 0) for k in val_weights)
    else:
        current_metric = val_metrics.get(checkpoint_selection, 0)

    if current_metric > best_val_metric:
        best_val_metric = current_metric
        best_checkpoint_state = {
            'model': copy.deepcopy(model.state_dict()),
            'latent_projectors': copy.deepcopy({k: v.state_dict() for k, v in latent_projectors.items()}),
            'epoch': epoch,
            'val_metrics': val_metrics,
        }
        print(f"  >> New best model checkpoint (val {checkpoint_selection}: {current_metric:.2f})")

    checkpoint_criteria = {
        'best_transfer':     'transfer_score',
        'best_peeking':      'peeking',
        'best_addition':     'addition_score',
        'best_ens_addition': 'ens_addition',
    }
    new_records = []
    for ckpt_name, metric_key in checkpoint_criteria.items():
        if metric_key in val_metrics and val_metrics[metric_key] > best_checkpoints[ckpt_name]['metric']:
            best_checkpoints[ckpt_name] = {
                'metric': val_metrics[metric_key],
                'epoch': epoch,
                'test_accs': periodic_test_accs.copy(),
            }
            new_records.append(f"{ckpt_name} (val {metric_key}: {val_metrics[metric_key]:.2f}%)")
            wandb.log({
                f'best/{ckpt_name}_test_transfer': periodic_test_accs['transfer'],
                f'best/{ckpt_name}_test_peeking': periodic_test_accs['peeking'],
                f'best/{ckpt_name}_test_addition': periodic_test_accs['addition'],
                f'best/{ckpt_name}_test_ens': periodic_test_accs['ens'],
                f'best/{ckpt_name}_epoch': epoch + 1,
                'epoch': epoch + 1,
            })

    if new_records:
        print(f"  >> New records at epoch {epoch+1}: {', '.join(new_records)}")
        print(f"     Test accs: transfer={periodic_test_accs['transfer']:.2f}%, "
              f"peeking={periodic_test_accs['peeking']:.2f}%, "
              f"addition={periodic_test_accs['addition']:.2f}%, "
              f"ens={periodic_test_accs['ens']:.2f}%")

    wandb.log({
        'val/peeking_acc': val_metrics.get('peeking', 0),
        'val/transfer_agreement': val_metrics.get('transfer_agreement', 0),
        'val/addition_agreement': val_metrics.get('addition', 0),
        'val/transfer_score': val_metrics.get('transfer_score', 0),
        'val/addition_score': val_metrics.get('addition_score', 0),
        'val/ens_addition': val_metrics.get('ens_addition', 0),
        'val/combined': sum(val_weights.get(k, 0) * val_metrics.get(k, 0) for k in val_weights),
        'epoch': epoch + 1
    })

    return best_val_metric, best_checkpoint_state, best_checkpoints, teacher_classifier


# SHOT_TRAINING!
def train_shot(
    model, train_loader, device, args, starting_modality, new_modality,
    latent_reconstruct_modalities: list[str],
    modality_bands_dict: dict = None,
    max_norm=1,
    distillation_temperature: float = 1.0,
    test_loader=None,
    eval_every_n_epochs: int = None,
    labeled_train_loader=None,
    labeled_frequency: float = 0.0,
    labeled_start_fraction: float = 0.0,
    active_losses: list[str] = None,
    loss_weights: dict = None,              # Per-loss scaling: {'latent': 0.5, ...}, default all 1.0
    weight_decay: float = 0.01,
    # Validation-based checkpoint selection
    val_unlabeled_loader=None,          # val2 (unlabeled multimodal) for peeking agreement
    val_labeled_loader=None,            # val1 (labeled monomodal) for peeking accuracy
    checkpoint_selection='combined',    # 'combined', 'peeking', 'transfer_score', 'addition_score'
    val_weights=None,                   # {'peeking': 0.5, 'transfer_score': 0.5}
    warmup_epochs: int = 1,             # Linear LR warmup epochs for cosine scheduler
    # Dataset options
    task_type: str = "classification",  # 'classification', 'multilabel', or 'segmentation'
    label_key: str = 'label',           # Batch key for labels
    num_classes: int = None,            # Required when task_type='segmentation', for mIoU computation
    ignore_index: int = -100,           # Label value to ignore in CE/mIoU (e.g. 19 for PASTIS void_label)
    unimodal_teacher=None,
    asym_lr_multiplier: float | None = None,  # If set, new components get lr * asym_lr_multiplier
    dyn_teacher: bool = False,
    use_mask_token: bool = False,
    protect_lrm: float = 0.0,
    latent_masked_only: bool = False,
    unprotect_starting_mod: bool = False,
    agree_ref: str = 'teacher',              # 'teacher' (default) or 'peeking'
):
    """
    End-to-end training with hybrid loss combining:
    - Latent reconstruction loss for latent_reconstruct_modalities (match frozen teacher features)
    - Bidirectional sequence projection loss (learn mappings between modality sequences)
    - Label distillation loss (train classifier heads with soft labels from teacher model)
    During modality dropout, uses projected sequences from available modalities.

    Mixed training mode (when labeled_train_loader is provided):
    - train_loader (train2): unlabeled multimodal data -> full SHOT losses with teacher distillation
    - labeled_train_loader (train1): labeled monomodal data -> CE loss with real labels + latent loss
    - labeled_frequency: probability of sampling from labeled_train_loader each iteration
    - labeled_start_fraction: fraction of training to complete before labeled mixing starts
        - 0.0: start labeled mixing from the beginning
        - 0.5: start labeled mixing at 50% of training
        - 1.0: never use labeled mixing
    """
    if use_mask_token and active_losses and 'prefusion' in active_losses:
        raise ValueError("use_mask_token=True is incompatible with 'prefusion' loss: "
                         "mask tokens do not use projectors, so there is nothing to supervise.")

    # Compute epoch at which labeled mixing starts
    batch_mising_start_epoch = int(args.epochs * labeled_start_fraction)
    protect_lrm_end_epoch = int(args.epochs * protect_lrm)

    print("\n" + "="*70)
    print(f"=== END TO END Delulu TRAINING===\n")
    print(f"  With Losses: {active_losses}")
    if "latent" in active_losses: print(f"  Latent modalities (feature matching): {latent_reconstruct_modalities}")
    if dyn_teacher:
        print(f"  Dynamic teacher distillation: enabled")
        print(f"    - starting_modality head: distills from student peeking (soft-vote)")
        print(f"    - newmod heads: distill from frozen unimodal teacher (unchanged)")
    if labeled_train_loader is not None and labeled_frequency > 0:
        print(f"  Mixed training mode: labeled_frequency={labeled_frequency:.2f}")
        print(f"    - labeled_start_fraction={labeled_start_fraction:.2f} (starts at epoch {batch_mising_start_epoch + 1}/{args.epochs})")
        print(f"    - Unlabeled batches (train2): latent + pre-fusion + distillation losses")
        print(f"    - Labeled batches (train1): CE + latent losses (newmod hallucinated)")

    print(f"  Active losses: {active_losses}")
    print(f"  Loss weights: {loss_weights}")

    all_modalities = [starting_modality, new_modality]

    if unimodal_teacher is None:
        unimodal_teacher = copy.deepcopy(model)
        unimodal_teacher.freeze_all()
        unimodal_teacher.eval()
        print(f"\nTeacher classifier created (frozen copy of model)")

    is_segmenter = hasattr(model, 'decoder_strategy')
    if is_segmenter:
        if model.decoder_strategy != 'ensemble':
            print(f"!! Converting decoder from {model.decoder_strategy} to ensemble mode for label distillation")
            model.switch_strategy('ensemble')
        for mod in all_modalities:
            if mod not in model.modality_decoders:
                model.instantiate_modality_decoder(mod)
    else:
        if model.classifier_strategy != 'ensemble':
            print(f"!! Converting classifier from {model.classifier_strategy} to ensemble mode for label distillation")
            model.switch_strategy('ensemble')
        for mod in all_modalities:
            if mod not in model.modality_classifiers:
                model.instantiate_modality_classifier(mod)

    model.freeze_all()
    model.set_requires_grad("all", clsreg=True, modality_encoders=True, mfla=False, msla=True, patch_embedders=True, head=True)
    model.set_requires_grad("backbone", blocks=True, norm=True)

    evan = model.evan
    embed_dim = evan.embed_dim

    trainable_latent_decoder = 0
    latent_decoders = nn.ModuleDict()
    if "latent" in active_losses:
        latent_decoders = create_latent_decoders(embed_dim, latent_reconstruct_modalities, device)
        trainable_latent_decoder = sum(p.numel() for p in latent_decoders.parameters())

    if use_mask_token:
        if hasattr(evan, 'projector_queries'):
            for p in evan.projector_queries.values():
                p.requires_grad = True
    else:
        evan.set_requires_grad("all", intermediate_projectors=True)

    trainable_in_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    trainable_in_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_intermediate_proj = sum(p.numel() for p in evan.intermediate_projectors.parameters())
    # trainable_in_model already includes intermediate_projectors (part of evan); don't double-count
    trainable_total = trainable_in_model + trainable_latent_decoder
    print(f"\nTrainable parameters: {trainable_total}")
    print(f"    Model (EVAN + Classifier): {trainable_in_model} ({100*trainable_in_model/(sum(p.numel() for p in model.parameters())):.2f}%)")
    print(f"      - EVAN backbone (excl. intermediate_projectors): {trainable_in_evan - trainable_intermediate_proj}")
    print(f"      - Intermediate projectors: {trainable_intermediate_proj}")
    print(f"    Latent decoder (CLS + Patch): {trainable_latent_decoder}")

    if asym_lr_multiplier is not None:
        # Identify "new" components that get a higher LR
        new_param_ids = set()

        for p in evan.intermediate_projectors.parameters():
            new_param_ids.add(id(p))
        if hasattr(evan, 'projector_queries'):
            for p in evan.projector_queries.parameters():
                new_param_ids.add(id(p))

        newmod_keys = [m for m in evan.supported_modalities if m != evan.starting_modality]
        mfla_dict = getattr(evan, 'modality_fusion_lora_adaptors', {})
        for newmod_key in newmod_keys:
            for subdict in [evan.patch_embedders, evan.modality_specific_layer_adaptors, mfla_dict]:
                if newmod_key in subdict:
                    for p in subdict[newmod_key].parameters():
                        new_param_ids.add(id(p))
            # modality_encodings and cls/storage tokens are ParameterDicts — values are plain tensors
            for tok_dict in [evan.cls_tokens, evan.storage_tokens, evan.modality_encodings]:
                if newmod_key in tok_dict:
                    new_param_ids.add(id(tok_dict[newmod_key]))

        if model.modality_heads is not None:
            for newmod_key in newmod_keys:
                if newmod_key in model.modality_heads:
                    for p in model.modality_heads[newmod_key].parameters():
                        new_param_ids.add(id(p))

        extra_new = []
        if 'latent' in active_losses:
            extra_new += list(latent_decoders.parameters())

        all_model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
        base_params = [p for p in all_model_params if id(p) not in new_param_ids]
        new_params  = [p for p in all_model_params if id(p) in new_param_ids] + extra_new

        print(f"Total trainable parameters: {sum(p.numel() for p in all_model_params) + sum(p.numel() for p in extra_new):,}")
        print(f"  Base LR group: {sum(p.numel() for p in base_params):,} params at lr={args.lr}")
        print(f"  High LR group: {sum(p.numel() for p in new_params):,} params at lr={args.lr * asym_lr_multiplier} ({asym_lr_multiplier}x)")

        optimizer = torch.optim.AdamW([
            {'params': base_params, 'lr': args.lr},
            {'params': new_params,  'lr': args.lr * asym_lr_multiplier},
        ], weight_decay=weight_decay)
    else:
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        if 'latent' in active_losses:
            params += list(latent_decoders.parameters())
        print(f"Total trainable parameters: {sum(p.numel() for p in params):,}")
        optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, warmup_epochs=warmup_epochs)

    assert task_type in ("classification", "multilabel", "segmentation"), f"Unknown task_type: {task_type!r}"
    starting_modality = evan.starting_modality
    newmod_list = [m for m in all_modalities if m != starting_modality]
    ce_fn = nn.BCEWithLogitsLoss() if task_type == "multilabel" else nn.CrossEntropyLoss(ignore_index=ignore_index)

    best_val_metric = -float('inf')
    best_checkpoint_state = None
    if val_weights is None:
        val_weights = {'peeking': 0.5, 'transfer_score': 0.5}
    best_checkpoints = {
        'best_transfer':     {'metric': -float('inf'), 'epoch': None, 'test_accs': None},
        'best_peeking':      {'metric': -float('inf'), 'epoch': None, 'test_accs': None},
        'best_addition':     {'metric': -float('inf'), 'epoch': None, 'test_accs': None},
        'best_ens_addition': {'metric': -float('inf'), 'epoch': None, 'test_accs': None},
    }

    teacher_baselines = {}
    for loader, set_name in zip([train_loader, val_labeled_loader, test_loader], ["adaptation_train", "val(labeled)", "test"]):
        if loader is not None:
            _, acc = evaluate_unimodal(
                unimodal_teacher, loader,
                nn.BCEWithLogitsLoss() if task_type == "multilabel" else nn.CrossEntropyLoss(ignore_index=ignore_index),
                device, modality_bands_dict, modalities_to_use=(starting_modality,),
                multilabel=(task_type == "multilabel"), label_key=label_key,
                segmentation=(task_type == "segmentation"), num_classes=num_classes, ignore_index=ignore_index,
            )
            metric_label = "mIoU" if task_type == "segmentation" else ("mAP" if task_type == "multilabel" else "accuracy")
            print(f"\nTeacher baseline {set_name} {metric_label} (starting modality only): {acc:.2f}%")
            teacher_baselines[set_name] = acc

    global_step = 0
    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()
        evan.train()
        latent_decoders.train()

        effective_labeled_freq = labeled_frequency if epoch >= batch_mising_start_epoch else 0.0
        effective_protect_lrm = epoch < protect_lrm_end_epoch

        epoch_losses = {'total': 0.0, 'latent': 0.0, 'prefusion': 0.0, 'distill': 0.0, 'ce': 0.0}
        train_count = labeled_count = unlabeled_count = 0

        if labeled_train_loader is not None and effective_labeled_freq > 0:
            total_batches = min(len(train_loader), len(labeled_train_loader))
        else:
            total_batches = len(train_loader)

        pbar = tqdm(
            mixed_batch_iterator(train_loader, labeled_train_loader, effective_labeled_freq),
            total=total_batches,
            desc=f"SHOT Epoch {epoch+1}/{args.epochs}"
        )

        for batch, is_labeled in pbar:
            if is_labeled:
                total_loss, loss_dict = _labeled_batch_step(
                    batch, model,
                    active_losses, loss_weights, starting_modality, newmod_list,
                    modality_bands_dict, ce_fn,
                    task_type, label_key, device,
                    use_mask_token=use_mask_token,
                )
                labeled_count += 1
            else:
                total_loss, loss_dict = _unlabeled_batch_step(
                    batch, model, unimodal_teacher, latent_decoders,
                    active_losses, loss_weights, starting_modality, newmod_list, all_modalities,
                    latent_reconstruct_modalities, modality_bands_dict,
                    args.token_mask_ratio, {
                        mod: (args.modality_dropout_startmod if mod == starting_modality else args.modality_dropout_newmod)
                             or args.modality_dropout
                        for mod in all_modalities
                    },
                    distillation_temperature,
                    effective_labeled_freq, task_type, device,
                    dyn_teacher=dyn_teacher,
                    use_mask_token=use_mask_token,
                    protect_lrm=effective_protect_lrm,
                    latent_masked_only=latent_masked_only,
                    unprotect_starting_mod=unprotect_starting_mod,
                )
                unlabeled_count += 1

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g['params']], max_norm=max_norm)
            optimizer.step()

            epoch_losses['total'] += total_loss.item()
            for k in ('latent', 'prefusion', 'distill', 'ce'):
                epoch_losses[k] += loss_dict[k]
            train_count += 1
            global_step += 1

            pbar.set_postfix({
                'loss': f"{total_loss.item():.4f}",
                'latent': f"{loss_dict['latent']:.4f}",
                'pre_fus': f"{loss_dict['prefusion']:.4f}",
                'distill': f"{loss_dict['distill']:.4f}",
                'ce': f"{loss_dict['ce']:.4f}",
                'L/U': f'{labeled_count}/{unlabeled_count}'
            })

            if global_step % 20 == 0:
                wandb.log({
                    'train_loss': total_loss.item(),
                    'latent_loss': loss_dict['latent'],
                    'pre_fusion': loss_dict['prefusion'],
                    'distill_loss': loss_dict['distill'],
                    'ce_loss': loss_dict['ce'],
                    'is_labeled': 1 if is_labeled else 0,
                    'effective_labeled_freq': effective_labeled_freq,
                    'grad_norm': grad_norm.item(),
                    'epoch': epoch + 1,
                    'lr': optimizer.param_groups[0]['lr']
                })

        # Epoch summary
        avg_total = epoch_losses['total'] / train_count
        avg_latent = epoch_losses['latent'] / train_count
        avg_prefusion = epoch_losses['prefusion'] / max(unlabeled_count, 1)
        avg_distill = epoch_losses['distill'] / max(unlabeled_count, 1)
        avg_ce = epoch_losses['ce'] / max(labeled_count, 1)
        labeled_ratio = labeled_count / train_count if train_count > 0 else 0

        scheduler.step()
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s):")
        print(f"  Train - Total: {avg_total:.4f}, Latent: {avg_latent:.4f}, Pre-fusion: {avg_prefusion:.4f}, Distill: {avg_distill:.4f}, CE: {avg_ce:.4f}")
        print(f"  Batches - Labeled: {labeled_count}, Unlabeled: {unlabeled_count}, Ratio: {labeled_ratio:.2f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        if eval_every_n_epochs is not None and test_loader is not None and (epoch + 1) % eval_every_n_epochs == 0:
            print(f"\n--- Periodic Evaluation at Epoch {epoch+1} ---")
            model.eval()

            (best_val_metric, best_checkpoint_state, best_checkpoints,
             unimodal_teacher) = _run_periodic_eval(
                epoch, model, unimodal_teacher,
                test_loader, val_unlabeled_loader, val_labeled_loader,
                modality_bands_dict, starting_modality, newmod_list, all_modalities,
                task_type, label_key, num_classes, ignore_index, device,
                checkpoint_selection, val_weights, best_val_metric, best_checkpoint_state,
                best_checkpoints, latent_decoders,
                use_mask_token=use_mask_token, agree_ref=agree_ref,
            )

            model.train()

    if best_checkpoint_state is not None:
        model.load_state_dict(best_checkpoint_state['model'])
        for k, v in best_checkpoint_state['latent_projectors'].items():
            latent_decoders[k].load_state_dict(v)
        print(f"\nRestored best checkpoint from epoch {best_checkpoint_state['epoch']+1}")
        print(f"  Val metrics: {best_checkpoint_state['val_metrics']}")

    print("\n=== Best Checkpoint Summary (test accuracies at best val) ===")
    best_checkpoint_summary = {}
    for ckpt_name, ckpt_data in best_checkpoints.items():
        if ckpt_data['test_accs'] is not None:
            test_accs = ckpt_data['test_accs']
            best_checkpoint_summary[ckpt_name] = {
                'epoch': ckpt_data['epoch'] + 1,
                'val_metric': ckpt_data['metric'],
                'test_transfer': test_accs['transfer'],
                'test_peeking': test_accs['peeking'],
                'test_addition': test_accs['addition'],
                'test_addition_ens': test_accs.get('ens', 0),
            }
            print(f"\n{ckpt_name} (epoch {ckpt_data['epoch']+1}, val={ckpt_data['metric']:.2f}%):")
            print(f"  test/transfer: {test_accs['transfer']:.2f}%")
            print(f"  test/peeking: {test_accs['peeking']:.2f}%")
            print(f"  test/addition: {test_accs['addition']:.2f}%")
            print(f"  test/ens: {test_accs.get('ens', 0):.2f}%")

    print("\n=== SHOT Training complete ===")
    return trainable_total, best_checkpoints, best_checkpoint_summary, teacher_baselines
