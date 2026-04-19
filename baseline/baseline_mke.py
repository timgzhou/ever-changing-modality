"""Multimodal Knowledge Expansion (MKE) Baseline (Xue et al., ICCV 2021).

A unimodal teacher (trained on labeled split 1) generates hard pseudo-labels on
unlabeled split 2.  A multimodal student is trained on those pseudo-labels using
weakly-augmented inputs.

Per the paper's equivalence result, the combined pseudo-label + consistency
regularisation loss reduces to plain CE against hard teacher pseudo-labels on
augmented student inputs — no consistency weight to tune.

For segmentation the pseudo-label mask is augmented with the same spatial
transform as the image to maintain correspondence.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import logging
import argparse
from datetime import datetime
import wandb
import csv
from tqdm import tqdm
import kornia.augmentation as K

from evan_main import evan_small, evan_base, evan_large, EVANClassifier, EvanSegmenter
from data_utils import get_loaders, create_multimodal_batch
from train_utils import make_scheduler, TrainMetricAccumulator, evaluate

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def make_augmentation(img_size: int):
    """Augmentation: random flips + random resized crop back to img_size.

    Image-only version (classification / multilabel).
    """
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomResizedCrop((img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        data_keys=["input"],
    )


def make_augmentation_with_mask(img_size: int):
    """Augmentation applied jointly to image + segmentation mask.

    Ensures the pseudo-label crop matches the student image crop exactly.
    """
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomResizedCrop((img_size, img_size), scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        data_keys=["input", "mask"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _n_chans(entry) -> int:
    if isinstance(entry, slice):
        return entry.stop - entry.start
    return len(entry)


def _evaluate_teacher(teacher_model, loader, ce_criterion, device, teacher_modality,
                       modality_bands_dict, multilabel, label_key, is_segmentation,
                       num_classes, ignore_index, split_name):
    """Evaluate teacher on a given split and print result."""
    _, metric = evaluate(
        teacher_model, loader, ce_criterion, device,
        modality_bands_dict, modalities_to_use=(teacher_modality,),
        multilabel=multilabel, label_key=label_key,
        segmentation=is_segmentation, num_classes=num_classes,
        ignore_index=ignore_index,
    )
    metric_name = "mIoU" if is_segmentation else ("mAP" if multilabel else "Acc")
    print(f"  Teacher {split_name} {metric_name}: {metric:.2f}%")
    return metric


def _compute_mke_teacher_agreement(
    student_model, teacher_model, val_loader, device,
    modality_bands_dict, student_modalities, teacher_modality,
    label_key, segmentation=False, ignore_index=-100,
) -> float:
    """Teacher-agreement on val2 for multimodal student checkpoint selection.

    Fraction of samples where student argmax == teacher argmax.
    Passes all student modalities to create_multimodal_batch (unlike the
    single-modality _compute_teacher_agreement in train_utils).
    """
    student_model.eval()
    teacher_model.eval()
    agree = total = 0

    with torch.no_grad():
        for batch in val_loader:
            student_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=tuple(student_modalities),
            )
            student_input = {k: v.to(device) for k, v in student_input.items()}
            teacher_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=(teacher_modality,),
            )
            teacher_input = {k: v.to(device) for k, v in teacher_input.items()}

            s_preds = student_model(student_input).argmax(dim=1)
            t_preds = teacher_model(teacher_input).argmax(dim=1)

            if segmentation:
                valid = t_preds != ignore_index
                agree += (s_preds == t_preds)[valid].sum().item()
                total += valid.sum().item()
            else:
                agree += (s_preds == t_preds).sum().item()
                total += s_preds.size(0)

    student_model.train()
    return 100.0 * agree / max(1, total)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MKE Baseline: multimodal student trained on teacher pseudo-labels')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'])
    parser.add_argument('--modalities', type=str, nargs='+', required=True,
                        help='Student modalities (must include teacher modality). '
                             'Example: --modalities s2 s1')
    parser.add_argument('--teacher_checkpoint', type=str, required=True,
                        help='Path to unimodal teacher checkpoint (trained on split 1)')
    parser.add_argument('--model', type=str, default='evan_base',
                        choices=['evan_small', 'evan_base', 'evan_large'])
    parser.add_argument('--use_dino_weights', action='store_true',
                        help='Init student backbone from DINOv2 ViT weights')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_time_steps', type=int, default=10,
                        help='Timestamps per PASTIS image (default: 10)')
    parser.add_argument('--tz_fusion_time', type=int, default=3)
    parser.add_argument('--tz_lora_rank', type=int, default=0)
    parser.add_argument('--tz_modality_specific_layer_augmenter', type=str, default='fft',
                        choices=['fft'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--global_rep', type=str, default='clstoken',
                        choices=['clstoken', 'mean_patch'])
    parser.add_argument('--train_mode', type=str, default='fft',
                        choices=['probe', 'adaptor', 'fft', 'emb+probe'])
    parser.add_argument('--val_per_epoch', type=int, default=1,
                        help='Run validation every N epochs (default: 1)')
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--results_csv', type=str, default=None)
    args = parser.parse_args()

    if args.results_csv is None:
        args.results_csv = f"res/baseline_mke_{args.dataset}.csv"

    teacher_checkpoint_path = args.teacher_checkpoint
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {args.dataset}, Student modalities: {args.modalities}")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Discover teacher modality from checkpoint
    # ------------------------------------------------------------------
    _ckpt_meta = torch.load(teacher_checkpoint_path, map_location='cpu')
    teacher_modality = _ckpt_meta['config']['evan_config'].get('starting_modality', 'rgb')
    normalization = _ckpt_meta.get('config', {}).get('normalization', 'zscore')
    del _ckpt_meta
    print(f"Teacher modality: {teacher_modality}")

    # Validate student modalities include teacher modality
    if teacher_modality not in args.modalities:
        parser.error(
            f"Student --modalities {args.modalities} must include teacher modality '{teacher_modality}'"
        )
    if len(args.modalities) < 2:
        parser.error("--modalities must specify at least 2 modalities for a multimodal student")

    # The 'new' modality (one the teacher doesn't have) for get_loaders
    other_modalities = [m for m in args.modalities if m != teacher_modality]
    new_modality = other_modalities[0]  # get_loaders needs one new modality to build the joint dataset

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_normalizer = None
    if normalization == 'div10000':
        from geobench_data_utils import make_div10000_normalizer
        data_normalizer = make_div10000_normalizer()
        print("Using div10000 normalizer (from teacher checkpoint config)")
    else:
        print("Using zscore normalizer (from teacher checkpoint config)")

    print("\n=== Creating datasets ===")
    train1_loader, _, train2_loader, val2_loader, test_loader, task_config = get_loaders(
        args.dataset, teacher_modality, args.batch_size, args.num_workers,
        data_normalizer=data_normalizer, num_time_steps=args.num_time_steps,
        new_modality=new_modality,
    )

    modality_bands_dict = task_config.modality_bands_dict
    is_segmentation = (task_config.task_type == 'segmentation')
    multilabel = task_config.multilabel
    label_key = task_config.label_key
    ignore_index = getattr(task_config, 'ignore_index', -100)
    metric_name = "mIoU" if is_segmentation else ("mAP" if multilabel else "Acc")
    num_classes = task_config.num_classes

    student_modalities = args.modalities
    all_n_chans = [_n_chans(modality_bands_dict[m]) for m in student_modalities]

    print(f"  train1 (labeled):   {len(train1_loader.dataset)} samples")
    print(f"  train2 (unlabeled): {len(train2_loader.dataset)} samples")
    print(f"  Student modalities: {student_modalities} ({all_n_chans} channels)")
    print(f"  Task: {task_config.task_type}, Metric: {metric_name}")

    # ------------------------------------------------------------------
    # Load teacher
    # ------------------------------------------------------------------
    print(f"\n=== Loading teacher from {teacher_checkpoint_path} ===")
    if is_segmentation:
        teacher_model = EvanSegmenter.from_checkpoint(teacher_checkpoint_path, device=device)
    else:
        teacher_model = EVANClassifier.from_checkpoint(teacher_checkpoint_path, device=device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    print(f"Teacher modality: {teacher_model.evan.starting_modality}")

    # ------------------------------------------------------------------
    # Teacher baseline evaluation (train1, train2, test)
    # ------------------------------------------------------------------
    if is_segmentation:
        _ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif multilabel:
        _ce = nn.BCEWithLogitsLoss()
    else:
        _ce = nn.CrossEntropyLoss()

    print(f"\n=== Teacher baseline ({teacher_modality.upper()}) ===")
    teacher_train1_metric = _evaluate_teacher(
        teacher_model, train1_loader, _ce, device, teacher_modality,
        modality_bands_dict, multilabel, label_key, is_segmentation,
        num_classes, ignore_index, split_name="train1",
    )
    teacher_train2_metric = _evaluate_teacher(
        teacher_model, train2_loader, _ce, device, teacher_modality,
        modality_bands_dict, multilabel, label_key, is_segmentation,
        num_classes, ignore_index, split_name="train2",
    )
    teacher_test_metric = _evaluate_teacher(
        teacher_model, test_loader, _ce, device, teacher_modality,
        modality_bands_dict, multilabel, label_key, is_segmentation,
        num_classes, ignore_index, split_name="test",
    )

    # ------------------------------------------------------------------
    # Student model
    # ------------------------------------------------------------------
    print("\n=== Creating student model ===")

    # S2 modality: pass RGB-band positions so DINO patch-embed init seeds correct channels
    _S2_RGB_INDICES = {
        'eurosat': [3, 2, 1],
        'benv2':   [3, 2, 1],
        'dfc2020': [3, 2, 1],
        'pastis':  [2, 1, 0],
    }
    primary_modality = student_modalities[0]
    rgb_in_s2_indices = (
        _S2_RGB_INDICES.get(args.dataset)
        if primary_modality == 's2' and args.use_dino_weights
        else None
    )

    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
    evan_model = model_fn(
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
        n_storage_tokens=4,
        starting_modality=student_modalities,
        starting_n_chans=all_n_chans,
        img_size=task_config.img_size,
        device=device,
        load_weights=args.use_dino_weights,
        rgb_in_s2_indices=rgb_in_s2_indices,
    )

    if is_segmentation:
        student_model = EvanSegmenter(
            evan_model, num_classes=num_classes,
            decoder_strategy="mean", device=device,
        )
    else:
        student_model = EVANClassifier(
            evan_model, num_classes=num_classes,
            classifier_strategy="mean", global_rep=args.global_rep, device=device,
        )
    student_model = student_model.to(device)

    # Freeze / unfreeze — apply modality-specific unfreezing for every student modality
    student_model.freeze_all()
    if args.train_mode == 'fft':
        student_model.set_requires_grad('backbone', blocks=True, norm=True)
        for mod in student_modalities:
            student_model.set_requires_grad(mod, msla=True, mfla=False,
                                            patch_embedders=True, clsreg=True, head=True)
        print("Mode=fft: training full backbone layers + head.")
    elif args.train_mode == 'adaptor':
        for mod in student_modalities:
            student_model.set_requires_grad(mod, patch_embedders=True, clsreg=True,
                                            msla=True, mfla=True, head=True)
        print("Mode=adaptor: training embedder, adaptors and head.")
    elif args.train_mode == 'probe':
        student_model.set_requires_grad('all', head=True)
        print("Mode=probe: training head only.")
    elif args.train_mode == 'emb+probe':
        student_model.set_requires_grad('all', patch_embedders=True, head=True)
        print("Mode=emb+probe: training embedder + head.")

    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # ------------------------------------------------------------------
    # Training setup
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs)

    if is_segmentation:
        ce_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif multilabel:
        ce_criterion = nn.BCEWithLogitsLoss()
    else:
        ce_criterion = nn.CrossEntropyLoss()

    # Augmentation (random flips + random resized crop back to img_size)
    aug = make_augmentation(task_config.img_size)
    aug_with_mask = make_augmentation_with_mask(task_config.img_size)


    # Channel counts per modality (for splitting combined tensor after aug)
    mod_chans = [_n_chans(modality_bands_dict[m]) for m in student_modalities]

    student_mods_str = '_'.join(student_modalities)

    # WandB
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"mke_{args.dataset}_{teacher_modality}_to_{student_mods_str}_{args.train_mode}",
        )
        wandb.log({
            'teacher/train1_' + metric_name: teacher_train1_metric,
            'teacher/train2_' + metric_name: teacher_train2_metric,
            'teacher/test_' + metric_name:   teacher_test_metric,
        })

    # Accumulator for train-step metrics
    accum = TrainMetricAccumulator(
        segmentation=is_segmentation,
        multilabel=multilabel,
        num_classes=num_classes,
        ignore_index=ignore_index,
    )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print(f"\n=== MKE training for {args.epochs} epochs ===")
    print(f"  Teacher: {teacher_modality.upper()} | Student: {[m.upper() for m in student_modalities]}")

    best_test_metric = 0.0
    best_epoch = 0
    best_agreement = -1.0
    valchecked_test_metric = None
    test_metric = 0.0

    trainable_param_list = [p for p in student_model.parameters() if p.requires_grad]

    for epoch in range(args.epochs):
        student_model.train()
        accum.reset()
        train_loss = 0.0

        pbar = tqdm(train2_loader, desc=f"MKE Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # ----------------------------------------------------------
            # 1. Teacher pseudo-labels (clean input, no grad)
            # ----------------------------------------------------------
            teacher_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=(teacher_modality,),
            )
            teacher_input = {k: v.to(device) for k, v in teacher_input.items()}

            with torch.no_grad():
                teacher_logits = teacher_model(teacher_input)

            if is_segmentation:
                pseudo_labels = teacher_logits.argmax(dim=1)          # [B, H, W]
            elif multilabel:
                pseudo_labels = (teacher_logits.sigmoid() > 0.5).float()  # [B, C]
            else:
                pseudo_labels = teacher_logits.argmax(dim=1)          # [B]

            # ----------------------------------------------------------
            # 2. Student multimodal input + augmentation
            # ----------------------------------------------------------
            student_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=tuple(student_modalities),
            )
            student_input = {k: v.to(device) for k, v in student_input.items()}

            if is_segmentation:
                # Stack all modality channels, augment jointly with pseudo-label mask
                combined = torch.cat([student_input[m] for m in student_modalities], dim=1)
                pseudo_mask = pseudo_labels.unsqueeze(1).float()      # [B, 1, H, W]
                combined_aug, pseudo_aug = aug_with_mask(combined, pseudo_mask)
                pseudo_labels_aug = pseudo_aug.squeeze(1).long()      # [B, H, W]

                # Split combined_aug back to per-modality
                student_input_aug = {}
                offset = 0
                for mod, nc in zip(student_modalities, mod_chans):
                    student_input_aug[mod] = combined_aug[:, offset:offset + nc]
                    offset += nc

                student_logits = student_model(student_input_aug)
                loss = ce_criterion(student_logits, pseudo_labels_aug)
                accum.update(student_logits.detach(), pseudo_labels_aug.detach())
            else:
                for mod in student_modalities:
                    student_input[mod] = aug(student_input[mod])
                student_logits = student_model(student_input)
                loss = ce_criterion(student_logits, pseudo_labels)
                accum.update(student_logits.detach(), pseudo_labels.detach())

            # ----------------------------------------------------------
            # 3. Backward
            # ----------------------------------------------------------
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_param_list, 10.0)
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        scheduler.step()
        train_loss /= len(train2_loader)
        train_metric = accum.compute()

        # ----------------------------------------------------------
        # Validation / test
        # ----------------------------------------------------------
        if (epoch + 1) % args.val_per_epoch == 0 or epoch == args.epochs - 1:
            _, test_metric = evaluate(
                student_model, test_loader, ce_criterion, device,
                modality_bands_dict, modalities_to_use=tuple(student_modalities),
                multilabel=multilabel, label_key=label_key,
                segmentation=is_segmentation, num_classes=num_classes,
                ignore_index=ignore_index,
            )
            print(f"Epoch {epoch+1}/{args.epochs} | loss={train_loss:.4f} | "
                  f"train {metric_name}={train_metric:.2f}% | test {metric_name}={test_metric:.2f}%")

            if test_metric > best_test_metric:
                best_test_metric = test_metric
                best_epoch = epoch + 1

            # Teacher-agreement checkpoint selection on val2
            # _compute_teacher_agreement only supports a single student modality;
            # use an inline version that passes all student modalities.
            agreement = _compute_mke_teacher_agreement(
                student_model, teacher_model, val2_loader, device,
                modality_bands_dict, student_modalities, teacher_modality,
                label_key, segmentation=is_segmentation, ignore_index=ignore_index,
            )
            print(f"  Val2 teacher-agreement: {agreement:.2f}%")
            if agreement > best_agreement:
                best_agreement = agreement
                valchecked_test_metric = test_metric

            if args.wandb_project:
                wandb.log({
                    'student/train_loss': train_loss,
                    f'student/train_{metric_name}': train_metric,
                    f'student/test_{metric_name}': test_metric,
                    'student/val2_teacher_agreement': agreement,
                    'epoch': epoch + 1,
                })
        else:
            print(f"Epoch {epoch+1}/{args.epochs} | loss={train_loss:.4f} | "
                  f"train {metric_name}={train_metric:.2f}%")

    print(f"\n=== MKE training complete ===")
    print(f"  Teacher train1 {metric_name}: {teacher_train1_metric:.2f}%")
    print(f"  Teacher train2 {metric_name}: {teacher_train2_metric:.2f}%")
    print(f"  Teacher test   {metric_name}: {teacher_test_metric:.2f}%")
    print(f"  Student test   {metric_name}: {test_metric:.2f}%")
    print(f"  Student best test {metric_name}: {best_test_metric:.2f}% (epoch {best_epoch})")

    # ------------------------------------------------------------------
    # CSV logging
    # ------------------------------------------------------------------
    filename = args.results_csv
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "model_type", "teacher_modality", "student_modalities", "train_mode",
        "tz_lora_rank", "tz_modality_specific_layer_augmenter",
        "learning_rate", "weight_decay", "trainable_params", "epochs",
        "use_dino_weights", "metric_name",
        "teacher_train1_metric", "teacher_train2_metric", "teacher_test_metric",
        "student_test_metric", "student_best_test_metric", "best_epoch",
        "best_val2_teacher_agreement", "valchecked_test_metric",
        "saved_checkpoint", "global_rep", "teacher_checkpoint",
    ]
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            args.model, teacher_modality, '+'.join(student_modalities), args.train_mode,
            args.tz_lora_rank, args.tz_modality_specific_layer_augmenter,
            args.lr, args.weight_decay, trainable_params, args.epochs,
            args.use_dino_weights, metric_name,
            f"{teacher_train1_metric:.2f}", f"{teacher_train2_metric:.2f}", f"{teacher_test_metric:.2f}",
            f"{test_metric:.2f}", f"{best_test_metric:.2f}", best_epoch,
            f"{best_agreement:.2f}",
            f"{valchecked_test_metric:.2f}" if valchecked_test_metric is not None else "",
            "", args.global_rep, teacher_checkpoint_path,
        ])
    print(f"Results appended to {filename}")

    if args.wandb_project:
        wandb.finish()


if __name__ == '__main__':
    main()


# DRYRUN examples
"""
# EuroSAT (classification): teacher on rgb, student gains vre
python -u baseline/baseline_mke.py --dataset eurosat \
    --modalities rgb vre \
    --teacher_checkpoint checkpoints/sft_evan_base_eurosat_rgb_fft_lr0.0001_20260414_012917.pt \
    --epochs 15

# BEN-v2 (multilabel): teacher on s2, student gains s1
python -u baseline/baseline_mke.py --dataset benv2 \
    --modalities s2 s1 \
    --teacher_checkpoint checkpoints/sft_evan_base_benv2_s2_fft_lr0.0005_20260414_012718.pt \
    --epochs 15

# DFC2020 (segmentation): teacher on s2, student gains s1
python -u baseline/baseline_mke.py --dataset dfc2020 \
    --modalities s2 s1 \
    --teacher_checkpoint checkpoints/sft_evan_base_dfc2020_s2_fft_lr0.0001_20260414_010633.pt \
    --epochs 15
"""
