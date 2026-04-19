"""Baseline Distillation: Train student on new modality using teacher's soft labels."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import os
import argparse
from datetime import datetime
import wandb
import csv
from tqdm import tqdm

from evan_main import evan_small, evan_base, evan_large, EVANClassifier, EvanSegmenter
from data_utils import get_loaders, create_multimodal_batch
from train_utils import _compute_map, compute_miou, evaluate

VALID_NEW_MODS = {
    'eurosat': ['vre', 'nir', 'swir', 'rgb', 's2'],
    'benv2':   ['s1', 's2'],
    'dfc2020': ['s1', 's2', 's2_rgb'],
}

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def _copy_evan_backbone_and_modality(teacher_state_dict, evan_config, student_modality,
                                      student_n_chans, teacher_modality, device):
    """
    Shared helper: build a fresh monomodal EVAN and copy backbone + modality-specific
    weights from the teacher state dict.  Returns the initialised EVAN model.
    """
    from evan_main import EVAN

    evan_config = evan_config.copy()
    evan_config['starting_modality'] = student_modality
    evan_config['starting_n_chans'] = student_n_chans
    evan_config.pop('supported_modalities', None)
    evan_config.pop('supported_modalities_in_chans', None)

    student_evan = EVAN(**evan_config, device=device)

    # --- backbone weights ---
    backbone_keys = [
        k for k in teacher_state_dict
        if k.startswith('evan.blocks.') or k.startswith('evan.norm.')
        or k == 'evan.mask_token' or k.startswith('evan.rope_embed')
    ]
    print(f"  Copying {len(backbone_keys)} backbone parameters from teacher")
    student_sd = student_evan.state_dict()
    for key in backbone_keys:
        evan_key = key.replace('evan.', '', 1)
        if evan_key in student_sd:
            student_sd[evan_key] = teacher_state_dict[key]

    # --- modality-specific weights (teacher_modality -> student_modality) ---
    modality_components = [
        'modality_specific_layer_adaptors',
        'modality_fusion_lora_adaptors',
        'cls_tokens',
        'storage_tokens',
        'modality_encoders',
        'modality_specific_mask_tokens',
    ]
    copied = 0
    for comp in modality_components:
        t_prefix = f'evan.{comp}.{teacher_modality}'
        s_prefix = f'{comp}.{student_modality}'
        for key in teacher_state_dict:
            if key.startswith(t_prefix):
                s_key = s_prefix + key[len(t_prefix):]
                if s_key in student_sd:
                    student_sd[s_key] = teacher_state_dict[key]
                    copied += 1

    print(f"  Copying {copied} modality-specific parameters ({teacher_modality} -> {student_modality})")
    print(f"  Patch embedder for '{student_modality}' randomly initialised ({student_n_chans} channels)")

    student_evan.load_state_dict(student_sd, strict=True)
    return student_evan


def init_student_from_teacher(
    teacher_checkpoint_path: str,
    student_modality: str,
    student_n_chans: int,
    task_type: str = 'classification',
    device: str = 'cpu',
):
    """
    Create a monomodal student (EVANClassifier or EvanSegmenter) initialised from a
    teacher checkpoint.  Backbone and modality-specific LoRA/FFT weights are copied;
    the patch embedder is randomly initialised.  The head is copied for classifiers and
    left at random init for segmenters (different spatial output shape).

    Args:
        teacher_checkpoint_path: Path to teacher checkpoint file.
        student_modality: Modality name for the student (e.g. 's1', 's2').
        student_n_chans: Number of input channels for student modality.
        task_type: 'classification', 'multilabel', or 'segmentation'.
        device: Device to load model to.

    Returns:
        EVANClassifier or EvanSegmenter with student modality initialised from teacher.
    """
    checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    config = checkpoint['config']
    evan_config = config['evan_config']
    teacher_state_dict = checkpoint['model_state_dict']
    teacher_modality = evan_config.get('starting_modality', 'rgb')

    if student_modality == teacher_modality:
        raise ValueError(
            f"Student modality ({student_modality}) must differ from teacher modality ({teacher_modality})."
        )

    print(f"=== Initialising student '{student_modality}' from teacher '{teacher_modality}' ===")

    student_evan = _copy_evan_backbone_and_modality(
        teacher_state_dict, evan_config, student_modality, student_n_chans, teacher_modality, device
    )

    if task_type == 'segmentation':
        student_model = EvanSegmenter(
            evan_model=student_evan,
            num_classes=config['num_classes'],
            decoder_strategy=config.get('decoder_strategy', 'mean'),
            device=device,
        )
        print("  Segmentation decoder randomly initialised (spatial head — no weight copying)")
    else:
        student_model = EVANClassifier(
            evan_model=student_evan,
            num_classes=config['num_classes'],
            classifier_strategy=config['classifier_strategy'],
            factor=config['factor'],
            global_rep=config['global_rep'],
            device=device,
        )
        # copy classifier head
        classifier_state = {
            k.replace('classifier.', ''): v
            for k, v in teacher_state_dict.items()
            if k.startswith('classifier.')
        }
        if classifier_state:
            student_model.classifier.load_state_dict(classifier_state)
            print("  Classifier weights copied from teacher")

    print("=== Student initialisation complete ===")
    return student_model


def evaluate_ensemble(student_model, teacher_model, test_loader, device,
                      student_modality_bands_dict, teacher_modality_bands_dict,
                      student_modality, teacher_modality, ensemble_mode='avg',
                      multilabel=False, label_key='label', segmentation=False,
                      num_classes=None, ignore_index=-100):
    """
    Evaluate ensemble of teacher and student models on test set.

    Args:
        student_model: Student EVAN classifier or EvanSegmenter
        teacher_model: Teacher EVAN classifier or EvanSegmenter
        test_loader: Test dataloader
        device: torch device
        student_modality_bands_dict: Band dict for student modality
        teacher_modality_bands_dict: Band dict for teacher modality
        student_modality: Student's modality name
        teacher_modality: Teacher's modality name
        ensemble_mode: 'avg' (average logits) or 'avg_softmax' (average probabilities)
        multilabel: If True, report mAP instead of accuracy
        label_key: Batch key for labels
        segmentation: If True, compute mIoU (logits are [B, C, H, W])
        num_classes: Required when segmentation=True
        ignore_index: Ignored label value for mIoU

    Returns:
        Ensemble metric: accuracy (%), mAP (%), or mIoU (%)
    """
    student_model.eval()
    teacher_model.eval()
    correct = 0
    total = 0
    all_outputs = [] if multilabel else None
    all_labels = [] if multilabel else None
    all_seg_preds = [] if segmentation else None
    all_seg_labels = [] if segmentation else None

    softmax_dim = 1  # works for both [B, C] and [B, C, H, W]

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval ensemble ({ensemble_mode})"):
            labels = batch[label_key].to(device)

            student_input = create_multimodal_batch(
                batch, modality_bands_dict=student_modality_bands_dict, modalities=(student_modality,)
            )
            student_input = {k: v.to(device) for k, v in student_input.items()}
            student_logits = student_model(student_input)

            teacher_input = create_multimodal_batch(
                batch, modality_bands_dict=teacher_modality_bands_dict, modalities=(teacher_modality,)
            )
            teacher_input = {k: v.to(device) for k, v in teacher_input.items()}
            teacher_logits = teacher_model(teacher_input)

            if ensemble_mode == 'avg':
                ensemble_logits = (student_logits + teacher_logits) / 2
            elif ensemble_mode == 'avg_softmax':
                if multilabel:
                    student_probs = torch.sigmoid(student_logits)
                    teacher_probs = torch.sigmoid(teacher_logits)
                else:
                    student_probs = F.softmax(student_logits, dim=softmax_dim)
                    teacher_probs = F.softmax(teacher_logits, dim=softmax_dim)
                ensemble_logits = (student_probs + teacher_probs) / 2
            else:
                raise ValueError(f"Unknown ensemble_mode: {ensemble_mode}")

            if segmentation:
                all_seg_preds.append(ensemble_logits.argmax(dim=1).cpu())
                all_seg_labels.append(labels.cpu())
            elif multilabel:
                all_outputs.append(ensemble_logits.cpu())
                all_labels.append(labels.cpu())
            else:
                _, predicted = ensemble_logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

    if segmentation:
        return compute_miou(torch.cat(all_seg_preds), torch.cat(all_seg_labels),
                            num_classes, ignore_index=ignore_index)
    if multilabel:
        return _compute_map(torch.cat(all_outputs), torch.cat(all_labels))
    return 100. * correct / total


def evaluate_with_teacher_classifier(student_model, teacher_model, test_loader, device,
                                      student_modality_bands_dict, student_modality,
                                      teacher_modality, global_rep='clstoken', label_key='label'):
    """
    Evaluate student features using teacher's classifier directly.

    Args:
        student_model: Student EVAN classifier
        teacher_model: Teacher EVAN classifier (use its classifier)
        test_loader: Test dataloader
        device: torch device
        student_modality_bands_dict: Band dict for student modality
        student_modality: Student's modality name
        teacher_modality: Teacher's modality name
        global_rep: 'clstoken' or 'mean_patch'

    Returns:
        Test accuracy using teacher's classifier on student features
    """
    student_model.eval()
    teacher_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Eval with teacher classifier"):
            labels = batch[label_key].to(device)

            student_input = create_multimodal_batch(
                batch, modality_bands_dict=student_modality_bands_dict, modalities=(student_modality,)
            )
            student_input = {k: v.to(device) for k, v in student_input.items()}

            # Get student features
            student_features = student_model.evan.forward_features(student_input)
            student_feat = student_features[student_modality]

            # Use teacher's classifier on student features
            if global_rep == 'clstoken':
                cls_token = student_feat['x_norm_clstoken']
            else:
                cls_token = student_feat['x_norm_patchtokens'].mean(1)

            logits = teacher_model.classifier(cls_token)
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def train_classifier_with_frozen_backbone(student_model, train_loader, test_loader, device,
                                           student_modality_bands_dict, student_modality,
                                           lr=1e-3, epochs=10, global_rep='clstoken',
                                           multilabel=False, label_key='label',
                                           val_loader=None, best_checkpoint_path=None,
                                           use_wandb=False, wandb_prefix=None,
                                           clip_norm=10, warmup_epochs=1,
                                           task_config=None):
    """
    Train a new classifier with frozen backbone under supervision.

    Val checkpoint criterion: val1 labeled metric (when val_loader is provided).

    Returns:
        Tuple of (final_test_metric, best_val_test_metric)
    """
    from train_utils import Trainer, make_criterion
    from data_utils import TaskConfig

    # Freeze everything, only train classifier
    student_model.freeze_all()
    student_model.set_requires_grad(student_modality, head=True)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()), lr=lr
    )

    if task_config is None:
        task_config = TaskConfig(
            dataset_name='', task_type='multilabel' if multilabel else 'classification',
            modality_a=student_modality, modality_b='',
            modality_a_channels=0, modality_b_channels=0,
            num_classes=0, multilabel=multilabel,
            label_key=label_key, modality_bands_dict=student_modality_bands_dict,
            img_size=0,
        )

    trainer = Trainer(
        student_model, optimizer, device, task_config,
        clip_norm=clip_norm, use_wandb=use_wandb, wandb_prefix=wandb_prefix,
        warmup_epochs=warmup_epochs,
    )
    return trainer.train_lp(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_epochs=epochs,
        modality=student_modality,
        best_checkpoint_path=best_checkpoint_path,
    )


def distillation_training_loop(
    student_model, teacher_model, train_loader, test_loader, device,
    student_modality_bands_dict, teacher_modality_bands_dict,
    optimizer, num_epochs, student_modality, teacher_modality,
    temperature=2.0, alpha=0.5, distillation_mode='regular', kl_type="kd",
    use_wandb=False, wandb_prefix=None, clip_norm=10,
    multilabel=False, label_key='label',
    segmentation=False, num_classes=None, ignore_index=-100,
    val_loader=None, best_checkpoint_path=None, warmup_epochs=1,
    task_config=None, student_modalities=None,
):
    """
    Knowledge distillation from teacher (teacher_modality) to student.

    student_modality: primary student modality (used for display / backwards compat).
    student_modalities: full tuple of student modality names; defaults to (student_modality,).

    Backward-compatible wrapper around Trainer.train_distillation.

    Returns:
        (train_metric, test_metric, best_test_metric, best_epoch, best_agreement)
    """
    from train_utils import Trainer
    from data_utils import TaskConfig

    if student_modalities is None:
        student_modalities = (student_modality,)

    if task_config is None:
        task_config = TaskConfig(
            dataset_name='',
            task_type='segmentation' if segmentation else ('multilabel' if multilabel else 'classification'),
            modality_a=student_modality, modality_b=teacher_modality,
            modality_a_channels=0, modality_b_channels=0,
            num_classes=num_classes or 0, multilabel=multilabel,
            label_key=label_key,
            modality_bands_dict={**student_modality_bands_dict, **teacher_modality_bands_dict},
            img_size=0, ignore_index=ignore_index,
        )

    trainer = Trainer(
        student_model, optimizer, device, task_config,
        clip_norm=clip_norm, use_wandb=use_wandb, wandb_prefix=wandb_prefix,
        warmup_epochs=warmup_epochs,
    )
    return trainer.train_distillation(
        train_loader=train_loader,
        val2_loader=val_loader,
        test_loader=test_loader,
        teacher_model=teacher_model,
        num_epochs=num_epochs,
        student_modality=student_modality,
        teacher_modality=teacher_modality,
        student_modality_bands_dict=student_modality_bands_dict,
        teacher_modality_bands_dict=teacher_modality_bands_dict,
        temperature=temperature,
        alpha=alpha,
        distillation_mode=distillation_mode,
        kl_type=kl_type,
        best_checkpoint_path=best_checkpoint_path,
        student_modalities=student_modalities,
    )


def main():
    parser = argparse.ArgumentParser(description='Baseline Distillation: Train student using teacher soft labels')
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'],
                        help='Dataset to train on (default: eurosat)')
    parser.add_argument('--teacher_checkpoint', type=str, required=True,
                        help='Path to teacher checkpoint file')
    parser.add_argument('--modalities', type=str, nargs='+', required=True,
                        help='Student modalities (each must be valid for --dataset; '
                             'pass multiple for a multimodal student, e.g. --modalities s1 s2)')
    parser.add_argument('--model', type=str, default='evan_base', choices=['evan_small', 'evan_base', 'evan_large'],
                        help='EVAN model size (default: evan_base)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--num_time_steps', type=int, default=10,
                        help='Timestamps to sample per PASTIS image before temporal aggregation (default: 10)')
    parser.add_argument('--tz_fusion_time', type=int, default=3,
                        help='n modality-independent layers before fusion')
    parser.add_argument('--tz_lora_rank', type=int, default=0,
                        help='rank of lora adaptors')
    parser.add_argument('--tz_modality_specific_layer_augmenter', type=str, default='fft', choices=['lora', 'fft'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='evan-distillation',
                        help='Wandb project name')
    parser.add_argument('--global_rep', type=str, default='clstoken', choices=['clstoken', 'mean_patch'])
    parser.add_argument('--train_mode', type=str, default='fft', choices=['probe', 'adaptor', 'fft', 'emb+probe'])
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--temperature', type=float, default=2.0,
                        help='Distillation temperature (default: 2.0)')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='Weight for distillation loss. alpha=1.0 means pure distillation (default: 1.0)')
    parser.add_argument('--distillation_mode', type=str, default='regular',
                        choices=['regular', 'with_guidance', 'feature'],
                        help='Distillation mode: regular (KL on logits), with_guidance (0.5 supervision + 0.5 distillation), feature (MSE on cls+patch tokens)')
    parser.add_argument('--kl_type', type=str, default='kd',
                        choices=['kd', 'ttm', 'wttm'])
    parser.add_argument('--init_from_teacher', action='store_true',
                        help='Initialize student backbone and adaptors from teacher weights (instead of DINO pretrained)')
    parser.add_argument('--results_csv', type=str, default=None,
                        help='Path to results CSV file (default: res/baseline_distillation_{dataset}.csv)')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Linear LR warmup epochs before cosine decay (default: 1)')
    args = parser.parse_args()

    # Validate each student modality against dataset
    valid_mods = VALID_NEW_MODS[args.dataset]
    for m in args.modalities:
        if m not in valid_mods:
            parser.error(f"--modalities {m!r} is not valid for --dataset {args.dataset}. "
                         f"Valid choices: {valid_mods}")
    student_modalities = tuple(args.modalities)
    primary_modality = student_modalities[0]
    student_label = '+'.join(student_modalities)

    # Default results CSV includes dataset name
    if args.results_csv is None:
        args.results_csv = f"res/baseline_distillation_{args.dataset}.csv"

    teacher_checkpoint_path = args.teacher_checkpoint
    if not os.path.exists(teacher_checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Read normalizer from teacher checkpoint config (must match training preprocessing)
    _ckpt_meta = torch.load(teacher_checkpoint_path, map_location='cpu')
    normalization = _ckpt_meta.get('config', {}).get('normalization', 'zscore')
    data_normalizer = None
    if normalization == 'div10000':
        from geobench_data_utils import make_div10000_normalizer
        data_normalizer = make_div10000_normalizer()
        print(f"Using div10000 normalizer (from teacher checkpoint config)")
    else:
        print(f"Using zscore normalizer (from teacher checkpoint config)")
    del _ckpt_meta

    # Create datasets (do this first so we know task_type before loading teacher)
    print("\n=== Creating datasets ===")
    # Load teacher checkpoint to discover teacher modality before building loaders,
    # so we can request all required modalities and get a complete modality_bands_dict.
    _ckpt_for_mod = torch.load(teacher_checkpoint_path, map_location='cpu')
    _teacher_modality = _ckpt_for_mod['config']['evan_config'].get('starting_modality', 'rgb')
    del _ckpt_for_mod
    # Request teacher modality as new_modality so the stacked batch contains all bands.
    # If teacher modality is already one of the student modalities, no extra request needed.
    _extra_mod = _teacher_modality if _teacher_modality not in student_modalities else None
    train1_loader, _, train2_loader, val2_loader, test_loader, task_config = \
        get_loaders(args.dataset, primary_modality, args.batch_size, args.num_workers,
                    data_normalizer=data_normalizer, num_time_steps=args.num_time_steps,
                    new_modality=_extra_mod)

    is_segmentation = (task_config.task_type == 'segmentation')
    multilabel = task_config.multilabel
    label_key = task_config.label_key
    ignore_index = getattr(task_config, 'ignore_index', -100)
    metric_name = "mIoU" if is_segmentation else ("mAP" if multilabel else "Acc")

    # Load teacher model — dispatch on task type
    print(f"\n=== Loading teacher from {teacher_checkpoint_path} ===")
    if is_segmentation:
        teacher_model = EvanSegmenter.from_checkpoint(teacher_checkpoint_path, device=device)
    else:
        teacher_model = EVANClassifier.from_checkpoint(teacher_checkpoint_path, device=device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    teacher_modality = teacher_model.evan.starting_modality
    print(f"Teacher modality: {teacher_modality}")
    print(f"Student modalities: {student_modalities}")

    if len(student_modalities) == 1 and student_modalities[0] == teacher_modality:
        raise ValueError(f"Single-modality student ({primary_modality}) must differ from teacher ({teacher_modality})")

    # Build per-modality band dicts from task_config
    modality_bands_dict = task_config.modality_bands_dict
    student_modality_bands_dict = {m: modality_bands_dict[m] for m in student_modalities}
    teacher_modality_bands_dict = {teacher_modality: modality_bands_dict[teacher_modality]}

    def _n_chans(entry):
        return entry.stop - entry.start if isinstance(entry, slice) else len(entry)

    all_student_n_chans = [_n_chans(modality_bands_dict[m]) for m in student_modalities]

    # Evaluate teacher on test split before distillation
    print(f"\n=== Teacher baseline ({teacher_modality.upper()}) on test split ===")
    if is_segmentation:
        _ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
    elif multilabel:
        _ce = nn.BCEWithLogitsLoss()
    else:
        _ce = nn.CrossEntropyLoss()
    _, teacher_test_metric = evaluate(
        teacher_model, test_loader, _ce, device,
        teacher_modality_bands_dict, modalities_to_use=(teacher_modality,),
        multilabel=multilabel, label_key=label_key,
        segmentation=is_segmentation, num_classes=task_config.num_classes,
        ignore_index=ignore_index,
    )
    print(f"  Teacher test {metric_name}: {teacher_test_metric:.2f}%")

    # Initialize wandb if enabled
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"{args.dataset}_distill_{args.distillation_mode}_{teacher_modality}->{student_label}_{args.train_mode}"
        )
        wandb.log({f"teacher_baseline/test_{metric_name}": teacher_test_metric})

    print(f"Model: {args.model}, Batch size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # Create student model (DINO-initialized, multimodal-capable like train_sft)
    print("\n=== Creating student model ===")

    if args.init_from_teacher:
        if len(student_modalities) > 1:
            raise ValueError("--init_from_teacher is only supported for single-modality students")
        print(f"Initializing student from teacher checkpoint: {teacher_checkpoint_path}")
        student_model = init_student_from_teacher(
            teacher_checkpoint_path=teacher_checkpoint_path,
            student_modality=primary_modality,
            student_n_chans=all_student_n_chans[0],
            task_type=task_config.task_type,
            device=device,
        )
        student_model = student_model.to(device)
    else:
        model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
        evan_model = model_fn(
            tz_fusion_time=args.tz_fusion_time,
            tz_lora_rank=args.tz_lora_rank,
            tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
            n_storage_tokens=4,
            starting_modality=list(student_modalities),
            starting_n_chans=all_student_n_chans,
            img_size=task_config.img_size,
            device=device,
            load_weights=True,
        )

        if is_segmentation:
            student_model = EvanSegmenter(
                evan_model, num_classes=task_config.num_classes,
                decoder_strategy="mean", device=device,
            )
        else:
            student_model = EVANClassifier(
                evan_model, num_classes=task_config.num_classes,
                classifier_strategy="mean", global_rep=args.global_rep, device=device,
            )
        student_model = student_model.to(device)

    # Freeze student backbone, train only specified components (all student modalities)
    student_model.freeze_all()
    if args.train_mode == 'fft':
        student_model.set_requires_grad('backbone', blocks=True, norm=True)
        student_model.set_requires_grad('all', patch_embedders=True, clsreg=True, msla=True, modality_encoders=True, head=True)
        print(f"Mode=fft: training full backbone layers + head.")
    elif args.train_mode == 'adaptor':
        student_model.set_requires_grad('all', patch_embedders=True, clsreg=True, msla=True, mfla=True, head=True)
        print(f"Mode=adaptor: training embedder, adaptors and classifier.")
    elif args.train_mode == 'probe':
        student_model.set_requires_grad('all', head=True)
        print(f"Mode=probe: training classifier only.")
    elif args.train_mode == 'emb+probe':
        student_model.set_requires_grad('all', patch_embedders=True, head=True)
        print(f"Mode=emb+probe: training embedder + classifier.")

    # Print parameter info
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Training setup
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.lr)
    num_epochs = args.epochs

    print(f"\n=== Distillation training for {num_epochs} epochs ===")
    print(f"Strategy: Distill from {teacher_modality.upper()} teacher to {student_label.upper()} student")
    print(f"Distillation mode: {args.distillation_mode}, Metric: {metric_name}")
    print(f"Temperature: {args.temperature}, Alpha: {args.alpha}")

    # Build task_config for distillation (student modality_bands_dict merged with teacher's)
    from data_utils import TaskConfig
    distill_task_config = TaskConfig(
        dataset_name=task_config.dataset_name,
        task_type=task_config.task_type,
        modality_a=primary_modality,
        modality_b=teacher_modality,
        modality_a_channels=all_student_n_chans[0],
        modality_b_channels=_n_chans(modality_bands_dict[teacher_modality]),
        num_classes=task_config.num_classes,
        multilabel=multilabel,
        label_key=label_key,
        modality_bands_dict={**student_modality_bands_dict, **teacher_modality_bands_dict},
        img_size=task_config.img_size,
        ignore_index=ignore_index,
    )

    # Run distillation training loop
    train_metric, test_metric, best_test_metric, best_epoch, best_agreement = distillation_training_loop(
        student_model, teacher_model, train2_loader, test_loader, device,
        student_modality_bands_dict, teacher_modality_bands_dict,
        optimizer, num_epochs, primary_modality, teacher_modality,
        temperature=args.temperature, alpha=args.alpha,
        distillation_mode=args.distillation_mode,
        use_wandb=bool(args.wandb_project), wandb_prefix='distill',
        multilabel=multilabel, label_key=label_key,
        segmentation=is_segmentation, num_classes=task_config.num_classes,
        ignore_index=ignore_index,
        val_loader=val2_loader, warmup_epochs=args.warmup_epochs,
        task_config=distill_task_config,
        student_modalities=student_modalities,
    )

    print("\n=== Distillation Training complete ===")

    # Ensemble evaluation
    print("\n=== Ensemble Evaluation (Teacher + Student) ===")
    ensemble_metric_avg = evaluate_ensemble(
        student_model, teacher_model, test_loader, device,
        student_modality_bands_dict, teacher_modality_bands_dict,
        primary_modality, teacher_modality, ensemble_mode='avg',
        multilabel=multilabel, label_key=label_key,
        segmentation=is_segmentation, num_classes=task_config.num_classes,
        ignore_index=ignore_index,
    )
    print(f"  Ensemble {metric_name} (avg logits): {ensemble_metric_avg:.2f}%")

    ensemble_metric_softmax = evaluate_ensemble(
        student_model, teacher_model, test_loader, device,
        student_modality_bands_dict, teacher_modality_bands_dict,
        primary_modality, teacher_modality, ensemble_mode='avg_softmax',
        multilabel=multilabel, label_key=label_key,
        segmentation=is_segmentation, num_classes=task_config.num_classes,
        ignore_index=ignore_index,
    )
    print(f"  Ensemble {metric_name} (avg softmax): {ensemble_metric_softmax:.2f}%")

    # For feature mode (classification only), run additional evaluations
    teacher_classifier_acc = None
    supervised_classifier_acc = None
    supervised_classifier_best_acc = None

    if args.distillation_mode == 'feature' and not multilabel and not is_segmentation:
        print("\n=== Feature Mode: Additional Evaluations ===")

        print("\n--- Evaluation 1: Using teacher's classifier on student features ---")
        teacher_classifier_acc = evaluate_with_teacher_classifier(
            student_model, teacher_model, test_loader, device,
            student_modality_bands_dict, primary_modality, teacher_modality,
            global_rep=args.global_rep, label_key=label_key,
        )
        print(f"  Test accuracy (teacher classifier): {teacher_classifier_acc:.2f}%")

        print("\n--- Evaluation 2: Training classifier with frozen backbone (supervised) ---")
        supervised_classifier_acc, supervised_classifier_best_acc = train_classifier_with_frozen_backbone(
            student_model, train1_loader, test_loader, device,
            student_modality_bands_dict, primary_modality,
            lr=args.lr, epochs=min(args.epochs, 10), global_rep=args.global_rep,
            multilabel=multilabel, label_key=label_key,
        )
        print(f"  Test accuracy (supervised classifier): {supervised_classifier_acc:.2f}% (best: {supervised_classifier_best_acc:.2f}%)")

    print(f"\nDistillation Final metrics ({student_label.upper()}):")
    print(f"  Train {metric_name}: {train_metric:.2f}%")
    print(f"  Test {metric_name}: {test_metric:.2f}%")
    print(f"  Ensemble {metric_name} (avg logits): {ensemble_metric_avg:.2f}%")
    print(f"  Ensemble {metric_name} (avg softmax): {ensemble_metric_softmax:.2f}%")
    if teacher_classifier_acc is not None:
        print(f"  Teacher classifier accuracy: {teacher_classifier_acc:.2f}%")
        print(f"  Supervised classifier accuracy: {supervised_classifier_acc:.2f}% (best: {supervised_classifier_best_acc:.2f}%)")

    filename = args.results_csv
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    fieldnames = ["model_type", "teacher_modality", "student_modality", "train_mode", "tz_lora_rank",
                  "tz_modality_specific_layer_augmenter", "learning_rate", "trainable_params",
                  "epoch", "temperature", "alpha", "distillation_mode", "kl_type",
                  "metric_name", "teacher_test_metric", "test_metric", "best_test_metric(oracle)", "best_epoch",
                  "best_val_agreement",
                  "ensemble_metric_logits", "ensemble_metric_softmax",
                  "teacher_classifier_acc", "supervised_classifier_acc", "supervised_classifier_best_acc",
                  "saved_checkpoint", "global_rep", "teacher_checkpoint"]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            args.model, teacher_modality, student_label, args.train_mode, args.tz_lora_rank,
            args.tz_modality_specific_layer_augmenter, args.lr, trainable_params,
            num_epochs, args.temperature, args.alpha, args.distillation_mode, args.kl_type, metric_name,
            f"{teacher_test_metric:.2f}", f"{test_metric:.2f}", f"{best_test_metric:.2f}", best_epoch,
            f"{best_agreement:.2f}" if best_agreement >= 0 else "",
            f"{ensemble_metric_avg:.2f}", f"{ensemble_metric_softmax:.2f}",
            f"{teacher_classifier_acc:.2f}" if teacher_classifier_acc is not None else "",
            f"{supervised_classifier_acc:.2f}" if supervised_classifier_acc is not None else "",
            f"{supervised_classifier_best_acc:.2f}" if supervised_classifier_best_acc is not None else "",
            "", args.global_rep, teacher_checkpoint_path,
        ])

    print(f"\nResults appended to {filename}")
    if args.wandb_project:
        wandb.finish()

if __name__ == '__main__':
    main()
