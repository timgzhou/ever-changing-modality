"""Baseline Distillation: Train student on new modality using teacher's soft labels."""

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

from evan_main import evan_small, evan_base, evan_large, EVANClassifier
from eurosat_data_utils import (
    get_loaders,
    get_modality_bands_dict,
    create_multimodal_batch,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def init_student_from_teacher(
    teacher_checkpoint_path: str,
    student_modality: str,
    student_n_chans: int,
    device: str = 'cpu',
) -> EVANClassifier:
    """
    Create a monomodal student EVANClassifier initialized from a teacher checkpoint.

    Copies the teacher's backbone weights (blocks, norm) and modality-specific
    components (MSLA, MFLA, CLS token, etc.) to the student's modality.
    The patch embedder is randomly initialized (different channel count),
    and the classifier is copied from the teacher.

    Args:
        teacher_checkpoint_path: Path to teacher checkpoint file
        student_modality: Modality name for the student (e.g., 'nir', 'swir')
        student_n_chans: Number of input channels for student modality
        device: Device to load model to

    Returns:
        EVANClassifier with student modality initialized from teacher weights
    """
    from evan_main import EVAN

    # Load teacher checkpoint
    checkpoint = torch.load(teacher_checkpoint_path, map_location=device)
    config = checkpoint['config']
    evan_config = config['evan_config'].copy()
    teacher_state_dict = checkpoint['model_state_dict']

    # Get teacher modality info
    teacher_modality = evan_config.get('starting_modality', 'rgb')

    if student_modality == teacher_modality:
        raise ValueError(
            f"Student modality ({student_modality}) must be different from "
            f"teacher modality ({teacher_modality}). Use from_checkpoint() instead."
        )

    print(f"=== Initializing student '{student_modality}' from teacher '{teacher_modality}' ===")

    # Update config for student modality
    evan_config['starting_modality'] = student_modality
    evan_config['starting_n_chans'] = student_n_chans
    # Remove multimodal info from config (student is monomodal)
    evan_config.pop('supported_modalities', None)
    evan_config.pop('supported_modalities_in_chans', None)

    # Create student EVAN model (monomodal, with student_modality)
    student_evan = EVAN(**evan_config, device=device)

    # === Copy backbone weights (shared across modalities) ===
    backbone_keys = []
    for key in teacher_state_dict.keys():
        if key.startswith('evan.blocks.') or key.startswith('evan.norm.') or \
           key == 'evan.mask_token' or key.startswith('evan.rope_embed'):
            backbone_keys.append(key)

    print(f"  Copying {len(backbone_keys)} backbone parameters from teacher")
    student_state_dict = student_evan.state_dict()
    for key in backbone_keys:
        # Remove 'evan.' prefix for EVAN state dict
        evan_key = key.replace('evan.', '', 1)
        if evan_key in student_state_dict:
            student_state_dict[evan_key] = teacher_state_dict[key]

    # === Copy modality-specific weights (teacher_modality -> student_modality) ===
    modality_components = [
        'modality_specific_layer_adaptors',
        'modality_fusion_lora_adaptors',
        'cls_tokens',
        'storage_tokens',
        'modality_encoders',
        'modality_specific_mask_tokens',
    ]

    copied_modality_keys = 0
    for component_name in modality_components:
        teacher_prefix = f'evan.{component_name}.{teacher_modality}'
        student_prefix = f'{component_name}.{student_modality}'

        for key in teacher_state_dict.keys():
            if key.startswith(teacher_prefix):
                suffix = key[len(teacher_prefix):]
                student_key = f'{student_prefix}{suffix}'
                if student_key in student_state_dict:
                    student_state_dict[student_key] = teacher_state_dict[key]
                    copied_modality_keys += 1

    print(f"  Copying {copied_modality_keys} modality-specific parameters "
          f"({teacher_modality} -> {student_modality})")
    print(f"  Patch embedder for '{student_modality}' randomly initialized "
          f"({student_n_chans} channels)")

    # Load the modified state dict into student EVAN
    student_evan.load_state_dict(student_state_dict, strict=True)

    # === Create classifier and copy weights from teacher ===
    student_model = EVANClassifier(
        evan_model=student_evan,
        num_classes=config['num_classes'],
        classifier_strategy=config['classifier_strategy'],
        factor=config['factor'],
        global_rep=config['global_rep'],
        device=device,
    )

    # Copy classifier weights
    classifier_state = {}
    for key in teacher_state_dict.keys():
        if key.startswith('classifier.'):
            classifier_state[key.replace('classifier.', '')] = teacher_state_dict[key]

    if classifier_state:
        student_model.classifier.load_state_dict(classifier_state)
        print(f"  Classifier weights copied from teacher")

    print(f"=== Student initialization complete ===")
    return student_model


def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """Compute KL divergence loss between student and teacher soft labels."""
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)


def feature_distillation_loss(student_features, teacher_features):
    """
    Compute MSE loss between student and teacher features (cls + patch tokens).

    Args:
        student_features: Dict with 'x_norm_clstoken' [B, D] and 'x_norm_patchtokens' [B, N, D]
        teacher_features: Dict with 'x_norm_clstoken' [B, D] and 'x_norm_patchtokens' [B, N, D]

    Returns:
        MSE loss averaged over cls and patch tokens
    """
    # CLS token loss
    cls_loss = F.mse_loss(student_features['x_norm_clstoken'], teacher_features['x_norm_clstoken'])

    # Patch tokens loss
    patch_loss = F.mse_loss(student_features['x_norm_patchtokens'], teacher_features['x_norm_patchtokens'])

    # Average the two losses
    return (cls_loss + patch_loss) / 2


def evaluate_ensemble(student_model, teacher_model, test_loader, device,
                      student_modality_bands_dict, teacher_modality_bands_dict,
                      student_modality, teacher_modality, ensemble_mode='avg'):
    """
    Evaluate ensemble of teacher and student models on test set.

    Args:
        student_model: Student EVAN classifier
        teacher_model: Teacher EVAN classifier
        test_loader: Test dataloader
        device: torch device
        student_modality_bands_dict: Band dict for student modality
        teacher_modality_bands_dict: Band dict for teacher modality
        student_modality: Student's modality name
        teacher_modality: Teacher's modality name
        ensemble_mode: 'avg' (average logits) or 'avg_softmax' (average probabilities)

    Returns:
        Test accuracy of ensemble predictions
    """
    student_model.eval()
    teacher_model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval ensemble ({ensemble_mode})"):
            labels = batch['label'].to(device)

            # Get student input and predictions
            student_input = create_multimodal_batch(
                batch, modality_bands_dict=student_modality_bands_dict, modalities=(student_modality,)
            )
            student_input = {k: v.to(device) for k, v in student_input.items()}
            student_logits = student_model(student_input)

            # Get teacher input and predictions
            teacher_input = create_multimodal_batch(
                batch, modality_bands_dict=teacher_modality_bands_dict, modalities=(teacher_modality,)
            )
            teacher_input = {k: v.to(device) for k, v in teacher_input.items()}
            teacher_logits = teacher_model(teacher_input)

            # Ensemble predictions
            if ensemble_mode == 'avg':
                # Average logits
                ensemble_logits = (student_logits + teacher_logits) / 2
                _, predicted = ensemble_logits.max(1)
            elif ensemble_mode == 'avg_softmax':
                # Average softmax probabilities
                student_probs = F.softmax(student_logits, dim=-1)
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                ensemble_probs = (student_probs + teacher_probs) / 2
                _, predicted = ensemble_probs.max(1)
            else:
                raise ValueError(f"Unknown ensemble_mode: {ensemble_mode}")

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100. * correct / total
    return accuracy


def evaluate_with_teacher_classifier(student_model, teacher_model, test_loader, device,
                                      student_modality_bands_dict, student_modality,
                                      teacher_modality, global_rep='clstoken'):
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
            labels = batch['label'].to(device)

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
                                           lr=1e-3, epochs=10, global_rep='clstoken'):
    """
    Train a new classifier with frozen backbone under supervision.

    Args:
        student_model: Student EVAN classifier (backbone frozen, classifier trainable)
        train_loader: Training dataloader
        test_loader: Test dataloader
        device: torch device
        student_modality_bands_dict: Band dict for student modality
        student_modality: Student's modality name
        lr: Learning rate for classifier
        epochs: Number of epochs to train classifier
        global_rep: 'clstoken' or 'mean_patch'

    Returns:
        Tuple of (final_test_acc, best_test_acc)
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Freeze everything, only train classifier
    student_model.freeze_all()
    student_model.set_requires_grad(student_modality, classifier=True)

    ce_criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()), lr=lr
    )
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    best_test_acc = 0

    for epoch in range(epochs):
        student_model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Classifier Train Epoch {epoch+1}/{epochs}")
        for batch in pbar:
            labels = batch['label'].to(device)

            student_input = create_multimodal_batch(
                batch, modality_bands_dict=student_modality_bands_dict, modalities=(student_modality,)
            )
            student_input = {k: v.to(device) for k, v in student_input.items()}

            optimizer.zero_grad()

            # Forward through frozen backbone
            with torch.no_grad():
                student_features = student_model.evan.forward_features(student_input)

            # Classify (only classifier has grad)
            logits = student_model.classify_from_features(student_features)
            loss = ce_criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train_loader)
        scheduler.step(train_loss)

        # Evaluate
        student_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in test_loader:
                labels = batch['label'].to(device)
                student_input = create_multimodal_batch(
                    batch, modality_bands_dict=student_modality_bands_dict, modalities=(student_modality,)
                )
                student_input = {k: v.to(device) for k, v in student_input.items()}
                logits = student_model(student_input)
                _, predicted = logits.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        test_acc = 100. * correct / total
        if test_acc > best_test_acc:
            best_test_acc = test_acc
        print(f"  Classifier Epoch {epoch+1}: Test Acc = {test_acc:.2f}% (best: {best_test_acc:.2f}%)")

    return test_acc, best_test_acc


def distillation_training_loop(
    student_model, teacher_model, train_loader, test_loader, device,
    student_modality_bands_dict, teacher_modality_bands_dict,
    optimizer, num_epochs, student_modality, teacher_modality,
    temperature=2.0, alpha=0.5, distillation_mode='regular',
    use_wandb=False, wandb_prefix=None, clip_norm=10
):
    """
    Training loop for knowledge distillation from teacher to student.

    Args:
        student_model: Student EVAN classifier (on student modality)
        teacher_model: Teacher EVAN classifier (on teacher modality)
        train_loader: Training dataloader
        test_loader: Test dataloader
        device: torch device
        student_modality_bands_dict: Band dict for student modality
        teacher_modality_bands_dict: Band dict for teacher modality
        optimizer: Optimizer for student
        num_epochs: Number of epochs
        student_modality: Student's modality name
        teacher_modality: Teacher's modality name
        temperature: Softmax temperature for distillation
        alpha: Weight for distillation loss (1-alpha for hard label loss if labels available)
        distillation_mode: One of 'regular' (KL on logits), 'with_guidance' (0.5 supervision + 0.5 distillation),
                          or 'feature' (MSE on cls+patch tokens)
        use_wandb: Whether to log to wandb
        wandb_prefix: Prefix for wandb metrics
        clip_norm: Max gradient norm for clipping

    Returns:
        Tuple of (train_acc, test_acc, best_test_acc, best_epoch)
    """
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from train_utils import evaluate

    teacher_model.eval()
    global_step = 0
    best_test_acc = 0
    best_epoch = 0
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=0, min_lr=1e-6)
    ce_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        student_model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        pbar = tqdm(train_loader, desc=f"Distill Epoch {epoch+1}/{num_epochs} [{student_modality.upper()}]")
        for batch in pbar:
            labels = batch['label'].to(device)

            # Get student input (student modality)
            student_input = create_multimodal_batch(
                batch, modality_bands_dict=student_modality_bands_dict, modalities=(student_modality,)
            )
            student_input = {k: v.to(device) for k, v in student_input.items()}

            # Get teacher input (teacher modality)
            teacher_input = create_multimodal_batch(
                batch, modality_bands_dict=teacher_modality_bands_dict, modalities=(teacher_modality,)
            )
            teacher_input = {k: v.to(device) for k, v in teacher_input.items()}

            optimizer.zero_grad()

            if distillation_mode == 'feature':
                # Feature distillation: MSE on cls + patch tokens
                with torch.no_grad():
                    teacher_features = teacher_model.evan.forward_features(teacher_input)
                    teacher_feat = teacher_features[teacher_modality]

                student_features = student_model.evan.forward_features(student_input)
                student_feat = student_features[student_modality]

                loss = feature_distillation_loss(student_feat, teacher_feat)
                # For accuracy tracking, still compute logits
                with torch.no_grad():
                    student_logits = student_model.classify_from_features(student_features)
            else:
                # Regular or with_guidance: use logit-based distillation
                with torch.no_grad():
                    teacher_logits = teacher_model(teacher_input)

                student_logits = student_model(student_input)
                distill_loss = distillation_loss(student_logits, teacher_logits, temperature)

                if distillation_mode == 'with_guidance':
                    # Fixed 0.5 supervision + 0.5 distillation
                    hard_loss = ce_criterion(student_logits, labels)
                    loss = 0.5 * distill_loss + 0.5 * hard_loss
                elif alpha < 1.0:
                    # Regular mode with optional hard label mixing
                    hard_loss = ce_criterion(student_logits, labels)
                    loss = alpha * distill_loss + (1 - alpha) * hard_loss
                else:
                    loss = distill_loss

            loss.backward()

            trainable_params = [p for p in student_model.parameters() if p.requires_grad]
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=clip_norm)

            optimizer.step()

            train_loss += loss.item()
            _, predicted = student_logits.max(1)
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

        # Evaluate student on test set
        test_loss, test_acc = evaluate(
            student_model, test_loader, ce_criterion, device,
            student_modality_bands_dict, modalities_to_use=(student_modality,)
        )
        scheduler.step(train_loss)

        if (epoch % 8 == 1) or (epoch + 1 == num_epochs):
            print(f"  Train ({student_modality.upper()}): Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"  Test ({student_modality.upper()}):  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}% (epoch {epoch+1}/{num_epochs})")

        if test_acc > best_test_acc:
            print(f"    New record: {test_acc:.2f} > previous {best_test_acc:.2f} at epoch {epoch+1}")
            best_test_acc = test_acc
            best_epoch = epoch + 1

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


def main():
    parser = argparse.ArgumentParser(description='Baseline Distillation: Train student using teacher soft labels')
    parser.add_argument('--teacher_path', type=str, required=True,
                        help='Teacher checkpoint name (will load from checkpoints/{teacher_path}_fft.pt)')
    parser.add_argument('--modality', type=str, required=True, choices=['rgb', 'vre', 'nir', 'swir', 'aw'],
                        help='Student modality (must be different from teacher)')
    parser.add_argument('--model', type=str, default='evan_small', choices=['evan_small', 'evan_base', 'evan_large'],
                        help='EVAN model size (default: evan_small)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='Number of training epochs (default: 1)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--tz_fusion_time', type=int, default=3,
                        help='n modality-independent layers before fusion')
    parser.add_argument('--tz_lora_rank', type=int, default=32,
                        help='rank of lora adaptors')
    parser.add_argument('--tz_modality_specific_layer_augmenter', type=str, default='fft', choices=['lora', 'fft'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat-distillation',
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
    parser.add_argument('--init_from_teacher', action='store_true',
                        help='Initialize student backbone and adaptors from teacher weights (instead of DINO pretrained)')
    args = parser.parse_args()

    # Load teacher checkpoint
    teacher_checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.teacher_path}_fft.pt')
    if not os.path.exists(teacher_checkpoint_path):
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_checkpoint_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load teacher model
    print(f"\n=== Loading teacher from {teacher_checkpoint_path} ===")
    teacher_model = EVANClassifier.from_checkpoint(teacher_checkpoint_path, device=device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Get teacher modality
    teacher_modality = teacher_model.evan.starting_modality
    print(f"Teacher modality: {teacher_modality}")

    # Validate that student modality is different from teacher
    if args.modality == teacher_modality:
        raise ValueError(f"Student modality ({args.modality}) must be different from teacher modality ({teacher_modality})")

    print(f"Student modality: {args.modality}")

    # Initialize wandb if enabled
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"distill_{args.distillation_mode}_{teacher_modality}->{args.modality}_{args.model}_{args.train_mode}"
        )

    print(f"Model: {args.model}, Batch size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # Band configuration
    student_modality_bands_dict = get_modality_bands_dict(args.modality)
    teacher_modality_bands_dict = get_modality_bands_dict(teacher_modality)
    bands_mod = student_modality_bands_dict[args.modality]

    # Create datasets
    print("\n=== Creating datasets ===")
    train1_loader, train2_loader, test_loader = get_loaders(args.batch_size, args.num_workers)

    # Create student EVAN model
    print("\n=== Creating student EVAN model ===")

    if args.init_from_teacher:
        # Initialize student from teacher weights (backbone + adaptors + classifier)
        print(f"Initializing student from teacher checkpoint: {teacher_checkpoint_path}")
        student_model = init_student_from_teacher(
            teacher_checkpoint_path=teacher_checkpoint_path,
            student_modality=args.modality,
            student_n_chans=len(bands_mod),
            device=device,
        )
        student_model = student_model.to(device)
    else:
        # Original path: Initialize from DINO pretrained weights
        model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
        evan = model_fn(
            tz_fusion_time=args.tz_fusion_time,
            tz_lora_rank=args.tz_lora_rank,
            tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
            n_storage_tokens=4,
            starting_modality=args.modality,
            starting_n_chans=len(bands_mod),
            device=device
        )

        # If student is RGB, reset patch embedder to random init (DINO pretrained weights are unfair)
        if args.modality == 'rgb':
            from evan.layers import PatchEmbed
            print("  Resetting RGB patch embedder to random initialization (removing DINO pretrained weights)")
            evan.patch_embedders['rgb'] = PatchEmbed(
                img_size=evan.img_size,
                patch_size=evan.patch_size,
                in_chans=len(bands_mod),
                embed_dim=evan.embed_dim,
                flatten_embedding=False,
            ).to(device)

        # Create classifier
        student_model = EVANClassifier(evan, num_classes=10, classifier_strategy="mean", global_rep=args.global_rep, device=device)
        student_model = student_model.to(device)

    # Freeze student backbone, train only specified components
    student_model.freeze_all()
    if args.train_mode == 'fft':
        student_model.set_requires_grad('backbone', blocks=True, norm=True)
        student_model.set_requires_grad(args.modality, msla=True, mfla=False, patch_embedders=True, clsreg=True, classifier=True)
        print(f"Mode=fft, Freezing lora paths, training full layers and classifier.")
    elif args.train_mode == 'adaptor':
        student_model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True, msla=True, mfla=True, classifier=True)
        print(f"Mode=adaptor, Freezing backbone, training embedder, lora or fft adaptors and classifier.")
    elif args.train_mode == 'probe':
        student_model.set_requires_grad(args.modality, classifier=True)
        print(f"Mode=Probe, Freezing backbone, only training classifier.")
    elif args.train_mode == 'emb+probe':
        student_model.set_requires_grad(args.modality, patch_embedders=True, classifier=True)
        print(f"Mode=Emb+Probe, Freezing backbone, training embedder(tokenizer) and classifier.")

    # Print parameter info
    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Training setup
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, student_model.parameters()), lr=args.lr)
    num_epochs = args.epochs

    print(f"\n=== Distillation training for {num_epochs} epochs ===")
    print(f"Strategy: Distill from {teacher_modality.upper()} teacher to {args.modality.upper()} student")
    print(f"Distillation mode: {args.distillation_mode}")
    print(f"Temperature: {args.temperature}, Alpha: {args.alpha}")

    # Run distillation training loop
    train_acc, test_acc, best_test_acc, best_epoch = distillation_training_loop(
        student_model, teacher_model, train1_loader, test_loader, device,
        student_modality_bands_dict, teacher_modality_bands_dict,
        optimizer, num_epochs, args.modality, teacher_modality,
        temperature=args.temperature, alpha=args.alpha,
        distillation_mode=args.distillation_mode,
        use_wandb=bool(args.wandb_project), wandb_prefix='distill'
    )

    print("\n=== Distillation Training complete ===")

    # Ensemble evaluation
    print("\n=== Ensemble Evaluation (Teacher + Student) ===")
    ensemble_acc_avg = evaluate_ensemble(
        student_model, teacher_model, test_loader, device,
        student_modality_bands_dict, teacher_modality_bands_dict,
        args.modality, teacher_modality, ensemble_mode='avg'
    )
    print(f"  Ensemble accuracy (avg logits): {ensemble_acc_avg:.2f}%")

    ensemble_acc_softmax = evaluate_ensemble(
        student_model, teacher_model, test_loader, device,
        student_modality_bands_dict, teacher_modality_bands_dict,
        args.modality, teacher_modality, ensemble_mode='avg_softmax'
    )
    print(f"  Ensemble accuracy (avg softmax): {ensemble_acc_softmax:.2f}%")

    # For feature mode, run additional evaluations
    teacher_classifier_acc = None
    supervised_classifier_acc = None
    supervised_classifier_best_acc = None

    if args.distillation_mode == 'feature':
        print("\n=== Feature Mode: Additional Evaluations ===")

        # 1. Evaluate using teacher's classifier directly
        print("\n--- Evaluation 1: Using teacher's classifier on student features ---")
        teacher_classifier_acc = evaluate_with_teacher_classifier(
            student_model, teacher_model, test_loader, device,
            student_modality_bands_dict, args.modality, teacher_modality,
            global_rep=args.global_rep
        )
        print(f"  Test accuracy (teacher classifier): {teacher_classifier_acc:.2f}%")

        # 2. Train classifier with frozen backbone under supervision
        print("\n--- Evaluation 2: Training classifier with frozen backbone (supervised) ---")
        supervised_classifier_acc, supervised_classifier_best_acc = train_classifier_with_frozen_backbone(
            student_model, train1_loader, test_loader, device,
            student_modality_bands_dict, args.modality,
            lr=args.lr, epochs=min(args.epochs, 10), global_rep=args.global_rep
        )
        print(f"  Test accuracy (supervised classifier): {supervised_classifier_acc:.2f}% (best: {supervised_classifier_best_acc:.2f}%)")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.checkpoint_name:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'evan_distill_{teacher_modality}_to_{args.modality}_{timestamp}.pt')

    student_model.save_checkpoint(checkpoint_path)
    print(f"\n=== Distillation checkpoint saved to: {checkpoint_path} ===")
    print(f"Distillation Final metrics ({args.modality.upper()}):")
    print(f"  Train accuracy: {train_acc:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")
    print(f"  Ensemble accuracy (avg logits): {ensemble_acc_avg:.2f}%")
    print(f"  Ensemble accuracy (avg softmax): {ensemble_acc_softmax:.2f}%")
    if args.distillation_mode == 'feature':
        print(f"  Teacher classifier accuracy: {teacher_classifier_acc:.2f}%")
        print(f"  Supervised classifier accuracy: {supervised_classifier_acc:.2f}% (best: {supervised_classifier_best_acc:.2f}%)")

    filename = "res/baseline_distillation.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = ["model_type", "teacher_modality", "student_modality", "train_mode", "tz_lora_rank",
                  "tz_modality_specific_layer_augmenter", "learning_rate", "trainable_params",
                  "epoch", "temperature", "alpha", "distillation_mode", "test_accuracy", "best_test_accuracy(oracle)",
                  "best_epoch", "ensemble_acc_avg", "ensemble_acc_softmax",
                  "teacher_classifier_acc", "supervised_classifier_acc", "supervised_classifier_best_acc",
                  "saved_checkpoint", "global_rep", "teacher_checkpoint"]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([args.model, teacher_modality, args.modality, args.train_mode, args.tz_lora_rank,
                         args.tz_modality_specific_layer_augmenter, args.lr, trainable_params,
                         num_epochs, args.temperature, args.alpha, args.distillation_mode, f"{test_acc:.2f}",
                         f"{best_test_acc:.2f}", best_epoch,
                         f"{ensemble_acc_avg:.2f}", f"{ensemble_acc_softmax:.2f}",
                         f"{teacher_classifier_acc:.2f}" if teacher_classifier_acc else "",
                         f"{supervised_classifier_acc:.2f}" if supervised_classifier_acc else "",
                         f"{supervised_classifier_best_acc:.2f}" if supervised_classifier_best_acc else "",
                         checkpoint_path, args.global_rep, teacher_checkpoint_path])

    # Finish wandb run
    if args.wandb_project:
        wandb.finish()
    return checkpoint_path


if __name__ == '__main__':
    main()
