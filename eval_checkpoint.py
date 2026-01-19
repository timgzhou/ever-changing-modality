"""Evaluate an EVAN checkpoint using supervised probing (classifier only).

This script loads an EVAN checkpoint and evaluates it by training only the classifier
while keeping all other components frozen. It automatically detects available modalities
from the checkpoint config.

Usage:
    python eval_checkpoint.py --checkpoint path/to/checkpoint.pt
    python eval_checkpoint.py --checkpoint path/to/checkpoint.pt --num_epochs 16 --lr 0.001
"""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import EuroSAT
from torchvision import transforms
import argparse
import os

from evan_main import evan_small, evan_base, evan_large, EVANClassifier
from eurosat_data_utils import (
    DictTransform,
    ALL_BAND_NAMES,
    get_modality_bands_dict,
    create_multimodal_batch,
)
from train_utils import (
    load_split_indices,
    single_modality_training_loop,
    supervised_training_loop,
)


def main():
    parser = argparse.ArgumentParser(description='Evaluate EVAN checkpoint with supervised probing')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to EVAN checkpoint')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=8,
                        help='Number of epochs for supervised probing (default: 8)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--eval_split', type=str, default='train2', choices=['train1', 'train2', 'full'],
                        help='Training split to use for probing (default: train2)')
    parser.add_argument('--skip_multimodal', action='store_true',
                        help='Skip multimodal evaluation (only do single-modality)')
    parser.add_argument('--skip_single_modal', action='store_true')
    parser.add_argument('--evan_cls_strategy', default='ensemble')
    args = parser.parse_args()

    # Load checkpoint
    print(f"\n{'='*70}")
    print(f"Loading checkpoint: {args.checkpoint}")
    print(f"{'='*70}")

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    config = checkpoint['config']

    print(f"\nCheckpoint config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Extract model config
    model_type = config['model_type']
    newmod = config.get('newmod', None)  # May not exist for stage 0 checkpoints

    # Determine available modalities
    modalities = ['rgb']
    if newmod:
        modalities.append(newmod)
        modality_bands_dict = get_modality_bands_dict('rgb', newmod)
    else:
        modality_bands_dict = get_modality_bands_dict('rgb')

    print(f"\nDetected modalities: {modalities}")

    # Create datasets
    print("\n=== Creating datasets ===")
    bands_full = tuple(ALL_BAND_NAMES)
    resize_transform = DictTransform(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))

    train_dataset_full = EuroSAT(
        root='datasets',
        split='train',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    test_dataset = EuroSAT(
        root='datasets',
        split='test',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    # Get training split
    if args.eval_split == 'train1':
        train_indices = load_split_indices('datasets/eurosat-train1.txt', train_dataset_full)
    elif args.eval_split == 'train2':
        train_indices = load_split_indices('datasets/eurosat-train2.txt', train_dataset_full)
    else:  # full
        train_indices = list(range(len(train_dataset_full)))

    train_dataset = Subset(train_dataset_full, train_indices)
    print(f"Train samples ({args.eval_split}): {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Recreate model
    print("\n=== Recreating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[model_type]
    evan = model_fn(
        tz_fusion_time=config['tz_fusion_time'],
        tz_lora_rank=config['tz_lora_rank'],
        n_storage_tokens=config.get('n_storage_tokens', 4),
        device=device
    )

    # Create modality components if needed
    if newmod and newmod not in evan.patch_embedders:
        bands_newmod = config.get('bands_newmod', modality_bands_dict[newmod])
        num_newmod_channels = len(bands_newmod)
        print(f"  Creating {newmod} modality components ({num_newmod_channels} channels)...")
        evan.create_modality_components(newmod, num_newmod_channels)

    model = EVANClassifier(evan, num_classes=config['num_classes'], classifier_strategy='mean', device=device)
    model = model.to(device)

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded model weights from checkpoint")

    criterion = nn.CrossEntropyLoss()
    results = {}
    trained_classifiers = nn.ModuleDict()  # Store trained classifiers for ensemble evaluation

    if not args.skip_single_modal:
        # ========== Single-modality evaluations ==========
        for modality in modalities:
            print(f"\n{'='*70}")
            print(f"=== Single-modality evaluation: {modality.upper()} ===")
            print(f"{'='*70}")

            # Reload checkpoint to reset classifier
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)

            # Freeze all, unfreeze classifier only
            model.freeze_all()
            model.set_requires_grad('all', classifier=True)

            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

            train_acc, test_acc, best_test_acc, best_epoch = single_modality_training_loop(
                model, train_loader, test_loader, device,
                modality_bands_dict, criterion, optimizer, args.num_epochs,
                modality=modality, phase_name=f"{modality.upper()} eval"
            )

            # Save trained classifier for ensemble evaluation
            trained_classifiers[modality] = copy.deepcopy(model.classifier)

            results[modality] = {
                'train_acc': train_acc,
                'test_acc': test_acc,
                'best_test_acc': best_test_acc,
                'best_epoch': best_epoch
            }
            print(f"\n{modality.upper()} Result: train={train_acc:.2f}% test={test_acc:.2f}% best={best_test_acc:.2f}% (epoch {best_epoch})")

        # ========== Ensemble of single-modality classifiers ==========
        if len(modalities) > 1 and len(trained_classifiers) == len(modalities):
            print(f"\n{'='*70}")
            print(f"=== Ensemble evaluation (separately trained classifiers) ===")
            print(f"{'='*70}")

            # Evaluate ensemble on test set
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch in test_loader:
                    labels = batch['label'].to(device)

                    # Get logits from each modality's classifier and average
                    all_logits = []
                    for modality in modalities:
                        modal_input = create_multimodal_batch(
                            batch, modality_bands_dict=modality_bands_dict, modalities=(modality,)
                        )
                        modal_input = {k: v.to(device) for k, v in modal_input.items()}
                        # Get features for each modality
                        features = model.evan.forward_features(modal_input)
                        cls_token = features[modality]['x_norm_clstoken']
                        logits = trained_classifiers[modality](cls_token)
                        all_logits.append(logits)
                        _, predicted = logits.max(1)
                        single_mod_acc=predicted.eq(labels).sum().item()

                    # Average logits (ensemble)
                    ensemble_logits = torch.stack(all_logits, dim=0).mean(dim=0)

                    _, predicted = ensemble_logits.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()

            ensemble_acc = 100. * correct / total
            results['ensemble_single'] = {'test_acc': ensemble_acc}
            print(f"\nEnsemble (separately trained) Result: test={ensemble_acc:.2f}%")

    # ========== Multimodal evaluation (if applicable) ==========
    if args.evan_cls_strategy=="ensemble": model.mean_to_ensemble()
    if args.evan_cls_strategy=="mean": model.ensemble_to_mean()
    
    if len(modalities) > 1 and not args.skip_multimodal:
        print(f"\n{'='*70}")
        print(f"=== Multimodal evaluation: {'+'.join(m.upper() for m in modalities)} ===")
        print(f"{'='*70}")

        # Reload checkpoint to reset classifier
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)

        # Freeze all, unfreeze classifier only
        model.freeze_all()
        model.set_requires_grad('all', classifier=True)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

        train_acc, _, _, test_acc_multi = supervised_training_loop(
            model=model,
            train_loader=train_loader,
            test_loader_full=test_loader,
            device=device,
            modality_bands_dict=modality_bands_dict,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.num_epochs,
            train_modalities=tuple(modalities),
            newmod=newmod,
            phase_name="Multimodal eval",
            eval_single_modalities=True,
        )

        results['multimodal'] = {
            'train_acc': train_acc,
            'test_acc': test_acc_multi,
        }
        print(f"\nMultimodal Result: train={train_acc:.2f}% test={test_acc_multi:.2f}%")

    # ========== Summary ==========
    print(f"\n{'='*70}")
    print("=== EVALUATION SUMMARY ===")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Eval split: {args.eval_split}")
    print(f"Epochs: {args.num_epochs}, LR: {args.lr}")
    print()

    for name, res in results.items():
        if 'best_test_acc' in res:
            print(f"  {name.upper():12s}: test={res['test_acc']:.2f}%  best={res['best_test_acc']:.2f}% (epoch {res['best_epoch']})")
        else:
            print(f"  {name.upper():12s}: test={res['test_acc']:.2f}%")


if __name__ == '__main__':
    main()

"""
# Basic usage
python eval_checkpoint.py --checkpoint checkpoints/my_checkpoint.pt

# With custom settings
python eval_checkpoint.py --checkpoint checkpoints/evan_eurosat_stage2_mae_20260118_080900.pt --num_epochs 4 --lr 0.001 --skip_multimodal # VRE

python eval_checkpoint.py --checkpoint checkpoints/evan_eurosat_stage2_mae_20260118_080900.pt --num_epochs 4 --lr 0.0003 --skip_single_modal  # SWIR

# python eval_checkpoint.py --checkpoint checkpoints/evan_eurosat_stage2_mae_20260119_150359.pt
"""