"""Stage 1: Train EVAN on EuroSAT RGB (train1 split)."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import EuroSAT
from torchvision import transforms
import logging
import os
import argparse
from datetime import datetime
import wandb

from evan_main import evan_small, evan_base, evan_large, EVANClassifier
from eurosat_data_utils import (
    DictTransform,
    ALL_BAND_NAMES,
    get_modality_bands_dict
)
from train_utils import (
    load_split_indices,
    supervised_training_loop,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Train EVAN on EuroSAT RGB (train1 split)')
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
    parser.add_argument('--tz_fusion_time', type=int, default=3,
                        help='n modality-independent layers before fusion')
    parser.add_argument('--tz_lora_rank', type=int, default=64,
                        help='rank of lora adaptors')
    parser.add_argument('--fusion_strategy', type=str, default='mean', choices=['mean', 'ensemble'],
                        help='Fusion strategy: mean (avg CLS tokens) or ensemble (per-modality classifiers)')
    parser.add_argument('--new_mod_group', type=str, default='vre', choices=['vre', 'nir', 'swir'],
                        help='New modality group for evaluation (not trained in stage 1)')
    parser.add_argument('--tz_modality_specific_layer_augmenter', type=str, default='lora', choices=['lora', 'fft'],
                        help='lora or fft for modality-specific mae.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat',
                        help='Wandb project name')
    args = parser.parse_args()

    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"stage1_{args.model}_{args.new_mod_group}"
        )

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
    print(f"Note: train2 split is reserved for stage 2 training")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader_full = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create EVAN model with pretrained DINO weights
    print("\n=== Creating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
    evan = model_fn(
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
        n_storage_tokens=4,  # DINOv3 uses 4 register tokens
        device=device
    )

    # Create classifier
    model = EVANClassifier(evan, num_classes=10, fusion_strategy=args.fusion_strategy, device=device)
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
        model._instantiate_modality_classifier('rgb')
        for param in model.modality_classifiers['rgb'].parameters():
            param.requires_grad = True
        print(f"Using ensemble fusion: RGB classifier pre-instantiated")

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

    # Run Stage 1 training loop (RGB-only training)
    train_acc, test_acc_rgb, _, test_acc_multi = supervised_training_loop(
        model, train_loader, test_loader_full, device,
        modality_bands_dict, criterion, optimizer, num_epochs,
        train_modalities=('rgb',), newmod=newmod, phase_name="Stage 1",
        eval_newmod_only=False,
        use_wandb=args.wandb, wandb_prefix='stage1'
    )

    print("\n=== Stage 1 Training complete ===")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(args.checkpoint_dir, f'evan_eurosat_stage1_{timestamp}.pt')

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
            'tz_fusion_time': args.tz_fusion_time,
            'tz_lora_rank': args.tz_lora_rank,
            'tz_modality_specific_layer_augmenter': args.tz_modality_specific_layer_augmenter,
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

    # Finish wandb run
    if args.wandb:
        wandb.finish()

    return checkpoint_path


if __name__ == '__main__':
    main()


# Example usage:
# python train_stage1.py --epochs 10 --fusion_strategy ensemble
# python train_stage1.py --model evan_base --epochs 5 --new_mod_group vre
