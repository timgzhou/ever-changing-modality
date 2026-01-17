"""Stage 2: Train new modality embedder and modality-specific LoRAs on train2 split.

Two modes:
- supervised: Train all new modality components (embedder, modality-specific LoRAs, fusion LoRAs, classifier)
- mae: Train only modality embedder and modality-specific LoRAs via MAE reconstruction (requires stage 3 for fusion)
"""

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
    supervised_finetune_phase,
    train_mae_phase,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Stage 2: Train new modality embedder and modality-specific LoRAs')
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Path to stage 1 checkpoint (required)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--stage2_ft_epochs', type=int, default=4,
                        help='Number of fine-tuning epochs for supervised mode (default: 4)')
    parser.add_argument('--stage2_lr', type=float, default=1e-3,
                        help='Learning rate for supervised mode (default: 1e-3)')
    parser.add_argument('--stage2_train_method', type=str, default='supervised',
                        choices=['supervised', 'mae'],
                        help='Stage 2 training method: supervised (all components) or mae (embedder + modality-specific only)')
    parser.add_argument('--stage2_mae_epochs', type=int, default=4,
                        help='Number of MAE epochs for mae mode (default: 4)')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.85,
                        help='Mask ratio for MAE training (default: 0.85)')
    parser.add_argument('--mae_lr', type=float, default=0.00001,
                        help='Learning rate for MAE phase (default: 0.00001)')
    parser.add_argument('--new_mod_group', type=str, default='vre', choices=['vre', 'nir', 'swir'],
                        help='New modality group to train')
    parser.add_argument('--freeze_rgb', action='store_true',
                        help='Freeze RGB classifier head during supervised training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat',
                        help='Wandb project name')
    args = parser.parse_args()

    # Load stage 1 checkpoint
    print(f"\n=== Loading Stage 1 checkpoint from: {args.stage1_checkpoint} ===")
    checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
    config = checkpoint['config']

    print(f"Stage 1 config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            config={**config, **vars(args)},
            name=f"stage2_{config['model_type']}_{args.new_mod_group}_{args.stage2_train_method}"
        )

    # Band configuration
    newmod = args.new_mod_group
    modality_bands_dict = get_modality_bands_dict('rgb', newmod)
    bands_rgb = modality_bands_dict['rgb']
    bands_newmod = modality_bands_dict[newmod]
    bands_full = tuple(ALL_BAND_NAMES)

    # Create datasets
    print("\n=== Creating datasets ===")

    resize_transform = DictTransform(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))

    train_dataset_full = EuroSAT(
        root='datasets',
        split='train',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    # Get indices for train2 split
    train2_indices = load_split_indices('datasets/eurosat-train2.txt', train_dataset_full)
    train2_dataset = Subset(train_dataset_full, train2_indices)
    print(f"Loaded {len(train2_indices)} samples from train2 split")

    # Test with all bands
    test_dataset_full = EuroSAT(
        root='datasets',
        split='test',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    print(f"Train samples (train2 split): {len(train2_dataset)}")
    print(f"Test samples: {len(test_dataset_full)}")

    # Create dataloaders
    train2_loader = DataLoader(train2_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader_full = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Recreate EVAN model with same config
    print("\n=== Recreating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[config['model_type']]
    evan = model_fn(
        tz_fusion_time=config['tz_fusion_time'],
        tz_lora_rank=config['tz_lora_rank'],
        tz_modality_specific_layer_augmenter=config['tz_modality_specific_layer_augmenter'],
        n_storage_tokens=4,
        device=device
    )

    # Create classifier
    model = EVANClassifier(evan, num_classes=config['num_classes'], classifier_strategy=config['fusion_strategy'], device=device)
    model = model.to(device)

    # For ensemble mode, pre-instantiate RGB classifier before loading state dict
    # so that the weights can be properly loaded
    if config['fusion_strategy'] == 'ensemble':
        model._instantiate_modality_classifier('rgb')

    # Load state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded model weights from stage 1 checkpoint")
    print(f"Stage 1 final accuracy - RGB: {checkpoint['test_acc_rgb']:.2f}%, Multi: {checkpoint['test_acc_multi']:.2f}%")

    # Training setup
    criterion = nn.CrossEntropyLoss()

    # ==================== STAGE 2 TRAINING ====================
    print("\n" + "="*70)
    print("=== STAGE 2: Training new modality components on train2 ===")
    print("="*70)

    if args.stage2_train_method == 'supervised':
        # ========== SUPERVISED TRAINING (trains everything) ==========
        print(f"\nUsing supervised training method (trains all new modality components)")

        # Run supervised fine-tuning phase (trains all new modality components)
        optimizer_stage2, train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi = supervised_finetune_phase(
            model, evan, train2_loader, test_loader_full, device, args,
            newmod, modality_bands_dict, criterion, phase_name="Stage 2",
            modality_masking={'rgb': 0.4, newmod: 0.1},
            freeze_rgb=args.freeze_rgb, unfreeze_modality_specific=True,
            use_wandb=args.wandb, wandb_prefix='stage2_supervised'
        )

        # Save stage 2 checkpoint
        timestamp_stage2 = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path_stage2 = os.path.join(args.checkpoint_dir, f'evan_eurosat_stage2_supervised_{timestamp_stage2}.pt')

        checkpoint_stage2 = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer_stage2.state_dict(),
            'epoch': args.stage2_ft_epochs,
            'train_acc': train_acc,
            'test_acc_rgb': test_acc_rgb,
            'test_acc_newmod': test_acc_newmod,
            'test_acc_multi': test_acc_multi,
            'stage1_checkpoint': args.stage1_checkpoint,
            'config': {
                **config,
                'train_split': 'train2',
                'stage2_train_method': args.stage2_train_method,
                'bands_newmod': bands_newmod,
                'newmod': newmod,
                'stage2_ft_epochs': args.stage2_ft_epochs,
                'stage2_lr': args.stage2_lr,
            }
        }

        torch.save(checkpoint_stage2, checkpoint_path_stage2)
        print(f"\n=== Stage 2 checkpoint saved to: {checkpoint_path_stage2} ===")
        print(f"Stage 2 Final metrics:")
        print(f"  Train accuracy (RGB+{newmod.upper()}, train2): {train_acc:.2f}%")
        print(f"  Test accuracy (RGB only): {test_acc_rgb:.2f}%")
        print(f"  Test accuracy ({newmod.capitalize()} only): {test_acc_newmod:.2f}%")
        print(f"  Test accuracy (RGB+{newmod.upper()}): {test_acc_multi:.2f}%")

    elif args.stage2_train_method == 'mae':
        # ========== MAE TRAINING (trains embedder + modality-specific only) ==========
        print(f"\nUsing MAE training method (trains embedder + modality-specific LoRAs only)")
        print(f"Note: Run stage 3 after this to train fusion LoRAs and classifier")

        # MAE SSL training for modality embedder and modality-specific layers
        train_mae_phase(
            model, evan, train2_loader, test_loader_full, device, args,
            bands_newmod, newmod, use_wandb=args.wandb
        )

        # Save stage 2 MAE checkpoint
        timestamp_stage2 = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path_stage2 = os.path.join(args.checkpoint_dir, f'evan_eurosat_stage2_mae_{timestamp_stage2}.pt')

        checkpoint_stage2 = {
            'model_state_dict': model.state_dict(),
            'stage1_checkpoint': args.stage1_checkpoint,
            'config': {
                **config,
                'train_split': 'train2',
                'stage2_train_method': args.stage2_train_method,
                'mae_mask_ratio': args.mae_mask_ratio,
                'mae_lr': args.mae_lr,
                'bands_newmod': bands_newmod,
                'newmod': newmod,
                'stage2_mae_epochs': args.stage2_mae_epochs,
            }
        }

        torch.save(checkpoint_stage2, checkpoint_path_stage2)
        print(f"\n=== Stage 2 (MAE) checkpoint saved to: {checkpoint_path_stage2} ===")
        print(f"Trained components: {newmod} patch embedder, {newmod} modality-specific LoRAs")
        print(f"Next step: Run stage 3 with this checkpoint to train fusion LoRAs and classifier")

    print("\n" + "="*70)
    print("=== STAGE 2 TRAINING COMPLETE ===")
    print("="*70)
    print(f"Stage 1 checkpoint: {args.stage1_checkpoint}")
    print(f"Stage 2 checkpoint: {checkpoint_path_stage2}")

    # Finish wandb run
    if args.wandb:
        wandb.finish()

    return checkpoint_path_stage2


if __name__ == '__main__':
    main()


# Example usage:
# Supervised mode (trains everything, no stage 3 needed):
# python train_stage2.py --stage1_checkpoint checkpoints/evan_eurosat_stage1_*.pt --stage2_train_method supervised

# MAE mode (trains embedder + modality-specific, requires stage 3):
# python train_stage2.py --stage1_checkpoint checkpoints/evan_eurosat_stage1_*.pt --stage2_train_method mae --stage2_mae_epochs 16
# Then run stage 3:
# python train_stage3.py --stage2_checkpoint checkpoints/evan_eurosat_stage2_mae_*.pt --stage3_ft_epochs 4
