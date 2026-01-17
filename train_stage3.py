"""Stage 3: Train fusion LoRAs and classifier on train2 split.

This stage is used after stage 2 MAE training to complete the new modality integration.

Two modes:
- supervised: Train modality encoding + fusion LoRAs + classifier together (default)
- mae+supervised: Phase A trains encoding + fusion LoRAs via multi-modal MAE,
                  Phase B trains classifier only with frozen fusion components

It trains:
- New modality fusion LoRAs
- New modality encoding
- Classifier(s)

While keeping frozen:
- New modality patch embedder (trained in stage 2 MAE)
- New modality modality-specific LoRAs (trained in stage 2 MAE)
- RGB components
- Shared DINO backbone
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
    train_multimodal_fusion_mae_phase,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Stage 3: Train fusion LoRAs and classifier (after stage 2 MAE)')
    parser.add_argument('--stage2_checkpoint', type=str, required=True,
                        help='Path to stage 2 MAE checkpoint (required)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--stage3_mode', type=str, default='supervised',
                        choices=['supervised', 'mae+supervised'],
                        help='Stage 3 training mode (default: supervised)')
    parser.add_argument('--stage3_ft_epochs', type=int, default=4,
                        help='Number of fine-tuning epochs (default: 4)')
    parser.add_argument('--stage3_lr', type=float, default=1e-3,
                        help='Learning rate for supervised phase (default: 1e-3)')
    # MAE+supervised specific arguments
    parser.add_argument('--mae_fusion_epochs', type=int, default=4,
                        help='Epochs for Phase A fusion MAE (default: 4)')
    parser.add_argument('--mae_fusion_lr', type=float, default=1e-4,
                        help='Learning rate for Phase A fusion MAE (default: 1e-4)')
    parser.add_argument('--rgb_modality_drop_prob', type=float, default=0.5,
                        help='Probability to drop RGB modality during MAE (default: 0.5)')
    parser.add_argument('--rgb_token_mask_ratio', type=float, default=0.25,
                        help='Token mask ratio for RGB when present (default: 0.25)')
    parser.add_argument('--newmod_token_mask_ratio', type=float, default=0.75,
                        help='Token mask ratio for new modality (default: 0.75)')
    parser.add_argument('--freeze_rgb', action='store_true',
                        help='Freeze RGB classifier head during training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable wandb logging')
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat',
                        help='Wandb project name')
    args = parser.parse_args()

    # Load stage 2 checkpoint
    print(f"\n=== Loading Stage 2 checkpoint from: {args.stage2_checkpoint} ===")
    checkpoint = torch.load(args.stage2_checkpoint, map_location='cpu')
    config = checkpoint['config']

    # Verify this is a MAE checkpoint
    if config.get('stage2_train_method') != 'mae':
        print(f"WARNING: Stage 2 checkpoint used '{config.get('stage2_train_method')}' method.")
        print("Stage 3 is typically used after stage 2 MAE training.")
        print("If you used supervised training in stage 2, the model is already fully trained.")

    print(f"Stage 2 config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Get newmod from config
    newmod = config['newmod']
    bands_newmod = config['bands_newmod']

    # Initialize wandb if enabled
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            config={**config, **vars(args)},
            name=f"stage3_{config['model_type']}_{newmod}_{args.stage3_mode}"
        )

    # Band configuration
    modality_bands_dict = get_modality_bands_dict('rgb', newmod)
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
    if config['fusion_strategy'] == 'ensemble':
        model._instantiate_modality_classifier('rgb')

    # Pre-instantiate new modality components before loading state dict
    # This ensures the MAE-trained weights (patch embedder, modality-specific LoRAs) are loaded
    num_newmod_channels = len(bands_newmod)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.create_modality_components(newmod, num_newmod_channels)

    # Load state dict from checkpoint - this loads the MAE-trained components
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded model weights from stage 2 checkpoint")
    print(f"MAE-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")

    # Training setup
    criterion = nn.CrossEntropyLoss()

    # Create a temporary args object with stage3 parameters mapped to what supervised_finetune_phase expects
    class Stage3Args:
        def __init__(self, args):
            self.stage2_lr = args.stage3_lr  # supervised_finetune_phase uses stage2_lr
            self.stage2_ft_epochs = args.stage3_ft_epochs  # supervised_finetune_phase uses stage2_ft_epochs

    stage3_args = Stage3Args(args)

    # ==================== STAGE 3 TRAINING ====================

    if args.stage3_mode == 'supervised':
        # ========== SUPERVISED MODE ==========
        # Train modality encoding + fusion LoRAs + classifier together
        print("\n" + "="*70)
        print("=== STAGE 3: Supervised Training (encoding + fusion + classifier) ===")
        print("="*70)

        # Run supervised fine-tuning phase (trains fusion LoRAs + classifier)
        # unfreeze_modality_specific=False keeps the MAE-trained components frozen
        optimizer_stage3, train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi = supervised_finetune_phase(
            model, evan, train2_loader, test_loader_full, device, stage3_args,
            newmod, modality_bands_dict, criterion, phase_name="Stage 3",
            modality_masking=None,
            freeze_rgb=args.freeze_rgb, unfreeze_modality_specific=False,
            use_wandb=args.wandb, wandb_prefix='stage3_finetune'
        )

        checkpoint_suffix = 'supervised'

    elif args.stage3_mode == 'mae+supervised':
        # ========== MAE + SUPERVISED MODE ==========
        # Phase A: Train encoding + fusion LoRAs via multi-modal MAE
        # Phase B: Train classifier only (freeze fusion components)

        print("\n" + "="*70)
        print("=== STAGE 3: MAE+Supervised Mode ===")
        print("="*70)
        print("Phase A: Multi-modal Fusion MAE (train encoding + fusion LoRAs)")
        print("Phase B: Supervised Classifier (freeze fusion, train classifier only)")

        # ========== PHASE A: Multi-modal Fusion MAE ==========
        mae_decoder = train_multimodal_fusion_mae_phase(
            model, evan, train2_loader, test_loader_full, device, args,
            newmod, modality_bands_dict, use_wandb=args.wandb
        )

        # ========== PHASE B: Supervised Classifier Training ==========
        print("\n" + "="*70)
        print("=== Stage 3 Phase B: Supervised Classifier Training ===")
        print("="*70)

        # Freeze modality encoding and fusion LoRAs (trained in Phase A)
        evan.modality_encoders[newmod].requires_grad_(False)
        evan.modality_fusion_lora_adaptors[newmod].requires_grad_(False)
        print(f"  Frozen: {newmod.capitalize()} modality encoding (trained in Phase A)")
        print(f"  Frozen: {newmod.capitalize()} fusion LoRAs (trained in Phase A)")

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only classifier(s)
        if model.fusion_strategy == 'mean':
            for param in model.classifier.parameters():
                param.requires_grad = True
            print("  Unfroze: Classifier (mean fusion)")
        elif model.fusion_strategy == 'ensemble':
            # Ensure new modality classifier exists
            if newmod not in model.modality_classifiers:
                print(f"  Creating {newmod} classifier")
                model._instantiate_modality_classifier(newmod)

            for param in model.modality_classifiers[newmod].parameters():
                param.requires_grad = True
            print(f"  Unfroze: {newmod.capitalize()} classifier (ensemble mode)")

            if args.freeze_rgb:
                print("  Frozen: RGB classifier head")
            else:
                for param in model.modality_classifiers['rgb'].parameters():
                    param.requires_grad = True
                print("  Unfroze: RGB classifier head")

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nTrainable parameters for Phase B: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

        # Create optimizer for classifier only
        optimizer_stage3 = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.stage3_lr
        )

        print(f"\n=== Phase B Training for {args.stage3_ft_epochs} epochs ===")
        print(f"Learning rate: {args.stage3_lr}")

        # Import supervised training loop
        from train_utils import supervised_training_loop

        # Run training loop for classifier only
        train_acc, test_acc_rgb, test_acc_newmod, test_acc_multi = supervised_training_loop(
            model, train2_loader, test_loader_full, device,
            modality_bands_dict, criterion, optimizer_stage3, args.stage3_ft_epochs,
            train_modalities=('rgb', newmod), newmod=newmod, phase_name="Stage 3 Phase B",
            eval_newmod_only=True, modality_masking=None,
            use_wandb=args.wandb, wandb_prefix='stage3_classifier'
        )

        print(f"\n=== Stage 3 Phase B complete ===")
        checkpoint_suffix = 'mae_supervised'

    # Save stage 3 checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp_stage3 = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path_stage3 = os.path.join(args.checkpoint_dir, f'evan_eurosat_stage3_{checkpoint_suffix}_{timestamp_stage3}.pt')

    checkpoint_stage3 = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer_stage3.state_dict(),
        'epoch': args.stage3_ft_epochs,
        'train_acc': train_acc,
        'test_acc_rgb': test_acc_rgb,
        'test_acc_newmod': test_acc_newmod,
        'test_acc_multi': test_acc_multi,
        'stage2_checkpoint': args.stage2_checkpoint,
        'stage1_checkpoint': checkpoint.get('stage1_checkpoint'),
        'config': {
            **config,
            'stage3_mode': args.stage3_mode,
            'stage3_ft_epochs': args.stage3_ft_epochs,
            'stage3_lr': args.stage3_lr,
            'mae_fusion_epochs': args.mae_fusion_epochs if args.stage3_mode == 'mae+supervised' else None,
            'mae_fusion_lr': args.mae_fusion_lr if args.stage3_mode == 'mae+supervised' else None,
        }
    }

    torch.save(checkpoint_stage3, checkpoint_path_stage3)
    print(f"\n=== Stage 3 checkpoint saved to: {checkpoint_path_stage3} ===")
    print(f"Stage 3 Final metrics:")
    print(f"  Train accuracy (RGB+{newmod.upper()}, train2): {train_acc:.2f}%")
    print(f"  Test accuracy (RGB only): {test_acc_rgb:.2f}%")
    print(f"  Test accuracy ({newmod.capitalize()} only): {test_acc_newmod:.2f}%")
    print(f"  Test accuracy (RGB+{newmod.upper()}): {test_acc_multi:.2f}%")

    print("\n" + "="*70)
    print("=== STAGE 3 TRAINING COMPLETE ===")
    print("="*70)
    print(f"Stage 2 checkpoint: {args.stage2_checkpoint}")
    print(f"Stage 3 checkpoint: {checkpoint_path_stage3}")

    # Finish wandb run
    if args.wandb:
        wandb.finish()

    return checkpoint_path_stage3


if __name__ == '__main__':
    main()


# Example usage:

# Supervised mode (trains encoding + fusion LoRAs + classifier together):
# python train_stage3.py --stage2_checkpoint checkpoints/evan_eurosat_stage2_mae_*.pt --stage3_mode supervised

# MAE+Supervised mode (Phase A: MAE for fusion, Phase B: classifier only):
# python train_stage3.py --stage2_checkpoint checkpoints/evan_eurosat_stage2_mae_*.pt --stage3_mode mae+supervised \
#     --mae_fusion_epochs 4 --stage3_ft_epochs 2 --wandb

# With custom masking ratios:
# python train_stage3.py --stage2_checkpoint checkpoints/evan_eurosat_stage2_mae_*.pt --stage3_mode mae+supervised \
#     --rgb_modality_drop_prob 0.5 --rgb_token_mask_ratio 0.25 --newmod_token_mask_ratio 0.75
