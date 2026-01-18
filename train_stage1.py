"""Stage 1: Train new modality embedder and modality-specific LoRAs on train2 split.

Two modes:
- supervised: Train all new modality components (embedder, modality-specific LoRAs, fusion LoRAs, classifier)
- mae: Train only modality embedder and modality-specific LoRAs via MAE reconstruction (requires stage 2 for fusion)
"""

import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    single_modality_training_loop,
    supervised_finetune_phase,
    train_distillrgb_phase,
    train_mae_phase,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Stage 1: Train new modality embedder and modality-specific LoRAs')
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to Stage 0 checkpoint (required)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--stage1_ft_epochs', type=int, default=4,
                        help='Number of fine-tuning epochs for supervised mode (default: 4)')
    parser.add_argument('--stage1_lr', type=float, default=1e-3,
                        help='Learning rate for supervised mode (default: 1e-3)')
    parser.add_argument('--stage1_train_method', type=str, default='mae',
                        choices=['supervised', 'mae', 'distillrgb'],
                        help='Stage 1 training method: supervised (all components) or mae (embedder + modality-specific only)')
    parser.add_argument('--stage1_ssl_epochs', type=int, default=4,
                        help='Number of MAE epochs for mae mode (default: 4)')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.85,
                        help='Mask ratio for MAE training (default: 0.85)')
    parser.add_argument('--ssl_lr', type=float, default=0.00003,
                        help='Learning rate for MAE phase (default: 0.00001)')
    parser.add_argument('--num_supervised_epochs', type=int, default=8)
    parser.add_argument('--new_mod_group', type=str, default='vre', choices=['vre', 'nir', 'swir'],
                        help='New modality group to train')
    parser.add_argument('--freeze_rgb', action='store_true',
                        help='Freeze RGB classifier head during supervised training')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat-stage1(modality-specific-layers)',
                        help='Wandb project name')
    args = parser.parse_args()

    # Load Stage 0 checkpoint
    print(f"\n=== Loading Stage 0 checkpoint from: {args.stage0_checkpoint} ===")
    checkpoint = torch.load(args.stage0_checkpoint, map_location='cpu')
    config = checkpoint['config']

    print(f"Stage 0 config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Initialize wandb if enabled
    wandb.init(
        project=args.wandb_project,
        config={**config, **vars(args)},
        name=f"stage1_{config['model_type']}_{args.new_mod_group}_{args.stage1_train_method}"
    )

    # Band configuration
    newmod = args.new_mod_group
    modality_bands_dict = get_modality_bands_dict('rgb', newmod)
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
        n_storage_tokens=4,
        device=device
    )

    # Create classifier
    num_newmod_channels = len(bands_newmod)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.create_modality_components(newmod,num_newmod_channels)
        
    model = EVANClassifier(evan, num_classes=config['num_classes'], classifier_strategy='mean', device=device)
    model = model.to(device)

    # Load state dict from checkpoint
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded model weights from Stage 0 checkpoint")
    print(f"Stage 0 final accuracy - RGB: {checkpoint['test_acc']:.2f}%")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    timestamp_stage1 = datetime.now().strftime('%Y%m%d_%H%M%S')
    # ==================== Stage 1 TRAINING ====================
    print("\n" + "="*70)
    print("=== Stage 1: Training new modality components on train2 split ===")
    print("="*70)
    
    if args.stage1_train_method == 'mae':
        # ========== MAE TRAINING (trains embedder + modality-specific only) ==========
        print(f"\nUsing MAE training method (trains embedder + modality-specific LoRAs only)")

        # MAE SSL training for modality embedder and modality-specific layers
        train_mae_phase(model, train2_loader, test_loader_full, device, args, bands_newmod, newmod)

        # Save Stage 1 MAE checkpoint
        checkpoint_path_stage1 = os.path.join(args.checkpoint_dir, f'evan_eurosat_stage1_mae_{timestamp_stage1}.pt')


        print(f"\n=== Stage 1 (MAE) checkpoint saved to: {checkpoint_path_stage1} ===")
        print(f"Trained components: {newmod} patch embedder, {newmod} cls and reg tokens, {newmod} modality-specific LoRAs")

    if args.stage1_train_method == 'distillrgb':
        # ========== Distill RGB TRAINING (trains embedder + modality-specific only) ==========
        print(f"\nUsing Distill RGB training method (trains embedder + modality-specific LoRAs for {newmod=})")
        train_distillrgb_phase(model, train2_loader, test_loader_full, device, args, bands_newmod, newmod, modality_bands_dict)
        checkpoint_path_stage1 = os.path.join(args.checkpoint_dir, f'evan_eurosat_stage1_distillrgb_{timestamp_stage1}.pt')

    checkpoint_stage1 = {
            'model_state_dict': model.state_dict(),
            'stage0_checkpoint': args.stage0_checkpoint,
            'config': {
                **config,
                'train_split': 'train2',
                'stage1_train_method': args.stage1_train_method,
                'ssl_lr': args.ssl_lr,
                'bands_newmod': bands_newmod,
                'newmod': newmod,
                'stage1_ssl_epochs': args.stage1_ssl_epochs,
            }
        }
    torch.save(checkpoint_stage1, checkpoint_path_stage1)
    print("\n" + "="*70)
    print("=== Stage 1 TRAINING COMPLETE ===")
    print("="*70)
    print(f"Stage 0 checkpoint: {args.stage0_checkpoint}")
    print(f"Stage 1 checkpoint: {checkpoint_path_stage1}")

    # ========== Evaluation By Supervised Probing ==========
    num_supervised_epochs = args.num_supervised_epochs
    print("\n" + "="*70)
    print("=== Evaluating MAE learned modality-specific features ===")
    model.freeze_all()
    model.set_requires_grad(newmod, patch_embedders=False, clsreg=False, msla=False, mfla=True, modality_encoders=True, classifier=True)
    print(f"Freezing modality-specific tokenizer, cls-reg tokens, modality-specific layer adaptors, training modality-fusion layer adaptors, modality_encoders, classifier.")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    print(f"\n=== Training for {num_supervised_epochs} epochs ===")
    print(f"Strategy: Train and evaluate on {newmod} only (train2 split)")
    train_acc, test_acc, best_test_acc, best_epoch = single_modality_training_loop(
        model, train2_loader, test_loader_full, device,
        modality_bands_dict, criterion, optimizer, num_supervised_epochs,
        modality=newmod, phase_name="Stage 1 supervised evaluation")
    print(f"Eval Result: \n      {train_acc=:.2f} {test_acc=:.2f} \n      {best_test_acc=:.2f} at epoch {best_epoch}")

    filename="train_stage1_res.csv"
    file_exists=os.path.isfile("train_stage0_res.csv")
    fieldnames=["model_type","new_modality","ssl_mode","ssl_lr","ssl_epoch","supervised_epoch","test_accuracy(afterSFT)","best_test_accuracy(oracle)","best_supervised_epoch","stage0_checkpoint","stage1_checkpoint"]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow(
            [
                config['model_type'],
                args.new_mod_group,
                args.stage1_train_method,
                args.ssl_lr,
                args.stage1_ssl_epochs,
                args.num_supervised_epochs,
                f"{test_acc:.2f}",
                f"{best_test_acc:.2f}",
                best_epoch,
                args.stage0_checkpoint,
                checkpoint_path_stage1
             ])
    
    wandb.finish()

    return


if __name__ == '__main__':
    main()
