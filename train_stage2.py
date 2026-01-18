"""stage 2: Train fusion LoRAs and classifier on train2 split.

It trains:
- New modality fusion LoRAs
- New modality encoding
- Classifier(s)

While keeping frozen:
- New modality patch embedder (trained in stage 1 MAE)
- New modality modality-specific LoRAs (trained in stage 1 MAE)
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
    evaluate,
    load_split_indices,
    supervised_finetune_phase,
    train_multimodal_fusion_mae_phase,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='stage 2: Train fusion LoRAs and classifier (after stage 1 MAE)')
    parser.add_argument('--stage1_checkpoint', type=str, required=True,
                        help='Path to stage 1 MAE checkpoint (required)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--stage2_mode', type=str, default='supervised',
                        choices=['supervised', 'mae+supervised'],
                        help='stage 2 training mode (default: supervised)')
    parser.add_argument('--stage2_lr', type=float, default=1e-3,
                        help='Learning rate for supervised phase (default: 1e-3)')
    parser.add_argument('--train_fusion_epochs', type=int, default=4,
                        help='Epochs for Phase A fusion MAE (default: 4)')
    parser.add_argument('--train_fusion_lr', type=float, default=1e-4,
                        help='Learning rate for Phase A fusion MAE (default: 1e-4)')
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat-stage2(modality-fusion)',
                        help='Wandb project name')
    args = parser.parse_args()

    # Load stage 1 checkpoint
    print(f"\n=== Loading stage 1 checkpoint from: {args.stage1_checkpoint} ===")
    checkpoint = torch.load(args.stage1_checkpoint, map_location='cpu')
    config = checkpoint['config']

    print(f"stage 1 config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Get newmod from config
    newmod = config['newmod']
    bands_newmod = config['bands_newmod']

    # Initialize wandb if enabled
    wandb.init(
        project=args.wandb_project,
        config={**config, **vars(args)},
        name=f"stage2_{config['model_type']}_{newmod}_{args.stage2_mode}"
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

    train1_indices = load_split_indices('datasets/eurosat-train1.txt', train_dataset_full)
    train1_dataset = Subset(train_dataset_full, train1_indices)
    train2_indices = load_split_indices('datasets/eurosat-train2.txt', train_dataset_full)
    train2_dataset = Subset(train_dataset_full, train2_indices)

    test_dataset_full = EuroSAT(
        root='datasets',
        split='test',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    print(f"Loaded {len(train1_indices)} and {len(train2_indices)} samples from train1 and train2 splits.")
    print(f"Test samples: {len(test_dataset_full)}")

    # Create dataloaders
    train1_loader = DataLoader(train1_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train2_loader = DataLoader(train2_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Recreate EVAN model with same config
    print("\n=== Recreating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[config['model_type']]
    evan = model_fn(
        tz_fusion_time=config['tz_fusion_time'],
        tz_lora_rank=config['tz_lora_rank'],
        tz_modality_specific_layer_augmenter=config['train_mode'],
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
    
    # Load state dict from checkpoint - this loads the MAE-trained components
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"Loaded model weights from stage 1 checkpoint")
    print(f"SSL-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")

    new_mod_test_loss, new_mod_test_acc = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device,
            modality_bands_dict, modalities_to_use=(newmod,)
        )
    rgb_test_loss, rgb_test_acc = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device,
            modality_bands_dict, modalities_to_use=('rgb',)
        )
    print(f"  {newmod} test acc: {new_mod_test_acc} \n  rgb test acc: {rgb_test_acc}")
    return


if __name__ == '__main__':
    main()


# python -u train_stage2.py --stage1_checkpoint checkpoints/evan_eurosat_stage1_distillrgb_20260118_000217.pt