"""Stage 0: Train EVAN on a single modality (train1 split)."""

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
import csv

from evan_main import evan_small, evan_base, evan_large, EVANClassifier
from eurosat_data_utils import (
    DictTransform,
    ALL_BAND_NAMES,
    get_modality_bands_dict
)
from train_utils import (
    load_split_indices,
    single_modality_training_loop,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Stage 0: Train EVAN on a single modality (using train1 split)')
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
    parser.add_argument('--tz_lora_rank', type=int, default=32,
                        help='rank of lora adaptors')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat-stage0(supervised)',
                        help='Wandb project name')
    parser.add_argument('--global_rep', type=str, default='clstoken', choices=['clstoken','mean_patch'])
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb','vre','nir','swir','aw'])
    parser.add_argument('--train_mode', type=str, default='probe', choices=['probe','lora','fft'])
    args = parser.parse_args()
    tz_modality_specific_layer_augmenter='fft' if args.train_mode=="fft" else 'lora'
    # Initialize wandb if enabled
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"stage0_{args.model}_{args.modality}_{args.train_mode}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # Band configuration (using constants from eurosat_data_utils)
    modality_bands_dict = get_modality_bands_dict(args.modality)
    bands_mod = modality_bands_dict[args.modality]
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

    print(f"Train samples (split=train1): {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset_full)}")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader_full = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # Create EVAN model with pretrained DINO weights
    print("\n=== Creating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
    evan = model_fn(
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter=tz_modality_specific_layer_augmenter,
        n_storage_tokens=4,
        device=device
    )

    # Pre-create modality components if not RGB (must happen before optimizer creation)
    # Otherwise, components created during forward pass won't be in optimizer
    if args.modality != 'rgb':
        in_chans = len(bands_mod)
        evan.create_modality_components(args.modality, in_chans)

    # Create classifier
    model = EVANClassifier(evan, num_classes=10, classifier_strategy="mean", global_rep=args.global_rep, device=device)
    model = model.to(device)

    # Freeze EVAN backbone, train only classifier(s)
    model.freeze_all()
    if args.train_mode == 'fft':
        # Train backbone blocks + classifier, freeze LoRA paths
        model.set_requires_grad('backbone', blocks=True, norm=True)
        model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True, classifier=True)
        print(f"Mode=fft, Freezing lora paths, training full layers and classifier.")
    elif args.train_mode == 'lora':
        # Train LoRA adaptors + patch embedder + classifier
        model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True, msla=True, mfla=True, classifier=True)
        print(f"Mode=lora, Freezing backbone, only training lora adaptors and classifier.")
    elif args.train_mode == 'probe':
        # Train only classifier
        model.set_requires_grad(args.modality, classifier=True)
        print(f"Mode=Probe, Freezing backbone, only training classifier.")

    # Print parameter info
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    num_epochs = args.epochs

    print(f"\n=== Training for {num_epochs} epochs ===")
    print(f"Strategy: Train and evaluate on {args.modality.upper()} only (train1 split)")

    # Run single-modality training loop
    train_acc, test_acc, best_test_acc, best_epoch = single_modality_training_loop(
        model, train_loader, test_loader_full, device,
        modality_bands_dict, criterion, optimizer, num_epochs,
        modality=args.modality, phase_name="Stage 0",
        use_wandb=bool(args.wandb_project), wandb_prefix='stage0' # , clip_norm=2
    )

    print("\n=== Stage 0 Training complete ===")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_path = os.path.join(args.checkpoint_dir, f'evan_eurosat_stage0_{args.modality}_{timestamp}.pt')

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'modality': args.modality,
        'config': {
            'model_type': args.model,
            'num_classes': 10,
            'train_split': 'train1',
            'tz_fusion_time': args.tz_fusion_time,
            'tz_lora_rank': args.tz_lora_rank,
            'train_mode': args.train_mode,
            'modality': args.modality,
            'bands': bands_mod,
            'bands_full': bands_full,
            'num_epochs': num_epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
        }
    }

    torch.save(checkpoint, checkpoint_path)
    print(f"\n=== Stage 0 checkpoint saved to: {checkpoint_path} ===")
    print(f"Stage 0 Final metrics ({args.modality.upper()}):")
    print(f"  Train accuracy: {train_acc:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")
    
    filename="train_stage0_res.csv"
    file_exists=os.path.isfile("train_stage0_res.csv")
    fieldnames=["model_type","modality","train_mode","tz_lora_rank","learning_rate","test_accuracy","epoch","best_test_accuracy(oracle)","best_epoch","saved_checkpoint"]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([args.model,args.modality,args.train_mode,args.tz_lora_rank,args.lr,f"{test_acc:.2f}",num_epochs,f"{best_test_acc:.2f}",best_epoch,checkpoint_path])
    
    # Finish wandb run
    if args.wandb_project:
        wandb.finish()

    return checkpoint_path


if __name__ == '__main__':
    main()

# Example usage:
# python train_stage0.py --model evan_base --epochs 5 --modality rgb # to train rgb-only evan (equiv to dino).
# python train_stage0.py --model evan_base --epochs 5 --modality vre
