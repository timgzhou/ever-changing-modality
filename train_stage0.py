"""Stage 0: Train EVAN on a single modality (train1 split)."""

import torch
import torch.nn as nn
import logging
import os
import argparse
from datetime import datetime
import wandb
import csv

from evan_main import evan_small, evan_base, evan_large, EVANClassifier
from eurosat_data_utils import (
    get_loaders,
    get_modality_bands_dict
)
from train_utils import (
    single_modality_training_loop,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='Stage 0: Train EVAN on a single modality (using train1 split)')
    parser.add_argument('--model', type=str, default='evan_small', choices=['evan_small', 'evan_base', 'evan_large'],
                        help='EVAN model size (default: evan_small)')
    parser.add_argument('--use_dino_weights', action="store_true")
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
    parser.add_argument('--tz_modality_specific_layer_augmenter',type=str, default='fft', choices=['lora', 'fft'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat-stage0(supervised)',
                        help='Wandb project name')
    parser.add_argument('--global_rep', type=str, default='clstoken', choices=['clstoken','mean_patch'])
    parser.add_argument('--modality', type=str, default='rgb', choices=['rgb','vre','nir','swir','aw'])
    parser.add_argument('--train_mode', type=str, default='emb+probe', choices=['probe','adaptor','fft','emb+probe'])
    parser.add_argument('--checkpoint_name', type=str, default=None)
    args = parser.parse_args()

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

    # Create datasets
    print("\n=== Creating datasets ===")
    train1_loader, train2_loader, test_loader = get_loaders(32,4)

    # Create EVAN model with pretrained DINO weights
    print("\n=== Creating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
    evan = model_fn(
        load_weights=args.use_dino_weights,
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
        n_storage_tokens=4,
        starting_modality=args.modality,
        starting_n_chans=len(bands_mod),
        device=device
    )

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
    elif args.train_mode == 'adaptor':
        # Train LoRA adaptors + patch embedder + classifier
        model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True, msla=True, mfla=True, classifier=True)
        print(f"Mode=adaptor, Freezing backbone, training embedder, lora or fft adaptors and classifier.")
    elif args.train_mode == 'probe':
        # Train only classifier
        model.set_requires_grad(args.modality, classifier=True)
        print(f"Mode=Probe, Freezing backbone, only training classifier.")
    elif args.train_mode == 'emb+probe':
        # Train embedder with  classifier
        model.set_requires_grad(args.modality, patch_embedders=True, classifier=True)
        print(f"Mode=Emb+Probe, Freezing backbone, training embedder(tokenizer) and classifier.")

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
        model, train1_loader, test_loader, device,
        modality_bands_dict, criterion, optimizer, num_epochs,
        modality=args.modality, phase_name="Stage 0",
        use_wandb=bool(args.wandb_project), wandb_prefix='stage0' # , clip_norm=2
    )

    print("\n=== Stage 0 Training complete ===")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.checkpoint_name:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'evan_eurosat_stage0_{args.modality}_{timestamp}.pt')
    
    model.save_checkpoint(checkpoint_path)
    print(f"\n=== Stage 0 checkpoint saved to: {checkpoint_path} ===")
    print(f"Stage 0 Final metrics ({args.modality.upper()}):")
    print(f"  Train accuracy: {train_acc:.2f}%")
    print(f"  Test accuracy: {test_acc:.2f}%")
    
    filename="res/train_stage0.csv"
    file_exists=os.path.isfile(filename)
    fieldnames=["model_type","modality","train_mode","tz_lora_rank","tz_modality_specific_layer_augmenter","learning_rate","trainable_params","epoch","test_accuracy","best_test_accuracy(oracle)","best_epoch","saved_checkpoint","global_rep"]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([args.model,args.modality,args.train_mode,args.tz_lora_rank,args.tz_modality_specific_layer_augmenter,args.lr,trainable_params,num_epochs,f"{test_acc:.2f}",f"{best_test_acc:.2f}",best_epoch,checkpoint_path,args.global_rep])
    
    # Finish wandb run
    if args.wandb_project:
        wandb.finish()
    return checkpoint_path


if __name__ == '__main__':
    main()
