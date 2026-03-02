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
from train_utils import single_modality_training_loop

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

EUROSAT_MODALITIES = ['rgb', 'vre', 'nir', 'swir', 'aw']
BENV2_MODALITIES   = ['s2', 's1']
PASTIS_MODALITIES  = ['s2', 's1']


def get_task_config_and_loaders(dataset, modality, batch_size, num_workers):
    """
    Return (train1_loader, test_loader, task_config, modality_bands_dict).

    modality_bands_dict maps modality name → band tuple (EuroSAT) or slice (GeoBench).
    For GeoBench datasets this is task_config.modality_slices filtered to {modality: ...}.
    """
    if dataset == 'eurosat':
        from eurosat_data_utils import get_loaders, get_modality_bands_dict
        train1_loader, _, test_loader = get_loaders(batch_size, num_workers)
        modality_bands_dict = get_modality_bands_dict(modality)

        from types import SimpleNamespace
        task_config = SimpleNamespace(
            dataset_name='eurosat',
            task_type='classification',
            modality_a_channels=len(modality_bands_dict[modality]),
            num_classes=10,
            multilabel=False,
            label_key='label',
            modality_slices=None,
            img_size=224,
        )
        return train1_loader, test_loader, task_config, modality_bands_dict

    elif dataset == 'benv2':
        from geobench_data_utils import get_benv2_loaders
        train1_loader, _, _, _, test_loader, task_config = get_benv2_loaders(
            batch_size=batch_size, num_workers=num_workers, starting_modality=modality
        )
        modality_bands_dict = {modality: task_config.modality_slices[modality]}
        return train1_loader, test_loader, task_config, modality_bands_dict

    elif dataset == 'pastis':
        from geobench_data_utils import get_pastis_loaders
        train1_loader, _, _, _, test_loader, task_config = get_pastis_loaders(
            batch_size=batch_size, num_workers=num_workers, starting_modality=modality
        )
        modality_bands_dict = {modality: task_config.modality_slices[modality]}
        return train1_loader, test_loader, task_config, modality_bands_dict

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(description='Stage 0: Train EVAN on a single modality (using train1 split)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eurosat', 'benv2', 'pastis'],
                        help='Dataset to train on')
    parser.add_argument('--modality', type=str, required=True,
                        help='Modality to train on. '
                             'EuroSAT: rgb/vre/nir/swir/aw. GeoBench: s2/s1.')
    parser.add_argument('--model', type=str, default='evan_small', choices=['evan_small', 'evan_base', 'evan_large'])
    parser.add_argument('--use_dino_weights', action='store_true')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tz_fusion_time', type=int, default=3,
                        help='n modality-independent layers before fusion')
    parser.add_argument('--tz_lora_rank', type=int, default=32,
                        help='rank of lora adaptors')
    parser.add_argument('--tz_modality_specific_layer_augmenter', type=str, default='fft',
                        choices=['lora', 'fft'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--global_rep', type=str, default='clstoken', choices=['clstoken', 'mean_patch'])
    parser.add_argument('--train_mode', type=str, default='emb+probe',
                        choices=['probe', 'adaptor', 'fft', 'emb+probe'])
    parser.add_argument('--checkpoint_name', type=str, default=None)
    args = parser.parse_args()

    # Validate modality against dataset
    valid_modalities = {
        'eurosat': EUROSAT_MODALITIES,
        'benv2':   BENV2_MODALITIES,
        'pastis':  PASTIS_MODALITIES,
    }[args.dataset]
    if args.modality not in valid_modalities:
        parser.error(f"--modality {args.modality!r} is not valid for --dataset {args.dataset}. "
                     f"Valid choices: {valid_modalities}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {args.dataset}, Modality: {args.modality}")
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # Data
    print("\n=== Creating datasets ===")
    train1_loader, test_loader, task_config, modality_bands_dict = get_task_config_and_loaders(
        args.dataset, args.modality, args.batch_size, args.num_workers
    )

    # Model
    print("\n=== Creating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
    evan = model_fn(
        load_weights=args.use_dino_weights,
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
        n_storage_tokens=4,
        starting_modality=args.modality,
        starting_n_chans=task_config.modality_a_channels,
        img_size=task_config.img_size,
        device=device,
    )

    model = EVANClassifier(
        evan,
        num_classes=task_config.num_classes,
        classifier_strategy="mean",
        global_rep=args.global_rep,
        device=device,
    )
    model = model.to(device)

    # Freeze / unfreeze according to train_mode
    model.freeze_all()
    if args.train_mode == 'fft':
        model.set_requires_grad('backbone', blocks=True, norm=True)
        model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True, classifier=True)
        print("Mode=fft: training full backbone layers + classifier.")
    elif args.train_mode == 'adaptor':
        model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True, msla=True, mfla=True, classifier=True)
        print("Mode=adaptor: training embedder, LoRA/FFT adaptors + classifier.")
    elif args.train_mode == 'probe':
        model.set_requires_grad(args.modality, classifier=True)
        print("Mode=probe: training classifier only.")
    elif args.train_mode == 'emb+probe':
        model.set_requires_grad(args.modality, patch_embedders=True, classifier=True)
        print("Mode=emb+probe: training embedder + classifier.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Loss
    if task_config.multilabel:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss (multilabel)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Loss: CrossEntropyLoss (classification)")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"stage0_{args.dataset}_{args.modality}_{args.train_mode}",
        )

    print(f"\n=== Training for {args.epochs} epochs ===")
    metric_name = "mAP" if task_config.multilabel else "Acc"

    train_metric, test_metric, best_test_metric, best_epoch = single_modality_training_loop(
        model, train1_loader, test_loader, device,
        modality_bands_dict, criterion, optimizer, args.epochs,
        modality=args.modality,
        phase_name="Stage 0",
        use_wandb=bool(args.wandb_project),
        wandb_prefix='stage0',
        multilabel=task_config.multilabel,
        label_key=task_config.label_key,
    )

    print(f"\n=== Stage 0 Training complete ===")
    print(f"  Train {metric_name}: {train_metric:.2f}%")
    print(f"  Test  {metric_name}: {test_metric:.2f}%")

    # Save checkpoint
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.checkpoint_name:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir,
                                       f'evan_{args.dataset}_stage0_{args.modality}_{timestamp}.pt')

    model.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    # CSV logging
    filename = "res/train_stage0.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "dataset", "model_type", "modality", "train_mode",
        "tz_lora_rank", "tz_modality_specific_layer_augmenter",
        "learning_rate", "trainable_params", "epoch",
        "test_metric", "best_test_metric(oracle)", "best_epoch",
        "metric_name", "saved_checkpoint", "global_rep",
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            args.dataset, args.model, args.modality, args.train_mode,
            args.tz_lora_rank, args.tz_modality_specific_layer_augmenter,
            args.lr, trainable_params, args.epochs,
            f"{test_metric:.2f}", f"{best_test_metric:.2f}", best_epoch,
            metric_name, checkpoint_path, args.global_rep,
        ])

    if args.wandb_project:
        wandb.finish()
    return checkpoint_path


if __name__ == '__main__':
    main()


# DRYRUN examples
"""
# EuroSAT RGB (original behaviour)
python -u train_stage0.py --dataset eurosat --modality rgb --epochs 5 --checkpoint_name eurosat_rgb_s0

# BEN-v2 S2
python -u train_stage0.py --dataset benv2 --modality s2 --epochs 2 --checkpoint_name benv2_s2_s0 --train_mode fft

# BEN-v2 S1 (start from S1 instead)
python -u train_stage0.py --dataset benv2 --modality s1 --epochs 2 --checkpoint_name benv2_s1_s0 --train_mode fft

# PASTIS S2
python -u train_stage0.py --dataset pastis --modality s2 --epochs 2 --batch_size 16 --checkpoint_name pastis_s2_s0
"""
