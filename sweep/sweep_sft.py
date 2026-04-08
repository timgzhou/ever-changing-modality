"""
Hyperparameter sweep runner for stage-0 SFT (train_sft.py).

Usage:
    # Create sweep first:
    python create_sweep.py --script sft --dataset benv2 --modalities s2_rgb
    # Then launch agent:
    wandb agent <sweep-id>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import logging
import argparse
import csv
from datetime import datetime
import wandb

from evan_main import evan_small, evan_base, evan_large, evan_small_s2, EVANClassifier, EvanSegmenter
from train_utils import single_modality_training_loop
from train_sft import get_task_config_and_loaders, _n_chans

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

# Band indices into the torchgeo 13-band S2 teacher (B1,B2,B3,B4,B5,B6,B7,B8,B8a,B9,B10,B11,B12)
# for each starting modality that supports s2dino init.
# Index meanings: B1=0, B2=1, B3=2, B4=3, B5=4, B6=5, B7=6, B8=7, B8a=8, B9=9, B10=10, B11=11, B12=12
S2DINO_BAND_INDICES = {
    's2':      None,           # full 13 bands — no slicing needed
    's2_rgb':  [3, 2, 1],     # B4, B3, B2
    's2_vre':  [4, 5, 6],     # B5, B6, B7
    's2_nir':  [7, 8],        # B8, B8a
    's2_swir': [11, 12],      # B11, B12
    's2_aw':   [0, 9],        # B1, B9
    # dataset-level full-S2 variants (fewer bands than 13)
    'benv2_s2':   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12],   # drop B10
    'pastis_s2':  [1, 2, 3, 4, 5, 6, 7, 8, 11, 12],          # drop B1, B9, B10
}


def main():
    parser = argparse.ArgumentParser(description='W&B sweep runner for stage-0 SFT')
    # Fixed args — passed at sweep creation time via create_sweep.py
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'])
    parser.add_argument('--modalities', type=str, nargs='+', required=True)
    parser.add_argument('--model', type=str, default='evan_small',
                        choices=['evan_small', 'evan_base', 'evan_large'])
    parser.add_argument('--init', type=str, default='random',
                        choices=['random', 'dino', 's2dino'],
                        help='Weight initialization: random, dino (ImageNet DINOv2), s2dino (torchgeo S2-DINO)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_time_steps', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_csv', type=str, default='res/sweep_sft_results.csv')
    parser.add_argument('--wandb_project', type=str, default='evan-sweep-sft')
    # Fixed training config (baked into sweep command, not swept)
    parser.add_argument('--train_mode', type=str, default='fft')
    parser.add_argument('--tz_lora_rank', type=int, default=0)
    parser.add_argument('--tz_fusion_time', type=int, default=3)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    # Swept hyperparameters — wandb overrides these via ${args}
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project)

    lr = args.lr
    weight_decay = args.weight_decay
    primary_modality = args.modalities[0]

    print(f"\n=== Sweep Configuration ===")
    print(f"  dataset: {args.dataset}, modalities: {args.modalities}, init: {args.init}")
    print(f"  lr: {lr}, weight_decay: {weight_decay}")
    print(f"  train_mode: {args.train_mode}, tz_lora_rank: {args.tz_lora_rank}, tz_fusion_time: {args.tz_fusion_time}")
    print(f"  epochs: {args.epochs}, warmup_epochs: {args.warmup_epochs}")

    wandb.config.update({
        'dataset': args.dataset,
        'modalities': args.modalities,
        'model': args.model,
        'init': args.init,
        'train_mode': args.train_mode,
        'tz_lora_rank': args.tz_lora_rank,
        'tz_fusion_time': args.tz_fusion_time,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'batch_size': args.batch_size,
    }, allow_val_change=True)

    modalities_str = '+'.join(args.modalities)
    wandb.run.name = f"{args.dataset}_{modalities_str}_{args.init}_lr{lr:.0e}_wd{weight_decay:.0e}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    print("\n=== Creating datasets ===")
    train1_loader, val1_loader, test_loader, task_config, modality_bands_dict = get_task_config_and_loaders(
        args.dataset, args.modalities, args.batch_size, args.num_workers,
        num_time_steps=args.num_time_steps,
    )

    print(f"\n=== Creating EVAN model (init={args.init}) ===")
    all_n_chans = [_n_chans(modality_bands_dict[m]) for m in args.modalities]
    common_kwargs = dict(
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter='fft',
        n_storage_tokens=4,
        starting_modality=args.modalities,
        starting_n_chans=all_n_chans,
        img_size=task_config.img_size,
        device=device,
    )
    if args.init == 's2dino':
        from torchgeo.models import ViTSmall16_Weights
        primary = args.modalities[0]
        # Resolve band indices: sub-band modalities (s2_rgb etc.) use their own slice;
        # full-s2 modalities use the dataset-specific slice to drop missing bands.
        if primary in S2DINO_BAND_INDICES:
            band_indices = S2DINO_BAND_INDICES[primary]
        else:
            dataset_key = f'{args.dataset}_s2'
            band_indices = S2DINO_BAND_INDICES.get(dataset_key)
        print(f"  s2dino: primary={primary}, band_indices={band_indices}")
        evan = evan_small_s2(
            weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO,
            band_indices=band_indices,
            **common_kwargs,
        )
    else:
        model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
        evan = model_fn(load_weights=(args.init == 'dino'), **common_kwargs)

    is_segmentation = (task_config.task_type == 'segmentation')
    if is_segmentation:
        model = EvanSegmenter(evan, num_classes=task_config.num_classes,
                              decoder_strategy="mean", device=device)
    else:
        model = EVANClassifier(evan, num_classes=task_config.num_classes,
                               classifier_strategy="mean", global_rep='clstoken', device=device)
    model = model.to(device)

    model.freeze_all()
    if args.train_mode == 'fft':
        model.set_requires_grad('backbone', blocks=True, norm=True)
        model.set_requires_grad('all', patch_embedders=True, clsreg=True, msla=True,
                                modality_encoders=True, head=True)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    ignore_index = getattr(task_config, 'ignore_index', -100)
    if task_config.multilabel:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'sweep_sft_{wandb.run.id}_{args.dataset}_{modalities_str}_{timestamp}.pt'
    )

    print(f"\n=== Training for {args.epochs} epochs ===")
    train_metric, test_metric, best_val_metric, best_val_test_metric = single_modality_training_loop(
        model, train1_loader, test_loader, device,
        modality_bands_dict, criterion, optimizer, args.epochs,
        modality=primary_modality,
        phase_name="Stage 0",
        use_wandb=True,
        wandb_prefix='stage0',
        multilabel=task_config.multilabel,
        label_key=task_config.label_key,
        segmentation=is_segmentation,
        num_classes=task_config.num_classes if is_segmentation else None,
        ignore_index=ignore_index if is_segmentation else -100,
        val_loader=val1_loader,
        best_checkpoint_path=checkpoint_path,
        warmup_epochs=args.warmup_epochs,
    )

    metric_name = "mIoU" if is_segmentation else ("mAP" if task_config.multilabel else "Acc")
    wandb.run.summary['best_val_metric'] = best_val_metric
    wandb.run.summary['best_val_test_metric'] = best_val_test_metric
    wandb.run.summary['final_test_metric'] = test_metric

    filename = args.results_csv
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "wandb_run_id", "dataset", "modalities", "model", "init", "train_mode",
        "tz_lora_rank", "tz_fusion_time", "lr", "weight_decay",
        "warmup_epochs", "epochs", "trainable_params", "metric_name",
        "final_test_metric", "best_val_metric", "best_val_test_metric",
        "checkpoint",
    ]
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            wandb.run.id, args.dataset, modalities_str, args.model, args.init, args.train_mode,
            args.tz_lora_rank, args.tz_fusion_time, lr, weight_decay,
            args.warmup_epochs, args.epochs, trainable_params, metric_name,
            f"{test_metric:.2f}",
            f"{best_val_metric:.2f}" if best_val_metric is not None else "",
            f"{best_val_test_metric:.2f}" if best_val_test_metric is not None else "",
            checkpoint_path,
        ])
    print(f"\nResults appended to {filename}")
    wandb.finish()


if __name__ == '__main__':
    main()
