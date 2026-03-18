"""Stage 0: Train EVAN on a single modality (train1 split)."""

import torch
import torch.nn as nn
import logging
import os
import argparse
from datetime import datetime
import wandb
import csv

from evan_main import evan_small, evan_base, evan_large, evan_small_s2, BENV2_BAND_INDICES, PASTIS_BAND_INDICES, EVANClassifier, EvanSegmenter
from train_utils import single_modality_training_loop

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

EUROSAT_MODALITIES = ['rgb', 'vre', 'nir', 'swir', 'aw']
BENV2_MODALITIES   = ['s2', 's1']
PASTIS_MODALITIES  = ['s2', 's1', 'rgb']


def get_task_config_and_loaders(dataset, modality, batch_size, num_workers, data_normalizer=None, num_time_steps=10):
    """
    Return (train1_loader, val1_loader, test_loader, task_config, modality_bands_dict).

    modality_bands_dict maps modality name → band tuple (EuroSAT) or slice (GeoBench).
    For GeoBench datasets this is task_config.modality_slices filtered to {modality: ...}.
    """
    if dataset == 'eurosat':
        from eurosat_data_utils import get_loaders_with_val, get_modality_bands_dict
        train1_loader, val1_loader, _, _, test_loader = get_loaders_with_val(batch_size, num_workers)
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
        return train1_loader, val1_loader, test_loader, task_config, modality_bands_dict

    elif dataset == 'benv2':
        from geobench_data_utils import get_benv2_loaders
        train1_loader, val1_loader, _, _, test_loader, task_config = get_benv2_loaders(
            batch_size=batch_size, num_workers=num_workers, starting_modality=modality
        )
        modality_bands_dict = {modality: task_config.modality_slices[modality]}
        return train1_loader, val1_loader, test_loader, task_config, modality_bands_dict

    elif dataset == 'pastis':
        from geobench_data_utils import get_pastis_loaders
        train1_loader, val1_loader, _, _, test_loader, task_config = get_pastis_loaders(
            batch_size=batch_size, num_workers=num_workers, starting_modality=modality,
            data_normalizer=data_normalizer, num_time_steps=num_time_steps,
        )
        modality_bands_dict = {modality: task_config.modality_slices[modality]}
        return train1_loader, val1_loader, test_loader, task_config, modality_bands_dict

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
    parser.add_argument('--use_s2dino_weights', action='store_true',
                        help='Init from torchgeo S2-DINO ViT-Small (SSL4EO-S12). '
                             'Only valid with --model evan_small and GeoBench datasets.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tz_fusion_time', type=int, default=3,
                        help='n modality-independent layers before fusion')
    parser.add_argument('--tz_lora_rank', type=int, default=32,
                        help='rank of lora adaptors')
    parser.add_argument('--tz_modality_specific_layer_augmenter', type=str, default='fft',
                        choices=['fft'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--global_rep', type=str, default='clstoken', choices=['clstoken', 'mean_patch'])
    parser.add_argument('--train_mode', type=str, default='emb+probe',
                        choices=['probe', 'adaptor', 'fft', 'emb+probe'])
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--num_time_steps', type=int, default=10,
                        help='Number of timestamps to sample per PASTIS image before temporal aggregation.')
    parser.add_argument('--val_per_epoch', type=int, default=1,
                        help='Run validation every N epochs (and always on the last epoch).')
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
    data_normalizer = None
    normalization = 'zscore'
    if args.dataset == 'pastis' and (
        args.use_s2dino_weights or (args.use_dino_weights and args.modality == 'rgb')
    ):
        from geobench_data_utils import make_div10000_normalizer
        data_normalizer = make_div10000_normalizer()
        normalization = 'div10000'
        print("Using /10000 normalizer to match torchgeo DINO pretraining.")
    train1_loader, val1_loader, test_loader, task_config, modality_bands_dict = get_task_config_and_loaders(
        args.dataset, args.modality, args.batch_size, args.num_workers, data_normalizer=data_normalizer,
        num_time_steps=args.num_time_steps,
    )

    # Model
    print("\n=== Creating EVAN model ===")
    common_kwargs = dict(
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
        n_storage_tokens=4,
        starting_modality=args.modality,
        starting_n_chans=task_config.modality_a_channels,
        img_size=task_config.img_size,
        device=device,
    )
    if args.use_s2dino_weights:
        if args.model != 'evan_small':
            parser.error('--use_s2dino_weights is only supported with --model evan_small')
        if args.modality != 's2':
            parser.error('--use_s2dino_weights requires --modality s2 (teacher is an S2 model)')
        if args.dataset == 'eurosat':
            parser.error('--use_s2dino_weights is not compatible with --dataset eurosat')
        from torchgeo.models import ViTSmall16_Weights
        band_indices = {'benv2': BENV2_BAND_INDICES, 'pastis': PASTIS_BAND_INDICES}.get(args.dataset)
        evan = evan_small_s2(
            weights=ViTSmall16_Weights.SENTINEL2_ALL_DINO,
            band_indices=band_indices,
            **common_kwargs,
        )
    else:
        model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
        evan = model_fn(load_weights=args.use_dino_weights, **common_kwargs)

    is_segmentation = (task_config.task_type == 'segmentation')

    if is_segmentation:
        model = EvanSegmenter(
            evan,
            num_classes=task_config.num_classes,
            decoder_strategy="mean",
            device=device,
        )
    else:
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
        model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True, msla=True, mfla=True, modality_encoders=True, head=True)
        print("Mode=fft: training full backbone layers + head.")
    elif args.train_mode == 'adaptor':
        model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True, msla=True, mfla=True, modality_encoders=True, head=True)
        print("Mode=adaptor: training embedder, LoRA/FFT adaptors + head.")
    elif args.train_mode == 'probe':
        model.set_requires_grad(args.modality, head=True)
        print("Mode=probe: training head only.")
    elif args.train_mode == 'emb+probe':
        model.set_requires_grad(args.modality, patch_embedders=True, head=True)
        print("Mode=emb+probe: training embedder + head.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Loss
    ignore_index = getattr(task_config, 'ignore_index', -100)
    if task_config.multilabel:
        criterion = nn.BCEWithLogitsLoss()
        print("Loss: BCEWithLogitsLoss (multilabel)")
    else:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)
        print(f"Loss: CrossEntropyLoss (ignore_index={ignore_index})")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"stage0_{args.dataset}_{args.modality}_{args.train_mode}",
        )

    print(f"\n=== Training for {args.epochs} epochs ===")
    if is_segmentation:
        metric_name = "mIoU"
    elif task_config.multilabel:
        metric_name = "mAP"
    else:
        metric_name = "Acc"

    # Determine checkpoint path before training so best-val checkpoint can be saved mid-run
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.checkpoint_name:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    else:
        checkpoint_path = os.path.join(args.checkpoint_dir,
                                       f'evan_{args.dataset}_stage0_{args.modality}_{timestamp}.pt')

    train_metric, test_metric, best_test_metric, best_epoch, best_val_metric = single_modality_training_loop(
        model, train1_loader, test_loader, device,
        modality_bands_dict, criterion, optimizer, args.epochs,
        modality=args.modality,
        phase_name="Stage 0",
        use_wandb=bool(args.wandb_project),
        wandb_prefix='stage0',
        multilabel=task_config.multilabel,
        label_key=task_config.label_key,
        segmentation=is_segmentation,
        num_classes=task_config.num_classes if is_segmentation else None,
        ignore_index=ignore_index if is_segmentation else -100,
        val_loader=val1_loader,
        best_checkpoint_path=checkpoint_path,
        val_per_epoch=args.val_per_epoch,
    )

    # Patch normalization into checkpoint config so shot_ete.py can read it back
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    ckpt['config']['normalization'] = normalization
    torch.save(ckpt, checkpoint_path)

    print(f"\n=== Stage 0 Training complete ===")
    print(f"  Train {metric_name}: {train_metric:.2f}%")
    print(f"  Test  {metric_name}: {test_metric:.2f}%")
    if best_val_metric is not None:
        print(f"  Best val {metric_name}: {best_val_metric:.2f}% — checkpoint: {checkpoint_path}")

    # CSV logging
    filename = f"res/train_stage0_{args.dataset}.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "dataset", "model_type", "modality", "train_mode",
        "tz_lora_rank", "tz_modality_specific_layer_augmenter",
        "learning_rate", "trainable_params", "epoch",
        "test_metric", "best_test_metric(oracle)", "best_epoch",
        "best_val_metric", "metric_name", "saved_checkpoint", "global_rep",
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
            f"{best_val_metric:.2f}" if best_val_metric is not None else "",
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

# BEN-v2 S2 (random init)
python -u train_stage0.py --dataset benv2 --modality s2 --epochs 2 --checkpoint_name benv2_s2_s0 --train_mode fft

# BEN-v2 S2 (init from torchgeo S2-DINO)
python -u train_stage0.py --dataset benv2 --modality s2 --epochs 2 --checkpoint_name benv2_s2dino_s0 --train_mode fft --use_s2dino_weights

# BEN-v2 S1 (start from S1 instead)
python -u train_stage0.py --dataset benv2 --modality s1 --epochs 2 --checkpoint_name benv2_s1_s0 --train_mode fft

# PASTIS S2
python -u train_stage0.py --dataset pastis --modality s2 --epochs 2 --batch_size 16 --checkpoint_name pastis_s2_s0 --train_mode fft --num_workers 4 --use_dino_weights 
python -u train_stage0.py --dataset pastis --modality s2 --epochs 16 --batch_size 16 --checkpoint_name pastis_s2_s0 --train_mode fft --num_workers 4  --use_s2dino_weights

# PASTIS RGB (init from DINOv3, 3-channel B04/B03/B02 subset of S2)
python -u train_stage0.py --dataset pastis --modality rgb --epochs 16 --batch_size 16 --checkpoint_name pastis_rgb_s0 --train_mode fft --num_workers 4 --use_dino_weights
"""
