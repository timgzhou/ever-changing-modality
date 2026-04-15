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

EUROSAT_MODALITIES    = ['rgb', 'vre', 'nir', 'swir', 'aw']
BENV2_MODALITIES      = ['s2', 's1', 's2_rgb', 's2_vre', 's2_nir', 's2_swir', 's2_aw']
BENV2FULL_MODALITIES  = ['s2', 's1', 's2_rgb', 's2_vre', 's2_nir', 's2_swir', 's2_aw']
PASTIS_MODALITIES     = ['s2', 's1', 'rgb', 's2_rgb', 's2_vre', 's2_nir', 's2_swir']
DFC2020_MODALITIES    = ['s2', 's1', 's2_rgb', 's2_vre', 's2_nir', 's2_swir', 's2_aw']


def get_task_config_and_loaders(dataset, modalities, batch_size, num_workers, data_normalizer=None, num_time_steps=10, data_root=None):
    """Return (train1_loader, val1_loader, test_loader, task_config, modality_bands_dict).

    modalities: list of modality names; first is primary (used for loaders).
    """
    from data_utils import get_loaders
    primary = modalities[0]
    train1, val1, _, _, test, task_config = get_loaders(
        dataset, primary, batch_size, num_workers,
        data_normalizer=data_normalizer, num_time_steps=num_time_steps,
        new_modality=None, data_root=data_root,
    )
    modality_bands_dict = {m: task_config.modality_bands_dict[m] for m in modalities}
    return train1, val1, test, task_config, modality_bands_dict


def _n_chans(entry) -> int:
    """Return channel count for a modality_bands_dict entry (slice or list)."""
    if isinstance(entry, slice):
        return entry.stop - entry.start
    return len(entry)  # list of indices


def main():
    parser = argparse.ArgumentParser(description='Train EVAN on a single modality (using train1 split)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eurosat', 'benv2', 'benv2full', 'pastis', 'dfc2020'],
                        help='Dataset to train on')
    parser.add_argument('--modalities', type=str, nargs='+', required=True,
                        help='Modalities to train on (first is primary). '
                             'EuroSAT: rgb/vre/nir/swir/aw. '
                             'GeoBench/DFC2020: s2/s1/s2_rgb/s2_vre/s2_nir/s2_swir/s2_aw.')
    parser.add_argument('--model', type=str, default='evan_base', choices=['evan_small', 'evan_base', 'evan_large'])
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
    parser.add_argument('--train_mode', type=str, default='fft',
                        choices=['probe', 'adaptor', 'fft', 'emb+probe'])
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--num_time_steps', type=int, default=10,
                        help='Number of timestamps to sample per PASTIS image before temporal aggregation.')
    parser.add_argument('--val_per_epoch', type=int, default=1,
                        help='Run validation every N epochs (and always on the last epoch).')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Linear LR warmup epochs before cosine decay (default: 1).')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Root directory for benv2full dataset. Should contain BigEarthNet-S2 and BigEarthNet-S1 subdirs. '
                             'Default: looks in current dir.')
    args = parser.parse_args()

    # Validate modalities against dataset
    valid_modalities = {
        'eurosat':   EUROSAT_MODALITIES,
        'benv2':     BENV2_MODALITIES,
        'benv2full': BENV2FULL_MODALITIES,
        'pastis':    PASTIS_MODALITIES,
        'dfc2020':   DFC2020_MODALITIES,
    }[args.dataset]
    for m in args.modalities:
        if m not in valid_modalities:
            parser.error(f"--modalities {m!r} is not valid for --dataset {args.dataset}. "
                         f"Valid choices: {valid_modalities}")
    primary_modality = args.modalities[0]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {args.dataset}, Modalities: {args.modalities}")
    print(f"Using device: {device}")
    print(f"Model: {args.model}, Batch size: {args.batch_size}, LR: {args.lr}, Epochs: {args.epochs}")

    # Data
    print("\n=== Creating datasets ===")
    data_normalizer = None
    normalization = 'zscore'
    train1_loader, val1_loader, test_loader, task_config, modality_bands_dict = get_task_config_and_loaders(
        args.dataset, args.modalities, args.batch_size, args.num_workers, data_normalizer=data_normalizer,
        num_time_steps=args.num_time_steps, data_root=args.data_root,
    )

    # Model
    print("\n=== Creating EVAN model ===")
    all_n_chans = [_n_chans(modality_bands_dict[m]) for m in args.modalities]
    common_kwargs = dict(
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
        n_storage_tokens=4,
        starting_modality=args.modalities,
        starting_n_chans=all_n_chans,
        img_size=task_config.img_size,
        device=device,
    )
    if args.use_s2dino_weights:
        if args.model != 'evan_small':
            parser.error('--use_s2dino_weights is only supported with --model evan_small')
        if primary_modality != 's2':
            parser.error('--use_s2dino_weights requires primary modality s2 (teacher is an S2 model)')
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
        # For s2 modality, pass the RGB-band positions so DINO patch weights are
        # copied into the correct channels of the s2 patch embedder.
        # EuroSAT s2 (13ch): B04=3, B03=2, B02=1
        # BEN-v2  s2 (12ch): same positions (B10 dropped at idx10, doesn't shift rgb)
        # PASTIS  s2 (10ch): B04=2, B03=1, B02=0 (B01/B09/B10 removed)
        _S2_RGB_INDICES = {
            'eurosat':   [3, 2, 1],
            'benv2':     [3, 2, 1],
            'benv2full': [3, 2, 1],
            'dfc2020':   [3, 2, 1],
            'pastis':    [2, 1, 0],
        }
        rgb_in_s2_indices = (
            _S2_RGB_INDICES.get(args.dataset)
            if primary_modality in ('s2', 's2_rgb') and args.use_dino_weights
            else None
        )
        model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
        evan = model_fn(load_weights=args.use_dino_weights, rgb_in_s2_indices=rgb_in_s2_indices, **common_kwargs)

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

    # Freeze / unfreeze according to train_mode (apply to all modalities)
    model.freeze_all()
    if args.train_mode == 'fft':
        model.set_requires_grad('backbone', blocks=True, norm=True)
        model.set_requires_grad('all', patch_embedders=True, clsreg=True, msla=True, modality_encoders=True, head=True)
        print("Mode=fft: training full backbone layers + head.")
    elif args.train_mode == 'probe':
        model.set_requires_grad('all', head=True)
        print("Mode=probe: training head only.")
    elif args.train_mode == 'emb+probe':
        model.set_requires_grad('all', patch_embedders=True, head=True)
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
        modalities_str = '+'.join(args.modalities)
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"stage0_{args.dataset}_{modalities_str}_{args.train_mode}",
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
        modalities_str = '+'.join(args.modalities)
        checkpoint_path = os.path.join(args.checkpoint_dir,
                                       f'sft_{args.model}_{args.dataset}_{modalities_str}_{args.train_mode}_lr{args.lr}_{timestamp}.pt')

    train_metric, test_metric, best_val_metric, best_val_test_metric = single_modality_training_loop(
        model, train1_loader, test_loader, device,
        modality_bands_dict, criterion, optimizer, args.epochs,
        modality=primary_modality,
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
        warmup_epochs=args.warmup_epochs,
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
    filename = f"res/train_sft/{args.dataset}.csv"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "dataset", "model_type", "modality", "train_mode",
        "tz_lora_rank", "tz_modality_specific_layer_augmenter",
        "learning_rate", "trainable_params", "epoch",
        "test_metric", "metric_name", "saved_checkpoint", "global_rep",
        "dino_init",
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            args.dataset, args.model, '+'.join(args.modalities), args.train_mode,
            args.tz_lora_rank, args.tz_modality_specific_layer_augmenter,
            args.lr, trainable_params, args.epochs,
            f"{best_val_test_metric:.2f}" if best_val_test_metric is not None else "",
            metric_name, checkpoint_path, args.global_rep,
            args.use_dino_weights,
        ])

    if args.wandb_project:
        wandb.finish()
    return checkpoint_path


if __name__ == '__main__':
    main()

# for run example see sh/{dataset}