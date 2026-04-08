"""
Hyperparameter sweep runner for baseline_distillation.py.

Usage:
    # Create sweep first:
    python create_sweep.py --script baseline_distill --dataset benv2 --teacher_checkpoint checkpoints/benv2_s2_s0.pt --modality s2_rgb
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

from evan_main import evan_small, evan_base, evan_large, EVANClassifier, EvanSegmenter
from data_utils import get_loaders
from train_utils import evaluate
from baseline.baseline_distillation import (
    init_student_from_teacher, distillation_training_loop, evaluate_ensemble,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='W&B sweep runner for baseline distillation')
    # Fixed args — passed at sweep creation time via create_sweep.py
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'])
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--modality', type=str, required=True,
                        help='Student modality (must differ from teacher)')
    parser.add_argument('--model', type=str, default='evan_small',
                        choices=['evan_small', 'evan_base', 'evan_large'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=16)
    parser.add_argument('--warmup_epochs', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_time_steps', type=int, default=10)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--results_csv', type=str, default='res/sweep_baseline_distill_results.csv')
    parser.add_argument('--wandb_project', type=str, default='evan-sweep-baseline-distill')
    parser.add_argument('--distillation_mode', type=str, default='regular',
                        choices=['regular', 'with_guidance', 'feature'])
    parser.add_argument('--alpha', type=float, default=1.0)
    # Swept hyperparameters — wandb overrides these via ${args}
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=2.0)
    args = parser.parse_args()

    wandb.init(project=args.wandb_project)

    lr = args.lr
    weight_decay = args.weight_decay
    temperature = args.temperature

    print(f"\n=== Sweep Configuration ===")
    print(f"  dataset: {args.dataset}, modality: {args.modality}")
    print(f"  lr: {lr}, weight_decay: {weight_decay}, temperature: {temperature}")
    print(f"  epochs: {args.epochs}, warmup_epochs: {args.warmup_epochs}")

    wandb.config.update({
        'dataset': args.dataset,
        'modality': args.modality,
        'model': args.model,
        'distillation_mode': args.distillation_mode,
        'alpha': args.alpha,
        'epochs': args.epochs,
        'warmup_epochs': args.warmup_epochs,
        'batch_size': args.batch_size,
        'teacher_checkpoint': args.teacher_checkpoint,
    }, allow_val_change=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Read normalizer from teacher checkpoint
    _ckpt_meta = torch.load(args.teacher_checkpoint, map_location='cpu')
    normalization = _ckpt_meta.get('config', {}).get('normalization', 'zscore')
    data_normalizer = None
    if normalization == 'div10000':
        from geobench_data_utils import make_div10000_normalizer
        data_normalizer = make_div10000_normalizer()
    del _ckpt_meta

    print("\n=== Creating datasets ===")
    train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config = get_loaders(
        args.dataset, args.modality, args.batch_size, args.num_workers,
        data_normalizer=data_normalizer, num_time_steps=args.num_time_steps,
    )

    is_segmentation = (task_config.task_type == 'segmentation')
    multilabel = task_config.multilabel
    label_key = task_config.label_key
    ignore_index = getattr(task_config, 'ignore_index', -100)
    metric_name = "mIoU" if is_segmentation else ("mAP" if multilabel else "Acc")

    print(f"\n=== Loading teacher from {args.teacher_checkpoint} ===")
    if is_segmentation:
        teacher_model = EvanSegmenter.from_checkpoint(args.teacher_checkpoint, device=device)
    else:
        teacher_model = EVANClassifier.from_checkpoint(args.teacher_checkpoint, device=device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    teacher_modality = teacher_model.evan.starting_modality
    print(f"Teacher: {teacher_modality}, Student: {args.modality}")

    wandb.run.name = f"{args.dataset}_{teacher_modality}->{args.modality}_lr{lr:.0e}_t{temperature:.1f}"

    modality_bands_dict = task_config.modality_bands_dict
    bands_mod = modality_bands_dict[args.modality]
    num_student_chans = len(range(*bands_mod.indices(999))) if isinstance(bands_mod, slice) else len(bands_mod)

    # Evaluate teacher baseline
    _ce = nn.CrossEntropyLoss(ignore_index=ignore_index) if not multilabel else nn.BCEWithLogitsLoss()
    _, teacher_test_metric = evaluate(
        teacher_model, test_loader, _ce, device,
        modality_bands_dict, modalities_to_use=(teacher_modality,),
        multilabel=multilabel, label_key=label_key,
        segmentation=is_segmentation, num_classes=task_config.num_classes,
        ignore_index=ignore_index,
    )
    print(f"Teacher test {metric_name}: {teacher_test_metric:.2f}%")
    wandb.run.summary['teacher_test_metric'] = teacher_test_metric

    print("\n=== Creating student model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[args.model]
    evan_model = model_fn(
        tz_fusion_time=3,
        tz_lora_rank=0,
        tz_modality_specific_layer_augmenter='fft',
        n_storage_tokens=4,
        starting_modality=args.modality,
        starting_n_chans=num_student_chans,
        img_size=task_config.img_size,
        device=device,
        load_weights=False,
    )
    if is_segmentation:
        student_model = EvanSegmenter(evan_model, num_classes=task_config.num_classes,
                                      decoder_strategy="mean", device=device)
    else:
        student_model = EVANClassifier(evan_model, num_classes=task_config.num_classes,
                                       classifier_strategy="mean", global_rep='clstoken', device=device)
    student_model = student_model.to(device)

    student_model.freeze_all()
    student_model.set_requires_grad('backbone', blocks=True, norm=True)
    student_model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True,
                                    msla=True, modality_encoders=True, head=True)

    trainable_params = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in student_model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, student_model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%m%d_%H%M')
    checkpoint_path = os.path.join(
        args.checkpoint_dir,
        f'sweep_distill_{wandb.run.id}_{args.dataset}_{teacher_modality}_to_{args.modality}_{timestamp}.pt'
    )

    print(f"\n=== Distillation training for {args.epochs} epochs ===")
    train_metric, test_metric, best_test_metric, best_epoch = distillation_training_loop(
        student_model, teacher_model, train2_loader, test_loader, device,
        modality_bands_dict, modality_bands_dict,
        optimizer, args.epochs, args.modality, teacher_modality,
        temperature=temperature, alpha=args.alpha,
        distillation_mode=args.distillation_mode,
        use_wandb=True, wandb_prefix='distill',
        multilabel=multilabel, label_key=label_key,
        segmentation=is_segmentation, num_classes=task_config.num_classes,
        ignore_index=ignore_index,
        val_loader=val2_loader, warmup_epochs=args.warmup_epochs,
        task_config=task_config,
    )

    ensemble_metric = evaluate_ensemble(
        student_model, teacher_model, test_loader, device,
        modality_bands_dict, modality_bands_dict,
        args.modality, teacher_modality, ensemble_mode='avg',
        multilabel=multilabel, label_key=label_key,
        segmentation=is_segmentation, num_classes=task_config.num_classes,
        ignore_index=ignore_index,
    )

    wandb.run.summary['best_test_metric'] = best_test_metric
    wandb.run.summary['final_test_metric'] = test_metric
    wandb.run.summary['ensemble_metric'] = ensemble_metric

    student_model.save_checkpoint(checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    filename = args.results_csv
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "wandb_run_id", "dataset", "teacher_modality", "student_modality",
        "model", "distillation_mode", "lr", "weight_decay", "temperature", "alpha",
        "warmup_epochs", "epochs", "trainable_params", "metric_name",
        "teacher_test_metric", "final_test_metric", "best_test_metric",
        "ensemble_metric_avg", "checkpoint", "teacher_checkpoint",
    ]
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            wandb.run.id, args.dataset, teacher_modality, args.modality,
            args.model, args.distillation_mode, lr, weight_decay, temperature, args.alpha,
            args.warmup_epochs, args.epochs, trainable_params, metric_name,
            f"{teacher_test_metric:.2f}", f"{test_metric:.2f}", f"{best_test_metric:.2f}",
            f"{ensemble_metric:.2f}", checkpoint_path, args.teacher_checkpoint,
        ])
    print(f"\nResults appended to {filename}")
    wandb.finish()


if __name__ == '__main__':
    main()
