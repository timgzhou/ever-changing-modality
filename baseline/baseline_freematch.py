"""FreeMatch Semi-Supervised Baseline: Self-adaptive thresholding for SSL."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import argparse
from datetime import datetime
import wandb
import csv
from tqdm import tqdm
import kornia.augmentation as K

from evan_main import evan_small, evan_base, evan_large, evan_small_s2, BENV2_BAND_INDICES, PASTIS_BAND_INDICES, EVANClassifier, EvanSegmenter
from data_utils import get_loaders, create_multimodal_batch
from train_utils import evaluate

VALID_MODALITIES = {
    'eurosat': ['rgb', 'vre', 'nir', 'swir'],
    'benv2':   ['s1', 's2', 's2_rgb'],
    'pastis':  ['s1', 's2', 'rgb'],
    'dfc2020': ['s1', 's2', 's2_rgb'],
}

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


# ---------------------------------------------------------------------------
# Augmentation factories
# ---------------------------------------------------------------------------

def make_weak_augmentation():
    """Weak augmentation: random flips only (channel-agnostic)."""
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        data_keys=["input"],
    )


def make_strong_augmentation():
    """Strong augmentation: flips + rotation + blur + intensity + erasing (channel-agnostic)."""
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        K.RandomRotation(degrees=90.0, p=0.5),
        K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.5),
        K.RandomBrightness(brightness=(0.8, 1.2), p=0.5),
        K.RandomContrast(contrast=(0.8, 1.2), p=0.5),
        K.RandomErasing(scale=(0.02, 0.15), ratio=(0.3, 3.3), p=0.3),
        data_keys=["input"],
    )


# ---------------------------------------------------------------------------
# FreeMatch adaptive thresholding state
# ---------------------------------------------------------------------------

def _replace_inf(t):
    """Replace inf with 0."""
    return torch.where(torch.isinf(t), torch.zeros_like(t), t)


class FreeMatchState:
    """Tracks EMA variables for FreeMatch self-adaptive thresholding.

    Maintains three EMA-updated state tensors:
      - time_p: global confidence threshold (scalar for clf/seg, [C] for multilabel)
      - p_model: EMA of predicted class probability distribution [C]
      - label_hist: EMA of hard-prediction histogram [C]
    """

    def __init__(self, num_classes: int, momentum: float = 0.999,
                 multilabel: bool = False, device: str = 'cuda'):
        self.num_classes = num_classes
        self.momentum = momentum
        self.multilabel = multilabel
        self.device = device

        if multilabel:
            # Per-class scalar threshold for multilabel
            self.time_p = torch.full((num_classes,), 0.5, device=device)
            self.p_model = torch.full((num_classes,), 0.5, device=device)
            self.label_hist = torch.full((num_classes,), 0.5, device=device)
        else:
            self.time_p = torch.tensor(1.0 / num_classes, device=device)
            self.p_model = torch.ones(num_classes, device=device) / num_classes
            self.label_hist = torch.ones(num_classes, device=device) / num_classes

    @torch.no_grad()
    def update(self, probs, max_probs, pred_classes):
        """Update EMA state from weak-augmentation predictions.

        Args:
            probs: softmax probs [N, C] (clf/seg) or sigmoid probs [N, C] (multilabel)
            max_probs: [N] (clf/seg) or [N, C] (multilabel)
            pred_classes: [N] argmax indices (clf/seg), unused for multilabel
        """
        m = self.momentum
        C = self.num_classes

        if self.multilabel:
            # Per-class EMA of sigmoid confidence (distance from 0.5)
            self.time_p = m * self.time_p + (1 - m) * max_probs.mean(dim=0)
            self.p_model = m * self.p_model + (1 - m) * probs.mean(dim=0)
            # label_hist: fraction of samples predicted positive per class
            pos_rate = (probs > 0.5).float().mean(dim=0)
            self.label_hist = m * self.label_hist + (1 - m) * pos_rate
        else:
            self.time_p = m * self.time_p + (1 - m) * max_probs.mean()
            self.p_model = m * self.p_model + (1 - m) * probs.mean(dim=0)
            hist = torch.bincount(pred_classes, minlength=C).float()
            hist = hist / hist.sum().clamp(min=1)
            self.label_hist = m * self.label_hist + (1 - m) * hist

    @torch.no_grad()
    def compute_mask(self, max_probs, pred_classes):
        """Compute adaptive pseudo-label mask.

        Args:
            max_probs: [N] (clf/seg) or [N, C] (multilabel)
            pred_classes: [N] (clf/seg), unused for multilabel

        Returns:
            mask: same shape as max_probs, float 0/1
        """
        if self.multilabel:
            # Per-class threshold: time_p already per-class [C]
            # No mod scaling needed — time_p tracks per-class confidence directly
            mask = (max_probs >= self.time_p.unsqueeze(0)).float()  # [N, C]
        else:
            mod = self.p_model / self.p_model.max()
            threshold = self.time_p * mod[pred_classes]  # [N]
            mask = (max_probs >= threshold).float()
        return mask

    @torch.no_grad()
    def entropy_loss(self, logits_strong, mask, segmentation=False):
        """Full FreeMatch de-biased class-fairness entropy loss.

        Encourages balanced predictions by cross-entropy between de-biased model
        distribution and de-biased batch prediction distribution.

        Args:
            logits_strong: [B, C] (clf), [B, C] (multilabel), or [B, C, H, W] (seg)
            mask: [B] (clf), [B, C] (multilabel), or [B, H, W] (seg)
            segmentation: whether task is segmentation

        Returns:
            entropy loss scalar (to be minimized — already negated)
        """
        if self.multilabel:
            # For multilabel: encourage balanced positive rates
            probs_s = torch.sigmoid(logits_strong)  # [B, C]
            # Only use masked samples (any-class mask)
            any_mask = mask.any(dim=1)  # [B]
            if any_mask.sum() == 0:
                return torch.tensor(0.0, device=logits_strong.device)
            avg_probs = probs_s[any_mask].mean(dim=0)  # [C]
            # Simple entropy: maximize entropy of per-class positive rate
            avg_probs = avg_probs.clamp(1e-8, 1 - 1e-8)
            ent = -(avg_probs * avg_probs.log() + (1 - avg_probs) * (1 - avg_probs).log()).mean()
            return -ent  # maximize entropy → minimize -entropy

        if segmentation:
            probs_s = F.softmax(logits_strong, dim=1)  # [B, C, H, W]
            # Flatten spatial
            B, C, H, W = probs_s.shape
            probs_flat = probs_s.permute(0, 2, 3, 1).reshape(-1, C)  # [N, C]
            mask_flat = mask.reshape(-1)  # [N]
            preds_flat = logits_strong.argmax(dim=1).reshape(-1)  # [N]
        else:
            probs_flat = F.softmax(logits_strong, dim=1)  # [B, C]
            mask_flat = mask  # [B]
            preds_flat = logits_strong.argmax(dim=1)  # [B]

        if mask_flat.sum() == 0:
            return torch.tensor(0.0, device=logits_strong.device)

        C = self.num_classes
        masked_idx = mask_flat.bool()

        # 1. hist_s: class histogram of masked strong predictions
        masked_preds = preds_flat[masked_idx]
        hist_s = torch.bincount(masked_preds, minlength=C).float()
        hist_s = hist_s / hist_s.sum().clamp(min=1)

        # 2. De-bias p_model by inverse of label_hist
        inv_label_hist = _replace_inf(1.0 / self.label_hist)
        mod_prob_model = self.p_model * inv_label_hist
        mod_prob_model = mod_prob_model / mod_prob_model.sum().clamp(min=1e-8)

        # 3. De-bias mean strong probs by inverse of hist_s
        mean_prob_s = probs_flat[masked_idx].mean(dim=0)
        inv_hist_s = _replace_inf(1.0 / hist_s)
        mod_mean_prob_s = mean_prob_s * inv_hist_s
        mod_mean_prob_s = mod_mean_prob_s / mod_mean_prob_s.sum().clamp(min=1e-8)

        # 4. Cross-entropy: H(mod_prob_model, mod_mean_prob_s)
        loss = (mod_prob_model * (mod_mean_prob_s + 1e-8).log()).sum()
        return -loss  # we minimize this, which maximizes the alignment


# ---------------------------------------------------------------------------
# Helper: get task config and loaders (mirrors train_sft.py)
# ---------------------------------------------------------------------------

def _n_chans(entry) -> int:
    if isinstance(entry, slice):
        return entry.stop - entry.start
    return len(entry)


def get_task_config_and_loaders(dataset, modality, batch_size, num_workers,
                                data_normalizer=None, num_time_steps=10):
    """Return all 5 loaders + task_config for FreeMatch.

    Unlike train_sft.py which only gets train1/val1/test, we need train2/val2
    as the unlabeled data source.
    """
    train1, val1, train2, val2, test, task_config = get_loaders(
        dataset, modality, batch_size, num_workers,
        data_normalizer=data_normalizer, num_time_steps=num_time_steps,
    )
    return train1, val1, train2, val2, test, task_config


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='FreeMatch Semi-Supervised Baseline')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'])
    parser.add_argument('--modality', type=str, required=True,
                        help='Modality to train on (single modality)')
    parser.add_argument('--model', type=str, default='evan_small',
                        choices=['evan_small', 'evan_base', 'evan_large'])
    parser.add_argument('--use_dino_weights', action='store_true')
    parser.add_argument('--use_s2dino_weights', action='store_true',
                        help='Init from torchgeo S2-DINO ViT-Small (SSL4EO-S12)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_time_steps', type=int, default=10,
                        help='Timestamps per PASTIS image (default: 10)')
    parser.add_argument('--tz_fusion_time', type=int, default=3)
    parser.add_argument('--tz_lora_rank', type=int, default=32)
    parser.add_argument('--tz_modality_specific_layer_augmenter', type=str, default='fft',
                        choices=['fft'])
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--global_rep', type=str, default='clstoken',
                        choices=['clstoken', 'mean_patch'])
    parser.add_argument('--train_mode', type=str, default='fft',
                        choices=['probe', 'adaptor', 'fft', 'emb+probe'])
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--val_per_epoch', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=1)
    # FreeMatch-specific
    parser.add_argument('--lambda_u', type=float, default=1.0,
                        help='Weight for unsupervised loss (default: 1.0)')
    parser.add_argument('--lambda_e', type=float, default=0.01,
                        help='Weight for entropy fairness loss (default: 0.01)')
    parser.add_argument('--ema_momentum', type=float, default=0.999,
                        help='EMA momentum for FreeMatch state (default: 0.999)')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for pseudo-label sharpening (default: 0.5)')
    parser.add_argument('--results_csv', type=str, default=None)
    args = parser.parse_args()

    # Validate modality
    valid_mods = VALID_MODALITIES[args.dataset]
    if args.modality not in valid_mods:
        parser.error(f"--modality {args.modality!r} not valid for {args.dataset}. "
                     f"Valid: {valid_mods}")

    if args.results_csv is None:
        args.results_csv = f"res/baseline_freematch_{args.dataset}.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Dataset: {args.dataset}, Modality: {args.modality}")
    print(f"Using device: {device}")

    # Normalizer (match train_sft.py pattern)
    data_normalizer = None
    if args.dataset == 'pastis' and (
        args.use_s2dino_weights or (args.use_dino_weights and args.modality == 'rgb')
    ):
        from geobench_data_utils import make_div10000_normalizer
        data_normalizer = make_div10000_normalizer()
        print("Using /10000 normalizer to match torchgeo DINO pretraining.")

    # Data
    print("\n=== Creating datasets ===")
    train1, val1, train2, val2, test, task_config = get_task_config_and_loaders(
        args.dataset, args.modality, args.batch_size, args.num_workers,
        data_normalizer=data_normalizer, num_time_steps=args.num_time_steps,
    )
    modality_bands_dict = task_config.modality_bands_dict
    num_chans = _n_chans(modality_bands_dict[args.modality])

    is_segmentation = (task_config.task_type == 'segmentation')
    multilabel = task_config.multilabel
    metric_name = "mIoU" if is_segmentation else ("mAP" if multilabel else "Acc")

    print(f"  train1 (labeled): {len(train1.dataset)} samples")
    print(f"  train2 (unlabeled): {len(train2.dataset)} samples")
    print(f"  Modality: {args.modality} ({num_chans} channels)")
    print(f"  Task: {task_config.task_type}, Metric: {metric_name}")

    # Model (mirrors train_sft.py)
    print("\n=== Creating EVAN model ===")
    common_kwargs = dict(
        tz_fusion_time=args.tz_fusion_time,
        tz_lora_rank=args.tz_lora_rank,
        tz_modality_specific_layer_augmenter=args.tz_modality_specific_layer_augmenter,
        n_storage_tokens=4,
        starting_modality=args.modality,
        starting_n_chans=num_chans,
        img_size=task_config.img_size,
        device=device,
    )
    if args.use_s2dino_weights:
        if args.model != 'evan_small':
            parser.error('--use_s2dino_weights only with --model evan_small')
        if args.modality != 's2':
            parser.error('--use_s2dino_weights requires modality s2')
        if args.dataset == 'eurosat':
            parser.error('--use_s2dino_weights not compatible with eurosat')
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

    if is_segmentation:
        model = EvanSegmenter(
            evan, num_classes=task_config.num_classes,
            decoder_strategy="mean", device=device,
        )
    else:
        model = EVANClassifier(
            evan, num_classes=task_config.num_classes,
            classifier_strategy="mean", global_rep=args.global_rep, device=device,
        )
    model = model.to(device)

    # Freeze/unfreeze (mirrors train_sft.py)
    model.freeze_all()
    if args.train_mode == 'fft':
        model.set_requires_grad('backbone', blocks=True, norm=True)
        model.set_requires_grad('all', patch_embedders=True, clsreg=True, msla=True,
                                modality_encoders=True, head=True)
        print("Mode=fft: training full backbone layers + head.")
    elif args.train_mode == 'adaptor':
        model.set_requires_grad(args.modality, patch_embedders=True, clsreg=True,
                                msla=True, mfla=True, head=True)
        print("Mode=adaptor: training embedder, adaptors and head.")
    elif args.train_mode == 'probe':
        model.set_requires_grad('all', head=True)
        print("Mode=probe: training head only.")
    elif args.train_mode == 'emb+probe':
        model.set_requires_grad('all', patch_embedders=True, head=True)
        print("Mode=emb+probe: training embedder + head.")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    # Wandb
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            config=vars(args),
            name=f"freematch_{args.dataset}_{args.modality}_{args.train_mode}",
        )

    # Augmentations
    weak_aug = make_weak_augmentation()
    strong_aug = make_strong_augmentation()

    # FreeMatch state
    fm_state = FreeMatchState(
        num_classes=task_config.num_classes,
        momentum=args.ema_momentum,
        multilabel=multilabel,
        device=device,
    )

    # Checkpoint path
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    if args.checkpoint_name:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    else:
        checkpoint_path = os.path.join(
            args.checkpoint_dir,
            f'evan_freematch_{args.dataset}_{args.modality}_{timestamp}.pt',
        )

    # Train
    print(f"\n=== FreeMatch training for {args.epochs} epochs ===")
    print(f"  lambda_u={args.lambda_u}, lambda_e={args.lambda_e}, "
          f"temperature={args.temperature}, ema_momentum={args.ema_momentum}")

    from train_utils import Trainer
    trainer = Trainer(
        model, optimizer, device, task_config,
        clip_norm=10.0,
        use_wandb=bool(args.wandb_project),
        wandb_prefix='freematch',
        warmup_epochs=args.warmup_epochs,
    )

    train_metric, test_metric, best_val_metric, best_val_test_metric = trainer.train_freematch(
        train1_loader=train1,
        train2_loader=train2,
        val1_loader=val1,
        test_loader=test,
        num_epochs=args.epochs,
        modality=args.modality,
        weak_aug=weak_aug,
        strong_aug=strong_aug,
        freematch_state=fm_state,
        temperature=args.temperature,
        lambda_u=args.lambda_u,
        lambda_e=args.lambda_e,
        best_checkpoint_path=checkpoint_path,
        val_per_epoch=args.val_per_epoch,
    )

    print(f"\n=== FreeMatch training complete ===")
    print(f"  Train {metric_name}: {train_metric:.2f}%")
    print(f"  Test  {metric_name}: {test_metric:.2f}%")
    if best_val_metric is not None:
        print(f"  Best val {metric_name}: {best_val_metric:.2f}%")
        print(f"  Best val→test {metric_name}: {best_val_test_metric:.2f}%")

    # CSV
    filename = args.results_csv
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "model_type", "modality", "train_mode", "learning_rate", "weight_decay",
        "trainable_params", "epochs", "lambda_u", "lambda_e", "ema_momentum",
        "temperature", "metric_name", "test_metric", "best_val_metric",
        "best_val_test_metric", "saved_checkpoint", "global_rep",
        "use_dino_weights", "use_s2dino_weights",
    ]
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            args.model, args.modality, args.train_mode, args.lr, args.weight_decay,
            trainable_params, args.epochs, args.lambda_u, args.lambda_e,
            args.ema_momentum, args.temperature, metric_name,
            f"{test_metric:.2f}",
            f"{best_val_metric:.2f}" if best_val_metric is not None else "",
            f"{best_val_test_metric:.2f}" if best_val_test_metric is not None else "",
            checkpoint_path, args.global_rep,
            args.use_dino_weights, args.use_s2dino_weights,
        ])
    print(f"Results appended to {filename}")

    if args.wandb_project:
        wandb.finish()
    return checkpoint_path


if __name__ == '__main__':
    main()


# DRYRUN examples
"""
# EuroSAT RGB
python -u baseline/baseline_freematch.py --dataset eurosat --modality rgb --epochs 5 --use_dino_weights

# BEN-v2 S2
python -u baseline/baseline_freematch.py --dataset benv2 --modality s2 --epochs 10 --use_dino_weights

# DFC2020 S2 (segmentation)
python -u baseline/baseline_freematch.py --dataset dfc2020 --modality s2 --epochs 10 --use_dino_weights
"""
