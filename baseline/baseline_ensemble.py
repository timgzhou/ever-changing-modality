"""Baseline Ensemble: Evaluate ensemble of two single-modality models."""

import torch
import torch.nn.functional as F
import logging
import os
import csv
from tqdm import tqdm
from itertools import combinations

from evan_main import EVANClassifier
from eurosat_data_utils import (
    get_loaders,
    get_modality_bands_dict,
    create_multimodal_batch,
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

MODALITIES = ['rgb', 'vre', 'nir', 'swir']

def evaluate_ensemble(model_a, model_b, test_loader, device,
                      modality_a, modality_b,
                      modality_bands_dict_a, modality_bands_dict_b,
                      temp_a=1.0, temp_b=1.0):
    """
    Evaluate ensemble of two models with temperature scaling.

    Args:
        model_a: First model (on modality_a)
        model_b: Second model (on modality_b)
        test_loader: Test dataloader
        device: torch device
        modality_a: First modality name
        modality_b: Second modality name
        modality_bands_dict_a: Band dict for modality_a
        modality_bands_dict_b: Band dict for modality_b
        temp_a: Temperature for model A logits
        temp_b: Temperature for model B logits

    Returns:
        Tuple of (ensemble_acc, oracle_acc, model_a_acc, model_b_acc)
    """
    model_a.eval()
    model_b.eval()

    ensemble_correct = 0
    oracle_correct = 0
    model_a_correct = 0
    model_b_correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Eval {modality_a}+{modality_b}"):
            labels = batch['label'].to(device)

            # Get inputs for both modalities
            input_a = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict_a, modalities=(modality_a,)
            )
            input_a = {k: v.to(device) for k, v in input_a.items()}

            input_b = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict_b, modalities=(modality_b,)
            )
            input_b = {k: v.to(device) for k, v in input_b.items()}

            # Get logits from both models
            logits_a = model_a(input_a)
            logits_b = model_b(input_b)

            # Ensemble: equal weight average of temperature-scaled softmax scores
            probs_a = F.softmax(logits_a / temp_a, dim=-1)
            probs_b = F.softmax(logits_b / temp_b, dim=-1)
            ensemble_probs = 0.5 * probs_a + 0.5 * probs_b

            _, ensemble_pred = ensemble_probs.max(1)
            _, pred_a = logits_a.max(1)
            _, pred_b = logits_b.max(1)

            # Count correct predictions
            correct_a = pred_a.eq(labels)
            correct_b = pred_b.eq(labels)

            ensemble_correct += ensemble_pred.eq(labels).sum().item()
            oracle_correct += (correct_a | correct_b).sum().item()
            model_a_correct += correct_a.sum().item()
            model_b_correct += correct_b.sum().item()
            total += labels.size(0)

    ensemble_acc = 100. * ensemble_correct / total
    oracle_acc = 100. * oracle_correct / total
    model_a_acc = 100. * model_a_correct / total
    model_b_acc = 100. * model_b_correct / total

    return ensemble_acc, oracle_acc, model_a_acc, model_b_acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Create loaders (train1, train2/val, test)
    print("\n=== Creating data loaders ===")
    _, val_loader, test_loader = get_loaders(32, 4)

    # Results storage
    results = []

    # Loop through all modality combinations
    for modality_a, modality_b in combinations(MODALITIES, 2):
        print(f"\n{'='*60}")
        print(f"Evaluating ensemble: {modality_a.upper()} + {modality_b.upper()}")
        print(f"{'='*60}")

        # Load checkpoints
        checkpoint_a = f"checkpoints/{modality_a}_fft.pt"
        checkpoint_b = f"checkpoints/{modality_b}_fft.pt"

        if not os.path.exists(checkpoint_a):
            print(f"  Skipping: {checkpoint_a} not found")
            continue
        if not os.path.exists(checkpoint_b):
            print(f"  Skipping: {checkpoint_b} not found")
            continue

        print(f"  Loading {checkpoint_a}...")
        model_a = EVANClassifier.from_checkpoint(checkpoint_a, device=device)
        model_a = model_a.to(device)
        model_a.eval()

        print(f"  Loading {checkpoint_b}...")
        model_b = EVANClassifier.from_checkpoint(checkpoint_b, device=device)
        model_b = model_b.to(device)
        model_b.eval()

        # Get band dicts
        modality_bands_dict_a = get_modality_bands_dict(modality_a)
        modality_bands_dict_b = get_modality_bands_dict(modality_b)

        # === Symmetric ensemble (equal weights, temp=1) ===
        print("\n  --- Symmetric Ensemble (temp=1) ---")
        ensemble_acc, oracle_acc, model_a_acc, model_b_acc = evaluate_ensemble(
            model_a, model_b, test_loader, device,
            modality_a, modality_b,
            modality_bands_dict_a, modality_bands_dict_b,
            temp_a=1.0, temp_b=1.0
        )

        print(f"    {modality_a.upper()} alone:    {model_a_acc:.2f}%")
        print(f"    {modality_b.upper()} alone:    {model_b_acc:.2f}%")
        print(f"    Ensemble:        {ensemble_acc:.2f}%")
        print(f"    Oracle:          {oracle_acc:.2f}%")


        results.append({
            'modality_a': modality_a,
            'modality_b': modality_b,
            'model_a_acc': model_a_acc,
            'model_b_acc': model_b_acc,
            'ensemble_acc': ensemble_acc,
            'oracle_acc': oracle_acc,
        })

        # Free memory
        del model_a, model_b, val_logits_a, val_logits_b, val_labels
        torch.cuda.empty_cache()

    # Save results to CSV
    filename = "res/baseline_ensemble.csv"
    os.makedirs("res", exist_ok=True)
    fieldnames = ["modality_a", "modality_b", "model_a_acc", "model_b_acc",
                  "ensemble_acc", "oracle_acc",
                  "best_temp_a", "best_temp_b"]

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({
                'modality_a': row['modality_a'],
                'modality_b': row['modality_b'],
                'model_a_acc': f"{row['model_a_acc']:.2f}",
                'model_b_acc': f"{row['model_b_acc']:.2f}",
                'ensemble_acc': f"{row['ensemble_acc']:.2f}",
                'oracle_acc': f"{row['oracle_acc']:.2f}",
                'best_temp_a': row['best_temp_a'],
                'best_temp_b': row['best_temp_b'],
            })

    print(f"\n{'='*60}")
    print(f"Results saved to {filename}")
    print(f"{'='*60}")

    # Print summary table
    print("\nSummary:")
    print(f"{'Pair':<12} {'A':>8} {'B':>8} {'Ensemble':>10} {'Oracle':>8}")
    print("-" * 70)
    for r in results:
        pair = f"{r['modality_a']}+{r['modality_b']}"
        print(f"{pair:<12} {r['model_a_acc']:>7.2f}% {r['model_b_acc']:>7.2f}% {r['ensemble_acc']:>9.2f}%{r['oracle_acc']:>7.2f}%")


if __name__ == '__main__':
    main()
