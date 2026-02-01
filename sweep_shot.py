"""
Hyperparameter sweep runner for SHOT training.
Wraps shot_ete.py training logic with W&B Sweeps integration.

Usage:
    wandb sweep sweep_config.yaml
    wandb agent <sweep-id>
"""

from shot import train_shot
import torch
import logging
import os
import argparse
import csv
from datetime import datetime
import wandb

from evan_main import EVANClassifier
from eurosat_data_utils import (
    ALL_BAND_NAMES,
    get_loaders_with_val,
    get_modality_bands_dict
)
from train_utils import _delulu_stage3_test

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    # Parse fixed args that won't be swept
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for SHOT training.')
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to stage 0 checkpoint (required)')
    parser.add_argument('--new_mod_group', type=str, required=True, choices=['vre', 'nir', 'swir', 'rgb'],
                        help='New modality group to train')
    parser.add_argument('--results_csv', type=str, default='res/sweep_results.csv',
                        help='Path to results CSV file')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=64,
                        help='Epochs for fusion MAE training (default: 64)')
    parser.add_argument('--eval_every_n_epochs', type=int, default=4,
                        help='Evaluate every N epochs during training')
    parser.add_argument('--wandb_project', type=str, default='delulu-sweep')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--mae_modalities', type=str, default="all", choices=["all", "newmod"])
    args = parser.parse_args()

    # Initialize wandb - sweep will override config
    wandb.init(project=args.wandb_project)
    config = wandb.config

    # Get swept hyperparameters from wandb config (with defaults)
    mae_mask_ratio = config.get('mae_mask_ratio', 0.75)
    modality_dropout = config.get('modality_dropout', 0.3)
    labeled_frequency = config.get('labeled_frequency', 0.3)
    labeled_start_fraction = config.get('labeled_start_fraction', 0.0)
    ssl_lr = config.get('ssl_lr', 1e-4)
    weight_decay = config.get('weight_decay', 0.01)
    use_mae = config.get('use_mae', True)
    use_latent = config.get('use_latent', True)

    # Build active_losses based on labeled_frequency and optional losses
    # Base losses: prefusion + distill always, +ce when using labeled data
    if labeled_frequency == 0:
        active_losses = ['prefusion', 'distill']
    else:
        active_losses = ['prefusion', 'distill', 'ce']

    # Optional losses
    if use_mae:
        active_losses.append('mae')
    if use_latent:
        active_losses.append('latent')

    print(f"\n=== Sweep Configuration ===")
    print(f"  mae_mask_ratio: {mae_mask_ratio}")
    print(f"  modality_dropout: {modality_dropout}")
    print(f"  labeled_frequency: {labeled_frequency}")
    print(f"  labeled_start_fraction: {labeled_start_fraction}")
    print(f"  ssl_lr: {ssl_lr}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  use_mae: {use_mae}")
    print(f"  use_latent: {use_latent}")
    print(f"  active_losses: {active_losses}")

    # Create a namespace object to pass to train_shot (mimicking argparse args)
    class Args:
        pass

    train_args = Args()
    train_args.ssl_lr = ssl_lr
    train_args.epochs = args.epochs
    train_args.mae_mask_ratio = mae_mask_ratio
    train_args.modality_dropout = modality_dropout

    # Load stage 0 checkpoint
    print(f"\n=== Loading Stage 0 checkpoint from: {args.stage0_checkpoint} ===")
    checkpoint = torch.load(args.stage0_checkpoint, map_location='cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EVANClassifier.from_checkpoint(args.stage0_checkpoint, device)
    model_config = checkpoint['config']
    evan_config = model_config['evan_config']
    starting_modality = evan_config['starting_modality']

    print(f"Stage 0 config:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")

    print(f"\nUsing device: {device}")

    newmod = args.new_mod_group
    modality_bands_dict = get_modality_bands_dict(starting_modality, newmod)
    bands_newmod = modality_bands_dict[newmod]

    # Update wandb config with full info
    wandb.config.update({
        'starting_modality': starting_modality,
        'new_modality': newmod,
        'active_losses': '+'.join(active_losses),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'stage0_checkpoint': args.stage0_checkpoint,
    }, allow_val_change=True)

    wandb.run.name = f"{starting_modality}+={newmod}_lr{ssl_lr:.0e}_wd{weight_decay:.0e}_lf{labeled_frequency}"

    # Create datasets
    print("\n=== Creating datasets ===")
    train1_loader, val1_loader, train2_loader, val2_loader, test_loader = get_loaders_with_val(
        args.batch_size, args.num_workers
    )

    evan = model.evan
    model = model.to(device)

    num_newmod_channels = len(bands_newmod)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.create_modality_components(newmod, num_newmod_channels)
        model = model.to(device)

    # Determine mae_modalities
    if args.mae_modalities == "all":
        mae_modalities = [starting_modality, newmod]
    else:
        mae_modalities = [newmod]

    # ========================================== TRAIN SHOT ===========================================
    print(f"\n=== Training with SHOT ===")
    _, _, intermediate_projectors, trainable_total = train_shot(
        model=model,
        train_loader=train2_loader,
        device=device,
        args=train_args,
        mae_modalities=mae_modalities,
        latent_reconstruct_modalities=[starting_modality],
        modality_bands_dict=modality_bands_dict,
        test_loader=test_loader,
        eval_every_n_epochs=args.eval_every_n_epochs,
        labeled_train_loader=train1_loader,
        labeled_frequency=labeled_frequency,
        labeled_start_fraction=labeled_start_fraction,
        active_losses=active_losses,
        weight_decay=weight_decay,
        val_loader=val2_loader,
        val_labeled_loader=val1_loader,
    )

    # ========================================= EVALUATION =====================================
    print("\n=== Evaluating trained model ===")

    all_modalities = [starting_modality, newmod]

    objectives = {
        "transfer": {"desc": f"Using only {newmod}, hallucinating {starting_modality}"},
        "peeking": {"desc": f"Using only {starting_modality}, hallucinating {newmod}"},
        "addition": {"desc": f"Using both {starting_modality} and {newmod}"}
    }

    accuracies = {}
    addition_ens_acc = None
    for objective, info in objectives.items():
        print(f"\n--- Evaluating: {objective.capitalize()} objective ---")
        print(f"    {info['desc']}")

        accuracy, ens_acc = _delulu_stage3_test(
            model=model,
            evan=model.evan,
            test_loader=test_loader,
            device=device,
            modality_bands_dict=modality_bands_dict,
            unlabeled_modalities=[newmod],
            labeled_modalities=[starting_modality],
            all_modalities=all_modalities,
            intermediate_projectors=intermediate_projectors,
            objective=objective
        )

        accuracies[objective] = accuracy
        if objective == "addition" and ens_acc is not None:
            addition_ens_acc = ens_acc
            wandb.log({"test/addition_ens_accuracy": ens_acc})
        print(f"{objective.capitalize()} accuracy: {accuracy:.2f}%")
        wandb.log({f"test/{objective}_accuracy": accuracy})

    # ========================================= CHECKPOINT =====================================
    timestamp_shot = datetime.now().strftime('%m%d_%H%M')
    checkpoint_shotete = os.path.join(args.checkpoint_dir, f'sweep_{wandb.run.id}_{timestamp_shot}.pt')

    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'intermediate_projectors_state_dict': intermediate_projectors.state_dict() if intermediate_projectors is not None else None,
        'sweep_config': {
            'mae_mask_ratio': mae_mask_ratio,
            'modality_dropout': modality_dropout,
            'labeled_frequency': labeled_frequency,
            'labeled_start_fraction': labeled_start_fraction,
            'ssl_lr': ssl_lr,
            'weight_decay': weight_decay,
            'use_mae': use_mae,
            'use_latent': use_latent,
            'active_losses': active_losses,
        }
    }
    torch.save(checkpoint_data, checkpoint_shotete)
    print(f"Checkpoint saved to: {checkpoint_shotete}")

    # ========================================= CSV LOGGING =====================================
    filename = args.results_csv
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "wandb_run_id", "starting_modality", "new_modality",
        "ssl_lr", "weight_decay", "epochs",
        "mask_ratio", "modality_dropout", "labeled_frequency", "labeled_start_fraction",
        "use_mae", "use_latent", "trainable_params", "active_losses",
        "transfer_acc", "peeking_acc", "addition_acc", "addition_ens_acc",
        "stage0_checkpoint", "shote2e_checkpoint"
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        active_losses_str = "+".join(active_losses)
        writer.writerow([
            wandb.run.id,
            starting_modality,
            newmod,
            ssl_lr,
            weight_decay,
            args.epochs,
            mae_mask_ratio,
            modality_dropout,
            labeled_frequency,
            labeled_start_fraction,
            use_mae,
            use_latent,
            trainable_total,
            active_losses_str,
            f"{accuracies['transfer']:.2f}",
            f"{accuracies['peeking']:.2f}",
            f"{accuracies['addition']:.2f}",
            f"{addition_ens_acc:.2f}" if addition_ens_acc is not None else "",
            args.stage0_checkpoint,
            checkpoint_shotete,
        ])

    print(f"\nResults appended to {filename}")
    wandb.finish()


if __name__ == '__main__':
    main()
