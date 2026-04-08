"""
Hyperparameter sweep runner for SHOT training.
Wraps shot_ete.py training logic with W&B Sweeps integration.

Usage:
    wandb sweep sweep_config.yaml
    wandb agent <sweep-id>
"""

import sys
import os
# Ensure repo root is on the path regardless of cwd or how the process was spawned
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shot import train_shot
import torch
import logging
import os
import argparse
import csv
from datetime import datetime
import wandb

from evan_main import EVANClassifier, EvanSegmenter
from data_utils import get_loaders
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    # Parse fixed args that won't be swept
    parser = argparse.ArgumentParser(description='Hyperparameter sweep for SHOT training.')
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'],
                        help='Dataset to train on (default: eurosat)')
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to stage 0 checkpoint (required)')
    parser.add_argument('--new_mod_group', type=str, required=True,
                        help='New modality to add (valid values depend on dataset)')
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
    # Swept hyperparameters - defaults here, wandb sweep overrides via command line
    parser.add_argument('--mae_mask_ratio', type=float, default=0.75)
    parser.add_argument('--modality_dropout', type=float, default=0.3)
    parser.add_argument('--labeled_frequency', type=float, default=0.3)
    parser.add_argument('--labeled_start_fraction', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--use_mae', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--use_latent', type=lambda x: x.lower() == 'true', default=True)
    parser.add_argument('--lambda_mae', type=float, default=1.0)
    parser.add_argument('--lambda_latent', type=float, default=1.0)
    parser.add_argument('--lambda_prefusion', type=float, default=1.0)
    parser.add_argument('--lambda_distill', type=float, default=1.0)
    parser.add_argument('--lambda_ce', type=float, default=1.0)
    args = parser.parse_args()

    # Initialize wandb - sweep will override config
    wandb.init(project=args.wandb_project)
    config = wandb.config

    # Get swept hyperparameters from args (passed by wandb sweep via command line)
    mae_mask_ratio = args.mae_mask_ratio
    modality_dropout = args.modality_dropout
    labeled_frequency = args.labeled_frequency
    labeled_start_fraction = args.labeled_start_fraction
    lr = args.lr
    weight_decay = args.weight_decay
    use_mae = args.use_mae
    use_latent = args.use_latent
    loss_weights = {
        'mae': args.lambda_mae, 'latent': args.lambda_latent,
        'prefusion': args.lambda_prefusion, 'distill': args.lambda_distill,
        'ce': args.lambda_ce,
    }

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
    print(f"  lr: {lr}")
    print(f"  weight_decay: {weight_decay}")
    print(f"  use_mae: {use_mae}")
    print(f"  use_latent: {use_latent}")
    print(f"  active_losses: {active_losses}")

    # Create a namespace object to pass to train_shot (mimicking argparse args)
    class Args:
        pass

    train_args = Args()
    train_args.lr = lr
    train_args.epochs = args.epochs
    train_args.mae_mask_ratio = mae_mask_ratio
    train_args.modality_dropout = modality_dropout

    # Load stage 0 checkpoint
    print(f"\n=== Loading Stage 0 checkpoint from: {args.stage0_checkpoint} ===")
    checkpoint = torch.load(args.stage0_checkpoint, map_location='cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_config = checkpoint['config']
    evan_config = model_config['evan_config']
    starting_modality = evan_config['starting_modality']
    is_segmentation = (model_config.get('task_type') == 'segmentation'
                       or args.dataset in ('pastis', 'dfc2020'))
    if is_segmentation:
        model = EvanSegmenter.from_checkpoint(args.stage0_checkpoint, device)
    else:
        model = EVANClassifier.from_checkpoint(args.stage0_checkpoint, device)

    print(f"Stage 0 config:")
    for k, v in model_config.items():
        print(f"  {k}: {v}")

    print(f"\nUsing device: {device}")

    newmod = args.new_mod_group

    # Update wandb config with full info
    wandb.config.update({
        'starting_modality': starting_modality,
        'new_modality': newmod,
        'active_losses': '+'.join(active_losses),
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'stage0_checkpoint': args.stage0_checkpoint,
    }, allow_val_change=True)

    wandb.run.name = f"{starting_modality}+={newmod}_lr{lr:.0e}_wd{weight_decay:.0e}_lf{labeled_frequency}"

    # Create datasets
    print("\n=== Creating datasets ===")
    train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config = get_loaders(
        args.dataset, starting_modality, args.batch_size, args.num_workers, new_modality=newmod
    )
    modality_bands_dict = task_config.modality_bands_dict
    bands_newmod = modality_bands_dict[newmod]

    evan = model.evan
    model = model.to(device)

    bands_spec = bands_newmod
    num_newmod_channels = (bands_spec.stop - bands_spec.start) if isinstance(bands_spec, slice) else len(bands_spec)
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
    trainable_total, best_checkpoints, _, teacher_baselines = train_shot(
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
        loss_weights=loss_weights,
        weight_decay=weight_decay,
        val_unlabeled_loader=val2_loader,
        val_labeled_loader=val1_loader,
        multilabel=task_config.multilabel,
        label_key=task_config.label_key,
        segmentation=is_segmentation,
        num_classes=task_config.num_classes if is_segmentation else None,
        ignore_index=getattr(task_config, 'ignore_index', -100),
    )

    # Log teacher baselines to wandb
    teacher_test_metric = teacher_baselines.get('test')
    if teacher_test_metric is not None:
        wandb.run.summary['teacher_test_metric'] = teacher_test_metric

    # Log best checkpoint test accuracies to wandb summary
    for ckpt_name, ckpt_data in best_checkpoints.items():
        if ckpt_data['test_accs'] is not None:
            wandb.run.summary[f'{ckpt_name}_test_transfer'] = ckpt_data['test_accs']['transfer']
            wandb.run.summary[f'{ckpt_name}_test_peeking'] = ckpt_data['test_accs']['peeking']
            wandb.run.summary[f'{ckpt_name}_test_addition'] = ckpt_data['test_accs']['addition']
            wandb.run.summary[f'{ckpt_name}_test_addition_ens'] = ckpt_data['test_accs'].get('addition_ens', 0)

    # ========================================= CHECKPOINT =====================================
    timestamp_shot = datetime.now().strftime('%m%d_%H%M')
    checkpoint_shotete = os.path.join(args.checkpoint_dir, f'sweep_{wandb.run.id}_{timestamp_shot}.pt')

    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'sweep_config': {
            'mae_mask_ratio': mae_mask_ratio,
            'modality_dropout': modality_dropout,
            'labeled_frequency': labeled_frequency,
            'labeled_start_fraction': labeled_start_fraction,
            'lr': lr,
            'weight_decay': weight_decay,
            'use_mae': use_mae,
            'use_latent': use_latent,
            'active_losses': active_losses,
        }
    }
    torch.save(checkpoint_data, checkpoint_shotete)
    print(f"Checkpoint saved to: {checkpoint_shotete}")

    # ========================================= CSV LOGGING =====================================
    # Extract val metrics and test accuracies from best checkpoints
    # Each objective uses its own best checkpoint (by validation metric)
    def get_ckpt_data(ckpt_name, test_key):
        ckpt = best_checkpoints.get(ckpt_name, {})
        val_metric = ckpt.get('metric')
        test_accs = ckpt.get('test_accs')
        test_acc = test_accs.get(test_key) if test_accs else None
        return val_metric, test_acc

    val_transfer, test_transfer = get_ckpt_data('best_transfer', 'transfer')
    val_peeking, test_peeking = get_ckpt_data('best_peeking', 'peeking')
    val_addition, test_addition = get_ckpt_data('best_addition', 'addition')
    val_ens_addition, test_ens_addition = get_ckpt_data('best_ens_addition', 'ens')

    filename = args.results_csv
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "wandb_run_id", "dataset", "starting_modality", "new_modality",
        "teacher_test_metric",
        "lr", "weight_decay", "epochs",
        "mask_ratio", "modality_dropout", "labeled_frequency", "labeled_start_fraction",
        "use_mae", "use_latent",
        "lambda_mae", "lambda_latent", "lambda_prefusion", "lambda_distill", "lambda_ce",
        "trainable_params", "active_losses",
        "val_transfer", "test_transfer",
        "val_peeking", "test_peeking",
        "val_addition", "test_addition",
        "val_ens_addition", "test_ens_addition",
        "stage0_checkpoint", "shote2e_checkpoint"
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        active_losses_str = "+".join(active_losses)
        writer.writerow([
            wandb.run.id,
            args.dataset,
            starting_modality,
            newmod,
            f"{teacher_test_metric:.2f}" if teacher_test_metric is not None else "",
            lr,
            weight_decay,
            args.epochs,
            mae_mask_ratio,
            modality_dropout,
            labeled_frequency,
            labeled_start_fraction,
            use_mae,
            use_latent,
            loss_weights['mae'],
            loss_weights['latent'],
            loss_weights['prefusion'],
            loss_weights['distill'],
            loss_weights['ce'],
            trainable_total,
            active_losses_str,
            f"{val_transfer:.2f}" if val_transfer is not None else "",
            f"{test_transfer:.2f}" if test_transfer is not None else "",
            f"{val_peeking:.2f}" if val_peeking is not None else "",
            f"{test_peeking:.2f}" if test_peeking is not None else "",
            f"{val_addition:.2f}" if val_addition is not None else "",
            f"{test_addition:.2f}" if test_addition is not None else "",
            f"{val_ens_addition:.2f}" if val_ens_addition is not None else "",
            f"{test_ens_addition:.2f}" if test_ens_addition is not None else "",
            args.stage0_checkpoint,
            checkpoint_shotete,
        ])

    print(f"\nResults appended to {filename}")
    wandb.finish()


if __name__ == '__main__':
    main()
