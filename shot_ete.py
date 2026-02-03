
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
    parser = argparse.ArgumentParser(description='End to end training for SHOT model.')
    # IMPORTANT
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to stage 0 checkpoint (required)')
    parser.add_argument('--new_mod_group', type=str, required=True, choices=['vre', 'nir', 'swir','rgb'],
                        help='New modality group to train')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.75,
                        help='Mask ratio for MAE training (default: 0.75)')
    parser.add_argument('--modality_dropout', type=float, default=0.3,
                        help='Probability of fully masking a modality')
    parser.add_argument('--mae_modalities', type=str, default="all", choices=["all","newmod"]) # jan30 update: all > newmod
    parser.add_argument('--labeled_frequency', type=float, default=0.3,
                        help='Frequency of labeled monomodal batches from train1 (0-1, default: 0.3)')
    parser.add_argument('--labeled_start_fraction', type=float, default=0.0,
                        help='Fraction of training before labeled mixing starts (0=start, 0.5=halfway, 1=never)')
    parser.add_argument('--active_losses', type=str, nargs='+', default=None,
                        choices=['mae', 'latent', 'prefusion', 'distill', 'ce'],
                        help='Which losses to activate (default: all)')
    parser.add_argument('--use_mfla', action='store_true',
                        help='Enable MFLA training for hallucinated modalities')
    parser.add_argument('--mfla_warmup_epochs', type=int, default=0,
                        help='Epochs where backbone + MFLA train together before freezing backbone (default: 0)')
    parser.add_argument('--results_csv', type=str, required=True,
                        help='Path to results CSV file')

    # UNIMPORTANT
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=4,
                        help='Epochs for fusion MAE training (default: 4)')
    parser.add_argument('--eval_every_n_epochs', type=int, default=4,
                        help='Evaluate every N epochs during training (default: None, only eval at end)')
    parser.add_argument('--ssl_lr', type=float, default=1e-4,
                        help='Learning rate for fusion MAE training (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW optimizer (default: 0.01)')
    parser.add_argument('--wandb_project', type=str, default='delulu-e2e-lossablate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    args = parser.parse_args()

    # Load stage 1 checkpoint
    print(f"\n=== Loading Stage 0 checkpoint from: {args.stage0_checkpoint} ===")
    checkpoint = torch.load(args.stage0_checkpoint, map_location='cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=EVANClassifier.from_checkpoint(args.stage0_checkpoint,device)
    config = checkpoint['config']
    evan_config = config['evan_config']
    starting_modality=evan_config['starting_modality']

    print(f"Stage 0 config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    print(f"\nUsing device: {device}")

    newmod = args.new_mod_group
    modality_bands_dict = get_modality_bands_dict(starting_modality, newmod)
    bands_newmod = modality_bands_dict[newmod] # list of band number
    bands_full = tuple(ALL_BAND_NAMES) # tuple of band number

    wandb.init(
        project=args.wandb_project,
        config={**config, **vars(args)},
        name=f"{starting_modality}+={newmod}--mae{args.mae_modalities}_lf{args.labeled_frequency}"
    )

    # Create datasets
    print("\n=== Creating datasets ===")

    train1_loader, val1_loader, train2_loader, val2_loader, test_loader = get_loaders_with_val(args.batch_size, args.num_workers)

    evan = model.evan
    model = model.to(device)
    print(f"SSL-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")

    num_newmod_channels = len(bands_newmod)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.create_modality_components(newmod,num_newmod_channels)
        model = model.to(device)  # Move newly created components to device
    
    # ========================================== TRAIN SHOT ===========================================
    print(f"\n Using SHOT (MAE + Latent Distillation + Sequence Projection) training method for fusion blocks")
    match args.mae_modalities:
        case "all":
            print(f"requiring mae from all modalities.")
            mae_modalities=[starting_modality,newmod]
        case "newmod":
            print(f"requiring mae from newmod only.")
            mae_modalities=[newmod]
        
    _,_,intermediate_projectors,trainable_total,best_checkpoints,best_checkpoint_summary=train_shot(
        model=model,
        train_loader=train2_loader,
        device=device,
        args=args,
        mae_modalities=mae_modalities,
        latent_reconstruct_modalities=[starting_modality],
        modality_bands_dict=modality_bands_dict,
        test_loader=test_loader,
        eval_every_n_epochs=args.eval_every_n_epochs,
        labeled_train_loader=train1_loader,
        labeled_frequency=args.labeled_frequency,
        labeled_start_fraction=args.labeled_start_fraction,
        active_losses=args.active_losses,
        weight_decay=args.weight_decay,
        # Validation-based checkpoint selection
        val_loader=val2_loader,           # unlabeled multimodal for teacher agreement
        val_labeled_loader=val1_loader,   # labeled monomodal for peeking accuracy
        # MFLA training options
        use_mfla=args.use_mfla,
        mfla_warmup_epochs=args.mfla_warmup_epochs,
    )

    # ========================================= EVALUATION =====================================
    print("\n=== Evaluating trained model ===")

    all_modalities = [starting_modality, newmod]

    # Define evaluation objectives
    objectives = {
        "transfer": {"desc": f"Using only {newmod}, hallucinating {starting_modality}"},
        "peeking": {"desc": f"Using only {starting_modality}, hallucinating {newmod}"},
        "addition": {"desc": f"Using both {starting_modality} and {newmod}"}
    }

    # Run evaluations and store accuracies
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
            objective=objective,
            use_mfla=args.use_mfla,
        )

        accuracies[objective] = accuracy
        if objective == "addition" and ens_acc is not None:
            addition_ens_acc = ens_acc
            wandb.log({"test/addition_ens_accuracy": ens_acc})
        print(f"{objective.capitalize()} accuracy: {accuracy:.2f}%")
        wandb.log({f"test/{objective}_accuracy": accuracy})

    # ========================================= CHECKPOINT =====================================
    timestamp_shot = datetime.now().strftime('%m%d_%H%M')
    checkpoint_shotete = os.path.join(args.checkpoint_dir, f'delulunet_eurosat_{timestamp_shot}.pt')
    if args.checkpoint_name:
        checkpoint_shotete = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    # Save model checkpoint with intermediate_projectors included
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'intermediate_projectors_state_dict': intermediate_projectors.state_dict() if intermediate_projectors is not None else None,
    }
    torch.save(checkpoint_data, checkpoint_shotete)
    print(f"SHOT checkpoint saved to: {checkpoint_shotete} (includes intermediate_projectors)")

    # Log results to CSV
    filename = args.results_csv
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "starting_modality","new_modality", "ssl_lr", "weight_decay", "epochs",
        "mask_ratio", "modality_dropout","labeled_frequency","labeled_start_fraction","trainable_params", "active_losses",
        "use_mfla", "mfla_warmup_epochs",
        "transfer_acc", "peeking_acc", "addition_acc", "addition_ens_acc",
        "valchecked_transfer_acc", "valchecked_peek_acc", "valchecked_add_acc", "valchecked_add_ens_acc",
        "stage0_checkpoint", "shote2e_checkpoint"
    ]

    # Extract val-checked test accuracies from best_checkpoint_summary
    def get_valchecked(ckpt_name, metric_key):
        if ckpt_name in best_checkpoint_summary:
            return f"{best_checkpoint_summary[ckpt_name][metric_key]:.2f}"
        return ""

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        # Format active_losses for CSV (join with +, or "all" if None was passed)
        active_losses_str = "+".join(args.active_losses) if args.active_losses else "all"
        writer.writerow([
            starting_modality,
            newmod,
            args.ssl_lr,
            args.weight_decay,
            args.epochs,
            args.mae_mask_ratio,
            args.modality_dropout,
            args.labeled_frequency,
            args.labeled_start_fraction,
            trainable_total,
            active_losses_str,
            args.use_mfla,
            args.mfla_warmup_epochs,
            f"{accuracies['transfer']:.2f}",
            f"{accuracies['peeking']:.2f}",
            f"{accuracies['addition']:.2f}",
            f"{addition_ens_acc:.2f}" if addition_ens_acc is not None else "",
            get_valchecked('best_transfer', 'test_transfer'),
            get_valchecked('best_peeking', 'test_peeking'),
            get_valchecked('best_addition', 'test_addition'),
            get_valchecked('best_ens_addition', 'test_addition_ens'),
            args.stage0_checkpoint,
            checkpoint_shotete,
        ])

    print(f"\nResults appended to {filename}")
    wandb.finish()
    return

if __name__ == '__main__':
    main()


# DRYRUN (nir to rgb)
"""
source env.sh
python -u shot_ete.py \
    --new_mod_group rgb \
    --checkpoint_name nir_to_rgb-dryrun \
    --stage0_checkpoint checkpoints/nir_fft.pt  \
    --epochs 4 \
    --eval_every_n_epochs 1 \
    --batch_size 64 \
    --results_csv res/shot_ete_dryrun.csv \
    --labeled_frequency 0.1 \
    --use_mfla \
    --mfla_warmup_epochs 1
"""