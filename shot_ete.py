
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
    get_loaders,
    get_modality_bands_dict
)
from train_utils import _delulu_stage3_test
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='End to end training for SHOT model.')
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to stage 0 checkpoint (required)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--new_mod_group', type=str, default='vre', choices=['vre', 'nir', 'swir','rgb'],
                        help='New modality group to train')
    parser.add_argument('--train_method', type=str, default='shot', choices=['shot'])
    parser.add_argument('--epochs', type=int, default=4,
                        help='Epochs for fusion MAE training (default: 4)')
    parser.add_argument('--ssl_lr', type=float, default=1e-4,
                        help='Learning rate for fusion MAE training (default: 1e-4)')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.75,
                        help='Mask ratio for MAE training (default: 0.75)')
    parser.add_argument('--modality_dropout', type=float, default=0.2,
                        help='Probability of fully masking a modality (default: 0.2)')
    parser.add_argument('--wandb_project', type=str, default='shot-end-to-end')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--multimodal_eval', action='store_true')
    parser.add_argument('--monomodal_eval', action='store_true')
    parser.add_argument('--train_components', type=str, default='full', choices=['full','adaptor'])
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
        name=f"{starting_modality}=+{newmod}--{args.mae_mask_ratio}mask"
    )

    # Create datasets
    print("\n=== Creating datasets ===")

    _, train2_loader, test_loader = get_loaders(32,4)

    evan = model.evan
    model = model.to(device)
    print(f"SSL-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")

    num_newmod_channels = len(bands_newmod)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.create_modality_components(newmod,num_newmod_channels)
        model = model.to(device)  # Move newly created components to device
    
    # ========================================== TRAIN SHOT ===========================================
    if args.train_method=="shot":
        print(f"\n Using SHOT (MAE + Latent Distillation + Sequence Projection) training method for fusion blocks")
        _,_,intermediate_projectors,trainable_total=train_shot(
            model=model,
            train_loader=train2_loader,
            device=device,
            args=args,
            mae_modalities=[starting_modality,newmod],  # new modality reconstructs pixels
            latent_reconstruct_modalities=[starting_modality],
            modality_bands_dict=modality_bands_dict,
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
    for objective, info in objectives.items():
        print(f"\n--- Evaluating: {objective.capitalize()} objective ---")
        print(f"    {info['desc']}")

        accuracy = _delulu_stage3_test(
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
        print(f"{objective.capitalize()} accuracy: {accuracy:.2f}%")
        wandb.log({f"test/{objective}_accuracy": accuracy})

    # ========================================= CHECKPOINT =====================================
    timestamp_shot = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_shotete = os.path.join(args.checkpoint_dir, f'evan_eurosat_{args.train_method}_{timestamp_shot}.pt')
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
    filename = "res/shot_e2e_e2e.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "starting_modality","new_modality", "ssl_mode", "ssl_lr", "fusion_epochs",
        "mask_ratio", "modality_dropout","train_components","trainable_params",
        "transfer_acc", "peeking_acc", "addition_acc",
        "stage0_checkpoint", "shote2e_checkpoint"
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            starting_modality,
            newmod,
            args.train_method,
            args.ssl_lr,
            args.epochs,
            args.mae_mask_ratio,
            args.modality_dropout,
            args.train_components,
            trainable_total,
            f"{accuracies['transfer']:.2f}",
            f"{accuracies['peeking']:.2f}",
            f"{accuracies['addition']:.2f}",
            args.stage0_checkpoint,
            checkpoint_shotete,
        ])

    print(f"\nResults appended to {filename}")
    wandb.finish()
    return

if __name__ == '__main__':
    main()

"""
python -u shot_ete.py \
    --checkpoint_name vre_to_rgb \
    --stage0_checkpoint checkpoints/vre_fft.pt  \
    --epochs 4 \
    --train_components full \
    --modality_dropout 0.4 \
    --mae_mask_ratio 0.85 \
    --new_mod_group rgb
"""