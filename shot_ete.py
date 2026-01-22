
from shot import train_shot
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchgeo.datasets import EuroSAT
from torchvision import transforms
import logging
import os
import argparse
import csv
from datetime import datetime
import wandb

from evan_main import EVANClassifier
from eurosat_data_utils import (
    DictTransform,
    ALL_BAND_NAMES,
    get_modality_bands_dict
)
from train_utils import (
    evaluate,
    load_split_indices,
    single_modality_training_loop,
    supervised_training_loop,
    train_mae_fusion_phase
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='End to end training for SHOT model.')
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to stage 0 checkpoint (required)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')
    parser.add_argument('--new_mod_group', type=str, default='vre', choices=['vre', 'nir', 'swir'],
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
    parser.add_argument('--num_supervised_epochs', type=int, default=2)
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
    modality_bands_dict = get_modality_bands_dict('rgb', newmod)
    bands_newmod = modality_bands_dict[newmod] # list of band number
    bands_full = tuple(ALL_BAND_NAMES) # tuple of band number

    wandb.init(
        project=args.wandb_project,
        config={**config, **vars(args)},
        name=f"{starting_modality}=+{newmod}--{args.mae_mask_ratio}mask"
    )

    # Create datasets
    print("\n=== Creating datasets ===")

    resize_transform = DictTransform(transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True))

    train_dataset_full = EuroSAT(
        root='datasets',
        split='train',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    train1_indices = load_split_indices('datasets/eurosat-train1.txt', train_dataset_full)
    train1_dataset = Subset(train_dataset_full, train1_indices)
    train2_indices = load_split_indices('datasets/eurosat-train2.txt', train_dataset_full)
    train2_dataset = Subset(train_dataset_full, train2_indices)

    test_dataset_full = EuroSAT(
        root='datasets',
        split='test',
        bands=bands_full,
        transforms=resize_transform,
        download=True,
        checksum=False
    )

    print(f"Loaded {len(train1_indices)} and {len(train2_indices)} samples from train1 and train2 splits.")
    print(f"Test samples: {len(test_dataset_full)}")

    # Create dataloaders
    train1_loader = DataLoader(train1_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    train2_loader = DataLoader(train2_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset_full, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    evan = model.evan
    # model = EVANClassifier(evan, num_classes=config['num_classes'], classifier_strategy='mean', device=device)
    model = model.to(device)
    
    # Load state dict from checkpoint - this loads the MAE-trained components
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"Loaded model weights from checkpoint {args.stage0_checkpoint}")
    print(f"SSL-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")

    num_newmod_channels = len(bands_newmod)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.create_modality_components(newmod,num_newmod_channels)
        model = model.to(device)  # Move newly created components to device
    
    # ========================================== TRAIN SHOT ===========================================
    if args.train_method=="shot":
        print(f"\n Using SHOT (MAE + Latent Distillation + CLS Projection) training method for fusion blocks")
        _,_,_,cls_projectors,trainable_total=train_shot(
            model=model,
            train_loader=train2_loader,
            device=device,
            args=args,
            mae_modalities=[newmod],  # new modality reconstructs pixels
            latent_reconstruct_modalities=['rgb'],  # rgb matches teacher latents
            modality_bands_dict=modality_bands_dict,
        )
        
    # ========================================= CHECKPOINT =====================================
    timestamp_shot = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_shotete = os.path.join(args.checkpoint_dir, f'evan_eurosat_{args.train_method}_{timestamp_shot}.pt')
    if args.checkpoint_name:
        checkpoint_shotete = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    # Save model checkpoint with cls_projectors included
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
        'cls_projectors_state_dict': cls_projectors.state_dict() if cls_projectors is not None else None,
    }
    torch.save(checkpoint_data, checkpoint_shotete)
    print(f"SHOT checkpoint saved to: {checkpoint_shotete} (includes cls_projectors)")

    # ================================================ EVALUATION =============================================
    eval_lr=1e-3
    model.freeze_all()
    model.set_requires_grad('all', classifier=True)
    criterion = nn.CrossEntropyLoss()
    # ========================================= MULTIMODALITY Evaluations =====================================
    print("\n" + "="*70)
    print("=== Evaluating fusion with multimodal supervised probing ===")
    print("="*70)
    train_acc, test_acc_multi=-1,-1
    if args.multimodal_eval:
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)
        print(f"\n=== Training for {args.num_supervised_epochs} epochs ===")
        print(f"Train and evaluate on {starting_modality}+{newmod} (multimodal, train1 split)")
        model.switch_strategy("ensemble",starting_modality)
        train_acc, _, _, test_acc_multi = supervised_training_loop(
            model=model,
            train_loader=train1_loader,
            test_loader_full=test_loader,
            device=device,
            modality_bands_dict=modality_bands_dict,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.num_supervised_epochs,
            train_modalities=(starting_modality, newmod),
            newmod=newmod,
            phase_name="SHOT supervised eval",
            use_wandb=True,
            wandb_prefix="shot_eval",
            eval_single_modalities=False,  # Skip redundant single-modality evals during multimodal training
        )

        print(f"\n(supervised) Multimodal Eval Result:")
        print(f"  Train acc (RGB+{newmod}): {train_acc:.2f}%")
        print(f"  Test acc (RGB+{newmod}): {test_acc_multi:.2f}%")

    # ========== Single-modality evaluations under supervision ==========
    # Reload checkpoint to reset classifier before single-modality evaluation
    train_acc_rgb, test_acc_rgb_single, best_test_acc_rgb, best_epoch_rgb=-1,-1,-1,-1
    train_acc_newmod, test_acc_newmod, best_test_acc_newmod, best_epoch_newmod=-1,-1,-1,-1
    if args.monomodal_eval:
        saved_checkpoint = torch.load(checkpoint_shotete, map_location=device)
        model.load_state_dict(saved_checkpoint['model_state_dict'], strict=True)

        # RGB-only evaluation
        print("\n" + "-"*50)
        print(f"=== (Supervised) Single-modality evaluation: RGB ===")
        model.freeze_all()
        model.set_requires_grad('all', classifier=True)
        optimizer_rgb = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)

        train_acc_rgb, test_acc_rgb_single, best_test_acc_rgb, best_epoch_rgb = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer_rgb, args.num_supervised_epochs,
            modality='rgb', phase_name="SHOT RGB-only eval w/ hallucination", hallucinate_modality=True, pseudo_modalities=[newmod],cls_projectors=cls_projectors
        )
        print(f"RGB-only Result: {train_acc_rgb=:.2f} {test_acc_rgb_single=:.2f} {best_test_acc_rgb=:.2f} at epoch {best_epoch_rgb}")

        # Reload checkpoint to reset classifier before newmod evaluation
        model.load_state_dict(saved_checkpoint['model_state_dict'], strict=True)

        # Newmod-only evaluation
        print("\n" + "-"*50)
        print(f"=== Single-modality evaluation: {newmod} ===")
        model.freeze_all()
        model.set_requires_grad('all', classifier=True)
        optimizer_newmod = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)

        train_acc_newmod, test_acc_newmod, best_test_acc_newmod, best_epoch_newmod = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer_newmod, args.num_supervised_epochs,
            modality=newmod, phase_name=f"SHOT {newmod}-only eval w/ hallucination", hallucinate_modality=True, pseudo_modalities=[starting_modality],cls_projectors=cls_projectors
        )
        print(f"(With Hallucination) {newmod}-only Result: {train_acc_newmod=:.2f} {test_acc_newmod=:.2f} {best_test_acc_newmod=:.2f} at epoch {best_epoch_newmod}")

        model.load_state_dict(saved_checkpoint['model_state_dict'], strict=True)
        # RGB-only evaluation
        print("\n" + "-"*50)
        print(f"=== (Supervised) Single-modality evaluation: RGB ===")
        model.freeze_all()
        model.set_requires_grad('all', classifier=True)
        optimizer_rgb = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)

        train_acc_rgb, test_acc_rgb_single, best_test_acc_rgb, best_epoch_rgb = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer_rgb, args.num_supervised_epochs,
            modality='rgb', phase_name="SHOT RGB-only eval"
        )
        print(f"RGB-only Result: {train_acc_rgb=:.2f} {test_acc_rgb_single=:.2f} {best_test_acc_rgb=:.2f} at epoch {best_epoch_rgb}")

        # Reload checkpoint to reset classifier before newmod evaluation
        model.load_state_dict(saved_checkpoint['model_state_dict'], strict=True)

        # Newmod-only evaluation
        print("\n" + "-"*50)
        print(f"=== Single-modality evaluation: {newmod} ===")
        model.freeze_all()
        model.set_requires_grad('all', classifier=True)
        optimizer_newmod = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)

        train_acc_newmod, test_acc_newmod, best_test_acc_newmod, best_epoch_newmod = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer_newmod, args.num_supervised_epochs,
            modality=newmod, phase_name=f"SHOT {newmod}-only eval"
        )
        print(f"{newmod}-only Result: {train_acc_newmod=:.2f} {test_acc_newmod=:.2f} {best_test_acc_newmod=:.2f} at epoch {best_epoch_newmod}")

    # Log results to CSV
    filename = "res/shot_e2e_res.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "starting_modality","new_modality", "ssl_mode", "ssl_lr", "fusion_epochs",
        "mask_ratio", "modality_dropout", "supervised_epochs","train_components","trainable_params",
        "test_acc_multi", "multimodal_gain",
        "test_acc_rgb_single", "best_test_acc_rgb",
        "test_acc_newmod_single", "best_test_acc_newmod",
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
            args.num_supervised_epochs,
            args.train_components,
            trainable_total,
            f"{test_acc_multi:.2f}",
            f"{test_acc_multi - test_acc_rgb_single:+.2f}",
            f"{test_acc_rgb_single:.2f}",
            f"{best_test_acc_rgb:.2f}",
            f"{test_acc_newmod:.2f}",
            f"{best_test_acc_newmod:.2f}",
            args.stage0_checkpoint,
            checkpoint_shotete,
        ])

    print(f"\nResults appended to {filename}")
    wandb.finish()
    return

if __name__ == '__main__':
    main()

# python -u shot_ete.py --stage0_checkpoint checkpoints/evan_eurosat_stage0_rgb_20260117_034834.pt --monomodal_eval --multimodal_eval

# checkpoints/evan_eurosat_stage0_rgb_20260121_012101.pt
# python -u shot_ete.py --stage0_checkpoint checkpoints/evan_eurosat_stage0_rgb_20260121_012101.pt --monomodal_eval --multimodal_eval
# python -u shot_ete.py --stage0_checkpoint checkpoints/evan_eurosat_stage0_rgb_20260121_012101.pt --monomodal_eval --multimodal_eval --epochs 1 --num_supervised_epochs 1