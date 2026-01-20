
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

from evan_main import evan_small, evan_base, evan_large, EVANClassifier
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
    parser = argparse.ArgumentParser(description='stage 2: Train fusion LoRAs and classifier (after stage 1 MAE)')
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
    parser.add_argument('--wandb_project', type=str, default='shot-end-to-end')
    parser.add_argument('--num_supervised_epochs', type=int, default=2)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--multimodal_eval', action='store_true')
    parser.add_argument('--monomodal_eval', action='store_true')
    args = parser.parse_args()

    # Load stage 1 checkpoint
    print(f"\n=== Loading Stage 0 checkpoint from: {args.stage0_checkpoint} ===")
    checkpoint = torch.load(args.stage0_checkpoint, map_location='cpu')
    config = checkpoint['config']

    print(f"Stage 0 config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    newmod = args.new_mod_group
    modality_bands_dict = get_modality_bands_dict('rgb', newmod)
    bands_newmod = modality_bands_dict[newmod] # list of band number
    bands_full = tuple(ALL_BAND_NAMES) # tuple of band number

    wandb.init(
        project=args.wandb_project,
        config={**config, **vars(args)},
        name=f"{config['model_type']}_{newmod}_{args.train_method}"
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

    # Recreate EVAN model with same config
    print("\n=== Recreating EVAN model ===")
    model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[config['model_type']]
    evan = model_fn(
        tz_fusion_time=config['tz_fusion_time'],
        tz_lora_rank=config['tz_lora_rank'],
        tz_modality_specific_layer_augmenter=config.get('tz_modality_specific_layer_augmenter', 'lora'),
        tz_modality_fusion_layer_augmenter=config.get('tz_modality_fusion_layer_augmenter', 'lora'),
        n_storage_tokens=config.get('n_storage_tokens', 4),
        device=device
    )

    # Create classifier
        
    model = EVANClassifier(evan, num_classes=config['num_classes'], classifier_strategy='mean', device=device)
    model = model.to(device)
    
    # Load state dict from checkpoint - this loads the MAE-trained components
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print(f"Loaded model weights from stage 1 checkpoint")
    print(f"SSL-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")

    num_newmod_channels = len(bands_newmod)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.create_modality_components(newmod,num_newmod_channels)
        
    _, new_mod_test_acc = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device,
            modality_bands_dict, modalities_to_use=(newmod,)
        )
    _, rgb_test_acc = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device,
            modality_bands_dict, modalities_to_use=('rgb',)
        )
    
    print(f"  {newmod} test acc: {new_mod_test_acc} \n  rgb test acc: {rgb_test_acc}")
    

    # ========================================== SHOT ===========================================
    if args.train_method=="shot":
        print(f"\n Using SHOT (a mix of MAE, Distillation, Contrastive) training method for fusion blocks")
        _,_,_,cls_projectors=train_shot(
            model=model,
            train_loader=train2_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            mae_modalities=[newmod],  # new modality reconstructs pixels
            latent_reconstruct_modalities=['rgb'],  # rgb matches teacher latents
            modality_bands_dict=modality_bands_dict,
        )
        
    # ========================================= CHECKPOINT =====================================
    timestamp_shot = datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_shotete = os.path.join(args.checkpoint_dir, f'evan_eurosat_{args.train_method}_{timestamp_shot}.pt')

    checkpoint_shot = {
        'model_state_dict': model.state_dict(),
        'stage1_checkpoint': args.stage0_checkpoint,
        'config': {
            **config,
            'train_split': 'train2',
            'train_method': args.train_method,
            'ssl_lr': args.ssl_lr,
            'epochs': args.epochs,
            'mae_mask_ratio': args.mae_mask_ratio,
            'newmod': newmod
        }
    }
    torch.save(checkpoint_shot, checkpoint_shotete)
    print(f"\n=== Stage 2 checkpoint saved to: {checkpoint_shotete} ===")

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
        print(f"Strategy: Train and evaluate on RGB+{newmod} (multimodal, train2 split)")
        train_acc, _, _, test_acc_multi = supervised_training_loop(
            model=model,
            train_loader=train1_loader,
            test_loader_full=test_loader,
            device=device,
            modality_bands_dict=modality_bands_dict,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=args.num_supervised_epochs,
            train_modalities=('rgb', newmod),
            newmod=newmod,
            phase_name="Stage 2 supervised evaluation",
            use_wandb=True,
            wandb_prefix="shot_eval",
            eval_single_modalities=False,  # Skip redundant single-modality evals during multimodal training
        )

        print(f"\nMultimodal Eval Result:")
        print(f"  Train acc (RGB+{newmod}): {train_acc:.2f}%")
        print(f"  Test acc (RGB+{newmod}): {test_acc_multi:.2f}%")

    # ========== Single-modality evaluations (like stage 1) ==========
    # Reload checkpoint to reset classifier before single-modality evaluation
    train_acc_rgb, test_acc_rgb_single, best_test_acc_rgb, best_epoch_rgb=-1,-1,-1,-1
    train_acc_newmod, test_acc_newmod, best_test_acc_newmod, best_epoch_newmod=-1,-1,-1,-1
    if args.monomodal_eval:
        model.load_state_dict(checkpoint_shot['model_state_dict'], strict=True)

        # RGB-only evaluation
        print("\n" + "-"*50)
        print(f"=== Single-modality evaluation: RGB ===")
        model.freeze_all()
        model.set_requires_grad('all', classifier=True)
        optimizer_rgb = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)

        train_acc_rgb, test_acc_rgb_single, best_test_acc_rgb, best_epoch_rgb = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer_rgb, args.num_supervised_epochs,
            modality='rgb', phase_name="SHOT RGB-only eval", hallucinate_modality=True, pseudo_modalities=[newmod],cls_projectors=cls_projectors
        )
        print(f"RGB-only Result: {train_acc_rgb=:.2f} {test_acc_rgb_single=:.2f} {best_test_acc_rgb=:.2f} at epoch {best_epoch_rgb}")

        # Reload checkpoint to reset classifier before newmod evaluation
        model.load_state_dict(checkpoint_shot['model_state_dict'], strict=True)

        # Newmod-only evaluation
        print("\n" + "-"*50)
        print(f"=== Single-modality evaluation: {newmod} ===")
        model.freeze_all()
        model.set_requires_grad('all', classifier=True)
        optimizer_newmod = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)

        train_acc_newmod, test_acc_newmod, best_test_acc_newmod, best_epoch_newmod = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer_newmod, args.num_supervised_epochs,
            modality=newmod, phase_name=f"SHOT newmod-o eval"
        )
        print(f"{newmod}-only Result: {train_acc_newmod=:.2f} {test_acc_newmod=:.2f} {best_test_acc_newmod=:.2f} at epoch {best_epoch_newmod}")

    # Log results to CSV
    filename = "shot_e2e_res.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "model_type", "new_modality", "ssl_mode", "ssl_lr", "fusion_epochs",
        "mask_ratio", "supervised_epochs",
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
            config['model_type'],
            newmod,
            args.train_method,
            args.ssl_lr,
            args.epochs,
            args.mae_mask_ratio,
            args.num_supervised_epochs,
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

# python -u shot_ete.py --stage0_checkpoint checkpoints/evan_eurosat_stage0_rgb_20260117_034834.pt --monomodal_eval