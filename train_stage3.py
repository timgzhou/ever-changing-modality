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
    train_mae_fusion_phase,
    train_pseudo_supervised,
    train_self_distillation,
    hallucination_supervised,
)
from shot import create_int_cls_projectors

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser(description='stage 2: Train fusion LoRAs and classifier (after stage 2 MAE)')
    parser.add_argument('--stage2_checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4,)
    parser.add_argument('--stage3_train_method', type=str, default='hallucination', choices=['distill','pseudo','hallucination'])
    parser.add_argument('--rgb_teacher_checkpoint', type=str, default='checkpoints/evan_eurosat_stage0_rgb_20260118_134254.pt',
                        help='RGB teacher checkpoint for distillation mode')
    parser.add_argument('--temperature', type=float, default=3.0)
    parser.add_argument('--stage3_lr', type=float, default=1e-3)
    parser.add_argument('--stage3_epochs', type=int, default=1)
    parser.add_argument('--classifier_strategy', type=str, default="ensemble", choices=["ensemble","mean"])
    parser.add_argument('--objective', type=str, default="multimodal", choices=["monomodal","multimodal"])
    parser.add_argument('--wandb_project', type=str, default='evan-eurosat-stage3(predictor)',help='Wandb project name')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    print(f"\n=== Loading checkpoint from: {args.stage2_checkpoint} ===")
    checkpoint = torch.load(args.stage2_checkpoint, map_location='cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model=EVANClassifier.from_checkpoint(args.stage2_checkpoint,device)
    config = checkpoint['config']
    evan_config = config['evan_config']
    starting_modality=evan_config['starting_modality']

    print(f"config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Get newmod from config
    newmod = "vre"
    modality_bands_dict = get_modality_bands_dict('rgb', newmod)
    bands_newmod = modality_bands_dict[newmod]  # derive from modality_bands_dict

    # Initialize wandb if enabled
    wandb.init(
        project=args.wandb_project,
        config={**config, **vars(args)},
        name=f"{newmod}_{args.stage3_train_method}_{args.classifier_strategy}"
    )
    bands_full = tuple(ALL_BAND_NAMES)

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

    model = model.to(device)
    # Load state dict from checkpoint - this loads the MAE-trained components
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded model weights from stage 2 checkpoint")
    print(f"SSL-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")
    
    print(f"Untrained starting modal accuracy (this should be different from pretraining)")
    _, new_mod_test_acc = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device,
            modality_bands_dict, modalities_to_use=(newmod,)
        )
    _, rgb_test_acc = evaluate(
            model, test_loader, nn.CrossEntropyLoss(), device,
            modality_bands_dict, modalities_to_use=('rgb',)
        )
    print(f"  {newmod} test acc: {new_mod_test_acc} \n  rgb test acc: {rgb_test_acc}")
    if args.objective=="multimodal":
        model.switch_strategy(args.classifier_strategy,"rgb")
    # ========================================= TRAIN =====================================
    if args.stage3_train_method=="distill":
        print(f"\n Using self distillation to train new classifier.")

        # Load RGB teacher model from stage0 checkpoint
        print(f"\n=== Loading RGB teacher from: {args.rgb_teacher_checkpoint} ===")
        teacher_checkpoint = torch.load(args.rgb_teacher_checkpoint, map_location='cpu')
        teacher_config = teacher_checkpoint['config']

        teacher_model_fn = {'evan_small': evan_small, 'evan_base': evan_base, 'evan_large': evan_large}[teacher_config['model_type']]
        teacher_evan = teacher_model_fn(
            tz_fusion_time=teacher_config['tz_fusion_time'],
            tz_lora_rank=teacher_config['tz_lora_rank'],
            tz_modality_specific_layer_augmenter=teacher_config.get('tz_modality_specific_layer_augmenter', 'lora'),
            tz_modality_fusion_layer_augmenter=teacher_config.get('tz_modality_fusion_layer_augmenter', 'lora'),
            n_storage_tokens=teacher_config.get('n_storage_tokens', 4),
            device=device
        )
        teacher_model = EVANClassifier(teacher_evan, num_classes=teacher_config['num_classes'], classifier_strategy='mean', device=device)
        teacher_model = teacher_model.to(device)
        teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'], strict=False)
        print(f"Loaded RGB teacher model from stage0 checkpoint")

        test_acc = train_self_distillation(
            model=model,
            train_loader=train2_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            modality_bands_dict=modality_bands_dict,
            student_mod=newmod,
            teacher_mod="rgb",
            teacher_model=teacher_model,
        )

    if args.stage3_train_method=="pseudo":
        print(f"\n Using pseudo clstoken to train new classifier.")
        test_acc = train_pseudo_supervised(
            model=model,
            unlabeled_train_loader=train2_loader,
            labeled_train_loader=train1_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            modality_bands_dict=modality_bands_dict,
            student_mod=newmod,
            teacher_mod="rgb",
        )

    if args.stage3_train_method=="hallucination":
        print("Warning! hallucination mode should only work with ensemble strategy, switching you over.")
        model.switch_strategy("ensemble","rgb")
        model.freeze_all()
        model.set_requires_grad("all",classifier=True)
        print(f"\n Using hallucination supervised training with SHOT cls_projectors.")

        # Load intermediate_cls_projectors from SHOT checkpoint
        if 'cls_projectors_state_dict' not in checkpoint or checkpoint['cls_projectors_state_dict'] is None:
            raise ValueError(
                f"Checkpoint {args.stage2_checkpoint} does not contain cls_projectors_state_dict. "
                "This checkpoint was likely not created by shot_ete.py. "
                "Please use a SHOT checkpoint that includes the trained cls_projectors."
            )

        # Recreate the cls_projectors structure and load state dict
        all_modalities = [starting_modality, newmod]
        intermediate_cls_projectors = create_int_cls_projectors(
            hidden_dim=model.evan.embed_dim,
            all_modalities=all_modalities,
            device=device
        )
        intermediate_cls_projectors.load_state_dict(checkpoint['cls_projectors_state_dict'])
        print(f"Loaded SHOT-trained intermediate_cls_projectors: {list(intermediate_cls_projectors.keys())}")

        test_acc = hallucination_supervised(
            model=model,
            unlabeled_train_loader=train2_loader,
            labeled_train_loader=train1_loader,
            test_loader=test_loader,
            device=device,
            args=args,
            modality_bands_dict=modality_bands_dict,
            student_mod=newmod,
            teacher_mod="rgb",
            intermediate_cls_projectors=intermediate_cls_projectors,
        )

    # Log results to CSV
    filename = "res/train_stage3_res.csv"
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "timestamp", "new_modality", "stage3_train_method", "objective",
        "classifier_strategy", "temperature", "stage3_lr", "stage3_epochs",
        "test_acc", "stage2_checkpoint", "rgb_teacher_checkpoint"
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([
            datetime.now().strftime("%Y%m%d_%H%M%S"),
            newmod,
            args.stage3_train_method,
            args.objective,
            args.classifier_strategy,
            args.temperature,
            args.stage3_lr,
            args.stage3_epochs,
            f"{test_acc:.2f}",
            args.stage2_checkpoint,
            args.rgb_teacher_checkpoint if args.stage3_train_method == "distill" else "N/A",
        ])

    print(f"\nResults appended to {filename}")
    wandb.finish()


if __name__ == '__main__':
    main()

#  python -u train_stage3.py --stage2_checkpoint checkpoints/evan_eurosat_shot_20260121_035727.pt --objective monomodal --stage3_train_method hallucination