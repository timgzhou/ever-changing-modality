
from shot import train_shot
from data_utils import get_loaders
import re
import torch
import logging
import os
import argparse
import csv
from datetime import datetime
import wandb

from evan_main import EVANClassifier, EvanSegmenter
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

VALID_NEW_MODS = {
    'eurosat': ['vre', 'nir', 'swir', 'rgb'],
    'benv2':   ['s1', 's2'],
    'pastis':  ['s1', 's2'],
    'dfc2020': ['s1', 's2'],
}


def main():
    parser = argparse.ArgumentParser(description='End to end training for SHOT model.')
    # IMPORTANT
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'],
                        help='Dataset to train on (default: eurosat)')
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to stage 0 checkpoint (required)')
    parser.add_argument('--new_mod_group', type=str, required=True,
                        help='New modality to add (eurosat: vre/nir/swir/rgb; benv2/pastis/dfc2020: s1/s2)')
    parser.add_argument('--mae_mask_ratio', type=float, default=0.75,
                        help='Mask ratio for MAE training (default: 0.75)')
    parser.add_argument('--modality_dropout', type=float, default=0.3,
                        help='Probability of fully masking a modality')
    parser.add_argument('--mae_modalities', type=str, default="all", choices=["all","newmod"])
    parser.add_argument('--labeled_frequency', type=float, default=0.3,
                        help='Frequency of labeled monomodal batches from train1 (0-1, default: 0.3)')
    parser.add_argument('--labeled_start_fraction', type=float, default=0.0,
                        help='Fraction of training before labeled mixing starts (0=start, 0.5=halfway, 1=never)')
    parser.add_argument('--active_losses', type=str, nargs='+', default=None,
                        choices=['mae', 'latent', 'prefusion', 'distill', 'ce'],
                        help='Which losses to activate (default: all)')
    parser.add_argument('--lambda_mae', type=float, default=1.0, help='Weight for MAE loss (default: 1.0)')
    parser.add_argument('--lambda_latent', type=float, default=1.0, help='Weight for latent loss (default: 1.0)')
    parser.add_argument('--lambda_prefusion', type=float, default=1.0, help='Weight for prefusion loss (default: 1.0)')
    parser.add_argument('--lambda_distill', type=float, default=1.0, help='Weight for distillation loss (default: 1.0)')
    parser.add_argument('--lambda_ce', type=float, default=1.0, help='Weight for CE loss (default: 1.0)')
    parser.add_argument('--use_mfla', action='store_true',
                        help='Enable MFLA training for hallucinated modalities')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Linear LR warmup epochs before cosine decay (default: 1)')

    parser.add_argument('--results_csv', type=str, required=True,
                        help='Path to results CSV file')
    parser.add_argument('--intermediate_projector_type', type=str, default='cross',
                        choices=['self', 'cross'],
                        help='Type of intermediate projector: self-attention (self) or cross-attention (cross)')
    parser.add_argument('--intermediate_projector_num_layers', type=int, default=2,
                        help='Number of layers in intermediate projector (default: 2)')

    parser.add_argument('--tz_fusion_time', type=int, default=3,
                        help='Number of modality-specific blocks before fusion (0=no MSLA, default: 3)')

    # UNIMPORTANT
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--eval_every_n_epochs', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--wandb_project', type=str, default='delulu-reBEN')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    args = parser.parse_args()

    # Validate new_mod_group against dataset
    valid_new_mods = VALID_NEW_MODS[args.dataset]
    if args.new_mod_group not in valid_new_mods:
        parser.error(f"--new_mod_group {args.new_mod_group!r} is not valid for --dataset {args.dataset}. "
                     f"Valid choices: {valid_new_mods}")

    # Load stage 0 checkpoint — detect Panopticon LP checkpoint vs EVAN checkpoint
    print(f"\n=== Loading Stage 0 checkpoint from: {args.stage0_checkpoint} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    raw_ckpt = torch.load(args.stage0_checkpoint, map_location='cpu')
    is_panopticon = not isinstance(raw_ckpt, dict) or 'config' not in raw_ckpt

    if is_panopticon:
        m = re.search(r'panopticon_lp_benv2_(\w+)_s0', args.stage0_checkpoint)
        starting_modality = m.group(1) if m else args.new_mod_group
        config = {'normalization': 'zscore', 'evan_config': {'starting_modality': starting_modality}}
        print(f"  Detected Panopticon LP checkpoint — starting modality: {starting_modality}")
        print(f"\nUsing device: {device}")
    else:
        checkpoint = raw_ckpt
        config = checkpoint['config']
        if 'decoder_strategy' in config:
            model = EvanSegmenter.from_checkpoint(args.stage0_checkpoint, device)
        else:
            model = EVANClassifier.from_checkpoint(args.stage0_checkpoint, device)
        evan_config = config['evan_config']
        starting_modality = evan_config['starting_modality']
        print(f"Stage 0 config:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"\nUsing device: {device}")

    newmod = args.new_mod_group

    wandb.init(
        project=args.wandb_project,
        config={**config, **vars(args)},
        name=f"{args.dataset}_{starting_modality}+={newmod}"
    )

    # Create datasets — use the same normalizer that was used during stage 0 training
    print("\n=== Creating datasets ===")
    data_normalizer = None
    normalization = config.get('normalization', 'zscore')
    if normalization == 'div10000':
        from geobench_data_utils import make_div10000_normalizer
        data_normalizer = make_div10000_normalizer()
        print(f"Using div10000 normalizer (from stage0 checkpoint config)")
    train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config = \
        get_loaders(args.dataset, starting_modality, args.batch_size, args.num_workers,
                    data_normalizer=data_normalizer, new_modality=newmod)

    modality_bands_dict = task_config.modality_bands_dict
    bands_newmod = modality_bands_dict[newmod]

    if is_panopticon:
        # Build Panopticon teacher (frozen backbone + trained LP head)
        from lp_panopticon_benv2 import PanopticonTeacher
        panopticon_teacher = PanopticonTeacher.from_checkpoint(
            head_state_path=args.stage0_checkpoint,
            modality=starting_modality,
            num_classes=task_config.num_classes,
            device=device,
        )
        # Build random-init EVAN student (no pretrained weights)
        from evan_main import evan_base
        evan_student = evan_base(
            load_weights=False,
            starting_modality=starting_modality,
            starting_n_chans=task_config.modality_a_channels,
            img_size=task_config.img_size,
            n_storage_tokens=4,
            tz_fusion_time=args.tz_fusion_time,
            device=device,
        )
        model = EVANClassifier(
            evan_student,
            num_classes=task_config.num_classes,
            classifier_strategy='ensemble',
            device=device,
        ).to(device)
        evan = model.evan
        # Dims for latent projector cross-architecture matching
        teacher_latent_dim = 768
        teacher_n_patches = (task_config.img_size // 14) ** 2
        print(f"Panopticon teacher loaded. Random-init EVAN student created.")
        print(f"  teacher_latent_dim={teacher_latent_dim}, teacher_n_patches={teacher_n_patches}")
    else:
        panopticon_teacher = None
        teacher_latent_dim = None
        teacher_n_patches = None
        evan = model.evan
        model = model.to(device)
        print(f"SSL-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")

    num_newmod_channels = len(bands_newmod) if not isinstance(bands_newmod, slice) else \
        (bands_newmod.stop - bands_newmod.start)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.intermediate_projector_type = args.intermediate_projector_type
        evan.intermediate_projector_num_layers = args.intermediate_projector_num_layers
        if args.intermediate_projector_type == "cross" and not hasattr(evan, 'projector_queries'):
            evan.projector_queries = torch.nn.ParameterDict()
        evan.create_modality_components(newmod, num_newmod_channels)
        model = model.to(device)

    # ========================================== TRAIN SHOT ===========================================
    print(f"\n Using SHOT (MAE + Latent Distillation + Sequence Projection) training method for fusion blocks")
    match args.mae_modalities:
        case "all":
            print(f"requiring mae from all modalities.")
            mae_modalities = [starting_modality, newmod]
        case "newmod":
            print(f"requiring mae from newmod only.")
            mae_modalities = [newmod]

    trainable_total, _, best_checkpoint_summary, teacher_baselines = train_shot(
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
        loss_weights={
            'mae': args.lambda_mae, 'latent': args.lambda_latent,
            'prefusion': args.lambda_prefusion, 'distill': args.lambda_distill,
            'ce': args.lambda_ce,
        },
        weight_decay=args.weight_decay,
        val_unlabeled_loader=val2_loader,
        val_labeled_loader=val1_loader,
        use_mfla=args.use_mfla,
        warmup_epochs=args.warmup_epochs,
        multilabel=task_config.multilabel,
        label_key=task_config.label_key,
        segmentation=(task_config.task_type == 'segmentation'),
        num_classes=task_config.num_classes,
        ignore_index=getattr(task_config, 'ignore_index', -100),

        teacher_classifier=panopticon_teacher,
        teacher_latent_dim=teacher_latent_dim,
        teacher_n_patches=teacher_n_patches,
    )

    if task_config.task_type == 'segmentation':
        metric_name = "mIoU"
    elif task_config.multilabel:
        metric_name = "mAP"
    else:
        metric_name = "Acc"

    # Log teacher baselines
    if teacher_baselines:
        print(f"\n=== Teacher Baselines ({metric_name}, starting modality only) ===")
        for set_name, acc in teacher_baselines.items():
            print(f"  {set_name}: {acc:.2f}%")
        wandb.log({f"teacher_baseline/{k.replace('(','').replace(')','').replace(' ','_')}": v
                   for k, v in teacher_baselines.items()})

    # Metrics already computed during training at best val epoch — use those directly
    best = best_checkpoint_summary.get('best_ens_addition', best_checkpoint_summary.get('best_addition', {}))
    metrics = {
        'transfer': best.get('test_transfer', 0.0),
        'peeking':  best.get('test_peeking', 0.0),
        'addition': best.get('test_addition', 0.0),
    }
    addition_ens_metric = best.get('test_addition_ens', None)

    # ========================================= CHECKPOINT =====================================
    timestamp_shot = datetime.now().strftime('%m%d_%H%M')
    checkpoint_shotete = os.path.join(args.checkpoint_dir, f'delulunet_{args.dataset}_{timestamp_shot}.pt')
    if args.checkpoint_name:
        checkpoint_shotete = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    checkpoint_data = {
        'model_state_dict': model.state_dict(),
        'config': model.get_config(),
    }
    torch.save(checkpoint_data, checkpoint_shotete)
    print(f"SHOT checkpoint saved to: {checkpoint_shotete}")

    # Log results to CSV
    filename = args.results_csv
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "dataset", "starting_modality", "new_modality", "lr", "weight_decay", "epochs",
        "mask_ratio", "modality_dropout", "labeled_frequency", "labeled_start_fraction",
        "trainable_params", "active_losses", "use_mfla", "warmup_epochs", "intermediate_projector_type", "tz_fusion_time", "metric_name",
        "transfer_metric", "peeking_metric", "addition_metric", "addition_ens_metric",
        "valchecked_transfer", "valchecked_peek", "valchecked_add", "valchecked_add_ens",
        "stage0_checkpoint", "shote2e_checkpoint"
    ]

    def get_valchecked(ckpt_name, metric_key):
        if ckpt_name in best_checkpoint_summary:
            return f"{best_checkpoint_summary[ckpt_name][metric_key]:.2f}"
        return ""

    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        active_losses_str = "+".join(args.active_losses) if args.active_losses else "all"
        writer.writerow([
            args.dataset,
            starting_modality,
            newmod,
            args.lr,
            args.weight_decay,
            args.epochs,
            args.mae_mask_ratio,
            args.modality_dropout,
            args.labeled_frequency,
            args.labeled_start_fraction,
            trainable_total,
            active_losses_str,
            args.use_mfla,
            args.warmup_epochs,
            args.intermediate_projector_type,
            args.tz_fusion_time,
            metric_name,
            f"{metrics['transfer']:.2f}",
            f"{metrics['peeking']:.2f}",
            f"{metrics['addition']:.2f}",
            f"{addition_ens_metric:.2f}" if addition_ens_metric is not None else "",
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


# DRYRUN examples
"""
# BEN-v2 using Panopticon LP checkpoint as teacher (s2 teacher → add s1)
python -u shot_ete.py \
    --dataset benv2 \
    --new_mod_group s1 \
    --stage0_checkpoint checkpoints/panopticon_lp_benv2_s2_s0.pt \
    --epochs 2 \
    --eval_every_n_epochs 1 \
    --batch_size 32 \
    --results_csv res/shot_ete_dryrun.csv \
    --labeled_frequency 0.3 \
    --tz_fusion_time 0

# BEN-v2 using Panopticon LP checkpoint as teacher (s1 teacher → add s2)
python -u shot_ete.py \
    --dataset benv2 \
    --new_mod_group s2 \
    --stage0_checkpoint checkpoints/panopticon_lp_benv2_s1_s0.pt \
    --epochs 2 \
    --eval_every_n_epochs 1 \
    --batch_size 32 \
    --results_csv res/shot_ete_dryrun.csv \
    --active_losses distill ce latent \
    --labeled_frequency 0.3

# BEN-v2 (s2 -> add s1)
python -u shot_ete.py \
    --dataset benv2 \
    --new_mod_group s1 \
    --checkpoint_name benv2_s2_to_s1 \
    --stage0_checkpoint checkpoints/benv2_s2_s0.pt \
    --epochs 2 \
    --eval_every_n_epochs 1 \
    --batch_size 32 \
    --results_csv res/shot_ete_benv2.csv \
    --labeled_frequency 0.3
    
# BENv2 s2 to s1
python -u shot_ete.py \
    --dataset benv2 \
    --new_mod_group s1 \
    --checkpoint_name benv2_s2_to_s1_dryrun \
    --stage0_checkpoint checkpoints/benv2_s2_s0.pt \
    --epochs 4 \
    --eval_every_n_epochs 1 \
    --batch_size 64 \
    --results_csv res/shot_ete_dryrun.csv \
    --labeled_frequency 0.3
    
# PASTIS s2 to s1
python -u shot_ete.py \
    --dataset pastis \
    --new_mod_group s1 \
    --checkpoint_name pastis_s2_to_s1_dryrun \
    --stage0_checkpoint checkpoints/pastis_s2_s0.pt \
    --epochs 4 \
    --eval_every_n_epochs 1 \
    --batch_size 32 \
    --results_csv res/shot_ete_dryrun.csv \
    --labeled_frequency 0.3
"""
