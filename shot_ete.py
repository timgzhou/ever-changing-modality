
from shot import train_shot
from data_utils import get_loaders
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
    'benv2':   ['s1', 's2', 's2_rgb', 's2_norgb'],
    'dfc2020': ['s1', 's2', 's2_rgb', 's2_norgb'],
}


def _parse_args():
    parser = argparse.ArgumentParser(description='End to end training for SHOT model.')
    # IMPORTANT
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['eurosat', 'benv2', 'dfc2020'],
                        help='Dataset to train on (default: eurosat)')
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to stage 0 checkpoint (required)')
    parser.add_argument('--new_mod_group', type=str, required=True,
                        help='New modality to add (eurosat: vre/nir/swir/rgb; benv2/pastis/dfc2020: s1/s2)')
    parser.add_argument('--token_mask_ratio', '--mae_mask_ratio', type=float, default=0.75,
                        help='Ratio of tokens masked per modality during training (default: 0.75)')
    parser.add_argument('--modality_dropout', type=float, default=0.3,
                        help='Probability of fully masking a modality')
    parser.add_argument('--labeled_frequency', type=float, default=0.3,
                        help='Frequency of labeled monomodal batches from train1 (0-1, default: 0.3)')
    parser.add_argument('--labeled_start_fraction', type=float, default=0.5,
                        help='Fraction of training before labeled mixing starts (0=start, 0.5=halfway, 1=never)')
    parser.add_argument('--active_losses', type=str, nargs='+', required=True,
                        choices=['latent', 'prefusion', 'distill', 'ce'],
                        help='Which losses to activate')
    parser.add_argument('--lambda_latent', type=float, default=1.0, help='Weight for latent loss (default: 1.0)')
    parser.add_argument('--lambda_prefusion', type=float, default=1.0, help='Weight for prefusion loss (default: 1.0)')
    parser.add_argument('--lambda_distill', type=float, default=1.0, help='Weight for distillation loss (default: 1.0)')
    parser.add_argument('--lambda_ce', type=float, default=1.0, help='Weight for CE loss (default: 1.0)')
    parser.add_argument('--use_mask_token', action='store_true',
                        help='Ablation: replace intermediate projectors with broadcast learned mask token '
                             '(projector_queries). Incompatible with --active_losses prefusion.')
    parser.add_argument('--protect_lrm', action='store_true',
                        help='Detach LRM modality features in prefusion loss so its encoder is not updated '
                             'to be easier to predict.')
    parser.add_argument('--latent_masked_only', action='store_true',
                        help='Only compute latent loss on masked patch positions (not unmasked ones).')
    parser.add_argument('--unprotect_starting_mod', action='store_true',
                        help='Do not protect new modalities from full dropout on unlabeled data even when '
                             'batch mixing is active. By default (False), new modalities are protected '
                             'from full dropout when effective_labeled_freq > 0.')
    parser.add_argument('--dyn_teacher', action='store_true',
                        help='Dynamic teacher distillation: starting_modality head trains against '
                             'student peeking (soft-vote), newmod heads train against frozen unimodal teacher')
    parser.add_argument('--warmup_epochs', type=int, default=3,
                        help='Linear LR warmup epochs before cosine decay (default: 3)')

    parser.add_argument('--results_csv', type=str, required=True,
                        help='Path to results CSV file')
    parser.add_argument('--select_by', type=str, default=None,
                        help='HP selection criterion used to pick this config (transfer/peeking/addition). '
                             'Informational only — logged to CSV if provided.')
    parser.add_argument('--intermediate_projector_num_layers', type=int, default=2,
                        help='Number of layers in intermediate projector (default: 2)')

    parser.add_argument('--tz_fusion_time', type=int, default=3,
                        help='Number of modality-specific blocks before fusion (0=no MSLA, default: 3)')

    # UNIMPORTANT
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--eval_every_n_epochs', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--asym_lr', type=float, default=None,
                        help='LR multiplier for new components (intermediate_projectors, '
                             'latent_projectors, new-modality patch_embedder/MSLA/CLS/storage/head). '
                             'If None (default), all params share --lr.')
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--wandb_project', type=str, default='delulu-apr21')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='Save final model checkpoint to --checkpoint_dir')
    args = parser.parse_args()

    if args.dyn_teacher:
        parser.error("--dyn_teacher is disabled; remove it from your sweep config or job script.")

    # Validate new_mod_group against dataset
    valid_new_mods = VALID_NEW_MODS[args.dataset]
    if args.new_mod_group not in valid_new_mods:
        parser.error(f"--new_mod_group {args.new_mod_group!r} is not valid for --dataset {args.dataset}. "
                     f"Valid choices: {valid_new_mods}")

    return args


def main(args=None):
    if args is None:
        args = _parse_args()

    print(f"\n=== Loading Stage 0 checkpoint from: {args.stage0_checkpoint} ===")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.stage0_checkpoint, map_location='cpu')
    config = checkpoint['config']
    if 'decoder_strategy' in config:
        model = EvanSegmenter.from_checkpoint(args.stage0_checkpoint, device)
    else:
        model = EVANClassifier.from_checkpoint(args.stage0_checkpoint, device)
    starting_modality = config['evan_config']['starting_modality']
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
    train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config = \
        get_loaders(args.dataset, starting_modality, args.batch_size, args.num_workers,
                    data_normalizer=data_normalizer, new_modality=newmod)

    modality_bands_dict = task_config.modality_bands_dict
    bands_newmod = modality_bands_dict[newmod]

    evan = model.evan

    checkpoint_ft = evan.tz_fusion_time
    if args.tz_fusion_time != checkpoint_ft:
        print(f"  Overriding tz_fusion_time: {checkpoint_ft} → {args.tz_fusion_time}")
        evan.rewire_fusion_time(args.tz_fusion_time)

    model = model.to(device)

    num_newmod_channels = len(bands_newmod) if not isinstance(bands_newmod, slice) else \
        (bands_newmod.stop - bands_newmod.start)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
        evan.intermediate_projector_type = "cross"
        evan.intermediate_projector_num_layers = args.intermediate_projector_num_layers
        if not hasattr(evan, 'projector_queries'):
            evan.projector_queries = torch.nn.ParameterDict()
        evan.create_modality_components(newmod, num_newmod_channels)
        model = model.to(device)

    # ========================================== TRAIN Delulu ===========================================
    trainable_total, best_checkpoints, best_checkpoint_summary, teacher_baselines = train_shot(
        model=model,
        train_loader=train2_loader,
        device=device,
        args=args,
        starting_modality=starting_modality,
        new_modality=newmod,
        latent_reconstruct_modalities=[starting_modality],
        modality_bands_dict=modality_bands_dict,
        test_loader=test_loader,
        eval_every_n_epochs=args.eval_every_n_epochs,
        labeled_train_loader=train1_loader,
        labeled_frequency=args.labeled_frequency,
        labeled_start_fraction=args.labeled_start_fraction,
        active_losses=args.active_losses,
        loss_weights={
            'latent': args.lambda_latent,
            'prefusion': args.lambda_prefusion, 'distill': args.lambda_distill,
            'ce': args.lambda_ce,
        },
        weight_decay=args.weight_decay,
        val_unlabeled_loader=val2_loader,
        val_labeled_loader=val1_loader,
        warmup_epochs=args.warmup_epochs,
        asym_lr_multiplier=args.asym_lr,
        dyn_teacher=args.dyn_teacher,
        use_mask_token=args.use_mask_token,
        protect_lrm=args.protect_lrm,
        latent_masked_only=args.latent_masked_only,
        unprotect_starting_mod=args.unprotect_starting_mod,
        task_type=task_config.task_type,
        label_key=task_config.label_key,
        num_classes=task_config.num_classes,
        ignore_index=getattr(task_config, 'ignore_index', -100),
    )

    # Log teacher baselines
    if teacher_baselines:
        wandb.log({f"teacher_baseline/{k.replace('(','').replace(')','').replace(' ','_')}": v
                   for k, v in teacher_baselines.items()})

    teacher_test_metric = teacher_baselines.get("test", None)
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
    if args.save_checkpoint:
        timestamp_shot = datetime.now().strftime('%m%d_%H%M')
        ckpt_path = os.path.join(args.checkpoint_dir, f'delulunet_{args.dataset}_{timestamp_shot}.pt')
        torch.save({'model_state_dict': model.state_dict(), 'config': model.get_config()}, ckpt_path)
        print(f"Checkpoint saved to: {ckpt_path}")

    # ========================================= CSV LOGGING =====================================
    def get_ckpt_data(ckpt_name, test_key):
        ckpt = best_checkpoints.get(ckpt_name, {})
        val_metric = ckpt.get('metric')
        test_accs = ckpt.get('test_accs')
        test_acc = test_accs.get(test_key) if test_accs else None
        return val_metric, test_acc

    val_transfer, test_transfer = get_ckpt_data('best_transfer', 'transfer')
    val_peeking, test_peeking   = get_ckpt_data('best_peeking',  'peeking')
    val_addition, test_addition = get_ckpt_data('best_addition', 'addition')
    val_ens_addition, test_ens_addition = get_ckpt_data('best_ens_addition', 'ens')

    filename = args.results_csv
    if d := os.path.dirname(filename):
        os.makedirs(d, exist_ok=True)
    file_exists = os.path.isfile(filename)
    fieldnames = [
        "wandb_run_id", "wandb_project", "dataset", "starting_modality", "new_modality",
        "teacher_test_metric",
        "lr", "asym_lr", "weight_decay", "epochs",
        "modality_dropout", "labeled_frequency", "labeled_start_fraction",
        "protect_lrm", "use_mask_token", "latent_masked_only", "unprotect_starting_mod",
        "lambda_latent", "lambda_prefusion", "lambda_distill", "mae_mask_ratio",
        "trainable_params", "active_losses", "select_by",
        "val_transfer", "test_transfer",
        "val_peeking", "test_peeking",
        "val_addition", "test_addition",
        "val_ens_addition", "test_ens_addition",
        "stage0_checkpoint",
    ]
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(fieldnames)
        active_losses_str = "+".join(args.active_losses) if args.active_losses else "all"
        writer.writerow([
            wandb.run.id,
            wandb.run.project,
            args.dataset,
            starting_modality,
            newmod,
            f"{teacher_test_metric:.2f}" if teacher_test_metric is not None else "",
            args.lr,
            args.asym_lr,
            args.weight_decay,
            args.epochs,
            args.modality_dropout,
            args.labeled_frequency,
            args.labeled_start_fraction,
            args.protect_lrm,
            args.use_mask_token,
            args.latent_masked_only,
            args.unprotect_starting_mod,
            args.lambda_latent,
            args.lambda_prefusion,
            args.lambda_distill,
            args.token_mask_ratio,
            trainable_total,
            active_losses_str,
            args.select_by or "",
            f"{val_transfer:.2f}" if val_transfer is not None else "",
            f"{test_transfer:.2f}" if test_transfer is not None else "",
            f"{val_peeking:.2f}" if val_peeking is not None else "",
            f"{test_peeking:.2f}" if test_peeking is not None else "",
            f"{val_addition:.2f}" if val_addition is not None else "",
            f"{test_addition:.2f}" if test_addition is not None else "",
            f"{val_ens_addition:.2f}" if val_ens_addition is not None else "",
            f"{test_ens_addition:.2f}" if test_ens_addition is not None else "",
            args.stage0_checkpoint,
        ])

    print(f"\nResults appended to {filename}")
    wandb.finish()


if __name__ == '__main__':
    main()


# DRYRUN examples
"""
# BEN-v2 (s2 -> add s1)
python -u shot_ete.py \
    --dataset benv2 \
    --new_mod_group s1 \
    --stage0_checkpoint checkpoints/sft_evan_base_benv2_s2_fft_lr0.001_20260418_112953.pt \
    --epochs 16 \
    --eval_every_n_epochs 2 \
    --batch_size 32 \
    --results_csv res/shot_ete_benv2.csv \
    --active_losses latent prefusion distill ce \
    --labeled_frequency 0.3 \
    --latent_masked_only \
    --lambda_latent 0.1 \
    --labeled_start_fraction 0
    
python -u shot_ete.py \
    --dataset benv2 \
    --new_mod_group s2 \
    --stage0_checkpoint checkpoints/sft_evan_base_benv2_s1_fft_lr0.0005_20260418_064233.pt \
    --lr 0.0004 \
    --weight_decay 0.0025 \
    --labeled_frequency 0.015 \
    --epochs 32 \
    --eval_every_n_epochs 2 \
    --batch_size 32 \
    --results_csv res/shot_ete_benv2.csv \
    --active_losses latent prefusion distill ce \
    --modality_dropout 0.3 \
    --labeled_frequency 0.17 \
    --latent_masked_only \
    --labeled_start_fraction 0 \
    --lambda_latent 0.1267 \
    --lambda_prefusion 0.37 \
    --lambda_distill 0.73 \
    --mae_mask_ratio 0.413 \
    --unprotect_starting_mod
"""
