
from shot import train_shot
import torch
import logging
import os
import argparse
import csv
from datetime import datetime
import wandb

from evan_main import EVANClassifier
from train_utils import _delulu_stage3_test
logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

VALID_NEW_MODS = {
    'eurosat': ['vre', 'nir', 'swir', 'rgb'],
    'benv2':   ['s1', 's2'],
    'pastis':  ['s1', 's2'],
}


def get_loaders_and_config(dataset, batch_size, num_workers):
    """Return (train1, val1, train2, val2, test, task_config, modality_bands_dict_full)."""
    if dataset == 'eurosat':
        from eurosat_data_utils import ALL_BAND_NAMES, get_loaders_with_val
        from types import SimpleNamespace
        train1, val1, train2, val2, test = get_loaders_with_val(batch_size, num_workers)
        # EuroSAT: full modality_bands_dict built per-run from get_modality_bands_dict
        # Here we just return the loaders; caller builds modality_bands_dict after knowing starting+new mod
        task_config = SimpleNamespace(
            dataset_name='eurosat',
            task_type='classification',
            num_classes=10,
            multilabel=False,
            label_key='label',
            modality_slices=None,
            img_size=224,
        )
        return train1, val1, train2, val2, test, task_config

    elif dataset == 'benv2':
        from geobench_data_utils import get_benv2_loaders
        train1, val1, train2, val2, test, task_config = get_benv2_loaders(
            batch_size=batch_size, num_workers=num_workers
        )
        return train1, val1, train2, val2, test, task_config

    elif dataset == 'pastis':
        from geobench_data_utils import get_pastis_loaders
        train1, val1, train2, val2, test, task_config = get_pastis_loaders(
            batch_size=batch_size, num_workers=num_workers
        )
        return train1, val1, train2, val2, test, task_config

    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def main():
    parser = argparse.ArgumentParser(description='End to end training for SHOT model.')
    # IMPORTANT
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'benv2', 'pastis'],
                        help='Dataset to train on (default: eurosat)')
    parser.add_argument('--stage0_checkpoint', type=str, required=True,
                        help='Path to stage 0 checkpoint (required)')
    parser.add_argument('--new_mod_group', type=str, required=True,
                        help='New modality to add (eurosat: vre/nir/swir/rgb; benv2/pastis: s1/s2)')
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
    parser.add_argument('--use_mfla', action='store_true',
                        help='Enable MFLA training for hallucinated modalities')
    parser.add_argument('--mfla_warmup_epochs', type=int, default=0,
                        help='Epochs where backbone + MFLA train together before freezing backbone (default: 0)')
    parser.add_argument('--results_csv', type=str, required=True,
                        help='Path to results CSV file')

    # UNIMPORTANT
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--eval_every_n_epochs', type=int, default=4)
    parser.add_argument('--ssl_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--wandb_project', type=str, default='delulu-e2e-lossablate')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    args = parser.parse_args()

    # Validate new_mod_group against dataset
    valid_new_mods = VALID_NEW_MODS[args.dataset]
    if args.new_mod_group not in valid_new_mods:
        parser.error(f"--new_mod_group {args.new_mod_group!r} is not valid for --dataset {args.dataset}. "
                     f"Valid choices: {valid_new_mods}")

    # Load stage 0 checkpoint
    print(f"\n=== Loading Stage 0 checkpoint from: {args.stage0_checkpoint} ===")
    checkpoint = torch.load(args.stage0_checkpoint, map_location='cpu')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = EVANClassifier.from_checkpoint(args.stage0_checkpoint, device)
    config = checkpoint['config']
    evan_config = config['evan_config']
    starting_modality = evan_config['starting_modality']

    print(f"Stage 0 config:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print(f"\nUsing device: {device}")

    newmod = args.new_mod_group

    # Build modality_bands_dict
    if args.dataset == 'eurosat':
        from eurosat_data_utils import ALL_BAND_NAMES, get_modality_bands_dict
        modality_bands_dict = get_modality_bands_dict(starting_modality, newmod)
        bands_newmod = modality_bands_dict[newmod]
    else:
        # GeoBench: slices come from task_config (built when we load loaders below)
        # We'll fill modality_bands_dict after loading
        modality_bands_dict = None
        bands_newmod = None

    wandb.init(
        project=args.wandb_project,
        config={**config, **vars(args)},
        name=f"{args.dataset}_{starting_modality}+={newmod}--mae{args.mae_modalities}_lf{args.labeled_frequency}"
    )

    # Create datasets
    print("\n=== Creating datasets ===")
    train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config = \
        get_loaders_and_config(args.dataset, args.batch_size, args.num_workers)

    if args.dataset != 'eurosat':
        modality_bands_dict = task_config.modality_slices
        bands_newmod = modality_bands_dict[newmod]

    evan = model.evan
    model = model.to(device)
    print(f"SSL-trained components loaded: {newmod} patch embedder, {newmod} modality-specific LoRAs")

    num_newmod_channels = len(bands_newmod) if not isinstance(bands_newmod, slice) else \
        (bands_newmod.stop - bands_newmod.start)
    if newmod not in evan.patch_embedders:
        print(f"  Creating {newmod} modality components...")
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

    _, _, intermediate_projectors, trainable_total, best_checkpoints, best_checkpoint_summary = train_shot(
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
        val_loader=val2_loader,
        val_labeled_loader=val1_loader,
        use_mfla=args.use_mfla,
        mfla_warmup_epochs=args.mfla_warmup_epochs,
        multilabel=task_config.multilabel,
        label_key=task_config.label_key,
    )

    # ========================================= EVALUATION =====================================
    print("\n=== Evaluating trained model ===")

    all_modalities = [starting_modality, newmod]
    metric_name = "mAP" if task_config.multilabel else "Acc"

    objectives = {
        "transfer": {"desc": f"Using only {newmod}, hallucinating {starting_modality}"},
        "peeking":  {"desc": f"Using only {starting_modality}, hallucinating {newmod}"},
        "addition": {"desc": f"Using both {starting_modality} and {newmod}"}
    }

    metrics = {}
    addition_ens_metric = None
    for objective, info in objectives.items():
        print(f"\n--- Evaluating: {objective.capitalize()} objective ---")
        print(f"    {info['desc']}")

        metric, ens_metric = _delulu_stage3_test(
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
            multilabel=task_config.multilabel,
            label_key=task_config.label_key,
        )

        metrics[objective] = metric
        if objective == "addition" and ens_metric is not None:
            addition_ens_metric = ens_metric
            wandb.log({f"test/addition_ens_{metric_name.lower()}": ens_metric})
        print(f"{objective.capitalize()} {metric_name}: {metric:.2f}%")
        wandb.log({f"test/{objective}_{metric_name.lower()}": metric})

    # ========================================= CHECKPOINT =====================================
    timestamp_shot = datetime.now().strftime('%m%d_%H%M')
    checkpoint_shotete = os.path.join(args.checkpoint_dir, f'delulunet_{args.dataset}_{timestamp_shot}.pt')
    if args.checkpoint_name:
        checkpoint_shotete = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
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
        "dataset", "starting_modality", "new_modality", "ssl_lr", "weight_decay", "epochs",
        "mask_ratio", "modality_dropout", "labeled_frequency", "labeled_start_fraction",
        "trainable_params", "active_losses", "use_mfla", "mfla_warmup_epochs", "metric_name",
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
# EuroSAT (nir to rgb) — original behaviour
python -u shot_ete.py \
    --dataset eurosat \
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
"""
