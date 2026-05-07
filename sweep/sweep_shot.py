"""
Thin W&B Sweeps wrapper around shot_ete.main().

All training logic lives in shot_ete.py. This file only:
  1. Parses swept + fixed args (using shot_ete's argument names/defaults).
  2. Converts the three boolean-as-string flags that W&B passes as "True"/"False" strings.
  3. Calls shot_ete.main(args).

Usage:
    wandb sweep sweep_config.yaml
    wandb agent <sweep-id>
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shot_ete


def main():
    parser = argparse.ArgumentParser(description='W&B Sweeps wrapper for SHOT training.')

    # ── Fixed args (non-swept) ────────────────────────────────────────────────
    parser.add_argument('--dataset', type=str, default='eurosat',
                        choices=['eurosat', 'benv2', 'pastis', 'dfc2020'])
    parser.add_argument('--stage0_checkpoint', type=str, required=True)
    parser.add_argument('--new_mod_group', type=str, required=True)
    parser.add_argument('--results_csv', type=str,
                        default='res/delulu-sweep/sweep_results_nomae.csv')
    parser.add_argument('--wandb_project', type=str, default='delulu-sweep')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=64)
    parser.add_argument('--eval_every_n_epochs', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--tz_fusion_time', type=int, default=3)
    parser.add_argument('--intermediate_projector_num_layers', type=int, default=2)

    # ── Swept continuous hyperparameters ─────────────────────────────────────
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--asym_lr', type=float, default=None)
    parser.add_argument('--modality_dropout', type=float, default=0.3)
    parser.add_argument('--labeled_frequency', type=float, default=0.3)
    parser.add_argument('--labeled_start_fraction', type=float, default=0.5)
    parser.add_argument('--token_mask_ratio', '--mae_mask_ratio', type=float, default=0.75)
    parser.add_argument('--lambda_latent', type=float, default=1.0)
    parser.add_argument('--lambda_prefusion', type=float, default=1.0)
    parser.add_argument('--lambda_distill', type=float, default=1.0)
    parser.add_argument('--lambda_ce', type=float, default=1.0)

    # ── Swept discrete flags — W&B passes these as "True"/"False" strings ────
    _bool = lambda x: x.lower() == 'true'
    parser.add_argument('--protect_lrm', type=float, default=0.0)
    parser.add_argument('--use_mask_token', type=_bool, default=False)
    parser.add_argument('--latent_masked_only', type=_bool, default=False)
    parser.add_argument('--unprotect_starting_mod', type=_bool, default=False)

    # ── Per-modality dropout overrides ───────────────────────────────────────
    parser.add_argument('--modality_dropout_startmod', type=float, default=None)
    parser.add_argument('--modality_dropout_newmod', type=float, default=None)

    # ── active_losses — W&B injects repeated flags: --active_losses X --active_losses Y
    parser.add_argument('--active_losses', type=str, action='append', required=True,
                        choices=['latent', 'prefusion', 'distill', 'ce'])

    args = parser.parse_args()

    # Provide defaults that shot_ete expects but sweep doesn't use
    args.dyn_teacher = False
    args.checkpoint_name = None
    args.save_checkpoint = False
    args.select_by = None

    shot_ete.main(args)


if __name__ == '__main__':
    main()
