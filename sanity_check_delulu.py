"""Quick multimodal eval of one adapted DeluluNet classifier checkpoint.

Loads an EVANClassifier checkpoint, builds the test loader for its dataset, and
reports transfer / peeking / addition metrics via evaluate_multimodal.

The original hardcoded checkpoint (delulunet_benv2_0501_0433.pt) no longer
exists and no adapted BEN-v2 classifier remains on disk, so the checkpoint is
now a required argument rather than a baked-in path.

  python -u sanity_check_delulu.py --checkpoint checkpoints/<ckpt>.pt
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from delulunet_main import EVANClassifier
from data_utils import get_loaders
from delulu import evaluate_multimodal


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True, help='Adapted EVANClassifier checkpoint (.pt)')
    ap.add_argument('--dataset', default='benv2', choices=['eurosat', 'benv2', 'dfc2020', 'biomassters'])
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EVANClassifier.from_checkpoint(args.checkpoint, device)
    model.eval()
    evan = model.evan

    mods = evan.supported_modalities
    if len(mods) < 2:
        raise SystemExit(f"Checkpoint has only {mods}; need an adapted (2-modality) model.")
    mod_a, mod_b = mods[0], mods[1]
    print(f"starting={mod_a}, new={mod_b}")

    *_, test_loader, task_config = get_loaders(
        dataset=args.dataset, starting_modality=mod_a, new_modality=mod_b,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )

    metrics = evaluate_multimodal(
        model=model, evan=evan,
        loader=test_loader, device=device,
        modality_bands_dict=task_config.modality_bands_dict,
        starting_modality=mod_a,
        newmod_modalities=[mod_b],
        all_modalities=[mod_a, mod_b],
        multilabel=task_config.multilabel, label_key=task_config.label_key,
        with_labels=True, desc="Test eval",
    )
    print(metrics)


if __name__ == '__main__':
    main()
