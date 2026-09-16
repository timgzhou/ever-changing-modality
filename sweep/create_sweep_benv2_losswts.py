"""Register the partial-transfer sweeps: DFC2020 config on BEN-v2, loss weights re-tuned.

Tests whether DeluluNet hyperparameters transfer across datasets when only the
two RECONSTRUCTION loss weights are re-tuned.

  pinned from configs/delulu_best_dfc2020.yaml, per selector:
      token_mask_ratio, modality_dropout_startmod, modality_dropout_newmod,
      labeled_frequency, lambda_distill, lr, weight_decay
  swept:
      lambda_latent, lambda_prefusion

One sweep per (direction, selector), so the transfer<->transfer,
peeking<->peeking and addition<->addition correspondence is preserved end to
end: each sweep pins the DFC2020 config tuned FOR THAT SELECTOR, and its result
is compared against the fully-swept BEN-v2 number FOR THAT SAME SELECTOR.
Mixing selectors would compare a config tuned for one objective against a
baseline tuned for another, which is not the question.

Usage (from repo root):
    python sweep/create_sweep_benv2_losswts.py [--dry-run] [--selectors transfer ...]
"""

import argparse
import json
import os

import yaml

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SWEEP_DIR)
_YAML_DIR = os.path.join(_SWEEP_DIR, 'sweep_yaml')
_TEACHERS = os.path.join(_ROOT, 'artifacts', 'sft_teachers.json')

SWEEP_YAML = 'sweep_benv2_losswts_only.yaml'
DFC_CONFIG = os.path.join(_ROOT, 'configs', 'delulu_best_dfc2020.yaml')

EPOCHS = '64'          # matches the benv2 runs this is compared against
BATCH_SIZE = '64'
N_JOBS = 12            # only 2 free params; 12 random draws covers the plane

# The two BEN-v2 directions with split1 teachers and a fully-swept baseline.
DIRECTIONS = [
    {'slug': 's1_to_s2',            'start': 's1',     'new_mod': 's2'},
    {'slug': 's2_to_s1',            'start': 's2',     'new_mod': 's1'},
    {'slug': 's2rgb_to_s2norgb',    'start': 's2_rgb', 'new_mod': 's2_norgb'},
    {'slug': 's2rgb_to_s1',         'start': 's2_rgb', 'new_mod': 's1'},
]

SELECTORS = ['transfer', 'peeking', 'addition']

# Pinned from the DFC2020 tuned config. `lr` is pinned too: this is a test of
# "loss weights only". If the hypothesis fails, freeing lr is the next step --
# lr is the single most predictive hparam for transfer (rho +0.36).
PINNED = ['token_mask_ratio', 'modality_dropout_startmod', 'modality_dropout_newmod',
          'labeled_frequency', 'lambda_distill', 'lr']

# The sweep CSV column is mae_mask_ratio but the CLI flag is token_mask_ratio.
_HPARAM_TO_FLAG = {'mae_mask_ratio': 'token_mask_ratio'}


def _teacher_for(modality: str) -> str:
    with open(_TEACHERS) as f:
        reg = json.load(f)
    key = f'benv2/{modality}/evan_base/cls/split1'
    entry = reg.get(key)
    if not entry or not entry.get('checkpoint'):
        raise SystemExit(f'No split1 teacher for {modality!r} (key {key})')
    return entry['checkpoint']


def _dfc_hparams(selector: str) -> dict:
    with open(DFC_CONFIG) as f:
        cfg = yaml.safe_load(f)
    h = dict(cfg['configs'][selector]['hparams'])
    fixed = cfg.get('fixed', {})
    out = {}
    for k, v in h.items():
        out[_HPARAM_TO_FLAG.get(k, k)] = v
    return out, fixed


def _load_merged() -> dict:
    with open(os.path.join(_YAML_DIR, 'base.yaml')) as f:
        base = yaml.safe_load(f)
    with open(os.path.join(_YAML_DIR, SWEEP_YAML)) as f:
        override = yaml.safe_load(f)
    merged = dict(override)
    # Only the sweep's OWN parameters are free. base.yaml contributes lr and
    # weight_decay, which this experiment deliberately pins -- so do NOT merge
    # them in, or they would silently become free again.
    merged['parameters'] = dict(override.get('parameters', {}))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--selectors', nargs='+', default=SELECTORS, choices=SELECTORS)
    ap.add_argument('--directions', nargs='+', default=None,
                    help='slugs to include (default: all)')
    args = ap.parse_args()

    dirs = [d for d in DIRECTIONS
            if args.directions is None or d['slug'] in args.directions]
    base = _load_merged()
    made = []

    for direction in dirs:
        teacher = _teacher_for(direction['start'])
        for selector in args.selectors:
            hp, fixed = _dfc_hparams(selector)
            extra = [
                '--dataset', 'benv2',
                '--stage0_checkpoint', teacher,
                '--new_mod_group', direction['new_mod'],
                '--epochs', EPOCHS,
                '--batch_size', BATCH_SIZE,
                '--select_by', selector,
                '--results_csv',
                f"res/delulu-sweep/losswts_benv2_{direction['slug']}_{selector}.csv",
            ]
            for k in PINNED:
                if k in hp:
                    extra.extend([f'--{k}', str(hp[k])])
            for k, v in fixed.items():
                if k == 'active_losses':
                    for loss in v:
                        extra.extend(['--active_losses', str(loss)])
                elif isinstance(v, bool):
                    # sweep_delulu.py parses its boolean flags with type=_bool
                    # (--latent_masked_only, --unprotect_starting_mod,
                    # --use_mask_token), so they REQUIRE a value. A bare
                    # `--latent_masked_only` -- which is what train_delulu.py
                    # takes -- makes argparse swallow the next token and the
                    # trial dies before training. Always emit the value.
                    extra.extend([f'--{k}', 'True' if v else 'False'])
                else:
                    extra.extend([f'--{k}', str(v)])

            cfg = dict(base)
            cfg['name'] = f"losswts-benv2-{direction['slug']}-{selector}"
            cfg['command'] = ['${env}', 'python', '${program}', '${args}'] + extra
            made.append((cfg['name'], cfg, extra))

    print(f'{len(made)} sweeps ({len(dirs)} directions x {len(args.selectors)} selectors), '
          f'{N_JOBS} trials each')
    for name, cfg, extra in made:
        pins = ' '.join(f'{a}={b}' for a, b in zip(extra[::2], extra[1::2])
                        if a.lstrip("-") in PINNED)
        print(f'  {name}')
        print(f'      free: {list(cfg["parameters"])}')
        print(f'      pinned: {pins[:150]}')

    if args.dry_run:
        print('\n--dry-run: nothing registered')
        return

    import wandb
    for name, cfg, _ in made:
        sid = wandb.sweep(cfg, project=os.environ.get('WANDB_PROJECT', 'delulu-losswts'))
        print(f'  registered {name}: {sid}')


if __name__ == '__main__':
    main()
