"""Register the MIRROR partial-transfer sweeps: BEN-v2 config on DFC2020.

The first arm (sweep/create_sweep_benv2_losswts.py) pinned a DFC2020-tuned
config onto BEN-v2 and re-tuned only the two reconstruction loss weights. This
runs the same test the other way round, and additionally frees lr.

  pinned from configs/delulu_best_benv2.yaml, per selector:
      token_mask_ratio, modality_dropout_startmod, modality_dropout_newmod,
      labeled_frequency, lambda_distill
  swept:
      lr, lambda_latent, lambda_prefusion

WHY lr IS FREE HERE. In the first arm lr was pinned, which isolated the loss
weights but stopped the partial sweep from following lr when the loss scale
moved -- and lr is the most predictive hyperparameter for transfer in the
BEN-v2 baseline (rho +0.294, p=2.7e-06). Freeing it tests the weaker, more
practical claim: the STRUCTURAL parameters transfer, the optimiser scale does
not. It also answers the question directly -- can tuning lr alone recover the
gap?

COMPARISON. With lr free this must be scored against the UNRESTRICTED full
sweep, not an lr-matched band; restricting the baseline would hand this arm an
advantage the baseline is denied.

  baseline: res/delulu-sweep/sweep_results_dfc2020_s1_to_s2norgb_deep.csv
            128 trials, 8 free params, post-latent-fix.

TEACHER. That baseline used the OLD s1 teacher (..._20260821_194707.pt,
test 49.05). This sweep pins the SAME checkpoint rather than reading the
teacher registry, which now points at the better 64-epoch teachers -- using
those would confound the teacher upgrade with the transfer measurement.

Only s1->s2_norgb is registered: it is the sole DFC2020 direction with a
post-latent-fix full sweep to compare against.

Usage (from repo root):
    python sweep/create_sweep_dfc2020_losswts.py [--dry-run]
"""

import argparse
import os

import yaml

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SWEEP_DIR)
_YAML_DIR = os.path.join(_SWEEP_DIR, 'sweep_yaml')

SWEEP_YAML = 'sweep_dfc2020_losswts_lr.yaml'
BENV2_CONFIG = os.path.join(_ROOT, 'configs', 'delulu_best_benv2.yaml')

EPOCHS = '64'
BATCH_SIZE = '8'       # dfc2020 segmentation; matches sh/train_delulu_crossconfig.sh
N_JOBS = 16            # 3 free params, so a few more draws than the 12 used for 2

# Pinned to the exact checkpoint the baseline sweep distilled from. NOT read
# from artifacts/sft_teachers.json: that registry now resolves to the 64-epoch
# teachers, which would confound the teacher upgrade with transfer.
TEACHER = 'checkpoints/sft_evan_base_dfc2020_s1_fft_lr0.0005_20260821_194707.pt'

DIRECTIONS = [
    {'slug': 's1_to_s2norgb', 'start': 's1', 'new_mod': 's2_norgb'},
]

SELECTORS = ['transfer', 'peeking', 'addition']

# lr is NOT here -- it is swept. Everything else structural is pinned.
PINNED = ['token_mask_ratio', 'modality_dropout_startmod', 'modality_dropout_newmod',
          'labeled_frequency', 'lambda_distill']

_HPARAM_TO_FLAG = {'mae_mask_ratio': 'token_mask_ratio'}


def _benv2_hparams(selector: str):
    with open(BENV2_CONFIG) as f:
        cfg = yaml.safe_load(f)
    h = dict(cfg['configs'][selector]['hparams'])
    return {_HPARAM_TO_FLAG.get(k, k): v for k, v in h.items()}, cfg.get('fixed', {})


def _load_sweep_cfg() -> dict:
    with open(os.path.join(_YAML_DIR, SWEEP_YAML)) as f:
        override = yaml.safe_load(f)
    merged = dict(override)
    # Deliberately does NOT merge base.yaml: that would re-free weight_decay.
    # lr IS free here, but from this file's own range, matched to the baseline.
    merged['parameters'] = dict(override.get('parameters', {}))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--selectors', nargs='+', default=SELECTORS, choices=SELECTORS)
    args = ap.parse_args()

    if not os.path.isfile(os.path.join(_ROOT, TEACHER)):
        raise SystemExit(f'teacher checkpoint missing: {TEACHER}')

    base = _load_sweep_cfg()
    made = []
    for direction in DIRECTIONS:
        for selector in args.selectors:
            hp, fixed = _benv2_hparams(selector)
            extra = [
                '--dataset', 'dfc2020',
                '--stage0_checkpoint', TEACHER,
                '--new_mod_group', direction['new_mod'],
                '--epochs', EPOCHS,
                '--batch_size', BATCH_SIZE,
                '--select_by', selector,
                '--results_csv',
                f"res/delulu-sweep/losswts_dfc2020_{direction['slug']}_{selector}.csv",
            ]
            for k in PINNED:
                if k in hp:
                    extra.extend([f'--{k}', str(hp[k])])
            for k, v in fixed.items():
                if k == 'active_losses':
                    for loss in v:
                        extra.extend(['--active_losses', str(loss)])
                elif isinstance(v, bool):
                    # sweep_delulu.py parses these with type=_bool, so they
                    # REQUIRE a value; a bare flag makes argparse swallow the
                    # next token and the trial dies before training.
                    extra.extend([f'--{k}', 'True' if v else 'False'])
                else:
                    extra.extend([f'--{k}', str(v)])

            cfg = dict(base)
            cfg['name'] = f"losswts-dfc2020-{direction['slug']}-{selector}"
            cfg['command'] = ['${env}', 'python', '${program}', '${args}'] + extra
            made.append((cfg['name'], cfg, extra))

    print(f'{len(made)} sweeps, {N_JOBS} trials each')
    for name, cfg, extra in made:
        pins = ' '.join(f'{a}={b}' for a, b in zip(extra[::2], extra[1::2])
                        if a.lstrip('-') in PINNED)
        print(f'  {name}')
        print(f'      free:   {list(cfg["parameters"])}')
        print(f'      pinned: {pins[:160]}')
        print(f'      teacher: {TEACHER.split("/")[-1]}')

    if args.dry_run:
        print('\n--dry-run: nothing registered')
        return

    import wandb
    for name, cfg, _ in made:
        sid = wandb.sweep(cfg, project=os.environ.get('WANDB_PROJECT', 'delulu-losswts'))
        print(f'  registered {name}: {sid}')


if __name__ == '__main__':
    main()
