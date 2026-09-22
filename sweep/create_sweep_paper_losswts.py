"""Per-(dataset, direction, selector) sweeps over lr + the two reconstruction
loss weights, for the three datasets the paper reports.

PROTOCOL
--------
1. The STRUCTURAL hyperparameters -- token mask ratio, both modality dropouts,
   labeled_frequency -- were tuned once (the DFC2020 128-trial deep sweep,
   res/delulu-sweep/best_deep.json) and TRANSFER across datasets. That
   transferability is a result the paper reports, not a confound, so the same
   per-selector values are pinned here for every dataset.
2. lambda_distill is likewise pinned: it is a task-loss weight on a shared
   logit/label scale, not a reconstruction term, so by the same hypothesis it
   carries over.
3. What does NOT transfer is the optimiser/loss SCALE. lr, lambda_latent and
   lambda_prefusion are therefore swept per (dataset, direction, selector).
4. The val-winning config of each sweep is then re-run with 3 seeds; those
   three runs are what the table's mean +/- std reports.

WHY lr IS FREE HERE. The existing benv2 sweeps
(res/delulu-sweep/losswts_benv2_*.csv) pinned lr per selector and swept only the
two loss weights. That makes benv2 cells tuned on fewer dimensions than any cell
swept with sweep_dfc2020_losswts_lr.yaml, which is not a like-for-like table.
Freeing lr everywhere is the uniform choice, and it supersedes those 11 sweeps.

TEACHERS come from artifacts/sft_teachers.json (the current registry), NOT the
pinned historical checkpoints used by create_sweep_dfc2020_losswts.py. That
script was measuring partial transfer, where the teacher had to be held at the
baseline's value; here we want the numbers the paper reports.

Usage:
    python sweep/create_sweep_paper_losswts.py --dry-run
    python sweep/create_sweep_paper_losswts.py --datasets dfc2020 eurosat
"""

import argparse
import json
import os

import yaml

_SWEEP_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_SWEEP_DIR)
_YAML_DIR = os.path.join(_SWEEP_DIR, 'sweep_yaml')

SWEEP_YAML = 'sweep_dfc2020_losswts_lr.yaml'   # lr + lambda_latent + lambda_prefusion
DEEP_JSON = os.path.join(_ROOT, 'res', 'delulu-sweep', 'best_deep.json')
DEEP_KEY = 'dfc2020_s1_to_s2norgb'
TEACHERS_JSON = os.path.join(_ROOT, 'artifacts', 'sft_teachers.json')

SELECTORS = ['transfer', 'peeking', 'addition']

# 3 free params. create_sweep_dfc2020_losswts.py used 16 for the same 3-param
# space and the existing 2-param benv2 sweeps used 8-11; 12 sits between them
# and keeps 27 sweeps affordable.
N_JOBS = 12

# Structural knobs pinned from the deep sweep, per selector. mae_mask_ratio is
# spelled --token_mask_ratio on train_delulu.py's CLI.
PINNED = ['token_mask_ratio', 'modality_dropout_startmod', 'modality_dropout_newmod',
          'labeled_frequency', 'lambda_distill']
_HPARAM_TO_FLAG = {'mae_mask_ratio': 'token_mask_ratio'}

# Per-dataset: the reported directions, the teacher registry key template, the
# batch size, and the epoch budget. Batch sizes match the existing launchers --
# dfc2020 is dense segmentation and needs the smaller batch.
DATASETS = {
    'dfc2020': {
        'teacher_key': 'dfc2020_cobench/{start}/evan_base/upernet/split1',
        'batch_size': '8',
        'epochs': '64',
        'directions': [
            {'slug': 's2rgb_to_s1',      'start': 's2_rgb', 'new_mod': 's1'},
            {'slug': 's2rgb_to_s2norgb', 'start': 's2_rgb', 'new_mod': 's2_norgb'},
            {'slug': 's1_to_s2',         'start': 's1',     'new_mod': 's2'},
            {'slug': 's2_to_s1',         'start': 's2',     'new_mod': 's1'},
        ],
    },
    'benv2': {
        'teacher_key': 'benv2/{start}/evan_base/cls/split1',
        'batch_size': '32',
        'epochs': '64',
        'directions': [
            {'slug': 's2rgb_to_s1',      'start': 's2_rgb', 'new_mod': 's1'},
            {'slug': 's2rgb_to_s2norgb', 'start': 's2_rgb', 'new_mod': 's2_norgb'},
            {'slug': 's1_to_s2',         'start': 's1',     'new_mod': 's2'},
            {'slug': 's2_to_s1',         'start': 's2',     'new_mod': 's1'},
        ],
    },
    'eurosat': {
        'teacher_key': 'eurosat/{start}/evan_base/cls/split1',
        'batch_size': '32',
        'epochs': '64',
        'directions': [
            {'slug': 'rgb_to_vre', 'start': 'rgb', 'new_mod': 'vre'},
        ],
    },
}


def _deep_hparams(selector):
    with open(DEEP_JSON) as f:
        deep = json.load(f)
    h = dict(deep[DEEP_KEY][selector]['hparams'])
    return {_HPARAM_TO_FLAG.get(k, k): v for k, v in h.items()}


def _teacher(dataset, start):
    with open(TEACHERS_JSON) as f:
        reg = json.load(f)
    key = DATASETS[dataset]['teacher_key'].format(start=start)
    entry = reg.get(key) or {}
    return key, entry.get('checkpoint', '')


def _load_sweep_cfg():
    with open(os.path.join(_YAML_DIR, SWEEP_YAML)) as f:
        override = yaml.safe_load(f)
    merged = dict(override)
    # Deliberately does NOT merge base.yaml -- that would re-free weight_decay.
    merged['parameters'] = dict(override.get('parameters', {}))
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--datasets', nargs='+', default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument('--selectors', nargs='+', default=SELECTORS, choices=SELECTORS)
    args = ap.parse_args()

    base = _load_sweep_cfg()
    made, missing = [], []
    for dataset in args.datasets:
        spec = DATASETS[dataset]
        for direction in spec['directions']:
            key, teacher = _teacher(dataset, direction['start'])
            if not teacher or not os.path.isfile(os.path.join(_ROOT, teacher)):
                missing.append(f'{dataset} {direction["slug"]}: no checkpoint for {key}')
                continue
            for selector in args.selectors:
                hp = _deep_hparams(selector)
                extra = [
                    '--dataset', dataset,
                    '--stage0_checkpoint', teacher,
                    '--new_mod_group', direction['new_mod'],
                    '--epochs', spec['epochs'],
                    '--batch_size', spec['batch_size'],
                    '--select_by', selector,
                    '--results_csv',
                    f"res/delulu-sweep/paper_losswts_{dataset}_{direction['slug']}_{selector}.csv",
                ]
                for k in PINNED:
                    if k in hp:
                        extra.extend([f'--{k}', str(hp[k])])
                extra.extend(['--weight_decay', '2e-4'])
                for loss in ('latent', 'prefusion', 'distill', 'ce'):
                    extra.extend(['--active_losses', loss])
                # sweep_delulu.py parses these with type=_bool, so they REQUIRE
                # a value; a bare flag makes argparse swallow the next token.
                extra.extend(['--latent_masked_only', 'True'])
                extra.extend(['--use_mask_token', 'False'])

                cfg = dict(base)
                cfg['name'] = f"paperlw-{dataset}-{direction['slug']}-{selector}"
                cfg['command'] = ['${env}', 'python', '${program}', '${args}'] + extra
                made.append((cfg['name'], cfg, extra, teacher))

    print(f'{len(made)} sweeps x {N_JOBS} trials = {len(made)*N_JOBS} runs')
    for name, cfg, extra, teacher in made:
        pins = ' '.join(f'{a}={b}' for a, b in zip(extra[::2], extra[1::2])
                        if a.lstrip('-') in PINNED)
        print(f'  {name}')
        print(f'      free:    {list(cfg["parameters"])}')
        print(f'      pinned:  {pins[:150]}')
        print(f'      teacher: {teacher.split("/")[-1]}')
    if missing:
        print('\nSKIPPED (no teacher):')
        for m in missing:
            print(f'  {m}')

    if args.dry_run:
        print('\n--dry-run: nothing registered')
        return

    import wandb
    ids = []
    for name, cfg, _, _ in made:
        sid = wandb.sweep(cfg, project=os.environ.get('WANDB_PROJECT', 'delulu-paperlw'))
        ids.append((name, sid))
        print(f'registered {name} -> {sid}')
    out = os.path.join(_ROOT, 'res', 'delulu-sweep', 'paper_losswts_sweep_ids.txt')
    with open(out, 'w') as f:
        for name, sid in ids:
            f.write(f'{name} {sid}\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
