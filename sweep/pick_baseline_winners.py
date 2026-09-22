"""Emit the val-winning baseline config per (dataset, cell) as a run plan.

One tab-separated line per cell: "<tag>\t<python arg string>". sh/baselines_seeds_all.sh
appends --seed N and submits each line three times.

The winners are READ FROM THE EXISTING SWEEP CSVs, never hardcoded, so the seed
replicates always reproduce the configuration the tables currently report. The
baselines need no new tuning: reBEN and EuroSAT already have 3-20 configs swept
per cell.

Only the cells the paper's tables actually report are emitted -- see
DIRECTIONS/SINGLES below, which mirror TRANSFER_PAIRS in res/results_BL.py.

Usage:
    python sweep/pick_baseline_winners.py --datasets benv2 eurosat
"""

import argparse
import csv
import glob
import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEACHERS_JSON = os.path.join(_ROOT, 'artifacts', 'sft_teachers.json')

# Cells the tables report, per dataset.
DIRECTIONS = {
    'benv2':   [('s2_rgb', 's1'), ('s2_rgb', 's2_norgb'), ('s1', 's2'), ('s2', 's1')],
    'eurosat': [('rgb', 'vre')],
}
# Single modalities for the teacher-free semi-supervised baselines. For peeking
# the table reports the STARTING modality, so these are the distinct starts.
SINGLES = {
    'benv2':   ['s2_rgb', 's1', 's2'],
    'eurosat': ['rgb'],
}
TEACHER_KEY = {
    'benv2':   'benv2/{mod}/evan_base/cls/split1',
    'eurosat': 'eurosat/{mod}/evan_base/cls/split1',
}
# Epoch budgets match the existing sweeps for each dataset/family.
EPOCHS = {'benv2': '20', 'eurosat': '20'}


def _rows(path, model='evan_base'):
    if not os.path.isfile(path):
        return []
    with open(path) as f:
        return [r for r in csv.DictReader(f) if r.get('model_type') == model]


def _best(rows, val_col):
    scored = []
    for r in rows:
        try:
            scored.append((float(r[val_col]), r))
        except (TypeError, ValueError, KeyError):
            continue
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    return scored[0][1]


def _teacher(dataset, mod):
    with open(TEACHERS_JSON) as f:
        reg = json.load(f)
    entry = reg.get(TEACHER_KEY[dataset].format(mod=mod)) or {}
    ckpt = entry.get('checkpoint', '')
    return ckpt if ckpt and os.path.isfile(os.path.join(_ROOT, ckpt)) else ''


def emit(tag, args, out):
    out.append(f'{tag}\t{args}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['benv2', 'eurosat'])
    args = ap.parse_args()

    lines, warn = [], []
    for ds in args.datasets:
        ep = EPOCHS[ds]

        # ---- distillation (KD/TTM): transfer column, unimodal student ----
        drows = []
        for f in glob.glob(os.path.join(_ROOT, f'res/baselines/distillation/{ds}/evan_base/*.csv')):
            drows.extend(_rows(f))
        for start, new in DIRECTIONS[ds]:
            teacher = _teacher(ds, start)
            if not teacher:
                warn.append(f'{ds} {start}->{new}: no teacher checkpoint'); continue
            for kl in ('kd', 'ttm'):
                cell = [r for r in drows
                        if r.get('teacher_modality') == start
                        and r.get('student_modality') == new
                        and r.get('kl_type') == kl]
                b = _best(cell, 'best_val_agreement')
                if b is None:
                    warn.append(f'{ds} {start}->{new} {kl}: no val-scored row'); continue
                init = str(b.get('init_from_teacher', '')).lower() in ('true', '1')
                emit(f'{ds}_distill_{kl}_{start}_to_{new}',
                     f"baseline/baseline_distillation.py --dataset {ds} --modalities {new} "
                     f"--teacher_checkpoint {teacher} --model evan_base "
                     f"--epochs {ep} --lr {b['learning_rate']} --kl_type {kl}"
                     + (' --init_from_teacher' if init else '') +
                     f" --results_csv res/baselines/distillation_{ds}_transfer.csv",
                     lines)

        # ---- MKE: addition column, bimodal student ----
        mrows = _rows(os.path.join(_ROOT, f'res/baselines/mke/{ds}.csv'))
        for start, new in DIRECTIONS[ds]:
            teacher = _teacher(ds, start)
            if not teacher:
                continue
            cell = [r for r in mrows
                    if r.get('teacher_modality') == start
                    and set(str(r.get('student_modalities', '')).split('+')) == {start, new}]
            b = _best(cell, 'valchecked_test_metric')
            if b is None:
                warn.append(f'{ds} MKE {start}+{new}: no val-scored row'); continue
            emit(f'{ds}_mke_{start}_to_{new}',
                 f"baseline/baseline_mke.py --dataset {ds} --modalities {start} {new} "
                 f"--teacher_checkpoint {teacher} --model evan_base "
                 f"--epochs {b.get('epochs', ep)} --lr {b['learning_rate']} "
                 f"--results_csv res/baselines/mke/{ds}.csv",
                 lines)

        # ---- FreeMatch / MixMatch: peeking column, teacher-free ----
        for fam, path, val_col in (
                ('freematch', f'res/baselines/freematch/baseline_freematch_{ds}.csv', 'best_val_metric'),
                ('mixmatch',  f'res/baselines/mixmatch/baseline_mixmatch_{ds}.csv',  'best_val_metric')):
            rows = _rows(os.path.join(_ROOT, path))
            for mod in SINGLES[ds]:
                b = _best([r for r in rows if r.get('modality') == mod], val_col)
                if b is None:
                    warn.append(f'{ds} {fam} {mod}: no val-scored row'); continue
                extra = f" --lambda_u {b['lambda_u']}" if b.get('lambda_u') else ''
                dino = ' --use_dino_weights' if str(b.get('use_dino_weights', '')).lower() in ('true', '1') else ''
                emit(f'{ds}_{fam}_{mod}',
                     f"baseline/baseline_{fam}.py --dataset {ds} --modality {mod} "
                     f"--model evan_base --epochs {b.get('epochs', ep)} "
                     f"--lr {b['learning_rate']}{extra}{dino} "
                     f"--results_csv {path}",
                     lines)

    for line in lines:
        print(line)
    if warn:
        import sys
        for w in warn:
            print(f'[warn] {w}', file=sys.stderr)


if __name__ == '__main__':
    main()
