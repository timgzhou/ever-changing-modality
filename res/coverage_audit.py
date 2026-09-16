"""Which paper-table cells do we actually have, now that dfc2020 and biomassters moved?

Prints, per table (Transfer / Peek / Addition), per dataset and per modality
pair, whether each method has results -- and how many seeds. The point is to
show what is MISSING, so the remaining runs can be launched.

Not a LaTeX generator: res/results_BL.py owns that. This is the audit that says
which of its cells are empty and why.

Run from repo root:  python res/coverage_audit.py
"""
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Transfer = train on M_B alone after starting from M_A; Peek = M_A real +
# M_B hallucinated; Addition = both real. Methods per table follow the paper.
TABLES = {
    'TRANSFER': ['KD', 'TTM', 'Delulu'],
    'PEEK':     ['MixMatch', 'Delulu'],
    'ADDITION': ['MKE', 'Delulu'],
}

PAIRS = {
    'benv2':       [('s2_rgb','s1'), ('s2_rgb','s2_norgb'), ('s1','s2'), ('s2','s1')],
    'dfc2020':     [('s2_rgb','s1'), ('s2_rgb','s2_norgb'), ('s1','s2'), ('s2','s1')],
    'eurosat':     [('rgb','vre')],
    'biomassters': [('s2_rgb','s1'), ('s2_rgb','s2_norgb'), ('s1','s2'), ('s2','s1')],
}

METRIC = {'benv2': 'mAP', 'dfc2020': 'mIoU', 'eurosat': 'Acc', 'biomassters': 'RMSE'}
# RMSE is lower-is-better; every other metric is higher-is-better.
LOWER_IS_BETTER = {'biomassters'}


def _read(path):
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
    except Exception:
        return None
    if 'dataset' in df.columns:
        df = df[df['dataset'] != 'dataset']
    return df if len(df) else None


def _n(df):
    return 0 if df is None else len(df)


# --- per-method locators: return (n_rows, note) for one (dataset, pair) ---

def kd_ttm(dataset, src, tgt):
    """KD / TTM live in the distill-transfer CSVs (flat for dfc2020/biomassters,
    per-arch dirs for benv2/eurosat)."""
    cands = [f'res/baselines/{dataset}_cobench_distill_transfer_upernet.csv',
             f'res/baselines/{dataset}_distill_transfer_upernet.csv',
             f'res/baselines/{dataset}_cobench_distillation_upernet.csv',
             f'res/baselines/{dataset}_distillation_upernet.csv']
    cands += glob.glob(f'res/baselines/distillation/{dataset}/evan_base/*.csv')
    tot = 0
    for p in cands:
        df = _read(p)
        if df is None or 'teacher_modality' not in df.columns:
            continue
        sc = 'student_modality' if 'student_modality' in df.columns else None
        if sc is None:
            continue
        m = (df['teacher_modality'].astype(str) == src) & (df[sc].astype(str) == tgt)
        tot += int(m.sum())
    return tot, ''


def mixmatch(dataset, src, tgt):
    for p in (f'res/baselines/{dataset}_cobench_mixmatch_upernet.csv',
              f'res/baselines/{dataset}_mixmatch_upernet.csv',
              f'res/baselines/mixmatch/baseline_mixmatch_{dataset}.csv'):
        df = _read(p)
        if df is None:
            continue
        # MixMatch CSVs key on a single 'modality' (the STARTING one).
        if 'modality' in df.columns:
            n = int((df['modality'].astype(str) == src).sum())
            if n:
                vals = df.loc[df['modality'].astype(str) == src, 'test_metric'].astype(float)
                bad = (vals < 25).sum() if dataset != 'biomassters' else 0
                return n, (f'{bad}/{n} look like failed runs (<25)' if bad else '')
        for c in ('starting_modality', 'teacher_modality'):
            if c in df.columns:
                n = int((df[c].astype(str) == src).sum())
                if n:
                    return n, ''
    return 0, ''


def mke(dataset, src, tgt):
    for p in (f'res/baselines/{dataset}_cobench_mke_upernet.csv',
              f'res/baselines/{dataset}_mke_upernet.csv',
              f'res/baselines/mke/{dataset}.csv'):
        df = _read(p)
        if df is None:
            continue
        if {'teacher_modality', 'student_modalities'} <= set(df.columns):
            want = {f'{src}+{tgt}', f'{tgt}+{src}'}
            m = ((df['teacher_modality'].astype(str) == src)
                 & (df['student_modalities'].astype(str).isin(want)))
            n = int(m.sum())
            if n:
                return n, ''
    return 0, ''


def delulu(dataset, src, tgt):
    """Delulu results are spread over many CSVs; count any row matching the pair."""
    tot = 0
    for p in glob.glob('res/delulu/*.csv'):
        b = os.path.basename(p)
        # Skip quarantined / ablation-only files.
        # Quarantined (wrong teacher/split/scale) or ablation-only files. NOTE:
        # crossconfig is NOT excluded -- it holds real tuned results, including
        # the only biomassters s2_rgb<->s2_norgb rows.
        if any(t in b for t in ('BROKEN', 'BADTEACHER', 'BADABLATION', 'LEAKYTEACHER',
                                'SMOKE', 'probe', 'recon_ab', 'halluc', 'depth',
                                'self_distill', 'head_analysis')):
            continue
        df = _read(p)
        if df is None or 'dataset' not in df.columns:
            continue
        if not {'starting_modality', 'new_modality'} <= set(df.columns):
            continue
        m = ((df['dataset'].astype(str) == dataset)
             & (df['starting_modality'].astype(str) == src)
             & (df['new_modality'].astype(str) == tgt))
        tot += int(m.sum())
    return tot, ''


LOADERS = {'KD': kd_ttm, 'TTM': kd_ttm, 'MixMatch': mixmatch, 'MKE': mke, 'Delulu': delulu}


def main():
    print('Coverage audit — rows found per (dataset, pair, method).')
    print('0 = MISSING.  Counts are raw CSV rows (seeds x configs), not seeds.\n')
    missing = []
    for table, methods in TABLES.items():
        print('=' * 78)
        print(f'  {table}')
        print('=' * 78)
        print(f"{'dataset':<13}{'metric':<7}{'pair':<22}" + ''.join(f'{m:>11}' for m in methods))
        for ds, pairs in PAIRS.items():
            for src, tgt in pairs:
                cells, notes = [], []
                for meth in methods:
                    n, note = LOADERS[meth](ds, src, tgt)
                    cells.append(n)
                    if note:
                        notes.append(f'{meth}: {note}')
                    if n == 0:
                        missing.append((table, ds, f'{src}->{tgt}', meth))
                print(f"{ds:<13}{METRIC[ds]:<7}{src+' -> '+tgt:<22}"
                      + ''.join(f'{c:>11}' for c in cells)
                      + ('   <- ' + '; '.join(notes) if notes else ''))
        print()

    print('=' * 78)
    print('  MISSING CELLS')
    print('=' * 78)
    if not missing:
        print('  none')
    else:
        by = {}
        for t, ds, pair, meth in missing:
            by.setdefault((t, ds, meth), []).append(pair)
        for (t, ds, meth), pairs in sorted(by.items()):
            print(f'  {t:<10}{ds:<13}{meth:<10} {len(pairs)} pair(s): {", ".join(pairs)}')
        print(f'\n  {len(missing)} missing cells total')


if __name__ == '__main__':
    main()
