"""Transfer / Peek / Addition tables with ViT-B and ViT-L side-by-side columns.

Outputs a single res/latex/BL_tables.tex with all three tabular environments.

Run from repo root:
    python res/results_BL.py
"""

import os
import glob
import sys
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Shared helpers (duplicated from results_table.py / to_latex.py to keep this
# script fully self-contained)
# ---------------------------------------------------------------------------

DATASETS = ['benv2', 'dfc2020', 'eurosat']

DATASET_NAMES = {
    'benv2':   'reBEN (Multi-Label Classification, mAP)',
    'dfc2020': 'DFC2020 (Semantic Segmentation, mIoU)',
    'eurosat': 'EuroSAT (Classification, Acc)',
}

VALID_TRANSFERS = {
    'benv2':   [('s2_rgb', 's1'), ('s2_rgb', 's2_norgb'), ('s1', 's2'), ('s2', 's1')],
    'dfc2020': [('s2_rgb', 's1'), ('s2_rgb', 's2_norgb'), ('s1', 's2'), ('s2', 's1')],
    'eurosat': [('rgb', 'nir'), ('rgb', 'vre'), ('rgb', 'swir')],
}

MOD_DISPLAY = {
    's2_rgb': 'S2-RGB', 's2': 'S2', 's1': 'S1',
    's2_norgb': 'S2-noRGB', 'rgb': 'RGB', 'vre': 'VRE',
    'nir': 'NIR', 'swir': 'SWIR',
}

COMBINED_RSFM_ALIASES = {
    ('s2', 's1'):           ['s2s1', 's2+s1', 's1+s2'],
    ('s1', 's2'):           ['s2s1', 's1+s2', 's2+s1'],
    ('s2_rgb', 's1'):       ['s2_rgb+s1', 's1+s2_rgb', 's2s1', 's2+s1'],
    ('s2_rgb', 's2_norgb'): ['s2_rgb+s2_norgb', 's2_norgb+s2_rgb', 's2'],
    ('rgb', 'nir'):         ['rgb+nir', 'nir+rgb'],
    ('rgb', 'vre'):         ['rgb+vre', 'vre+rgb'],
    ('rgb', 'swir'):        ['rgb+swir', 'swir+rgb'],
    ('swir', 'nir'):        ['swir+nir', 'nir+swir'],
    ('swir', 'rgb'):        ['swir+rgb', 'rgb+swir'],
    ('swir', 'vre'):        ['swir+vre', 'vre+swir'],
    ('vre', 'nir'):         ['vre+nir', 'nir+vre'],
    ('vre', 'rgb'):         ['vre+rgb', 'rgb+vre'],
}

OUT_DIR = 'res/latex'


def _read_csv(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    if 'dataset' in df.columns:
        df = df[df['dataset'] != 'dataset']
    return df if not df.empty else None


def _fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '--'
    return f'{float(val):.{decimals}f}'


def _fmt_meanstd(val, decimals=2):
    if val is None:
        return '--'
    mean, std = val
    if np.isnan(mean):
        return '--'
    if np.isnan(std):
        return f'{mean:.{decimals}f}'
    return f'{mean:.{decimals}f}±{std:.{decimals}f}'


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_distillation(dataset, arch):
    base = f'res/baselines/distillation/{dataset}/{arch}'
    if not os.path.isdir(base):
        return {}
    result = {}
    for fpath in glob.glob(f'{base}/*.csv'):
        df = _read_csv(fpath)
        if df is None or 'teacher_modality' not in df.columns:
            continue
        df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
        for (teacher, student, kl_type), grp in df.groupby(['teacher_modality', 'student_modality', 'kl_type']):
            topk = grp.nlargest(5, 'test_metric')['test_metric']
            result[(teacher, student, kl_type)] = (topk.mean(), topk.std())
    return result


def _load_delulu(dataset, arch, val_col, test_col):
    df = _read_csv('res/delulu/hptuned_apr21.csv')
    if df is None:
        return {}
    df = df[(df['dataset'] == dataset) & (df['model_arch'] == arch)]
    df[test_col] = pd.to_numeric(df[test_col], errors='coerce')
    df[val_col]  = pd.to_numeric(df[val_col],  errors='coerce')
    result = {}
    for (start, new), grp in df.groupby(['starting_modality', 'new_modality']):
        top1 = grp.nlargest(1, val_col)[test_col]
        result[(start, new)] = (top1.mean(), top1.std())
    return result


def _load_rsfm(dataset):
    df = _read_csv('res/rsfm/rsfm_results.csv')
    if df is None:
        return {}
    df = df[(df['dataset'] == dataset) & (df['train_mode'] == 'fft')]
    df = df[~df['model'].str.lower().str.contains('dino')]
    df['val_metric']  = pd.to_numeric(df['val_metric'],  errors='coerce')
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    result = {}
    for (model, modality), grp in df.groupby(['model', 'modality']):
        best = grp.sort_values('val_metric', ascending=False).iloc[0]
        result[(model, modality)] = best['test_metric']
    return result


def _load_sft_dino(dataset, arch='evan_base'):
    """DINO-init SFT for given arch: modality → test_metric (val-selected)."""
    df = _read_csv(f'res/train_sft/{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == arch]
    df['dino_init'] = df['dino_init'].astype(str).str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False})
    df = df[df['dino_init'] == True]
    if df.empty:
        return {}
    df['val_metric']  = pd.to_numeric(df['val_metric'],  errors='coerce')
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    result = {}
    for modality, grp in df.groupby('modality'):
        best = grp.sort_values('val_metric', ascending=False).iloc[0]
        result[modality] = best['test_metric']
    return result


def _load_mke_addition(dataset, arch='evan_base'):
    df = _read_csv(f'res/baselines/mke/{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == arch]
    if df.empty:
        return {}
    df['valchecked_test_metric'] = pd.to_numeric(df['valchecked_test_metric'], errors='coerce')
    result = {}
    for (teacher, student_mods), grp in df.groupby(['teacher_modality', 'student_modalities']):
        if '+' not in str(student_mods):
            continue
        parts     = [p.strip() for p in student_mods.split('+')]
        new_parts = [p for p in parts if p != teacher]
        if len(new_parts) != 1:
            continue
        result[(teacher, new_parts[0])] = grp['valchecked_test_metric'].max()
    return result


def _load_mixmatch_peek(dataset, arch='evan_base'):
    df = _read_csv(f'res/baselines/mixmatch/baseline_mixmatch_{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == arch]
    if df.empty:
        return {}
    df['best_val_metric']      = pd.to_numeric(df['best_val_metric'],      errors='coerce')
    df['best_val_test_metric'] = pd.to_numeric(df['best_val_test_metric'], errors='coerce')
    result = {}
    for modality, grp in df.groupby('modality'):
        top3 = grp.nlargest(3, 'best_val_metric')['best_val_test_metric']
        result[modality] = (top3.mean(), top3.std())
    return result


def _load_sft_combined_dino(dataset, arch='evan_base'):
    """DINO-init SFT on combined modality for given arch."""
    df = _read_csv(f'res/train_sft/{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == arch]
    df['dino_init'] = df['dino_init'].astype(str).str.lower().map(
        {'true': True, 'false': False, '1': True, '0': False})
    df = df[df['dino_init'] == True]
    if df.empty:
        return {}
    df['val_metric']  = pd.to_numeric(df['val_metric'],  errors='coerce')
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    result = {}
    for modality, grp in df.groupby('modality'):
        if '+' not in str(modality):
            continue
        best = grp.sort_values('val_metric', ascending=False).iloc[0]
        result[modality] = best['test_metric']
    return result


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_transfer_BL(dataset):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    dist_b = _load_distillation(dataset, 'evan_base')
    dist_l = _load_distillation(dataset, 'evan_large')
    del_b  = _load_delulu(dataset, 'evan_base',  'valchecked_val_transfer', 'valchecked_transfer')
    del_l  = _load_delulu(dataset, 'evan_large', 'valchecked_val_transfer', 'valchecked_transfer')
    sft_b  = _load_sft_dino(dataset, 'evan_base')
    sft_l  = _load_sft_dino(dataset, 'evan_large')
    rsfm   = _load_rsfm(dataset)

    rows = []
    for (start, new) in transfers:
        start_d = MOD_DISPLAY.get(start, start)
        new_d   = MOD_DISPLAY.get(new,   new)
        rows.append({
            'Dataset':                 DATASET_NAMES[dataset].split(' ')[0],
            'Start(M_A)':              start_d,
            'Transfer(M_B)':           new_d,
            'DINO-SFT-B(M_A)':         _fmt(sft_b.get(start)),
            'DINO-SFT-L(M_A)':         _fmt(sft_l.get(start)),
            'KD-B':                    _fmt_meanstd(dist_b.get((start, new, 'kd'))),
            'KD-L':                    _fmt_meanstd(dist_l.get((start, new, 'kd'))),
            'TTM-B':                   _fmt_meanstd(dist_b.get((start, new, 'ttm'))),
            'TTM-L':                   _fmt_meanstd(dist_l.get((start, new, 'ttm'))),
            'Delulu-B':                _fmt_meanstd(del_b.get((start, new))),
            'Delulu-L':                _fmt_meanstd(del_l.get((start, new))),
            'DINO-SFT-B(M_B oracle)':  _fmt(sft_b.get(new)),
            'DINO-SFT-L(M_B oracle)':  _fmt(sft_l.get(new)),
            'Panopticon-B':            _fmt(rsfm.get(('panopticon',     new))),
            'OlmoEarth-B':             _fmt(rsfm.get(('olmoearth-base', new))),
            'OlmoEarth-L':             _fmt(rsfm.get(('olmoearth-large',new))),
        })
    return pd.DataFrame(rows)


def build_peek_BL(dataset):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    mm_b  = _load_mixmatch_peek(dataset, 'evan_base')
    mm_l  = _load_mixmatch_peek(dataset, 'evan_large')
    del_b = _load_delulu(dataset, 'evan_base',  'valchecked_val_peek', 'valchecked_peek')
    del_l = _load_delulu(dataset, 'evan_large', 'valchecked_val_peek', 'valchecked_peek')
    sft_b = _load_sft_dino(dataset, 'evan_base')
    sft_l = _load_sft_dino(dataset, 'evan_large')
    rsfm  = _load_rsfm(dataset)

    rows = []
    for (start, new) in transfers:
        start_d = MOD_DISPLAY.get(start, start)
        new_d   = MOD_DISPLAY.get(new,   new)
        rows.append({
            'Dataset':        DATASET_NAMES[dataset].split(' ')[0],
            'Start':          start_d,
            'Peeked':         new_d,
            'DINO-SFT-B':     _fmt(sft_b.get(start)),
            'DINO-SFT-L':     _fmt(sft_l.get(start)),
            'MixMatch-B':     _fmt_meanstd(mm_b.get(start)),
            'MixMatch-L':     _fmt_meanstd(mm_l.get(start)),
            'Delulu-B':       _fmt_meanstd(del_b.get((start, new))),
            'Delulu-L':       _fmt_meanstd(del_l.get((start, new))),
            'Panopticon-B':   _fmt(rsfm.get(('panopticon',     start))),
            'OlmoEarth-B':    _fmt(rsfm.get(('olmoearth-base', start))),
            'OlmoEarth-L':    _fmt(rsfm.get(('olmoearth-large',start))),
        })
    return pd.DataFrame(rows)


def build_addition_BL(dataset):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    mke_b     = _load_mke_addition(dataset, 'evan_base')
    mke_l     = _load_mke_addition(dataset, 'evan_large')
    del_b     = _load_delulu(dataset, 'evan_base',  'valchecked_val_add_ens', 'valchecked_add_ens')
    del_l     = _load_delulu(dataset, 'evan_large', 'valchecked_val_add_ens', 'valchecked_add_ens')
    sft_b     = _load_sft_dino(dataset, 'evan_base')
    sft_l     = _load_sft_dino(dataset, 'evan_large')
    comb_b    = _load_sft_combined_dino(dataset, 'evan_base')
    comb_l    = _load_sft_combined_dino(dataset, 'evan_large')
    rsfm      = _load_rsfm(dataset)

    rows = []
    for (start, new) in transfers:
        start_d = MOD_DISPLAY.get(start, start)
        new_d   = MOD_DISPLAY.get(new,   new)

        lsft_b = comb_b.get(f'{start}+{new}') or comb_b.get(f'{new}+{start}')
        lsft_l = comb_l.get(f'{start}+{new}') or comb_l.get(f'{new}+{start}')

        aliases = COMBINED_RSFM_ALIASES.get((start, new), [f'{start}+{new}', f'{new}+{start}'])
        pan_score = olmo_b = olmo_l = None
        for alias in aliases:
            if pan_score is None: pan_score = rsfm.get(('panopticon',      alias))
            if olmo_b    is None: olmo_b    = rsfm.get(('olmoearth-base',  alias))
            if olmo_l    is None: olmo_l    = rsfm.get(('olmoearth-large', alias))

        rows.append({
            'Dataset':                  DATASET_NAMES[dataset].split(' ')[0],
            'Start→New':                f'{start_d}→{new_d}',
            'DINO-SFT-B(M_A)':          _fmt(sft_b.get(start)),
            'DINO-SFT-L(M_A)':          _fmt(sft_l.get(start)),
            'MKE-B':                    _fmt(mke_b.get((start, new))),
            'MKE-L':                    _fmt(mke_l.get((start, new))),
            'Delulu-B':                 _fmt_meanstd(del_b.get((start, new))),
            'Delulu-L':                 _fmt_meanstd(del_l.get((start, new))),
            'DINO-SFT-B(M_A+M_B ora)':  _fmt(lsft_b),
            'DINO-SFT-L(M_A+M_B ora)':  _fmt(lsft_l),
            'Panopticon-B':             _fmt(pan_score),
            'OlmoEarth-B':              _fmt(olmo_b),
            'OlmoEarth-L':              _fmt(olmo_l),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# LaTeX helpers (self-contained subset of to_latex.py)
# ---------------------------------------------------------------------------

_MODALITY_DISPLAY = {
    's2-rgb':   r'$\mathrm{S2}_{rgb}$',
    's2-norgb': r'$\mathrm{S2}_{\neg rgb}$',
    's2':       r'$\mathrm{S2}$',
    's1':       r'$\mathrm{S1}$',
}


def _render_modality(s):
    return _MODALITY_DISPLAY.get(str(s).lower(), str(s))


def _escape(s):
    s = _render_modality(s)
    return s.replace('→', r'$\to$').replace('±', r'$\pm$')


def _bold(s):
    return r'\textbf{' + str(s) + '}'


def _multirow(n, s):
    return rf'\multirow{{{n}}}{{*}}{{{s}}}'


def _multicolumn(n, align, s):
    return rf'\multicolumn{{{n}}}{{{align}}}{{{s}}}'


def _num(s):
    if pd.isna(s) or str(s).strip() == '--':
        return float('nan')
    m = re.match(r'[-+]?\d*\.?\d+', str(s).strip())
    return float(m.group()) if m else float('nan')


def _bold_max_per_row(df, cols):
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(object)
    for i, row in df.iterrows():
        vals  = {c: _num(row[c]) for c in cols}
        valid = {c: v for c, v in vals.items() if not np.isnan(v)}
        if not valid:
            continue
        best_col = max(valid, key=valid.__getitem__)
        df.at[i, best_col] = _bold(_escape(row[best_col]))
    return df


def _apply_multirow(col_values):
    result = []
    i = 0
    while i < len(col_values):
        val = col_values[i]
        j = i + 1
        while j < len(col_values) and col_values[j] == val:
            j += 1
        n = j - i
        result.append(_multirow(n, _escape(val)) if n > 1 else _escape(val))
        result.extend([''] * (n - 1))
        i = j
    return result


def _df_to_latex_rows(df, merge_cols):
    df = df.copy()
    merged = {col: _apply_multirow(list(df[col])) for col in merge_cols}
    rows = []
    for i in range(len(df)):
        row = []
        for col in df.columns:
            if col in merged:
                row.append(merged[col][i])
            else:
                row.append(_escape(str(df.at[i, col])))
        rows.append(row)
    return rows


def _rows_to_tex(rows, col_spec, header_rows, midrule_after=None):
    lines = [r'\begin{tabular}{' + col_spec + '}', r'\toprule']
    lines += header_rows
    lines.append(r'\midrule')
    for i, row in enumerate(rows):
        lines.append('  ' + ' & '.join(row) + r' \\')
        if midrule_after and i in midrule_after:
            lines.append(r'  \midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def _midrule_after_datasets(df):
    col = list(df['Dataset'])
    s = set()
    for i in range(len(col) - 1):
        if col[i] != col[i + 1]:
            s.add(i)
    return s


# ---------------------------------------------------------------------------
# LaTeX table makers
# ---------------------------------------------------------------------------

def make_transfer_tex(df):
    baseline_cols = ['KD-B', 'KD-L', 'TTM-B', 'TTM-L', 'Delulu-B', 'Delulu-L']
    df = _bold_max_per_row(df, baseline_cols)
    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start(M_A)'])
    midrule = _midrule_after_datasets(df)

    # 3 id | 2 f0 | 4 baselines | 2 ours | 5 oracle
    col_spec = 'llr | rr | rrrr | rr | rrrrr'

    super_row = (
        _multicolumn(3, 'c', '') + ' & ' +
        _multicolumn(2, 'c|', r'$f_0(M_A)$') + ' & ' +
        _multicolumn(4, 'c|', 'Baselines') + ' & ' +
        _multicolumn(2, 'c|', 'Ours') + ' & ' +
        _multicolumn(5, 'c',  r'Oracle ($M_B$)') +
        r' \\'
    )
    DISPLAY = {
        'Dataset':                 'Dataset',
        'Start(M_A)':              r'\shortstack[c]{Start\\($M_A$)}',
        'Transfer(M_B)':           r'\shortstack[c]{Transfer\\($M_B$)}',
        'DINO-SFT-B(M_A)':         r'\shortstack[c]{DINO\\SFT (B)}',
        'DINO-SFT-L(M_A)':         r'\shortstack[c]{DINO\\SFT (L)}',
        'KD-B':                    r'\shortstack[c]{KD\\(B)}',
        'KD-L':                    r'\shortstack[c]{KD\\(L)}',
        'TTM-B':                   r'\shortstack[c]{TTM\\(B)}',
        'TTM-L':                   r'\shortstack[c]{TTM\\(L)}',
        'Delulu-B':                r'\shortstack[c]{Delulu\\(B)}',
        'Delulu-L':                r'\shortstack[c]{Delulu\\(L)}',
        'DINO-SFT-B(M_B oracle)':  r'\shortstack[c]{DINO\\SFT (B)}',
        'DINO-SFT-L(M_B oracle)':  r'\shortstack[c]{DINO\\SFT (L)}',
        'Panopticon-B':            r'\shortstack[c]{Panop.\\(B)}',
        'OlmoEarth-B':             r'\shortstack[c]{OlmoE.\\(B)}',
        'OlmoEarth-L':             r'\shortstack[c]{OlmoE.\\(L)}',
    }
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'
    return _rows_to_tex(rows, col_spec, [super_row, r'\midrule', col_row], midrule)


def make_peek_tex(df):
    baseline_cols = ['MixMatch-B', 'MixMatch-L', 'Delulu-B', 'Delulu-L']
    df = _bold_max_per_row(df, baseline_cols)
    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start'])
    midrule = _midrule_after_datasets(df)

    # 3 id | 2 f0 | 2 baselines | 2 ours | 3 oracle
    col_spec = 'llr | rr | rr | rr | rrr'

    super_row = (
        _multicolumn(3, 'c', '') + ' & ' +
        _multicolumn(2, 'c|', r'$f_0(M_A)$') + ' & ' +
        _multicolumn(2, 'c|', 'Baselines') + ' & ' +
        _multicolumn(2, 'c|', 'Ours') + ' & ' +
        _multicolumn(3, 'c',  r'Oracle ($M_A$)') +
        r' \\'
    )
    DISPLAY = {
        'Dataset':      'Dataset',
        'Start':        r'\shortstack[c]{Start\\($M_A$)}',
        'Peeked':       r'\shortstack[c]{Peeked\\($M_B$)}',
        'DINO-SFT-B':   r'\shortstack[c]{DINO\\SFT (B)}',
        'DINO-SFT-L':   r'\shortstack[c]{DINO\\SFT (L)}',
        'MixMatch-B':   r'\shortstack[c]{MixMatch\\(B)}',
        'MixMatch-L':   r'\shortstack[c]{MixMatch\\(L)}',
        'Delulu-B':     r'\shortstack[c]{Delulu\\(B)}',
        'Delulu-L':     r'\shortstack[c]{Delulu\\(L)}',
        'Panopticon-B': r'\shortstack[c]{Panop.\\(B)}',
        'OlmoEarth-B':  r'\shortstack[c]{OlmoE.\\(B)}',
        'OlmoEarth-L':  r'\shortstack[c]{OlmoE.\\(L)}',
    }
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'
    return _rows_to_tex(rows, col_spec, [super_row, r'\midrule', col_row], midrule)


def make_addition_tex(df):
    # Split Start→New into two id columns
    split = df['Start→New'].str.split('→', expand=True)
    df = df.copy()
    df.insert(df.columns.get_loc('Start→New'), 'Start(M_A)', split[0])
    df.insert(df.columns.get_loc('Start→New') + 1, 'New(M_B)', split[1])
    df = df.drop(columns=['Start→New'])

    baseline_cols = ['MKE-B', 'MKE-L', 'Delulu-B', 'Delulu-L']
    df = _bold_max_per_row(df, baseline_cols)
    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start(M_A)'])
    midrule = _midrule_after_datasets(df)

    # 3 id | 2 f0 | 2 baselines | 2 ours | 5 oracle
    col_spec = 'llr | rr | rr | rr | rrrrr'

    super_row = (
        _multicolumn(3, 'c', '') + ' & ' +
        _multicolumn(2, 'c|', r'$f_0(M_A)$') + ' & ' +
        _multicolumn(2, 'c|', 'Baselines') + ' & ' +
        _multicolumn(2, 'c|', 'Ours') + ' & ' +
        _multicolumn(5, 'c',  r'Oracle ($M_A$+$M_B$)') +
        r' \\'
    )
    DISPLAY = {
        'Dataset':                 'Dataset',
        'Start(M_A)':              r'\shortstack[c]{Start\\($M_A$)}',
        'New(M_B)':                r'\shortstack[c]{New\\($M_B$)}',
        'DINO-SFT-B(M_A)':         r'\shortstack[c]{DINO\\SFT (B)}',
        'DINO-SFT-L(M_A)':         r'\shortstack[c]{DINO\\SFT (L)}',
        'MKE-B':                   r'\shortstack[c]{MKE\\(B)}',
        'MKE-L':                   r'\shortstack[c]{MKE\\(L)}',
        'Delulu-B':                r'\shortstack[c]{Delulu\\(B)}',
        'Delulu-L':                r'\shortstack[c]{Delulu\\(L)}',
        'DINO-SFT-B(M_A+M_B ora)': r'\shortstack[c]{DINO\\SFT (B)}',
        'DINO-SFT-L(M_A+M_B ora)': r'\shortstack[c]{DINO\\SFT (L)}',
        'Panopticon-B':            r'\shortstack[c]{Panop.\\(B)}',
        'OlmoEarth-B':             r'\shortstack[c]{OlmoE.\\(B)}',
        'OlmoEarth-L':             r'\shortstack[c]{OlmoE.\\(L)}',
    }
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'
    return _rows_to_tex(rows, col_spec, [super_row, r'\midrule', col_row], midrule)


# ---------------------------------------------------------------------------
# Tall (pivoted) table makers — methods as rows, columns = dataset→start→new
# ---------------------------------------------------------------------------

# Each entry: (method_label, group_label, col_B, col_L)
# col_L=None means no L variant (e.g. Panopticon).
_TRANSFER_METHOD_ROWS = [
    (r'$f_0$',       r'$f_0(M_A)$', 'DINO-SFT-B(M_A)',        'DINO-SFT-L(M_A)'        ),
    ('KD',           'Baselines',   'KD-B',                    'KD-L'                   ),
    ('TTM',          'Baselines',   'TTM-B',                   'TTM-L'                  ),
    ('Delulu',       'Ours',        'Delulu-B',                'Delulu-L'               ),
    ('DINO SFT',     'Oracle',      'DINO-SFT-B(M_B oracle)',  'DINO-SFT-L(M_B oracle)' ),
    ('Panopticon',   'Oracle',      'Panopticon-B',             None                    ),
    ('OlmoEarth',    'Oracle',      'OlmoEarth-B',             'OlmoEarth-L'            ),
]

_PEEK_METHOD_ROWS = [
    (r'$f_0$',      r'$f_0(M_A)$', 'DINO-SFT-B',  'DINO-SFT-L' ),
    ('MixMatch',    'Baselines',   'MixMatch-B',  'MixMatch-L' ),
    ('Delulu',      'Ours',        'Delulu-B',    'Delulu-L'   ),
    ('Panopticon',  'Oracle',      'Panopticon-B', None        ),
    ('OlmoEarth',   'Oracle',      'OlmoEarth-B', 'OlmoEarth-L'),
]

_ADDITION_METHOD_ROWS = [
    (r'$f_0$',      r'$f_0(M_A)$', 'DINO-SFT-B(M_A)',         'DINO-SFT-L(M_A)'        ),
    ('MKE',         'Baselines',   'MKE-B',                   'MKE-L'                  ),
    ('Delulu',      'Ours',        'Delulu-B',                'Delulu-L'               ),
    ('DINO SFT',    'Oracle',      'DINO-SFT-B(M_A+M_B ora)', 'DINO-SFT-L(M_A+M_B ora)'),
    ('Panopticon',  'Oracle',      'Panopticon-B',             None                    ),
    ('OlmoEarth',   'Oracle',      'OlmoEarth-B',             'OlmoEarth-L'            ),
]


def _make_tall_tex(df, dataset_col, start_col, new_col, method_rows):
    """Single combined tall table: (group, method, B/L) as rows, (dataset, start, new) as columns.

    Header has 3 levels: dataset → start_mod → new_mod (one value per cell).
    Body has 3 id columns: group (multirow), method (multirow over B+L), size (B/L).
    Methods with col_L=None emit only a B row.
    """
    keys = list(df[[dataset_col, start_col, new_col]].itertuples(index=False, name=None))
    n = len(keys)

    # --- column spec: 3 id cols | one 'r' per transfer, grouped by dataset with | ---
    ds_groups = []
    for ds, start, new in keys:
        if not ds_groups or ds_groups[-1][0] != ds:
            ds_groups.append((ds, 0))
        ds_groups[-1] = (ds_groups[-1][0], ds_groups[-1][1] + 1)

    col_spec = 'lll | ' + ' | '.join(' '.join(['r'] * cnt) for _, cnt in ds_groups)

    # --- header row 1: dataset spans ---
    ds_spans = {}
    for ds, start, new in keys:
        ds_spans[ds] = ds_spans.get(ds, 0) + 1
    h1_cells = [_multicolumn(3, 'c', '')]
    for i, (ds, span) in enumerate(ds_spans.items()):
        align = 'c|' if i < len(ds_spans) - 1 else 'c'
        h1_cells.append(_multicolumn(span, align, _escape(ds)))
    header1 = ' & '.join(h1_cells) + r' \\'

    # --- header row 2: start_mod spans (consecutive runs within each dataset) ---
    start_runs = []
    for ds, start, new in keys:
        if not start_runs or (start_runs[-1][0], start_runs[-1][1]) != (ds, start):
            start_runs.append([ds, start, 0])
        start_runs[-1][2] += 1

    h2_cells = [_multicolumn(3, 'c', '')]
    for i, (ds, start, span) in enumerate(start_runs):
        is_last_in_ds   = (i == len(start_runs) - 1) or (start_runs[i + 1][0] != ds)
        is_last_overall = (i == len(start_runs) - 1)
        align = 'c' if is_last_overall else ('c|' if is_last_in_ds else 'c')
        h2_cells.append(_multicolumn(span, align, _escape(start)))
    header2 = ' & '.join(h2_cells) + r' \\'

    # --- header row 3: new_mod leaf labels ---
    h3_cells = [r'\shortstack[c]{Method\\Group}', 'Method', 'Size']
    for i, (ds, start, new) in enumerate(keys):
        h3_cells.append(_escape(new))
    header3 = ' & '.join(h3_cells) + r' \\'

    # --- body: two sub-rows per method (B and L), one row if col_L is None ---
    # flatten to (group, method, size_label, col) tuples
    flat_rows = []
    for method_label, group, col_b, col_l in method_rows:
        flat_rows.append((group, method_label, '(B)', col_b))
        if col_l is not None:
            flat_rows.append((group, method_label, '(L)', col_l))

    group_col  = [g   for g, _, _, _ in flat_rows]
    method_col = [m   for _, m, _, _ in flat_rows]
    group_rendered  = _apply_multirow(group_col)
    method_rendered = _apply_multirow(method_col)

    lines = [r'\begin{tabular}{' + col_spec + '}', r'\toprule']
    lines.append('  ' + header1)
    lines.append('  ' + header2)
    lines.append('  ' + header3)
    lines.append(r'  \midrule')
    prev_group = None
    for i, (group, method, size, col) in enumerate(flat_rows):
        if prev_group is not None and group != prev_group:
            lines.append(r'  \midrule')
        cells = []
        for ds, start, new in keys:
            row = df[(df[dataset_col] == ds) & (df[start_col] == start) & (df[new_col] == new)]
            cells.append(_escape(str(row[col].iloc[0])) if not row.empty and col in df.columns else '--')
        line = group_rendered[i] + ' & ' + method_rendered[i] + ' & ' + size + ' & ' + ' & '.join(cells) + r' \\'
        lines.append('  ' + line)
        prev_group = group
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def make_transfer_tex_tall(df):
    return _make_tall_tex(df, 'Dataset', 'Start(M_A)', 'Transfer(M_B)', _TRANSFER_METHOD_ROWS)


def make_peek_tex_tall(df):
    return _make_tall_tex(df, 'Dataset', 'Start', 'Peeked', _PEEK_METHOD_ROWS)


def make_addition_tex_tall(df):
    split = df['Start→New'].str.split('→', expand=True)
    df = df.copy()
    df['_start'] = split[0]
    df['_new']   = split[1]
    return _make_tall_tex(df, 'Dataset', '_start', '_new', _ADDITION_METHOD_ROWS)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tall', action='store_true',
                        help='Pivot: methods as rows, transfers as columns (one table per dataset)')
    args = parser.parse_args()

    # ---- collect dataframes ----
    transfer_frames, peek_frames, addition_frames = [], [], []
    for dataset in DATASETS:
        df = build_transfer_BL(dataset)
        if df is not None and not df.empty:
            transfer_frames.append(df)
        df = build_peek_BL(dataset)
        if df is not None and not df.empty:
            peek_frames.append(df)
        df = build_addition_BL(dataset)
        if df is not None and not df.empty:
            addition_frames.append(df)

    # ---- print tables ----
    for label, frames in [('TRANSFER', transfer_frames), ('PEEK', peek_frames), ('ADDITION', addition_frames)]:
        print(f'\n{"="*80}\n  {label}\n{"="*80}')
        if frames:
            print(pd.concat(frames, ignore_index=True).to_string(index=False))
        else:
            print('  (no data)')

    # ---- write single .tex ----
    os.makedirs(OUT_DIR, exist_ok=True)

    if args.tall:
        wide_fns  = [make_transfer_tex_tall, make_peek_tex_tall, make_addition_tex_tall]
        out = f'{OUT_DIR}/BL_tables_tall.tex'
    else:
        wide_fns  = [make_transfer_tex, make_peek_tex, make_addition_tex]
        out = f'{OUT_DIR}/BL_tables.tex'

    sections = []
    for label, frames, fn in zip(
        ['Transfer', 'Peek', 'Addition'],
        [transfer_frames, peek_frames, addition_frames],
        wide_fns,
    ):
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        sections.append(f'% Table: {label}\n' + fn(combined))

    with open(out, 'w') as f:
        f.write('\n\n\\bigskip\n\n'.join(sections) + '\n')
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()

# python res/results_BL.py
