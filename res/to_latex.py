"""Convert saved CSVs to LaTeX tables for the paper.

Run from repo root after results_table.py:
    python res/to_latex.py

Outputs: res/latex/table_b.tex  table_c.tex  table_d.tex
"""

import os
import re
import numpy as np
import pandas as pd

OUT_DIR = 'res/latex'

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _num(s):
    """Extract leading float from a cell like '49.75' or '48.96±1.32'. Returns nan if '--'."""
    if pd.isna(s) or str(s).strip() == '--':
        return float('nan')
    m = re.match(r'[-+]?\d*\.?\d+', str(s).strip())
    return float(m.group()) if m else float('nan')


def _bold(s):
    return r'\textbf{' + str(s) + '}'


def _escape(s):
    """Minimal LaTeX escaping for cell content."""
    return str(s).replace('→', r'$\to$').replace('±', r'$\pm$')


def _multirow(n, s):
    return rf'\multirow{{{n}}}{{*}}{{{s}}}'


def _multicolumn(n, align, s):
    return rf'\multicolumn{{{n}}}{{{align}}}{{{s}}}'


def _bold_max_per_row(df, cols):
    # Return a copy of df with the maximum numeric value in cols bolded per row.
    # Convert affected columns to object dtype first so strings can be assigned.
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(object)
    for i, row in df.iterrows():
        vals = {c: _num(row[c]) for c in cols}
        valid = {c: v for c, v in vals.items() if not np.isnan(v)}
        if not valid:
            continue
        best_col = max(valid, key=valid.__getitem__)
        df.at[i, best_col] = _bold(_escape(row[best_col]))
    return df


def _apply_multirow(col_values):
    # Replace repeated consecutive values with \multirow{n}{*}{val} + empty strings.
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
    """Convert df to list-of-lists of LaTeX cell strings, with multirow for merge_cols."""
    df = df.copy()
    merged = {}
    for col in merge_cols:
        merged[col] = _apply_multirow(list(df[col]))

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
    # Assemble full tabular. midrule_after: set of row indices after which to insert \midrule.
    lines = [r'\begin{tabular}{' + col_spec + '}', r'\toprule']
    lines += header_rows
    lines.append(r'\midrule')
    for i, row in enumerate(rows):
        lines.append('  ' + ' & '.join(row) + r' \\')
        if midrule_after and i in midrule_after:
            lines.append(r'  \midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Table B
# ---------------------------------------------------------------------------

def make_table_b():
    df = pd.read_csv(f'{OUT_DIR}/transfer.csv')

    # Drop rand-ST columns — keep only DINO context + baselines + oracle
    drop_cols = ['rand-ST(M_A)', 'rand-ST(M_B oracle)']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Structure after drop:
    #   id:         Dataset | Start(M_A) | Transfer(M_B)
    #   M_A ctx:    DINO-SFT (M_A)
    #   baselines:  KD(M_B) | TTM(M_B)
    #   ours:       Delulu(M_B)
    #   M_B oracle: DINO-SFT(M_B oracle) | Panopticon(M_B oracle) | OlmoEarth(M_B oracle)

    baseline_cols = ['KD(M_B)', 'TTM(M_B)', 'Delulu(M_B)']

    df = _bold_max_per_row(df, baseline_cols)
    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start(M_A)'])

    dataset_col = list(df['Dataset'])
    midrule_after = set()
    for i in range(len(dataset_col) - 1):
        if dataset_col[i] != dataset_col[i + 1]:
            midrule_after.add(i)

    # col spec: 3 id | 1 M_A | vline | 2 baselines | 1 ours | vline | 3 oracle
    col_spec = 'llr | r | rrr | rrr'

    th = r'\thead'
    n_id, n_ma, n_base, n_ours, n_ora = 3, 1, 2, 1, 3
    super_row = (
        _multicolumn(n_id, 'c', '') + ' & ' +
        _multicolumn(n_ma, 'c|', r'\shortstack[c]{$M_A$\\context}') + ' & ' +
        _multicolumn(n_base, 'c|', 'Baselines') + ' & ' +
        _multicolumn(n_ours, 'c|', 'Ours') + ' & ' +
        _multicolumn(n_ora, 'c', r'\shortstack[c]{$M_B$\\oracle}') +
        r' \\'
    )

    DISPLAY = {
        'Dataset':               'Dataset',
        'Start(M_A)':            r'\shortstack[c]{Start\\($M_A$)}',
        'Transfer(M_B)':         r'\shortstack[c]{Transfer\\($M_B$)}',
        'DINO-SFT (M_A)':        r'\shortstack[c]{DINO\\SFT}',
        'KD(M_B)':               'KD',
        'TTM(M_B)':              'TTM',
        'Delulu(M_B)':           'Delulu',
        'DINO-SFT(M_B oracle)':  r'\shortstack[c]{DINO\\SFT}',
        'Panopticon(M_B oracle)': 'Panopticon',
        'OlmoEarth(M_B oracle)': 'OlmoEarth',
    }
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'

    header_rows = [super_row, r'\midrule', col_row]
    return _rows_to_tex(rows, col_spec, header_rows, midrule_after)


# ---------------------------------------------------------------------------
# Table C
# ---------------------------------------------------------------------------

def make_table_c():
    df = pd.read_csv(f'{OUT_DIR}/peek.csv')

    # Baselines to bold max over per row: MixMatch, Delulu (Panopticon/OlmoEarth are oracles)
    baseline_cols = ['MixMatch', 'Delulu']
    oracle_cols   = ['Panopticon', 'OlmoEarth']

    df = _bold_max_per_row(df, baseline_cols)

    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start'])

    dataset_col = list(df['Dataset'])
    midrule_after = set()
    for i in range(len(dataset_col) - 1):
        if dataset_col[i] != dataset_col[i + 1]:
            midrule_after.add(i)

    # col spec: 3 id | vline | 2 baseline | vline | 1 ours | vline | 2 oracle
    col_spec = 'llr | r | rr | rr'

    n_id = 3
    super_row = (
        _multicolumn(n_id, 'c', '') + ' & ' +
        _multicolumn(1, 'c|', r'$M_A$ context') + ' & ' +
        _multicolumn(1, 'c|', 'Baseline') + ' & ' +
        _multicolumn(1, 'c|', 'Ours') + ' & ' +
        _multicolumn(2, 'c', r'$M_A$ oracle') +
        r' \\'
    )

    DISPLAY = {
        'Dataset':    'Dataset',
        'Start':      r'\shortstack[c]{Start\\($M_A$)}',
        'Peeked':     r'\shortstack[c]{Peeked\\($M_B$)}',
        'DINO SFT':   r'\shortstack[c]{DINO\\SFT}',
        'MixMatch':   'MixMatch',
        'Delulu':     'Delulu',
        'Panopticon': 'Panopticon',
        'OlmoEarth':  'OlmoEarth',
    }
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'

    header_rows = [super_row, r'\midrule', col_row]
    return _rows_to_tex(rows, col_spec, header_rows, midrule_after)


# ---------------------------------------------------------------------------
# Table D
# ---------------------------------------------------------------------------

def make_table_d():
    df = pd.read_csv(f'{OUT_DIR}/addition.csv')

    # Split 'Start→New' into separate Start and New columns
    split = df['Start→New'].str.split('→', expand=True)
    df.insert(df.columns.get_loc('Start→New'), 'Start(M_A)', split[0])
    df.insert(df.columns.get_loc('Start→New') + 1, 'New(M_B)', split[1])
    df = df.drop(columns=['Start→New'])

    baseline_cols = ['MKE', 'Delulu']

    df = _bold_max_per_row(df, baseline_cols)
    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start(M_A)'])

    dataset_col = list(df['Dataset'])
    midrule_after = set()
    for i in range(len(dataset_col) - 1):
        if dataset_col[i] != dataset_col[i + 1]:
            midrule_after.add(i)

    # col spec: 3 id | 1 f0(M_A) | vline | 1 baseline | 1 ours | vline | 3 oracle
    col_spec = 'llr | r | rr | rrr'

    super_row = (
        _multicolumn(3, 'c', '') + ' & ' +
        _multicolumn(1, 'c|', r'$f_0(M_A)$') + ' & ' +
        _multicolumn(1, 'c|', 'Baseline') + ' & ' +
        _multicolumn(1, 'c|', 'Ours') + ' & ' +
        _multicolumn(3, 'c', r'$M_A+M_B$ oracle') +
        r' \\'
    )

    DISPLAY = {
        'Dataset':              'Dataset',
        'Start(M_A)':           r'\shortstack[c]{Start\\($M_A$)}',
        'New(M_B)':             r'\shortstack[c]{New\\($M_B$)}',
        'DINO-SFT(M_A)':        r'\shortstack[c]{DINO\\SFT}',
        'MKE':                  'MKE',
        'Delulu':               'Delulu',
        'DINO(M_A+M_B oracle)': r'\shortstack[c]{DINO\\SFT}',
        'Panopticon':           'Panopticon',
        'OlmoEarth':            'OlmoEarth',
    }
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'

    header_rows = [super_row, r'\midrule', col_row]
    return _rows_to_tex(rows, col_spec, header_rows, midrule_after)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    for name, fn in [('transfer', make_table_b), ('peek', make_table_c), ('addition', make_table_d)]:
        src = f'{OUT_DIR}/{name}.csv'  # name is already transfer/peek/addition
        if not os.path.isfile(src):
            print(f'  {name}.csv not found — run results_table.py first')
            continue
        tex = fn()
        out = f'{OUT_DIR}/{name}.tex'
        with open(out, 'w') as f:
            f.write(tex + '\n')
        print(f'  wrote {out}')


if __name__ == '__main__':
    main()

# python res/to_latex.py
