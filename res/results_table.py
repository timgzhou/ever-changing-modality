"""Unified per-dataset result tables.

Table 1 (SFT): rows = method × init × model size, cols = modality
Table 2 (Distillation/MKE): rows = method × model, cols = teacher_mod → student_mod

Run from repo root:
    python res/results_table.py
"""

import os
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# evan model_type → display label
EVAN_MODEL_LABELS = {
    'evan_small': 'ViT-S',
    'evan_base':  'ViT-B',
    'evan_large': 'ViT-L',
}

# RSFM model string → (init_label, model_label)
# model strings in rsfm_results.csv: panopticon, olmoearth, olmoearth-large,
# dinov3-vitb-lvd, dinov3-vitl-lvd, dinov3-vitl-sat, etc.
def _rsfm_model_info(model_str):
    """Return (init_group, model_label) matching ROW_ORDER init_group names."""
    m = model_str.lower()
    if 'panopticon' in m:
        return 'Panopticon', 'Panopticon-B'
    if 'olmoearth' in m:
        size = 'OlmoEarth-L' if 'large' in m else 'OlmoEarth-B'
        return 'Olmoearth', size
    if 'dinov3' in m or 'dino' in m:
        if 'vitl' in m:
            variant = 'SAT' if 'sat' in m else 'LVD'
            return 'DINO v3', f'DINO-L {variant}'
        if 'vitb' in m:
            variant = 'SAT' if 'sat' in m else 'LVD'
            return 'DINO v3', f'DINO-B {variant}'
    return 'Pretrained', model_str


# Per-dataset: display column name → modality aliases in CSVs
COLUMNS = {
    'benv2': {
        'S2-RGB': ['s2_rgb'],
        'S2':     ['s2'],
        'S1':     ['s1'],
        'S2+S1':  ['s2+s1', 's1+s2', 's2s1', 's1s2'],
    },
    'dfc2020': {
        'S2-RGB': ['s2_rgb'],
        'S2':     ['s2'],
        'S1':     ['s1'],
        'S2+S1':  ['s2+s1', 's1+s2', 's2s1', 's1s2'],
    },
    'eurosat': {
        'RGB':  ['rgb'],
        'VRE':  ['vre'],
        'SWIR': ['swir'],
        'S2':   ['s2'],
    },
}

DATASETS = list(COLUMNS.keys())

DATASET_NAMES = {
    'benv2':   'reBEN (Multi Label Classification)',
    'dfc2020': 'DFC2020 (Semantic Segmentation)',
    'eurosat': 'EuroSAT (Classification)',
}

# Row order: (section, init_group, model_label)
# init_group is the merged row-group label (printed only on first row of group).
# The grouping matches the paper table: Random init / DINO v3 / Panopticon / Olmoearth.
ROW_ORDER = [
    ('SFT', 'Random init', 'ViT-S'),
    ('SFT', 'Random init', 'ViT-B'),
    ('SFT', 'Random init', 'ViT-L'),
    ('SFT', 'DINO v3',    'ViT-S'),
    ('SFT', 'DINO v3',    'ViT-B'),
    ('SFT', 'DINO v3',    'ViT-L'),
    ('SFT', 'Panopticon', 'Panopticon-B'),
    ('SFT', 'Olmoearth',  'OlmoEarth-B'),
    ('SFT', 'Olmoearth',  'OlmoEarth-L'),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _best(df, group_cols, score_col):
    return (
        df.sort_values(score_col, ascending=False)
        .groupby(group_cols, as_index=False)
        .first()
    )


def _top3(df, group_cols, score_col):
    """Return top-3 rows per group by score_col (descending)."""
    return (
        df.sort_values(score_col, ascending=False)
        .groupby(group_cols, as_index=False)
        .head(3)
    )


def _params_str(params):
    """Format trainable_params as e.g. '87M'."""
    try:
        p = int(params)
        if p >= 1_000_000:
            return f'{round(p / 1_000_000)}M'
        if p >= 1_000:
            return f'{round(p / 1_000)}K'
        return str(p)
    except (TypeError, ValueError):
        return ''


# ---------------------------------------------------------------------------
# Loaders — each returns a list of dicts:
#   {dataset, col_aliases, method, init, model_label, params, score}
# ---------------------------------------------------------------------------

def load_sft():
    records = []
    for dataset in DATASETS:
        df = _read_csv(f'res/train_sft/{dataset}.csv')
        if df is None:
            continue
        select_col = 'val_metric' if 'val_metric' in df.columns else 'test_metric'
        top3 = _top3(df, ['modality', 'model_type', 'dino_init'], select_col)
        for (modality, model_type, dino_init), grp in top3.groupby(['modality', 'model_type', 'dino_init']):
            model_label = EVAN_MODEL_LABELS.get(model_type, model_type)
            init_group  = 'DINO v3' if dino_init else 'Random init'
            records.append(dict(
                dataset=dataset,
                col_aliases=[modality],
                init_group=init_group, model_label=model_label,
                params=grp['trainable_params'].iloc[0],
                scores=grp['test_metric'].tolist(),
            ))
    return records


def load_rsfm():
    df = _read_csv('res/rsfm/rsfm_results.csv')
    if df is None:
        return []
    df = df[df['train_mode'] == 'fft']
    # Exclude DINO v3 models — those come from train_sft instead
    df = df[~df['model'].str.lower().str.contains('dino')]
    # Select top-3 checkpoints by val_metric (HP selection), report their test_metrics
    select_col = 'val_metric' if 'val_metric' in df.columns else 'test_metric'
    top3 = _top3(df, ['dataset', 'modality', 'model'], select_col)
    records = []
    for (dataset, modality, model), grp in top3.groupby(['dataset', 'modality', 'model']):
        if dataset not in COLUMNS:
            continue
        init_group, model_label = _rsfm_model_info(model)
        records.append(dict(
            dataset=dataset,
            col_aliases=[modality],
            init_group=init_group, model_label=model_label,
            params=grp['trainable_params'].iloc[0],
            scores=grp['test_metric'].tolist(),
        ))
    return records


# ---------------------------------------------------------------------------
# Transfer table: teacher mod → student mod sub-columns
# Each entry: (teacher_display, student_display, student_aliases)
# Distillation student = single other modality
# MKE student = strict superset (teacher + others), aliases are '+'-joined sorted sets
# ---------------------------------------------------------------------------

TRANSFER_COLS = {
    'benv2': {
        # teacher_display → list of (student_display, [csv_aliases])
        'S2-RGB': [
            ('S2',    ['s2']),
            ('S1',    ['s1']),
            ('+S2',   ['s2_rgb+s2', 's2+s2_rgb']),
            ('+S1',   ['s2_rgb+s1', 's1+s2_rgb']),
            ('S2+S1', ['s2+s1', 's1+s2']),
            ('All',   ['s2_rgb+s2+s1', 's2+s2_rgb+s1', 's1+s2+s2_rgb']),
        ],
        'S2': [
            ('S1',   ['s1']),
            ('+S1',  ['s2+s1', 's1+s2']),
            ('All',  ['s2+s2_rgb+s1', 's2+s1+s2_rgb']),
        ],
        'S1': [
            ('S2',   ['s2']),
            ('+S2',  ['s1+s2', 's2+s1']),
            ('All',  ['s1+s2_rgb+s2', 's1+s2+s2_rgb']),
        ],
    },
    'dfc2020': {
        'S2-RGB': [
            ('S2',    ['s2']),
            ('S1',    ['s1']),
            ('+S2',   ['s2_rgb+s2', 's2+s2_rgb']),
            ('+S1',   ['s2_rgb+s1', 's1+s2_rgb']),
            ('S2+S1', ['s2+s1', 's1+s2']),
            ('All',   ['s2_rgb+s2+s1', 's2+s2_rgb+s1']),
        ],
        'S2': [
            ('S1',   ['s1']),
            ('+S1',  ['s2+s1', 's1+s2']),
            ('All',  ['s2+s2_rgb+s1', 's2+s1+s2_rgb']),
        ],
        'S1': [
            ('S2',   ['s2']),
            ('+S2',  ['s1+s2', 's2+s1']),
            ('All',  ['s1+s2_rgb+s2', 's1+s2+s2_rgb']),
        ],
    },
    'eurosat': {
        'RGB': [
            ('VRE',      ['vre']),
            ('NIR',      ['nir']),
            ('SWIR',     ['swir']),
            ('+VRE',     ['rgb+vre', 'vre+rgb']),
            ('+NIR',     ['rgb+nir', 'nir+rgb']),
            ('+SWIR',    ['rgb+swir', 'swir+rgb']),
            ('+VRE+NIR', ['rgb+vre+nir']),
            ('All',      ['rgb+vre+nir+swir']),
        ],
        'VRE': [
            ('RGB',      ['rgb']),
            ('NIR',      ['nir']),
            ('SWIR',     ['swir']),
            ('+RGB',     ['vre+rgb', 'rgb+vre']),
            ('+NIR',     ['vre+nir', 'nir+vre']),
            ('+SWIR',    ['vre+swir', 'swir+vre']),
            ('All',      ['vre+rgb+nir+swir']),
        ],
        'NIR': [
            ('RGB',      ['rgb']),
            ('VRE',      ['vre']),
            ('SWIR',     ['swir']),
            ('+RGB',     ['nir+rgb', 'rgb+nir']),
            ('+VRE',     ['nir+vre', 'vre+nir']),
            ('+SWIR',    ['nir+swir', 'swir+nir']),
            ('All',      ['nir+rgb+vre+swir']),
        ],
        'SWIR': [
            ('RGB',      ['rgb']),
            ('VRE',      ['vre']),
            ('NIR',      ['nir']),
            ('+RGB',     ['swir+rgb', 'rgb+swir']),
            ('+VRE',     ['swir+vre', 'vre+swir']),
            ('+NIR',     ['swir+nir', 'nir+swir']),
            ('All',      ['swir+rgb+vre+nir']),
        ],
    },
}

TRANSFER_ROW_ORDER = [
    ('Distillation', 'ViT-S'),
    ('Distillation', 'ViT-B'),
    ('Distillation', 'ViT-L'),
    ('MKE',          'ViT-S'),
    ('MKE',          'ViT-B'),
    ('MKE',          'ViT-L'),
]


# ---------------------------------------------------------------------------
# Loaders for transfer table
# ---------------------------------------------------------------------------

def _norm_modset(s):
    """Canonicalise a '+'-joined modality string: sort parts, lower, strip."""
    return '+'.join(sorted(p.strip().lower() for p in s.split('+')))


def load_distillation(dataset):
    """
    Returns list of dicts:
      {model_label, teacher_aliases, student_aliases, score}
    teacher_aliases / student_aliases: lists of canonical modality strings
    """
    records = []
    base = f'res/baselines/distillation/{dataset}'
    if not os.path.isdir(base):
        return records

    # benv2 has per-model subdirs; dfc2020 does not
    subdirs = []
    for entry in os.listdir(base):
        full = os.path.join(base, entry)
        if os.path.isdir(full):
            subdirs.append((entry, full))   # entry = model name or whatever
    if not subdirs:
        subdirs = [('', base)]

    for subdir_name, subdir_path in subdirs:
        model_label = EVAN_MODEL_LABELS.get(subdir_name, subdir_name if subdir_name else None)
        for fname in os.listdir(subdir_path):
            if not fname.endswith('.csv'):
                continue
            df = _read_csv(os.path.join(subdir_path, fname))
            if df is None:
                continue
            # Use best_test_metric(oracle) if present, else test_metric
            score_col = 'best_test_metric(oracle)' if 'best_test_metric(oracle)' in df.columns else 'test_metric'
            best = _best(df, ['model_type', 'teacher_modality', 'student_modality'], score_col)
            for _, row in best.iterrows():
                ml = EVAN_MODEL_LABELS.get(row['model_type'], row['model_type'])
                if model_label and ml != model_label:
                    continue  # subdir model doesn't match row
                records.append(dict(
                    model_label=ml,
                    teacher_aliases=[_norm_modset(row['teacher_modality'])],
                    student_aliases=[_norm_modset(row['student_modality'])],
                    score=row[score_col],
                ))
    return records


def load_mke(dataset):
    """
    Returns list of dicts:
      {model_label, teacher_aliases, student_aliases, score}
    student_aliases includes the teacher (strict superset).
    """
    path = f'res/baselines/mke/{dataset}.csv'
    df = _read_csv(path)
    if df is None:
        return []
    score_col = 'student_best_test_metric'
    best = _best(df, ['model_type', 'teacher_modality', 'student_modalities'], score_col)
    records = []
    for _, row in best.iterrows():
        ml = EVAN_MODEL_LABELS.get(row['model_type'], row['model_type'])
        # student_modalities stored as e.g. "s2+s1" or "s2 s1" — normalise
        raw_stud = row['student_modalities'].replace(' ', '+')
        records.append(dict(
            model_label=ml,
            teacher_aliases=[_norm_modset(row['teacher_modality'])],
            student_aliases=[_norm_modset(raw_stud)],
            score=row[score_col],
        ))
    return records


# ---------------------------------------------------------------------------
# Build and print transfer table
# ---------------------------------------------------------------------------

def build_transfer_table(dataset, distill_records, mke_records):
    teacher_defs = TRANSFER_COLS.get(dataset, {})
    if not teacher_defs:
        return None

    # Build flat column list: (teacher_display, student_display)
    col_list = []
    for t_disp, student_cols in teacher_defs.items():
        for s_disp, _ in student_cols:
            col_list.append((t_disp, s_disp))

    def _lookup(records, method_label):
        # score_map: (method, model, teacher_display, student_display) → score
        score_map = {}
        for rec in records:
            ml = rec['model_label']
            for t_disp, student_cols in teacher_defs.items():
                t_aliases_display = COLUMNS[dataset].get(t_disp, [t_disp.lower()])
                t_aliases_norm = [_norm_modset(a) for a in t_aliases_display]
                if not any(a in rec['teacher_aliases'] for a in t_aliases_norm):
                    continue
                for s_disp, s_aliases in student_cols:
                    s_aliases_norm = [_norm_modset(a) for a in s_aliases]
                    if not any(a in rec['student_aliases'] for a in s_aliases_norm):
                        continue
                    key = (method_label, ml, t_disp, s_disp)
                    prev = score_map.get(key)
                    if prev is None or rec['score'] > prev:
                        score_map[key] = rec['score']
        return score_map

    distill_map = _lookup(distill_records, 'Distillation')
    mke_map     = _lookup(mke_records,     'MKE')

    # MKE only valid for strict superset student mods (student contains '→' and teacher)
    # Already enforced by data, but also by the TRANSFER_COLS structure (supersets listed)

    # Use unique internal keys to avoid collisions (same student label under different teachers),
    # then rename to student-only display labels for printing.
    col_keys = []      # internal df column names
    col_display = []   # display labels (student only)
    for t_disp, student_cols in teacher_defs.items():
        for s_disp, _ in student_cols:
            col_keys.append(f'{t_disp}||{s_disp}')
            col_display.append(s_disp)

    rows = []
    prev_method = None
    for method, model_label in TRANSFER_ROW_ORDER:
        score_map = distill_map if method == 'Distillation' else mke_map
        row_vals = {(t, s): score_map.get((method, model_label, t, s)) for t, s in col_list}
        if all(v is None for v in row_vals.values()):
            continue
        row = {
            'Method': method if method != prev_method else '',
            'Model':  model_label,
        }
        for (t_disp, s_disp), ck in zip(col_list, col_keys):
            v = row_vals[(t_disp, s_disp)]
            row[ck] = f'{v:.2f}' if v is not None else ''
        rows.append(row)
        prev_method = method

    if not rows:
        return None

    df = pd.DataFrame(rows)
    # Rename internal keys to student-only display labels
    rename = {'Method': 'Method', 'Model': 'Model'}
    rename.update(dict(zip(col_keys, col_display)))
    df = df.rename(columns=rename)
    # col_keys / col_display are returned so the banner knows group widths
    return df, col_keys, col_display


def _print_transfer_table(dataset, df, col_keys, col_display):
    """Print teacher-group banner (centered, with | separators) then the data table."""
    teacher_defs = TRANSFER_COLS[dataset]
    table_str = df.to_string(index=False)
    lines = table_str.split('\n')
    header_row = lines[0]

    # Find the start position of the first student column in the header
    first_col = col_display[0]
    prefix_len = header_row.index(first_col)
    prefix = ' ' * prefix_len

    # For each teacher group, find the span [start, end) in the header string.
    # We scan left-to-right through col_display to find each label's position
    # strictly after the previous one, avoiding false matches for repeated names.
    idx = 0
    groups = []
    search_from = prefix_len
    for t_disp, student_cols in teacher_defs.items():
        n = len(student_cols)
        sub_labels = col_display[idx:idx + n]
        start = header_row.index(sub_labels[0], search_from)
        pos = start
        for lbl in sub_labels:
            pos = header_row.index(lbl, pos)
            pos += len(lbl)
        end = pos
        groups.append((t_disp, start, end))
        search_from = end
        idx += n

    # Build banner line: centered teacher label within block, | between groups
    banner = prefix
    for i, (t_disp, start, end) in enumerate(groups):
        block_w = end - start
        label = f' {t_disp} '
        centered = label.center(block_w)
        if i < len(groups) - 1:
            banner += centered + '|'
        else:
            banner += centered

    # Build separator line (dashes under each block, | between)
    sep = ' ' * prefix_len
    for i, (_, start, end) in enumerate(groups):
        block_w = end - start
        if i < len(groups) - 1:
            sep += '-' * block_w + '+'
        else:
            sep += '-' * block_w

    # Insert | separators into data rows at group boundaries
    def _insert_seps(line):
        out = line
        offset = 0
        for i, (_, start, end) in enumerate(groups[:-1]):
            pos = end + offset
            out = out[:pos] + '|' + out[pos:]
            offset += 1
        return out

    print(banner)
    print(sep)
    for line in lines:
        print(_insert_seps(line))


# ---------------------------------------------------------------------------
# Build and print tables
# ---------------------------------------------------------------------------

def _fmt_scores(scores):
    """Format a list of test scores as 'mean±std' (1 decimal place)."""
    import numpy as np
    if not scores:
        return ''
    if len(scores) == 1:
        return f'{scores[0]:.1f}'
    return f'{float(np.mean(scores)):.1f}±{float(np.std(scores)):.1f}'


def build_table(dataset, all_records):
    col_defs  = COLUMNS[dataset]
    col_names = list(col_defs.keys())

    # lookup: (init_group, model_label, col_name) → list of test scores
    scores_lookup = {}
    params_lookup = {}

    for rec in all_records:
        if rec['dataset'] != dataset:
            continue
        key = (rec['init_group'], rec['model_label'])
        for col_name, aliases in col_defs.items():
            if any(a in rec['col_aliases'] for a in aliases):
                lk = (*key, col_name)
                scores_lookup.setdefault(lk, []).extend(rec['scores'])
                params_lookup[key] = rec.get('params')

    rows = []
    prev_section    = None
    prev_init_group = None
    for section, init_group, model_label in ROW_ORDER:
        key      = (init_group, model_label)
        row_vals = {col: scores_lookup.get((*key, col)) for col in col_names}
        if all(v is None for v in row_vals.values()):
            continue

        params = params_lookup.get(key)
        row = {
            'Section': section    if section    != prev_section    else '',
            'Init':    init_group if init_group != prev_init_group else '',
            'Model':   model_label,
            'Params':  _params_str(params),
        }
        for col in col_names:
            row[col] = _fmt_scores(row_vals[col])
        rows.append(row)
        prev_section    = section
        prev_init_group = init_group

    return pd.DataFrame(rows) if rows else None


def main():
    all_records = load_sft() + load_rsfm()

    for dataset in DATASETS:
        title = DATASET_NAMES[dataset]
        print(f'\n{"="*80}')
        print(f'  {title}')
        print(f'{"="*80}')

        # --- Table 1: SFT ---
        df = build_table(dataset, all_records)
        print(f'\n  -- SFT --')
        if df is None or df.empty:
            print('  (no data)')
        else:
            print(df.to_string(index=False))

        # --- Table 2: Distillation / MKE ---
        distill_records = load_distillation(dataset)
        mke_records     = load_mke(dataset)
        result2 = build_transfer_table(dataset, distill_records, mke_records)
        print(f'\n  -- Distillation / MKE (teacher → student) --')
        if result2 is None:
            print('  (no data)')
        else:
            df2, col_keys2, col_display2 = result2
            _print_transfer_table(dataset, df2, col_keys2, col_display2)


if __name__ == '__main__':
    main()

# python res/results_table.py
