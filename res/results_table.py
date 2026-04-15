"""Unified per-dataset result tables.

Rows:  method × init × model size
Cols:  modality (per dataset)

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
        'S2+S1':  ['s2+s1', 's1+s2'],
    },
    'dfc2020': {
        'S2-RGB': ['s2_rgb'],
        'S2':     ['s2'],
        'S1':     ['s1'],
        'S2+S1':  ['s2+s1', 's1+s2'],
    },
    'eurosat': {
        'RGB':  ['rgb'],
        'VRE':  ['vre'],
        'NIR':  ['nir'],
        'SWIR': ['swir'],
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
    ('SFT', 'DINO v3',    'DINO-B LVD'),
    ('SFT', 'DINO v3',    'DINO-L LVD'),
    ('SFT', 'DINO v3',    'DINO-L SAT'),
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
        best = _best(df, ['modality', 'model_type', 'dino_init'], 'test_metric')
        for _, row in best.iterrows():
            model_label = EVAN_MODEL_LABELS.get(row['model_type'], row['model_type'])
            init_group  = 'DINO v3' if row['dino_init'] else 'Random init'
            records.append(dict(
                dataset=dataset,
                col_aliases=[row['modality']],
                init_group=init_group, model_label=model_label,
                params=row.get('trainable_params'),
                score=row['test_metric'],
            ))
    return records


def load_rsfm():
    df = _read_csv('res/rsfm/rsfm_results.csv')
    if df is None:
        return []
    df = df[df['train_mode'] == 'fft']
    best = _best(df, ['dataset', 'modality', 'model'], 'test_metric')
    records = []
    for _, row in best.iterrows():
        dataset = row['dataset']
        if dataset not in COLUMNS:
            continue
        init_group, model_label = _rsfm_model_info(row['model'])
        records.append(dict(
            dataset=dataset,
            col_aliases=[row['modality']],
            init_group=init_group, model_label=model_label,
            params=row.get('trainable_params'),
            score=row['test_metric'],
        ))
    return records


# ---------------------------------------------------------------------------
# Build and print tables
# ---------------------------------------------------------------------------

def build_table(dataset, all_records):
    col_defs  = COLUMNS[dataset]
    col_names = list(col_defs.keys())

    # lookup: (init_group, model_label, col_name) → score
    score_lookup  = {}
    params_lookup = {}

    for rec in all_records:
        if rec['dataset'] != dataset:
            continue
        key = (rec['init_group'], rec['model_label'])
        for col_name, aliases in col_defs.items():
            if any(a in rec['col_aliases'] for a in aliases):
                lk = (*key, col_name)
                prev = score_lookup.get(lk)
                if prev is None or rec['score'] > prev:
                    score_lookup[lk]   = rec['score']
                    params_lookup[key] = rec.get('params')

    rows = []
    prev_section    = None
    prev_init_group = None
    for section, init_group, model_label in ROW_ORDER:
        key      = (init_group, model_label)
        row_vals = {col: score_lookup.get((*key, col)) for col in col_names}
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
            v = row_vals[col]
            row[col] = f'{v:.2f}' if v is not None else ''
        rows.append(row)
        prev_section    = section
        prev_init_group = init_group

    return pd.DataFrame(rows) if rows else None


def main():
    all_records = load_sft() + load_rsfm()

    for dataset in DATASETS:
        df = build_table(dataset, all_records)
        title = DATASET_NAMES[dataset]
        print(f'\n{"="*80}')
        print(f'  {title}')
        print(f'{"="*80}')
        if df is None or df.empty:
            print('  (no data)')
        else:
            print(df.to_string(index=False))


if __name__ == '__main__':
    main()

# python res/results_table.py
