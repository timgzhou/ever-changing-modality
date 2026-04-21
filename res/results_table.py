"""Unified per-dataset result tables.

Table A (SFT baseline): rows = init × model size, cols = modality
Table B (Transfer): one row per (dataset, starting_mod → new_mod),
  cols = random-init SFT | DINO SFT | KD | MTD | Delulu (ours) | labeled SFT (oracle) | Panopticon | OlmoEarth

Run from repo root:
    python res/results_table.py
"""

import os
import glob
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAN_MODEL_LABELS = {
    'evan_small': 'ViT-S',
    'evan_base':  'ViT-B',
    'evan_large': 'ViT-L',
}

DATASET_NAMES = {
    'benv2':   'reBEN (Multi-Label Classification, mAP)',
    'dfc2020': 'DFC2020 (Semantic Segmentation, mIoU)',
    'eurosat': 'EuroSAT (Classification, Acc)',
}

DATASETS = ['benv2', 'dfc2020', 'eurosat']

# Valid transfers per dataset for the Transfer table
VALID_TRANSFERS = {
    'benv2':   [('s2_rgb', 's1'), ('s2_rgb', 's2_norgb'), ('s1', 's2'), ('s2', 's1')],
    'dfc2020': [('s2_rgb', 's1'), ('s2_rgb', 's2_norgb'), ('s1', 's2'), ('s2', 's1')],
    'eurosat': [('rgb', 'nir'), ('rgb', 'vre'), ('swir', 'nir'), ('swir', 'rgb'), ('swir', 'vre'), ('vre', 'nir'), ('vre', 'rgb')],
}

# Display names for modalities
MOD_DISPLAY = {
    's2_rgb': 'S2-RGB', 's2': 'S2', 's1': 'S1',
    's2_norgb': 'S2-noRGB', 'rgb': 'RGB', 'vre': 'VRE',
    'nir': 'NIR', 'swir': 'SWIR',
}

# SFT table: modality columns per dataset
SFT_COLUMNS = {
    'benv2':   {'S2-RGB': ['s2_rgb'], 'S2': ['s2'], 'S1': ['s1'], 'S2+S1': ['s2+s1', 's1+s2', 's2s1']},
    'dfc2020': {'S2-RGB': ['s2_rgb'], 'S2': ['s2'], 'S1': ['s1'], 'S2+S1': ['s2+s1', 's1+s2', 's2s1']},
    'eurosat': {'RGB': ['rgb'], 'VRE': ['vre'], 'SWIR': ['swir'], 'S2': ['s2']},
}

# SFT row order: (init_group, model_label)
SFT_ROW_ORDER = [
    ('Random init', 'ViT-S'),
    ('Random init', 'ViT-B'),
    ('Random init', 'ViT-L'),
    ('DINO v3',    'ViT-S'),
    ('DINO v3',    'ViT-B'),
    ('DINO v3',    'ViT-L'),
    ('Panopticon', 'Panopticon-B'),
    ('OlmoEarth',  'OlmoEarth-B'),
    ('OlmoEarth',  'OlmoEarth-L'),
]


def _rsfm_model_info(model_str):
    m = model_str.lower()
    if 'panopticon' in m:
        return 'Panopticon', 'Panopticon-B'
    if 'olmoearth' in m:
        size = 'OlmoEarth-L' if 'large' in m else 'OlmoEarth-B'
        return 'OlmoEarth', size
    if 'dino' in m:
        if 'vitl' in m:
            variant = 'SAT' if 'sat' in m else 'LVD'
            return 'DINO v3', f'DINO-L {variant}'
        variant = 'SAT' if 'sat' in m else 'LVD'
        return 'DINO v3', f'DINO-B {variant}'
    return 'Pretrained', model_str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(path):
    if not os.path.isfile(path):
        return None
    df = pd.read_csv(path)
    # Drop header-repeat rows (CSVs sometimes have repeated header rows)
    if 'dataset' in df.columns:
        df = df[df['dataset'] != 'dataset']
    return df if not df.empty else None


def _fmt(val, decimals=2):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '--'
    return f'{float(val):.{decimals}f}'


def _best_val_test(df, group_cols, val_col, test_col):
    """For each group, pick row with best val_col, return test_col value."""
    return (
        df.sort_values(val_col, ascending=False)
        .groupby(group_cols, as_index=False)
        .first()
    )[group_cols + [test_col]]


def _avg_test(df, group_cols, test_col):
    """Average test_col over HP runs per group (no val selection available)."""
    return df.groupby(group_cols, as_index=False)[test_col].mean()


# ---------------------------------------------------------------------------
# SFT table loaders
# ---------------------------------------------------------------------------

def load_sft_records():
    """Returns list of dicts: {dataset, modality_aliases, init_group, model_label, test_metric}"""
    records = []

    # train_sft (DINO-init only so far)
    for dataset in DATASETS:
        df = _read_csv(f'res/train_sft/{dataset}.csv')
        if df is None:
            continue
        df['val_metric'] = pd.to_numeric(df['val_metric'], errors='coerce')
        df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
        df['dino_init'] = df['dino_init'].astype(str).str.lower().map(
            {'true': True, 'false': False, '1': True, '0': False})
        for (modality, model_type, dino_init), grp in df.groupby(['modality', 'model_type', 'dino_init']):
            best = grp.sort_values('val_metric', ascending=False).iloc[0]
            model_label = EVAN_MODEL_LABELS.get(model_type, model_type)
            init_group = 'DINO v3' if dino_init else 'Random init'
            records.append(dict(
                dataset=dataset,
                col_aliases=[modality],
                init_group=init_group,
                model_label=model_label,
                test_metric=best['test_metric'],
            ))

    # rsfm (panopticon, olmoearth — exclude dino which is in train_sft)
    df = _read_csv('res/rsfm/rsfm_results.csv')
    if df is not None:
        df = df[df['train_mode'] == 'fft']
        df = df[~df['model'].str.lower().str.contains('dino')]
        df['val_metric'] = pd.to_numeric(df['val_metric'], errors='coerce')
        df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
        for (dataset, modality, model), grp in df.groupby(['dataset', 'modality', 'model']):
            if dataset not in DATASETS:
                continue
            best = grp.sort_values('val_metric', ascending=False).iloc[0]
            init_group, model_label = _rsfm_model_info(model)
            records.append(dict(
                dataset=dataset,
                col_aliases=[modality],
                init_group=init_group,
                model_label=model_label,
                test_metric=best['test_metric'],
            ))

    return records


def build_sft_table(dataset, records):
    col_defs = SFT_COLUMNS[dataset]
    col_names = list(col_defs.keys())

    scores = {}
    for rec in records:
        if rec['dataset'] != dataset:
            continue
        key = (rec['init_group'], rec['model_label'])
        for col_name, aliases in col_defs.items():
            if any(a in rec['col_aliases'] for a in aliases):
                scores[(key, col_name)] = rec['test_metric']

    rows = []
    prev_init = None
    for init_group, model_label in SFT_ROW_ORDER:
        key = (init_group, model_label)
        row_vals = {col: scores.get((key, col)) for col in col_names}
        if all(v is None for v in row_vals.values()):
            continue
        row = {
            'Init':  init_group if init_group != prev_init else '',
            'Model': model_label,
        }
        for col in col_names:
            row[col] = _fmt(row_vals[col])
        rows.append(row)
        prev_init = init_group

    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Transfer table loaders
# ---------------------------------------------------------------------------

def _load_distillation_transfer(dataset):
    """
    No val metric available — average test_metric per kl_type over HP runs.
    Returns dict: (teacher_mod, student_mod, kl_type) → score
    Only for evan_base (ViT-B).
    """
    base = f'res/baselines/distillation/{dataset}/evan_base'
    if not os.path.isdir(base):
        return {}
    result = {}
    for fpath in glob.glob(f'{base}/*.csv'):
        df = _read_csv(fpath)
        if df is None or 'teacher_modality' not in df.columns:
            continue
        df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
        for (teacher, student, kl_type), grp in df.groupby(['teacher_modality', 'student_modality', 'kl_type']):
            result[(teacher, student, kl_type)] = grp['test_metric'].mean()
    return result


def _load_mke_addition(dataset):
    """
    MKE (Multimodal Knowledge Expansion): multimodal student trained from unimodal teacher.
    Val-selected via valchecked_test_metric (teacher-agreement proxy).
    Returns dict: (start_mod, new_mod) → best valchecked_test_metric
    Only for evan_base (ViT-B).
    """
    df = _read_csv(f'res/baselines/mke/{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == 'evan_base']
    df['valchecked_test_metric'] = pd.to_numeric(df['valchecked_test_metric'], errors='coerce')
    result = {}
    for (teacher, student_mods), grp in df.groupby(['teacher_modality', 'student_modalities']):
        if '+' not in str(student_mods):
            continue
        parts = [p.strip() for p in student_mods.split('+')]
        new_parts = [p for p in parts if p != teacher]
        if len(new_parts) != 1:
            continue
        new_mod = new_parts[0]
        result[(teacher, new_mod)] = grp['valchecked_test_metric'].max()
    return result


def _load_mke_transfer(dataset):
    """
    Uses valchecked_test_metric (val-selected via teacher-agreement proxy).
    Returns dict: (teacher_mod, student_mod) → best valchecked_test_metric
    Only for evan_base (ViT-B).
    """
    df = _read_csv(f'res/baselines/mke/{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == 'evan_base']
    df['valchecked_test_metric'] = pd.to_numeric(df['valchecked_test_metric'], errors='coerce')
    result = {}
    for (teacher, student_mods), grp in df.groupby(['teacher_modality', 'student_modalities']):
        best = grp['valchecked_test_metric'].max()
        result[(teacher, student_mods)] = best
    return result


def _load_delulu_transfer(dataset):
    """
    Uses valchecked_transfer (test metric at best val_transfer checkpoint).
    Returns dict: (starting_mod, new_mod) → best valchecked_transfer over HP runs
    Only for evan_base (ViT-B).
    """
    df = _read_csv('res/delulu/hptuned_apr21.csv')
    if df is None:
        return {}
    df = df[(df['dataset'] == dataset) & (df['model_arch'] == 'evan_base')]
    df['valchecked_transfer'] = pd.to_numeric(df['valchecked_transfer'], errors='coerce')
    df['valchecked_val_transfer'] = pd.to_numeric(df['valchecked_val_transfer'], errors='coerce')
    result = {}
    for (start, new), grp in df.groupby(['starting_modality', 'new_modality']):
        # pick HP run with best val_transfer metric
        best_row = grp.sort_values('valchecked_val_transfer', ascending=False).iloc[0]
        result[(start, new)] = best_row['valchecked_transfer']
    return result


def _load_labeled_sft_transfer(dataset):
    """
    Labeled SFT oracle: train_sft test metric for the new modality (DINO init, ViT-B).
    This is the upper bound — a model trained directly on the new modality with labels.
    Returns dict: new_mod → test_metric
    """
    df = _read_csv(f'res/train_sft/{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == 'evan_base']
    df['val_metric'] = pd.to_numeric(df['val_metric'], errors='coerce')
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    result = {}
    for modality, grp in df.groupby('modality'):
        best = grp.sort_values('val_metric', ascending=False).iloc[0]
        result[modality] = best['test_metric']
    return result


def _load_rsfm_transfer(dataset):
    """
    Panopticon and OlmoEarth-B/L test metrics for each modality (val-selected).
    Returns dict: (model_name, modality) → test_metric
    """
    df = _read_csv('res/rsfm/rsfm_results.csv')
    if df is None:
        return {}
    df = df[(df['dataset'] == dataset) & (df['train_mode'] == 'fft')]
    df = df[~df['model'].str.lower().str.contains('dino')]
    df['val_metric'] = pd.to_numeric(df['val_metric'], errors='coerce')
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    result = {}
    for (model, modality), grp in df.groupby(['model', 'modality']):
        best = grp.sort_values('val_metric', ascending=False).iloc[0]
        result[(model, modality)] = best['test_metric']
    return result


def _load_train_sft_starting(dataset):
    """DINO-init SFT on starting modality (ViT-B). Returns dict: modality → test_metric"""
    return _load_labeled_sft_transfer(dataset)


def _load_mixmatch_peek(dataset):
    """
    MixMatch baseline for peeking: trains on starting modality with unlabeled data.
    Has val metric (best_val_metric) → use best_val_test_metric for reporting.
    Returns dict: starting_mod → best_val_test_metric  (evan_base only)
    """
    df = _read_csv(f'res/baselines/mixmatch/baseline_mixmatch_{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == 'evan_base']
    df['best_val_metric'] = pd.to_numeric(df['best_val_metric'], errors='coerce')
    df['best_val_test_metric'] = pd.to_numeric(df['best_val_test_metric'], errors='coerce')
    result = {}
    for modality, grp in df.groupby('modality'):
        best = grp.sort_values('best_val_metric', ascending=False).iloc[0]
        result[modality] = best['best_val_test_metric']
    return result


def _load_delulu_peek(dataset):
    """
    Delulu peeking: val-selected via valchecked_val_peek, report valchecked_peek.
    Returns dict: (starting_mod, peeked_mod) → valchecked_peek
    """
    df = _read_csv('res/delulu/hptuned_apr21.csv')
    if df is None:
        return {}
    df = df[(df['dataset'] == dataset) & (df['model_arch'] == 'evan_base')]
    df['valchecked_peek'] = pd.to_numeric(df['valchecked_peek'], errors='coerce')
    df['valchecked_val_peek'] = pd.to_numeric(df['valchecked_val_peek'], errors='coerce')
    result = {}
    for (start, new), grp in df.groupby(['starting_modality', 'new_modality']):
        best_row = grp.sort_values('valchecked_val_peek', ascending=False).iloc[0]
        result[(start, new)] = best_row['valchecked_peek']
    return result


def _load_delulu_addition(dataset):
    """
    Delulu addition: val-selected via valchecked_val_add_ens, report valchecked_add_ens.
    Returns dict: (starting_mod, new_mod) → valchecked_add_ens
    """
    df = _read_csv('res/delulu/hptuned_apr21.csv')
    if df is None:
        return {}
    df = df[(df['dataset'] == dataset) & (df['model_arch'] == 'evan_base')]
    df['valchecked_add_ens'] = pd.to_numeric(df['valchecked_add_ens'], errors='coerce')
    df['valchecked_val_add_ens'] = pd.to_numeric(df['valchecked_val_add_ens'], errors='coerce')
    result = {}
    for (start, new), grp in df.groupby(['starting_modality', 'new_modality']):
        best_row = grp.sort_values('valchecked_val_add_ens', ascending=False).iloc[0]
        result[(start, new)] = best_row['valchecked_add_ens']
    return result


def _load_labeled_sft_addition(dataset):
    """
    Labeled SFT oracle on combined (start+new) modality (DINO init, ViT-B).
    Returns dict: (start, new) → test_metric  — keyed by sorted pair for lookup.
    """
    df = _read_csv(f'res/train_sft/{dataset}.csv')
    if df is None:
        return {}
    df = df[df['model_type'] == 'evan_base']
    df['val_metric'] = pd.to_numeric(df['val_metric'], errors='coerce')
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    result = {}
    combined_aliases = SFT_COLUMNS[dataset].get('S2+S1', []) + SFT_COLUMNS[dataset].get('S2+S2', [])
    for modality, grp in df.groupby('modality'):
        if '+' not in str(modality) and modality not in combined_aliases:
            continue
        best = grp.sort_values('val_metric', ascending=False).iloc[0]
        result[modality] = best['test_metric']
    return result


def _load_rsfm_addition(dataset):
    """
    RSFM baselines on combined modality.
    Returns dict: (model_name, combined_mod_alias) → test_metric
    """
    return _load_rsfm_transfer(dataset)


def build_addition_table(dataset):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    mke_add      = _load_mke_addition(dataset)
    delulu_add   = _load_delulu_addition(dataset)
    sft_combined = _load_labeled_sft_addition(dataset)
    rsfm         = _load_rsfm_transfer(dataset)

    # Combined modality aliases for RSFM lookup
    COMBINED_RSFM_ALIASES = {
        ('s2', 's1'): ['s2s1', 's1+s2', 's2+s1'],
        ('s1', 's2'): ['s2s1', 's1+s2', 's2+s1'],
        ('s2_rgb', 's1'): ['s2s1', 's1+s2', 's2+s1'],
        ('s2_rgb', 's2_norgb'): ['s2', 's2s1'],
    }

    rows = []
    for (start, new) in transfers:
        start_d = MOD_DISPLAY.get(start, start)
        new_d   = MOD_DISPLAY.get(new, new)

        mke_score = mke_add.get((start, new))
        del_score = delulu_add.get((start, new))

        # Labeled SFT on combined: try common alias patterns
        lsft_score = None
        for alias in [f'{start}+{new}', f'{new}+{start}'] + list(SFT_COLUMNS[dataset].get('S2+S1', [])):
            if alias in sft_combined:
                lsft_score = sft_combined[alias]
                break

        # RSFM on combined modality
        rsfm_aliases = COMBINED_RSFM_ALIASES.get((start, new), [f'{start}+{new}', f'{new}+{start}'])
        pan_score  = None
        olmo_score = None
        for alias in rsfm_aliases:
            if pan_score is None:
                pan_score = rsfm.get(('panopticon', alias))
            if olmo_score is None:
                olmo_score = rsfm.get(('olmoearth-base', alias)) or rsfm.get(('olmoearth-large', alias))

        rows.append({
            'Dataset':    DATASET_NAMES[dataset].split(' ')[0],
            'Start→New':  f'{start_d}→{new_d}',
            'rand SFT':   '--',
            'MKE':        _fmt(mke_score),
            'Delulu':     _fmt(del_score),
            'lbl SFT':    _fmt(lsft_score),
            'Panopticon': _fmt(pan_score),
            'OlmoEarth':  _fmt(olmo_score),
        })

    return pd.DataFrame(rows) if rows else None


def build_peek_table(dataset):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    mixmatch  = _load_mixmatch_peek(dataset)
    delulu    = _load_delulu_peek(dataset)
    sft_start = _load_train_sft_starting(dataset)
    rsfm      = _load_rsfm_transfer(dataset)

    rows = []
    for (start, new) in transfers:
        start_d = MOD_DISPLAY.get(start, start)
        new_d   = MOD_DISPLAY.get(new, new)

        dino_start = sft_start.get(start)
        mm_score   = mixmatch.get(start)
        del_score  = delulu.get((start, new))
        pan_score  = rsfm.get(('panopticon', start))
        olmo_score = rsfm.get(('olmoearth-base', start))

        rows.append({
            'Dataset':    DATASET_NAMES[dataset].split(' ')[0],
            'Start':      start_d,
            'Peeked':     new_d,
            'rand SFT':   '--',
            'DINO SFT':   _fmt(dino_start),
            'MixMatch':   _fmt(mm_score),
            'Delulu':     _fmt(del_score),
            'Panopticon': _fmt(pan_score),
            'OlmoEarth':  _fmt(olmo_score),
        })

    return pd.DataFrame(rows) if rows else None


def build_transfer_table(dataset):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    distill  = _load_distillation_transfer(dataset)
    delulu   = _load_delulu_transfer(dataset)
    sft_new  = _load_labeled_sft_transfer(dataset)
    sft_start = _load_train_sft_starting(dataset)
    rsfm     = _load_rsfm_transfer(dataset)

    # For random-init SFT on starting modality: not available yet
    # For DINO SFT on starting modality: from sft_start

    rows = []
    for (start, new) in transfers:
        start_d = MOD_DISPLAY.get(start, start)
        new_d   = MOD_DISPLAY.get(new, new)

        # DINO SFT metric for starting modality (teacher performance)
        dino_start = sft_start.get(start)

        # KD and TTM (distillation, no val selection — avg test_metric per kl_type)
        kd_score  = distill.get((start, new, 'kd'))
        ttm_score = distill.get((start, new, 'ttm'))

        # Delulu
        del_score = delulu.get((start, new))

        # Labeled SFT oracle for new modality
        lsft_score = sft_new.get(new)

        # Panopticon for new modality
        pan_score = rsfm.get(('panopticon', new))

        # OlmoEarth-B for new modality
        olmo_score = rsfm.get(('olmoearth-base', new))

        rows.append({
            'Dataset':    DATASET_NAMES[dataset].split(' ')[0],
            'Start→New':  f'{start_d}→{new_d}',
            'rand SFT':   '--',
            'DINO SFT':   _fmt(dino_start),
            'KD':         _fmt(kd_score),
            'TTM':        _fmt(ttm_score),
            'Delulu':     _fmt(del_score),
            'lbl SFT':    _fmt(lsft_score),
            'Panopticon': _fmt(pan_score),
            'OlmoEarth':  _fmt(olmo_score),
        })

    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('\n' + '='*80)
    print('  TABLE A: SFT Baselines (best val checkpoint → test metric)')
    print('='*80)

    sft_records = load_sft_records()
    for dataset in DATASETS:
        print(f'\n  --- {DATASET_NAMES[dataset]} ---')
        df = build_sft_table(dataset, sft_records)
        if df is None or df.empty:
            print('  (no data)')
        else:
            print(df.to_string(index=False))

    print('\n\n' + '='*80)
    print('  TABLE B: Transfer (start mod → new mod, ViT-B / evan_base)')
    print('  Columns: rand SFT = random-init SFT on starting mod (not yet available)')
    print('           DINO SFT = DINO-init SFT on starting mod (teacher perf)')
    print('           KD / TTM = Distillation baselines by kl_type (avg over HP, no val sel)')
    print('           Delulu   = Delulu ours (val-selected transfer metric)')
    print('           lbl SFT  = Labeled SFT oracle on new mod (upper bound)')
    print('           Panopticon / OlmoEarth = RSFM baselines on new mod (val-selected)')
    print('='*80)

    all_rows = []
    for dataset in DATASETS:
        df = build_transfer_table(dataset)
        if df is not None and not df.empty:
            all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        print(combined.to_string(index=False))
    else:
        print('  (no data)')

    print('\n\n' + '='*80)
    print('  TABLE C: Peeking (predict on start mod, peek unlabeled new mod, ViT-B)')
    print('  Columns: rand SFT   = random-init SFT on starting mod (not yet available)')
    print('           DINO SFT   = DINO-init SFT on starting mod (no peeking baseline)')
    print('           MixMatch   = semi-supervised on starting mod (val-selected)')
    print('           Delulu     = Delulu ours (val-selected peek metric)')
    print('           Panopticon / OlmoEarth = RSFM baselines on starting mod (val-selected)')
    print('='*80)

    peek_rows = []
    for dataset in DATASETS:
        df = build_peek_table(dataset)
        if df is not None and not df.empty:
            peek_rows.append(df)

    if peek_rows:
        combined = pd.concat(peek_rows, ignore_index=True)
        print(combined.to_string(index=False))
    else:
        print('  (no data)')

    print('\n\n' + '='*80)
    print('  TABLE D: Addition (predict on both start+new mod, ViT-B)')
    print('  Columns: rand SFT   = random-init SFT on combined mod (not yet available)')
    print('           MKE        = Multimodal Knowledge Expansion: multimodal student')
    print('           Delulu     = Delulu ours (val-selected addition ensemble metric)')
    print('           lbl SFT    = Labeled SFT oracle on combined mod (upper bound)')
    print('           Panopticon / OlmoEarth = RSFM baselines on combined mod (val-selected)')
    print('='*80)

    add_rows = []
    for dataset in DATASETS:
        df = build_addition_table(dataset)
        if df is not None and not df.empty:
            add_rows.append(df)

    if add_rows:
        combined = pd.concat(add_rows, ignore_index=True)
        print(combined.to_string(index=False))
    else:
        print('  (no data)')


if __name__ == '__main__':
    main()

# python res/results_table.py
