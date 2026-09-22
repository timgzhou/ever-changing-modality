"""Transfer / Peek / Addition tables with ViT-B and ViT-L side-by-side columns.

Outputs a single res/latex/BL_tables.tex with all three tabular environments.

Run from repo root:
    python res/results_BL.py [--arch B|L|BL] [--tall]
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
# Datasets rendered in the focused per-dataset tables (--datasets).
ALL_DATASETS = ['benv2', 'dfc2020', 'eurosat', 'biomassters']

DATASET_NAMES = {
    'benv2':   'reBEN (Multi-Label Classification, mAP)',
    'dfc2020': 'DFC2020 (Semantic Segmentation, mIoU)',
    'eurosat': 'EuroSAT (Classification, Acc)',
    'biomassters': 'BioMassters (AGB Regression, RMSE -- LOWER IS BETTER)',
}

DATASET_DISPLAY = {
    'benv2':   r'\shortstack[c]{reBEN\\(mAP)}',
    'dfc2020': r'\shortstack[c]{DFC2020\\(mIoU)}',
    'eurosat': r'\shortstack[c]{EuroSAT\\(Acc)}',
    'biomassters': r'\shortstack[c]{BioMassters\\(RMSE $\downarrow$)}',
}

VALID_TRANSFERS = {
    'benv2':   [('s2_rgb', 's1'), ('s2_rgb', 's2_norgb'), ('s1', 's2'), ('s2', 's1')],
    'dfc2020': [('s2_rgb', 's1'), ('s2_rgb', 's2_norgb'), ('s1', 's2'), ('s2', 's1')],
    'eurosat': [('rgb', 'vre')],
    'biomassters': [('s2_rgb', 's1'), ('s2_rgb', 's2_norgb'), ('s1', 's2'), ('s2', 's1')],
}

# Datasets whose metric is lower-is-better (see LOWER_IS_BETTER_DATASETS below,
# which is the row-level counterpart used at bolding time).
_LOWER_IS_BETTER_DS = ('biomassters',)


def _best_by_val(grp, dataset, col='val_metric'):
    """Pick the best-by-val row, respecting the metric's direction.

    BioMassters is RMSE, so ascending=False silently selected the WORST model:
    the s2_rgb teacher came back as test 76.9 (an undertrained lr 1e-4 run) when
    the best is 41.96.
    """
    ascending = dataset in _LOWER_IS_BETTER_DS
    return grp.sort_values(col, ascending=ascending).iloc[0]


MOD_DISPLAY = {
    's2_rgb': 'S2-RGB', 's2': 'S2', 's1': 'S1',
    's2_norgb': 'S2-noRGB', 'rgb': 'RGB', 'vre': 'VRE',
    'nir': 'NIR', 'swir': 'SWIR',
}

# EXACT combined-modality names only -- every alias here must name a run on the
# SAME pair of modalities as the cell.
#
# Loose fallbacks were removed 2026-09-21: they filled 11 of 27 oracle Addition
# cells with a number from a DIFFERENT experiment. 's2s1' is a genuine S2+S1
# run, so it is right for ('s1','s2')/('s2','s1') but wrong for
# ('s2_rgb','s1') -- RGB-only is not full S2. Worse, ('s2_rgb','s2_norgb') fell
# back to plain 's2', a UNIMODAL score printed in a bimodal cell. A missing
# oracle run must read '--', not borrow a neighbour's number.
COMBINED_RSFM_ALIASES = {
    ('s2', 's1'):           ['s2s1', 's2+s1', 's1+s2'],
    ('s1', 's2'):           ['s2s1', 's1+s2', 's2+s1'],
    ('s2_rgb', 's1'):       ['s2_rgb+s1', 's1+s2_rgb'],
    ('s2_rgb', 's2_norgb'): ['s2_rgb+s2_norgb', 's2_norgb+s2_rgb'],
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
    try:
        df = pd.read_csv(path)
    except pd.errors.ParserError:
        # A handful of older eurosat distillation CSVs have rows with one extra
        # field (an unquoted comma inside a value). Drop just those rows rather
        # than losing the whole file -- and say so, so a genuinely corrupt new
        # file is not silently half-read.
        df = pd.read_csv(path, on_bad_lines='skip', engine='python')
        print(f'  [warn] {path}: skipped malformed row(s)')
    if 'dataset' in df.columns:
        df = df[df['dataset'] != 'dataset']
    return df if not df.empty else None


def _fmt(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return '--'
    return f'{float(val):.{decimals}f}'


def _fmt_meanstd(val, decimals=1):
    if val is None:
        return '--'
    mean, std = val
    if np.isnan(mean):
        return '--'
    if np.isnan(std):
        return f'{mean:.{decimals}f}'
    return f'{mean:.{decimals}f}±{std:.{decimals}f}'


# Three seeds of each per-cell sweep winner (sh/delulu_seeds_all.sh). Holds all
# datasets; _load_delulu prefers these rows for any cell they cover.
PAPER_SEEDS_CSV = 'res/delulu/paper_seeds.csv'

DELULU_CSV = 'res/delulu/hptuned_masking_may6.csv'  # overridden by --apr21 / --may5 flags

# Flat per-dataset baseline CSVs, for datasets that never used the
# res/baselines/<family>/<dataset>/<arch>/ directory layout.
#
# dfc2020 (Copernicus-Bench) and biomassters write ONE flat file per family
# under res/baselines/, e.g. dfc2020_cobench_distill_transfer_upernet.csv. The
# loaders only globbed the directory layout, so KD/TTM/MixMatch/MKE came back
# empty for both datasets even though the results existed.
#
# Distillation: prefer the *distill_transfer* file. It is the correct analogue of
# Delulu's transfer metric (unimodal student on the NEW modality); the plain
# *distillation* file passes --modalities <teacher> <new>, a BIMODAL student,
# which is the ADDITION setting. See sh/distill_transfer_dfc2020.sh.
BASELINE_FLAT_CSVS = {
    'distillation': {
        'dfc2020':     ['res/baselines/dfc2020_cobench_distill_transfer_upernet_seeds.csv',
                        'res/baselines/dfc2020_cobench_distill_transfer_upernet.csv'],
        'biomassters': ['res/baselines/biomassters_distill_transfer_upernet.csv'],
        # benv2/eurosat transfer seeds live in their own flat files, written by
        # sweep/pick_baseline_winners.py + sh/baselines_seeds_all.sh. Without
        # these the loader globs only the per-cell sweep dirs, which have no
        # seed column, and the table falls back to a config spread.
        'benv2':       ['res/baselines/distillation_benv2_transfer_seeds.csv'],
        'eurosat':     ['res/baselines/distillation_eurosat_transfer_seeds.csv'],
    },
    'distillation_add': {   # bimodal student -> the addition/ensemble column
        'dfc2020':     ['res/baselines/dfc2020_cobench_distillation_upernet.csv'],
        'biomassters': ['res/baselines/biomassters_distillation_upernet.csv'],
    },
    'mke': {
        # *_initteacher.csv FIRST: the teacher-init arm is the like-for-like
        # analogue of Delulu's student_init='teacher'. MKE is the same EVAN +
        # EvanSegmenter stack minus masking and hallucination, so Delulu's own
        # init path applies -- load the teacher as the student, then add the
        # second modality (blocks seeded from the backbone). It gets its own
        # file because the original CSVs have a fixed 21-column schema and an
        # extra field would silently shift every column left in pandas.
        'dfc2020':     ['res/baselines/dfc2020_cobench_mke_upernet_initteacher_seeds.csv',
                        'res/baselines/dfc2020_cobench_mke_upernet_initteacher.csv',
                        'res/baselines/dfc2020_cobench_mke_upernet.csv'],
        'biomassters': ['res/baselines/biomassters_mke_upernet_initteacher.csv',
                        'res/baselines/biomassters_mke_upernet.csv'],
        # benv2/eurosat were previously absent, so _load_mke_addition fell
        # through to its single-file fallback and never saw the *_seeds.csv.
        'benv2':       ['res/baselines/mke/benv2_seeds.csv',
                        'res/baselines/mke/benv2.csv'],
        'eurosat':     ['res/baselines/mke/eurosat_seeds.csv',
                        'res/baselines/mke/eurosat.csv'],
    },
    'freematch': {
        # FreeMatch is semi-supervised and TEACHER-FREE, like MixMatch, so these
        # rows are unaffected by any teacher change. Absent for biomassters by
        # design: its self-adaptive thresholding on max(softmax) plus a class
        # histogram are K-way classification constructs with no regression
        # analogue (arXiv 2205.07246 never mentions regression), so the column
        # is structurally empty there rather than merely unrun.
        'dfc2020':     ['res/baselines/dfc2020_cobench_freematch_upernet_seeds.csv',
                        'res/baselines/dfc2020_cobench_freematch_upernet.csv'],
        # benv2/eurosat predate the flat-CSV convention: they live in the
        # freematch/ subdir under the baseline_freematch_<ds> name, not at
        # res/baselines/<ds>_freematch.csv. The old paths matched no file, so
        # both columns silently read '--' despite the runs existing.
        'benv2':       ['res/baselines/freematch/baseline_freematch_benv2_seeds.csv',
                        'res/baselines/freematch/baseline_freematch_benv2.csv'],
        'eurosat':     ['res/baselines/freematch/baseline_freematch_eurosat_seeds.csv',
                        'res/baselines/freematch/baseline_freematch_eurosat.csv'],
    },
    'mixmatch': {
        # RERUN first: the original dfc2020 rows all used lambda_u=75, which
        # collapses training (0.34-23.39 mIoU vs 54-59 at lambda_u 0.5-1.0).
        'dfc2020':     ['res/baselines/dfc2020_cobench_mixmatch_upernet_seeds.csv',
                        'res/baselines/dfc2020_cobench_mixmatch_upernet_RERUN.csv',
                        'res/baselines/dfc2020_cobench_mixmatch_lambdau_upernet.csv'],
        'biomassters': ['res/baselines/biomassters_mixmatch_upernet.csv'],
        # benv2/eurosat were previously absent here, so _load_mixmatch_peek fell
        # through to its single-file fallback and never saw the *_seeds.csv.
        'benv2':       ['res/baselines/mixmatch/baseline_mixmatch_benv2_seeds.csv',
                        'res/baselines/mixmatch/baseline_mixmatch_benv2.csv'],
        'eurosat':     ['res/baselines/mixmatch/baseline_mixmatch_eurosat_seeds.csv',
                        'res/baselines/mixmatch/baseline_mixmatch_eurosat.csv'],
    },
}


def _flat_frames(family, dataset, arch):
    """Concatenated flat baseline CSVs for (family, dataset), or None."""
    paths = list(BASELINE_FLAT_CSVS.get(family, {}).get(dataset, []))
    # MKE's teacher-init arm lives in its OWN FILE (the original CSVs have a
    # fixed 21-column schema, so an extra column would silently shift fields).
    # When that file has rows, report it alone -- the random/DINO-init arm is
    # not comparable to Delulu's teacher-initialised student.
    if family == 'mke':
        ti = [p for p in paths if '_initteacher' in p]
        if any((_read_csv(p) is not None and len(_read_csv(p))) for p in ti):
            paths = ti
    frames = []
    for path in paths:
        df = _read_csv(path)
        if df is None:
            continue
        if 'model_type' in df.columns:
            df = df[df['model_type'] == arch]
        # TRANSFER distillation only: report the TEACHER-INIT arm.
        #
        # NOTE teacher-init is NOT the baselines' best configuration. Measured
        # on dfc2020 2026-09-17: KD/TTM are -0.56 mIoU worse with it (better in
        # 2/8 cells), MKE +0.16 (2/4). train_delulu.py:248 already documented
        # why -- a transfer student must UNLEARN the teacher's modality-specific
        # features. Reporting it anyway moves Delulu 4/8 -> 7/8 on dfc2020
        # Transfer+Addition, so any write-up must say the baselines are matched
        # to Delulu's init rather than tuned.
        #
        # This is deliberate: different inits want different TRAINING (separate
        # lrs for pretrained vs newly-added parameters), and the current recipe
        # is tuned for random init, so a best-per-method table would confound
        # init with "which init suits the existing hyperparameters". Matched
        # init under one recipe is the controlled comparison. Revisiting it
        # needs per-arm lr tuning first, not just flipping this filter.
        #
        # KD/TTM students were built with load_weights=False -- neither teacher
        # weights nor DINO -- while Delulu's transfer student defaults to
        # student_init='teacher' and MKE/MixMatch both get DINO. That gave the
        # Transfer column's baselines strictly less initialisation than every
        # other method in the tables. The teacher-init arm
        # (--init_from_teacher) is the like-for-like analogue of Delulu's
        # setting: backbone and modality-specific blocks copied, patch embedder
        # random (the channel counts differ).
        #
        # Both arms live in the same CSV, so without this filter the table would
        # silently pool them and report whichever won by val.
        if family == 'distillation' and 'init_from_teacher' in df.columns:
            mask = df['init_from_teacher'].astype(str).str.lower().isin(['true', '1'])
            if mask.any():
                df = df[mask]
        if len(df):
            frames.append(df)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True, sort=False)


# Datasets reported from a single crossconfig sweep, which REPLACES DELULU_CSV
# and DELULU_EXTRA_CSVS rather than adding to them (see _load_delulu).
#
# dfc2020 moved to the Copernicus-Bench split in August and biomassters was
# added in September; neither writes into the pooled April/May DELULU_CSV. Each
# has since been swept as one grid of (config x seed) on the current code, so
# that grid is the whole candidate set -- anything else on disk is an earlier,
# non-comparable run.
#
# Only files whose rows carry `select_by` are usable: the table picks a config
# per (start, new, selector), and a file with select_by unset cannot answer that
# unless --ignore_select_by is passed.
DELULU_CROSSCONFIG_CSVS = {
    'dfc2020':     'res/delulu/dfc2020_crossconfig.csv',
    'biomassters': 'res/delulu/biomassters_crossconfig.csv',
}

# BIOMASSTERS_TPOOLED=1 builds the biomassters table from the mean-pooled runs
# (input-pooled T, non-temporal model) instead of the T=12 ones. They are
# separate experiments on different inputs and must never be mixed in one cell,
# so they live in separate CSVs and the stage-0 teachers come from the
# '/tpooled' registry keys. Off by default, so existing tables are unchanged.
_TPOOLED_ON = os.environ.get('BIOMASSTERS_TPOOLED', '0') not in ('0', '', 'false', 'False')
if _TPOOLED_ON:
    DELULU_CROSSCONFIG_CSVS['biomassters'] = 'res/delulu/biomassters_crossconfig_tpooled.csv'


def _TPOOLED(dataset):
    """True when this dataset's table should read mean-pooled (T<0) stage-0 rows."""
    return _TPOOLED_ON and dataset == 'biomassters'

# Per-dataset Delulu result files, read IN ADDITION to DELULU_CSV. Only consulted
# for datasets absent from DELULU_CROSSCONFIG_CSVS above.
DELULU_EXTRA_CSVS = {}

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _distill_agg(df, id_cols, result, dataset=None):
    """Aggregate one distillation frame into result[(teacher, student, kl)]."""
    df = df.copy()
    df['test_metric']        = pd.to_numeric(df['test_metric'],        errors='coerce')
    df['best_val_agreement'] = pd.to_numeric(df['best_val_agreement'], errors='coerce')
    if 'seed' in df.columns:
        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
    for (teacher, student, kl_type), grp in df.groupby(
            ['teacher_modality', 'student_modality', 'kl_type']):
        seeded = _seed_agg(grp, 'best_val_agreement', 'test_metric', dataset)
        if seeded is not None:
            result[(teacher, student, kl_type)] = seeded[:2]
            continue
        val_rows = grp[grp['best_val_agreement'].notna()]
        if val_rows.empty:
            topk = grp.nlargest(5, 'test_metric')['test_metric']
            result[(teacher, student, kl_type)] = (topk.mean(), topk.std())
            continue
        top2 = val_rows.nlargest(2, 'best_val_agreement')
        all_test = list(top2['test_metric'])
        no_val = grp[grp['best_val_agreement'].isna()]
        use_ids = [c for c in id_cols if c in grp.columns]
        for _, row in top2.iterrows():
            if use_ids:
                mask = (no_val[use_ids] == row[use_ids].values).all(axis=1)
                all_test.extend(no_val[mask]['test_metric'].tolist())
        s = pd.Series(all_test, dtype=float).dropna()
        result[(teacher, student, kl_type)] = (s.mean(), s.std())
    return result


def _load_distillation(dataset, arch):
    # Flat-file datasets (dfc2020 cobench, biomassters) have no per-arch dir.
    flat = _flat_frames('distillation', dataset, arch)
    if flat is not None and 'teacher_modality' in flat.columns:
        id_cols = list(flat.columns[6:13]) + ['teacher_checkpoint']
        return _distill_agg(flat, id_cols, {}, dataset)

    base = f'res/baselines/distillation/{dataset}/{arch}'
    if not os.path.isdir(base):
        return {}
    id_cols = list(pd.read_csv(glob.glob(f'{base}/*.csv')[0], nrows=0).columns[6:13]) + ['teacher_checkpoint'] if glob.glob(f'{base}/*.csv') else []
    result = {}
    for fpath in glob.glob(f'{base}/*.csv'):
        df = _read_csv(fpath)
        if df is None or 'teacher_modality' not in df.columns:
            continue
        df['test_metric']          = pd.to_numeric(df['test_metric'],          errors='coerce')
        df['best_val_agreement']   = pd.to_numeric(df['best_val_agreement'],   errors='coerce')
        for (teacher, student, kl_type), grp in df.groupby(['teacher_modality', 'student_modality', 'kl_type']):
            val_rows = grp[grp['best_val_agreement'].notna()]
            if val_rows.empty:
                # fallback: no val data available
                topk = grp.nlargest(5, 'test_metric')['test_metric']
                result[(teacher, student, kl_type)] = (topk.mean(), topk.std())
                continue
            top2 = val_rows.nlargest(2, 'best_val_agreement')
            # collect test scores from top-2 val rows + their paired NaN-val rows (same id cols)
            all_test = list(top2['test_metric'])
            no_val = grp[grp['best_val_agreement'].isna()]
            for _, row in top2.iterrows():
                mask = (no_val[id_cols] == row[id_cols].values).all(axis=1)
                all_test.extend(no_val[mask]['test_metric'].tolist())
            s = pd.Series(all_test, dtype=float).dropna()
            result[(teacher, student, kl_type)] = (s.mean(), s.std())
    return result


def _load_distillation_ens(dataset, arch):
    """Like _load_distillation but uses ensemble_metric_logits as the target metric."""
    base = f'res/baselines/distillation/{dataset}/{arch}'
    if not os.path.isdir(base):
        return {}
    result = {}
    for fpath in glob.glob(f'{base}/*.csv'):
        df = _read_csv(fpath)
        if df is None or 'teacher_modality' not in df.columns:
            continue
        df['ensemble_metric_logits'] = pd.to_numeric(df['ensemble_metric_logits'], errors='coerce')
        df['best_val_agreement']     = pd.to_numeric(df['best_val_agreement'],     errors='coerce')
        for (teacher, student, kl_type), grp in df.groupby(['teacher_modality', 'student_modality', 'kl_type']):
            val_rows = grp[grp['best_val_agreement'].notna()]
            if val_rows.empty:
                vals = grp.nlargest(5, 'ensemble_metric_logits')['ensemble_metric_logits'].dropna()
            else:
                vals = val_rows.nlargest(2, 'best_val_agreement')['ensemble_metric_logits'].dropna()
            if len(vals):
                result[(teacher, student, kl_type)] = (vals.mean(), vals.std())
    return result


def _load_delulu(dataset, arch, val_col, test_col, ignore_select_by=False):
    # old col name → new col name (may5 format)
    COL_MAP = {
        'valchecked_transfer':     'test_transfer',
        'valchecked_peek':         'test_peeking',
        'valchecked_add':          'test_addition',
        'valchecked_add_ens':      'test_ens_addition',
        'valchecked_val_transfer': 'val_transfer',
        'valchecked_val_peek':     'val_peeking',
        'valchecked_val_add':      'val_addition',
        'valchecked_val_add_ens':  'val_ens_addition',
    }
    SELECT_MAP = {
        'val_transfer':     'transfer',
        'val_peeking':      'peeking',
        'val_addition':     'addition',
        'val_ens_addition': 'addition',
    }
    norm_val  = COL_MAP.get(val_col,  val_col)
    norm_test = COL_MAP.get(test_col, test_col)

    # A dataset with a crossconfig file is reported from that file ALONE: it is
    # the complete, current sweep (every config x seed on the current code), so
    # pooling it with the legacy CSVs can only add older runs to the candidate
    # set that val-selection then picks from. Those runs are not comparable --
    # dfc2020's legacy file is the leaky-teacher era, and biomassters' best_*
    # files are single-pair snapshots -- and mixing them inflates the spread
    # (biomassters addition read 69.9+-23.8 pooled).
    frames = []
    crossconfig = DELULU_CROSSCONFIG_CSVS.get(dataset)
    if crossconfig is not None:
        df = _read_csv(crossconfig)
        if df is None:
            return {}
    else:
        base = _read_csv(DELULU_CSV)
        if base is not None:
            frames.append(base)
        for extra in DELULU_EXTRA_CSVS.get(dataset, []):
            e = _read_csv(extra)
            if e is not None:
                frames.append(e)
        if not frames:
            return {}
        df = pd.concat(frames, ignore_index=True, sort=False)

    df = df[df['dataset'] == dataset]

    # train_delulu stores regression metrics NEGATED (delulu.py _neg_rmse) so
    # that val-selection can maximise uniformly across task types. Flip them
    # back for display, after the dataset filter so only regression rows move.
    if dataset in _LOWER_IS_BETTER_DS:
        for c in df.columns:
            if c.startswith(('val_', 'test_')):
                df[c] = -pd.to_numeric(df[c], errors='coerce')

    # apr21 has model_arch; may5 does not (all evan_base)
    if 'model_arch' in df.columns:
        df = df[df['model_arch'] == arch]

    # apr21 uses old col names; remap if needed
    if val_col in df.columns:
        df = df.rename(columns={val_col: norm_val, test_col: norm_test})

    if norm_val not in df.columns or norm_test not in df.columns:
        return {}

    # may5: filter by select_by when present (skip if ignore_select_by)
    if not ignore_select_by and 'select_by' in df.columns and df['select_by'].notna().any():
        sel = SELECT_MAP.get(norm_val)
        if sel is not None:
            df = df[df['select_by'] == sel]

    df[norm_test] = pd.to_numeric(df[norm_test], errors='coerce')
    df[norm_val]  = pd.to_numeric(df[norm_val],  errors='coerce')
    if 'seed' in df.columns:
        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')

    # The paper seed runs -- 3 seeds of each per-cell sweep winner, written by
    # sh/delulu_seeds_all.sh -- are the only Delulu rows that yield a real seed
    # std. Append them, tagged so the per-cell loop can prefer them: pooling
    # them with the sweep trials would let a lucky trial win val-selection and
    # put the cell back on a config spread.
    seeded = _read_csv(PAPER_SEEDS_CSV)
    if seeded is not None and 'dataset' in seeded.columns:
        seeded = seeded[seeded['dataset'] == dataset].copy()
        if len(seeded):
            for c in (norm_val, norm_test, 'seed'):
                if c in seeded.columns:
                    seeded[c] = pd.to_numeric(seeded[c], errors='coerce')
            if not ignore_select_by and 'select_by' in seeded.columns:
                sel = SELECT_MAP.get(norm_val)
                if sel is not None:
                    seeded = seeded[seeded['select_by'] == sel]
            if norm_val in seeded.columns and norm_test in seeded.columns and len(seeded):
                seeded['_paper_seed'] = True
                df = df.assign(_paper_seed=False)
                df = pd.concat([df, seeded], ignore_index=True, sort=False)

    result = {}
    for (start, new), grp in df.groupby(['starting_modality', 'new_modality']):
        if '_paper_seed' in grp.columns and grp['_paper_seed'].fillna(False).any():
            grp = grp[grp['_paper_seed'].fillna(False)]
        # Prefer a real seed std: pick the config by val, then spread across
        # THAT config's seeds. Falls back to top-k-by-val for cells that have
        # no seeded rows yet (biomassters, and any pre-seed Delulu sweep).
        seeded = _seed_agg(grp, norm_val, norm_test, dataset)
        if seeded is not None:
            result[(start, new)] = seeded[:2]
            continue
        # Direction matters: the regression columns were un-negated above, so
        # nlargest would pick the WORST (highest-RMSE) configs for biomassters.
        top3 = _nbest(grp, norm_val, 3, dataset).index
        top3 = grp.loc[top3, norm_test]
        result[(start, new)] = (top3.mean(), top3.std())
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
        best = _best_by_val(grp, dataset)
        result[(model, modality)] = best['test_metric']
    return result


# The SFT CSV, decoder and train_split that each dataset's table is built from.
# Getting any of these wrong silently reports a different model:
#   - dfc2020 results live in dfc2020_cobench.csv; the older dfc2020.csv is
#     LINEAR-decoder only, so reading it gave s2_rgb = 33.0 instead of 59.0.
#   - a `full` teacher has already seen train2 with labels, which is the stage-1
#     unlabeled pool, so mixing splits leaks. split1 only.
#   - dfc2020/benv2 tables are upernet; biomassters is upernet+relu.
SFT_SOURCE = {
    # benv2 is multi-label CLASSIFICATION: its head is 'cls', never upernet.
    # Asking for upernet here matched zero rows and blanked every benv2 stage-0
    # cell (f_0 and the DINOv3 oracle) in all three tables.
    'benv2':       dict(csv='benv2',           decoder='cls',          split='split1'),
    'dfc2020':     dict(csv='dfc2020_cobench', decoder='upernet',      split='split1'),
    'eurosat':     dict(csv='eurosat',         decoder=None,           split=None),
    'biomassters': dict(csv='biomassters',     decoder='upernet+relu', split='split1'),
}


def _sft_frame(dataset, arch):
    """Load the SFT rows for a dataset's table: right file, decoder and split."""
    spec = SFT_SOURCE.get(dataset, dict(csv=dataset, decoder=None, split=None))
    df = _read_csv(f"res/train_sft/{spec['csv']}.csv")
    if df is None:
        return None
    df = df[df['model_type'] == arch]
    # Filter only on columns the file actually has: older CSVs predate them, and
    # a missing column means the file is single-decoder / single-split anyway.
    if spec['decoder'] and 'decoder' in df.columns:
        want = spec['decoder']
        got = df['decoder'].astype(str)
        # 'upernet+relu' is recorded as 'upernet' plus relu_output in some files.
        df = df[(got == want) | (got == want.split('+')[0])] if '+' in want else df[got == want]
    if spec['split'] and 'train_split' in df.columns:
        df = df[df['train_split'].astype(str) == spec['split']]
    # Temporal regime. biomassters rows carry num_time_steps: 12 for the
    # temporal stack, -12 for the input-mean-pooled cache. They are different
    # models on different inputs, so the f_0 / oracle rows must come from the
    # same regime as the Delulu rows they are compared against -- otherwise a
    # pooled Delulu number is scored against a temporal teacher.
    if 'num_time_steps' in df.columns:
        nts = pd.to_numeric(df['num_time_steps'], errors='coerce')
        df = df[nts < 0] if _TPOOLED(dataset) else df[~(nts < 0)]
    return df if len(df) else None


def _load_sft_dino(dataset, arch='evan_base'):
    """Supervised SFT for given arch: modality → test_metric (val-selected).

    Init (DINO vs random) is chosen by val, not pinned -- see note below.
    """
    df = _sft_frame(dataset, arch)
    if df is None:
        return {}
    df = df.copy()
    # NOTE: do NOT filter to dino_init == True. Which init wins flips with the
    # DECODER, not the train split. DFC2020, paired over matched
    # (modality, epoch, lr, wd):
    #     linear  : DINO wins,   mean -5.3 to -5.9 mIoU, 47/48 pairs
    #     upernet : random wins, mean +2.1 to +2.6 mIoU, 37/45 pairs
    # Both splits agree within a decoder; fully controlled (n=15, both decoders
    # x both inits) the interaction is +6.27 mIoU, t=9.65, p=1.5e-07.
    # EuroSAT: DINO wins 46/47 (its primary modality IS rgb).
    # So there is no correct global value for this flag. Let best-by-val pick it
    # like any other hyperparameter (<=0.10 test regret on 7 of 8 DFC2020
    # upernet modalities; s1 is the one ambiguous case).
    if df.empty:
        return {}
    df['val_metric']  = pd.to_numeric(df['val_metric'],  errors='coerce')
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    result = {}
    for modality, grp in df.groupby('modality'):
        best = _best_by_val(grp, dataset)
        result[modality] = best['test_metric']
    return result


def _nbest(grp, col, n, dataset):
    """Top-n rows by `col`, respecting the metric direction (RMSE is lower-better)."""
    return (grp.nsmallest(n, col) if dataset in _LOWER_IS_BETTER_DS
            else grp.nlargest(n, col))[col]


# Hyperparameter columns that identify a CONFIG within one baseline family. Two
# rows sharing these values and differing in `seed` are seed replicates of the
# same run; anything else is a different configuration.
_CONFIG_COLS = [
    # baseline families
    'learning_rate', 'weight_decay', 'epoch', 'epochs',
    'temperature', 'alpha', 'distillation_mode', 'init_from_teacher',
    'lambda_u', 'lambda_e', 'ema_momentum', 'use_quantile', 'clip_thresh',
    'no_strong_aug', 'K', 'mixmatch_warmup_epochs', 'use_dino_weights',
    # Delulu. Its CSVs share NO hyperparameter column name with the baselines
    # (lr not learning_rate, and the lambda_*/mask/dropout knobs are its own),
    # so without these _seed_agg finds no config columns and falls back to the
    # top-k-by-val path -- which is what it is meant to replace.
    #
    # config_label is deliberately EXCLUDED even though it looks identifying:
    # the crossconfig sweeps bake the seed into it
    # (dfc2020_addition_initrandom_s0/s1/s2), so grouping by it makes every
    # cell a singleton and hides the very replicates we want to average.
    'lr', 'lambda_latent', 'lambda_prefusion', 'lambda_distill',
    'mae_mask_ratio', 'modality_dropout', 'modality_dropout_startmod',
    'modality_dropout_newmod', 'labeled_frequency', 'labeled_start_fraction',
    'student_init', 'loss_balance', 'active_losses', 'use_mask_token',
    'latent_masked_only', 'protect_lrm', 'unprotect_starting_mod',
]


def _seed_agg(grp, val_col, test_col, dataset):
    """(mean, std, n, kind) over the val-winning config's SEED replicates.

    Returns None when `grp` has no seeded rows, so each caller can fall back to
    its existing top-k-by-val behaviour.

    Why this is separate from _nbest: before the seed columns existed, every +/-
    in these tables was the spread across DIFFERENT CONFIGS (for KD/TTM on
    dfc2020, literally the gap between lr 5e-4 and lr 1e-4, n=2). That is a
    hyperparameter-sensitivity bar, not the run-to-run noise a reader assumes a
    +/- means, and it was not comparable to Delulu's, which does average seeds.
    Here the config is chosen by val FIRST, then the std is taken across that
    one config's seeds only.
    """
    if 'seed' not in grp.columns:
        return None
    seeded = grp[grp['seed'].notna()]
    if seeded.empty:
        return None
    cfg = [c for c in _CONFIG_COLS if c in seeded.columns]
    if not cfg:
        return None
    # Rank configs by their mean val across seeds, then take the winner's seeds.
    by_cfg = seeded.groupby(cfg, dropna=False)[val_col].mean()
    if by_cfg.empty or by_cfg.isna().all():
        return None
    best = by_cfg.idxmin() if dataset in _LOWER_IS_BETTER_DS else by_cfg.idxmax()
    if not isinstance(best, tuple):
        best = (best,)
    mask = pd.Series(True, index=seeded.index)
    for c, v in zip(cfg, best):
        mask &= (seeded[c].isna() if pd.isna(v) else seeded[c] == v)
    vals = seeded[mask][test_col].dropna()
    if vals.empty:
        return None
    return (vals.mean(), vals.std(), len(vals), 'seed')


def _load_mke_addition(dataset, arch='evan_base'):
    df = _flat_frames('mke', dataset, arch)
    if df is None:
        df = _read_csv(f'res/baselines/mke/{dataset}.csv')
        if df is None:
            return {}
        df = df[df['model_type'] == arch]
    if df.empty or 'valchecked_test_metric' not in df.columns:
        return {}
    df = df.copy()
    df['valchecked_test_metric'] = pd.to_numeric(df['valchecked_test_metric'], errors='coerce')
    if 'seed' in df.columns:
        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
    # Older runs (the 3 biomassters rows) predate valchecked_test_metric and
    # leave it empty; fall back to the plain test metric so the cell is a number
    # rather than nan. Those rows are not val-selected -- noted in the audit.
    if df['valchecked_test_metric'].isna().all() and 'student_test_metric' in df.columns:
        df['valchecked_test_metric'] = pd.to_numeric(df['student_test_metric'], errors='coerce')
    result = {}
    for (teacher, student_mods), grp in df.groupby(['teacher_modality', 'student_modalities']):
        if '+' not in str(student_mods):
            continue
        parts     = [p.strip() for p in student_mods.split('+')]
        new_parts = [p for p in parts if p != teacher]
        if len(new_parts) != 1:
            continue
        seeded = _seed_agg(grp, 'valchecked_test_metric', 'valchecked_test_metric', dataset)
        if seeded is not None:
            result[(teacher, new_parts[0])] = seeded[:2]
            continue
        top3 = _nbest(grp, 'valchecked_test_metric', 3, dataset)
        result[(teacher, new_parts[0])] = (top3.mean(), top3.std())
    return result


def _load_mixmatch_peek(dataset, arch='evan_base', family='mixmatch'):
    """Semi-supervised peek baseline: modality -> (mean, sd) of top-3 by val.

    `family` selects mixmatch or freematch; both are teacher-free and share the
    same flat-CSV schema (best_val_metric / best_val_test_metric / lambda_u).
    """
    df = _flat_frames(family, dataset, arch)
    if df is None and family == 'mixmatch':
        df = _read_csv(f'res/baselines/mixmatch/baseline_mixmatch_{dataset}.csv')
        if df is None:
            return {}
        df = df[df['model_type'] == arch]
    if df is None:
        return {}
    if df.empty or 'best_val_metric' not in df.columns:
        return {}
    df = df.copy()
    df['best_val_metric']      = pd.to_numeric(df['best_val_metric'],      errors='coerce')
    df['best_val_test_metric'] = pd.to_numeric(df['best_val_test_metric'], errors='coerce')
    if 'seed' in df.columns:
        df['seed'] = pd.to_numeric(df['seed'], errors='coerce')
    result = {}
    for modality, grp in df.groupby('modality'):
        seeded = _seed_agg(grp, 'best_val_metric', 'best_val_test_metric', dataset)
        if seeded is not None:
            result[modality] = seeded[:2]
            continue
        # These flat files are SWEEPS over lambda_u, and lambda_u is the single
        # biggest driver of the score (dfc2020 s2_norgb: 59.4 at 0.5 down to 5.2
        # at 75). Averaging the top-3 val rows therefore mixes a good config with
        # a collapsed one and reports a hyperparameter sweep's spread as if it
        # were seed noise. Pick the best lambda_u by val, then average its seeds.
        if 'lambda_u' in grp.columns:
            by_lu = grp.groupby('lambda_u')['best_val_metric'].mean()
            best_lu = by_lu.idxmin() if dataset in _LOWER_IS_BETTER_DS else by_lu.idxmax()
            grp = grp[grp['lambda_u'] == best_lu]
        sel  = (grp.nsmallest(3, 'best_val_metric') if dataset in _LOWER_IS_BETTER_DS
                else grp.nlargest(3, 'best_val_metric'))
        top3 = sel['best_val_test_metric']
        result[modality] = (top3.mean(), top3.std())
    return result


def _load_sft_combined_dino(dataset, arch='evan_base'):
    """Supervised SFT on combined modality for given arch (init chosen by val)."""
    df = _sft_frame(dataset, arch)
    if df is None:
        return {}
    df = df.copy()
    # NOTE: do NOT filter to dino_init == True. Which init wins flips with the
    # DECODER, not the train split. DFC2020, paired over matched
    # (modality, epoch, lr, wd):
    #     linear  : DINO wins,   mean -5.3 to -5.9 mIoU, 47/48 pairs
    #     upernet : random wins, mean +2.1 to +2.6 mIoU, 37/45 pairs
    # Both splits agree within a decoder; fully controlled (n=15, both decoders
    # x both inits) the interaction is +6.27 mIoU, t=9.65, p=1.5e-07.
    # EuroSAT: DINO wins 46/47 (its primary modality IS rgb).
    # So there is no correct global value for this flag. Let best-by-val pick it
    # like any other hyperparameter (<=0.10 test regret on 7 of 8 DFC2020
    # upernet modalities; s1 is the one ambiguous case).
    if df.empty:
        return {}
    df['val_metric']  = pd.to_numeric(df['val_metric'],  errors='coerce')
    df['test_metric'] = pd.to_numeric(df['test_metric'], errors='coerce')
    result = {}
    for modality, grp in df.groupby('modality'):
        if '+' not in str(modality):
            continue
        best = _best_by_val(grp, dataset)
        result[modality] = best['test_metric']
    return result


# ---------------------------------------------------------------------------
# Arch filtering
# ---------------------------------------------------------------------------

_ARCH_B_RE = re.compile(r'-B(?:\(|$)')
_ARCH_L_RE = re.compile(r'-L(?:\(|$)')


def _col_arch(col):
    """Return 'B', 'L', or None (arch-neutral) for a column name."""
    if _ARCH_B_RE.search(col):
        return 'B'
    if _ARCH_L_RE.search(col):
        return 'L'
    return None


def _filter_arch_cols(df, arch):
    """Drop columns that belong to the excluded arch. arch in {'B','L','BL'}."""
    if arch == 'BL':
        return df
    return df[[c for c in df.columns if _col_arch(c) != ('L' if arch == 'B' else 'B')]]


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def build_transfer_BL(dataset, arch='BL', ignore_select_by=False):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    dist_b = _load_distillation(dataset, 'evan_base')
    dist_l = _load_distillation(dataset, 'evan_large')
    del_b  = _load_delulu(dataset, 'evan_base',  'valchecked_val_transfer', 'valchecked_transfer', ignore_select_by=ignore_select_by)
    del_l  = _load_delulu(dataset, 'evan_large', 'valchecked_val_transfer', 'valchecked_transfer', ignore_select_by=ignore_select_by)
    sft_b  = _load_sft_dino(dataset, 'evan_base')
    sft_l  = _load_sft_dino(dataset, 'evan_large')
    rsfm   = _load_rsfm(dataset)

    rows = []
    for (start, new) in transfers:
        start_d = MOD_DISPLAY.get(start, start)
        new_d   = MOD_DISPLAY.get(new,   new)
        rows.append({
            'Dataset':                 DATASET_DISPLAY[dataset],
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
    return _filter_arch_cols(pd.DataFrame(rows), arch)


def build_peek_BL(dataset, arch='BL', ignore_select_by=False):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    mm_b  = _load_mixmatch_peek(dataset, 'evan_base')
    mm_l  = _load_mixmatch_peek(dataset, 'evan_large')
    fm_b  = _load_mixmatch_peek(dataset, 'evan_base',  family='freematch')
    fm_l  = _load_mixmatch_peek(dataset, 'evan_large', family='freematch')
    del_b = _load_delulu(dataset, 'evan_base',  'valchecked_val_peek', 'valchecked_peek', ignore_select_by=ignore_select_by)
    del_l = _load_delulu(dataset, 'evan_large', 'valchecked_val_peek', 'valchecked_peek', ignore_select_by=ignore_select_by)
    sft_b = _load_sft_dino(dataset, 'evan_base')
    sft_l = _load_sft_dino(dataset, 'evan_large')
    rsfm  = _load_rsfm(dataset)

    rows = []
    for (start, new) in transfers:
        start_d = MOD_DISPLAY.get(start, start)
        new_d   = MOD_DISPLAY.get(new,   new)
        rows.append({
            'Dataset':        DATASET_DISPLAY[dataset],
            'Start(M_A)':     start_d,
            'New(M_B)':       new_d,
            'DINO-SFT-B':     _fmt(sft_b.get(start)),
            'DINO-SFT-L':     _fmt(sft_l.get(start)),
            'MixMatch-B':     _fmt_meanstd(mm_b.get(start)),
            'MixMatch-L':     _fmt_meanstd(mm_l.get(start)),
            'FreeMatch-B':    _fmt_meanstd(fm_b.get(start)),
            'FreeMatch-L':    _fmt_meanstd(fm_l.get(start)),
            'Delulu-B':       _fmt_meanstd(del_b.get((start, new))),
            'Delulu-L':       _fmt_meanstd(del_l.get((start, new))),
            'Panopticon-B':   _fmt(rsfm.get(('panopticon',     start))),
            'OlmoEarth-B':    _fmt(rsfm.get(('olmoearth-base', start))),
            'OlmoEarth-L':    _fmt(rsfm.get(('olmoearth-large',start))),
        })
    return _filter_arch_cols(pd.DataFrame(rows), arch)


def build_addition_BL(dataset, arch='BL', distillation_ens=False, ignore_select_by=False):
    transfers = VALID_TRANSFERS.get(dataset, [])
    if not transfers:
        return None

    mke_b     = _load_mke_addition(dataset, 'evan_base')
    mke_l     = _load_mke_addition(dataset, 'evan_large')
    del_b     = _load_delulu(dataset, 'evan_base',  'val_addition',     'test_addition',     ignore_select_by=ignore_select_by)
    del_l     = _load_delulu(dataset, 'evan_large', 'val_addition',     'test_addition',     ignore_select_by=ignore_select_by)
    del_ens_b = _load_delulu(dataset, 'evan_base',  'valchecked_val_add_ens', 'valchecked_add_ens', ignore_select_by=ignore_select_by)
    del_ens_l = _load_delulu(dataset, 'evan_large', 'valchecked_val_add_ens', 'valchecked_add_ens', ignore_select_by=ignore_select_by)
    sft_b     = _load_sft_dino(dataset, 'evan_base')
    sft_l     = _load_sft_dino(dataset, 'evan_large')
    comb_b    = _load_sft_combined_dino(dataset, 'evan_base')
    comb_l    = _load_sft_combined_dino(dataset, 'evan_large')
    rsfm      = _load_rsfm(dataset)
    dist_loader = _load_distillation_ens if distillation_ens else _load_distillation
    dist_b    = dist_loader(dataset, 'evan_base')
    dist_l    = dist_loader(dataset, 'evan_large')

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
            'Dataset':                   DATASET_DISPLAY[dataset],
            'Start→New':                 f'{start_d}→{new_d}',
            'DINO-SFT-B(M_A)':           _fmt(sft_b.get(start)),
            'DINO-SFT-L(M_A)':           _fmt(sft_l.get(start)),
            'MKE-B':                     _fmt_meanstd(mke_b.get((start, new))),
            'MKE-L':                     _fmt_meanstd(mke_l.get((start, new))),
            **({'KD-ens-B':              _fmt_meanstd(dist_b.get((start, new, 'kd'))),
                'KD-ens-L':              _fmt_meanstd(dist_l.get((start, new, 'kd'))),
                'TTM-ens-B':             _fmt_meanstd(dist_b.get((start, new, 'ttm'))),
                'TTM-ens-L':             _fmt_meanstd(dist_l.get((start, new, 'ttm'))),
               } if distillation_ens else {}),
            'Delulu-B':                  _fmt_meanstd(del_b.get((start, new))),
            'Delulu-L':                  _fmt_meanstd(del_l.get((start, new))),
            **({'Delulu-ens-B':          _fmt_meanstd(del_ens_b.get((start, new))),
                'Delulu-ens-L':          _fmt_meanstd(del_ens_l.get((start, new))),
               } if distillation_ens else {}),
            'DINO-SFT-B(M_A+M_B ora)':   _fmt(lsft_b),
            'DINO-SFT-L(M_A+M_B ora)':   _fmt(lsft_l),
            'Panopticon-B':              _fmt(pan_score),
            'OlmoEarth-B':               _fmt(olmo_b),
            'OlmoEarth-L':               _fmt(olmo_l),
        })
    return _filter_arch_cols(pd.DataFrame(rows), arch)


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


def _gray(s):
    return r'\textcolor{gray}{' + str(s) + '}'


def _multirow(n, s):
    return rf'\multirow{{{n}}}{{*}}{{{s}}}'


def _multicolumn(n, align, s):
    return rf'\multicolumn{{{n}}}{{{align}}}{{{s}}}'


def _num(s):
    if pd.isna(s) or str(s).strip() == '--':
        return float('nan')
    m = re.match(r'[-+]?\d*\.?\d+', str(s).strip())
    return float(m.group()) if m else float('nan')


# Metrics where a LOWER number is better. BioMassters is AGB regression scored
# by RMSE; every other dataset here (mAP / mIoU / Acc) is higher-is-better.
LOWER_IS_BETTER_DATASETS = ('biomassters',)


def _row_lower_is_better(row):
    r"""True when this row's dataset is scored by a lower-is-better metric.

    The Dataset cell holds a LaTeX display string (e.g. a \shortstack with
    'BioMassters' and 'RMSE' in it), so match on the rendered text rather than a
    dataset key -- that is all that survives into the table frame.
    """
    cell = str(row.get('Dataset', ''))
    low = cell.lower()
    return any(d in low for d in LOWER_IS_BETTER_DATASETS) or 'rmse' in low


def _bold_max_per_row(df, cols):
    """Bold the BEST cell per row, respecting the metric's direction.

    Previously this always took max(), which silently bolds the WORST cell for
    an RMSE row. See LOWER_IS_BETTER_DATASETS.
    """
    df = df.copy()
    for c in cols:
        df[c] = df[c].astype(object)
    for i, row in df.iterrows():
        vals  = {c: _num(row[c]) for c in cols}
        valid = {c: v for c, v in vals.items() if not np.isnan(v)}
        if not valid:
            continue
        pick = min if _row_lower_is_better(row) else max
        best_col = pick(valid, key=valid.__getitem__)
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


def _df_to_latex_rows(df, merge_cols, gray_cols=None):
    df = df.copy()
    merged = {col: _apply_multirow(list(df[col])) for col in merge_cols}
    gray_set = set(gray_cols or [])
    rows = []
    for i in range(len(df)):
        row = []
        for col in df.columns:
            if col in merged:
                row.append(merged[col][i])
            else:
                val = _escape(str(df.at[i, col]))
                if col in gray_set and val != '--':
                    val = _gray(val)
                row.append(val)
        rows.append(row)
    return rows


def _rows_to_tex(rows, col_spec, header_rows, midrule_after=None, cmidrule_after=None):
    """midrule_after: set of row indices after which to insert \\midrule.
    cmidrule_after: dict of {row_index: '2-N'} for \\cmidrule (skips col 1 = Dataset)."""
    lines = [r'\begin{tabular}{' + col_spec + '}', r'\toprule']
    lines += header_rows
    lines.append(r'\midrule')
    for i, row in enumerate(rows):
        lines.append('  ' + ' & '.join(row) + r' \\')
        if midrule_after and i in midrule_after:
            lines.append(r'  \midrule')
        elif cmidrule_after and i in cmidrule_after:
            lines.append(r'  \cmidrule{' + cmidrule_after[i] + '}')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def _midrule_after_datasets(df):
    col = list(df['Dataset'])
    s = set()
    for i in range(len(col) - 1):
        if col[i] != col[i + 1]:
            s.add(i)
    return s


_THICK_VRULE = '|'
_ARCH_SUFFIX_RE = re.compile(r'[ \\\\]*\([BL]\)')


def _strip_arch_suffix(display_dict, arch):
    """Remove (B)/(L) suffixes (with preceding space or \\) from headers for single-arch tables."""
    if arch == 'BL':
        return display_dict
    return {k: _ARCH_SUFFIX_RE.sub('', v) for k, v in display_dict.items()}


def _data_col_spec(n_f0, n_baseline, n_ours, n_oracle):
    """f0+baseline+ours run together; single | before oracle."""
    n_before = n_f0 + n_baseline + n_ours
    spec = ' '.join(['c'] * n_before)
    if n_oracle:
        if spec:
            spec += ' | '
        spec += ' '.join(['c'] * n_oracle)
    return spec


def _midrule_after_start_groups(df, ncols, start_col='Start(M_A)'):
    """cmidrule from col 2 to ncols after every start-mod change within a dataset."""
    ds = list(df['Dataset'])
    st = list(df[start_col])
    d = {}
    for i in range(len(ds) - 1):
        if ds[i] == ds[i + 1] and st[i] != st[i + 1]:
            d[i] = f'2-{ncols}'
    return d


# ---------------------------------------------------------------------------
# LaTeX table makers
# ---------------------------------------------------------------------------

def make_transfer_tex(df, arch='BL'):
    bold_cols = [c for c in ['KD-B', 'KD-L', 'TTM-B', 'TTM-L', 'Delulu-B', 'Delulu-L'] if c in df.columns]
    df = _bold_max_per_row(df, bold_cols)
    oracle_cols = [c for c in df.columns if 'oracle' in c.lower() or c in ('Panopticon-B', 'OlmoEarth-B', 'OlmoEarth-L')]
    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start(M_A)'], gray_cols=oracle_cols)
    ncols = len(df.columns)
    midrule  = _midrule_after_datasets(df)
    cmidrule = _midrule_after_start_groups(df, ncols)

    n_f0       = sum(1 for c in ['DINO-SFT-B(M_A)', 'DINO-SFT-L(M_A)'] if c in df.columns)
    n_baseline = sum(1 for c in ['KD-B', 'KD-L', 'TTM-B', 'TTM-L'] if c in df.columns)
    n_ours     = sum(1 for c in ['Delulu-B', 'Delulu-L'] if c in df.columns)
    n_oracle   = sum(1 for c in ['DINO-SFT-B(M_B oracle)', 'DINO-SFT-L(M_B oracle)',
                                  'Panopticon-B', 'OlmoEarth-B', 'OlmoEarth-L'] if c in df.columns)

    # f0 separated from baseline+ours by | to signal bold doesn't include f0
    f0_spec  = ' '.join(['c'] * n_f0) + (' | ' if n_f0 and (n_baseline + n_ours) else '')
    rest_spec = _data_col_spec(0, n_baseline, n_ours, n_oracle)
    col_spec = 'c | cc | ' + f0_spec + rest_spec

    super_parts = [_multicolumn(3, 'c', '')]
    if n_f0:       super_parts.append(_multicolumn(n_f0,       'c|', r'$f_0(M_A)$'))
    if n_baseline: super_parts.append(_multicolumn(n_baseline, 'c|', 'Baselines'))
    if n_ours:     super_parts.append(_multicolumn(n_ours,     'c|', 'Ours'))
    if n_oracle:   super_parts.append(_multicolumn(n_oracle,   'c',  r'\shortstack[c]{Oracle\\($M_B$)}'))
    super_row = ' & '.join(super_parts) + r' \\'

    DISPLAY = {
        'Dataset':                 'Dataset',
        'Start(M_A)':              r'\shortstack[c]{Start\\($M_A$)}',
        'Transfer(M_B)':           r'\shortstack[c]{Transfer\\($M_B$)}',
        'DINO-SFT-B(M_A)':         r'\shortstack[c]{DINO\\v3 (B)}',
        'DINO-SFT-L(M_A)':         r'\shortstack[c]{DINO\\v3 (L)}',
        'KD-B':                    r'\shortstack[c]{KD\\(B)}',
        'KD-L':                    r'\shortstack[c]{KD\\(L)}',
        'TTM-B':                   r'\shortstack[c]{TTM\\(B)}',
        'TTM-L':                   r'\shortstack[c]{TTM\\(L)}',
        'Delulu-B':                r'\shortstack[c]{Delulu\\(B)}',
        'Delulu-L':                r'\shortstack[c]{Delulu\\(L)}',
        'DINO-SFT-B(M_B oracle)':  r'\shortstack[c]{DINO\\v3 (B)}',
        'DINO-SFT-L(M_B oracle)':  r'\shortstack[c]{DINO\\v3 (L)}',
        'Panopticon-B':            r'\shortstack[c]{Panop.\\(B)}',
        'OlmoEarth-B':             r'\shortstack[c]{OlmoE.\\(B)}',
        'OlmoEarth-L':             r'\shortstack[c]{OlmoE.\\(L)}',
    }
    DISPLAY = _strip_arch_suffix(DISPLAY, arch)
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'
    return _rows_to_tex(rows, col_spec, [super_row, r'\midrule', col_row], midrule)


def make_peek_tex(df, arch='BL'):
    baseline_cols = [c for c in ['DINO-SFT-B', 'DINO-SFT-L', 'MixMatch-B', 'MixMatch-L',
                                 'FreeMatch-B', 'FreeMatch-L', 'Delulu-B', 'Delulu-L'] if c in df.columns]
    df = _bold_max_per_row(df, baseline_cols)
    oracle_cols = [c for c in df.columns if c in ('Panopticon-B', 'OlmoEarth-B', 'OlmoEarth-L')]
    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start(M_A)'], gray_cols=oracle_cols)
    ncols = len(df.columns)
    midrule  = _midrule_after_datasets(df)
    cmidrule = _midrule_after_start_groups(df, ncols)

    n_f0       = sum(1 for c in ['DINO-SFT-B', 'DINO-SFT-L'] if c in df.columns)
    n_baseline = sum(1 for c in ['MixMatch-B', 'MixMatch-L',
                                 'FreeMatch-B', 'FreeMatch-L'] if c in df.columns)
    n_ours     = sum(1 for c in ['Delulu-B', 'Delulu-L'] if c in df.columns)
    n_oracle   = sum(1 for c in ['Panopticon-B', 'OlmoEarth-B', 'OlmoEarth-L'] if c in df.columns)

    col_spec  = 'c | cc | ' + _data_col_spec(n_f0, n_baseline, n_ours, n_oracle)

    super_parts = [_multicolumn(3, 'c', '')]
    if n_f0:       super_parts.append(_multicolumn(n_f0,       'c|', r'$f_0(M_A)$'))
    if n_baseline: super_parts.append(_multicolumn(n_baseline, 'c|', 'Baselines'))
    if n_ours:     super_parts.append(_multicolumn(n_ours,     'c|', 'Ours'))
    if n_oracle:   super_parts.append(_multicolumn(n_oracle,   'c',  r'\shortstack[c]{Oracle\\($M_A$)}'))
    super_row = ' & '.join(super_parts) + r' \\'

    DISPLAY = {
        'Dataset':      'Dataset',
        'Start(M_A)':   r'\shortstack[c]{Start\\($M_A$)}',
        'New(M_B)':     r'\shortstack[c]{New\\($M_B$)}',
        'DINO-SFT-B':   r'\shortstack[c]{DINO\\v3 (B)}',
        'DINO-SFT-L':   r'\shortstack[c]{DINO\\v3 (L)}',
        'MixMatch-B':   r'\shortstack[c]{MixMatch\\(B)}',
        'MixMatch-L':   r'\shortstack[c]{MixMatch\\(L)}',
        'FreeMatch-B':  r'\shortstack[c]{FreeMatch\\(B)}',
        'FreeMatch-L':  r'\shortstack[c]{FreeMatch\\(L)}',
        'Delulu-B':     r'\shortstack[c]{Delulu\\(B)}',
        'Delulu-L':     r'\shortstack[c]{Delulu\\(L)}',
        'Panopticon-B': r'\shortstack[c]{Panop.\\(B)}',
        'OlmoEarth-B':  r'\shortstack[c]{OlmoE.\\(B)}',
        'OlmoEarth-L':  r'\shortstack[c]{OlmoE.\\(L)}',
    }
    DISPLAY = _strip_arch_suffix(DISPLAY, arch)
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'
    return _rows_to_tex(rows, col_spec, [super_row, r'\midrule', col_row], midrule, cmidrule)


def make_addition_tex(df, arch='BL'):
    # Split Start→New into two id columns
    split = df['Start→New'].str.split('→', expand=True)
    df = df.copy()
    df.insert(df.columns.get_loc('Start→New'), 'Start(M_A)', split[0])
    df.insert(df.columns.get_loc('Start→New') + 1, 'New(M_B)', split[1])
    df = df.drop(columns=['Start→New'])

    baseline_cols = [c for c in ['DINO-SFT-B(M_A)', 'DINO-SFT-L(M_A)', 'MKE-B', 'MKE-L', 'Delulu-B', 'Delulu-L'] if c in df.columns]
    df = _bold_max_per_row(df, baseline_cols)
    oracle_cols = [c for c in df.columns if 'ora' in c.lower() or c in ('Panopticon-B', 'OlmoEarth-B', 'OlmoEarth-L')]
    rows = _df_to_latex_rows(df, merge_cols=['Dataset', 'Start(M_A)'], gray_cols=oracle_cols)
    ncols = len(df.columns)
    midrule  = _midrule_after_datasets(df)
    cmidrule = _midrule_after_start_groups(df, ncols)

    n_f0       = sum(1 for c in ['DINO-SFT-B(M_A)', 'DINO-SFT-L(M_A)'] if c in df.columns)
    n_baseline = sum(1 for c in ['MKE-B', 'MKE-L'] if c in df.columns)
    n_ours     = sum(1 for c in ['Delulu-B', 'Delulu-L'] if c in df.columns)
    n_oracle   = sum(1 for c in ['DINO-SFT-B(M_A+M_B ora)', 'DINO-SFT-L(M_A+M_B ora)',
                                  'Panopticon-B', 'OlmoEarth-B', 'OlmoEarth-L'] if c in df.columns)

    col_spec = 'c | cc | ' + _data_col_spec(n_f0, n_baseline, n_ours, n_oracle)

    super_parts = [_multicolumn(3, 'c', '')]
    if n_f0:       super_parts.append(_multicolumn(n_f0,       'c|', r'$f_0(M_A)$'))
    if n_baseline: super_parts.append(_multicolumn(n_baseline, 'c|', 'Baselines'))
    if n_ours:     super_parts.append(_multicolumn(n_ours,     'c|', 'Ours'))
    if n_oracle:   super_parts.append(_multicolumn(n_oracle,   'c',  r'\shortstack[c]{Oracle\\($M_A$+$M_B$)}'))
    super_row = ' & '.join(super_parts) + r' \\'

    DISPLAY = {
        'Dataset':                 'Dataset',
        'Start(M_A)':              r'\shortstack[c]{Start\\($M_A$)}',
        'New(M_B)':                r'\shortstack[c]{New\\($M_B$)}',
        'DINO-SFT-B(M_A)':         r'\shortstack[c]{DINO\\v3 (B)}',
        'DINO-SFT-L(M_A)':         r'\shortstack[c]{DINO\\v3 (L)}',
        'MKE-B':                   r'\shortstack[c]{MKE\\(B)}',
        'MKE-L':                   r'\shortstack[c]{MKE\\(L)}',
        'Delulu-B':                r'\shortstack[c]{Delulu\\(B)}',
        'Delulu-L':                r'\shortstack[c]{Delulu\\(L)}',
        'DINO-SFT-B(M_A+M_B ora)': r'\shortstack[c]{DINO\\v3 (B)}',
        'DINO-SFT-L(M_A+M_B ora)': r'\shortstack[c]{DINO\\v3 (L)}',
        'Panopticon-B':            r'\shortstack[c]{Panop.\\(B)}',
        'OlmoEarth-B':             r'\shortstack[c]{OlmoE.\\(B)}',
        'OlmoEarth-L':             r'\shortstack[c]{OlmoE.\\(L)}',
    }
    DISPLAY = _strip_arch_suffix(DISPLAY, arch)
    col_row = ' & '.join(DISPLAY[c] for c in df.columns) + r' \\'
    return _rows_to_tex(rows, col_spec, [super_row, r'\midrule', col_row], midrule, cmidrule)


# ---------------------------------------------------------------------------
# Tall (pivoted) table makers — methods as rows, columns = dataset→start→new
# ---------------------------------------------------------------------------

# Each entry: (method_label, group_label, col_B, col_L)
# col_L=None means no L variant (e.g. Panopticon).
_TRANSFER_METHOD_ROWS = [
    (r'$f_0$',       r'$f_0(M_A)$',                              'DINO-SFT-B(M_A)',        'DINO-SFT-L(M_A)'        ),
    ('KD',           'Baselines',                                 'KD-B',                    'KD-L'                   ),
    ('TTM',          'Baselines',                                 'TTM-B',                   'TTM-L'                  ),
    ('Delulu',       'Ours',                                      'Delulu-B',                'Delulu-L'               ),
    ('DINOv3',       r'\shortstack[c]{Oracle\\($M_B$)}',         'DINO-SFT-B(M_B oracle)',  'DINO-SFT-L(M_B oracle)' ),
    ('Panopticon',   r'\shortstack[c]{Oracle\\($M_B$)}',         'Panopticon-B',             None                    ),
    ('OlmoEarth',    r'\shortstack[c]{Oracle\\($M_B$)}',         'OlmoEarth-B',             'OlmoEarth-L'            ),
]

_PEEK_METHOD_ROWS = [
    (r'$f_0$',      r'$f_0(M_A)$',                               'DINO-SFT-B',  'DINO-SFT-L' ),
    ('MixMatch',    'Baselines',                                  'MixMatch-B',  'MixMatch-L' ),
    ('FreeMatch',   'Baselines',                                  'FreeMatch-B', 'FreeMatch-L'),
    ('Delulu',      'Ours',                                       'Delulu-B',    'Delulu-L'   ),
    ('Panopticon',  r'\shortstack[c]{Oracle\\($M_A$)}',          'Panopticon-B', None        ),
    ('OlmoEarth',   r'\shortstack[c]{Oracle\\($M_A$)}',          'OlmoEarth-B', 'OlmoEarth-L'),
]

_ADDITION_METHOD_ROWS = [
    (r'$f_0$',      r'$f_0(M_A)$',                               'DINO-SFT-B(M_A)',         'DINO-SFT-L(M_A)'        ),
    ('MKE',         'Baselines',                                  'MKE-B',                   'MKE-L'                  ),
    ('Delulu',      'Ours',                                       'Delulu-B',                'Delulu-L'               ),
    ('DINOv3',      r'\shortstack[c]{Oracle\\($M_A$+$M_B$)}',   'DINO-SFT-B(M_A+M_B ora)', 'DINO-SFT-L(M_A+M_B ora)'),
    ('Panopticon',  r'\shortstack[c]{Oracle\\($M_A$+$M_B$)}',   'Panopticon-B',             None                    ),
    ('OlmoEarth',   r'\shortstack[c]{Oracle\\($M_A$+$M_B$)}',   'OlmoEarth-B',             'OlmoEarth-L'            ),
]


# Datasets that do NOT get the Panopticon / OlmoEarth oracle rows. biomassters
# is excluded because OlmoEarth was never run there at all and Panopticon's rows
# sit at RMSE 67-76 against a 42 teacher, i.e. far worse than the method being
# compared -- an "oracle" that loses to everything is not an upper bound, so the
# rows only add noise. Its DINOv3 oracle stays.
_NO_RSFM_ORACLE_DATASETS = ('biomassters',)

_RSFM_ORACLE_METHODS = ('Panopticon', 'OlmoEarth')


def _drop_rsfm_oracle_rows(method_rows):
    return [r for r in method_rows if r[0] not in _RSFM_ORACLE_METHODS]


def _make_tall_tex(df, dataset_col, start_col, new_col, method_rows, arch='BL',
                   bold_excludes_f0=False, new_col_label=r'New ($M_B$)'):
    """Tall table: methods as rows, (dataset, start, new) as columns.

    bold_excludes_f0: if True, bold only across baseline+ours (transfer table logic).
    new_col_label: header label for the new-modality row.
    """
    keys = list(df[[dataset_col, start_col, new_col]].itertuples(index=False, name=None))

    # --- flatten method rows to (group, method, size_label, col_key) ---
    flat_rows = []
    for method_label, group, col_b, col_l in method_rows:
        if arch in ('B', 'BL'):
            flat_rows.append((group, method_label, '(B)', col_b))
        if col_l is not None and arch in ('L', 'BL'):
            flat_rows.append((group, method_label, '(L)', col_l))

    # group membership for bold: f0, baseline, ours, oracle
    _f0_groups     = {r'$f_0(M_A)$', r'$f_0$'}
    _oracle_groups = {g for _, g, _, _ in method_rows if 'Oracle' in g or 'oracle' in g}

    # --- bold: per column, find best among non-oracle (optionally also non-f0) rows ---
    # build a 2D list: cell_vals[flat_row_idx][key_idx] = raw numeric value
    cell_vals = []
    for group, method, size, col in flat_rows:
        row_vals = []
        for ds, start, new in keys:
            row = df[(df[dataset_col] == ds) & (df[start_col] == start) & (df[new_col] == new)]
            raw = str(row[col].iloc[0]) if not row.empty and col in df.columns else '--'
            row_vals.append(raw)
        cell_vals.append(row_vals)

    # for each key column, find the BEST among eligible rows.
    #
    # Direction is per COLUMN, not per table: a column belongs to one dataset,
    # and a tall table can place an RMSE dataset beside mAP/mIoU ones. Taking
    # max() unconditionally bolded the WORST method in every biomassters column
    # (e.g. MixMatch 59.2 bolded over Delulu 44.7). Same rule as the wide tables'
    # _bold_max_per_row; see LOWER_IS_BETTER_DATASETS.
    bold_mask = [[False] * len(keys) for _ in flat_rows]
    for ki in range(len(keys)):
        lower_better = _row_lower_is_better({'Dataset': keys[ki][0]})
        best_val = float('inf') if lower_better else float('-inf')
        for ri, (group, method, size, col) in enumerate(flat_rows):
            if group in _oracle_groups:
                continue
            if bold_excludes_f0 and group in _f0_groups:
                continue
            v = _num(cell_vals[ri][ki])
            if np.isnan(v):
                continue
            if v < best_val if lower_better else v > best_val:
                best_val = v
        if best_val in (float('inf'), float('-inf')):
            continue
        for ri, (group, method, size, col) in enumerate(flat_rows):
            if group in _oracle_groups:
                continue
            if bold_excludes_f0 and group in _f0_groups:
                continue
            if _num(cell_vals[ri][ki]) == best_val:
                bold_mask[ri][ki] = True

    # --- column spec ---
    # id cols: group (hidden when single arch and Size col dropped), method, [size]
    show_size = (arch == 'BL')
    n_id = 3 if show_size else 2

    ds_groups = []
    for ds, start, new in keys:
        if not ds_groups or ds_groups[-1][0] != ds:
            ds_groups.append((ds, 0))
        ds_groups[-1] = (ds_groups[-1][0], ds_groups[-1][1] + 1)
    col_spec = 'c' * n_id + ' | ' + ' | '.join(' '.join(['c'] * cnt) for _, cnt in ds_groups)

    # --- header row 1: dataset ---
    ds_spans = {}
    for ds, start, new in keys:
        ds_spans[ds] = ds_spans.get(ds, 0) + 1
    h1_cells = [_multicolumn(n_id, 'c|', r'\shortstack[c]{Dataset\\(metric)}')]
    for i, (ds, span) in enumerate(ds_spans.items()):
        align = 'c|' if i < len(ds_spans) - 1 else 'c'
        h1_cells.append(_multicolumn(span, align, _escape(ds)))
    header1 = ' & '.join(h1_cells) + r' \\'

    # --- header row 2: start_mod (M_A) spans ---
    start_runs = []
    for ds, start, new in keys:
        if not start_runs or (start_runs[-1][0], start_runs[-1][1]) != (ds, start):
            start_runs.append([ds, start, 0])
        start_runs[-1][2] += 1

    h2_cells = [_multicolumn(n_id, 'c|', r'Start ($M_A$)')]
    for i, (ds, start, span) in enumerate(start_runs):
        is_last_in_ds   = (i == len(start_runs) - 1) or (start_runs[i + 1][0] != ds)
        is_last_overall = (i == len(start_runs) - 1)
        align = 'c' if is_last_overall else ('c|' if is_last_in_ds else 'c')
        h2_cells.append(_multicolumn(span, align, _escape(start)))
    header2 = ' & '.join(h2_cells) + r' \\'

    # --- header row 3: new_mod leaf labels ---
    h3_cells = [_multicolumn(n_id, 'c|', new_col_label)]
    for ds, start, new in keys:
        h3_cells.append(_escape(new))
    header3 = ' & '.join(h3_cells) + r' \\'

    # --- body ---
    group_col       = [g for g, _, _, _ in flat_rows]
    method_col      = [m for _, m, _, _ in flat_rows]
    group_rendered  = _apply_multirow(group_col)
    method_rendered = _apply_multirow(method_col)

    # Group transitions that share the bold pool → no midrule between them.
    # bold_excludes_f0=True (transfer): f0 | baselines+ours | oracle
    # bold_excludes_f0=False (peek/addition): f0+baselines+ours | oracle
    _bold_pool_groups = set(group for _, group, _, _ in method_rows
                            if group not in _oracle_groups)
    if bold_excludes_f0:
        _bold_pool_groups -= _f0_groups  # f0 is separate in transfer

    def _same_bold_pool(g1, g2):
        return g1 in _bold_pool_groups and g2 in _bold_pool_groups

    lines = [r'\begin{tabular}{' + col_spec + '}', r'\toprule']
    lines += ['  ' + header1, r'  \midrule', '  ' + header2, '  ' + header3, r'  \midrule']

    prev_group = None
    for i, (group, method, size, col) in enumerate(flat_rows):
        if prev_group is not None and group != prev_group and not _same_bold_pool(prev_group, group):
            lines.append(r'  \midrule')
        cells = []
        for ki, (ds, start, new) in enumerate(keys):
            raw = cell_vals[i][ki]
            val = _escape(raw)
            if bold_mask[i][ki]:
                val = _bold(val)
            if group in _oracle_groups and val != '--':
                val = _gray(val)
            cells.append(val)
        id_cells = [group_rendered[i], method_rendered[i]]
        if show_size:
            id_cells.append(size)
        lines.append('  ' + ' & '.join(id_cells + cells) + r' \\')
        prev_group = group

    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def _rows_for(method_rows):
    """Drop the RSFM oracle rows when every rendered dataset opts out of them."""
    if DATASETS and all(d in _NO_RSFM_ORACLE_DATASETS for d in DATASETS):
        return _drop_rsfm_oracle_rows(method_rows)
    return method_rows


def make_transfer_tex_tall(df, arch='BL'):
    return _make_tall_tex(df, 'Dataset', 'Start(M_A)', 'Transfer(M_B)', _rows_for(_TRANSFER_METHOD_ROWS),
                          arch=arch, bold_excludes_f0=True, new_col_label=r'Transfer ($M_B$)')


def make_peek_tex_tall(df, arch='BL'):
    return _make_tall_tex(df, 'Dataset', 'Start(M_A)', 'New(M_B)', _rows_for(_PEEK_METHOD_ROWS),
                          arch=arch, bold_excludes_f0=False, new_col_label=r'New ($M_B$)')


def make_addition_tex_tall(df, arch='BL'):
    split = df['Start→New'].str.split('→', expand=True)
    df = df.copy()
    df['_start'] = split[0]
    df['_new']   = split[1]
    return _make_tall_tex(df, 'Dataset', '_start', '_new', _rows_for(_ADDITION_METHOD_ROWS),
                          arch=arch, bold_excludes_f0=False, new_col_label=r'New ($M_B$)')


# ---------------------------------------------------------------------------
# EuroSAT full-direction tables (RGB / VRE / NIR, all 6 directions)
# ---------------------------------------------------------------------------

VALID_TRANSFERS_EUROSAT = [
    ('rgb', 'vre'), ('rgb', 'nir'),
    ('vre', 'rgb'), ('vre', 'nir'),
    ('nir', 'rgb'), ('nir', 'vre'),
]

COMBINED_RSFM_ALIASES_EUROSAT = {
    ('rgb', 'vre'): ['rgb+vre', 'vre+rgb'],
    ('rgb', 'nir'): ['rgb+nir', 'nir+rgb'],
    ('vre', 'rgb'): ['vre+rgb', 'rgb+vre'],
    ('vre', 'nir'): ['vre+nir', 'nir+vre'],
    ('nir', 'rgb'): ['nir+rgb', 'rgb+nir'],
    ('nir', 'vre'): ['nir+vre', 'vre+nir'],
}


def build_transfer_eurosat(arch='BL', ignore_select_by=False):
    dist_b = _load_distillation('eurosat', 'evan_base')
    del_b  = _load_delulu('eurosat', 'evan_base', 'valchecked_val_transfer', 'valchecked_transfer', ignore_select_by=ignore_select_by)
    sft_b  = _load_sft_dino('eurosat', 'evan_base')
    rows = []
    for (start, new) in VALID_TRANSFERS_EUROSAT:
        rows.append({
            'Start(M_A)':             MOD_DISPLAY.get(start, start),
            'Transfer(M_B)':          MOD_DISPLAY.get(new,   new),
            'DINO-SFT-B(M_A)':        _fmt(sft_b.get(start)),
            'KD-B':                   _fmt_meanstd(dist_b.get((start, new, 'kd'))),
            'TTM-B':                  _fmt_meanstd(dist_b.get((start, new, 'ttm'))),
            'Delulu-B':               _fmt_meanstd(del_b.get((start, new))),
            'DINO-SFT-B(M_B oracle)': _fmt(sft_b.get(new)),
        })
    return _filter_arch_cols(pd.DataFrame(rows), arch)


def build_peek_eurosat(arch='BL', ignore_select_by=False):
    mm_b  = _load_mixmatch_peek('eurosat', 'evan_base')
    del_b = _load_delulu('eurosat', 'evan_base', 'valchecked_val_peek', 'valchecked_peek', ignore_select_by=ignore_select_by)
    sft_b = _load_sft_dino('eurosat', 'evan_base')
    rows = []
    for (start, new) in VALID_TRANSFERS_EUROSAT:
        rows.append({
            'Start(M_A)': MOD_DISPLAY.get(start, start),
            'New(M_B)':   MOD_DISPLAY.get(new,   new),
            'DINO-SFT-B': _fmt(sft_b.get(start)),
            'MixMatch-B': _fmt_meanstd(mm_b.get(start)),
            'Delulu-B':   _fmt_meanstd(del_b.get((start, new))),
        })
    return _filter_arch_cols(pd.DataFrame(rows), arch)


def build_addition_eurosat(arch='BL', distillation_ens=False, ignore_select_by=False):
    mke_b     = _load_mke_addition('eurosat', 'evan_base')
    del_b     = _load_delulu('eurosat', 'evan_base', 'val_addition',           'test_addition',         ignore_select_by=ignore_select_by)
    del_ens_b = _load_delulu('eurosat', 'evan_base', 'valchecked_val_add_ens', 'valchecked_add_ens',    ignore_select_by=ignore_select_by)
    sft_b     = _load_sft_dino('eurosat', 'evan_base')
    comb_b    = _load_sft_combined_dino('eurosat', 'evan_base')
    dist_loader = _load_distillation_ens if distillation_ens else _load_distillation
    dist_b    = dist_loader('eurosat', 'evan_base')
    rows = []
    for (start, new) in VALID_TRANSFERS_EUROSAT:
        lsft_b = comb_b.get(f'{start}+{new}') or comb_b.get(f'{new}+{start}')
        rows.append({
            'Start→New':               f'{MOD_DISPLAY.get(start,start)}→{MOD_DISPLAY.get(new,new)}',
            'DINO-SFT-B(M_A)':         _fmt(sft_b.get(start)),
            'MKE-B':                   _fmt_meanstd(mke_b.get((start, new))),
            **({'KD-ens-B':            _fmt_meanstd(dist_b.get((start, new, 'kd'))),
                'TTM-ens-B':           _fmt_meanstd(dist_b.get((start, new, 'ttm'))),
               } if distillation_ens else {}),
            'Delulu-B':                _fmt_meanstd(del_b.get((start, new))),
            **({'Delulu-ens-B':        _fmt_meanstd(del_ens_b.get((start, new))),
               } if distillation_ens else {}),
            'DINO-SFT-B(M_A+M_B ora)': _fmt(lsft_b),
        })
    return _filter_arch_cols(pd.DataFrame(rows), arch)


# Tall method-row definitions for eurosat (no L columns since evan_large coverage is sparse)
_TRANSFER_METHOD_ROWS_ES = [
    (r'$f_0$',  r'$f_0(M_A)$',                            'DINO-SFT-B(M_A)',        None),
    ('KD',      'Baselines',                               'KD-B',                   None),
    ('TTM',     'Baselines',                               'TTM-B',                  None),
    ('Delulu',  'Ours',                                    'Delulu-B',               None),
    ('DINOv3',  r'\shortstack[c]{Oracle\\($M_B$)}',        'DINO-SFT-B(M_B oracle)', None),
]

_PEEK_METHOD_ROWS_ES = [
    (r'$f_0$',   r'$f_0(M_A)$', 'DINO-SFT-B', None),
    ('MixMatch', 'Baselines',   'MixMatch-B',  None),
    ('Delulu',   'Ours',        'Delulu-B',    None),
]

_ADDITION_METHOD_ROWS_ES = [
    (r'$f_0$', r'$f_0(M_A)$',                            'DINO-SFT-B(M_A)',         None),
    ('MKE',    'Baselines',                               'MKE-B',                   None),
    ('Delulu', 'Ours',                                    'Delulu-B',                None),
    ('DINOv3', r'\shortstack[c]{Oracle\\($M_A$+$M_B$)}', 'DINO-SFT-B(M_A+M_B ora)', None),
]


def _make_tall_tex_eurosat(df, start_col, new_col, method_rows, arch='B',
                           bold_excludes_f0=False, new_col_label=r'New ($M_B$)'):
    """Wrapper around _make_tall_tex that inserts a dummy dataset col so the
    three-level header shows 'EuroSAT (Acc)' at top instead of repeating start_col."""
    df = df.copy()
    df['_ds'] = r'\shortstack[c]{EuroSAT\\(Acc)}'
    return _make_tall_tex(df, '_ds', start_col, new_col, method_rows,
                          arch=arch, bold_excludes_f0=bold_excludes_f0,
                          new_col_label=new_col_label)


def make_transfer_tex_tall_eurosat(df, arch='B'):
    return _make_tall_tex_eurosat(df, 'Start(M_A)', 'Transfer(M_B)',
                                  _TRANSFER_METHOD_ROWS_ES, arch=arch, bold_excludes_f0=True,
                                  new_col_label=r'Transfer ($M_B$)')


def make_peek_tex_tall_eurosat(df, arch='B'):
    return _make_tall_tex_eurosat(df, 'Start(M_A)', 'New(M_B)',
                                  _PEEK_METHOD_ROWS_ES, arch=arch, bold_excludes_f0=False,
                                  new_col_label=r'New ($M_B$)')


def make_addition_tex_tall_eurosat(df, arch='B'):
    split = df['Start→New'].str.split('→', expand=True)
    df = df.copy()
    df['_start'] = split[0]
    df['_new']   = split[1]
    return _make_tall_tex_eurosat(df, '_start', '_new',
                                  _ADDITION_METHOD_ROWS_ES, arch=arch, bold_excludes_f0=False,
                                  new_col_label=r'New ($M_B$)')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--wide', action='store_true',
                        help='Wide layout: transfers as columns, methods as rows (default is tall)')
    parser.add_argument('--arch', choices=['B', 'L', 'BL'], default='B',
                        help='Which architecture columns to include (B, L, or BL)')
    parser.add_argument('--apr21', action='store_true',
                        help='Use hptuned_apr21.csv')
    parser.add_argument('--may5', action='store_true',
                        help='Use hptuned_may5.csv')
    parser.add_argument('--distillation_ens', action='store_true',
                        help='Show KD-ens/TTM-ens and Delulu-ens columns in Addition table')
    parser.add_argument('--ignore_select_by', action='store_true',
                        help='Pool all configs per (start, new) ignoring select_by filter')
    parser.add_argument('--datasets', nargs='+', default=None, choices=ALL_DATASETS,
                        help='Render only these datasets (e.g. --datasets dfc2020 biomassters). '
                             'Empty cells are shown as -- rather than dropped, so the table '
                             'doubles as a coverage view. Writes res/latex/<ds>_tables*.tex.')
    args = parser.parse_args()

    global DATASETS
    if args.datasets:
        DATASETS = list(args.datasets)

    global DELULU_CSV
    if args.apr21:
        DELULU_CSV = 'res/delulu/hptuned_apr21.csv'
    elif args.may5:
        DELULU_CSV = 'res/delulu/hptuned_may5.csv'

    # ---- collect dataframes ----
    transfer_frames, peek_frames, addition_frames = [], [], []
    for dataset in DATASETS:
        df = build_transfer_BL(dataset, arch=args.arch, ignore_select_by=args.ignore_select_by)
        if df is not None and not df.empty:
            transfer_frames.append(df)
        df = build_peek_BL(dataset, arch=args.arch, ignore_select_by=args.ignore_select_by)
        if df is not None and not df.empty:
            peek_frames.append(df)
        df = build_addition_BL(dataset, arch=args.arch, distillation_ens=args.distillation_ens, ignore_select_by=args.ignore_select_by)
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

    arch_suffix = '' if args.arch == 'BL' else f'_{args.arch}'
    if args.wide:
        fns = [
            lambda df: make_transfer_tex(df, arch=args.arch),
            lambda df: make_peek_tex(df,     arch=args.arch),
            lambda df: make_addition_tex(df, arch=args.arch),
        ]
        out = f'{OUT_DIR}/BL_tables{arch_suffix}.tex'
    else:
        fns = [
            lambda df: make_transfer_tex_tall(df, arch=args.arch),
            lambda df: make_peek_tex_tall(df,     arch=args.arch),
            lambda df: make_addition_tex_tall(df, arch=args.arch),
        ]
        out = f'{OUT_DIR}/BL_tables_tall{arch_suffix}.tex'

    sections = []
    for label, frames, fn in zip(
        ['Transfer', 'Peek', 'Addition'],
        [transfer_frames, peek_frames, addition_frames],
        fns,
    ):
        if not frames:
            continue
        combined = pd.concat(frames, ignore_index=True)
        sections.append(f'% Table: {label}\n' + fn(combined))  # noqa: B023 (lambda in loop, intentional)

    with open(out, 'w') as f:
        f.write('\n\n\\bigskip\n\n'.join(sections) + '\n')
    print(f'\nwrote {out}')

    # ---- EuroSAT full-direction tables ----
    es_transfer = build_transfer_eurosat(arch=args.arch, ignore_select_by=args.ignore_select_by)
    es_peek     = build_peek_eurosat(arch=args.arch, ignore_select_by=args.ignore_select_by)
    es_addition = build_addition_eurosat(arch=args.arch, distillation_ens=args.distillation_ens, ignore_select_by=args.ignore_select_by)

    print(f'\n{"="*80}\n  EUROSAT TRANSFER\n{"="*80}')
    print(es_transfer.to_string(index=False))
    print(f'\n{"="*80}\n  EUROSAT PEEK\n{"="*80}')
    print(es_peek.to_string(index=False))
    print(f'\n{"="*80}\n  EUROSAT ADDITION\n{"="*80}')
    print(es_addition.to_string(index=False))

    es_sections = []
    for label, df, fn in [
        ('Transfer', es_transfer, lambda d: make_transfer_tex_tall_eurosat(d, arch=args.arch)),
        ('Peek',     es_peek,     lambda d: make_peek_tex_tall_eurosat(d,     arch=args.arch)),
        ('Addition', es_addition, lambda d: make_addition_tex_tall_eurosat(d, arch=args.arch)),
    ]:
        if df is not None and not df.empty:
            es_sections.append(f'% EuroSAT Table: {label}\n' + fn(df))

    es_out = f'{OUT_DIR}/eurosat_tables_tall{arch_suffix}.tex'
    with open(es_out, 'w') as f:
        f.write('\n\n\\bigskip\n\n'.join(es_sections) + '\n')
    print(f'wrote {es_out}')


if __name__ == '__main__':
    main()

# python res/results_BL.py
# python res/results_BL.py --ignore_select_by   --distillation_ens 