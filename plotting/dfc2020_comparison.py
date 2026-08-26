"""
DFC2020 / Copernicus-Bench comparison tables.

Design of the comparison
------------------------
Everything below is on the Copernicus-Bench official split (3156/986/986,
8 classes, upernet decoder unless noted).

  init / lower bound   SFT split1  — supervised on train1 only. This is the
                                     teacher every stage-1 method starts from,
                                     so it is the "do nothing" baseline.
  label oracle         SFT full    — supervised on train1+train2, i.e. the same
                                     data with train2's labels revealed. Upper
                                     bound for what unlabeled train2 could buy.
                                     Matched on TEST-TIME INPUT modality.

  transfer   (test on new mod only)      vs  distillation baseline
  peeking    (test on start mod, new mod seen unlabeled) vs freematch / mixmatch
  addition   (test on both mods)         vs  MKE baseline

`transfer` is evaluated with only the NEW modality at test time, so its oracle
is SFT(new_mod). `peeking` uses the START modality, oracle SFT(start).
`addition` uses both, oracle SFT(start+new).
"""
import csv, json, os
from collections import defaultdict

SFT = 'res/train_sft/dfc2020_cobench.csv'
SWEEP = 'res/delulu-sweep/sweep_recovered_dfc2020_split1.csv'
DEC = 'upernet'


def sft_best(modality, split, decoder=DEC):
    """Best-by-val test mIoU for an SFT config."""
    best = None
    for r in csv.DictReader(open(SFT)):
        if (r['modality'] == modality and r['train_split'] == split
                and r['decoder'] == decoder):
            v = float(r['val_metric'])
            if best is None or v > best[0]:
                best = (v, float(r['test_metric']))
    return best[1] if best else None


def load_baseline(name, decoder=DEC):
    """{(teacher_mod, new_mod): best test mIoU} for a teacher-based baseline."""
    f = f'res/baselines/dfc2020_cobench_{name}_{decoder}.csv'
    out = defaultdict(list)
    if not os.path.exists(f):
        return {}
    for r in csv.DictReader(open(f)):
        # student field is '+'-joined, e.g. 's1+s2_rgb' (teacher first)
        t = r['teacher_modality']
        field = r['student_modality'] if name == 'distillation' else r['student_modalities']
        mods = field.split('+')
        new = [m for m in mods if m != t]
        k = (t, new[0] if new else mods[-1])
        v = float(r['test_metric'] if name == 'distillation' else r['student_test_metric'])
        out[k].append(v)
    return {k: max(v) for k, v in out.items()}


def load_distill_transfer_sweep(decoder=DEC, init=None, kl=None):
    """{(teacher, new_mod): best test mIoU} from the 16-trial transfer HP sweep.

    `init` filters on init_from_teacher: 'False' = RANDOM init (baseline_distillation.py builds the student with
    load_weights=False, so there is no DINO prior),
    'True' = student starts from the teacher's weights. Reported separately
    because they differ systematically -- teacher-init is 2.5-5.9 mIoU WORSE in every
    direction than starting from scratch -- a transfer student appears to have to
    unlearn the teacher's modality-specific features before it can learn the new
    modality.
    """
    f = f'res/baselines/dfc2020_cobench_distill_transfer_sweep_{decoder}.csv'
    out = defaultdict(list)
    if not os.path.exists(f):
        return {}
    for r in csv.DictReader(open(f)):
        if init is not None and r.get('init_from_teacher') != init:
            continue
        if kl is not None and r.get('kl_type') != kl:
            continue
        out[(r['teacher_modality'], r['student_modality'])].append(float(r['test_metric']))
    return {k: max(v) for k, v in out.items()}


def load_distill_transfer(decoder=DEC, kl=None):
    """{(teacher, new_mod): best test mIoU} for the TRANSFER distillation runs.

    These use a UNIMODAL student on the new modality (--modalities <new>), which
    is the correct analogue of delulu's `transfer`. The other distillation file
    holds bimodal students and belongs under Addition.
    """
    f = f'res/baselines/dfc2020_cobench_distill_transfer_{decoder}.csv'
    out = defaultdict(list)
    if not os.path.exists(f):
        return {}
    for r in csv.DictReader(open(f)):
        if kl is not None and r.get('kl_type') != kl:
            continue
        out[(r['teacher_modality'], r['student_modality'])].append(float(r['test_metric']))
    return {k: max(v) for k, v in out.items()}


def load_semisl(name, decoder=DEC):
    """{modality: best val-selected test mIoU} — no teacher, single modality."""
    # MixMatch's default lambda_u=75 is mis-scaled for the dense CE objective
    # (see sh/mixmatch_lambdau_dfc2020.sh); prefer the lambda_u sweep when it
    # exists so the baseline is represented at its tuned setting.
    cands = [f'res/baselines/dfc2020_cobench_{name}_lambdau_{decoder}.csv',
             f'res/baselines/dfc2020_cobench_{name}_{decoder}.csv']
    out = defaultdict(list)
    for f in cands:
        if not os.path.exists(f):
            continue
        for r in csv.DictReader(open(f)):
            out[r['modality']].append(float(r['best_val_test_metric']))
    return {k: max(v) for k, v in out.items()}


SLUG2DIR = {
    's1_to_s2norgb':    ('s1', 's2_norgb'),
    's2rgb_to_s2norgb': ('s2_rgb', 's2_norgb'),
    's2norgb_to_s2rgb': ('s2_norgb', 's2_rgb'),
}


def load_delulu():
    """Best trial per direction, selected on val_addition (the sweep objective)."""
    best = {}
    for r in csv.DictReader(open(SWEEP)):
        d = SLUG2DIR[r['slug']]
        try:
            va = float(r['val_addition'])
        except (TypeError, ValueError):
            continue
        if d not in best or va > best[d][0]:
            best[d] = (va, {
                'transfer': float(r['test_transfer']),
                'peeking':  float(r['test_peeking']),
                'addition': float(r['test_addition']),
                'teacher':  float(r['teacher_test']),
                'lr': r['lr'], 'lambda_latent': r['lambda_latent'],
            })
    return {k: v[1] for k, v in best.items()}


def pair_name(a, b):
    """SFT modality key for the two-modality oracle, matching train_sft naming."""
    for cand in (f'{a}+{b}', f'{b}+{a}'):
        for r in csv.DictReader(open(SFT)):
            if r['modality'] == cand:
                return cand
    return None


def main():
    delulu = load_delulu()
    distill = load_baseline('distillation')
    mke = load_baseline('mke')
    fm = load_semisl('freematch')
    mm = load_semisl('mixmatch')

    dirs = sorted(delulu)
    W = 118

    print('=' * W)
    print('DFC2020 · Copernicus-Bench split · upernet decoder · test mIoU')
    print('SFT split1 = init / lower bound   |   SFT full = label oracle '
          '(train2 labels revealed)')
    print('=' * W)

    # ---- 1. TRANSFER: test on NEW modality only ----------------------------
    # No baseline belongs here. Both distillation and MKE build a BIMODAL
    # student (student_modality is always 'teacher+new'), so they are evaluated
    # on both modalities and have no new-modality-only arm. Putting them in this
    # panel compares different test-time inputs.
    ddino = load_distill_transfer_sweep(init='False')
    dt_kd = load_distill_transfer_sweep(init='True', kl='kd')
    dt_ttm = load_distill_transfer_sweep(init='True', kl='ttm')
    print('\n[1] TRANSFER  (test-time input = NEW modality only)')
    print('     distillation = UNIMODAL-student variant, best of the 16-trial HP sweep')
    print('     KD and TTM are the two KD variants (kl_type), reported separately')
    print(f"{'direction':24s} {'SFT teacher':>12s} {'KD':>8s} {'TTM':>8s} "
          f"{'delulu':>8s} {'SFT oracle(new,full)':>21s}   [rand-init]")
    for s, n in dirs:
        tch = sft_best(s, 'split1')          # the teacher the method starts from
        orc = sft_best(n, 'full')            # oracle on the STUDENT modality
        d = delulu[(s, n)]['transfer']
        a, b = dt_kd.get((s, n)), dt_ttm.get((s, n))
        rnd = ddino.get((s, n))
        print(f'{s+" -> +"+n:24s} {tch:12.2f} '
              f'{(f"{a:.2f}" if a else "--"):>8s} {(f"{b:.2f}" if b else "--"):>8s} '
              f'{d:8.2f} {orc:21.2f}   {(f"{rnd:.2f}" if rnd else "--"):>9s}')

    # ---- 2. PEEKING: test on START modality -> vs semi-supervised ----------
    print('\n[2] PEEKING  (test-time input = START modality; new modality seen '
          'unlabeled at train time)')
    print(f"{'direction':24s} {'init SFT(start)':>16s} {'freematch':>10s} "
          f"{'mixmatch':>9s} {'delulu':>9s} {'oracle SFT(start,full)':>23s}")
    for s, n in dirs:
        init = sft_best(s, 'split1')
        orc = sft_best(s, 'full')
        d = delulu[(s, n)]['peeking']
        print(f'{s+" -> +"+n:24s} {init:16.2f} {fm.get(s, float("nan")):10.2f} '
              f'{mm.get(s, float("nan")):9.2f} {d:9.2f} {orc:23.2f}')

    # ---- 3. ADDITION: test on BOTH -> vs MKE -------------------------------
    print('\n[3] ADDITION  (test-time input = BOTH modalities)')
    print('     distillation and MKE both belong here: their students are bimodal')
    print(f"{'direction':24s} {'init SFT(start)':>16s} {'distill':>9s} {'MKE':>9s} "
          f"{'delulu':>9s} {'oracle SFT(both,full)':>22s}")
    for s, n in dirs:
        init = sft_best(s, 'split1')
        pn = pair_name(s, n)
        orc = sft_best(pn, 'full') if pn else None
        d = delulu[(s, n)]['addition']
        bd, bm = distill.get((s, n)), mke.get((s, n))
        print(f'{s+" -> +"+n:24s} {init:16.2f} '
              f'{(f"{bd:.2f}" if bd else "--"):>9s} '
              f'{(f"{bm:.2f}" if bm else "--"):>9s} {d:9.2f} '
              f'{(f"{orc:.2f}" if orc else "n/a"):>22s}')

    print('\nbest delulu hyperparameters (selected on val_addition):')
    for s, n in dirs:
        d = delulu[(s, n)]
        print(f'  {s+" -> +"+n:24s} lr={d["lr"]}  lambda_latent={d["lambda_latent"]}')


if __name__ == '__main__':
    main()
