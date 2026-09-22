#!/usr/bin/env python3
"""Check the Spearman helpers in analysis/analyze_hallucination_correlation.py.

The rank transform is hand-rolled (torch has no rank primitive), so it is worth
pinning against scipy.stats.spearmanr -- particularly under ties, where an
average-rank convention is the whole point and an argsort alone gets it wrong.

Run inside a job env (needs torch + scipy):  python test_spearman_correlation.py
"""

import importlib.util
import os
import sys

import numpy as np
import torch
from scipy.stats import rankdata, spearmanr

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    'hc', os.path.join(_HERE, 'analysis', 'analyze_hallucination_correlation.py'))
hc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hc)

_fail = 0


def check(name, got, want, tol=1e-5):
    global _fail
    err = float(np.abs(np.asarray(got) - np.asarray(want)).max())
    ok = err <= tol
    _fail += (not ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: max_err={err:.3e}")


def check_true(name, cond, extra=''):
    global _fail
    _fail += (not cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}{(' ' + extra) if extra else ''}")


torch.manual_seed(0)

# ── Ranks match scipy, with and without ties, on either axis ────────────────
x = torch.randint(0, 4, (3, 5, 7)).float()          # heavy ties
check('ranks dim=-1 (ties)', hc._average_ranks(x, -1).numpy(),
      np.apply_along_axis(rankdata, -1, x.numpy()) - 1)     # scipy is 1-based

x0 = torch.randint(0, 3, (6, 4, 3)).float()
check('ranks dim=0 (ties)', hc._average_ranks(x0, 0).numpy(),
      np.apply_along_axis(rankdata, 0, x0.numpy()) - 1)

x = torch.randn(4, 9, 16)                            # no ties
check('ranks dim=-1 (no ties)', hc._average_ranks(x, -1).numpy(),
      np.apply_along_axis(rankdata, -1, x.numpy()) - 1)

# ── patch_spearman / batch_spearman match scipy elementwise ─────────────────
a = torch.randint(0, 5, (3, 4, 20)).float()
b = torch.randint(0, 5, (3, 4, 20)).float()
check('patch_spearman vs scipy (ties)', hc.patch_spearman(a, b).numpy(),
      [[spearmanr(a[i, j].numpy(), b[i, j].numpy()).statistic for j in range(4)]
       for i in range(3)], tol=1e-4)

a = torch.randn(3, 4, 32)
b = torch.randn(3, 4, 32)
check('patch_spearman vs scipy (cont)', hc.patch_spearman(a, b).numpy(),
      [[spearmanr(a[i, j].numpy(), b[i, j].numpy()).statistic for j in range(4)]
       for i in range(3)], tol=1e-4)

a = torch.randn(30, 4, 5)
b = torch.randn(30, 4, 5)
check('batch_spearman vs scipy', hc.batch_spearman(a, b).numpy(),
      [[spearmanr(a[:, j, k].numpy(), b[:, j, k].numpy()).statistic for k in range(5)]
       for j in range(4)], tol=1e-4)

# ── Invariances. The monotone case is the one that distinguishes rank from
# linear correlation, so assert the Pearson comparison is genuinely below 1. ─
a = torch.randn(8, 16, 32)
check('patch_spearman self == 1', hc.patch_spearman(a, a).numpy(), np.ones((8, 16)), tol=1e-4)
check('batch_spearman self == 1', hc.batch_spearman(a, a).numpy(), np.ones((16, 32)), tol=1e-4)
check('patch_spearman negated == -1', hc.patch_spearman(a, -a).numpy(), -np.ones((8, 16)), tol=1e-4)
check('patch_spearman monotone == 1', hc.patch_spearman(a, a.exp()).numpy(), np.ones((8, 16)), tol=1e-4)
check('batch_spearman monotone == 1', hc.batch_spearman(a, a.exp()).numpy(), np.ones((16, 32)), tol=1e-4)
_pp = hc.patch_pearson(a, a.exp()).mean().item()
check_true('monotone control: patch_pearson < 1', _pp < 0.99, f'(got {_pp:.4f})')

# ── Null floor: independent inputs score ~0 on both axes ────────────────────
u = torch.randn(400, 4, 8)
v = torch.randn(400, 4, 8)
for nm, fn in (('batch_spearman', hc.batch_spearman), ('patch_spearman', hc.patch_spearman)):
    m = fn(u, v).mean().item()
    check_true(f'{nm} null ~ 0', abs(m) < 0.02, f'(got {m:+.4f})')

# ── Shuffled controls: None below B=2, right shape above ────────────────────
one = torch.randn(1, 4, 8)
check_true('shuffled_patch_spearman B=1 -> None', hc.shuffled_patch_spearman(one, one) is None)
check_true('shuffled_batch_spearman B=1 -> None', hc.shuffled_batch_spearman(one, one) is None)
_sp = hc.shuffled_batch_spearman(u, v, seed=0)
check_true('shuffled_batch_spearman shape', _sp is not None and tuple(_sp.shape) == (4, 8))

# ── Constant tokens must not produce nan (they would poison later means) ────
const = torch.ones(5, 4, 8)
for nm, t in (('const patch', hc.patch_spearman(const, const)),
              ('const batch', hc.batch_spearman(const, const)),
              ('mixed patch', hc.patch_spearman(torch.randn(5, 4, 8), const))):
    check_true(f'{nm}: no nan', not bool(torch.isnan(t).any()))

# ── GPU and CPU agree ───────────────────────────────────────────────────────
if torch.cuda.is_available():
    g = torch.randn(20, 8, 16, device='cuda')
    h = torch.randn(20, 8, 16, device='cuda')
    check('cuda batch_spearman == cpu', hc.batch_spearman(g, h).cpu().numpy(),
          hc.batch_spearman(g.cpu(), h.cpu()).numpy(), tol=1e-4)
    check('cuda patch_spearman == cpu', hc.patch_spearman(g, h).cpu().numpy(),
          hc.patch_spearman(g.cpu(), h.cpu()).numpy(), tol=1e-4)
else:
    print('[skip] cuda checks (no gpu)')

print(f'\nFAILURES: {_fail}')
sys.exit(1 if _fail else 0)
