"""Three-panel comparison: transfer / peeking / addition vs their baselines."""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dfc2020_comparison import (sft_best, load_baseline, load_semisl,
                                load_delulu, pair_name)

OUT = Path(sys.argv[1] if len(sys.argv) > 1
           else "figs/dfc2020_cobench_comparison.png")

# validated categorical slots (dataviz validator, light surface, --pairs all: PASS)
C = {"blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "violet": "#4a3aa7"}
SURFACE, INK, INK2, INK3, GRID = "#fcfcfb", "#1a1a19", "#4a4a48", "#8a8a86", "#e3e3df"

delulu = load_delulu()
distill = load_baseline('distillation')
mke = load_baseline('mke')
fm, mm = load_semisl('freematch'), load_semisl('mixmatch')
DIRS = sorted(delulu)
LBL = {('s1','s2_norgb'):'s1 → +s2_norgb',
       ('s2_rgb','s2_norgb'):'s2_rgb → +s2_norgb',
       ('s2_norgb','s2_rgb'):'s2_norgb → +s2_rgb'}

# panel: (title, subtitle, series list of (name, colour, value_fn), init_fn, oracle_fn)
panels = [
 ("Transfer", "test input = NEW modality only (no comparable baseline)",
  [("DeluluNet",    C["orange"], lambda s,n: delulu[(s,n)]['transfer'])],
  lambda s,n: sft_best(n,'split1'), lambda s,n: sft_best(n,'full')),
 ("Peeking", "test input = START modality; new modality unlabeled at train",
  [("FreeMatch", C["aqua"],   lambda s,n: fm.get(s)),
   ("MixMatch",  "#8a6a3a",   lambda s,n: mm.get(s)),
   ("DeluluNet", C["orange"], lambda s,n: delulu[(s,n)]['peeking'])],
  lambda s,n: sft_best(s,'split1'), lambda s,n: sft_best(s,'full')),
 ("Addition", "test input = BOTH modalities — distillation & MKE students are bimodal",
  [("distillation", C["blue"],   lambda s,n: distill.get((s,n))),
   ("MKE",          C["violet"], lambda s,n: mke.get((s,n))),
   ("DeluluNet",    C["orange"], lambda s,n: delulu[(s,n)]['addition'])],
  lambda s,n: sft_best(s,'split1'),
  lambda s,n: sft_best(pair_name(s,n),'full') if pair_name(s,n) else None),
]

plt.rcParams.update({"font.size":9,"axes.facecolor":SURFACE,"figure.facecolor":SURFACE,
  "axes.edgecolor":GRID,"axes.labelcolor":INK2,"text.color":INK,"xtick.color":INK2,
  "ytick.color":INK2,"axes.titlecolor":INK,"axes.spines.top":False,"axes.spines.right":False})

fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.4), sharey=True)
fig.subplots_adjust(left=0.055, right=0.99, top=0.72, bottom=0.28, wspace=0.09)

allv = [v for _,_,ss,i_f,o_f in panels for s,n in DIRS
        for v in [i_f(s,n), o_f(s,n)] + [f(s,n) for _,_,f in ss] if v is not None]
lo, hi = min(allv), max(allv)
pad = (hi-lo)*0.10
# MixMatch collapses to ~6 mIoU and would flatten everything else;
# clip the axis to the informative band and annotate any bar below it.
ylim = (max(0.0, lo-pad), hi+pad)
FLOOR = 40.0
if lo < FLOOR: ylim = (FLOOR, hi+pad)

for ax, (title, sub, series, init_fn, orc_fn) in zip(axes, panels):
    nb = len(series)
    w = 0.72/nb
    for i,(s,n) in enumerate(DIRS):
        init, orc = init_fn(s,n), orc_fn(s,n)
        # init (lower bound) and oracle (upper bound) as a shaded span + rules
        if init is not None and orc is not None:
            ax.add_patch(plt.Rectangle((i-0.42, min(init,orc)), 0.84, abs(orc-init),
                                       facecolor=INK3, alpha=0.07, zorder=0))
        for val,ls,lab in ((init,'-','init'),(orc,(0,(4,3)),'oracle')):
            if val is not None:
                ax.hlines(val, i-0.42, i+0.42, color=INK3, lw=1.4, ls=ls, zorder=2)
        for k,(nm,col,fn) in enumerate(series):
            v = fn(s,n)
            if v is None: continue
            x = i - 0.36 + w*(k+0.5)
            if v < ylim[0]:
                # below the clipped axis: draw a stub and label the true value
                ax.bar(x, pad*0.35, bottom=ylim[0], width=w*0.82, color=col,
                       zorder=3, edgecolor=SURFACE, linewidth=1.0, hatch='///')
                ax.text(x, ylim[0]+pad*0.45, f'{v:.1f}\n↓', ha='center', va='bottom',
                        fontsize=7, color=INK2, zorder=4, linespacing=0.9)
            else:
                ax.bar(x, v-ylim[0], bottom=ylim[0], width=w*0.82, color=col,
                       zorder=3, edgecolor=SURFACE, linewidth=1.0)
                ax.text(x, v+pad*0.10, f'{v:.1f}', ha='center', va='bottom',
                        fontsize=7.5, color=INK2, zorder=4)
    ax.set_xticks(range(len(DIRS)))
    ax.set_xticklabels([LBL[d] for d in DIRS], rotation=18, ha='right', fontsize=8.5)
    ax.set_xlim(-0.6, len(DIRS)-0.4); ax.set_ylim(*ylim)
    ax.set_title(f'{title}\n{sub}', fontsize=10.5, pad=9, loc='left', color=INK)
    ax.grid(axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)
axes[0].set_ylabel('test mIoU')

handles = [Line2D([],[],color=INK3,lw=1.4,ls='-',label='init: SFT split1 (teacher)'),
           Line2D([],[],color=INK3,lw=1.4,ls=(0,(4,3)),label='oracle: SFT full (train2 labels revealed)')]
seen=set()
for _,_,ss,_,_ in panels:
    for nm,col,_ in ss:
        if nm in seen: continue
        seen.add(nm)
        handles.append(Line2D([],[],marker='s',ls='',ms=8,mfc=col,mec=SURFACE,label=nm))
fig.legend(handles=handles, loc='lower center', ncol=6, frameon=False, fontsize=8.5,
           labelcolor=INK2, bbox_to_anchor=(0.52, 0.015))

fig.suptitle('DFC2020 · Copernicus-Bench split · DeluluNet vs baselines '
             '(upernet decoder, test mIoU)', x=0.055, ha='left', y=0.965,
             fontsize=13, color=INK)
fig.text(0.055, 0.915,
         'Each panel controls for TEST-TIME INPUT MODALITY, so the oracle differs per panel. '
         'Shaded band = the gap unlabeled train2 could close.',
         ha='left', fontsize=8.5, color=INK3)
fig.text(0.055, 0.878,
         'distillation and MKE appear only under Addition — their students consume BOTH '
         'modalities at test time. Y-axis clipped at 40; hatched stubs mark values below it.',
         ha='left', fontsize=8.5, color=INK3)
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=170)
print('wrote', OUT)
