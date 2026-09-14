"""Asymmetric Duos temperature weighting for DeluluNet's transfer path.

Zhou et al., "Asymmetric Duos: Sidekicks Improve Uncertainty" (arXiv 2505.18636)
aggregate two unequal models by a weighted logit sum with two learned scalars:

    f_Duo(X) = f_large(X) * T_large + f_small(X) * T_small

fit by minimising NLL on a validation set. If the weak model is useless the fit
can drive T_small -> 0 and recover the strong model alone, so the method cannot
do worse than the strong head (up to fitting noise).

Our asymmetric duo, under TRANSFER evaluation (only the new modality is real):
    strong = head[new_mod]    reads the REAL modality
    weak   = head[start_mod]  reads the HALLUCINATED modality

DeluluNet's current combiner is the equal-weight soft vote (T=0.5, 0.5), which
the head analysis showed sits BELOW the strong head alone on average.

Targets. The honest, label-free choice is the frozen unimodal TEACHER's
prediction on val2 -- that is what the method has access to at adaptation time
and needs no val labels. We also fit against val2 ground-truth labels as an
upper reference, to separate "the combiner is fixable" from "teacher targets are
good enough to fix it with".

Fitting is on val2 only; TEST is never touched during fitting.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse, csv, os
import torch
import torch.nn.functional as F
from delulunet_main import EvanSegmenter, hallucinate_intermediate_features, merge_intermediate_features
from data_utils import create_multimodal_batch, get_loaders
from train_utils import compute_miou


def _make_batch(batch, mbd, mods, device):
    mb = create_multimodal_batch(batch, modality_bands_dict=mbd, modalities=mods)
    return {k: v.to(device) for k, v in mb.items()}


@torch.no_grad()
def collect(model, teacher, loader, mbd, all_mods, real_mods, start_mod,
            label_key, device, verify=True):
    """Return per-head logits, teacher argmax, and labels for one split (CPU)."""
    hal = tuple(m for m in all_mods if m not in real_mods)
    L = {m: [] for m in sorted(all_mods)}
    T, Y = [], []
    verified = False

    for batch in loader:
        mm = _make_batch(batch, mbd, tuple(all_mods), device)
        labels = batch[label_key].to(device)

        inter = model.evan.forward_modality_specific_features(mm)
        hf = hallucinate_intermediate_features(inter, tuple(real_mods), hal, model.evan, use_mask_token=False)
        fi = merge_intermediate_features(inter, hf, tuple(real_mods), hal)
        fused = model.evan.forward_fusion_from_modality_features(fi, hallucinated_modalities=set(hal))
        logits = {m: model.get_modality_logits(fused, m) for m in sorted(all_mods)}

        if verify and not verified:
            ref = model.predict_from_real_modalities(inter, tuple(real_mods), tuple(all_mods), use_mask_token=False)
            mine = torch.stack([logits[m] for m in sorted(all_mods)]).mean(dim=0)
            md = (ref - mine).abs().max().item()
            assert md < 1e-4, f"head reconstruction != production path (max|diff|={md:.3e})"
            print(f"    [verify] heads reproduce predict_from_real_modalities (max|diff|={md:.2e})")
            verified = True

        for m in sorted(all_mods):
            L[m].append(logits[m].float().cpu())
        # teacher sees ONLY the start modality (its training modality)
        t_logits = teacher({start_mod: mm[start_mod]})
        T.append(t_logits.argmax(dim=1).cpu())
        Y.append(labels.cpu())

    return ({m: torch.cat(v) for m, v in L.items()}, torch.cat(T), torch.cat(Y))


def fit_temps(strong, weak, target, ignore_index, iters=300, lr=0.05):
    """Fit f = strong*T_s + weak*T_w by NLL against `target`. Returns (T_s, T_w)."""
    # init at the equal-weight soft vote DeluluNet currently uses
    ts = torch.tensor(0.5, requires_grad=True)
    tw = torch.tensor(0.5, requires_grad=True)
    opt = torch.optim.LBFGS([ts, tw], lr=lr, max_iter=iters)
    # [N,C,H,W] -> [pixels, C], keeping only non-ignored pixels.
    C = strong.shape[1]
    s = strong.permute(0, 2, 3, 1).reshape(-1, C)
    w = weak.permute(0, 2, 3, 1).reshape(-1, C)
    y = target.reshape(-1)
    keep = y != ignore_index
    s, w, y = s[keep], w[keep], y[keep]
    assert s.shape[0] == y.shape[0] == w.shape[0] and s.shape[1] == C, \
        f"flatten misaligned: s{tuple(s.shape)} w{tuple(w.shape)} y{tuple(y.shape)}"
    # equal-weight init must reproduce the soft vote's loss on this same data
    with torch.no_grad():
        _base = F.cross_entropy(s * 0.5 + w * 0.5, y).item()

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(s * ts + w * tw, y)
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        _fit = F.cross_entropy(s * ts + w * tw, y).item()
    # The fit starts AT the equal-weight solution, so NLL must not get worse.
    assert _fit <= _base + 1e-4, f"fit worsened NLL: {_base:.4f} -> {_fit:.4f}"
    print(f"    [verify] val2 NLL {_base:.4f} (equal vote) -> {_fit:.4f} (duo)")
    return ts.detach().item(), tw.detach().item()


def miou(logits, labels, nc, ii):
    return compute_miou(logits.argmax(dim=1), labels, nc, ignore_index=ii)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--teacher_checkpoint', required=True)
    ap.add_argument('--start_mod', required=True)
    ap.add_argument('--new_mod', required=True)
    ap.add_argument('--variant', default='?')
    ap.add_argument('--dataset', default='dfc2020')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--out_csv', default='res/delulu/duo_temps.csv')
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EvanSegmenter.from_checkpoint(args.checkpoint, dev).to(dev).eval()
    teacher = EvanSegmenter.from_checkpoint(args.teacher_checkpoint, dev).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    all_mods = (args.start_mod, args.new_mod)
    _, _, _, val2_loader, test_loader, tc = get_loaders(
        args.dataset, args.start_mod, args.batch_size, 4,
        data_normalizer=None, new_modality=args.new_mod)
    nc, ii = tc.num_classes, getattr(tc, 'ignore_index', 255)
    real = (args.new_mod,)   # TRANSFER: only the new modality is real

    print(f"\n=== {args.start_mod} -> +{args.new_mod}  [{args.variant}] ===")
    print("  collecting val2 (fitting split)...")
    vL, vT, vY = collect(model, teacher, val2_loader, tc.modality_bands_dict,
                         all_mods, real, args.start_mod, tc.label_key, dev)
    print("  collecting test (held out)...")
    tL, tT, tY = collect(model, teacher, test_loader, tc.modality_bands_dict,
                         all_mods, real, args.start_mod, tc.label_key, dev)

    S, W = args.new_mod, args.start_mod          # strong=real, weak=hallucinated
    base_ens = (tL[S] + tL[W]) / 2               # equal-weight soft vote (current)

    res = {
        'head_real(strong)':  miou(tL[S], tY, nc, ii),
        'head_hal(weak)':     miou(tL[W], tY, nc, ii),
        'equal_vote(current)': miou(base_ens, tY, nc, ii),
    }
    # sanity: teacher quality on val2, to contextualise it as a target
    res['teacher_on_val2'] = compute_miou(vT, vY, nc, ii)

    fits = {}
    for tgt_name, tgt in (('teacher', vT), ('labels', vY)):
        ts, tw = fit_temps(vL[S], vL[W], tgt, ii)
        fits[tgt_name] = (ts, tw)
        res[f'duo_{tgt_name}'] = miou(tL[S] * ts + tL[W] * tw, tY, nc, ii)

    print(f"  {'teacher mIoU on val2':<26} {res['teacher_on_val2']:6.2f}")
    print(f"  {'head[real]   (strong)':<26} {res['head_real(strong)']:6.2f}")
    print(f"  {'head[hal]    (weak)':<26} {res['head_hal(weak)']:6.2f}")
    print(f"  {'equal vote   (current)':<26} {res['equal_vote(current)']:6.2f}")
    for k, (ts, tw) in fits.items():
        print(f"  {'duo/' + k:<26} {res['duo_' + k]:6.2f}   (T_strong={ts:+.3f}, T_weak={tw:+.3f})")

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    ex = os.path.isfile(args.out_csv)
    with open(args.out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if not ex:
            w.writerow(['variant', 'start_mod', 'new_mod', 'teacher_val2_miou',
                        'head_real', 'head_hal', 'equal_vote',
                        'duo_teacher', 'T_strong_teacher', 'T_weak_teacher',
                        'duo_labels', 'T_strong_labels', 'T_weak_labels'])
        w.writerow([args.variant, args.start_mod, args.new_mod,
                    f"{res['teacher_on_val2']:.2f}",
                    f"{res['head_real(strong)']:.2f}", f"{res['head_hal(weak)']:.2f}",
                    f"{res['equal_vote(current)']:.2f}",
                    f"{res['duo_teacher']:.2f}", f"{fits['teacher'][0]:.4f}", f"{fits['teacher'][1]:.4f}",
                    f"{res['duo_labels']:.2f}", f"{fits['labels'][0]:.4f}", f"{fits['labels'][1]:.4f}"])


if __name__ == '__main__':
    main()
