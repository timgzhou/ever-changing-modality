"""Per-head accuracy analysis for DeluluNet's transfer path.

Transfer mode: only the NEW modality is real; the START modality is hallucinated
through the intermediate projectors. The reported prediction is a soft vote --
_soft_vote() averages the per-modality head logits:

    pred = mean_over_mods( get_modality_logits(fused, mod) )

This script reproduces that exact forward pass and additionally scores each head
on its own, to see whether the hallucinated-modality head is dragging the
ensemble down.

For each direction we report, under transfer mode:
  head[start]  -- head reading the HALLUCINATED start modality
  head[new]    -- head reading the REAL new modality
  ensemble     -- the soft vote actually used (sanity: matches the results CSV)
  oracle_pick  -- per-pixel best of the two heads (upper bound on any weighting)

Peeking mode is included as the mirror image (start real, new hallucinated).
"""
import argparse, csv, os, sys
import torch
from delulunet_main import EvanSegmenter, hallucinate_intermediate_features, merge_intermediate_features
from data_utils import create_multimodal_batch, get_loaders
from train_utils import compute_miou


def _make_batch(batch, modality_bands_dict, mods, device):
    mb = create_multimodal_batch(batch, modality_bands_dict=modality_bands_dict, modalities=mods)
    return {k: v.to(device) for k, v in mb.items()}


@torch.no_grad()
def per_head_eval(model, loader, modality_bands_dict, all_mods, real_mods,
                  label_key, num_classes, ignore_index, device):
    """Return {head_name: mIoU, 'ensemble': mIoU, 'oracle_pick': mIoU} for one path."""
    hallucinated = tuple(m for m in all_mods if m not in real_mods)
    _verified = {'done': False}
    preds = {m: [] for m in sorted(all_mods)}
    ens_preds, orc_preds, all_labels = [], [], []

    for batch in loader:
        mm = _make_batch(batch, modality_bands_dict, tuple(all_mods), device)
        labels = batch[label_key].to(device)

        intermediate = model.evan.forward_modality_specific_features(mm)
        hal = hallucinate_intermediate_features(
            intermediate, tuple(real_mods), hallucinated, model.evan, use_mask_token=False)
        fusion_input = merge_intermediate_features(
            intermediate, hal, tuple(real_mods), hallucinated)
        fused = model.evan.forward_fusion_from_modality_features(
            fusion_input, hallucinated_modalities=set(hallucinated))

        # per-head logits off the SAME fused tensor the ensemble uses
        logits = {m: model.get_modality_logits(fused, m) for m in sorted(all_mods)}

        # VERIFICATION: our manual reconstruction must reproduce, bit-for-bit,
        # what the production path (predict_from_real_modalities -> _soft_vote)
        # returns. If this ever drifts, the per-head numbers describe a
        # different forward pass than the one being reported and are meaningless.
        if not _verified['done']:
            ref = model.predict_from_real_modalities(
                intermediate, tuple(real_mods), tuple(all_mods), use_mask_token=False)
            mine = torch.stack([logits[m] for m in sorted(all_mods)]).mean(dim=0)
            md = (ref - mine).abs().max().item()
            assert md < 1e-4, (
                f"reconstruction diverges from predict_from_real_modalities "
                f"(max|diff|={md:.3e}) -- per-head analysis would be invalid")
            print(f"    [verify] matches predict_from_real_modalities (max|diff|={md:.2e})")

            # VERIFICATION 2: the hallucinated features must actually DIFFER
            # from the real ones. If the projectors were a no-op / leaked the
            # real modality, transfer would silently be measuring `addition`.
            n_st = model.evan.n_storage_tokens
            for hm in hallucinated:
                real_f = intermediate[hm]          # [B, 1+n_storage+P, D]
                hal_f = fusion_input[hm]           # [B, 1+P, D] (projector: no storage tokens)
                # Compare CLS + patches, dropping the storage slots the projector omits.
                if real_f.shape[1] != hal_f.shape[1]:
                    real_cmp = torch.cat([real_f[:, :1], real_f[:, n_st + 1:]], dim=1)
                else:
                    real_cmp = real_f
                assert real_cmp.shape == hal_f.shape, (
                    f"shape mismatch after stripping storage tokens: "
                    f"{tuple(real_cmp.shape)} vs {tuple(hal_f.shape)}")
                rel = (real_cmp - hal_f).norm().item() / (real_cmp.norm().item() + 1e-9)
                assert rel > 1e-3, (
                    f"hallucinated '{hm}' is identical to the real features "
                    f"(rel diff {rel:.2e}) -- transfer is not actually hallucinating")
                print(f"    [verify] '{hm}' hallucinated != real (rel diff {rel:.3f})")
            _verified['done'] = True
        for m, lg in logits.items():
            preds[m].append(lg.argmax(dim=1).cpu())

        stacked = torch.stack([logits[m] for m in sorted(all_mods)])   # [M,B,C,H,W]
        ens_preds.append(stacked.mean(dim=0).argmax(dim=1).cpu())

        # oracle: per pixel, take whichever head is right (upper bound on reweighting)
        ph = torch.stack([logits[m].argmax(dim=1) for m in sorted(all_mods)])  # [M,B,H,W]
        correct = (ph == labels.unsqueeze(0))
        pick = torch.where(correct.any(dim=0), labels, ph[0])
        orc_preds.append(pick.cpu())
        all_labels.append(labels.cpu())

    lab = torch.cat(all_labels)
    out = {}
    for m in sorted(all_mods):
        tag = f"head[{m}]" + ("*HAL" if m in hallucinated else "")
        out[tag] = compute_miou(torch.cat(preds[m]), lab, num_classes, ignore_index=ignore_index)
    out['ensemble'] = compute_miou(torch.cat(ens_preds), lab, num_classes, ignore_index=ignore_index)
    out['oracle_pick'] = compute_miou(torch.cat(orc_preds), lab, num_classes, ignore_index=ignore_index)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start_mod', required=True)
    ap.add_argument('--new_mod', required=True)
    ap.add_argument('--variant', default='?', help='label for the CSV (full / distill_only)')
    ap.add_argument('--dataset', default='dfc2020')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--out_csv', default='res/delulu/head_analysis.csv')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EvanSegmenter.from_checkpoint(args.checkpoint, device).to(device).eval()

    all_mods = (args.start_mod, args.new_mod)
    # Same call convention as shot_ete.py: starting_modality positional,
    # new_modality by keyword, normalizer left to the dataset default.
    _, _, _, _, test_loader, tc = get_loaders(
        args.dataset, args.start_mod, args.batch_size, 4,
        data_normalizer=None, new_modality=args.new_mod)

    common = dict(model=model, loader=test_loader,
                  modality_bands_dict=tc.modality_bands_dict, all_mods=all_mods,
                  label_key=tc.label_key, num_classes=tc.num_classes,
                  ignore_index=getattr(tc, 'ignore_index', 255), device=device)

    print(f"\n=== {args.start_mod} -> +{args.new_mod}  [{args.variant}] ===")
    rows = []
    for path, real in (('transfer', (args.new_mod,)), ('peeking', (args.start_mod,))):
        res = per_head_eval(real_mods=real, **common)
        print(f"  {path}:")
        for k, v in res.items():
            print(f"    {k:<22} {v:6.2f}")
        rows.append((path, res))

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    exists = os.path.isfile(args.out_csv)
    with open(args.out_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(['variant', 'start_mod', 'new_mod', 'path', 'head_start',
                        'head_new', 'hallucinated', 'ensemble', 'oracle_pick', 'checkpoint'])
        for path, res in rows:
            hal = args.start_mod if path == 'transfer' else args.new_mod
            hs = res.get(f"head[{args.start_mod}]*HAL", res.get(f"head[{args.start_mod}]"))
            hn = res.get(f"head[{args.new_mod}]*HAL", res.get(f"head[{args.new_mod}]"))
            w.writerow([args.variant, args.start_mod, args.new_mod, path,
                        f"{hs:.2f}", f"{hn:.2f}", hal,
                        f"{res['ensemble']:.2f}", f"{res['oracle_pick']:.2f}", args.checkpoint])
    print(f"  -> appended to {args.out_csv}")


if __name__ == '__main__':
    main()
