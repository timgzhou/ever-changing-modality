"""Train ONLY the cross-modal projector on a frozen encoder, and measure hallucination.

Why this exists
---------------
The recon A/B (res/delulu/dfc2020_recon_ab.csv) trained the whole delulu stack
and scored mIoU; the cosine terms lost. The hallucination analysis on those
checkpoints then showed the MSE projector is NOT mean-collapsed (batch_pearson
0.40, far from the ~0 a conditional-mean predictor would score) and that no
cosine variant improved per-image recovery.

Both of those measured the projector through a long chain: fusion blocks, a
decoder, a task head, and three other loss terms all pulling on the same
encoder. This probe removes the chain. The encoder is frozen and only the
projector trains, on the reconstruction loss ALONE. If a loss function changes
what a cross-modal projector can learn, it has to show up here -- there is
nothing else in the graph to absorb or mask the difference.

Setup
-----
  frozen encoder --> forward_modality_specific_features (under no_grad)
      src_seq  = features[src_mod]   [B, 1+n_storage+n_patches, D]
      tgt_seq  = features[tgt_mod]   (the reconstruction target)
  trainable    --> a FRESH projector (+ its target queries, for 'cross' type)
      pred     = projector(src_seq)  [B, 1+n_patches, D]
  loss         --> mse | mse_ccos | ccos   on the PATCH tokens

No fusion blocks, no decoder, no task head, no CE/distill/latent terms.

The CLS row is dropped from both sides: 'ccos' centers across the token axis,
which is only meaningful for the patch grid, and the downstream task head reads
x_norm_patchtokens anyway.

Reports patch_pearson (aligned + derangement-shuffled) and batch_pearson on a
held-out split, reusing the exact functions from
analysis/analyze_hallucination_correlation.py so the numbers are comparable to
the full-stack results.
"""

import argparse
import csv
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from delulunet_main import EVANClassifier, EvanSegmenter
from data_utils import create_multimodal_batch, get_loaders
from analysis.analyze_hallucination_correlation import (
    patch_pearson, shuffled_patch_pearson, batch_pearson,
)


def _load_model(path, device):
    config = torch.load(path, map_location='cpu', weights_only=False)['config']
    if 'classifier_strategy' in config:
        return EVANClassifier.from_checkpoint(path, device)
    if 'decoder_strategy' in config:
        return EvanSegmenter.from_checkpoint(path, device)
    raise ValueError(f"Cannot determine head type for {path}")


def centered_cos(pred, target, eps=1e-6):
    """1 - cosine between per-sample-mean-centered tokens -> [B, N].

    Identical to delulu._centered_cos; duplicated here so the probe does not
    depend on the training module's import graph.
    """
    pc = pred - pred.mean(dim=1, keepdim=True)
    tc = target - target.mean(dim=1, keepdim=True)
    return 1.0 - F.cosine_similarity(pc, tc, dim=-1, eps=eps)


def make_loss(kind, cos_weight):
    """kind: mse | mse_ccos | ccos. Applied to patch tokens [B, N, D]."""
    if kind == 'mse':
        return lambda p, t: F.mse_loss(p, t)
    if kind == 'mse_ccos':
        return lambda p, t: F.mse_loss(p, t) + cos_weight * centered_cos(p, t).mean()
    if kind == 'ccos':
        return lambda p, t: centered_cos(p, t).mean()
    raise ValueError(f"unknown loss {kind}")


def build_projector(evan, src_mod, tgt_mod, depth, device):
    """A FRESH projector at the requested depth, plus fresh target queries.

    evan.intermediate_projector_num_layers is mutated so _make_intermediate_projector
    builds at `depth`; the encoder itself is unaffected by that attribute.
    """
    old = evan.intermediate_projector_num_layers
    evan.intermediate_projector_num_layers = depth
    try:
        proj = evan._make_intermediate_projector(tgt_mod).to(device)
    finally:
        evan.intermediate_projector_num_layers = old

    params = list(proj.parameters())
    queries = None
    if evan.intermediate_projector_type == 'cross':
        # Re-initialise the target queries too: reusing the trained ones would
        # hand every arm a head start from the checkpoint's own objective.
        q = torch.empty(1, 2, evan.embed_dim, device=device)
        torch.nn.init.trunc_normal_(q, std=0.02)
        queries = torch.nn.Parameter(q)
        params.append(queries)
    return proj, queries, params


def project(evan, proj, queries, src_seq):
    if evan.intermediate_projector_type == 'cross':
        return proj(src_seq, queries=queries, rope_embed=evan.rope_embed, src_patch_mask=None)
    return proj(src_seq)


@torch.no_grad()
def encode(evan, batch, bands, mods, device):
    x = create_multimodal_batch(batch, bands, mods)
    x = {k: v.to(device, non_blocking=True) for k, v in x.items()}
    return evan.forward_modality_specific_features(x)


@torch.no_grad()
def evaluate(evan, proj, queries, loader, bands, mods, src_mod, tgt_mod,
             device, n_batches, n_prefix):
    proj.eval()
    al, sh, bp, losses = [], [], [], []
    seen = 0
    for batch in loader:
        if seen >= n_batches:
            break
        feats = encode(evan, batch, bands, mods, device)
        pred = project(evan, proj, queries, feats[src_mod])[:, 1:]     # drop CLS
        tgt = feats[tgt_mod][:, n_prefix:]                             # drop CLS+storage
        al.append(patch_pearson(pred, tgt).flatten().cpu())
        s = shuffled_patch_pearson(pred, tgt)
        if s is not None:
            sh.append(s.flatten().cpu())
        bp.append(batch_pearson(pred, tgt).flatten().cpu())
        losses.append(F.mse_loss(pred, tgt).item())
        seen += 1
    proj.train()
    cat = lambda xs: torch.cat(xs) if xs else torch.tensor([float('nan')])
    a, s_, b = cat(al), cat(sh), cat(bp)
    return {
        'aligned': a.mean().item(),
        'shuffled': s_.mean().item(),
        'gap': a.mean().item() - s_.mean().item(),
        'batch_r': b.mean().item(),
        'batch_r_median': b.median().item(),
        'val_mse': sum(losses) / max(1, len(losses)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--checkpoint', required=True, help='trained model; encoder is frozen')
    ap.add_argument('--dataset', default='dfc2020')
    ap.add_argument('--src_mod', required=True, help='source modality (the one we have)')
    ap.add_argument('--tgt_mod', required=True, help='target modality (the one we hallucinate)')
    ap.add_argument('--depth', type=int, required=True, help='projector num_layers')
    ap.add_argument('--loss', required=True, choices=['mse', 'mse_ccos', 'ccos'])
    ap.add_argument('--cos_weight', type=float, default=1.0,
                    help='weight on the centered-cosine term for mse_ccos. The patch-token '
                         'MSE here is ~0.01, so 1.0 makes cosine dominate; that is a '
                         'deliberate arm, not an oversight.')
    ap.add_argument('--epochs', type=int, default=6)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight_decay', type=float, default=0.01)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--eval_batches', type=int, default=24)
    ap.add_argument('--max_steps_per_epoch', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--results_csv', default='res/delulu/projector_probe.csv')
    ap.add_argument('--tag', default='')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    dev = args.device

    model = _load_model(args.checkpoint, dev)
    evan = model.evan
    evan.eval()
    for p in evan.parameters():          # encoder frozen for every arm
        p.requires_grad_(False)

    mods = (args.src_mod, args.tgt_mod)
    for m in mods:
        if m not in evan.supported_modalities:
            raise ValueError(f"{m!r} not in checkpoint modalities {evan.supported_modalities}")

    train_loader, _, val_loader, _, test_loader, task = get_loaders(
        args.dataset, starting_modality=args.src_mod, batch_size=args.batch_size,
        num_workers=args.num_workers, new_modality=args.tgt_mod,
    )
    bands = task.modality_bands_dict
    n_prefix = evan.n_storage_tokens + 1

    proj, queries, params = build_projector(evan, args.src_mod, args.tgt_mod, args.depth, dev)
    n_par = sum(p.numel() for p in params)
    loss_fn = make_loss(args.loss, args.cos_weight)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps = args.max_steps_per_epoch or len(train_loader)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs * steps)

    print(f"=== projector probe | {args.src_mod} -> {args.tgt_mod} | "
          f"depth={args.depth} loss={args.loss} "
          f"{'w=' + str(args.cos_weight) if args.loss == 'mse_ccos' else ''} ===")
    print(f"trainable params: {n_par:,} (encoder frozen)")

    t0 = time.time()
    for ep in range(args.epochs):
        run, nb = 0.0, 0
        for i, batch in enumerate(train_loader):
            if args.max_steps_per_epoch and i >= args.max_steps_per_epoch:
                break
            feats = encode(evan, batch, bands, mods, dev)
            pred = project(evan, proj, queries, feats[args.src_mod])[:, 1:]
            tgt = feats[args.tgt_mod][:, n_prefix:]
            loss = loss_fn(pred, tgt)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            run += loss.item(); nb += 1
        m = evaluate(evan, proj, queries, val_loader, bands, mods,
                     args.src_mod, args.tgt_mod, dev, args.eval_batches, n_prefix)
        print(f"  ep {ep+1}/{args.epochs} train_loss={run/max(1,nb):.5f} "
              f"val_mse={m['val_mse']:.5f} aligned={m['aligned']:.4f} "
              f"shuf={m['shuffled']:.4f} gap={m['gap']:.4f} batch_r={m['batch_r']:.4f}")

    test = evaluate(evan, proj, queries, test_loader, bands, mods,
                    args.src_mod, args.tgt_mod, dev, args.eval_batches, n_prefix)
    print("\n=== TEST ===")
    for k, v in test.items():
        print(f"  {k:<16} {v:.4f}")

    os.makedirs(os.path.dirname(args.results_csv) or '.', exist_ok=True)
    row = dict(tag=args.tag, dataset=args.dataset, src_mod=args.src_mod, tgt_mod=args.tgt_mod,
               depth=args.depth, loss=args.loss, cos_weight=args.cos_weight,
               epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, seed=args.seed,
               trainable_params=n_par, checkpoint=os.path.basename(args.checkpoint),
               minutes=round((time.time() - t0) / 60, 2),
               **{f'test_{k}': round(v, 6) for k, v in test.items()})
    new = not os.path.exists(args.results_csv)
    with open(args.results_csv, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"\nAppended to {args.results_csv}")
    print(json.dumps(row))


if __name__ == '__main__':
    main()
