"""
Is the projector's cross-attention read positional or content-driven?

Every patch query in CrossSequenceProjector is the same learned prototype, told
apart only by RoPE. If the read is a near-positional lookup (output patch i
attends to source patch i and its neighbours), output i is just the
contextualised source token at i, and the shared query costs nothing: capacity
belongs on the source side (self-attention). If the attention is broad, a
content-aware query (a second cross-attention block, or query self-attention)
has room to help.

Per projector direction and per head, over test images, for each patch query:
  own        attention mass on the source patch at the same position
  3x3 / 5x5  mass within that neighbourhood (uniform baseline in brackets)
  prefix     mass on source CLS + storage tokens
  eff_keys   exp(entropy) over all keys: how many keys the read effectively uses
  dist       attention-weighted distance to the source patch, in patch units
  img_TV     total-variation distance between two images' attention rows at
             the same (head, position): 0 = purely positional, same map for
             every image; 1 = disjoint maps (content decides where to look)

Runs with no source mask, i.e. the transfer/addition setting.

    python analysis/cross_attn_locality.py --checkpoint checkpoints/<ckpt>.pt --dataset dfc2020
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_utils import create_multimodal_batch, get_loaders
from delulunet.layers.attention import CrossAttention
from delulunet.layers import attention as attn_module

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_hallucination_correlation import _load_model

_CAPTURE = []


def _forward_capture(self, tgt, memory, rope_tgt=None, rope_memory=None,
                     prefix_len_tgt=0, prefix_len_memory=0, attn_mask=None):
    """CrossAttention.forward with the softmax computed explicitly and stored."""
    B, Nq, C = tgt.shape
    Nkv = memory.shape[1]
    hd = C // self.num_heads
    q = self.q_proj(tgt).reshape(B, Nq, self.num_heads, hd).transpose(1, 2)
    kv = self.kv_proj(memory).reshape(B, Nkv, 2, self.num_heads, hd)
    k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)
    if rope_tgt is not None:
        sin, cos = rope_tgt
        qp = attn_module.rope_apply(q[:, :, prefix_len_tgt:].to(sin.dtype), sin, cos)
        q = torch.cat([q[:, :, :prefix_len_tgt], qp.to(q.dtype)], dim=2)
    if rope_memory is not None:
        sin, cos = rope_memory
        kp = attn_module.rope_apply(k[:, :, prefix_len_memory:].to(sin.dtype), sin, cos)
        k = torch.cat([k[:, :, :prefix_len_memory], kp.to(k.dtype)], dim=2)
    logits = (q.float() @ k.float().transpose(-1, -2)) * hd ** -0.5
    if attn_mask is not None:
        logits = logits + attn_mask
    probs = logits.softmax(-1)                                    # [B, h, Nq, Nkv]
    _CAPTURE.append((probs[:, :, prefix_len_tgt:], prefix_len_memory))
    x = (probs.to(v.dtype) @ v).transpose(1, 2).reshape(B, Nq, C)
    return self.proj_drop(self.proj(x))


def _metrics(probs, n_prefix, grid):
    """probs: [B, h, N, n_prefix+N] for patch queries. Returns dict of [h] arrays."""
    B, h, N, _ = probs.shape
    pp = probs[..., n_prefix:]                                    # patch keys only
    yy, xx = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing='ij')
    pos = torch.stack([yy.flatten(), xx.flatten()], -1).float().to(probs.device)
    cheb = (pos[:, None] - pos[None]).abs().amax(-1)              # [N, N] Chebyshev
    eucl = (pos[:, None] - pos[None]).norm(dim=-1)
    ent = -(probs.clamp_min(1e-12).log() * probs).sum(-1)         # [B, h, N]
    out = {
        'own':      pp.diagonal(dim1=-2, dim2=-1),
        'nbr3':     (pp * (cheb <= 1)).sum(-1),
        'nbr5':     (pp * (cheb <= 2)).sum(-1),
        'prefix':   probs[..., :n_prefix].sum(-1),
        'eff_keys': ent.exp(),
        'dist':     (pp * eucl).sum(-1) / pp.sum(-1).clamp_min(1e-12),
    }
    # Same position and head, different images (neighbouring batch index).
    out['img_TV'] = 0.5 * (probs - probs.roll(1, dims=0)).abs().sum(-1)
    return {k: v.mean(dim=(0, 2)).cpu().numpy() for k, v in out.items()}   # -> [h]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', default=None)
    p.add_argument('--n_batches', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--out_json', default=None)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ds = args.dataset or ckpt.get('dataset')
    model, _ = _load_model(args.checkpoint, args.device)
    model.eval()
    evan = model.evan
    mod_a, mod_b = evan.supported_modalities[:2]
    _, _, _, _, test_loader, tc = get_loaders(
        ds, starting_modality=mod_a, batch_size=args.batch_size, num_workers=4,
        new_modality=mod_b, num_time_steps=ckpt.get('num_time_steps') or 10)
    slices = tc.modality_bands_dict

    # Sanity: the capturing forward must reproduce the SDPA forward exactly.
    b0 = next(iter(test_loader))
    x0 = {k: v.to(args.device) for k, v in create_multimodal_batch(b0, slices, (mod_a, mod_b)).items()}
    with torch.no_grad():
        e0 = evan.forward_modality_specific_features(x0)
        ref0 = evan._project_sequence(e0[mod_a], f'{mod_a}_to_{mod_b}', mod_b)
        CrossAttention.forward = _forward_capture
        new0 = evan._project_sequence(e0[mod_a], f'{mod_a}_to_{mod_b}', mod_b)
    err = (ref0.float() - new0.float()).abs().max().item()
    print(f'capture-vs-SDPA max abs diff: {err:.2e}')
    assert err < 1e-3 * ref0.float().abs().max().item(), 'capturing forward diverges from SDPA'
    grid = None
    acc = {}
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.n_batches:
                break
            x = create_multimodal_batch(batch, slices, (mod_a, mod_b))
            x = {k: v.to(args.device) for k, v in x.items()}
            emb = evan.forward_modality_specific_features(x)
            for src, tgt in ((mod_a, mod_b), (mod_b, mod_a)):
                _CAPTURE.clear()
                ref = evan._project_sequence(emb[src], f'{src}_to_{tgt}', tgt)
                probs, n_prefix = _CAPTURE[-1]
                grid = int(round(probs.shape[2] ** 0.5))
                m = _metrics(probs, n_prefix, grid)
                for k, v in m.items():
                    acc.setdefault(f'{src}_to_{tgt}', {}).setdefault(k, []).append(v)

    N = grid * grid
    base = {'own': 1 / N, 'nbr3': 9 / N, 'nbr5': 25 / N}
    result = {}
    for key, d in acc.items():
        per_head = {k: np.mean(v, 0) for k, v in d.items()}      # [h]
        result[key] = {k: {'mean': float(v.mean()), 'per_head': [round(float(x), 4) for x in v]}
                       for k, v in per_head.items()}
        print(f'\n=== {key}  (grid {grid}x{grid}, {len(per_head["own"])} heads) ===')
        for k, v in per_head.items():
            b = f'  [uniform {base[k]:.4f}]' if k in base else ''
            print(f'  {k:9s} mean {v.mean():.4f}{b}   per-head '
                  + ' '.join(f'{x:.3f}' for x in v))
    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        with open(args.out_json, 'w') as f:
            json.dump({'checkpoint': args.checkpoint, 'grid': grid, 'directions': result}, f, indent=2)
        print(f'\nSaved {args.out_json}')


if __name__ == '__main__':
    main()
