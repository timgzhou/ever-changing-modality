"""
Profile inference cost on BEN-v2 (s2→s1, 19-class multilabel) for:
  - DeluluNet-s2:          s2 only, s1 hallucinated via CrossSequenceProjector
  - DeluluNet-s1:          s1 only, s2 hallucinated via CrossSequenceProjector
  - DeluluNet-addition:    s2 + s1 both real
  - EVAN SFT (dino-s1/s2/s2+s1): vanilla EVAN classifier, no SHOT
  - Panopticon ViT-B ×3:  s1 / s2 / s2+s1  (BN1d+Linear head)

All models use the same head: BatchNorm1d(affine=False) + Linear.
Throughput is measured by ramping batch size until OOM, then reporting
images/second at the largest successful batch.

Usage:
  python profile_delulunet.py [--delulu_checkpoint PATH]
"""

import sys, os, time, argparse, gc, json
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------- #
# BEN-v2 config
# --------------------------------------------------------------------------- #
DATASET = 'benv2'
STARTING_MOD = 's2'
NEW_MOD = 's1'
ALL_MODS = (STARTING_MOD, NEW_MOD)
IMG_SIZE = 128
S2_CHANS = 12
S1_CHANS = 2
NUM_CLASSES = 19      # BEN-v2 multilabel
FEATURE_DIM_VIT_B = 768

DEFAULT_CHECKPOINT = 'checkpoints/delulunet_benv2_0420_1051.pt'

# --------------------------------------------------------------------------- #
# Shared head: BN1d(affine=False) + Linear  (matches rsfm_sft.create_classification_head)
# --------------------------------------------------------------------------- #

def make_head(feature_dim: int, num_classes: int, device) -> nn.Module:
    return nn.Sequential(
        nn.BatchNorm1d(feature_dim, affine=False),
        nn.Linear(feature_dim, num_classes),
    ).to(device)


# --------------------------------------------------------------------------- #
# DeluluNet forward wrappers
# --------------------------------------------------------------------------- #

class PeekWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_s2):
        from evan_main import hallucinate_intermediate_features, merge_intermediate_features
        evan = self.model.evan
        intermediate = evan.forward_modality_specific_features({STARTING_MOD: x_s2})
        hallucinated = hallucinate_intermediate_features(
            intermediate, (STARTING_MOD,), (NEW_MOD,), evan
        )
        merged = merge_intermediate_features(
            intermediate, hallucinated, (STARTING_MOD,), (NEW_MOD,)
        )
        fused = evan.forward_fusion_from_modality_features(merged)
        return self.model._soft_vote(fused, ALL_MODS)


class S1PeekWrapper(nn.Module):
    """s1 only → hallucinate s2 → fusion."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_s1):
        from evan_main import hallucinate_intermediate_features, merge_intermediate_features
        evan = self.model.evan
        intermediate = evan.forward_modality_specific_features({NEW_MOD: x_s1})
        hallucinated = hallucinate_intermediate_features(
            intermediate, (NEW_MOD,), (STARTING_MOD,), evan
        )
        merged = merge_intermediate_features(
            intermediate, hallucinated, (NEW_MOD,), (STARTING_MOD,)
        )
        fused = evan.forward_fusion_from_modality_features(merged)
        return self.model._soft_vote(fused, ALL_MODS)


class AdditionWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x_s2, x_s1):
        evan = self.model.evan
        intermediate = evan.forward_modality_specific_features(
            {STARTING_MOD: x_s2, NEW_MOD: x_s1}
        )
        fused = evan.forward_fusion_from_modality_features(intermediate)
        return self.model._soft_vote(fused, ALL_MODS)


# --------------------------------------------------------------------------- #
# Parameter counting helpers (DeluluNet only)
# --------------------------------------------------------------------------- #

def count_params_unique(modules_and_params):
    seen, total = set(), 0
    for item in modules_and_params:
        if isinstance(item, torch.nn.Parameter):
            if id(item) not in seen:
                seen.add(id(item))
                total += item.numel()
        else:
            for p in item.parameters():
                if id(p) not in seen:
                    seen.add(id(p))
                    total += p.numel()
    return total


def count_params_touched(model, forward_fn, inputs):
    """Count parameters belonging to modules that execute at least one op during forward_fn(*inputs).

    Uses forward hooks: any module whose forward() is called contributes its
    own (non-recursive) parameters. Shared parameters are counted once.
    """
    touched_param_ids = set()
    hooks = []

    def make_hook(mod):
        def hook(m, inp, out):
            for p in m.parameters(recurse=False):
                touched_param_ids.add(id(p))
        return hook

    for mod in model.modules():
        hooks.append(mod.register_forward_hook(make_hook(mod)))

    with torch.no_grad():
        forward_fn(*inputs)

    for h in hooks:
        h.remove()

    # Sum unique touched params
    seen, total = set(), 0
    for mod in model.modules():
        for p in mod.parameters(recurse=False):
            if id(p) in touched_param_ids and id(p) not in seen:
                seen.add(id(p))
                total += p.numel()
    return total


def active_peek_items(model):
    evan = model.evan
    items = [
        evan.blocks,
        evan.patch_embedders[STARTING_MOD],
        evan.modality_specific_layer_adaptors[STARTING_MOD],
        evan.cls_tokens[STARTING_MOD],
        evan.modality_encodings[STARTING_MOD],
        evan.intermediate_projectors,
        model.modality_heads,
    ]
    if STARTING_MOD in evan.storage_tokens:
        items.append(evan.storage_tokens[STARTING_MOD])
    if hasattr(evan, 'projector_queries'):
        items.extend(evan.projector_queries.values())
    if hasattr(evan, 'modality_fusion_lora_adaptors'):
        items.append(evan.modality_fusion_lora_adaptors)
    if hasattr(evan, 'norm'):
        items.append(evan.norm)
    return items


def active_s1peek_items(model):
    """s1 input, s2 hallucinated — mirror of active_peek_items but for the new modality."""
    evan = model.evan
    items = [
        evan.blocks,
        evan.patch_embedders[NEW_MOD],
        evan.modality_specific_layer_adaptors[NEW_MOD],
        evan.cls_tokens[NEW_MOD],
        evan.modality_encodings[NEW_MOD],
        evan.intermediate_projectors,
        model.modality_heads,
    ]
    if NEW_MOD in evan.storage_tokens:
        items.append(evan.storage_tokens[NEW_MOD])
    if hasattr(evan, 'projector_queries'):
        items.extend(evan.projector_queries.values())
    if hasattr(evan, 'modality_fusion_lora_adaptors'):
        items.append(evan.modality_fusion_lora_adaptors)
    if hasattr(evan, 'norm'):
        items.append(evan.norm)
    return items


def active_addition_items(model):
    evan = model.evan
    items = [
        evan.blocks,
        evan.patch_embedders[STARTING_MOD],
        evan.patch_embedders[NEW_MOD],
        evan.modality_specific_layer_adaptors[STARTING_MOD],
        evan.modality_specific_layer_adaptors[NEW_MOD],
        evan.cls_tokens[STARTING_MOD],
        evan.cls_tokens[NEW_MOD],
        evan.modality_encodings[STARTING_MOD],
        evan.modality_encodings[NEW_MOD],
        model.modality_heads,
    ]
    for mod in (STARTING_MOD, NEW_MOD):
        if mod in evan.storage_tokens:
            items.append(evan.storage_tokens[mod])
    if hasattr(evan, 'modality_fusion_lora_adaptors'):
        items.append(evan.modality_fusion_lora_adaptors)
    if hasattr(evan, 'norm'):
        items.append(evan.norm)
    return items


# --------------------------------------------------------------------------- #
# FLOPs  (single image, B=1)
# --------------------------------------------------------------------------- #

def measure_gflops(fn, inputs, device, n_warmup=5):
    """fn(*inputs) must run a single forward pass. inputs are tensors on device.

    Uses FlopCounterMode (PyTorch dispatcher-level) which correctly counts
    F.scaled_dot_product_attention / Flash Attention — unlike torch.profiler
    with_flops=True which silently misses fused SDPA kernels.
    """
    with torch.no_grad():
        for _ in range(n_warmup):
            fn(*inputs)
    if device != 'cpu':
        torch.cuda.synchronize()

    with FlopCounterMode(display=False) as fcm:
        fn(*inputs)

    # FlopCounterMode reports total FLOPs (each MAC = 2 FLOPs); divide by 2 for MACs.
    gmacs = fcm.get_total_flops() / 2 / 1e9
    return gmacs


# --------------------------------------------------------------------------- #
# Throughput: ramp batch size until OOM, report img/s at largest success
# --------------------------------------------------------------------------- #

BS_CACHE_PATH = 'res/profile_bs_cache.json'

def _load_bs_cache():
    if os.path.exists(BS_CACHE_PATH):
        with open(BS_CACHE_PATH) as f:
            return json.load(f)
    return {}

def _save_bs_cache(cache):
    os.makedirs(os.path.dirname(BS_CACHE_PATH), exist_ok=True)
    with open(BS_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=2)


def measure_throughput(fn, make_inputs_fn, device, n_steps=20, cache_key=None):
    """
    Ramp B = 1, 2, 4, 8, ... until OOM.  At the largest successful B,
    time n_steps forward passes and return (best_B, imgs_per_sec).

    If cache_key is given, the discovered max batch size is saved to
    BS_CACHE_PATH and reused on subsequent runs (skipping the ramp).

    make_inputs_fn(B) -> tuple of tensors already on device.
    fn(*inputs) -> forward pass (no grad).
    """
    bs_cache = _load_bs_cache()

    if cache_key and cache_key in bs_cache:
        best_B = bs_cache[cache_key]
        print(f'  [bs cache] {cache_key}: using cached max_bs={best_B}')
    else:
        best_B = 1
        B = 1
        while True:
            try:
                inputs = make_inputs_fn(B)
                with torch.no_grad():
                    fn(*inputs)
                if device != 'cpu':
                    torch.cuda.synchronize()
                best_B = B
                del inputs
                gc.collect()
                if device != 'cpu':
                    torch.cuda.empty_cache()
                B *= 2
            except RuntimeError as e:
                if 'out of memory' in str(e).lower():
                    break
                raise

        if cache_key:
            bs_cache[cache_key] = best_B
            _save_bs_cache(bs_cache)
            print(f'  [bs cache] {cache_key}: saved max_bs={best_B}')

    # Benchmark at best_B
    inputs = make_inputs_fn(best_B)
    # warmup
    with torch.no_grad():
        for _ in range(5):
            fn(*inputs)
    if device != 'cpu':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_steps):
            fn(*inputs)
    if device != 'cpu':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    imgs_per_sec = best_B * n_steps / elapsed
    return best_B, imgs_per_sec


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def _split_model(name):
    """Split 'DeluluNet-s2' → ('DeluluNet', 's2'), 'dino-s1' → ('DINOv3', 's1'), etc."""
    if name.startswith('DeluluNet-'):
        return 'DeluluNet', name[len('DeluluNet-'):]
    if name.startswith('dino-'):
        return 'DINOv3', name[len('dino-'):]
    if name.startswith('Panopticon-'):
        return 'Panopticon', name[len('Panopticon-'):]
    return name, ''


def print_table(rows):
    """rows: list of dicts with keys: model, params_M, active_M, gmacs, best_B, throughput"""
    cols     = ['Model', 'Modality', 'Params (M)', 'Active (M)', 'GMACs', 'Batch Size', 'Throughput']
    col_keys = ['_arch',  '_mod',    'params_M',   'active_M',   'gmacs', 'best_B',     'throughput']

    augmented = []
    for r in rows:
        arch, mod = _split_model(r['model'])
        augmented.append({**r, '_arch': arch, '_mod': mod})

    widths = [max(len(c), max(len(str(r[k])) for r in augmented)) + 2
              for c, k in zip(cols, col_keys)]

    header = '  '.join(c.ljust(w) for c, w in zip(cols, widths))
    sep    = '  '.join('-' * w for w in widths)
    print()
    print(header)
    print(sep)
    prev_arch = None
    for r in augmented:
        arch = r['_arch']
        display_arch = arch if arch != prev_arch else ''
        prev_arch = arch
        vals = [display_arch] + [str(r[k]) for k in col_keys[1:]]
        print('  '.join(v.ljust(w) for v, w in zip(vals, widths)))
    print()


def write_latex_table(rows, path):
    """Write a LaTeX booktabs table with merged Model column using multirow."""
    lines = []
    lines += [
        r'\begin{table}[t]',
        r'\centering',
        r'\caption{Inference cost on BEN-v2 (img\_size=128, 19-class multilabel).}',
        r'\label{tab:computation}',
        r'\begin{tabular}{cccccc}',
        r'\toprule',
        r'Model & Modality & Params (M) & GMACs & Batch Size & Throughput \\',
        r'\midrule',
    ]

    # Group rows by architecture
    groups = []
    for r in rows:
        arch, mod = _split_model(r['model'])
        if not groups or groups[-1][0] != arch:
            groups.append((arch, []))
        groups[-1][1].append((mod, r))

    for g_idx, (arch, members) in enumerate(groups):
        n = len(members)
        for i, (mod, r) in enumerate(members):
            arch_cell = (
                rf'\multirow{{{n}}}{{*}}{{{arch}}}' if i == 0 else ''
            )
            row_str = ' & '.join([
                arch_cell, mod,
                r['active_M'], r['gmacs'],
                str(r['best_B']), r['throughput'],
            ]) + r' \\'
            lines.append(row_str)
        if g_idx < len(groups) - 1:
            lines.append(r'\midrule')

    lines += [
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ]

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'LaTeX table saved to {path}')


DINO_S1_CHECKPOINT  = 'checkpoints/sft_evan_base_benv2_s1_fft_lr0.0005_20260418_064233.pt'
DINO_S2_CHECKPOINT  = 'checkpoints/sft_evan_base_benv2_s2_fft_lr0.001_20260418_112953.pt'
DINO_S2S1_CHECKPOINT = 'checkpoints/sft_evan_base_benv2_s2+s1_fft_lr0.0005_20260418_064727.pt'


def profile_evan_sft(label, checkpoint, input_mods, in_chans_map, device):
    """Profile a vanilla EVAN SFT classifier (no hallucination).

    active_M uses forward hooks to count only params in modules that actually
    execute — excludes intermediate_projectors / projector_queries which exist
    in the checkpoint but are never called during a plain forward pass.
    """
    from evan_main import EVANClassifier
    model = EVANClassifier.from_checkpoint(checkpoint, device=device)
    model.eval()
    total = sum(p.numel() for p in model.parameters())

    def fwd(*tensors):
        x = {mod: t for mod, t in zip(input_mods, tensors)}
        return model(x)

    inputs_1 = tuple(
        torch.randn(1, in_chans_map[m], IMG_SIZE, IMG_SIZE, device=device)
        for m in input_mods
    )
    active = count_params_touched(model, fwd, inputs_1)
    gmacs = measure_gflops(fwd, inputs_1, device)

    def make_inputs(B):
        return tuple(
            torch.randn(B, in_chans_map[m], IMG_SIZE, IMG_SIZE, device=device)
            for m in input_mods
        )

    best_B, thr = measure_throughput(fwd, make_inputs, device, cache_key=label)

    del model
    gc.collect()
    if device != 'cpu':
        torch.cuda.empty_cache()

    return {
        'model':      label,
        'params_M':   f'{total / 1e6:.1f}',
        'active_M':   f'{active / 1e6:.1f}',
        'gmacs':      f'{gmacs:.1f}',
        'best_B':     best_B,
        'throughput': f'{thr:.0f}',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--delulu_checkpoint', type=str, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    results = []

    IN_CHANS = {STARTING_MOD: S2_CHANS, NEW_MOD: S1_CHANS}

    # ------------------------------------------------------------------ #
    # 1. EVAN SFT (dino) — vanilla classifier, no hallucination
    # ------------------------------------------------------------------ #
    print('\n=== EVAN SFT: s1 ===')
    results.append(profile_evan_sft(
        'dino-s1', DINO_S1_CHECKPOINT, (NEW_MOD,), IN_CHANS, device,
    ))
    print('\n=== EVAN SFT: s2 ===')
    results.append(profile_evan_sft(
        'dino-s2', DINO_S2_CHECKPOINT, (STARTING_MOD,), IN_CHANS, device,
    ))

    # ------------------------------------------------------------------ #
    # 2. DeluluNet
    # ------------------------------------------------------------------ #
    print(f'\n=== Loading DeluluNet from {args.delulu_checkpoint} ===')
    from evan_main import EVANClassifier
    model = EVANClassifier.from_checkpoint(args.delulu_checkpoint, device=device)
    model.eval()
    evan = model.evan
    print(f'  img_size={evan.img_size}, embed_dim={evan.embed_dim}, depth={len(evan.blocks)}')
    print(f'  supported_modalities={evan.supported_modalities}')
    total_delulu = sum(p.numel() for p in model.parameters())

    x_s2_1 = torch.randn(1, S2_CHANS, IMG_SIZE, IMG_SIZE, device=device)
    x_s1_1 = torch.randn(1, S1_CHANS, IMG_SIZE, IMG_SIZE, device=device)

    # -- DeluluNet-s1: feed s1, hallucinate s2 --
    s1peek_wrap = S1PeekWrapper(model).to(device)
    s1peek_active = count_params_unique(active_s1peek_items(model))
    s1peek_gflops = measure_gflops(s1peek_wrap, (x_s1_1,), device)
    s1peek_B, s1peek_thr = measure_throughput(
        s1peek_wrap,
        lambda B: (torch.randn(B, S1_CHANS, IMG_SIZE, IMG_SIZE, device=device),),
        device, cache_key='DeluluNet-s1',
    )
    results.append({
        'model':      'DeluluNet-s1',
        'params_M':   f'{total_delulu / 1e6:.1f}',
        'active_M':   f'{s1peek_active / 1e6:.1f}',
        'gmacs':      f'{s1peek_gflops:.1f}',
        'best_B':     s1peek_B,
        'throughput': f'{s1peek_thr:.0f}',
    })

    # -- DeluluNet-s2: feed s2, hallucinate s1 --
    peek_wrap = PeekWrapper(model).to(device)
    peek_active = count_params_unique(active_peek_items(model))
    peek_gflops = measure_gflops(peek_wrap, (x_s2_1,), device)
    peek_B, peek_thr = measure_throughput(
        peek_wrap,
        lambda B: (torch.randn(B, S2_CHANS, IMG_SIZE, IMG_SIZE, device=device),),
        device, cache_key='DeluluNet-s2',
    )
    results.append({
        'model':      'DeluluNet-s2',
        'params_M':   f'{total_delulu / 1e6:.1f}',
        'active_M':   f'{peek_active / 1e6:.1f}',
        'gmacs':      f'{peek_gflops:.1f}',
        'best_B':     peek_B,
        'throughput': f'{peek_thr:.0f}',
    })

    # -- DeluluNet-s1+s2: both modalities real --
    add_wrap = AdditionWrapper(model).to(device)
    add_active = count_params_unique(active_addition_items(model))
    add_gflops = measure_gflops(add_wrap, (x_s2_1, x_s1_1), device)
    add_B, add_thr = measure_throughput(
        add_wrap,
        lambda B: (
            torch.randn(B, S2_CHANS, IMG_SIZE, IMG_SIZE, device=device),
            torch.randn(B, S1_CHANS, IMG_SIZE, IMG_SIZE, device=device),
        ),
        device, cache_key='DeluluNet-s1+s2',
    )
    results.append({
        'model':      'DeluluNet-s1+s2',
        'params_M':   f'{total_delulu / 1e6:.1f}',
        'active_M':   f'{add_active / 1e6:.1f}',
        'gmacs':      f'{add_gflops:.1f}',
        'best_B':     add_B,
        'throughput': f'{add_thr:.0f}',
    })

    del model, peek_wrap, s1peek_wrap, add_wrap
    gc.collect()
    if device != 'cpu':
        torch.cuda.empty_cache()

    # ------------------------------------------------------------------ #
    # 3. Panopticon ViT-B/14 — s1 / s2 / s2+s1
    # ------------------------------------------------------------------ #
    from rsfm_sft import load_foundation_model, create_classification_head

    def profile_fm(model_name, label, modality, in_chans):
        wrap, dim = load_foundation_model(
            model_name, device, modality=modality, dataset=DATASET, raw_pixels=True
        )
        wrap.eval()
        head = create_classification_head(dim, NUM_CLASSES, device)
        head.eval()
        total = sum(p.numel() for p in wrap.parameters()) + \
                sum(p.numel() for p in head.parameters())

        def fwd(x):
            return head(wrap(x).last_hidden_state)

        x1 = torch.randn(1, in_chans, IMG_SIZE, IMG_SIZE, device=device)
        gmacs = measure_gflops(fwd, (x1,), device)
        best_B, thr = measure_throughput(
            fwd,
            lambda B: (torch.randn(B, in_chans, IMG_SIZE, IMG_SIZE, device=device),),
            device, cache_key=label,
        )

        del wrap, head
        gc.collect()
        if device != 'cpu':
            torch.cuda.empty_cache()

        return {
            'model':      label,
            'params_M':   f'{total / 1e6:.1f}',
            'active_M':   f'{total / 1e6:.1f}',
            'gmacs':      f'{gmacs:.1f}',
            'best_B':     best_B,
            'throughput': f'{thr:.0f}',
        }

    print('\n=== Panopticon ViT-B/14: s1 ===')
    results.append(profile_fm('panopticon', 'Panopticon-s1',    's1',    S1_CHANS))
    print('\n=== Panopticon ViT-B/14: s2 ===')
    results.append(profile_fm('panopticon', 'Panopticon-s2',    's2',    S2_CHANS))
    print('\n=== Panopticon ViT-B/14: s1+s2 ===')
    results.append(profile_fm('panopticon', 'Panopticon-s1+s2', 's2+s1', S2_CHANS + S1_CHANS))

    # OlmoEarth omitted: always runs full multi-modal compute regardless of
    # which modality is passed (mask gates the loss, not the tokens), so
    # single-modality GMACs are not meaningful.

    # ------------------------------------------------------------------ #
    # 4. Summary table
    # ------------------------------------------------------------------ #
    print('\n' + '=' * 70)
    print('  BEN-v2 inference profile  (img_size=128, 19-class, BN1d+Linear head)')
    print('=' * 70)
    print_table(results)

    write_latex_table(results, 'res/latex/computation.tex')

    print('Notes:')
    print('  Active (M)   = params actually executed in this mode (DeluluNet only)')
    print('  GMACs        = multiply-accumulate ops per image (B=1)')
    print('  Batch Size   = largest batch size without OOM')
    print('  Throughput   = images/sec at Batch Size')
    print('  Panopticon s1+s2: channels routed by wavelength ID → same token count as s2\n')


if __name__ == '__main__':
    main()

# python -u profile_delulunet.py