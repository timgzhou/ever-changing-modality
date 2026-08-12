"""
Analyze Pearson correlation between hallucinated and real patch tokens.

For each modality (A, B), compares:
  corr(hal_A, real_A)   -- hallucinated A (from B) vs real A  [should be high]
  corr(real_B, real_A)  -- cross-modal baseline               [should be lower]
  corr(real_A, real_A)  -- sanity                             [should be ~1.0]

Two additional controls disentangle per-image content recovery from shared
latent geometry (see shuffled_patch_pearson / batch_pearson):
  shuffled control  -- hal from image i vs real from image j != i
  across-sample r   -- Pearson over the sample axis, dataset mean removed

Visualizes samples where hallucination is best: raw S2 (B04/B03/B02 true-color), S1 (2ch composite),
and patch token grids (PCA→RGB) for real and hallucinated modalities.

Supports BEN-v2 (classification) and BioMassters (temporal regression); the
dataset and task head are inferred from the checkpoint config.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['font.family'] = 'serif'
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from delulunet_main import EVANClassifier, EvanSegmenter
from data_utils import create_multimodal_batch, get_loaders

_DATASETS = ('eurosat', 'benv2', 'dfc2020', 'biomassters')

# Modality pairs that identify a dataset when the checkpoint doesn't record one.
# EuroSAT modalities are S2 sub-bands; benv2/dfc2020/biomassters use s1/s2.
_EUROSAT_MODS = {'rgb', 'vre', 'nir', 'swir', 'aw'}


def _infer_dataset(mods, ckpt_config):
    """Best-effort dataset name from checkpoint metadata, else from modality names."""
    ds = ckpt_config.get('dataset')
    if ds:
        return ds
    if any(m in _EUROSAT_MODS for m in mods):
        return 'eurosat'
    # s1/s2 pairs are ambiguous across benv2 / dfc2020 / biomassters.
    raise ValueError(
        f"Checkpoint does not record a dataset and modalities {list(mods)} are "
        f"ambiguous. Pass --dataset explicitly (one of {list(_DATASETS)})."
    )


_GEOBENCH_S2_BANDS = {
    'benv2':       ('geobench_data_utils',     'BENV2_S2_BANDS'),
    'biomassters': ('biomassters_data_utils',  'BIOMASSTERS_S2_BANDS'),
}


def _geobench_s2_bands(ds_name):
    """S2 band order for GeoBench-style datasets; () when unknown (-> grayscale).

    Read from the already-imported dataset module (get_loaders imports it), so a
    missing optional dependency never silently downgrades the S2 panel.
    """
    entry = _GEOBENCH_S2_BANDS.get(ds_name)
    if entry is None:
        return ()
    mod_name, attr = entry
    mod = sys.modules.get(mod_name)
    if mod is None:
        import importlib
        mod = importlib.import_module(mod_name)
    return getattr(mod, attr, ())


def _rgb_indices(bands):
    """(R, G, B) positions of B04/B03/B02 within a modality's band list.

    Returns None when the modality has no true-colour triple (e.g. EuroSAT
    'vre'/'nir', or S1), in which case the caller falls back to a grayscale view.
    """
    b = [str(x) for x in bands]
    try:
        return b.index('B04'), b.index('B03'), b.index('B02')
    except ValueError:
        return None


def _load_model(path, device):
    """Load an EVAN predictor, dispatching on the checkpoint's task head.

    Classification checkpoints carry 'classifier_strategy'; dense-prediction
    (segmentation / regression) checkpoints carry 'decoder_strategy'.
    """
    config = torch.load(path, map_location='cpu', weights_only=False)['config']
    if 'classifier_strategy' in config:
        return EVANClassifier.from_checkpoint(path, device), 'classifier'
    if 'decoder_strategy' in config:
        return EvanSegmenter.from_checkpoint(path, device), 'segmenter'
    raise ValueError(
        f"Cannot determine head type for {path}: config has neither "
        f"'classifier_strategy' nor 'decoder_strategy'. Keys: {sorted(config)}"
    )


def patch_pearson(a, b):
    """Per-patch Pearson r across feature dim. a,b: [B, N, D] → [B, N]

    Centering is per-token across D, so this removes each token's own DC offset
    but NOT any component shared across the dataset (mean token direction,
    layernorm geometry, positional structure). See batch_pearson() for the
    across-samples variant that does remove it.
    """
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)


def shuffled_patch_pearson(a, b, generator=None):
    """patch_pearson with b's samples derangement-shuffled → [B, N].

    Control for the shared-region component: pairs hallucinated tokens from
    image i with real tokens from image j != i. If this matches the aligned
    score, patch_pearson is measuring latent geometry rather than per-image
    content recovery.
    """
    B = a.shape[0]
    if B < 2:
        return None
    # Derangement via cyclic shift: guarantees no index maps to itself.
    shift = 1 if generator is None else int(torch.randint(1, B, (1,), generator=generator).item())
    return patch_pearson(a, b.roll(shift, dims=0))


def batch_pearson(a, b, eps=1e-8):
    """Pearson r across the SAMPLE axis, per (patch, feature). a,b: [B, N, D] → [N, D]

    Centering across B subtracts the dataset mean token, removing the shared
    centroid that patch_pearson leaves in. This isolates per-image information:
    a projector that collapses to the conditional mean scores ~0 here regardless
    of how high its patch_pearson is.
    """
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    num = (a * b).sum(0)
    den = a.norm(dim=0) * b.norm(dim=0)
    return num / (den + eps)


def _stretch(arr_hwc):
    """Joint percentile stretch across all channels. arr_hwc: np.float32 [H,W,C] → uint8."""
    lo = np.percentile(arr_hwc, 2)
    hi = np.percentile(arr_hwc, 98)
    if hi > lo:
        out = np.clip((arr_hwc - lo) / (hi - lo), 0, 1)
    else:
        out = np.zeros_like(arr_hwc)
    return (out * 255).astype(np.uint8)


def _drop_time(x):
    """[C, T, H, W] -> [C, H, W] by mean over T; [C, H, W] passes through."""
    return x.mean(1) if x.dim() == 4 else x


def modality_to_rgb(img_chw, band_spec, rgb_idx=None):
    """Render one modality as an image with joint percentile stretch.

    True-colour when the modality contains B04/B03/B02 (rgb_idx given), otherwise
    a grayscale channel-average. Works for any channel count and for temporal
    [C, T, H, W] inputs.
    """
    x = _drop_time(img_chw[band_spec]).cpu().numpy().astype(np.float32)
    if rgb_idx is not None:
        r, g, b = rgb_idx
        return _stretch(np.stack([x[r], x[g], x[b]], axis=-1))
    avg = x.mean(0)  # [H, W]
    return _stretch(np.stack([avg, avg, avg], axis=-1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/delulunet_benv2_0501_0433.pt')
    parser.add_argument('--dataset', default=None, choices=sorted(_DATASETS),
                        help='Dataset loaders to use. Default: read from the checkpoint, '
                             'else inferred from modality names.')
    parser.add_argument('--num_time_steps', type=int, default=None,
                        help='Temporal datasets (biomassters): timesteps to load. '
                             'Default: taken from the checkpoint config.')
    parser.add_argument('--out_dir', default='res/hallucination_correlation')
    parser.add_argument('--n_batches', type=int, default=20)
    parser.add_argument('--n_vis', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load model (dispatch on task head) ──────────────────────────────────
    # Run metadata ('dataset', 'num_time_steps') lives at the checkpoint top
    # level; model hyperparameters live under 'config'. Merge so lookups below
    # find either, with 'config' taking precedence on key collisions.
    _ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    ckpt_config = {k: v for k, v in _ckpt.items() if k != 'model_state_dict'}
    ckpt_config.update(_ckpt.get('config', {}))
    del _ckpt
    model, head_kind = _load_model(args.checkpoint, args.device)
    model.eval()
    evan = model.evan
    n_storage = evan.n_storage_tokens

    mods = evan.supported_modalities   # e.g. ['s1', 's2'] or ['s2', 's1']
    mod_a, mod_b = mods[0], mods[1]
    print(f"Modalities: A={mod_a}, B={mod_b}")
    print(f"Projector type: {evan.intermediate_projector_type}")
    print(f"Head type: {head_kind}")

    # ── Load data ────────────────────────────────────────────────────────────
    # get_loaders() normalizes every dataset to the same 5-loader + TaskConfig
    # interface, including EuroSAT (band-name tuples) vs GeoBench (slices).
    ds_name = args.dataset or _infer_dataset(mods, ckpt_config)
    if ds_name not in _DATASETS:
        raise ValueError(f"Unsupported dataset {ds_name!r}; choose from {list(_DATASETS)}")
    print(f"Dataset: {ds_name}")

    # Match the checkpoint's T unless overridden; the model mean-pools over it.
    n_t = args.num_time_steps or ckpt_config.get('num_time_steps') or 10
    if ds_name == 'biomassters':
        print(f"num_time_steps: {n_t}")

    _, _, _, _, test_loader, task_config = get_loaders(
        ds_name,
        starting_modality=mod_a,
        batch_size=args.batch_size,
        num_workers=4,
        new_modality=mod_b,
        num_time_steps=n_t,
    )
    modality_slices = task_config.modality_bands_dict

    # Preflight: catch a dataset/checkpoint mismatch here rather than as an
    # opaque conv2d channel error deep in the patch embedder.
    expected_chans = dict(zip(evan.supported_modalities, evan.supported_modalities_in_chans))
    for mod in (mod_a, mod_b):
        spec_bands = modality_slices[mod]
        got = (spec_bands.stop - spec_bands.start) if isinstance(spec_bands, slice) else len(spec_bands)
        want = expected_chans.get(mod)
        if want is not None and got != want:
            raise ValueError(
                f"Channel mismatch for modality {mod!r}: checkpoint expects {want} "
                f"channels but dataset {ds_name!r} provides {got}. The checkpoint was "
                f"likely trained on a different dataset — pass --dataset explicitly."
            )

    # Per-modality true-colour indices for visualization. EuroSAT stores band
    # names directly in modality_bands_dict; GeoBench stores slices into the
    # stacked image, so the S2 band order comes from the dataset module.
    # None -> render that modality as a grayscale channel-average.
    def _viz_rgb(mod):
        band_spec = modality_slices[mod]
        if not isinstance(band_spec, slice):
            return _rgb_indices(band_spec)          # EuroSAT: tuple of band names
        if mod == 's2':
            return _rgb_indices(_geobench_s2_bands(ds_name))
        return None                                  # s1 and friends: no true colour

    rgb_idx = {mod: _viz_rgb(mod) for mod in (mod_a, mod_b)}
    # Texture filter runs on whichever modality has a true-colour view.
    tex_mod = next((m for m in (mod_a, mod_b) if rgb_idx[m] is not None), mod_a)
    """
    # ── Evaluate checkpoint on test split (both modalities) ─────────────────
    print('\n=== Evaluating checkpoint on test split (multimodal) ===')
    all_eval_outputs = []
    all_eval_labels = []
    with torch.no_grad():
        for batch in test_loader:
            labels = batch['label'].float().to(args.device)
            modal_input = create_multimodal_batch_geobench(batch, modality_slices, (mod_a, mod_b))
            modal_input = {k: v.to(args.device) for k, v in modal_input.items()}
            logits = model(modal_input)
            all_eval_outputs.append(logits.cpu())
            all_eval_labels.append(labels.cpu())
    test_map = _compute_map(torch.cat(all_eval_outputs), torch.cat(all_eval_labels))
    print(f'  Test mAP (both modalities): {test_map:.2f}%')
    """
    # ── Accumulate tokens and correlations ──────────────────────────────────
    all_corr_hal_a = []   # corr(hal_A, real_A)
    all_corr_xmod_a = []  # corr(real_B, real_A)
    all_corr_hal_b = []   # corr(hal_B, real_B)
    all_corr_xmod_b = []  # corr(real_A, real_B)
    all_corr_aa = []      # sanity
    all_corr_bb = []      # sanity
    # extra pairs for 4×4 matrix
    all_corr_haha = []    # corr(hal_A, hal_A)
    all_corr_hbhb = []    # corr(hal_B, hal_B)
    all_corr_hahb = []    # corr(hal_A, hal_B)
    all_corr_ha_rb = []   # corr(hal_A, real_B)
    all_corr_hb_ra = []   # corr(hal_B, real_A)

    # ── Additional analysis (does not affect the metrics above) ─────────────
    # Shuffled control: hal from image i vs real from image j != i.
    all_shuf_hal_a = []   # corr(hal_A, real_A[shuffled])
    all_shuf_hal_b = []   # corr(hal_B, real_B[shuffled])
    # Raw tokens, kept to compute across-sample correlation on full-dataset
    # statistics (per-batch means would bias batch_pearson at small B).
    tok_pa, tok_ha, tok_pb, tok_hb = [], [], [], []

    # For visualization: keep raw images and patch tokens for high-corr samples
    vis_candidates = []  # (mean_corr_a, img_raw, patches_a, hal_patches_a, patches_b, hal_patches_b, corr_map_a, corr_map_b)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.n_batches:
                break

            img_raw = batch['image']  # [B, C_total, H, W] — z-score normalized, percentile-stretched for viz
            x = create_multimodal_batch(batch, modality_slices, (mod_a, mod_b))
            x_a = x[mod_a].to(args.device)
            x_b = x[mod_b].to(args.device)

            embedded = evan.forward_modality_specific_features({mod_a: x_a, mod_b: x_b})
            seq_a = embedded[mod_a]   # [B, 1+n_storage+N, D]
            seq_b = embedded[mod_b]

            # Hallucinate A from B
            hal_a = evan._project_sequence(seq_b, f'{mod_b}_to_{mod_a}', mod_a)  # [B, 1+N, D]

            # Hallucinate B from A
            hal_b = evan._project_sequence(seq_a, f'{mod_a}_to_{mod_b}', mod_b)  # [B, 1+N, D]

            # Patch tokens (skip CLS + storage for real; skip CLS for hallucinated cross)
            pa  = seq_a[:, 1 + n_storage:, :].float()   # [B, N, D]
            pb  = seq_b[:, 1 + n_storage:, :].float()
            ha  = hal_a[:, 1:, :].float()                # [B, N, D]
            hb  = hal_b[:, 1:, :].float()

            corr_hal_a  = patch_pearson(ha, pa)    # [B, N]
            corr_xmod_a = patch_pearson(pb, pa)
            corr_hal_b  = patch_pearson(hb, pb)
            corr_xmod_b = patch_pearson(pa, pb)
            corr_aa     = patch_pearson(pa, pa)
            corr_bb     = patch_pearson(pb, pb)

            all_corr_hal_a.append(corr_hal_a.cpu())
            all_corr_xmod_a.append(corr_xmod_a.cpu())
            all_corr_hal_b.append(corr_hal_b.cpu())
            all_corr_xmod_b.append(corr_xmod_b.cpu())
            all_corr_aa.append(corr_aa.cpu())
            all_corr_bb.append(corr_bb.cpu())
            all_corr_haha.append(patch_pearson(ha, ha).cpu())
            all_corr_hbhb.append(patch_pearson(hb, hb).cpu())
            all_corr_hahb.append(patch_pearson(ha, hb).cpu())
            all_corr_ha_rb.append(patch_pearson(ha, pb).cpu())
            all_corr_hb_ra.append(patch_pearson(hb, pa).cpu())

            # Shuffled control + raw tokens for the across-sample metric.
            shuf_a = shuffled_patch_pearson(ha, pa)
            shuf_b = shuffled_patch_pearson(hb, pb)
            if shuf_a is not None:
                all_shuf_hal_a.append(shuf_a.cpu())
                all_shuf_hal_b.append(shuf_b.cpu())
            tok_pa.append(pa.cpu()); tok_ha.append(ha.cpu())
            tok_pb.append(pb.cpu()); tok_hb.append(hb.cpu())

            B = pa.shape[0]
            for s in range(B):
                # Skip flat/uniform tiles (cloud, ocean). Measure texture on the
                # true-colour bands when available, else on the whole modality.
                # The threshold is distribution-relative (applied after the loop)
                # so it transfers across datasets with different normalizations.
                tex = _drop_time(img_raw[s, modality_slices[tex_mod]])
                rgb_std = (tex[list(rgb_idx[tex_mod])] if rgb_idx[tex_mod] is not None else tex).std().item()
                mean_a = corr_hal_a[s].mean().item()
                vis_candidates.append((
                    mean_a,
                    img_raw[s].cpu(),
                    pa[s].cpu(), ha[s].cpu(),
                    pb[s].cpu(), hb[s].cpu(),
                    corr_hal_a[s].cpu(), corr_hal_b[s].cpu(),
                    rgb_std,
                ))

    # ── Print metrics ────────────────────────────────────────────────────────
    def stats(tensors, label):
        t = torch.cat(tensors).flatten()
        print(f"  {label:40s}  mean={t.mean():.4f}  std={t.std():.4f}")

    print('\n=== Patch-level Pearson Correlation ===')
    print(f'  (A={mod_a}, B={mod_b}, N patches per sample)')
    stats(all_corr_hal_a,  f'corr(hal_{mod_a},   real_{mod_a})   [KEY]')
    stats(all_corr_xmod_a, f'corr(real_{mod_b}, real_{mod_a}) [baseline]')
    stats(all_corr_hal_b,  f'corr(hal_{mod_b},   real_{mod_b})   [KEY]')
    stats(all_corr_xmod_b, f'corr(real_{mod_a}, real_{mod_b}) [baseline]')
    stats(all_corr_aa,     f'corr(real_{mod_a}, real_{mod_a}) [sanity≈1]')
    stats(all_corr_bb,     f'corr(real_{mod_b}, real_{mod_b}) [sanity≈1]')

    def _ms(tensors):
        t = torch.cat(tensors).flatten()
        return t.mean().item(), t.std().item()

    def _cell(tensors):
        m, s = _ms(tensors)
        return f'${m:.3f}\\pm{s:.3f}$'

    # ── Additional analysis 1: shuffled-image control ────────────────────────
    # Isolates the shared-region component. If the shuffled score is close to
    # the aligned score, patch_pearson reflects latent geometry (all real S2
    # tokens occupying a common region) rather than per-image content recovery.
    print('\n=== Shuffled-image control (patch_pearson, mismatched pairs) ===')
    if all_shuf_hal_a:
        stats(all_shuf_hal_a, f'corr(hal_{mod_a},   real_{mod_a}[shuf])')
        stats(all_shuf_hal_b, f'corr(hal_{mod_b},   real_{mod_b}[shuf])')
        m_al_a, _ = _ms(all_corr_hal_a)
        m_sh_a, _ = _ms(all_shuf_hal_a)
        m_al_b, _ = _ms(all_corr_hal_b)
        m_sh_b, _ = _ms(all_shuf_hal_b)
        print(f'  gap ({mod_a}): aligned {m_al_a:.4f} - shuffled {m_sh_a:.4f} = {m_al_a - m_sh_a:.4f}')
        print(f'  gap ({mod_b}): aligned {m_al_b:.4f} - shuffled {m_sh_b:.4f} = {m_al_b - m_sh_b:.4f}')
        print('  Large gap → per-image recovery. Small gap → shared latent geometry.')
    else:
        print('  skipped (batch size < 2)')

    # ── Additional analysis 2: across-sample Pearson ─────────────────────────
    # Centering across the dataset removes the mean token, so a projector that
    # collapses to the conditional mean scores ~0 here however high its
    # patch_pearson is. Computed on pooled tokens for unbiased dataset means.
    print('\n=== Across-sample Pearson (per patch-position & feature) ===')
    cat_pa = torch.cat(tok_pa); cat_ha = torch.cat(tok_ha)
    cat_pb = torch.cat(tok_pb); cat_hb = torch.cat(tok_hb)
    n_samples = cat_pa.shape[0]
    print(f'  pooled over {n_samples} samples')
    if n_samples < 2:
        print('  skipped (need >= 2 samples)')
    else:
        bp_a = batch_pearson(cat_ha, cat_pa)   # [N, D]
        bp_b = batch_pearson(cat_hb, cat_pb)
        bp_x = batch_pearson(cat_pb, cat_pa)   # cross-modal baseline
        for t, label in ((bp_a, f'corr(hal_{mod_a}, real_{mod_a})   [KEY]'),
                         (bp_b, f'corr(hal_{mod_b}, real_{mod_b})   [KEY]'),
                         (bp_x, f'corr(real_{mod_b}, real_{mod_a}) [baseline]')):
            f = t.flatten()
            print(f'  {label:40s}  mean={f.mean():.4f}  std={f.std():.4f}  median={f.median():.4f}')

    ma, mb = mod_a.upper(), mod_b.upper()

    # rows/cols order: real_A, real_B, hal_A, hal_B
    # symmetric pairs reuse the same accumulator (Pearson is symmetric)
    c = {
        ('rA','rA'): _cell(all_corr_aa),
        ('rA','rB'): _cell(all_corr_xmod_b),   # corr(real_A, real_B)
        ('rA','hA'): _cell(all_corr_hal_a),     # corr(hal_A, real_A)
        ('rA','hB'): _cell(all_corr_hb_ra),     # corr(hal_B, real_A)
        ('rB','rB'): _cell(all_corr_bb),
        ('rB','hA'): _cell(all_corr_ha_rb),     # corr(hal_A, real_B)
        ('rB','hB'): _cell(all_corr_hal_b),     # corr(hal_B, real_B)
        ('hA','hA'): _cell(all_corr_haha),
        ('hA','hB'): _cell(all_corr_hahb),
        ('hB','hB'): _cell(all_corr_hbhb),
    }
    # fill symmetric lower triangle
    for (r, c_), v in list(c.items()):
        c[(c_, r)] = v

    keys  = ['rA', 'rB', 'hA', 'hB']
    names = [f'real {ma}', f'real {mb}', f'hall {ma}', f'hall {mb}']

    def trow(i):
        cells = ' & '.join(c[(keys[i], keys[j])] for j in range(4))
        return f'  {names[i]} & {cells} \\\\'

    print(f"""
\\begin{{table}}[h]
\\centering
\\caption{{Patch-level Pearson correlation matrix (mean$\\pm$std) for real and hallucinated {ma}/{mb} tokens.}}
\\label{{tab:hallucination_corr}}
\\begin{{tabular}}{{lcccc}}
\\toprule
 & real {ma} & real {mb} & hall {ma} & hall {mb} \\\\
\\midrule
{trow(0)}
{trow(1)}
{trow(2)}
{trow(3)}
\\bottomrule
\\end{{tabular}}
\\end{{table}}""")

    # ── Correlation heatmap ──────────────────────────────────────────────────
    accumulators = {
        ('rA','rA'): all_corr_aa,
        ('rA','rB'): all_corr_xmod_b,
        ('rA','hA'): all_corr_hal_a,
        ('rA','hB'): all_corr_hb_ra,
        ('rB','rB'): all_corr_bb,
        ('rB','hA'): all_corr_ha_rb,
        ('rB','hB'): all_corr_hal_b,
        ('hA','hA'): all_corr_haha,
        ('hA','hB'): all_corr_hahb,
        ('hB','hB'): all_corr_hbhb,
    }
    for (r, c_), v in list(accumulators.items()):
        accumulators[(c_, r)] = v

    keys  = ['rA', 'rB', 'hA', 'hB']
    labels = [f'real {ma}', f'real {mb}', f'hall {ma}', f'hall {mb}']
    n = len(keys)
    mean_mat = np.zeros((n, n))
    std_mat  = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            m, s = _ms(accumulators[(ki, kj)])
            mean_mat[i, j] = m
            std_mat[i, j]  = s

    # Mask upper triangle (above diagonal) to show only lower triangle + diagonal
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    display_mat = np.where(mask, np.nan, mean_mat)
    # diagonal shown as flat gray, not on the correlation colorscale
    for i in range(n):
        display_mat[i, i] = np.nan  # will be painted gray via Rectangle below

    fig, ax = plt.subplots(figsize=(5, 3.4))
    import seaborn as sns
    cmap = sns.diverging_palette(240, 10, as_cmap=True)  # vlag equivalent
    cmap.set_bad('white')
    im = ax.imshow(display_mat, cmap=cmap, vmin=-1, vmax=1, aspect=0.8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=12, rotation=0, ha='center')
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=12)
    ax.text(0.98, 0.98, "Patch-level\nPearson\nCorrelation", fontsize=13,
            ha='right', va='top', transform=ax.transAxes, color='black')
    for i in range(n):
        ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1,
                         fill=True, facecolor='lightgray', edgecolor='none', zorder=2))
        ax.text(i, i, '1.0', ha='center', va='center', fontsize=11, zorder=3, color='dimgray')
        for j in range(n):
            if j >= i:
                continue
            ax.text(j, i, f'{mean_mat[i,j]:.3f}\n±{std_mat[i,j]:.3f}',
                    ha='center', va='center', fontsize=11, zorder=3,
                    color='white' if abs(mean_mat[i,j]) > 0.5 else 'black')
    plt.tight_layout()
    heatmap_path = os.path.join(args.out_dir, 'corr_matrix.pdf')
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {heatmap_path}')

    def make_pca_rgb(pca, lo, hi, tokens):
        """Project tokens into a pre-fit PCA and normalize to [0,1] with fixed range."""
        proj = pca.transform(tokens.float().numpy())
        return np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)

    # ── Visualize top-n_vis samples ──────────────────────────────────────────
    # Drop the flattest 25% of tiles (cloud/ocean) using a distribution-relative
    # cutoff, so this works regardless of the dataset's normalization scheme.
    if vis_candidates:
        std_cut = float(np.percentile([c[-1] for c in vis_candidates], 25))
        kept = [c for c in vis_candidates if c[-1] >= std_cut]
        print(f'\nViz: {len(kept)}/{len(vis_candidates)} tiles pass texture filter '
              f'(RGB std >= {std_cut:.3f})')
    else:
        kept = []
    kept.sort(key=lambda x: x[0], reverse=True)
    top = kept[:args.n_vis]

    slice_a = modality_slices[mod_a]
    slice_b = modality_slices[mod_b]
    label_a = f'{mod_a.upper()} (RGB)' if rgb_idx[mod_a] is not None else f'{mod_a.upper()} (mean)'
    label_b = f'{mod_b.upper()} (RGB)' if rgb_idx[mod_b] is not None else f'{mod_b.upper()} (mean)'

    if not top:
        print('No tiles available for visualization; skipping sample figures.')
        print('\nDone.')
        return

    n_patches = top[0][2].shape[0]
    grid_size = int(n_patches ** 0.5)  # 8 for BEN-v2, 16 for BioMassters

    # Layout (2 rows × 4 cols):
    #   cols 0-1: S2 / S1 raw images, each spanning both rows (full height)
    #   col 2: real_A tokens (row 0), real_B tokens (row 1)
    #   col 3: hal_A tokens (row 0), hal_B tokens (row 1)
    #   token cols are narrow so two stacked token grids match the height of the input images.
    _TOK = 0.45    # token col width relative to image col; two stacked squares → ~same height as one image
    _col_ratios = [1, 1, _TOK, _TOK]

    for idx, (mean_corr_a, img_raw, pa, ha, pb, hb, corr_a_map, corr_b_map, _std) in enumerate(top):
        fig = plt.figure(figsize=(10, 5))

        # PCA fit on both real token sets jointly; hallucinated tokens projected into same space.
        real_both = np.concatenate([pa.float().numpy(), pb.float().numpy()], axis=0)
        pca = PCA(n_components=3).fit(real_both)
        proj_real = pca.transform(real_both)
        lo, hi = proj_real.min(0), proj_real.max(0)

        rgb_pa = make_pca_rgb(pca, lo, hi, pa).reshape(grid_size, grid_size, 3)
        rgb_ha = make_pca_rgb(pca, lo, hi, ha).reshape(grid_size, grid_size, 3)
        rgb_pb = make_pca_rgb(pca, lo, hi, pb).reshape(grid_size, grid_size, 3)
        rgb_hb = make_pca_rgb(pca, lo, hi, hb).reshape(grid_size, grid_size, 3)

        # Pixel-based layout (fig is 10×5 in at 150 dpi = 1500×750 px).
        # Left images: 224×224 px. Token panels: 100×100 px, gap 24 px between rows.
        # All panels share the same top and bottom edge.
        FW, FH = 1500, 750   # figure size in pixels at 150 dpi
        px = lambda v: v / FW  # horizontal fraction
        py = lambda v: v / FH  # vertical fraction

        img_px = 224
        tok_px = 100
        gap_col = 20    # horizontal gap between panels
        gap_mid = 50    # wider gap between real col and hall col (for arrows)
        gap_row = 24    # vertical gap between the two token rows

        # Bottom edge: vertically centre the 224px block in the figure
        img_bottom = (FH - img_px) / 2          # 263 px from bottom
        tok_bottom_lo = img_bottom               # lower token row aligns with image bottom
        tok_bottom_hi = img_bottom + tok_px + gap_row  # upper token row

        x0 = 20
        x1 = x0 + img_px + gap_col
        x2 = x1 + img_px + gap_col   # real col
        x3 = x2 + tok_px + gap_mid   # hall col (wider gap)

        ax_s2 = fig.add_axes([px(x0), py(img_bottom), px(img_px), py(img_px)])
        ax_s1 = fig.add_axes([px(x1), py(img_bottom), px(img_px), py(img_px)])
        ax_pa = fig.add_axes([px(x2), py(tok_bottom_hi), px(tok_px), py(tok_px)])
        ax_ha = fig.add_axes([px(x3), py(tok_bottom_hi), px(tok_px), py(tok_px)])
        ax_pb = fig.add_axes([px(x2), py(tok_bottom_lo), px(tok_px), py(tok_px)])
        ax_hb = fig.add_axes([px(x3), py(tok_bottom_lo), px(tok_px), py(tok_px)])

        # Arrow: bottom-right of real s2 (ax_pa) → upper-left of hall s1 (ax_hb)
        # Arrow: upper-right of real s1 (ax_pb) → lower-left of hall s2 (ax_ha)
        arrow_kw = dict(arrowstyle='->', color='dimgray', lw=1.2,
                        mutation_scale=10)
        fig.add_artist(matplotlib.patches.FancyArrowPatch(
            (px(x2 + tok_px), py(tok_bottom_hi)),
            (px(x3),          py(tok_bottom_lo + tok_px)),
            transform=fig.transFigure, **arrow_kw))
        fig.add_artist(matplotlib.patches.FancyArrowPatch(
            (px(x2 + tok_px), py(tok_bottom_lo + tok_px)),
            (px(x3),          py(tok_bottom_hi)),
            transform=fig.transFigure, **arrow_kw))

        for ax in (ax_s2, ax_s1, ax_pa, ax_ha, ax_pb, ax_hb):
            ax.axis('off')

        ax_s2.imshow(modality_to_rgb(img_raw, slice_a, rgb_idx[mod_a]))
        ax_s2.text(0.5, -0.02, label_a, fontsize=11, ha='center', va='top', transform=ax_s2.transAxes)

        ax_s1.imshow(modality_to_rgb(img_raw, slice_b, rgb_idx[mod_b]))
        ax_s1.text(0.5, -0.02, label_b, fontsize=11, ha='center', va='top', transform=ax_s1.transAxes)

        ax_pa.imshow(rgb_pa, interpolation='nearest')
        ax_pa.text(0.5, -0.04, f'real {mod_a}', fontsize=11, ha='center', va='top', transform=ax_pa.transAxes)

        ax_ha.imshow(rgb_ha, interpolation='nearest')
        ax_ha.text(0.5, -0.04, f'hall {mod_a}', fontsize=11, ha='center', va='top', transform=ax_ha.transAxes)

        ax_pb.imshow(rgb_pb, interpolation='nearest')
        ax_pb.text(0.5, -0.04, f'real {mod_b}', fontsize=11, ha='center', va='top', transform=ax_pb.transAxes)

        ax_hb.imshow(rgb_hb, interpolation='nearest')
        ax_hb.text(0.5, -0.04, f'hall {mod_b}', fontsize=11, ha='center', va='top', transform=ax_hb.transAxes)

        out_path = os.path.join(args.out_dir, f'sample_{idx:03d}.pdf')
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f'Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()


# BEN-v2 (classification):
# python -u analyze_hallucination_correlation.py --checkpoint checkpoints/delulu-checkpoints/sweep_lr7ygzoh_0501_1505.pt
# python -u analyze_hallucination_correlation.py --checkpoint checkpoints/delulu-checkpoints/delulunet_benv2_0501_0635.pt
# python -u analyze_hallucination_correlation.py --checkpoint checkpoints/delulu-checkpoints/delulunet_benv2_0501_1943.pt
#
# BioMassters (temporal regression) — dataset/T/head inferred from the checkpoint.
# Use a smaller batch: 12 timesteps at 256px is far heavier than BEN-v2.
# python -u analyze_hallucination_correlation.py \
#     --checkpoint checkpoints/delulunet_biomassters_s1s2_addition_rank1_seed2.pt \
#     --batch_size 4 --n_batches 40 --out_dir res/hallucination_correlation/biomassters