"""
Analyze Pearson and Spearman correlation between hallucinated and real patch tokens.

For each modality (A, B), compares:
  corr(hal_A, real_A)   -- hallucinated A (from B) vs real A  [should be high]
  corr(real_B, real_A)  -- cross-modal baseline               [should be lower]
  corr(real_A, real_A)  -- sanity                             [should be ~1.0]

Two additional controls disentangle per-image content recovery from shared
latent geometry (see shuffled_patch_pearson / batch_pearson):
  shuffled control  -- hal from image i vs real from image j != i
  across-sample r   -- Pearson over the sample axis, dataset mean removed

Both the per-patch and the across-sample metrics are also reported on ranks
(patch_spearman / batch_spearman). The pair is informative as a difference:
  Pearson >> Spearman  -- the linear score rides on a few high-magnitude
                          channels, not on the token as a whole
  Spearman >> Pearson  -- recovery is monotone but at the wrong amplitude,
                          which batch_pearson penalizes and rank does not

Visualizes samples where hallucination is best: raw S2 (B04/B03/B02 true-color), S1 (2ch composite),
and patch token grids (PCA→RGB) for real and hallucinated modalities.

Supports BEN-v2 (classification) and BioMassters (temporal regression); the
dataset and task head are inferred from the checkpoint config.
"""

import argparse
import json
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def sample_pearson(a, b, eps=1e-8):
    """Per-sample Pearson r over (patch, feature), sample-mean-token removed. → [B]

    Centering is over N: each sample's own mean token is subtracted, so a
    projector that emits a constant within an image scores ~0 here however high
    its patch_pearson. Unlike batch_pearson this needs only one sample, so it
    can annotate the per-image visualizations -- but it removes only the
    per-image mean, not the dataset mean shared across images, so it still
    wants a shuffled control.
    """
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    num = (a * b).flatten(1).sum(-1)
    den = a.flatten(1).norm(dim=-1) * b.flatten(1).norm(dim=-1)
    return num / (den + eps)


def shuffled_sample_pearson(a, b, seed=0):
    """sample_pearson with b's samples deranged → [B]. Must be ~0 if valid."""
    B = a.shape[0]
    if B < 2:
        return None
    return sample_pearson(a, b[_derangement(B, seed)])


def feat_pearson(a, b, eps=1e-8):
    """Per-feature Pearson r over the N patches, per sample. a,b: [B,N,D] → [B,D]

    Like sample_pearson this centers over N, but it correlates each feature
    channel separately instead of flattening (N, D) into one vector. Flattening
    fits ONE scale across all coordinates, so a channel the projector learned
    with an inverted sign cancels a channel it got right; here each channel
    keeps its own sign.

    Report the signed mean AND the absolute mean. Signed answers "is there a
    coherent direction", absolute answers "is there structure of either sign",
    and a large gap between them means the projector is sign-heterogeneous.

    Caveat on the absolute mean: |r| is biased positive under the null
    (E|r| ~ 0.8/sqrt(N) ~ 0.10 at N=64), which is the same size as the weaker
    arms' real signal. Always report it as aligned-minus-shuffled, never raw.
    The signed mean carries no such bias.
    """
    a = a - a.mean(1, keepdim=True)
    b = b - b.mean(1, keepdim=True)
    num = (a * b).sum(1)                                  # [B, D]
    den = a.norm(dim=1) * b.norm(dim=1)
    return num / (den + eps)


def shuffled_feat_pearson(a, b, seed=0):
    """feat_pearson with b's samples deranged → [B, D]. Gives the |r| null floor."""
    B = a.shape[0]
    if B < 2:
        return None
    return feat_pearson(a, b[_derangement(B, seed)])


def _derangement(B, seed):
    """Random permutation with no fixed points, via rejection sampling."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.arange(B)
    for _ in range(1000):
        perm = torch.randperm(B, generator=g)
        if not torch.any(perm == idx):
            return perm
    return idx.roll(1)   # B == 2 has exactly one derangement


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


def shuffled_batch_pearson(a, b, seed=0):
    """batch_pearson with b's samples derangement-shuffled → [N, D].

    The audit batch_pearson owes the reader: pairs image i's hallucination with
    image j != i's real tokens. Centering already removes the dataset mean, so a
    working metric must land at ~0 here -- if it does not, batch_pearson is
    scoring shared structure rather than per-image recovery, exactly the failure
    shuffled_patch_pearson exposes in patch_pearson.

    Uses a random derangement rather than a cyclic shift: samples are pooled in
    load order, so roll(1) can pair spatially adjacent tiles and inflate the
    score for reasons that have nothing to do with the projector.
    """
    B = a.shape[0]
    if B < 2:
        return None
    return batch_pearson(a, b[_derangement(B, seed)])


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


def _average_ranks(x, dim):
    """Tie-averaged ranks of x along `dim`, as floats of x's shape.

    Ties must share a rank, or Spearman is not scale-free in the way it
    advertises: a tied pair broken arbitrarily injects a spurious ordering that
    the correlation then scores. torch has no rank primitive, so this is the
    usual double-argsort with a tie-averaging pass on top.
    """
    dim = dim % x.dim()          # scatter/zeros below index dim positionally
    x = x.float()
    n = x.shape[dim]
    order = x.argsort(dim=dim)
    shape = [1] * x.dim()
    shape[dim] = n
    arange = torch.arange(n, device=x.device, dtype=x.dtype)

    # Tie-averaging: walk the sorted values, and give every run of equal values
    # the mean of the ranks it spans. Done vectorially via a cumulative-sum of
    # run boundaries, which assigns each element its run id, then a scatter-mean
    # of ranks over run ids.
    srt = x.gather(dim, order)
    new_run = torch.ones_like(srt, dtype=torch.bool)
    lead = srt.narrow(dim, 1, n - 1)
    prev = srt.narrow(dim, 0, n - 1)
    new_run.narrow(dim, 1, n - 1).copy_(lead != prev)
    run_id = (new_run.cumsum(dim) - 1).long()             # [..., n], 0-based

    sorted_ranks = arange.view(shape).expand_as(x)
    # n slots is the worst case (every element its own run), so this never
    # overflows and no max() sync is needed to size it.
    run_sum = torch.zeros_like(x)
    run_cnt = torch.zeros_like(x)
    run_sum.scatter_add_(dim, run_id, sorted_ranks)
    run_cnt.scatter_add_(dim, run_id, torch.ones_like(x))
    # The buffer is sized at the worst case, so slots past the last real run stay
    # empty; clamp the divisor to keep them 0 rather than nan. They are never
    # gathered, but a nan here would propagate through a later reduction.
    run_mean = run_sum / run_cnt.clamp(min=1.0)
    return run_mean.gather(dim, run_id).gather(dim, _invert_perm(order, dim))


def _invert_perm(order, dim):
    """Index that undoes the sort `order` along `dim`."""
    dim = dim % order.dim()
    inv = torch.empty_like(order)
    n = order.shape[dim]
    shape = [1] * order.dim()
    shape[dim] = n
    arange = torch.arange(n, device=order.device).view(shape).expand_as(order)
    inv.scatter_(dim, order, arange)
    return inv


def patch_spearman(a, b):
    """Per-patch Spearman rho across the feature dim. a,b: [B, N, D] → [B, N]

    patch_pearson's rank counterpart, and it answers a different question: it
    asks whether the projector gets the ORDER of a token's coordinates right,
    not whether it gets their values right. That matters here because the
    token distributions are heavy-tailed -- a handful of high-magnitude
    channels can carry a Pearson r that the remaining hundreds of coordinates
    do not support. Spearman weights every coordinate equally, so a large
    Pearson-minus-Spearman gap means the score is riding on a few outlier
    dimensions.

    Ranking is per-token across D, mirroring patch_pearson's centering, so this
    inherits the same caveat: it leaves the dataset-shared component in. Read
    it against shuffled_patch_spearman.
    """
    return patch_pearson(_average_ranks(a, -1), _average_ranks(b, -1))


def shuffled_patch_spearman(a, b, generator=None):
    """patch_spearman with b's samples derangement-shuffled → [B, N]."""
    B = a.shape[0]
    if B < 2:
        return None
    shift = 1 if generator is None else int(torch.randint(1, B, (1,), generator=generator).item())
    return patch_spearman(a, b.roll(shift, dims=0))


def batch_spearman(a, b):
    """Spearman rho across the SAMPLE axis, per (patch, feature). [B,N,D] → [N,D]

    batch_pearson's rank counterpart. Ranking over B makes the metric immune to
    the per-feature scale mismatch between real and hallucinated tokens: a
    projector whose outputs are a monotone but compressed version of the target
    is penalized by batch_pearson and not by this. Since the open question is
    whether the projector carries per-image information at all -- not whether it
    carries it at the right amplitude -- this is the more forgiving and arguably
    fairer read of the same claim.

    Ranking over B removes the dataset mean by construction (every position's
    ranks are a permutation of 0..B-1), so it keeps batch_pearson's key
    property: a conditional-mean-collapsed projector scores ~0.
    """
    return batch_pearson(_average_ranks(a, 0), _average_ranks(b, 0))


def shuffled_batch_spearman(a, b, seed=0):
    """batch_spearman with b's samples deranged → [N, D]. Must be ~0."""
    B = a.shape[0]
    if B < 2:
        return None
    return batch_spearman(a, b[_derangement(B, seed)])


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
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for the shuffled-control derangements.')
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

    # Spearman counterparts of the two KEY patch-level arms. Only the arms that
    # are actually read are ranked: _average_ranks is a double-argsort, so
    # ranking all ten matrix cells would dominate the loop's cost for cells
    # nobody interprets.
    all_sp_hal_a = []     # spearman(hal_A, real_A)
    all_sp_hal_b = []     # spearman(hal_B, real_B)
    all_sp_xmod_a = []    # spearman(real_B, real_A)  [baseline]

    # ── Additional analysis (does not affect the metrics above) ─────────────
    # Shuffled control: hal from image i vs real from image j != i.
    all_shuf_hal_a = []   # corr(hal_A, real_A[shuffled])
    all_shuf_hal_b = []   # corr(hal_B, real_B[shuffled])
    all_shuf_sp_a = []    # spearman(hal_A, real_A[shuffled])
    all_shuf_sp_b = []    # spearman(hal_B, real_B[shuffled])
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

            all_sp_hal_a.append(patch_spearman(ha, pa).cpu())
            all_sp_hal_b.append(patch_spearman(hb, pb).cpu())
            all_sp_xmod_a.append(patch_spearman(pb, pa).cpu())

            # Shuffled control + raw tokens for the across-sample metric.
            shuf_a = shuffled_patch_pearson(ha, pa)
            shuf_b = shuffled_patch_pearson(hb, pb)
            if shuf_a is not None:
                all_shuf_hal_a.append(shuf_a.cpu())
                all_shuf_hal_b.append(shuf_b.cpu())
                all_shuf_sp_a.append(shuffled_patch_spearman(ha, pa).cpu())
                all_shuf_sp_b.append(shuffled_patch_spearman(hb, pb).cpu())
            tok_pa.append(pa.cpu()); tok_ha.append(ha.cpu())
            tok_pb.append(pb.cpu()); tok_hb.append(hb.cpu())

            B = pa.shape[0]
            # Rank samples for visualization by sample_pearson, not patch_pearson:
            # the latter is ~0.98 regardless of hallucination quality, so ranking
            # on it picks samples essentially at random.
            sp_a = sample_pearson(ha, pa)   # [B]

            for s in range(B):
                # Skip flat/uniform tiles (cloud, ocean). Measure texture on the
                # true-colour bands when available, else on the whole modality.
                # The threshold is distribution-relative (applied after the loop)
                # so it transfers across datasets with different normalizations.
                tex = _drop_time(img_raw[s, modality_slices[tex_mod]])
                rgb_std = (tex[list(rgb_idx[tex_mod])] if rgb_idx[tex_mod] is not None else tex).std().item()
                mean_a = sp_a[s].item()
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

    print('\n=== Token-wise Pearson Correlation ===')
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

    # ── Token-wise Spearman ──────────────────────────────────────────────────
    # Same axis as the Pearson block above (per token, across the feature dim),
    # but on ranks. Read the two together: Pearson >> Spearman means the linear
    # score is carried by a few high-magnitude channels rather than by the
    # token's coordinates as a whole.
    print('\n=== Token-wise Spearman Correlation ===')
    stats(all_sp_hal_a,  f'rho(hal_{mod_a},   real_{mod_a})   [KEY]')
    stats(all_sp_hal_b,  f'rho(hal_{mod_b},   real_{mod_b})   [KEY]')
    stats(all_sp_xmod_a, f'rho(real_{mod_b}, real_{mod_a}) [baseline]')
    for sp_acc, pe_acc, mod in ((all_sp_hal_a, all_corr_hal_a, mod_a),
                                (all_sp_hal_b, all_corr_hal_b, mod_b)):
        m_sp, _ = _ms(sp_acc)
        m_pe, _ = _ms(pe_acc)
        print(f'  pearson - spearman ({mod}): {m_pe:.4f} - {m_sp:.4f} = {m_pe - m_sp:+.4f}'
              f'   (large + => carried by outlier channels)')

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
        # Same control on ranks. The gap, not the aligned score, is the number
        # to compare across arms here too.
        stats(all_shuf_sp_a, f'rho(hal_{mod_a},   real_{mod_a}[shuf])')
        stats(all_shuf_sp_b, f'rho(hal_{mod_b},   real_{mod_b}[shuf])')
        for sp_acc, shuf_acc, mod in ((all_sp_hal_a, all_shuf_sp_a, mod_a),
                                      (all_sp_hal_b, all_shuf_sp_b, mod_b)):
            m_al, _ = _ms(sp_acc)
            m_sh, _ = _ms(shuf_acc)
            print(f'  spearman gap ({mod}): aligned {m_al:.4f} - shuffled {m_sh:.4f} = {m_al - m_sh:.4f}')
    else:
        print('  skipped (batch size < 2)')

    # ── Additional analysis 2: across-sample Pearson ─────────────────────────
    # Centering across the dataset removes the mean token, so a projector that
    # collapses to the conditional mean scores ~0 here however high its
    # patch_pearson is. Computed on pooled tokens for unbiased dataset means.
    _bp_summary = None   # filled below unless the across-sample block is skipped
    _bs_summary = None   # ditto, for the rank version
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
        # Shuffled control: hallucinate from the wrong image. Must be ~0, else
        # batch_pearson is measuring shared structure, not per-image recovery.
        bp_sa = shuffled_batch_pearson(cat_ha, cat_pa, seed=args.seed)
        bp_sb = shuffled_batch_pearson(cat_hb, cat_pb, seed=args.seed)
        for t, label in ((bp_a,  f'corr(hal_{mod_a}, real_{mod_a})   [KEY]'),
                         (bp_b,  f'corr(hal_{mod_b}, real_{mod_b})   [KEY]'),
                         (bp_x,  f'corr(real_{mod_b}, real_{mod_a}) [baseline]'),
                         (bp_sa, f'corr(hal_{mod_a}, real_{mod_a}[shuf]) [null]'),
                         (bp_sb, f'corr(hal_{mod_b}, real_{mod_b}[shuf]) [null]')):
            f = t.flatten()
            print(f'  {label:44s}  mean={f.mean():.4f}  std={f.std():.4f}  median={f.median():.4f}')
        # ── Across-sample Spearman ───────────────────────────────────────────
        # batch_pearson on ranks. Ranking over the sample axis makes this blind
        # to a monotone rescaling of the projector's output, so a hallucination
        # that orders the samples correctly at the wrong amplitude scores here
        # but not on batch_pearson. Same shuffled control, same requirement: the
        # null must land at ~0, or the metric is scoring shared structure.
        print('\n=== Across-sample Spearman (per patch-position & feature) ===')
        bs_a = batch_spearman(cat_ha, cat_pa)   # [N, D]
        bs_b = batch_spearman(cat_hb, cat_pb)
        bs_x = batch_spearman(cat_pb, cat_pa)   # cross-modal baseline
        bs_sa = shuffled_batch_spearman(cat_ha, cat_pa, seed=args.seed)
        bs_sb = shuffled_batch_spearman(cat_hb, cat_pb, seed=args.seed)
        for t, label in ((bs_a,  f'rho(hal_{mod_a}, real_{mod_a})   [KEY]'),
                         (bs_b,  f'rho(hal_{mod_b}, real_{mod_b})   [KEY]'),
                         (bs_x,  f'rho(real_{mod_b}, real_{mod_a}) [baseline]'),
                         (bs_sa, f'rho(hal_{mod_a}, real_{mod_a}[shuf]) [null]'),
                         (bs_sb, f'rho(hal_{mod_b}, real_{mod_b}[shuf]) [null]')):
            f = t.flatten()
            print(f'  {label:44s}  mean={f.mean():.4f}  std={f.std():.4f}  median={f.median():.4f}')
        for bs, bss, bp, mod in ((bs_a, bs_sa, bp_a, mod_a), (bs_b, bs_sb, bp_b, mod_b)):
            m, sd = bs.flatten().mean().item(), bss.flatten().std().item()
            print(f'  margin ({mod}): aligned {m:.4f} vs null 0 +/- {sd:.4f}'
                  f'  ({m / (sd + 1e-8):.1f} sigma)')
            print(f'    pearson - spearman ({mod}): {bp.flatten().mean().item():.4f}'
                  f' - {m:.4f} = {bp.flatten().mean().item() - m:+.4f}'
                  f'   (negative => monotone-but-rescaled recovery)')

        # ── Per-sample Pearson (sample mean token removed) ───────────────────
        # Complements batch_pearson: it is computable on a single image, and it
        # catches a projector that is constant WITHIN an image but varies across
        # images -- which batch_pearson rewards (it carries per-image info) but
        # which has no spatial structure at all.
        print('\n=== Per-sample Pearson (sample mean token removed) ===')
        for ha_, pa_, mod in ((cat_ha, cat_pa, mod_a), (cat_hb, cat_pb, mod_b)):
            sp = sample_pearson(ha_, pa_)
            sps = shuffled_sample_pearson(ha_, pa_, seed=args.seed)
            print(f'  corr(hal_{mod}, real_{mod})        mean={sp.mean():.4f}  std={sp.std():.4f}')
            print(f'  corr(hal_{mod}, real_{mod}[shuf])  mean={sps.mean():.4f}  std={sps.std():.4f}'
                  f'   gap={sp.mean() - sps.mean():.4f}')

        # ── Per-feature Pearson (signed vs absolute) ─────────────────────────
        # sample_pearson flattens (N, D) and so fits one scale across all
        # coordinates: a sign-inverted channel cancels a correct one. Here each
        # channel keeps its own sign. |r| is biased positive under the null, so
        # the absolute row is only meaningful as aligned - shuffled.
        print('\n=== Per-feature Pearson over patches (signed | absolute) ===')
        for ha_, pa_, mod in ((cat_ha, cat_pa, mod_a), (cat_hb, cat_pb, mod_b)):
            fp = feat_pearson(ha_, pa_)
            fps = shuffled_feat_pearson(ha_, pa_, seed=args.seed)
            print(f'  {mod}: signed={fp.mean():+.4f} (shuf {fps.mean():+.4f})'
                  f'   abs={fp.abs().mean():.4f} (shuf {fps.abs().mean():.4f},'
                  f' gap={fp.abs().mean() - fps.abs().mean():+.4f})')
            print(f'      sign-heterogeneity: abs - |signed| = '
                  f'{fp.abs().mean() - fp.mean().abs():+.4f}'
                  f'   (large => channels disagree in sign)')

        # The paper's claim, stated as a margin over the empirical null.
        for bp, bps, mod in ((bp_a, bp_sa, mod_a), (bp_b, bp_sb, mod_b)):
            m, s = bp.flatten().mean().item(), bps.flatten().std().item()
            print(f'  margin ({mod}): aligned {m:.4f} vs null 0 +/- {s:.4f}'
                  f'  ({m / (s + 1e-8):.1f} sigma)')

        # Stashed for the summary.json dump, written once the patch-level
        # accumulators below have been reduced.
        _bp_summary = {
            mod_a: {'aligned': float(bp_a.mean()), 'aligned_std': float(bp_a.std()),
                    'shuffled': float(bp_sa.mean()), 'shuffled_std': float(bp_sa.std())},
            mod_b: {'aligned': float(bp_b.mean()), 'aligned_std': float(bp_b.std()),
                    'shuffled': float(bp_sb.mean()), 'shuffled_std': float(bp_sb.std())},
            'cross_baseline': float(bp_x.mean()),
            'n_samples': int(n_samples),
        }
        _bs_summary = {
            mod_a: {'aligned': float(bs_a.mean()), 'aligned_std': float(bs_a.std()),
                    'shuffled': float(bs_sa.mean()), 'shuffled_std': float(bs_sa.std())},
            mod_b: {'aligned': float(bs_b.mean()), 'aligned_std': float(bs_b.std()),
                    'shuffled': float(bs_sb.mean()), 'shuffled_std': float(bs_sb.std())},
            'cross_baseline': float(bs_x.mean()),
            'n_samples': int(n_samples),
        }

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
\\caption{{Token-wise Pearson correlation matrix (mean$\\pm$std) for real and hallucinated {ma}/{mb} tokens.}}
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

    # Machine-readable dump so a multi-model table can be assembled without
    # re-parsing stdout. _bp_summary is absent when the across-sample block was
    # skipped (n_samples < 2).
    summary = {
        'checkpoint': args.checkpoint, 'dataset': ds_name,
        'mod_a': mod_a, 'mod_b': mod_b,
        'batch_pearson': _bp_summary,
        'batch_spearman': _bs_summary,
        'patch_spearman': {
            mod_a: {'aligned': _ms(all_sp_hal_a),
                    'shuffled': _ms(all_shuf_sp_a) if all_shuf_sp_a else None},
            mod_b: {'aligned': _ms(all_sp_hal_b),
                    'shuffled': _ms(all_shuf_sp_b) if all_shuf_sp_b else None},
            'cross_baseline': _ms(all_sp_xmod_a),
        },
        'patch_pearson_matrix': {
            f'{labels[i]}|{labels[j]}': [mean_mat[i, j], std_mat[i, j]]
            for i in range(n) for j in range(n)
        },
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    print(f"Saved {os.path.join(args.out_dir, 'summary.json')}")

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
    ax.text(0.98, 0.98, "Token-wise\nPearson\nCorrelation", fontsize=13,
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