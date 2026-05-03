"""
Analyze Pearson correlation between hallucinated and real patch tokens.

For each modality (A, B), compares:
  corr(hal_A, real_A)   -- hallucinated A (from B) vs real A  [should be high]
  corr(real_B, real_A)  -- cross-modal baseline               [should be lower]
  corr(real_A, real_A)  -- sanity                             [should be ~1.0]

Visualizes samples where hallucination is best: raw S2 (B04/B03/B02 true-color), S1 (2ch composite),
and 8×8 patch token grids (PCA→RGB) for real and hallucinated modalities.
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
from evan_main import EVANClassifier
from geobench_data_utils import get_benv2_loaders, create_multimodal_batch_geobench, BENV2_S2_BANDS

# BEN-v2 S2 band indices for RGB and NIR (within the s2 slice)
_S2_BANDS = list(BENV2_S2_BANDS)
_S2_R = _S2_BANDS.index('B04')
_S2_G = _S2_BANDS.index('B03')
_S2_B = _S2_BANDS.index('B02')


def patch_pearson(a, b):
    """Per-patch Pearson r across feature dim. a,b: [B, N, D] → [B, N]"""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)


def _stretch(arr_hwc):
    """Joint percentile stretch across all channels. arr_hwc: np.float32 [H,W,C] → uint8."""
    lo = np.percentile(arr_hwc, 2)
    hi = np.percentile(arr_hwc, 98)
    if hi > lo:
        out = np.clip((arr_hwc - lo) / (hi - lo), 0, 1)
    else:
        out = np.zeros_like(arr_hwc)
    return (out * 255).astype(np.uint8)


def s2_to_rgb(img_chw, s2_slice):
    """Visualize S2 as true-color RGB (B04/B03/B02) with joint percentile stretch."""
    s2 = img_chw[s2_slice].cpu().numpy().astype(np.float32)
    rgb = np.stack([s2[_S2_R], s2[_S2_G], s2[_S2_B]], axis=-1)  # [H, W, 3]
    return _stretch(rgb)


def s1_to_rgb(img_chw, s1_slice):
    """Visualize S1 (VV, VH) as grayscale average with joint percentile stretch."""
    s1 = img_chw[s1_slice].cpu().numpy().astype(np.float32)
    avg = (s1[0] + s1[1]) / 2  # [H, W]
    avg_hwc = np.stack([avg, avg, avg], axis=-1)  # [H, W, 3] grayscale
    return _stretch(avg_hwc)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/delulunet_benv2_0501_0433.pt')
    parser.add_argument('--n_batches', type=int, default=20)
    parser.add_argument('--n_vis', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    os.makedirs('res/hallucination_correlation', exist_ok=True)

    # ── Load model ──────────────────────────────────────────────────────────
    model = EVANClassifier.from_checkpoint(args.checkpoint, args.device)
    model.eval()
    evan = model.evan
    n_storage = evan.n_storage_tokens

    mods = evan.supported_modalities   # e.g. ['s1', 's2'] or ['s2', 's1']
    mod_a, mod_b = mods[0], mods[1]
    print(f"Modalities: A={mod_a}, B={mod_b}")
    print(f"Projector type: {evan.intermediate_projector_type}")

    # ── Load data ────────────────────────────────────────────────────────────
    _, _, _, _, test_loader, task_config = get_benv2_loaders(
        batch_size=args.batch_size,
        num_workers=4,
        starting_modality=mod_a,
        new_modality=mod_b,
    )
    modality_slices = task_config.modality_bands_dict
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

    # For visualization: keep raw images and patch tokens for high-corr samples
    vis_candidates = []  # (mean_corr_a, img_raw, patches_a, hal_patches_a, patches_b, hal_patches_b, corr_map_a, corr_map_b)

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= args.n_batches:
                break

            img_raw = batch['image']  # [B, C_total, H, W] — z-score normalized, percentile-stretched for viz
            x = create_multimodal_batch_geobench(batch, modality_slices, (mod_a, mod_b))
            x_a = x[mod_a].to(args.device)
            x_b = x[mod_b].to(args.device)

            embedded = evan.forward_modality_specific_features({mod_a: x_a, mod_b: x_b})
            seq_a = embedded[mod_a]   # [B, 1+n_storage+N, D]
            seq_b = embedded[mod_b]

            # Hallucinate A from B
            seq_b_norm = F.layer_norm(seq_b, [seq_b.shape[-1]])
            hal_a = evan._project_sequence(seq_b_norm, f'{mod_b}_to_{mod_a}', mod_a)  # [B, 1+N, D]

            # Hallucinate B from A
            seq_a_norm = F.layer_norm(seq_a, [seq_a.shape[-1]])
            hal_b = evan._project_sequence(seq_a_norm, f'{mod_a}_to_{mod_b}', mod_b)  # [B, 1+N, D]

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

            B = pa.shape[0]
            s2_sl = modality_slices['s2']
            for s in range(B):
                # Skip flat/uniform tiles (cloud, ocean) — filter on RGB bands specifically
                s2 = img_raw[s, s2_sl]
                rgb_std = s2[[_S2_R, _S2_G, _S2_B]].std().item()
                if rgb_std < 0.5:
                    continue
                mean_a = corr_hal_a[s].mean().item()
                vis_candidates.append((
                    mean_a,
                    img_raw[s].cpu(),
                    pa[s].cpu(), ha[s].cpu(),
                    pb[s].cpu(), hb[s].cpu(),
                    corr_hal_a[s].cpu(), corr_hal_b[s].cpu(),
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
    heatmap_path = 'res/hallucination_correlation/corr_matrix.pdf'
    plt.savefig(heatmap_path, bbox_inches='tight')
    plt.close()
    print(f'Saved {heatmap_path}')

    def make_pca_rgb(pca, lo, hi, tokens):
        """Project tokens into a pre-fit PCA and normalize to [0,1] with fixed range."""
        proj = pca.transform(tokens.float().numpy())
        return np.clip((proj - lo) / (hi - lo + 1e-8), 0, 1)

    # ── Visualize top-n_vis samples ──────────────────────────────────────────
    vis_candidates.sort(key=lambda x: x[0], reverse=True)
    top = vis_candidates[:args.n_vis]

    s2_slice = modality_slices['s2']
    s1_slice = modality_slices['s1']

    n_patches = top[0][2].shape[0]
    grid_size = int(n_patches ** 0.5)  # 8 for BEN-v2

    # Layout (2 rows × 4 cols):
    #   cols 0-1: S2 / S1 raw images, each spanning both rows (full height)
    #   col 2: real_A tokens (row 0), real_B tokens (row 1)
    #   col 3: hal_A tokens (row 0), hal_B tokens (row 1)
    #   token cols are narrow so two stacked token grids match the height of the input images.
    _TOK = 0.45    # token col width relative to image col; two stacked squares → ~same height as one image
    _col_ratios = [1, 1, _TOK, _TOK]

    for idx, (mean_corr_a, img_raw, pa, ha, pb, hb, corr_a_map, corr_b_map) in enumerate(top):
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

        ax_s2.imshow(s2_to_rgb(img_raw, s2_slice))
        ax_s2.text(0.5, -0.02, 'S2 (RGB)', fontsize=11, ha='center', va='top', transform=ax_s2.transAxes)

        ax_s1.imshow(s1_to_rgb(img_raw, s1_slice))
        ax_s1.text(0.5, -0.02, 'S1 (VV+VH)', fontsize=11, ha='center', va='top', transform=ax_s1.transAxes)

        ax_pa.imshow(rgb_pa, interpolation='nearest')
        ax_pa.text(0.5, -0.04, f'real {mod_a}', fontsize=11, ha='center', va='top', transform=ax_pa.transAxes)

        ax_ha.imshow(rgb_ha, interpolation='nearest')
        ax_ha.text(0.5, -0.04, f'hall {mod_a}', fontsize=11, ha='center', va='top', transform=ax_ha.transAxes)

        ax_pb.imshow(rgb_pb, interpolation='nearest')
        ax_pb.text(0.5, -0.04, f'real {mod_b}', fontsize=11, ha='center', va='top', transform=ax_pb.transAxes)

        ax_hb.imshow(rgb_hb, interpolation='nearest')
        ax_hb.text(0.5, -0.04, f'hall {mod_b}', fontsize=11, ha='center', va='top', transform=ax_hb.transAxes)

        out_path = f'res/hallucination_correlation/sample_{idx:03d}.pdf'
        plt.savefig(out_path, bbox_inches='tight')
        plt.close()
        print(f'Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()


# python -u analyze_hallucination_correlation.py --checkpoint checkpoints/sweep_lr7ygzoh_0501_1505.pt
# python -u analyze_hallucination_correlation.py --checkpoint checkpoints/delulunet_benv2_0501_0635.pt
# python -u analyze_hallucination_correlation.py --checkpoint checkpoints/delulunet_benv2_0501_1943.pt