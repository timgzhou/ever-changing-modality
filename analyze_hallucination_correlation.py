"""
Analyze Pearson correlation between hallucinated and real patch tokens.

For each modality (A, B), compares:
  corr(hal_A, real_A)   -- hallucinated A (from B) vs real A  [should be high]
  corr(real_B, real_A)  -- cross-modal baseline               [should be lower]
  corr(real_A, real_A)  -- sanity                             [should be ~1.0]

Visualizes samples where hallucination is best: raw S2 (PCA→RGB), S1 (2ch composite),
and 8×8 patch token grids (PCA→RGB) for real and hallucinated modalities.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

sys.path.insert(0, os.path.dirname(__file__))
from evan_main import EVANClassifier
from geobench_data_utils import get_benv2_loaders, create_multimodal_batch_geobench


def patch_pearson(a, b):
    """Per-patch Pearson r across feature dim. a,b: [B, N, D] → [B, N]"""
    a = a - a.mean(-1, keepdim=True)
    b = b - b.mean(-1, keepdim=True)
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1) + 1e-8)


def percentile_stretch(arr, lo=2, hi=98):
    """Stretch [C,H,W] raw values to [0,1] per channel using percentile clipping."""
    out = np.empty_like(arr, dtype=np.float32)
    for c in range(arr.shape[0]):
        p_lo, p_hi = np.percentile(arr[c], lo), np.percentile(arr[c], hi)
        out[c] = np.clip((arr[c] - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)
    return out


def s2_to_rgb(img_chw, s2_slice):
    """Visualize S2 via PCA of all bands → RGB, with percentile stretch."""
    s2 = img_chw[s2_slice].float().cpu().numpy()  # [C, H, W]
    C, H, W = s2.shape
    pixels = s2.reshape(C, -1).T                  # [H*W, C]
    pca = PCA(n_components=3).fit(pixels)
    rgb = pca.transform(pixels).reshape(H, W, 3)  # [H, W, 3]
    # percentile stretch each output channel
    out = np.empty_like(rgb)
    for c in range(3):
        p_lo, p_hi = np.percentile(rgb[:, :, c], 2), np.percentile(rgb[:, :, c], 98)
        out[:, :, c] = np.clip((rgb[:, :, c] - p_lo) / (p_hi - p_lo + 1e-8), 0, 1)
    return out


def s1_to_rgb(img_chw, s1_slice):
    """Visualize S1 (VV, VH) as 2-channel composite from raw values."""
    s1 = img_chw[s1_slice].float().cpu().numpy()   # [2, H, W]
    s1_norm = percentile_stretch(s1)               # [2, H, W]
    avg = (s1_norm[0:1] + s1_norm[1:2]) / 2
    rgb = np.concatenate([s1_norm[0:1], s1_norm[1:2], avg], axis=0)
    return rgb.transpose(1, 2, 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/delulunet_benv2_0420_1051.pt')
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
    from geobench_data_utils import IdentityNormalizer
    _, _, _, _, test_loader, task_config = get_benv2_loaders(
        batch_size=args.batch_size,
        num_workers=4,
        starting_modality=mod_a,
        new_modality=mod_b,
    )
    _, _, _, _, test_loader_raw, _ = get_benv2_loaders(
        batch_size=args.batch_size,
        num_workers=4,
        starting_modality=mod_a,
        new_modality=mod_b,
        data_normalizer=IdentityNormalizer,
    )
    modality_slices = task_config.modality_bands_dict

    # ── Accumulate tokens and correlations ──────────────────────────────────
    all_corr_hal_a = []   # corr(hal_A, real_A)
    all_corr_xmod_a = []  # corr(real_B, real_A)
    all_corr_hal_b = []   # corr(hal_B, real_B)
    all_corr_xmod_b = []  # corr(real_A, real_B)
    all_corr_aa = []      # sanity
    all_corr_bb = []      # sanity

    # For visualization: keep raw images and patch tokens for high-corr samples
    vis_candidates = []  # (mean_corr_a, img_raw, patches_a, hal_patches_a, patches_b, hal_patches_b, corr_map_a, corr_map_b)

    with torch.no_grad():
        for i, (batch, batch_raw) in enumerate(zip(test_loader, test_loader_raw)):
            if i >= args.n_batches:
                break

            img_raw = batch_raw['image']  # [B, C_total, H, W] — raw reflectance for viz
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

            B = pa.shape[0]
            for s in range(B):
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

    def pca_patch_rgb(real_patches, hal_patches):
        """Fit PCA on real [N,D], normalize range from real, apply same to hal. Returns (real_rgb, hal_rgb) as [N,3] in [0,1]."""
        p_real = real_patches.float().numpy()
        p_hal  = hal_patches.float().numpy()
        pca = PCA(n_components=3).fit(p_real)
        proj_real = pca.transform(p_real)
        proj_hal  = pca.transform(p_hal)
        lo, hi = proj_real.min(0), proj_real.max(0)
        real_rgb = np.clip((proj_real - lo) / (hi - lo + 1e-8), 0, 1)
        hal_rgb  = np.clip((proj_hal  - lo) / (hi - lo + 1e-8), 0, 1)
        return real_rgb, hal_rgb

    # ── Visualize top-n_vis samples ──────────────────────────────────────────
    vis_candidates.sort(key=lambda x: x[0], reverse=True)
    top = vis_candidates[:args.n_vis]

    s2_slice = modality_slices['s2']
    s1_slice = modality_slices['s1']

    n_patches = top[0][2].shape[0]
    grid_size = int(n_patches ** 0.5)  # 8 for BEN-v2

    for idx, (mean_corr_a, img_raw, pa, ha, pb, hb, corr_a_map, corr_b_map) in enumerate(top):
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        fig.suptitle(
            f'Sample {idx}  corr(hal_{mod_a}, real_{mod_a})={mean_corr_a:.3f}',
            fontsize=12
        )

        real_rgb_a, hal_rgb_a = pca_patch_rgb(pa, ha)
        real_rgb_b, hal_rgb_b = pca_patch_rgb(pb, hb)

        for row, (real_rgb, hal_rgb, corr_map, label) in enumerate([
            (real_rgb_a, hal_rgb_a, corr_a_map, mod_a),
            (real_rgb_b, hal_rgb_b, corr_b_map, mod_b),
        ]):
            ax_s2 = axes[row, 0]
            ax_s1 = axes[row, 1]
            ax_real = axes[row, 2]
            ax_hal  = axes[row, 3]
            ax_corr = axes[row, 4]

            if row == 0:
                ax_s2.imshow(s2_to_rgb(img_raw, s2_slice))
                ax_s2.set_title('S2 (RGB)')
                ax_s2.axis('off')
                ax_s1.imshow(s1_to_rgb(img_raw, s1_slice))
                ax_s1.set_title('S1 (VV/VH)')
                ax_s1.axis('off')
            else:
                ax_s2.axis('off')
                ax_s1.axis('off')

            real_rgb = real_rgb.reshape(grid_size, grid_size, 3)
            hal_rgb  = hal_rgb.reshape(grid_size, grid_size, 3)
            corr_grid = corr_map.numpy().reshape(grid_size, grid_size)

            ax_real.imshow(real_rgb, interpolation='nearest')
            ax_real.set_title(f'real {label} tokens')
            ax_real.axis('off')

            ax_hal.imshow(hal_rgb, interpolation='nearest')
            ax_hal.set_title(f'hal {label} (from {"B" if row==0 else "A"})')
            ax_hal.axis('off')

            im = ax_corr.imshow(corr_grid, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            ax_corr.set_title(f'corr(hal, real) {label}')
            ax_corr.axis('off')
            plt.colorbar(im, ax=ax_corr, fraction=0.046, pad=0.04)

        plt.tight_layout()
        out_path = f'res/hallucination_correlation/sample_{idx:03d}.png'
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'Saved {out_path}')

    print('\nDone.')


if __name__ == '__main__':
    main()
