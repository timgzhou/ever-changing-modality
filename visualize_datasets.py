"""
Visualize first 3 samples from BEN-v2 and PASTIS datasets.
Outputs: vis_benv2.png, vis_pastis.png
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from geobench_data_utils import (
    get_benv2_loaders, get_pastis_loaders,
    BENV2_S2_BANDS,
    PASTIS_S2_BANDS,
)
# geobench_data_utils already adds GEO-Bench-2 to sys.path
from geobench_v2.datasets.benv2 import GeoBenchBENV2
from geobench_v2.datasets.pastis import GeoBenchPASTIS


def to_uint8(t):
    a = t.numpy().astype(np.float32)
    lo, hi = np.percentile(a, 2), np.percentile(a, 98)
    if hi > lo:
        a = np.clip((a - lo) / (hi - lo), 0, 1)
    else:
        a = np.zeros_like(a)
    return (a * 255).astype(np.uint8)


def show_rgb(ax, img_chw, r_idx, g_idx, b_idx, title):
    rgb = np.stack([to_uint8(img_chw[i]) for i in (r_idx, g_idx, b_idx)], axis=-1)
    ax.imshow(rgb)
    ax.set_title(title, fontsize=7)
    ax.axis('off')


def show_single(ax, img_chw, c_idx, title, cmap='viridis'):
    ax.imshow(to_uint8(img_chw[c_idx]), cmap=cmap)
    ax.set_title(title, fontsize=7)
    ax.axis('off')


# ─── BEN-v2 ────────────────────────────────────────────────────────────────

print("Loading BEN-v2...")
t1, v1, t2, v2, te, cfg_ben = get_benv2_loaders(batch_size=3, num_workers=0)
batch = next(iter(t2))          # [3, 14, 120, 120]

label_names = GeoBenchBENV2.label_names
s2_bands = list(BENV2_S2_BANDS)
r, g, b = s2_bands.index('B04'), s2_bands.index('B03'), s2_bands.index('B02')
nir = s2_bands.index('B08')

# 3 samples × 4 panels (RGB, NIR, VV, VH)
fig, axes = plt.subplots(3, 4, figsize=(15, 9))
fig.suptitle('BEN-v2 — first 3 samples', fontsize=12, fontweight='bold')

for i in range(3):
    img = batch['image'][i]
    label = batch['label'][i]
    s2 = img[cfg_ben.modality_slices['s2']]
    s1 = img[cfg_ben.modality_slices['s1']]
    active = [label_names[j] for j in range(19) if label[j].item() == 1]
    print(f"  Sample {i+1} labels: {active}")

    show_rgb(axes[i, 0], s2, r, g, b, 'S2 RGB' if i == 0 else '')
    show_single(axes[i, 1], s2, nir, 'S2 NIR' if i == 0 else '', cmap='YlGn')
    show_single(axes[i, 2], s1, 0, 'S1 VV' if i == 0 else '', cmap='gray')
    show_single(axes[i, 3], s1, 1, 'S1 VH' if i == 0 else '', cmap='gray')

    # Label names as text to the right of VH panel
    label_text = '\n'.join(active) if active else '(none)'
    axes[i, 3].text(
        1.04, 0.5, label_text,
        transform=axes[i, 3].transAxes,
        fontsize=6.5, va='center', ha='left',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', edgecolor='gray', alpha=0.8)
    )

plt.tight_layout()
fig.savefig('vis_benv2.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved vis_benv2.png")


# ─── PASTIS ────────────────────────────────────────────────────────────────

print("Loading PASTIS...")
t1p, v1p, t2p, v2p, tep, cfg_pas = get_pastis_loaders(batch_size=3, num_workers=0)
batchp = next(iter(t2p))        # [3, 16, 128, 128]

classes = list(GeoBenchPASTIS.classes)
n_classes = len(classes)
cmap_seg = plt.cm.get_cmap('tab20', n_classes)

s2_bands_p = list(PASTIS_S2_BANDS)
r_p, g_p, b_p = s2_bands_p.index('B04'), s2_bands_p.index('B03'), s2_bands_p.index('B02')
nir_p = s2_bands_p.index('B08')

# 3 samples × 5 panels (RGB, NIR, VV_asc, VV_desc, mask)
fig2, axes2 = plt.subplots(3, 5, figsize=(16, 10))
fig2.suptitle('PASTIS — first 3 samples (temporal mean)', fontsize=12, fontweight='bold')

for i in range(3):
    imgp = batchp['image'][i]
    mask = batchp['mask'][i].numpy()
    s2p = imgp[cfg_pas.modality_slices['s2']]
    s1p = imgp[cfg_pas.modality_slices['s1']]

    present_ids = np.unique(mask)
    present_names = [classes[j] for j in present_ids if j < n_classes]
    print(f"  Sample {i+1} classes: {present_names}")

    show_rgb(axes2[i, 0], s2p, r_p, g_p, b_p, 'S2 RGB' if i == 0 else '')
    show_single(axes2[i, 1], s2p, nir_p, 'S2 NIR' if i == 0 else '', cmap='YlGn')
    show_single(axes2[i, 2], s1p, 0, 'S1 VV_asc' if i == 0 else '', cmap='gray')
    show_single(axes2[i, 3], s1p, 3, 'S1 VV_desc' if i == 0 else '', cmap='gray')

    axes2[i, 4].imshow(mask, cmap=cmap_seg, vmin=0, vmax=n_classes - 1, interpolation='nearest')
    axes2[i, 4].set_title('Mask' if i == 0 else '', fontsize=7)
    axes2[i, 4].axis('off')

    # Legend beside each mask
    patches = [
        mpatches.Patch(color=cmap_seg(j), label=f'{j}: {classes[j]}')
        for j in present_ids if j < n_classes
    ]
    axes2[i, 4].legend(
        handles=patches, loc='upper left',
        bbox_to_anchor=(1.02, 1), borderaxespad=0,
        fontsize=6, framealpha=0.9
    )

plt.tight_layout()
fig2.savefig('vis_pastis.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved vis_pastis.png")
