"""
Linear probe of Panopticon on reBEN (BEN-v2, 19-class multilabel).

Loads Panopticon ViT-B/14 (frozen), extracts CLS features, trains a linear
BCE head on the BEN-v2 train split, and evaluates mAP on val and test.

Supports three modality modes:
  s2   — 12-band S2 (chn_ids: optical wavelengths in nm)
  s1   — 2-band S1 VV/VH (chn_ids: -1, -2)
  s2s1 — 12+2 bands stacked, chn_ids concatenated

Usage:
  python lp_panopticon_benv2.py --modality s2
  python lp_panopticon_benv2.py --modality s1
  python lp_panopticon_benv2.py --modality s2s1 --epochs 20 --lr 1e-3
"""

import argparse
import csv
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Add Panopticon hub to path so its internal imports resolve
PANOPTICON_HUB = os.path.expanduser('~/.cache/torch/hub/Panopticon-FM_panopticon_main')
if PANOPTICON_HUB not in sys.path:
    sys.path.insert(0, PANOPTICON_HUB)

# S2 and S1 channel IDs for Panopticon (from ben-s2.yaml / sentinel2.yaml + sentinel1.yaml)
# S2: Sentinel-2 band centre wavelengths in nm (12 bands: B01..B12)
S2_CHN_IDS = torch.tensor([442, 492, 559, 664, 704, 740, 782, 827, 864, 945, 1613, 2203],
                           dtype=torch.int16)
# S1: VV=-1, VH=-2 (Panopticon SAR convention)
S1_CHN_IDS = torch.tensor([-1, -2], dtype=torch.int16)
S2S1_CHN_IDS = torch.cat([S2_CHN_IDS, S1_CHN_IDS])  # 14 channels


def build_chn_ids(modality: str, batch_size: int, device) -> torch.Tensor:
    """Return chn_ids tensor of shape [B, C] for given modality."""
    base = {'s2': S2_CHN_IDS, 's1': S1_CHN_IDS, 's2s1': S2S1_CHN_IDS}[modality]
    return base.unsqueeze(0).expand(batch_size, -1).to(device)


def random_flip_crop(imgs: torch.Tensor, scale: tuple = (0.8, 1.0)) -> torch.Tensor:
    """Random horizontal flip, vertical flip, and resized crop. Applied per batch."""
    B, C, H, W = imgs.shape
    # flips
    if torch.rand(1) > 0.5:
        imgs = imgs.flip(dims=(-1,))
    if torch.rand(1) > 0.5:
        imgs = imgs.flip(dims=(-2,))
    # random resized crop
    ratio = scale[0] + torch.rand(1).item() * (scale[1] - scale[0])
    ch, cw = int(H * ratio), int(W * ratio)
    top  = torch.randint(0, H - ch + 1, (1,)).item()
    left = torch.randint(0, W - cw + 1, (1,)).item()
    imgs = imgs[:, :, top:top+ch, left:left+cw]
    imgs = F.interpolate(imgs, size=(H, W), mode='bilinear', align_corners=False)
    return imgs


@torch.no_grad()
def extract_features(model, loader, modality, modality_slices, device,
                     cache_path: str | None = None, augment: bool = False):
    """
    Run Panopticon in frozen mode over loader, return (features, labels).

    features: [N, 768]  (L2-normalised CLS token from Panopticon)
    labels:   [N, 19]   (float binary multi-label)

    If cache_path is given and the file exists, features are loaded from disk
    instead of recomputed. If it doesn't exist, features are computed and saved.
    If augment=True, random flip+crop is applied to each batch before encoding.
    """
    if cache_path is not None and os.path.isfile(cache_path):
        print(f'    loading cached features from {cache_path}')
        saved = torch.load(cache_path, map_location='cpu')
        return saved['feats'], saved['labels']

    all_feats = []
    all_labels = []
    model.eval()

    s2_slice = modality_slices.get('s2')
    s1_slice = modality_slices.get('s1')

    for batch in tqdm(loader, desc=f'Extracting [{modality}]', leave=False):
        imgs_full = batch['image'].to(device)   # [B, C_total, H, W]
        labels    = batch['label'].float()       # [B, 19]

        if modality == 's2':
            imgs = imgs_full[:, s2_slice, :, :]
            chn_ids = build_chn_ids('s2', imgs.shape[0], device)
        elif modality == 's1':
            imgs = imgs_full[:, s1_slice, :, :]
            chn_ids = build_chn_ids('s1', imgs.shape[0], device)
        else:  # s2s1
            imgs = imgs_full
            chn_ids = build_chn_ids('s2s1', imgs.shape[0], device)

        if augment:
            imgs = random_flip_crop(imgs)

        x_dict = dict(imgs=imgs, chn_ids=chn_ids)
        cls = model(x_dict)   # [B, 768], already L2-normalised

        all_feats.append(cls.cpu())
        all_labels.append(labels)

    feats  = torch.cat(all_feats)
    labels = torch.cat(all_labels)

    if cache_path is not None:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save({'feats': feats, 'labels': labels}, cache_path)
        print(f'    saved features to {cache_path}')

    return feats, labels


def compute_map(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute mean Average Precision (mAP) for multilabel classification."""
    from sklearn.metrics import average_precision_score
    import numpy as np
    scores = torch.sigmoid(logits).numpy()
    targets = labels.numpy()
    aps = []
    for c in range(targets.shape[1]):
        if targets[:, c].sum() > 0:
            aps.append(average_precision_score(targets[:, c], scores[:, c]))
    return float(np.mean(aps)) * 100.0 if aps else 0.0


def train_linear_probe(train_feats, train_labels, val_feats, val_labels,
                       num_classes, epochs, lr, weight_decay, batch_size, device,
                       use_bn: bool = True):
    """Train a linear BCE head on pre-extracted features. Returns best-val head."""
    feat_dim = train_feats.shape[1]
    if use_bn:
        linear = nn.Linear(feat_dim, num_classes)
        nn.init.normal_(linear.weight, std=0.01)
        nn.init.zeros_(linear.bias)
        head = nn.Sequential(nn.BatchNorm1d(feat_dim, affine=False), linear).to(device)
    else:
        head = nn.Linear(feat_dim, num_classes).to(device)
        nn.init.normal_(head.weight, std=0.01)
        nn.init.zeros_(head.bias)

    criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=0)

    train_ds = TensorDataset(train_feats, train_labels)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs * len(train_loader), eta_min=0)

    best_val_map = 0.0
    best_state = None

    for epoch in range(epochs):
        head.train()
        for feats, lbls in train_loader:
            feats, lbls = feats.to(device), lbls.to(device)
            optimizer.zero_grad()
            loss = criterion(head(feats), lbls)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # Validation
        head.eval()
        with torch.no_grad():
            val_logits = head(val_feats.to(device)).cpu()
        val_map = compute_map(val_logits, val_labels)
        if val_map > best_val_map:
            best_val_map = val_map
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
        print(f'  Epoch {epoch+1}/{epochs}  val mAP={val_map:.2f}%  (best={best_val_map:.2f}%)')

    head.load_state_dict(best_state)
    return head, best_val_map


def main():
    parser = argparse.ArgumentParser(description='LP Panopticon on reBEN')
    parser.add_argument('--modality', type=str, default='s2', choices=['s2', 's1', 's2s1'])
    parser.add_argument('--epochs',   type=int,   default=50)
    parser.add_argument('--lr',       type=float, default=1e-2)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--batch_size',   type=int,   default=256)
    parser.add_argument('--num_workers',  type=int,   default=4)
    parser.add_argument('--results_csv',  type=str,   default='res/baselines/benv2_panopticon.csv')
    parser.add_argument('--train_stage0', action='store_true',
                        help='LP on train1 split only (matches train_stage0.py convention). '
                             'Saves head checkpoint and logs to res/train_stage0_benv2.csv.')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--aug_views', type=int, default=1,
                        help='Number of augmented views to cache for train set. '
                             '1 = no augmentation (view0). Each extra view adds a '
                             'random flip+crop pass and is cached separately.')
    parser.add_argument('--no_bn', action='store_true',
                        help='Disable BatchNorm1d before the linear head.')
    parser.add_argument('--feature_cache_dir', type=str, default='cache/panopticon_benv2',
                        help='Directory to cache extracted features. Set to empty string to disable.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}  |  modality: {args.modality}')

    # ------------------------------------------------------------------ data
    print('\n=== Loading BEN-v2 loaders ===')
    from geobench_data_utils import get_benv2_loaders
    train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config = \
        get_benv2_loaders(batch_size=args.batch_size, num_workers=args.num_workers)

    if args.train_stage0:
        # stage0: train on train1 only (matches train_stage0.py convention)
        extraction_loader = DataLoader(train1_loader.dataset, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers, timeout=120)
    else:
        # incremental LP: merge train1+train2 for full training set
        from torch.utils.data import ConcatDataset
        full_train_ds = ConcatDataset([train1_loader.dataset, train2_loader.dataset])
        extraction_loader = DataLoader(full_train_ds, batch_size=args.batch_size,
                                       shuffle=False, num_workers=args.num_workers, timeout=120)
    # Use val1 for selection, val2 for reporting, test for final
    # (keeping same split semantics as stage0 but both halves available)
    modality_slices = task_config.modality_bands_dict  # {'s2': slice(0,12), 's1': slice(12,14)}
    num_classes = task_config.num_classes  # 19

    all_modalities = ['s2', 's1', 's2s1']
    cache_dir = args.feature_cache_dir or None

    # ------------------------------------------------------------------ model (skip if all cached)
    # train has aug_views cache files (view0 = no aug, view1+ = augmented)
    # val/test always have a single cache file (no augmentation)
    train_split_prefix = 'train1' if args.train_stage0 else 'train'
    train_cache_names = [f'{train_split_prefix}_view{v}' for v in range(args.aug_views)]
    all_cached = cache_dir and all(
        os.path.isfile(os.path.join(cache_dir, f'{split}_{mod}.pt'))
        for mod in all_modalities
        for split in ([*train_cache_names, 'val', 'test'])
    )
    if all_cached:
        print('\n=== All features cached — skipping model load ===')
        model = None
    else:
        print('\n=== Loading Panopticon ViT-B/14 ===')
        model = torch.hub.load('Panopticon-FM/panopticon', 'panopticon_vitb14',
                               trust_repo=True)
        model = model.to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        total_params = sum(p.numel() for p in model.parameters())
        print(f'Panopticon params: {total_params:,} (all frozen)')

    # ------------------------------------------------------------------ extract (all modalities)
    # Extract features for every modality so we can eval the probe on all three,
    # regardless of which modality was used for training.
    print('\n=== Extracting features (all modalities) ===')

    train_feats_all, val1_feats_all, test_feats_all = {}, {}, {}
    for mod in all_modalities:
        print(f'  [{mod}]')
        def _cache(split, _mod=mod):
            return os.path.join(cache_dir, f'{split}_{_mod}.pt') if cache_dir else None

        # Extract one unaugmented view + (aug_views-1) augmented views, concatenate
        view_feats, view_labels = [], []
        for v in range(args.aug_views):
            augment = (v > 0)  # view0 is always clean; subsequent views are augmented
            cache_name = f'{train_split_prefix}_view{v}'
            f, l = extract_features(
                model, extraction_loader, mod, modality_slices, device,
                _cache(cache_name), augment=augment)
            view_feats.append(f)
            view_labels.append(l)
        train_feats_all[mod] = torch.cat(view_feats)
        train_labels = torch.cat(view_labels)

        val1_feats_all[mod], val1_labels = extract_features(
            model, val1_loader, mod, modality_slices, device, _cache('val'))
        test_feats_all[mod], test_labels = extract_features(
            model, test_loader, mod, modality_slices, device, _cache('test'))
    print(f'  feature dim: {train_feats_all["s2"].shape[1]}  '
          f'train={train_feats_all["s2"].shape[0]} ({args.aug_views} view(s))  '
          f'val={val1_feats_all["s2"].shape[0]}  '
          f'test={test_feats_all["s2"].shape[0]}')

    # ------------------------------------------------------------------ stage0 branch
    if args.train_stage0:
        print(f'\n=== Stage-0 LP on train1 [{args.modality}] ===')
        head, _ = train_linear_probe(
            train_feats_all[args.modality], train_labels,
            val1_feats_all[args.modality], val1_labels,
            num_classes=num_classes,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            device=device,
            use_bn=not args.no_bn,
        )
        head.eval()
        test_maps = {}
        for eval_mod in all_modalities:
            with torch.no_grad():
                test_logits = head(test_feats_all[eval_mod].to(device)).cpu()
            test_maps[eval_mod] = compute_map(test_logits, test_labels)
            marker = ' <-- trained' if eval_mod == args.modality else ''
            print(f'  eval [{eval_mod}]  test={test_maps[eval_mod]:.2f}%{marker}')

        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir,
                                 f'panopticon_lp_benv2_{args.modality}_s0.pt')
        torch.save(head.state_dict(), ckpt_path)
        print(f'  Saved head to {ckpt_path}')

        csv_path = 'res/train_stage0_benv2.csv'
        s0_fieldnames = ['dataset', 'model_type', 'modality', 'optimizer', 'aug_views',
                         'epochs', 'lr', 'metric_name',
                         'test_s2', 'test_s1', 'test_s2s1', 'saved_checkpoint']
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        file_exists = os.path.isfile(csv_path)
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=s0_fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'dataset': 'benv2',
                'model_type': 'panopticon_base',
                'modality': args.modality,
                'optimizer': 'sgd',
                'aug_views': args.aug_views,
                'epochs': args.epochs,
                'lr': args.lr,
                'metric_name': 'mAP',
                'test_s2':   f'{test_maps["s2"]:.2f}',
                'test_s1':   f'{test_maps["s1"]:.2f}',
                'test_s2s1': f'{test_maps["s2s1"]:.2f}',
                'saved_checkpoint': ckpt_path,
            })
        print(f'Logged to {csv_path}')
        return

    # ------------------------------------------------------------------ incremental LP loop
    os.makedirs(os.path.dirname(args.results_csv), exist_ok=True)
    fieldnames = ['model', 'dataset', 'train_modality', 'optimizer', 'aug_views',
                  'epochs', 'lr', 'metric_name', 'train_fraction',
                  'test_s2', 'test_s1', 'test_s2s1']
    file_exists = os.path.isfile(args.results_csv)

    fractions = [i / 20 for i in range(1, 21)]  # 5%, 10%, ..., 100%
    n_train = train_feats_all[args.modality].shape[0]

    with open(args.results_csv, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for frac in fractions:
            n = max(1, int(n_train * frac))
            # Fixed subsample: same indices for reproducibility across fractions
            idx = torch.arange(n)

            print(f'\n=== Training probe on [{args.modality}]  fraction={frac:.0%}  n={n} ===')
            head, _ = train_linear_probe(
                train_feats_all[args.modality][idx], train_labels[idx],
                val1_feats_all[args.modality], val1_labels,
                num_classes=num_classes,
                epochs=args.epochs,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                device=device,
                use_bn=not args.no_bn,
            )

            head.eval()
            test_maps = {}
            for eval_mod in all_modalities:
                with torch.no_grad():
                    test_logits = head(test_feats_all[eval_mod].to(device)).cpu()
                test_maps[eval_mod] = compute_map(test_logits, test_labels)
                marker = ' <-- trained' if eval_mod == args.modality else ''
                print(f'  eval [{eval_mod}]  test={test_maps[eval_mod]:.2f}%{marker}')

            writer.writerow({
                'model': 'panopticon_vitb14',
                'dataset': 'benv2',
                'train_modality': args.modality,
                'optimizer': 'sgd',
                'aug_views': args.aug_views,
                'epochs': args.epochs,
                'lr': args.lr,
                'metric_name': 'mAP',
                'train_fraction': frac,
                'test_s2':   f'{test_maps["s2"]:.2f}',
                'test_s1':   f'{test_maps["s1"]:.2f}',
                'test_s2s1': f'{test_maps["s2s1"]:.2f}',
            })
            f.flush()

    print(f'Logged to {args.results_csv}')


class PanopticonTeacher(nn.Module):
    """Frozen Panopticon ViT-B/14 + trained LP head, implementing the teacher interface for shot.py.

    Exposes:
      - __call__(input_dict)                   → logits [B, num_classes]   (distill loss)
      - .evan.forward_features(input_dict)     → {mod: {x_norm_clstoken, x_norm_patchtokens}}
      - .evan.starting_modality                (string)
      - .freeze_all()                          (freezes all parameters)

    `input_dict` format (same as what shot.py passes): {mod_name: tensor [B, C, H, W]}
    where the tensor is GEO-Bench z-score normalised (consistent with LP training).

    Note on shapes (for latent loss):
      Panopticon patch_size=14, so for 128×128 input: N_patches = floor(128/14)^2 = 81.
      EVAN student has patch_size=16, N_patches = 64. The latent projectors handle
      this mismatch when train_shot is called with teacher_latent_dim=768, teacher_n_patches=81.
    """

    class _EvanStub:
        """Minimal stub satisfying the .evan interface used by shot.py."""

        def __init__(self, starting_modality: str, backbone: nn.Module, chn_ids: torch.Tensor):
            self.starting_modality = starting_modality
            self._backbone = backbone
            self._chn_ids = chn_ids  # [C] int16

        @torch.no_grad()
        def forward_features(self, input_dict: dict) -> dict:
            """Return {mod: {'x_norm_clstoken': [B,768], 'x_norm_patchtokens': [B,N,768]}}."""
            imgs = next(iter(input_dict.values()))  # [B, C, H, W]
            B = imgs.shape[0]
            chn_ids = self._chn_ids.unsqueeze(0).expand(B, -1).to(imgs.device)
            out = self._backbone.forward_features(dict(imgs=imgs, chn_ids=chn_ids))
            mod = next(iter(input_dict.keys()))
            return {mod: {
                'x_norm_clstoken':    out['x_norm_clstoken'],     # [B, 768]
                'x_norm_patchtokens': out['x_norm_patchtokens'],   # [B, N, 768]
            }}

    def __init__(self, backbone: nn.Module, head: nn.Module,
                 modality: str, chn_ids: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.modality = modality
        self._chn_ids = chn_ids                                    # [C] int16
        self.evan = self._EvanStub(modality, backbone, chn_ids)

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, input_dict: dict) -> torch.Tensor:
        """Return logits [B, num_classes]."""
        imgs = next(iter(input_dict.values()))   # [B, C, H, W]
        B = imgs.shape[0]
        chn_ids = self._chn_ids.unsqueeze(0).expand(B, -1).to(imgs.device)
        with torch.no_grad():
            cls = self.backbone(dict(imgs=imgs, chn_ids=chn_ids))  # [B, 768], L2-norm
        return self.head(cls)                                       # [B, num_classes]

    @classmethod
    def from_checkpoint(cls, head_state_path: str, modality: str,
                        num_classes: int, device) -> "PanopticonTeacher":
        """Load from a stage0 LP checkpoint (head.state_dict only).

        Args:
            head_state_path: path to `head.state_dict()` saved by --train_stage0
                             (nn.Sequential(BatchNorm1d(768, affine=False), Linear(768, num_classes)))
            modality: 's2', 's1', or 's2s1'
            num_classes: output classes (19 for BEN-v2)
            device: torch device
        """
        hub = os.path.expanduser('~/.cache/torch/hub/Panopticon-FM_panopticon_main')
        if hub not in sys.path:
            sys.path.insert(0, hub)

        backbone = torch.hub.load('Panopticon-FM/panopticon', 'panopticon_vitb14', trust_repo=True)
        backbone = backbone.to(device).eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        head = nn.Sequential(nn.BatchNorm1d(768, affine=False), nn.Linear(768, num_classes))
        head.load_state_dict(torch.load(head_state_path, map_location=device))
        head = head.to(device).eval()

        chn_ids = {'s2': S2_CHN_IDS, 's1': S1_CHN_IDS, 's2s1': S2S1_CHN_IDS}[modality]
        return cls(backbone, head, modality, chn_ids).to(device)


if __name__ == '__main__':
    main()
