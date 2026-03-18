"""
Sanity-check script: fine-tune the torchgeo ViT-Small (S2-DINO) directly on PASTIS
with a simple segmentation head. This bypasses EVAN entirely and serves as a baseline
to verify that the PASTIS dataloader + normalizer + mIoU metric are all correct before
trusting EvanSegmenter results.

Example:
    python -u finetune_geotorch_model.py --train_mode fft --epochs 10 --checkpoint_name pastis_geotorch_fft
    python -u finetune_geotorch_model.py --train_mode probe --epochs 20 --checkpoint_name pastis_geotorch_probe
"""

import os
import csv
import argparse
import logging
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from geobench_data_utils import get_pastis_loaders, PASTIS_S2_BANDS, make_div10000_normalizer
from evan_main import PASTIS_BAND_INDICES, UNetDecoder
from train_utils import compute_miou

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

NUM_CLASSES   = 19   # PASTIS semantic classes
IGNORE_INDEX  = 19   # void label
PATCH_SIZE    = 16
EMBED_DIM     = 384


class TorchGeoSegmenter(nn.Module):
    """
    Wraps a torchgeo/timm VisionTransformer and adds a minimal segmentation head.

    The ViT outputs patch tokens of shape [B, N, D] (N = (H/P)*(W/P)).
    We reshape them to a 2-D feature map and upsample back to the input resolution
    with a small Conv head — same approach as linear probing on patch tokens.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, img_size: int = 128, patch_size: int = 16,
                 decoder: str = 'linear'):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size   # e.g. 8 for 128/16
        self.decoder_type = decoder

        if decoder == 'unet':
            self.head = UNetDecoder(EMBED_DIM, num_classes, patch_hw=self.grid)
        else:
            # Simple head: 1x1 conv on patch features → bilinear upsample
            self.head = nn.Conv2d(EMBED_DIM, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            logits: [B, num_classes, H, W]
        """
        features = self.backbone.forward_features(x)   # [B, 1+N, D]
        patch_tokens = features[:, 1:, :]              # [B, N, D]

        if self.decoder_type == 'unet':
            return self.head(patch_tokens)   # [B, C, H, W]

        B, N, D = patch_tokens.shape
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, D, self.grid, self.grid)  # [B, D, g, g]
        logits_small = self.head(feat_map)   # [B, C, g, g]
        return F.interpolate(logits_small, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)


def evaluate(model, loader, criterion, device, ignore_index, num_classes):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x = batch['image'][:, :len(PASTIS_S2_BANDS)].to(device)   # S2 channels only
            labels = batch['mask'].to(device)
            outputs = model(x)
            total_loss += criterion(outputs, labels).item()
            all_preds.append(outputs.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    miou = compute_miou(torch.cat(all_preds), torch.cat(all_labels), num_classes, ignore_index)
    return total_loss / len(loader), miou


def main():
    parser = argparse.ArgumentParser(description='Finetune torchgeo ViT-Small on PASTIS (sanity check)')
    parser.add_argument('--train_mode', type=str, default='fft', choices=['probe', 'fft'],
                        help='probe: head only; fft: full backbone + head')
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--checkpoint_name', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='geotorch_model_finetune_runs')
    parser.add_argument('--val_per_epoch', type=int, default=4,
                        help='evaluate every val_per_epoch epochs')
    parser.add_argument('--normalization', type=str, default='div10000', choices=['zscore', 'div10000'],
                        help='div10000: divide by 10000 (matches S2-DINO pretraining); zscore: per-band z-score')
    parser.add_argument('--decoder', type=str, default='linear', choices=['linear', 'unet'],
                        help='linear: 1x1 conv + bilinear upsample; unet: UNetDecoder from evan_main')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"train_mode={args.train_mode}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}")

    normalizer = make_div10000_normalizer() if args.normalization == 'div10000' else None
    print(f"Normalization: {args.normalization}")

    print("\n=== Creating PASTIS dataloaders ===")
    train1_loader, val1_loader, _, _, test_loader, task_config = get_pastis_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        starting_modality='s2',
        data_normalizer=normalizer,
    )
    print(f"  img_size={task_config.img_size}, num_classes={task_config.num_classes}, ignore_index={task_config.ignore_index}")

    # Model
    print("\n=== Loading torchgeo ViT-Small (S2-DINO) ===")
    import timm
    from torchgeo.models import ViTSmall16_Weights
    weights = ViTSmall16_Weights.SENTINEL2_ALL_DINO
    sd = weights.get_state_dict(progress=True)
    # Create with img_size matching the dataset so timm won't assert on input resolution.
    # timm interpolates pos_embed from 224 → 128 automatically when strict=False.
    backbone = timm.create_model(
        'vit_small_patch16_224', img_size=task_config.img_size,
        in_chans=13, pretrained=False,
    )
    # Interpolate pos_embed from 224 (196 patches + 1 cls = 197) → 128 (64 patches + 1 cls = 65)
    pe = sd['pos_embed']                      # [1, 197, 384]
    cls_pe, patch_pe = pe[:, :1], pe[:, 1:]  # [1, 1, 384], [1, 196, 384]
    patch_pe = patch_pe.reshape(1, 14, 14, 384).permute(0, 3, 1, 2)  # [1, 384, 14, 14]
    grid = task_config.img_size // PATCH_SIZE
    patch_pe = F.interpolate(patch_pe, size=(grid, grid), mode='bicubic', align_corners=False)
    patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, grid * grid, 384)
    sd['pos_embed'] = torch.cat([cls_pe, patch_pe], dim=1)  # [1, 65, 384]
    backbone.load_state_dict(sd, strict=False)

    # Adapt patch embedder: slice to 10 S2 bands (same as PASTIS_BAND_INDICES)
    with torch.no_grad():
        old_proj = backbone.patch_embed.proj
        new_proj = nn.Conv2d(
            len(PASTIS_BAND_INDICES), EMBED_DIM, kernel_size=PATCH_SIZE, stride=PATCH_SIZE,
            bias=(old_proj.bias is not None),
        )
        new_proj.weight = nn.Parameter(old_proj.weight[:, PASTIS_BAND_INDICES, :, :].clone())
        if old_proj.bias is not None:
            new_proj.bias = nn.Parameter(old_proj.bias.clone())
        backbone.patch_embed.proj = new_proj
        print(f"  Sliced patch embedder: 13 → {len(PASTIS_BAND_INDICES)} bands")

    model = TorchGeoSegmenter(
        backbone, num_classes=NUM_CLASSES, img_size=task_config.img_size, patch_size=PATCH_SIZE,
        decoder=args.decoder,
    ).to(device)

    # Freeze / unfreeze
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    if args.train_mode == 'fft':
        for p in model.backbone.parameters():
            p.requires_grad = True
        print("Mode=fft: training full backbone + head.")
    else:
        print("Mode=probe: training head only.")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )

    if args.wandb_project:
        import wandb
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=f"geotorch_pastis_{args.train_mode}_{args.decoder}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    if args.checkpoint_name:
        checkpoint_path = os.path.join(args.checkpoint_dir, f'{args.checkpoint_name}.pt')
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(args.checkpoint_dir, f'geotorch_pastis_{args.train_mode}_{ts}.pt')

    best_val_miou = 0.0
    best_test_miou = 0.0
    best_epoch = 0

    val_interval = max(1, args.val_per_epoch)
    print(f"\n=== Training for {args.epochs} epochs (eval every {val_interval} epochs) ===")
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_preds, train_labels_list = [], []

        pbar = tqdm(train1_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            x      = batch['image'][:, :len(PASTIS_S2_BANDS)].to(device)
            labels = batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0
            )
            optimizer.step()

            train_loss += loss.item()
            train_preds.append(outputs.detach().argmax(dim=1).cpu())
            train_labels_list.append(labels.detach().cpu())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        train_loss /= len(train1_loader)
        train_miou = compute_miou(
            torch.cat(train_preds), torch.cat(train_labels_list), NUM_CLASSES, IGNORE_INDEX
        )

        print(f"  Epoch {epoch+1}: train_loss={train_loss:.4f} train_mIoU={train_miou:.2f}%", end="")

        if (epoch + 1) % val_interval == 0 or epoch == args.epochs - 1:
            val_loss, val_miou = evaluate(model, val1_loader, criterion, device, IGNORE_INDEX, NUM_CLASSES)
            test_loss, test_miou = evaluate(model, test_loader, criterion, device, IGNORE_INDEX, NUM_CLASSES)
            print(f"  val_mIoU={val_miou:.2f}%  test_mIoU={test_miou:.2f}%")

            if val_miou > best_val_miou:
                best_val_miou = val_miou
                best_test_miou = test_miou
                best_epoch = epoch + 1
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch+1,
                            'val_miou': val_miou, 'test_miou': test_miou}, checkpoint_path)
                print(f"    Saved best checkpoint → {checkpoint_path}")

            if args.wandb_project:
                import wandb
                wandb.log({'train_loss': train_loss, 'train_miou': train_miou,
                           'val_miou': val_miou, 'test_miou': test_miou}, step=epoch+1)
        else:
            print()
            if args.wandb_project:
                import wandb
                wandb.log({'train_loss': train_loss, 'train_miou': train_miou}, step=epoch+1)

    print(f"\n=== Done ===")
    print(f"  Best val mIoU: {best_val_miou:.2f}% at epoch {best_epoch}")
    print(f"  Corresponding test mIoU: {best_test_miou:.2f}%")

    # CSV log
    filename = 'res/finetune_geotorch_pastis.csv'
    file_exists = os.path.isfile(filename)
    fieldnames = ['train_mode', 'epochs', 'lr', 'batch_size', 'trainable_params',
                  'best_val_miou', 'best_test_miou', 'best_epoch', 'checkpoint']
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(fieldnames)
        writer.writerow([args.train_mode, args.epochs, args.lr, args.batch_size,
                         trainable, f'{best_val_miou:.2f}', f'{best_test_miou:.2f}',
                         best_epoch, checkpoint_path])

    if args.wandb_project:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()

"""
python -u finetune_geotorch_model.py --normalization div10000 --epochs 4
python -u finetune_geotorch_model.py --normalization zscore
"""