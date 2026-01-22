"""Training utilities for EVAN on EuroSAT."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from einops import rearrange
from eurosat_data_utils import create_multimodal_batch
import wandb
import numpy as np


class FullSequenceMAEDecoder(nn.Module):
    """
    MAE decoder that takes the full sequence including mask token positions.

    Unlike SimpleMAEDecoder which expects only unmasked tokens + ids_restore,
    this decoder takes the full sequence where masked positions already contain
    the mask token. Simpler interface for fusion MAE training.
    """

    def __init__(self, embed_dim, num_channels, patch_size, decoder_depth=1, decoder_heads=8, ffn_factor=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim

        # Transformer decoder
        from torch.nn import TransformerEncoderLayer, TransformerEncoder
        decoder_layer = TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=decoder_heads,
            dim_feedforward=embed_dim * ffn_factor,
            batch_first=True,
            norm_first=True
        )
        self.decoder = TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Linear projection to reconstruct pixels
        self.pred = nn.Linear(embed_dim, patch_size * patch_size * num_channels)

    def forward(self, x):
        """
        Args:
            x: [B, num_patches, embed_dim] - Full sequence with mask tokens at masked positions

        Returns:
            reconstructed patches: [B, num_patches, patch_size^2 * channels]
        """
        x = self.decoder(x)
        return self.pred(x)


def evaluate(model, dataloader, criterion, device, modality_bands_dict,
             modalities_to_use=('rgb',)):
    """
    Evaluate model on a dataloader.

    Args:
        model: EVAN classifier
        dataloader: DataLoader (RGB only or full bands)
        criterion: Loss function
        device: torch device
        modality_bands_dict: Dict mapping modality names to their band tuples
                            e.g., {'rgb': ('B04', 'B03', 'B02'), 'infrared': ('B08', 'B8A', 'B09', 'B10')}
        modalities_to_use: Tuple of modality names to use for evaluation (e.g., ('rgb',) or ('rgb', 'infrared'))
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch['label'].to(device)

            # Create multi-modal input with specified modalities
            modal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict, modalities=modalities_to_use
            )
            modal_input = {k: v.to(device) for k, v in modal_input.items()}
            outputs = model(modal_input)

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ==================== MAE Helper Functions ====================

def mae_reconstruction_loss(pred, target, mask):
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # Mean over pixels in patch
    loss = (loss * mask).sum() / mask.sum()
    return loss

def patchify(imgs, patch_size):
    patches = rearrange(imgs, 'b c (h ph) (w pw) -> b (h w) (ph pw c)', ph=patch_size, pw=patch_size)
    return patches
 
def create_mae_decoders(hidden_dim,patch_size,modality_bands_dict,mae_modalities,device):
    mae_decoders = nn.ModuleDict()
    for mod in mae_modalities:
        num_channels = len(modality_bands_dict[mod])
        mae_decoders[mod] = FullSequenceMAEDecoder(
            embed_dim=hidden_dim,
            num_channels=num_channels,
            patch_size=patch_size,
            decoder_depth=1,
            ffn_factor=2
        ).to(device)
        print(f"  Initialized FullSequenceMAEDecoder for {mod}, num_channels={num_channels}")
    return mae_decoders

def create_latent_projectors(hidden_dim,latent_reconstruct_modalities,device):
    latent_projectors = nn.ModuleDict()
    for mod in latent_reconstruct_modalities:
        latent_projectors[mod] = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        ).to(device)
        print(f"  Initialized Latent Projector for {mod}")
    return latent_projectors

def create_mask_tokens(evan,hidden_dim,all_modalities,device):
    for mod in all_modalities:
        evan.modality_specific_mask_tokens[mod] = nn.Parameter(torch.randn(hidden_dim, device=device),requires_grad=True)
    return evan.modality_specific_mask_tokens

def create_int_cls_projectors(hidden_dim, all_modalities, device, factor=4):
    """Create bidirectional projectors to map CLS tokens between all modality pairs."""
    # So we have 2*(m choose 2) projectors
    cls_projectors = nn.ModuleDict()
    for src_mod in all_modalities:
        for tgt_mod in all_modalities:
            if src_mod != tgt_mod:
                key = f"{src_mod}_to_{tgt_mod}"
                cls_projectors[key] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * factor),
                    nn.GELU(),
                    nn.Linear(hidden_dim * factor, hidden_dim),
                ).to(device)
                print(f"  Initialized CLS Projector: {src_mod} -> {tgt_mod}")
    return cls_projectors

def create_fuse_cls_projectors(hidden_dim, mae_modalities, latent_reconstruct_modalities, device,factor=4):
    """
    These are the supervision provided from latent_reconstruction_modalities to mae_modalities.
    Note we do this to ensure the features extracted from mae_modalities are semantically meaningful.
    """
    fused_cls_projector=nn.ModuleDict()
    for src_mod in mae_modalities:
        for tgt_mod in latent_reconstruct_modalities:
            if src_mod != tgt_mod:
                key = f"{src_mod}_to_{tgt_mod}"
                fused_cls_projector[key] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim*factor),
                    nn.GELU(),
                    nn.Linear(hidden_dim*factor, hidden_dim),
                ).to(device)
                print(f"  Initialized CLS Projector: {src_mod} -> {tgt_mod}")
    return fused_cls_projector
def mask_input(mask_tokens,pre_fusion_cls_projectors,batch_size,n_storage_tokens,num_patches,mae_mask_ratio,all_modalities,prefusion_features,modality_dropout,device):
    len_keep = int(num_patches * (1 - mae_mask_ratio))
    modality_masks = {}  # {mod: [B, num_patches] bool tensor, True=masked}
    modality_dropped = {}
    drop_candidates = [mod for mod in all_modalities if np.random.rand() < modality_dropout]
    if len(drop_candidates) == len(all_modalities):
        # Don't drop all - randomly keep one
        drop_candidates.pop(np.random.randint(len(drop_candidates)))
    available_modalities = [mod for mod in all_modalities if mod not in drop_candidates]
    for mod in all_modalities:
        noise = torch.rand(batch_size, num_patches, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.ones(batch_size, num_patches, device=device, dtype=torch.bool)
        is_dropped = mod in drop_candidates
        modality_dropped[mod] = is_dropped
        len_keep_moddrop = 0 if is_dropped else len_keep
        mask.scatter_(1, ids_shuffle[:, :len_keep_moddrop], False)
        modality_masks[mod] = mask
    masked_mod_features = {}
    for mod, features in prefusion_features.items():
        # features: [B, 1+n_storage+num_patches, embed_dim]
        n_prefix = n_storage_tokens + 1
        cls_token = features[:, 0:1, :]  # [B, 1, embed_dim]
        storage = features[:, 1:n_prefix, :]  # [B, n_storage, embed_dim]
        patches = features[:, n_prefix:, :]  # [B, num_patches, embed_dim]

        # Replace masked positions with mask_token (using this modality's mask in mask_tokens)
        mask_expanded = modality_masks[mod].unsqueeze(-1)  # [B, num_patches, 1]
        mask_token_expanded = mask_tokens[mod].expand(batch_size, num_patches, -1)
        masked_patches = torch.where(mask_expanded, mask_token_expanded, patches)

        # If modality is fully dropped, use projected CLS from available modalities
        # and mask storage tokens
        if modality_dropped[mod]:
            # Compute mean of projected CLS from all available modalities
            projected_cls_list = []
            for avail_mod in available_modalities:
                avail_cls = prefusion_features[avail_mod][:, 0, :].detach()  # [B, embed_dim]
                key = f"{avail_mod}_to_{mod}"
                projected = pre_fusion_cls_projectors[key](avail_cls)  # [B, embed_dim]
                projected_cls_list.append(projected)
            # Mean of projections, detached so gradients don't flow back through projector during MAE/latent loss
            cls_token = torch.stack(projected_cls_list).mean(dim=0).unsqueeze(1).detach()  # [B, 1, embed_dim]
            storage = mask_tokens[mod].unsqueeze(0).unsqueeze(0).expand(batch_size, n_storage_tokens, -1)

        # Reconstruct full sequence with masked patches
        masked_mod_features[mod] = torch.cat([cls_token, storage, masked_patches], dim=1)
    return modality_masks, masked_mod_features
            
def train_shot(
    model, train_loader, device, args,
    mae_modalities: list[str],
    latent_reconstruct_modalities: list[str] = ["rgb"],
    modality_bands_dict: dict = None,
    max_norm=4
):
    """
    End-to-end training with hybrid loss combining:
    - MAE reconstruction loss for mae_modalities (reconstruct raw pixels)
    - Latent reconstruction loss for latent_reconstruct_modalities (match frozen teacher features)
    - Bidirectional CLS projection loss (learn mappings between all modality CLS tokens)
    During modality dropout, uses projected CLS from available modalities instead of mask tokens.
    """
    print("\n" + "="*70)
    print(f"=== END TO END SHOT TRAINING===\n")
    print(f"  MAE modalities (pixel reconstruction): {mae_modalities}")
    print(f"  Latent modalities (feature matching): {latent_reconstruct_modalities}")
    all_modalities = list(set(mae_modalities + latent_reconstruct_modalities))

    model.freeze_all()
    model.set_requires_grad("all", clsreg=True, modality_encoders=True, mfla=True, msla=True, patch_embedders=True) # blocks are true, no need for mfla and msla
    if args.train_components=="full":
        model.set_requires_grad("backbone", mask_token=False, blocks=True, norm=True) #use new mask tokens instead, see below
    
    evan = model.evan
    embed_dim = evan.embed_dim
    patch_size = evan.patch_size
    num_patches = (evan.img_size // patch_size) ** 2

    # Create updatable componenets for training
    mae_decoders=create_mae_decoders(embed_dim,patch_size,modality_bands_dict,mae_modalities,device)
    latent_projectors = create_latent_projectors(embed_dim,latent_reconstruct_modalities,device)
    mask_tokens = create_mask_tokens(evan,embed_dim,all_modalities,device)
    pre_fusion_cls_projectors = create_int_cls_projectors(embed_dim, all_modalities, device)
    post_fusion_cls_projector = create_fuse_cls_projectors(embed_dim, mae_modalities, latent_reconstruct_modalities, device)

    # Create frozen teacher for latent targets
    # TODO check if teacher_evan here is still behaving right when it only takes rgb, it should be.
    teacher_evan = copy.deepcopy(evan)
    teacher_evan.freeze_all()
    teacher_evan.eval()
    trainable_in_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    total_in_evan = sum(p.numel() for p in evan.parameters())
    trainable_decoder = sum(p.numel() for p in mae_decoders.parameters())
    trainable_projector = sum(p.numel() for p in latent_projectors.parameters())
    trainable_cls_proj = sum(p.numel() for p in pre_fusion_cls_projectors.parameters())
    trainable_fused_cls_projector = sum(p.numel() for p in post_fusion_cls_projector.parameters())
    trainable_total = trainable_in_evan + trainable_decoder + trainable_projector + trainable_cls_proj + trainable_fused_cls_projector 
    print(f"\nTrainable parameters: {trainable_total}")
    print(f"    EVAN: {trainable_in_evan} ({100*trainable_in_evan/total_in_evan:.2f}%)")
    print(f"    MAE decoders: {trainable_decoder}")
    print(f"    Latent projectors: {trainable_projector}")
    print(f"    Pre fusion CLS projectors: {trainable_cls_proj}")
    print(f"    Post fusion CLS projectors: {trainable_cls_proj}")

    params = (
        list(filter(lambda p: p.requires_grad, evan.parameters())) +
        list(mae_decoders.parameters()) +
        list(latent_projectors.parameters()) +
        list(pre_fusion_cls_projectors.parameters()) +
        list(post_fusion_cls_projector.parameters())
    )
    num_params = sum(p.numel() for p in params)
    print(f"Total trainable parameters: {num_params:,}")
    assert num_params==trainable_total, f"{num_params=} != {trainable_total=}"

    optimizer = torch.optim.AdamW(params, lr=args.ssl_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1, min_lr=1e-6)

    # Loss functions
    mse_fn = nn.MSELoss()

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        evan.train()
        mae_decoders.train()
        latent_projectors.train()
        pre_fusion_cls_projectors.train()
        post_fusion_cls_projector.train()
        train_loss = 0.0
        train_mae_loss = 0.0
        train_latent_loss = 0.0
        train_pre_fusion_cls_loss = 0.0
        train_post_fusion_cls_loss=0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"SHOT Training Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            full_multimodal_input = create_multimodal_batch(
                batch, modality_bands_dict=modality_bands_dict,
                modalities=tuple(all_modalities)
            )
            full_multimodal_input = {k: v.to(device) for k, v in full_multimodal_input.items()}
            batch_size = next(iter(full_multimodal_input.values())).shape[0]
            prefusion_features = evan.forward_modality_specific_features(full_multimodal_input) # pre_fusion features (independent processing for all)
            # mod_specific looks like {mod: [B, 1+n_storage+num_patches, embed_dim] for mod in all_modalities}

            # Step 1: Get teacher targets (unmasked, frozen)
            with torch.no_grad():
                teacher_input={teacher_mod:full_multimodal_input[teacher_mod] for teacher_mod in latent_reconstruct_modalities}
                teacher_out = teacher_evan.forward_features(teacher_input)
                # looks like {"rgb": {'x_norm_patchtokens': [B, num_patches, embed_dim], ...}}

            # Step 2: Compute pre-fusion CLS projection loss # TODO: try removing detach here
            # Train bidirectional projectors: src_cls -> tgt_cls for all pairs
            # Both src and tgt are detached - we only train the projector, not the pre-fusion weights 
            batch_pre_fusion_cls_loss = 0.0
            prefusion_cls_loss = torch.tensor(0.0, device=device)
            pre_fusion_loss_count = 0
            for src_mod in all_modalities:
                src_cls = prefusion_features[src_mod][:, 0, :].detach()  # [B, embed_dim]
                for tgt_mod in all_modalities:
                    if src_mod != tgt_mod:
                        tgt_cls = prefusion_features[tgt_mod][:, 0, :].detach()  # [B, embed_dim]
                        key = f"{src_mod}_to_{tgt_mod}"
                        projected_cls = pre_fusion_cls_projectors[key](src_cls)  # [B, embed_dim]
                        prefusion_cls_loss = prefusion_cls_loss + mse_fn(projected_cls, tgt_cls)
                        pre_fusion_loss_count += 1
            prefusion_cls_loss = prefusion_cls_loss / pre_fusion_loss_count if pre_fusion_loss_count > 0 else prefusion_cls_loss
            batch_pre_fusion_cls_loss = prefusion_cls_loss.item()
            
            modality_masks, masked_mod_features = mask_input(mask_tokens,pre_fusion_cls_projectors,batch_size,evan.n_storage_tokens,num_patches,args.mae_mask_ratio,all_modalities,prefusion_features,args.modality_dropout,device)
            
            # Step 3: Forward through fusion blocks
            student_fused = evan.forward_fusion_from_modality_features(masked_mod_features)
            # {mod: {'x_norm_patchtokens': [B, num_patches, embed_dim], ...}}

            # Step 4: Compute losses
            total_loss = prefusion_cls_loss  # Start with CLS projection loss computed earlier
            batch_mae_loss = 0.0
            batch_latent_loss = 0.0
            batch_fused_cls_loss=0.0

            # MAE loss: decode to pixels, loss on masked patches only
            for mod in mae_modalities:
                student_patches = student_fused[mod]['x_norm_patchtokens']  # [B, num_patches, embed_dim]
                pred_pixels = mae_decoders[mod](student_patches)  # [B, num_patches, patch_size^2 * C]
                target_img = full_multimodal_input[mod]  # [B, C, H, W]
                target_patches = patchify(target_img, patch_size)  # [B, num_patches, patch_size^2 * C]
                mask_float = modality_masks[mod].float()
                mae_loss = mae_reconstruction_loss(pred_pixels, target_patches, mask_float)
                total_loss = total_loss + mae_loss
                batch_mae_loss += mae_loss.item()

            # Latent loss: project and match teacher, loss on masked patches + CLS token
            for mod in latent_reconstruct_modalities:
                student_patches = student_fused[mod]['x_norm_patchtokens']  # [B, num_patches, embed_dim]
                teacher_patches = teacher_out[mod]['x_norm_patchtokens'].detach()  # [B, num_patches, embed_dim]
                student_cls = student_fused[mod]['x_norm_clstoken']  # [B, embed_dim]
                teacher_cls = teacher_out[mod]['x_norm_clstoken'].detach()  # [B, embed_dim]
                # TODO: here the cls token and the patch token are being projected equivalently. 
                # This is not right, clstoken should have its own projection.
                # Concatenate CLS with patches: treat CLS as an extra "patch" for projection
                student_all = torch.cat([student_cls.unsqueeze(1), student_patches], dim=1)  # [B, 1+num_patches, embed_dim]
                teacher_all = torch.cat([teacher_cls.unsqueeze(1), teacher_patches], dim=1)  # [B, 1+num_patches, embed_dim]

                projected = latent_projectors[mod](student_all)  # [B, 1+num_patches, embed_dim]

                latent_loss = mse_fn(projected, teacher_all)
                total_loss = total_loss + latent_loss
                batch_latent_loss += latent_loss.item()
            # Fused CLStoken loss: clstoken from mae_modalities should be mlp predictable by clstokens from latent_reconstruction_modalities
            for mod_mae in mae_modalities:
                for mod_lat in latent_reconstruct_modalities:
                    mod_mae_cls=student_fused[mod_mae]['x_norm_clstoken']
                    predicted_lat_cls=post_fusion_cls_projector[f"{mod_mae}_to_{mod_lat}"](mod_mae_cls)
                    target_lat_cls=student_fused[mod_lat]['x_norm_clstoken'].detach()
                    fused_cls_loss=mse_fn(predicted_lat_cls, target_lat_cls)
                    total_loss = total_loss + fused_cls_loss
                    batch_fused_cls_loss += fused_cls_loss.item()
            
            # Step 9: Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
            optimizer.step()

            train_loss += total_loss.item()
            train_mae_loss += batch_mae_loss
            train_latent_loss += batch_latent_loss
            train_pre_fusion_cls_loss += batch_pre_fusion_cls_loss
            train_post_fusion_cls_loss += batch_fused_cls_loss
            train_count += 1
            global_step += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'mae': f'{batch_mae_loss:.4f}',
                'latent': f'{batch_latent_loss:.4f}',
                'pre_fusion_cls': f'{batch_pre_fusion_cls_loss:.4f}',
                'post_fusion_cls': f'{batch_fused_cls_loss:.4f}',
                'grad': f'{grad_norm:.4f}'
            })

            # Log to wandb every step
            wandb.log({
                'train_loss': total_loss.item(),
                'mae_loss': batch_mae_loss,
                'latent_loss': batch_latent_loss,
                'pre_fusion_cls': batch_pre_fusion_cls_loss,
                'post_fusion_cls': batch_fused_cls_loss,
                'grad_norm': grad_norm.item(),
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[0]['lr'],
            })

        # Epoch summary
        train_loss /= train_count
        train_mae_loss /= train_count
        train_latent_loss /= train_count
        train_pre_fusion_cls_loss /= train_count
        train_post_fusion_cls_loss /= train_count

        scheduler.step(train_loss)
        print(f"\nFusion MAE Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Total: {train_loss:.4f}, MAE: {train_mae_loss:.4f}, Latent: {train_latent_loss:.4f}, CLS: {train_pre_fusion_cls_loss:.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")

    print("\n=== Phase 2 (Fusion MAE Training) complete ===")
    return mae_decoders, latent_projectors, mask_tokens, pre_fusion_cls_projectors, trainable_total
