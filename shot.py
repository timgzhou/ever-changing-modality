"""Training utilities for EVAN on EuroSAT."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from einops import rearrange
from eurosat_data_utils import create_multimodal_batch
from train_utils import _delulu_stage3_test
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


class SequenceProjector(nn.Module):
    def __init__(self, embed_dim, num_heads=8, ffn_factor=4, num_layers=1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_factor,
            batch_first=True,
            norm_first=True
        )
        self.layers = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        return self.layers(x)



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

def distillation_loss(student_logits, teacher_logits, temperature=2.0):
    """Compute KL divergence loss between student and teacher soft labels."""
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(student_soft, teacher_soft, reduction='batchmean') * (temperature ** 2)

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

# ==================== SHOT TRAINING COMPONENT ====================

def create_latent_projectors(hidden_dim, latent_reconstruct_modalities, device, num_heads=8, ffn_factor=4):
    """Create transformer-based projectors for latent matching (CLS + patches jointly)."""
    projectors = nn.ModuleDict()
    for mod in latent_reconstruct_modalities:
        projectors[mod] = SequenceProjector(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            ffn_factor=ffn_factor,
            num_layers=2
        ).to(device)
        print(f"  Initialized Latent Projector (Transformer) for {mod}")
    return projectors

def create_intermediate_projectors(hidden_dim, all_modalities, device, num_heads=8, ffn_factor=4):
    """Create bidirectional projectors to map full sequences (CLS + storage + patches) between all modality pairs."""
    # So we have 2*(m choose 2) projectors
    projectors = nn.ModuleDict()
    for src_mod in all_modalities:
        for tgt_mod in all_modalities:
            if src_mod != tgt_mod:
                key = f"{src_mod}_to_{tgt_mod}"
                projectors[key] = SequenceProjector(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    ffn_factor=ffn_factor,
                    num_layers=2
                ).to(device)
                print(f"  Initialized Sequence Projector: {src_mod} -> {tgt_mod}")
    return projectors

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

# MASKING HELPER FUNCTION
def mask_input(intermediate_projectors, batch_size, n_storage_tokens, num_patches, mae_mask_ratio, all_modalities, prefusion_features, modality_dropout, device, protected_modalities=None):
    """
    Apply masking using projected sequences from other modalities.

    For partially masked modalities: replace masked token positions with projected tokens.
    For fully dropped modalities: replace entire sequence with mean of projected sequences.

    Args:
        protected_modalities: List of modalities that should never be fully dropped.
                             Used to ensure newmod is always present in unlabeled batches.
    """
    len_keep = int(num_patches * (1 - mae_mask_ratio))
    modality_masks = {}  # {mod: [B, num_patches] bool tensor, True=masked}
    modality_dropped = {}

    # Determine which modalities to drop (excluding protected modalities)
    drop_candidates = [
        mod for mod in all_modalities
        if np.random.rand() < modality_dropout
        and (protected_modalities is None or mod not in protected_modalities)
    ]
    if len(drop_candidates) == len(all_modalities):
        # Don't drop all - randomly keep one
        drop_candidates.pop(np.random.randint(len(drop_candidates)))
    available_modalities = [mod for mod in all_modalities if mod not in drop_candidates]
    has_modality_dropout = len(drop_candidates) > 0

    # Generate masks for each modality
    # Token masking and modality dropout are mutually exclusive to avoid information leakage
    for mod in all_modalities:
        is_dropped = mod in drop_candidates
        modality_dropped[mod] = is_dropped

        if is_dropped:
            # Fully mask dropped modalities
            mask = torch.ones(batch_size, num_patches, device=device, dtype=torch.bool)
        elif has_modality_dropout:
            # No token masking when simulating missing modalities
            mask = torch.zeros(batch_size, num_patches, device=device, dtype=torch.bool)
        else:
            # Normal MAE token masking when all modalities present
            noise = torch.rand(batch_size, num_patches, device=device)
            ids_shuffle = torch.argsort(noise, dim=1)
            mask = torch.ones(batch_size, num_patches, device=device, dtype=torch.bool)
            mask.scatter_(1, ids_shuffle[:, :len_keep], False)

        modality_masks[mod] = mask

    # Compute projected sequences from all available modalities to all target modalities
    # This is done once upfront to avoid redundant computation
    projected_sequences = {}  # {(src_mod, tgt_mod): [B, seq_len, embed_dim]}
    for src_mod in available_modalities:
        src_seq = prefusion_features[src_mod]  # [B, seq_len, embed_dim]
        src_seq_norm = F.layer_norm(src_seq, [src_seq.shape[-1]])
        for tgt_mod in all_modalities:
            if src_mod != tgt_mod:
                key = f"{src_mod}_to_{tgt_mod}"
                projected_sequences[(src_mod, tgt_mod)] = intermediate_projectors[key](src_seq_norm)

    masked_mod_features = {}
    n_prefix = n_storage_tokens + 1

    for mod in all_modalities:
        features = prefusion_features[mod]  # [B, 1+n_storage+num_patches, embed_dim]

        if modality_dropped[mod]:
            # Fully dropped: use mean of all projected sequences from available modalities
            projected_list = [projected_sequences[(avail_mod, mod)] for avail_mod in available_modalities]
            # Mean of projections, detached so gradients don't flow back during MAE/latent loss
            masked_mod_features[mod] = torch.stack(projected_list).mean(dim=0)
        else:
            # Partially masked: replace masked positions with projected tokens
            # Use mean of projections from available modalities (excluding self)
            other_available = [m for m in available_modalities if m != mod]
            if other_available:
                projected_list = [projected_sequences[(avail_mod, mod)] for avail_mod in other_available]
                projected_seq = torch.stack(projected_list).mean(dim=0)  # [B, seq_len, embed_dim]
            else:
                # No other modalities available, use zeros (edge case)
                projected_seq = torch.zeros_like(features)

            # Create full sequence mask: [B, seq_len, 1]
            # CLS and storage tokens are never masked for partial masking
            prefix_mask = torch.zeros(batch_size, n_prefix, device=device, dtype=torch.bool)
            full_mask = torch.cat([prefix_mask, modality_masks[mod]], dim=1)  # [B, seq_len]
            full_mask_expanded = full_mask.unsqueeze(-1)  # [B, seq_len, 1]

            # Replace masked positions with projected tokens
            masked_mod_features[mod] = torch.where(full_mask_expanded, projected_seq, features)

    return modality_masks, masked_mod_features


def mixed_batch_iterator(unlabeled_loader, labeled_loader, labeled_freq):
    """
    Yield (batch, is_labeled) tuples, mixing batches from both loaders.

    Args:
        unlabeled_loader: Primary loader (train2, unlabeled multimodal)
        labeled_loader: Secondary loader (train1, labeled monomodal), can be None
        labeled_freq: Probability of sampling from labeled_loader each iteration

    Yields:
        (batch, is_labeled): Tuple of batch dict and boolean indicating if from labeled loader
    """
    unlabeled_iter = iter(unlabeled_loader)
    labeled_iter = iter(labeled_loader) if labeled_loader else None

    # Total batches = length of unlabeled loader (primary)
    for _ in range(len(unlabeled_loader)):
        use_labeled = labeled_iter is not None and np.random.rand() < labeled_freq

        if use_labeled:
            try:
                batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                batch = next(labeled_iter)
            yield batch, True
        else:
            try:
                batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                batch = next(unlabeled_iter)
            yield batch, False


# SHOT_TRAINING!
def train_shot(
    model, train_loader, device, args,
    mae_modalities: list[str],
    latent_reconstruct_modalities: list[str] = ["rgb"],
    modality_bands_dict: dict = None,
    max_norm=4,
    distillation_temperature: float = 2.0,
    test_loader=None,
    eval_every_n_epochs: int = None,
    labeled_train_loader=None,
    labeled_frequency: float = 0.0,
    active_losses: list[str] = None,
):
    """
    End-to-end training with hybrid loss combining:
    - MAE reconstruction loss for mae_modalities (reconstruct raw pixels)
    - Latent reconstruction loss for latent_reconstruct_modalities (match frozen teacher features)
    - Bidirectional sequence projection loss (learn mappings between modality sequences)
    - Label distillation loss (train classifier heads with soft labels from teacher model)
    During modality dropout, uses projected sequences from available modalities.

    Mixed training mode (when labeled_train_loader is provided):
    - train_loader (train2): unlabeled multimodal data -> full SHOT losses with teacher distillation
    - labeled_train_loader (train1): labeled monomodal data -> CE loss with real labels + latent loss
    - labeled_frequency: probability of sampling from labeled_train_loader each iteration
    """
    print("\n" + "="*70)
    print(f"=== END TO END SHOT TRAINING===\n")
    print(f"  MAE modalities (pixel reconstruction): {mae_modalities}")
    print(f"  Latent modalities (feature matching): {latent_reconstruct_modalities}")
    if labeled_train_loader is not None and labeled_frequency > 0:
        print(f"  Mixed training mode: labeled_frequency={labeled_frequency:.2f}")
        print(f"    - Unlabeled batches (train2): MAE + latent + pre-fusion + distillation losses")
        print(f"    - Labeled batches (train1): CE + latent losses (newmod hallucinated)")

    # Validate and set default for active_losses
    all_loss_names = ['mae', 'latent', 'prefusion', 'distill', 'ce']
    if active_losses is None:
        active_losses = all_loss_names
    else:
        invalid = set(active_losses) - set(all_loss_names)
        if invalid:
            raise ValueError(f"Invalid loss names: {invalid}. Valid: {all_loss_names}")
    print(f"  Active losses: {active_losses}")

    all_modalities = list(set(mae_modalities + latent_reconstruct_modalities))

    # Create frozen copy of full classifier for both feature and label distillation
    # The model already has a trained classifier on starting_modality
    teacher_classifier = copy.deepcopy(model)
    teacher_classifier.freeze_all()
    teacher_classifier.eval()
    print(f"\nTeacher classifier created (frozen copy of model for feature + label distillation)")
    
    # Ensure classifier is in ensemble mode for per-modality distillation
    if model.classifier_strategy != 'ensemble':
        print(f"!! Converting classifier from {model.classifier_strategy} to ensemble mode for label distillation")
        model.switch_strategy('ensemble')
        # Instantiate classifiers for any new modalities that don't have them yet
        for mod in all_modalities:
            if mod not in model.modality_classifiers:
                model.instantiate_modality_classifier(mod)

    model.freeze_all()
    model.set_requires_grad("all", clsreg=True, modality_encoders=True, mfla=False, msla=True, patch_embedders=True, classifier=True)
    model.set_requires_grad("backbone", mask_token=False, blocks=True, norm=True)
    
    evan = model.evan
    embed_dim = evan.embed_dim
    patch_size = evan.patch_size
    num_patches = (evan.img_size // patch_size) ** 2

    # Create updatable componenets for training
    mae_decoders=create_mae_decoders(embed_dim,patch_size,modality_bands_dict,mae_modalities,device)
    latent_projectors = create_latent_projectors(embed_dim, latent_reconstruct_modalities, device)
    intermediate_projectors = create_intermediate_projectors(embed_dim, all_modalities, device)


    trainable_in_evan = sum(p.numel() for p in evan.parameters() if p.requires_grad)
    trainable_in_model = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_decoder = sum(p.numel() for p in mae_decoders.parameters())
    trainable_projector = sum(p.numel() for p in latent_projectors.parameters())
    trainable_intermediate_proj = sum(p.numel() for p in intermediate_projectors.parameters())
    trainable_total = trainable_in_model + trainable_decoder + trainable_projector + trainable_intermediate_proj
    print(f"\nTrainable parameters: {trainable_total}")
    print(f"    Model (EVAN + Classifier): {trainable_in_model} ({100*trainable_in_model/(sum(p.numel() for p in model.parameters())):.2f}%)")
    print(f"      - EVAN backbone: {trainable_in_evan}")
    print(f"    MAE decoders: {trainable_decoder}")
    print(f"    Latent projectors (CLS + Patch): {trainable_projector}")
    print(f"    Intermediate projectors: {trainable_intermediate_proj}")

    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    params += list(intermediate_projectors.parameters())
    if 'mae' in active_losses:
        params += list(mae_decoders.parameters())
    if 'latent' in active_losses:
        params += list(latent_projectors.parameters())
    num_params = sum(p.numel() for p in params)
    print(f"Total trainable parameters: {num_params:,}")
    # assert num_params==trainable_total, f"{num_params=} != {trainable_total=}"

    optimizer = torch.optim.AdamW(params, lr=args.ssl_lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=0.7, patience=12, min_lr=1e-6)

    # Loss functions
    mse_fn = nn.MSELoss()

    # Identify newmod (modalities that are not the starting modality)
    starting_modality = evan.starting_modality
    newmod_list = [m for m in all_modalities if m != starting_modality]
    ce_fn = nn.CrossEntropyLoss()

    # Training loop
    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        evan.train()
        mae_decoders.train()
        latent_projectors.train()
        intermediate_projectors.train()

        # Track losses separately for labeled and unlabeled batches
        train_loss = 0.0
        train_mae_loss = 0.0
        train_latent_loss = 0.0
        train_pre_fusion_loss = 0.0
        train_distill_loss = 0.0
        train_ce_loss = 0.0
        train_count = 0
        labeled_count = 0
        unlabeled_count = 0

        pbar = tqdm(
            mixed_batch_iterator(train_loader, labeled_train_loader, labeled_frequency),
            total=len(train_loader),
            desc=f"SHOT Epoch {epoch+1}/{args.epochs}"
        )

        for batch, is_labeled in pbar:
            if is_labeled:
                # ===== LABELED MONOMODAL PATH (train1) =====
                # Only starting_modality available, use real labels
                labels = batch['label'].to(device)

                # 1. Extract only starting_modality
                monomodal_input = create_multimodal_batch(
                    batch, modality_bands_dict=modality_bands_dict,
                    modalities=(starting_modality,)
                )
                monomodal_input = {k: v.to(device) for k, v in monomodal_input.items()}
                batch_size = monomodal_input[starting_modality].shape[0]

                # 2. Get prefusion features for starting_modality
                prefusion_features = evan.forward_modality_specific_features(monomodal_input)

                # 3. Hallucinate newmod via projectors
                for newmod in newmod_list:
                    src_seq = prefusion_features[starting_modality]
                    src_seq_norm = F.layer_norm(src_seq, [src_seq.shape[-1]])
                    key = f"{starting_modality}_to_{newmod}"
                    prefusion_features[newmod] = intermediate_projectors[key](src_seq_norm)

                # 4. Forward through fusion
                student_fused = evan.forward_fusion_from_modality_features(prefusion_features)

                # 5. Latent loss (match teacher on starting_modality only)
                batch_latent_loss = 0.0
                latent_loss = torch.tensor(0.0, device=device)
                if 'latent' in active_losses:
                    with torch.no_grad():
                        teacher_out = teacher_classifier.evan.forward_features(monomodal_input)

                    for mod in latent_reconstruct_modalities:
                        if mod == starting_modality:  # Only compute for real modality
                            student_patches = student_fused[mod]['x_norm_patchtokens']
                            teacher_patches = teacher_out[mod]['x_norm_patchtokens'].detach()
                            student_cls = student_fused[mod]['x_norm_clstoken']
                            teacher_cls = teacher_out[mod]['x_norm_clstoken'].detach()

                            student_seq = torch.cat([student_cls.unsqueeze(1), student_patches], dim=1)
                            projected_seq = latent_projectors[mod](student_seq)
                            projected_cls = projected_seq[:, 0, :]
                            projected_patches = projected_seq[:, 1:, :]

                            latent_loss = latent_loss + mse_fn(projected_cls, teacher_cls) + mse_fn(projected_patches, teacher_patches)
                            batch_latent_loss += latent_loss.item()

                # 6. Hard label CE loss
                total_loss = latent_loss
                batch_ce_loss = 0.0
                if 'ce' in active_losses:
                    student_logits = model.classify_from_features(student_fused)
                    ce_loss = ce_fn(student_logits, labels)
                    total_loss = total_loss + ce_loss
                    batch_ce_loss = ce_loss.item()

                # No MAE, pre-fusion, or distillation loss for labeled batches
                batch_mae_loss = 0.0
                batch_pre_fusion_loss = 0.0
                batch_distill_loss = 0.0

                train_ce_loss += batch_ce_loss
                labeled_count += 1

            else:
                # ===== UNLABELED MULTIMODAL PATH (train2) =====
                # Both modalities available, use teacher distillation
                full_multimodal_input = create_multimodal_batch(
                    batch, modality_bands_dict=modality_bands_dict,
                    modalities=tuple(all_modalities)
                )
                full_multimodal_input = {k: v.to(device) for k, v in full_multimodal_input.items()}
                batch_size = next(iter(full_multimodal_input.values())).shape[0]
                prefusion_features = evan.forward_modality_specific_features(full_multimodal_input)

                # Step 1: Get teacher targets (unmasked, frozen)
                with torch.no_grad():
                    teacher_input = {teacher_mod: full_multimodal_input[teacher_mod] for teacher_mod in latent_reconstruct_modalities}
                    teacher_out = teacher_classifier.evan.forward_features(teacher_input)

                # Step 2: Compute pre-fusion sequence projection loss
                batch_pre_fusion_loss = 0.0
                prefusion_loss = torch.tensor(0.0, device=device)
                if 'prefusion' in active_losses:
                    pre_fusion_loss_count = 0
                    for src_mod in all_modalities:
                        src_seq = prefusion_features[src_mod].detach() if src_mod in latent_reconstruct_modalities else prefusion_features[src_mod]
                        for tgt_mod in all_modalities:
                            if src_mod != tgt_mod:
                                tgt_seq = prefusion_features[tgt_mod].detach() if tgt_mod in latent_reconstruct_modalities else prefusion_features[tgt_mod]
                                key = f"{src_mod}_to_{tgt_mod}"
                                src_seq_norm = F.layer_norm(src_seq, [src_seq.shape[-1]])
                                projected_seq = intermediate_projectors[key](src_seq_norm)
                                prefusion_loss = prefusion_loss + mse_fn(projected_seq, tgt_seq)
                                pre_fusion_loss_count += 1
                    prefusion_loss = prefusion_loss / pre_fusion_loss_count if pre_fusion_loss_count > 0 else prefusion_loss
                    batch_pre_fusion_loss = prefusion_loss.item()

                # Masking with protected_modalities to ensure newmod is never dropped (only when batch mixing)
                modality_masks, masked_mod_features = mask_input(
                    intermediate_projectors, batch_size, evan.n_storage_tokens, num_patches,
                    args.mae_mask_ratio, all_modalities, prefusion_features, args.modality_dropout, device,
                    protected_modalities=newmod_list if labeled_frequency > 0 else None
                )

                # Step 3: Forward through fusion blocks
                student_fused = evan.forward_fusion_from_modality_features(masked_mod_features)

                # Step 4: Compute losses
                total_loss = prefusion_loss
                batch_mae_loss = 0.0
                batch_latent_loss = 0.0

                # MAE loss
                if 'mae' in active_losses:
                    for mod in mae_modalities:
                        mask_float = modality_masks[mod].float()
                        if mask_float.sum() == 0:
                            continue
                        student_patches = student_fused[mod]['x_norm_patchtokens']
                        pred_pixels = mae_decoders[mod](student_patches)
                        target_img = full_multimodal_input[mod]
                        target_patches = patchify(target_img, patch_size)
                        mae_loss = mae_reconstruction_loss(pred_pixels, target_patches, mask_float)
                        total_loss = total_loss + mae_loss
                        batch_mae_loss += mae_loss.item()

                # Latent loss
                if 'latent' in active_losses:
                    for mod in latent_reconstruct_modalities:
                        student_patches = student_fused[mod]['x_norm_patchtokens']
                        teacher_patches = teacher_out[mod]['x_norm_patchtokens'].detach()
                        student_cls = student_fused[mod]['x_norm_clstoken']
                        teacher_cls = teacher_out[mod]['x_norm_clstoken'].detach()

                        student_seq = torch.cat([student_cls.unsqueeze(1), student_patches], dim=1)
                        projected_seq = latent_projectors[mod](student_seq)
                        projected_cls = projected_seq[:, 0, :]
                        projected_patches = projected_seq[:, 1:, :]

                        latent_loss = mse_fn(projected_cls, teacher_cls) + mse_fn(projected_patches, teacher_patches)
                        total_loss = total_loss + latent_loss
                        batch_latent_loss += latent_loss.item()

                # Label distillation
                batch_distill_loss = 0.0
                if 'distill' in active_losses:
                    with torch.no_grad():
                        teacher_modality = teacher_classifier.evan.starting_modality
                        teacher_input = {teacher_modality: full_multimodal_input[teacher_modality]}
                        teacher_logits = teacher_classifier(teacher_input)

                    student_logits = model.classify_from_features(student_fused)
                    distill_loss = distillation_loss(student_logits, teacher_logits, distillation_temperature)
                    total_loss = total_loss + distill_loss
                    batch_distill_loss = distill_loss.item()

                # No CE loss for unlabeled batches
                batch_ce_loss = 0.0
                unlabeled_count += 1

            # Backward pass (same for both paths)
            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
            optimizer.step()

            train_loss += total_loss.item()
            train_mae_loss += batch_mae_loss
            train_latent_loss += batch_latent_loss
            train_pre_fusion_loss += batch_pre_fusion_loss
            train_distill_loss += batch_distill_loss
            train_count += 1
            global_step += 1

            pbar.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'mae': f'{batch_mae_loss:.4f}',
                'latent': f'{batch_latent_loss:.4f}',
                'pre_fus': f'{batch_pre_fusion_loss:.4f}',
                'distill': f'{batch_distill_loss:.4f}',
                'ce': f'{batch_ce_loss:.4f}',
                'L/U': f'{labeled_count}/{unlabeled_count}'
            })

            # Log to wandb every step
            wandb.log({
                'train_loss': total_loss.item(),
                'mae_loss': batch_mae_loss,
                'latent_loss': batch_latent_loss,
                'pre_fusion': batch_pre_fusion_loss,
                'distill_loss': batch_distill_loss,
                'ce_loss': batch_ce_loss,
                'is_labeled': 1 if is_labeled else 0,
                'grad_norm': grad_norm.item(),
                'epoch': epoch + 1,
                'lr': optimizer.param_groups[0]['lr']
            })

        # Epoch summary
        train_loss /= train_count
        train_mae_loss /= max(unlabeled_count, 1)  # MAE only from unlabeled
        train_latent_loss /= train_count
        train_pre_fusion_loss /= max(unlabeled_count, 1)  # Pre-fusion only from unlabeled
        train_distill_loss /= max(unlabeled_count, 1)  # Distill only from unlabeled
        train_ce_loss /= max(labeled_count, 1)  # CE only from labeled
        labeled_ratio = labeled_count / train_count if train_count > 0 else 0

        scheduler.step(train_loss)

        print(f"\nEpoch {epoch+1}/{args.epochs}:")
        print(f"  Train - Total: {train_loss:.4f}, MAE: {train_mae_loss:.4f}, Latent: {train_latent_loss:.4f}, Pre-fusion: {train_pre_fusion_loss:.4f}, Distill: {train_distill_loss:.4f}, CE: {train_ce_loss:.4f}")
        print(f"  Batches - Labeled: {labeled_count}, Unlabeled: {unlabeled_count}, Ratio: {labeled_ratio:.2f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Periodic evaluation
        if eval_every_n_epochs is not None and test_loader is not None and (epoch + 1) % eval_every_n_epochs == 0:
            print(f"\n--- Periodic Evaluation at Epoch {epoch+1} ---")
            model.eval()
            unlabeled_modalities = [mod for mod in mae_modalities if mod not in latent_reconstruct_modalities]
            if not unlabeled_modalities:
                unlabeled_modalities = mae_modalities[:1]
            labeled_modalities = latent_reconstruct_modalities

            for objective in ["transfer", "peeking", "addition"]:
                accuracy, ens_acc = _delulu_stage3_test(
                    model=model,
                    evan=evan,
                    test_loader=test_loader,
                    device=device,
                    modality_bands_dict=modality_bands_dict,
                    unlabeled_modalities=unlabeled_modalities,
                    labeled_modalities=labeled_modalities,
                    all_modalities=all_modalities,
                    intermediate_projectors=intermediate_projectors,
                    objective=objective
                )
                print(f"  {objective.capitalize()} accuracy: {accuracy:.2f}%")
                wandb.log({f"test/{objective}_accuracy": accuracy, "epoch": epoch + 1})
                if objective == "addition" and ens_acc is not None:
                    wandb.log({f"test/addition_ens_accuracy": ens_acc, "epoch": epoch + 1})
            model.train()

    print("\n=== Phase 2 (Fusion MAE Training) complete ===")
    return mae_decoders, latent_projectors, intermediate_projectors, trainable_total
