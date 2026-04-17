"""
Compare teacher logits between baseline_distillation and SHOT's _unlabeled_batch_step path.

Baseline distillation:
    teacher_logits = teacher_model({s2: batch_s2})

SHOT unlabeled step (teacher_is_peeking=False):
    teacher_out = teacher_classifier.evan.forward_features({s2: batch_s2})  # unused
    teacher_logits = teacher_classifier({s2: batch_s2})

These should be identical. This script verifies that.
"""

import torch
import torch.nn.functional as F
from data_utils import get_loaders, create_multimodal_batch
from evan_main import EVANClassifier
import copy

STAGE0_CKPT = "checkpoints/sft_evan_base_benv2_s2_fft_lr0.0001_20260414_020254.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4

print(f"Device: {DEVICE}")
print(f"Loading stage0 checkpoint: {STAGE0_CKPT}")

# --- Load model (student) ---
model = EVANClassifier.from_checkpoint(STAGE0_CKPT, DEVICE)
model = model.to(DEVICE)

raw_ckpt = torch.load(STAGE0_CKPT, map_location='cpu')
config = raw_ckpt['config']
starting_modality = config['evan_config']['starting_modality']
print(f"Starting modality: {starting_modality}")

# --- Load a small batch from train2 (unlabeled multimodal) ---
train1_loader, val1_loader, train2_loader, val2_loader, test_loader, task_config = \
    get_loaders('benv2', starting_modality, BATCH_SIZE, num_workers=2, new_modality='s1')

modality_bands_dict = task_config.modality_bands_dict
batch = next(iter(train2_loader))

# Extract s2 input (same as both baselines would use)
s2_input = create_multimodal_batch(batch, modality_bands_dict=modality_bands_dict, modalities=(starting_modality,))
s2_input = {k: v.to(DEVICE) for k, v in s2_input.items()}

# ============================================================
# Baseline distillation teacher path:
#   teacher_logits = teacher_model(teacher_input)
# ============================================================
teacher_model = copy.deepcopy(model)
teacher_model.freeze_all()
teacher_model.eval()

with torch.no_grad():
    baseline_teacher_logits = teacher_model(s2_input)

print(f"\nBaseline teacher logits shape: {baseline_teacher_logits.shape}")
print(f"Baseline teacher logits (first sample, first 5 classes):\n  {baseline_teacher_logits[0, :5].cpu().tolist()}")

# ============================================================
# SHOT unlabeled step teacher path (teacher_is_peeking=False):
#   teacher_out = teacher_classifier.evan.forward_features(teacher_input)  # unused
#   _teacher_input = {teacher_modality: full_multimodal_input[teacher_modality]}
#   teacher_logits = teacher_classifier(_teacher_input)
# ============================================================

# Simulate what train_shot does: deepcopy model, then switch student to ensemble
# (teacher is deepcopied BEFORE switch_strategy in the current code)
shot_teacher = copy.deepcopy(model)
shot_teacher.freeze_all()
shot_teacher.eval()

# Now simulate what happens in _unlabeled_batch_step
full_multimodal_input = create_multimodal_batch(
    batch, modality_bands_dict=modality_bands_dict, modalities=('s2', 's1')
)
full_multimodal_input = {k: v.to(DEVICE) for k, v in full_multimodal_input.items()}

latent_reconstruct_modalities = [starting_modality]  # [s2]

with torch.no_grad():
    teacher_input = {m: full_multimodal_input[m] for m in latent_reconstruct_modalities}

    # Line 903: forward_features — result is NOT used downstream (dead code)
    teacher_out_unused = shot_teacher.evan.forward_features(teacher_input)

    # Line 904-906: actual teacher logits
    teacher_modality = shot_teacher.evan.starting_modality
    _teacher_input = {teacher_modality: full_multimodal_input[teacher_modality]}
    shot_teacher_logits = shot_teacher(_teacher_input)

print(f"\nSHOT teacher logits shape: {shot_teacher_logits.shape}")
print(f"SHOT teacher logits (first sample, first 5 classes):\n  {shot_teacher_logits[0, :5].cpu().tolist()}")

# ============================================================
# Compare
# ============================================================
max_diff = (baseline_teacher_logits - shot_teacher_logits).abs().max().item()
are_equal = torch.allclose(baseline_teacher_logits, shot_teacher_logits, atol=1e-5)

print(f"\n{'='*50}")
print(f"Max absolute difference: {max_diff:.2e}")
print(f"Logits identical (atol=1e-5): {are_equal}")

if not are_equal:
    print("\nDIFFERENCE DETECTED — teacher logits are NOT the same!")
    print("Diff per sample (max across classes):")
    for i in range(BATCH_SIZE):
        d = (baseline_teacher_logits[i] - shot_teacher_logits[i]).abs().max().item()
        print(f"  sample {i}: {d:.4e}")
else:
    print("\nOK — teacher logits are identical between both paths.")

# ============================================================
# Also compare the distillation_loss functions
# baseline uses train_utils.distillation_loss (with kl_type param)
# SHOT uses shot.distillation_loss (no kl_type param, always KD)
# ============================================================
print(f"\n{'='*50}")
print("Comparing distillation_loss implementations...")

from shot import distillation_loss as shot_distill_loss
from train_utils import distillation_loss as baseline_distill_loss

# Make a fake student logits tensor (random, simulating a partially-trained s1 head)
torch.manual_seed(42)
fake_student_logits = torch.randn(BATCH_SIZE, 19, device=DEVICE)

shot_loss = shot_distill_loss(fake_student_logits, baseline_teacher_logits, temperature=1.0)
baseline_loss_kd = baseline_distill_loss(fake_student_logits, baseline_teacher_logits, temperature=1.0, kl_type='kd')
baseline_loss_ttm = baseline_distill_loss(fake_student_logits, baseline_teacher_logits, temperature=1.0, kl_type='ttm')

print(f"SHOT distill_loss (temp=1.0):           {shot_loss.item():.6f}")
print(f"Baseline distill_loss kd (temp=1.0):    {baseline_loss_kd.item():.6f}")
print(f"Baseline distill_loss ttm (temp=1.0):   {baseline_loss_ttm.item():.6f}")

shot_loss2 = shot_distill_loss(fake_student_logits, baseline_teacher_logits, temperature=2.0)
baseline_loss2 = baseline_distill_loss(fake_student_logits, baseline_teacher_logits, temperature=2.0, kl_type='kd')
print(f"SHOT distill_loss (temp=2.0):           {shot_loss2.item():.6f}")
print(f"Baseline distill_loss kd (temp=2.0):    {baseline_loss2.item():.6f}")

losses_match_kd = torch.allclose(shot_loss, baseline_loss_kd, atol=1e-5)
print(f"\nSHOT loss == Baseline KD loss: {losses_match_kd}")

# ============================================================
# Verify student initialization: which param groups are trainable
# and whether newly-added s1 components are actually in the optimizer.
# Reproduces the exact setup in train_shot().
# ============================================================
print(f"\n{'='*50}")
print("Verifying student initialization and optimizer param groups...")

import sys
sys.path.insert(0, '.')
from shot import train_shot
from evan_main import EVANClassifier
import copy, types, argparse

# Reload fresh model from checkpoint
student = EVANClassifier.from_checkpoint(STAGE0_CKPT, DEVICE).to(DEVICE)
evan = student.evan

# Replicate shot_ete.py: create s1 components on evan
from data_utils import get_loaders
task_config2 = task_config  # reuse from above
from evan_main import EVAN
evan.intermediate_projector_type = 'cross'
evan.intermediate_projector_num_layers = 2
if not hasattr(evan, 'projector_queries'):
    evan.projector_queries = torch.nn.ParameterDict()
evan.create_modality_components('s1', 2)
student = student.to(DEVICE)

# Replicate train_shot setup (lines 1216-1277 of shot.py)
import torch.nn as nn

teacher = copy.deepcopy(student)
teacher.freeze_all()
teacher.eval()

# switch_strategy + instantiate s1 head
if student.classifier_strategy != 'ensemble':
    student.switch_strategy('ensemble')
all_modalities = ['s2', 's1']
for mod in all_modalities:
    if mod not in student.modality_classifiers:
        student.instantiate_modality_classifier(mod)

student.freeze_all()
student.set_requires_grad("all", clsreg=True, modality_encoders=True, mfla=False, msla=True, patch_embedders=True, head=True)
student.set_requires_grad("backbone", blocks=True, norm=True)

from shot import create_mae_decoders, create_latent_projectors
embed_dim = evan.embed_dim
patch_size = evan.patch_size
modality_bands_dict2 = task_config.modality_bands_dict
mae_modalities = ['s2', 's1']
latent_reconstruct_modalities = ['s2']
active_losses = ['distill', 'ce']  # mimicking user's test run

mae_decoders = create_mae_decoders(embed_dim, patch_size, modality_bands_dict2, mae_modalities, DEVICE)
latent_projectors = create_latent_projectors(embed_dim, latent_reconstruct_modalities, DEVICE)
evan.set_requires_grad("all", intermediate_projectors=True)

params = list(filter(lambda p: p.requires_grad, student.parameters()))
if 'mae' in active_losses:
    params += list(mae_decoders.parameters())
if 'latent' in active_losses:
    params += list(latent_projectors.parameters())

param_ids_in_optimizer = {id(p) for p in params}

def check_group(name, param_iter):
    all_p = list(param_iter)
    in_opt = sum(1 for p in all_p if id(p) in param_ids_in_optimizer)
    req_grad = sum(1 for p in all_p if p.requires_grad)
    total = len(all_p)
    status = "OK" if in_opt == total else "MISSING FROM OPTIMIZER"
    print(f"  {name:45s} total={total:3d}  requires_grad={req_grad:3d}  in_optimizer={in_opt:3d}  [{status}]")

print(f"\nOptimizer has {len(params)} param tensors total ({sum(p.numel() for p in params):,} values)")
print()
check_group("s2 patch_embedder",        evan.patch_embedders['s2'].parameters())
check_group("s1 patch_embedder",        evan.patch_embedders['s1'].parameters())
check_group("s2 MSLA blocks",           evan.modality_specific_layer_adaptors['s2'].parameters())
check_group("s1 MSLA blocks",           evan.modality_specific_layer_adaptors['s1'].parameters())
check_group("s2 cls_token",             [evan.cls_tokens['s2']])
check_group("s1 cls_token",             [evan.cls_tokens['s1']])
check_group("s2 modality_encoding",     [evan.modality_encodings['s2']])
check_group("s1 modality_encoding",     [evan.modality_encodings['s1']])
check_group("intermediate_projectors",  evan.intermediate_projectors.parameters())
check_group("shared backbone blocks",   evan.blocks.parameters())
check_group("shared backbone norm",     evan.norm.parameters())
check_group("s2 classifier head",       student.modality_classifiers['s2'].parameters())
check_group("s1 classifier head",       student.modality_classifiers['s1'].parameters())
check_group("mae_decoders (s2+s1)",     mae_decoders.parameters())
check_group("latent_projectors (s2)",   latent_projectors.parameters())
