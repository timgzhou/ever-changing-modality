from shot import create_intermediate_projectors, SequenceProjector
import argparse
import torch
import torch.nn as nn
import logging
import csv
import os

from evan_main import EVANClassifier
from eurosat_data_utils import (
    get_loaders,
    get_loaders_with_val,
    get_modality_bands_dict
)
from train_utils import (
    delulu_supervision,
    single_modality_training_loop,
    supervised_training_loop
)

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='checkpoints/presetE.pt', help='Path to checkpoint file')
parser.add_argument('--num_supervised_epochs', type=int, default=8)
parser.add_argument('--csv_suffix', type=str, default=None, help='suffix to save experiment results to.')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Fraction of train2 to use for validation (default: 0.1)')
parser.add_argument('--teacher_checkpoint', type=str, default=None,
                    help='Path to teacher checkpoint for distillation (monomodal on starting_modality)')
parser.add_argument('--temperature', type=float, default=2.0,
                    help='Softmax temperature for KL divergence (default: 2.0)')
args = parser.parse_args()

train1_loader, val1_loader, train2_loader, val2_loader, test_loader = get_loaders_with_val(32, 4, val_ratio=args.val_ratio)
eval_lr=1e-3 #1e-4, jan28
criterion = nn.CrossEntropyLoss()
checkpoint_path = args.checkpoint
classifier_strategy="ensemble"
num_supervised_epochs=args.num_supervised_epochs

checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = checkpoint['config']
evan_config = config['evan_config']
print(f"config:")
for k, v in config.items():
    print(f"  {k}: {v}")


device = 'cuda' if torch.cuda.is_available() else "cpu"

# CSV logging setup
csv_results = []
csv_path = f"res/modality-transfer_{args.csv_suffix}.csv"

def load_model_and_unfreeze_classifier(checkpoint_path,classifier_strategy,device):
    model=EVANClassifier.from_checkpoint(checkpoint_path,device)
    model.switch_strategy(classifier_strategy)
    model.freeze_all()
    model.set_requires_grad('all', classifier=True)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model

model = load_model_and_unfreeze_classifier(checkpoint_path,classifier_strategy,device)
modalities = model.evan.supported_modalities
modality_bands_dict = get_modality_bands_dict(*modalities)
intermediate_projectors = create_intermediate_projectors(
    hidden_dim=model.evan.embed_dim,
    all_modalities=modalities,
    device=device
)

# ======================!!UNSUPERVISED TRAINING!!======================
# Evaluate all three delulu objectives: transfer, addition, peeking
print("\n" + "="*70)
print("=== Evaluating delulu supervision with all objectives ===")
print("="*70)

# Support both old and new checkpoint key names
projector_state_dict = checkpoint.get('intermediate_projectors_state_dict') or checkpoint.get('cls_projectors_state_dict')
intermediate_projectors.load_state_dict(projector_state_dict)
print(f"Loaded SHOT-trained intermediate_projectors: {list(intermediate_projectors.keys())}")
starting_modality = model.evan.starting_modality
unlabeled_modalities = list(set(model.evan.supported_modalities)-{starting_modality})

# Load teacher model if provided (for distillation)
teacher_model = None
if args.teacher_checkpoint:
    teacher_model = EVANClassifier.from_checkpoint(args.teacher_checkpoint, device)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    print(f"Loaded teacher from {args.teacher_checkpoint}")

# Run 3 modes: no_distillation, alternating, distill_only
# distill_mode: "none" (no distillation), "alternating", "distill_only"
distill_modes = ["none"]  # Always run without distillation
if teacher_model is not None:
    distill_modes.extend(["alternating", "distill_only"])
distill_modes=["none"] # TODO:TEMP
for distill_mode in distill_modes:
    print("\n" + "="*70)
    print(f"=== Running {distill_mode} ===")
    print("="*70)

    for objective in ["transfer", "addition", "peeking"]:
        print("\n" + "-"*70)
        print(f"--- Running delulu_supervision with objective={objective}, distill_mode={distill_mode} ---")
        print("-"*70)

        model = load_model_and_unfreeze_classifier(checkpoint_path, classifier_strategy, device)

        best_test_acc, test_acc = delulu_supervision(
            model=model,
            unlabeled_train_loader=train2_loader,
            labeled_train_loader=train1_loader,
            test_loader=test_loader,
            device=device,
            modality_bands_dict=modality_bands_dict,
            unlabeled_modalities=unlabeled_modalities,
            labeled_modalities=[starting_modality],
            intermediate_projectors=intermediate_projectors,
            lr=eval_lr,
            epochs=num_supervised_epochs,
            stage2epochs=8,
            eval_every_n_epochs=1,
            objective=objective,
            val1_loader=val1_loader,
            val2_loader=val2_loader,
            teacher_model=teacher_model if distill_mode != "none" else None,
            temperature=args.temperature,
            distill_only=(distill_mode == "distill_only")
        )

        # Determine real/hallucinated modalities based on objective for logging
        real_mod,hal_mod = "none","none"
        if objective == "transfer":
            # Test on unlabeled only, hallucinate labeled
            real_mod = "+".join(unlabeled_modalities)
            hal_mod = starting_modality
        elif objective == "addition":
            # Test on both modalities (all real)
            real_mod = "+".join([starting_modality] + unlabeled_modalities)
            hal_mod = None
        elif objective == "peeking":
            # Test on labeled only, hallucinate unlabeled
            real_mod = starting_modality
            hal_mod = "+".join(unlabeled_modalities)

        csv_results.append({
            "eval_type": f"delulu-{objective}",
            "distillation": distill_mode,
            "real_modality": real_mod,
            "hallucinated_modality": hal_mod,
            "test_acc": test_acc,
            "best_test_acc": best_test_acc,
        })

# Write CSV results (append to single file)
fieldnames = [
    "checkpoint",
    "num_supervised_epochs",
    "starting_modality",
    "supported_modalities",
    "eval_type",
    "distillation",
    "real_modality",
    "hallucinated_modality",
    "test_acc",
    "best_test_acc",
]
file_exists = os.path.exists(csv_path)
with open(csv_path, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    for result in csv_results:
        row = {
            "checkpoint": checkpoint_path,
            "num_supervised_epochs": num_supervised_epochs,
            "starting_modality": starting_modality,
            "supported_modalities": "+".join(modalities),
            **result
        }
        writer.writerow(row)
print(f"\nResults appended to: {csv_path}")

# python -u evaluate_shot.py