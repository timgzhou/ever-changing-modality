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
parser.add_argument('--num_supervised_epochs', type=int, default=1, help='Path to checkpoint file')
parser.add_argument('--csv_suffix', type=str, default=None, help='suffix to save experiment results to.')
parser.add_argument('--val_ratio', type=float, default=0.1, help='Fraction of train2 to use for validation (default: 0.1)')
args = parser.parse_args()

train1_loader, val1_loader, train2_loader, val2_loader, test_loader = get_loaders_with_val(32, 4, val_ratio=args.val_ratio)
eval_lr=1e-4
criterion = nn.CrossEntropyLoss()
checkpoint_path = args.checkpoint
clssifier_strategy="ensemble"
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

def load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device):
    model=EVANClassifier.from_checkpoint(checkpoint_path,device)
    model.switch_strategy(clssifier_strategy)
    model.freeze_all()
    model.set_requires_grad('all', classifier=True)
    model = model.to(device)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    return model

model = load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device)
modalities = model.evan.supported_modalities
modality_bands_dict = get_modality_bands_dict(*modalities)
intermediate_projectors = create_intermediate_projectors(
    hidden_dim=model.evan.embed_dim,
    all_modalities=modalities,
    device=device
)
"""
# ================================================ SUPERVISED TRAINING =============================================
print("\n\nSUPERVISED EVALUATION\n")

# ========================================= -----------MULTIMODAL------------- =====================================
print("\n" + "="*70)
print("=== Evaluating fusion with multimodal supervised probing ===")
print("="*70)
model = load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)
criterion = nn.CrossEntropyLoss()
train_acc, test_acc = supervised_training_loop(
    model=model,
    train_loader=train1_loader,
    test_loader_full=test_loader,
    device=device,
    modality_bands_dict=modality_bands_dict,
    criterion=criterion,
    optimizer=optimizer,
    num_epochs=num_supervised_epochs,
    train_modalities=tuple(modalities),
    phase_name="SHOT supervised eval"
)
print(f"{test_acc=}")
csv_results.append({
    "eval_type": "real-multimodal-real-supervision",
    "real_modality": "+".join(modalities),
    "hallucinated_modality": None,
    "test_acc": test_acc,
    "best_test_acc": test_acc,
})

# ========================================= -----------hallucinated multimodal under real supervision ------------- =====================================
print("\n" + "="*70)
print("=== Evaluating (hallucinated) fusion with monomodal under real supervision ===")
print("="*70)
# model = load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device)
# Support both old and new checkpoint key names
projector_state_dict = checkpoint.get('intermediate_projectors_state_dict') or checkpoint.get('cls_projectors_state_dict')
intermediate_projectors.load_state_dict(projector_state_dict, strict=False)

for evaluated_modality in modalities:
    for hallucinate_modality in modalities:
        if evaluated_modality==hallucinate_modality: continue
        model = load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device)
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)
        print(f"hallucinated supervision: real {evaluated_modality} and hallucinated {hallucinate_modality}")
        _, test_acc_wh, best_test_acc_wh, _ = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer, num_supervised_epochs,
            modality=evaluated_modality, phase_name=f"SHOT {evaluated_modality}-only eval w/ hallucination {hallucinate_modality}",
            hallucinate_modality=True, pseudo_modalities=[hallucinate_modality], intermediate_projectors=intermediate_projectors
        )
        print(f"\nREAL {evaluated_modality} + Hallucinated {hallucinate_modality} Result: {test_acc_wh=:.2f} {best_test_acc_wh=:.2f}\n\n")
        csv_results.append({
            "eval_type": "hallucinated-multimodal-real-supervision",
            "real_modality": evaluated_modality,
            "hallucinated_modality": hallucinate_modality,
            "test_acc": test_acc_wh,
            "best_test_acc": best_test_acc_wh,
        })

for evaluated_modality in modalities:
    model = load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)
    _, test_acc_woh, best_test_acc_woh, _ = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer, num_supervised_epochs,
            modality=evaluated_modality, phase_name=f"SHOT {evaluated_modality}-only eval w/o hallucination.",
        )
    print(f"\n{evaluated_modality}-only Result: {test_acc_woh=:.2f} {best_test_acc_woh=:.2f}\n\n")
    csv_results.append({
        "eval_type": "real-monomodal-real-supervision",
        "real_modality": evaluated_modality,
        "hallucinated_modality": None,
        "test_acc": test_acc_woh,
        "best_test_acc": best_test_acc_woh,
    })
"""


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
unlabeld_modalities = list(set(model.evan.supported_modalities)-{starting_modality})

for objective in ["transfer", "addition", "peeking"]:
    print("\n" + "-"*70)
    print(f"--- Running delulu_supervision with objective={objective} ---")
    print("-"*70)

    model = load_model_and_unfreeze_clssifier(checkpoint_path, clssifier_strategy, device)

    best_test_acc, test_acc = delulu_supervision(
        model=model,
        unlabeled_train_loader=train2_loader,
        labeled_train_loader=train1_loader,
        test_loader=test_loader,
        device=device,
        modality_bands_dict=modality_bands_dict,
        unlabeled_modalities=unlabeld_modalities,
        labeled_modalities=[starting_modality],
        intermediate_projectors=intermediate_projectors,
        lr=eval_lr,
        epochs=num_supervised_epochs,
        eval_every_n_epochs=1,
        objective=objective,
        val1_loader=val1_loader,
        val2_loader=val2_loader
    )

    # Determine real/hallucinated modalities based on objective for logging
    if objective == "transfer":
        # Test on unlabeled only, hallucinate labeled
        real_mod = "+".join(unlabeld_modalities)
        hal_mod = starting_modality
    elif objective == "addition":
        # Test on both modalities (all real)
        real_mod = "+".join([starting_modality] + unlabeld_modalities)
        hal_mod = None
    elif objective == "peeking":
        # Test on labeled only, hallucinate unlabeled
        real_mod = starting_modality
        hal_mod = "+".join(unlabeld_modalities)

    csv_results.append({
        "eval_type": f"delulu-{objective}",
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