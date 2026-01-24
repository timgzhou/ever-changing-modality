
from shot import create_int_cls_projectors
import argparse
import torch
import torch.nn as nn
import logging

from evan_main import EVANClassifier
from eurosat_data_utils import (
    get_loaders,
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
args = parser.parse_args()

train1_loader, train2_loader, test_loader = get_loaders(32,4)
eval_lr=1e-4
criterion = nn.CrossEntropyLoss()
checkpoint_path = args.checkpoint
clssifier_strategy="ensemble"
num_supervised_epochs=12

checkpoint = torch.load(checkpoint_path, map_location='cpu')
config = checkpoint['config']
evan_config = config['evan_config']
print(f"config:")
for k, v in config.items():
    print(f"  {k}: {v}")


device = 'cuda' if torch.cuda.is_available() else "cpu"

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
intermediate_cls_projectors = create_int_cls_projectors(
    hidden_dim=model.evan.embed_dim,
    all_modalities=modalities,
    device=device
)

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

# ========================================= -----------hallucinated multimodal under real supervision ------------- =====================================
print("\n" + "="*70)
print("=== Evaluating (hallucinated) fusion with monomodal under real supervision ===")
print("="*70)
model = load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device)
intermediate_cls_projectors.load_state_dict(checkpoint['cls_projectors_state_dict'], strict=False)

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
            hallucinate_modality=True, pseudo_modalities=[hallucinate_modality],cls_projectors=intermediate_cls_projectors
        )
        print(f"\nREAL {evaluated_modality} + Hallucinated {hallucinate_modality} Result: {test_acc_wh=:.2f} {best_test_acc_wh=:.2f}\n\n")

for evaluated_modality in modalities:
    model = load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=eval_lr)
    _, test_acc_woh, best_test_acc_woh, _ = single_modality_training_loop(
            model, train1_loader, test_loader, device,
            modality_bands_dict, criterion, optimizer, num_supervised_epochs,
            modality=evaluated_modality, phase_name=f"SHOT {evaluated_modality}-only eval w/o hallucination.", 
        )
    print(f"\n{evaluated_modality}-only Result: {test_acc_woh=:.2f} {best_test_acc_woh=:.2f}\n\n")



# ======================!!UNSUPERVISED TRAINING!!======================
# 1. hallucinate supervision transfer
print("\n" + "="*70)
print("=== Evaluating (hallucinated) fusion with monomodal under pseudo supervision ===")
print("="*70)

model = load_model_and_unfreeze_clssifier(checkpoint_path,clssifier_strategy,device)
intermediate_cls_projectors.load_state_dict(checkpoint['cls_projectors_state_dict'])
print(f"Loaded SHOT-trained intermediate_cls_projectors: {list(intermediate_cls_projectors.keys())}")
starting_modality = model.evan.starting_modality
unlabeld_modalities = list(set(model.evan.supported_modalities)-{starting_modality})

test_acc = delulu_supervision(
    model=model,
    unlabeled_train_loader=train2_loader,
    labeled_train_loader=train1_loader,
    test_loader=test_loader,
    device=device,
    modality_bands_dict=modality_bands_dict,
    unlabeled_modalities=unlabeld_modalities,
    labeled_modalities=[starting_modality],
    intermediate_cls_projectors=intermediate_cls_projectors,
    lr=eval_lr,
    epochs=num_supervised_epochs,
    eval_every_n_epochs=1
)

# python -u evaluate_shot.py