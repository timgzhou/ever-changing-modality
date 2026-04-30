import sys, os, torch
sys.path.insert(0, os.path.dirname(__file__))
from evan_main import EVANClassifier
from geobench_data_utils import get_benv2_loaders
from shot import evaluate_multimodal

CHECKPOINT = 'checkpoints/delulunet_benv2_0420_1051.pt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = EVANClassifier.from_checkpoint(CHECKPOINT, DEVICE)
model.eval()
evan = model.evan
mods = evan.supported_modalities  # e.g. ['s1', 's2']
mod_a, mod_b = mods[0], mods[1]
print(f"starting={mod_a}, new={mod_b}")

_, _, _, _, test_loader, task_config = get_benv2_loaders(
    batch_size=64, num_workers=4,
    starting_modality=mod_a, new_modality=mod_b,
)

metrics = evaluate_multimodal(
    model=model, evan=evan,
    loader=test_loader, device=DEVICE,
    modality_bands_dict=task_config.modality_bands_dict,
    starting_modality=mod_a,
    newmod_modalities=[mod_b],
    all_modalities=[mod_a, mod_b],
    multilabel=True, label_key='label',
    with_labels=True, desc="Test eval",
)
print(metrics)
