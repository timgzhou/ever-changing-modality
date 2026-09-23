"""Teacher memorisation check: RMSE on train1 (should be near-fit) vs
train2 / val1 / val2 / test (should all be close to each other).

If train2 tracks train1 rather than val/test, the "unlabeled" stage-1 pool
has been seen with labels and the whole setting leaks.
"""
import sys
import torch

sys.path.insert(0, ".")

from biomassters_data_utils import get_biomassters_loaders
from delulunet_main import EvanSegmenter
from train_utils import make_criterion, evaluate

CKS = {
    "s1": "checkpoints/sft_evan_base_biomassters_s1_fft_lr0.0005_20260919_125600.pt",
    "s2": "checkpoints/sft_evan_base_biomassters_s2_fft_lr0.0005_20260919_125845.pt",
}

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BS = 16
# The teachers were trained on the pooled cache (num_time_steps = -12).
NTS = 12
POOL = True

print(f"device={DEV}  batch_size={BS}  temporal_pool={POOL}  T={NTS}\n")

for mod, ck in CKS.items():
    print("=" * 66)
    print(f"  teacher: {mod}   {ck.split('/')[-1]}")
    print("=" * 66)

    train1, val1, train2, val2, test, tc = get_biomassters_loaders(
        batch_size=BS, num_workers=4, starting_modality=mod,
        new_modality=None, num_time_steps=NTS, temporal_pool=POOL,
    )

    model = EvanSegmenter.from_checkpoint(ck, device=DEV)
    model.to(DEV).eval()

    crit = make_criterion(tc)
    mbd = tc.modality_bands_dict

    common = dict(
        criterion=crit, device=DEV, modality_bands_dict=mbd,
        modalities_to_use=(mod,), label_key=tc.label_key,
        segmentation=False, regression=True,
        regression_scale=getattr(tc, "regression_scale", 1.0),
        regression_mask_above=getattr(tc, "regression_mask_above", None),
    )

    for name, ld in [("train1 (SEEN, labeled)", train1),
                     ("train2 (stage-1 pool)", train2),
                     ("val1", val1), ("val2", val2), ("test", test)]:
        loss, metric = evaluate(model, ld, **common)
        print(f"  {name:24s} RMSE {metric:7.2f}   loss {loss:.4f}")
    print()
    del model
    torch.cuda.empty_cache()
