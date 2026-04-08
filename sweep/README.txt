Hyperparameter Sweep Infrastructure
=====================================

Directory layout
----------------
sweep/
  create_sweep.py              # Register a new sweep with W&B (run once)
  sweep_shot.py                # Runner: SHOT (shot_ete.py), all datasets
  sweep_sft.py                 # Runner: stage-0 SFT (train_sft.py)
  sweep_baseline_distill.py    # Runner: baseline distillation
  run_sweep.sh                 # SBATCH script — 4 agents on 4 H100s
  sweep_registry.txt           # Log of created sweep IDs
  sweep_yaml/
    base.yaml                  # Shared HPs: lr, weight_decay
    sweep_shot.yaml            # SHOT-specific HPs
    sweep_sft.yaml             # SFT-specific HPs
    sweep_baseline_distill.yaml # Distill-specific HPs


All commands run from the repo root.

Step 1: Create the sweep (run once)
-------------------------------------
# EuroSAT SHOT (uses hardcoded checkpoint map):
python sweep/create_sweep.py --script shot --starting rgb --newmod nir

# Any other dataset (provide checkpoint explicitly):
python sweep/create_sweep.py --script shot \
    --dataset dfc2020 \
    --starting s2_rgb --newmod s1 \
    --stage0_checkpoint checkpoints/dfc2020_s2rgb_dinov3init_sft.pt

# Stage-0 SFT:
python sweep/create_sweep.py --script sft --dataset dfc2020 --modalities s2_rgb

# Baseline distillation:
python sweep/create_sweep.py --script baseline_distill \
    --dataset dfc2020 \
    --teacher_checkpoint checkpoints/dfc2020_s2_rgb_s0.pt \
    --modality s1

Prints the full sweep ID (entity/project/sweep_id).
Optionally paste it into sweep_registry.txt as a personal log.


Step 2: Submit to the cluster
------------------------------
sbatch sweep/run_sweep.sh tgz/delulu-sweep-dfc2020-s2_rgb-s1/<sweep_id>

To smoke-test locally (single trial, no SLURM):
  wandb agent tgz/delulu-sweep-dfc2020-s2_rgb-s1/<sweep_id> --count 1


Swept hyperparameters (sweep_shot.yaml)
----------------------------------------
  lr, weight_decay, mae_mask_ratio, modality_dropout,
  labeled_frequency, labeled_start_fraction, use_mae, use_latent


Adding a new swept hyperparameter
----------------------------------
1. Add it to the relevant sweep_yaml/sweep_*.yaml under `parameters:`
2. Add a matching --argparse argument in the runner script
3. Re-create the sweep (W&B sweeps are immutable once created)
