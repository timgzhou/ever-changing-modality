Final BEN-v2 SHOT Sweep
=======================

4 configs × 2 directions (s2→s1, s1→s2) = 8 sweeps × 32 runs = 256 jobs total.

Configs
-------
  delulu        protect_lrm=F, use_mask_token=F, lsf=0.5, prefusion+latent+distill+ce
  mask-token    protect_lrm=F, use_mask_token=T, lsf=0.5, latent+distill+ce  (no prefusion)
  no-batch-mix  protect_lrm=F, use_mask_token=F, lsf=1.0, lf=0, prefusion+latent+distill  (no ce)
  no-prefusion  protect_lrm=F, use_mask_token=F, lsf=0.5, latent+distill+ce

Continuous HPs swept (shared across all configs)
-------------------------------------------------
  lr, weight_decay         log-uniform [1e-5, 1e-3]
  modality_dropout         uniform [0.1, 0.5]
  labeled_frequency        uniform [0.1, 0.5]  (fixed=0 for no-batch-mix)
  lambda_latent            uniform [0, 0.5]
  lambda_prefusion         uniform [0, 1]
  lambda_distill           uniform [0, 1]
  mae_mask_ratio           uniform [0.1, 0.8]

Step 1: Register the 8 sweeps (run once)
-----------------------------------------
  python sweep/create_sweep_final.py           # registers + prints sbatch loops
  python sweep/create_sweep_final.py --dry-run # inspect without registering

Step 2: Submit jobs
--------------------
  The script prints a ready-to-run sbatch loop per sweep, e.g.:
    for i in $(seq 1 32); do sbatch sweep/run_sweep.sh 'entity/project/sweep_id'; done

  Smoke-test a single trial locally (no SLURM):
    wandb agent entity/project/sweep_id --count 1

Output
------
  res/delulu-sweep/sweep_results_benv2_final.csv  (same schema as sweep_results_benv2.csv)

Files
-----
  create_sweep_final.py         register sweeps
  sweep_shot.py                 W&B agent runner (one trial per job)
  run_sweep.sh                  SBATCH wrapper (l40s, 8 CPUs, 64 GB, 3h)
  sweep_yaml/base.yaml          shared lr + weight_decay ranges
  sweep_yaml/sweep_benv2_final.yaml  continuous HP space + command template
  sweep_registry.txt            log of registered sweep IDs
