#!/bin/bash
#SBATCH --time=0:20:00
#SBATCH --account=aip-gpleiss
#SBATCH --output=logs/splitcount/%j.out
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
source sh/env.sh
mkdir -p logs/splitcount
python -u -c "
import tacoreader, os
root='datasets/geoben2/biomassters'
paths=[f'geobench_biomassters.{i:04d}.part.tortilla' for i in range(3)]
df=tacoreader.load([os.path.join(root,p) for p in paths])
print('TOTAL ROWS:', len(df))
print(df['tortilla:data_split'].value_counts())
import geobench_v2; print('geobench_v2 version:', getattr(geobench_v2,'__version__','unknown'))
"
