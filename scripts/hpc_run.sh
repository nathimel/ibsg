#!/bin/sh

# load conda and activate environment
module load miniconda3/4.8.5
conda init bash
source /opt/apps/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate ibsg

echo 'python src/scripts/run_simulations.py "$@"'
python src/scripts/run_simulations.py "$@"

echo 'python src/scripts/curve.py "$@"'
python src/scripts/curve.py "$@"

echo 'python src/scripts/plot.py "$@"'
python src/scripts/plot.py "$@"

