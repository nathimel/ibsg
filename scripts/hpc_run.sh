#!/bin/sh

# load conda and activate environment
module load miniconda3/4.8.5
conda init bash
source /opt/apps/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate ibsg

echo 'python src/run_simulations.py "$@"'
python src/run_simulations.py "$@"

echo 'python src/curve.py "$@"'
python src/curve.py "$@"

echo 'python src/plot.py "$@"'
python src/plot.py "$@"

