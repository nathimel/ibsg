#!/bin/sh

# load conda and activate environment
module load miniconda3/4.8.5
conda init bash
source /opt/apps/miniconda3/4.12.0/etc/profile.d/conda.sh
conda activate ibsg

echo 'python src/setup.py "$@"'
python src/setup.py "$@"

echo 'python src/curve.py "$@"'
python src/curve.py "$@"

echo 'python src/run_simulations.py "$@"'
python src/run_simulations.py "$@"

echo 'python src/measure.py "$@"'
python src/measure.py "$@"

# echo 'python src/plot.py "$@"'
# python src/plot.py "$@"

echo 'python src/get_all_data.py analysis_data/all_data.csv'
python src/get_all_data.py analysis_data/all_data.csv # this will overwrite with new data, but not remove old data
