#!/bin/sh

# load conda and activate environment
# n.b.: deactivate any current conda env before slurm submission.
module load miniconda3/4.8.5
conda init bash
source /opt/apps/miniconda3/4.8.5/etc/profile.d/conda.sh
conda activate ibsg

# run programs
echo 'python src/setup.py "$@"'
python src/setup.py "$@"
