#!/bin/sh

# load conda and activate environment
module load miniconda3/4.8.5
conda init bash
source /opt/apps/miniconda3/4.8.5/etc/profile.d/conda.sh
conda activate ibsg

# run programs
echo
echo "python src/setup.py $@"
python src/setup.py "$@"

# echo
# echo "python src/curve.py $@"
# python src/curve.py "$@"

echo
echo "src/run_simulations.py $@"
python src/run_simulations.py "$@"

# echo 
# echo "src/measure.py $@"
# python src/measure.py "$@"

# echo
# echo "src/plot.py $@"
# python src/plot.py "$@"

# echo 
# echo "src/movies.py $@"
# python src/movies.py "$@"

# echo 
# echo "src/get_all_data.py analysis_data/all_data.csv"
# python src/get_all_data.py analysis_data/all_data.csv # this will overwrite with new data, but not remove old data
