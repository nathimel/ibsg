#!/bin/sh

python src/setup.py "$@"

python src/curve.py "$@"

python src/run_simulations.py "$@"

python src/measure.py "$@"

python src/plot.py "$@"

python src/get_all_data.py analysis_data/all_data.csv # this will overwrite with new data, but not remove old data
