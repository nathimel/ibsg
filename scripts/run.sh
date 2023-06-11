#!/bin/sh

python src/curve.py "$@"

python src/run_simulations.py "$@"

python src/efficiency.py "$@"

python src/plot.py "$@"
