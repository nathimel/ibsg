#!/bin/sh

python src/setup.py "$@"

python src/curve.py "$@"

python src/run_simulations.py "$@"

python src/finite_sample.py "$@"

python src/efficiency.py "$@"

python src/plot.py "$@"
