#!/bin/sh

python src/setup.py "$@" # optional pre-experiment checks

python src/curve.py "$@"

python src/run_simulations.py "$@"

python src/measure.py "$@"

python src/plot.py "$@"
