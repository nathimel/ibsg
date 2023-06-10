#!/bin/sh

python src/curve.py "$@"

python src/run_simulations.py "$@"

python src/plot.py "$@"
