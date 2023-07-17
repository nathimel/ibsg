# ibsg: an Information Bottleneck analysis of sim-max signaling games

This repository contains code for constructing sim-max games, simulating  dynamics, and quantifying the evolved signaling languages' efficiency w.r.t the Information Bottleneck bound.

The codebase is organized around the following steps of the experiment.

## Setting up an experiment

This codebase uses [hydra](https://hydra.cc/) to organize configurations and outputs:

- The [conf](./conf/) folder contains the main `config.yaml` file, which can be overriden with additional YAML files or command-line arguments.

- Running the shell script [scripts/run.sh](scripts/run.sh) will generate folders and files in [outputs](outputs), where a `.hydra` folder will be found with a `config.yaml` file. Reference this file as an exhaustive list of config fields to override.

## Requirements  

Step 1. Create the conda environment:

- Get the required packages by running

    `conda env create -f environment.yml`

Step 2. Install ALTK via git:

- Additionally, this project requires [the artificial language toolkit (ALTK)](https://github.com/nathimel/altk). Install it via git with

    `python -m pip install 'git+https://github.com/CLMBRs/altk.git'`

## Replicating the experimental results

The main experimental results can be reproduced by running `./scripts/run.sh`.

This will perform four basic steps by running the following scripts:

1. Estimate the information curve

    `python src/curve.py`

    Estimate the IB curve

2. Simulate evolution

    `python src/run_simulations.py`

    Run one or more trials of an evolutionary dynamics simulation on a sim-max game, and save the resulting data.

    - This script can also generate and save hypothetical variants of the emergent systems for comparison.

3. Measure efficiency

    `python src/measure.py`

    Measure the optimality of emergent systems w.r.t the IB  bound, record most similar theoretically optimal systems, and save resulting data.
    - Optionally: approximate each emergent sender (encoder) distribution via limited sampling of its behavior. This is an exploratory simulation of how real data collection might skew efficiency measurements.

4. Get a basic plot

    `python src/plot.py`

    Produce a basic plot of the emergent systems on the information plane.

    - This script technically requires steps 3,4
    - Code for the more detailed plots can be found in [notebooks/paper_figures.ipynb](src/notebooks/paper_figures.ipynb).
