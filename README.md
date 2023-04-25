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

    `python -m pip install git+https://github.com/nathimel/altk.git`

## Replicating the experimental results

The main experimental results can be reproduced by running `./scripts/run.sh`.

This will perform four basic steps by running the following scripts:

1. Simulate evolution:

    `python3 src/run_simulations.py`

    Run one or more trials of an evolutionary dynamics simulation on a sim-max game, and save the resulting (complexity, distortion) points to a csv.

    - This script will also generate and save hypothetical variants of the emergent systems for comparison.

2. Estimate Pareto frontier

    `python3 src/curve.py`

    Estimate the IB curve

4. Get a basic plot

    `python3 src/plot.py`

    Produce a basic plot of the emergent systems on the information plane.

    Code for the more detailed plots can be found in [notebooks/single_figures.ipynb](src/notebooks/paper_figures.ipynb).
