# ibsg: an Information Bottleneck analysis of sim-max signaling games

This repository contains code for constructing sim-max games, simulating  dynamics, and quantifying the evolved signaling languages' efficiency w.r.t the Information Bottleneck bound.


## Setting up an experiment

This codebase uses [hydra](https://hydra.cc/) to organize configurations and outputs:

- The [conf](./conf/) folder contains the main `config.yaml` file, which can be overriden with additional YAML files or command-line arguments.

- Running the shell script [scripts/run.sh](scripts/run.sh) will generate folders and files in [outputs](outputs), where a `.hydra` folder will be found with a `config.yaml` file. Reference this file as an exhaustive list of config fields to override.

<details>
<summary>Example</summary>
<br>

Here is an example command that will execute an experiment, overriding the hydra config defaults.

```    
./scripts/run.sh -m \
"game.universe=2ball_300" \
"game.prior=2ball_300_power_2" \
"game.num_signals=300" \
"game.discriminative_need_gamma=0" \
"simulation.num_runs=10" \
"simulation.dynamics.imprecise_imitation_gamma=range(-3, 4)" \
"simulation/dynamics=two_population_rd, nowak_krakauer"
```

Description of command line args, in order of appearance:

- `./scripts/run.sh -m `
    - The `-m` flag indicates to hydra that we are performing a 'multirun' sweep over configs.

- The next three commands specify parameters of both the signaling games, and of the IB theoretical bound (see [src/game/game.py](src/game/game.py)).
- `game.universe=2ball_300`
    - We specify a universe of 300 referents sampled from a unit sphere in 3 dimensions. This universe is loaded from a CSV file at [data/universe/2ball_300.csv](data/universe/2cube_300.csv) folder, so we pass the filename using `"game.universe=2ball_300"` (see the [nballs](src/notebooks/nballs.ipynb) notebook). You can encode a universe with any structure you like into a csv file; the default universe is just $\{1, \dots, 10\}$.
- `game.prior=2ball_300_power_2`
    - We specify a power-law distributed prior over meanings at [data/prior/2ball_300_power_2.csv](data/prior/2ball_300_power_2.csv) (see the [power_prior](src/notebooks/power_prior.ipynb) notebook). Use any prior you like, encoded as a CSV file. If we omit this, a uniform prior will be inferred.
- `game.num_signals=300`
    - We let Sender and Receiver have 300 possible signals for all rounds of the signaling game (thus allowing for perfectly accurate languages).


- `game.discriminative_need_gamma=0`
    - We set the degree of tolerable pragmatic slack / discriminative need in a signaling game to be moderate. This is the one parameter for payoff / utility / fitness in the signaling game. It will be exponentiated by 10 (see `generate_sim_matrix` at [src/game/perception.py](src/game/perception.py)).

- `simulation.num_runs=10`
    - We simulate evolution ten different times, varying initial conditions. By default, these runs are executed in parallel using all available CPU cores. The number of processes to run, and whether to multiprocess, can be overriden. See [conf/simulation/basic.yaml](conf/simulation/basic.yaml). We vary the entropy of the initial populations' average behavior. The range of this variation can be specified by `population_init_minalpha` and `population_init_maxalpha`.

- The next two arguments ask hydra to *sweep* over different parameters, holding all other parameters equal. Sweeps are performed locally and serially (but see https://hydra.cc/docs/plugins/joblib_launcher/).

- `simulation.dynamics.imprecise_imitation_gamma=range(-3, 4)`
    - We sweep over different levels (-3, -2, ..., 3) of perceptual/mutation noise in the signaling game dynamics (see [src/game/perception.py](src/game/perception.py)). We have therefore now requested that hydra execute 7 jobs.

- `simulation/dynamics=two_population_rd, nowak_krakauer`
    - We sweep over two different dynamics inspired by the replicator equation (see [src/simulation/dynamics.py](src/simulation/dynamics.py)). We now have requested 14 jobs (however, the IB theoretical bound is appropriate for all 14 simulation sweeps, so it is only estimated once).

For each of the 14 jobs, outputs will be written to folders that are unique to that job, under [multirun](multirun/), but hierarchically organized as appropriate. For example, one job will output results to

- `multirun/universe=2ball_300/num_signals=300/prior=2ball_300_power_2/dist=squared_dist/meaning_certainty=0/dynamics=two_population_rd/ii=-3/population_size=None/num_runs=10/seed=42/discr_need=0/`

while the IB curve and optimal encoders will be written once to

- `multirun/universe=2ball_300/num_signals=300/prior=2ball_300_power_2/dist=squared_dist/meaning_certainty=0/`

Happy exploring!

</details>

## Requirements  

Step 1. Create the conda environment:

- Get the required packages by running

    `conda env create -f environment.yml`

Step 2. Install ALTK via git:

- Additionally, this project requires [the artificial language toolkit (ALTK)](https://github.com/nathimel/altk). Install it via git with

    `python -m pip install 'git+https://github.com/CLMBRs/altk.git'`

## Replicating the experimental results

The main experimental results can be reproduced by running `./scripts/run.sh`.

This will perform four basic steps by running the following scripts (with the appropriate config overrides):

1. Estimate the information curve

    `python src/curve.py`

    Estimate the appropriate IB curve.

2. Simulate evolution

    `python src/run_simulations.py`

    Run one or more trials of an evolutionary dynamics simulation on a sim-max game, and save the resulting data.

3. Measure efficiency

    `python src/measure.py`

    Measure the optimality of emergent systems w.r.t the IB  bound, record most similar theoretically optimal systems, and save resulting data.
    Optionally:
    - approximate each emergent sender (encoder) distribution via limited sampling of its behavior. This is an exploratory simulation of how real data collection might skew efficiency measurements.
    - generate and save hypothetical variants of the emergent systems for comparison.

4. Visualize results

    `python src/plot.py`

    Produce plots of the emergent systems on the information plane and compare the distributions of each emergent encoder to different variants (e.g. its nearest optimal, sample-approximated, and rotated variant encoders).

    - Code for the more detailed plots can be found in [notebooks/paper_figures.ipynb](src/notebooks/paper_figures.ipynb).

## Citation

To cite this work, please use the following:

```
@unpublished{Imel2023,
author    = {Imel, Nathaniel, and Franke, Michael, and Futrell, Richard},
title     = {Evolutionary dynamics lead to the emergence of efficiently compressed meaning systems},
year      = {2023},
}```
