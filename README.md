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
"simulation.num_runs=8" \
"simulation.dynamics.population_init_gamma=range(-3,4)" \
"simulation.dynamics.imprecise_imitation_gamma=range(-3, 4)" \
"simulation/dynamics=replicator_diffusion, nowak_krakauer"
```

Description of command line args, in order of appearance:

- `./scripts/run.sh -m `
    - The `-m` flag indicates to hydra that we are performing a 'multirun' sweep over configs.

- The next three overrides specify parameters of both the signaling games, and of the IB theoretical bound (see [https://github.com/nathimel/ibsg/tree/main/src/game/game.py](src/game/game.py)). Note that all overrides can be specified in any order.
- `game.universe=2ball_300`
    - We specify a universe of 300 referents sampled from a unit sphere in 3 dimensions. This universe is loaded from a CSV file at [data/universe/2ball_300.csv](https://github.com/nathimel/ibsg/tree/main/data/universe/2cube_300.csv) folder, so we pass the filename using `"game.universe=2ball_300"` (see the [nballs](https://github.com/nathimel/ibsg/tree/main/src/notebooks/old_analyses/nballs.ipynb) notebook for some visualization). You can encode a universe with any structure you like into a csv file; the default universe is just $\\{1, \dots, 10\\}$.
- `game.prior=2ball_300_power_2`
    - We specify a power-law distributed prior over meanings at [data/prior/2ball_300_power_2.csv](data/prior/2ball_300_power_2.csv) (see the [power_prior](https://github.com/nathimel/ibsg/tree/main/src/notebooks/old_analyses/power_prior.ipynb) notebook for visualizations). Use any prior you like, encoded as a CSV file. If we omit this, a uniform prior will be inferred.
- `game.num_signals=300`
    - We let Sender and Receiver have 300 possible signals for all rounds of the signaling game (thus allowing for perfectly accurate languages).


- `game.discriminative_need_gamma=0`
    - We set the degree of tolerable pragmatic slack / discriminative need in a signaling game to be moderate. This is the one integer parameter for payoff / utility / fitness in the signaling game. It will be the exponent of $10$, i.e., the actual parameter supplied to the utility function is $1$. (see `generate_sim_matrix` at [src/game/perception.py](https://github.com/nathimel/ibsg/tree/main/src/game/perception.py)).

- `simulation.num_runs=8`
    - We simulate evolution eight different times. Since some evolutionary dynamics are nondeterministic, this can be important. By default, these runs are executed in parallel using all available CPU cores. The number of processes to run, and whether to multiprocess, can be overriden. See [conf/simulation/basic.yaml](https://github.com/nathimel/ibsg/tree/main/conf/simulation/basic.yaml).

- The next three overrides ask hydra to *sweep* over different parameters, holding all other parameters equal. Sweeps are performed locally and serially (but see https://hydra.cc/docs/plugins/joblib_launcher/).

- `simulation.dynamics.population_init_gamma=range(-3,4)`
    - We seep over different initial conditions of the initial population of senders and receivers The integers in this list to sweep  (-3, -2, ..., 3) correspond to an exponent of ten for an energy-based initialization (see [random_stochastic_matrix](https://github.com/nathimel/ibsg/tree/main/src/misc/tools.py)). We have therefore now requested that hydra execute 7 jobs, each of them running 8 (runs) simulations.

- `simulation.dynamics.imprecise_imitation_gamma=range(-3, 4)`
    - We sweep over different levels of perceptual/mutation noise in the signaling game dynamics (see [src/game/perception.py](https://github.com/nathimel/ibsg/tree/main/src/game/perception.py)). We have therefore now requested that hydra execute 49 jobs.

- `simulation/dynamics=replicator_diffusion, nowak_krakauer`
    - We sweep over two different dynamics inspired by the replicator equation (see [src/simulation/dynamics.py](https://github.com/nathimel/ibsg/tree/main/src/simulation/dynamics.py)). We now have requested 98 jobs (however, the IB theoretical bound is appropriate for all 98 simulation sweeps, so it is only estimated once).

For each of the 98 jobs, unique folders will be generated and outputs will be written to them under [multirun](https://github.com/nathimel/ibsg/tree/main/multirun/). These folders are hierarchically organized by the parameters described above.

Happy exploring!

</details>

## Requirements  

Step 1. Create the conda environment:

- Get the required packages by running

    `conda env create -f environment.yml`

Step 2. Install ALTK via git:

- Additionally, this project requires [the unnatural language toolkit (ALTK)](https://clmbr.shane.st/ultk/ultk.html). Install it via git with

    `python -m pip install 'git+https://github.com/CLMBRs/ultk.git'`

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

5. Collect the results of all multisweeps into one dataframe

    `python src/get_all_data.py`

    Search recursively for the results of simulation sweeps and IB curve estimations, collect them into dataframes annotated with their appropriate parameters as columns, and write as one (long) dataframe to [all_data](analysis_data/all_data.csv) in tidy data format. This can be used to produce and directly compare plots as in [this notebook](src/notebooks/analyze.ipynb).
    - NOTE: It is important to run this script on the same machine used to run simulations or estimate curves. The reason for this is because this script will look for numpy binaries (e.g., `betas.npy` or `optimal_encoders.npy`) generated during those steps. However, **all numpy binaries are excluded from git history** via the `.gitignore` due to large file limits. So, if you ran experiments using a remote server, you must run this script on the same server if you want perform data analysis on those experiments.

## References

This codebase represents the efforts of an extension of the following paper:

> Imel, N. (2023). "The evolution of efficient compression in signaling games." *Proceedings of the 45th Annual Meeting of the Cognitive Science Society*.

and uses code from the following repositories:

- The above paper's [codebase](https://github.com/nathimel/rdsg/tree/main),
- Noga Zaslavsky's [ib-color-naming](https://github.com/nogazs/ib-color-naming) model,
- Michael Franke and Pedro Correia's [vagueness-games](https://github.com/josepedrocorreia/vagueness-games) simulations.
