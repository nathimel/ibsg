# ibsg: an Information Bottleneck analysis of sim-max signaling games

This repository contains code for constructing sim-max games, simulating  dynamics, and quantifying the evolved signaling languages' efficiency w.r.t the Information Bottleneck bound.

## Replicating

The full experiment involves fairly extensive parameter sweeping. This codebase is integrated with a [SLURM](https://slurm.schedmd.com/quickstart.html) job scheduler to efficiency run many simulations in parallel. The top-level script to run the full experiment is in [hpc/scripts/array.sub](hpc/scripts/array.sub). This reads over 800 parameter combinations specified in [hpc/configs/sim.txt](hpc/configs/sim.txt) and submits one job per line.

To run with different parameter configurations, you can use the [hpc/misc/helper.py] script to automatically write to the lines of a config file with the combinations you want.

If you want to explore running locally (e.g., for just a few iterations of a dynamics and only a few parameter combinations), read the sections below.

## Setting up an experiment

This codebase uses [hydra](https://hydra.cc/) to organize configurations and outputs:

- The [conf](./conf/) folder contains the main `config.yaml` file, which can be overriden with additional YAML files or command-line arguments.

- Running the shell script [scripts/run.sh](scripts/run.sh) will generate folders and files in [outputs](outputs), where a `.hydra` folder will be found with a `config.yaml` file. Reference this file as an exhaustive list of config fields to override.

<details>
<summary>Example</summary>
<br>

Here is an example command that will execute an experiment, overriding the hydra config defaults.

```    
TODO
```

Description of command line args, in order of appearance:

- `./scripts/run.sh -m `
    - The `-m` flag indicates to hydra that we are performing a 'multirun' sweep over configs.


For each of the N jobs, unique folders will be generated and outputs will be written to them under [multirun](https://github.com/nathimel/ibsg/tree/main/multirun/). These folders are hierarchically organized by the parameters described above.


</details>

## Requirements  

Step 1. Create the conda environment:

- Get the required packages by running

    `conda env create -f environment.yml`

Step 2. Install ULTK via git:

- Additionally, this project requires [the unnatural language toolkit (ULTK)](https://clmbr.shane.st/ultk/ultk.html). Install it via git with

    `python -m pip install 'git+https://github.com/CLMBRs/ultk.git'`

## Replicating the experimental results

The main experimental results can be reproduced by running `./scripts/run.sh`.

This will perform four basic steps by running the following scripts (with the appropriate config overrides):

1. Compute the IB curve

    `python src/curve.py`

2. Simulate evolution

    `python src/run_simulations.py`

    Run one or more trials of an evolutionary dynamics simulation on a sim-max game, and save the resulting data.

3. Measure efficiency and other data of trajectories

    `python src/measure.py`

    Measure the optimality of emergent systems w.r.t the IB  bound, record most similar theoretically optimal systems, and save resulting data.
    Optionally:

4. Collect the results of all multisweeps into one dataframe

    `python src/get_all_data.py`

    Search recursively for the results of simulation sweeps and IB curve estimations, collect them into dataframes annotated with their appropriate parameters as columns, and write as one (long) dataframe to [all_data](analysis_data/all_data.csv) in tidy data format. This can be used to produce plots as in [this notebook](src/notebooks/journal.ipynb).
    - NOTE: It is important to run this script on the same machine used to run simulations or estimate curves. The reason for this is because this script will look for numpy binaries (e.g., `betas.npy` or `optimal_encoders.npy`) generated during those steps. However, **all numpy binaries are excluded from git history** via the `.gitignore` due to large file limits. So, if you ran experiments using a remote server, you must run this script on the same server if you want perform data analysis on those experiments.

## References

This codebase represents the efforts of an extension of the following paper:

> Imel, N. (2023). "The evolution of efficient compression in signaling games." *Proceedings of the 45th Annual Meeting of the Cognitive Science Society*.

and uses code from the following repositories:

- The above paper's [codebase](https://github.com/nathimel/rdsg/tree/main),
- Noga Zaslavsky's [ib-color-naming](https://github.com/nogazs/ib-color-naming) model,
- Michael Franke and Pedro Correia's [vagueness-games](https://github.com/josepedrocorreia/vagueness-games) simulations.
