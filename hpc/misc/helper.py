"""Simple script to automatically generate config files that we can use for slurm. Ideally we'd integrate with https://hydra.cc/docs/plugins/submitit_launcher/."""

import sys
import numpy as np

out_fn = "hpc/configs/sim.txt"


overrides = [
    "game.meaning_dist_pi=0.5",
    "simulation/dynamics=imprecise_conditional_imitation",
    "simulation.dynamics.imprecise_imitation_alpha=0.5",
    "game.discriminative_need_gamma=REPLACE",
    "game.universe=100",
    "game.num_signals=100",
    "simulation.dynamics.population_init_tau=null",
    "seed=REPLACE",
    "simulation.dynamics.max_its=100000.0",
    "simulation.dynamics.threshold=0.0001",
    "simulation.multiprocessing=False",
]

# Configs to manipulate ... we could use sys argv
##############################################################################
gammas = np.logspace(-10, 1, 100)
seeds = range(8)
##############################################################################

def main():

    base_overrides_str = " ".join(["-m"] + overrides)
    overrides_strings = []
    # generate config lines
    for gamma in gammas:
        for seed in seeds:

            override_str = base_overrides_str.replace(
                "seed=REPLACE", 
                f"seed={seed}",
            ).replace(
                "game.discriminative_need_gamma=REPLACE", 
                f"game.discriminative_need_gamma={gamma}",
            )

            overrides_strings.append(override_str)
    
    file_str = "\n".join(overrides_strings) + "\n"

    with open(out_fn, "w") as f:
        f.write(file_str)
    
    print(f"Wrote {len(overrides_strings)} lines to {out_fn}")

if __name__ == "__main__":
    main()
