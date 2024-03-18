# temporary notes

The long command that I have used to sweep over conditions is the following:

`./scripts/run.sh -m  "simulation/dynamics=imprecise_conditional_imitation" "game.discriminative_need_gamma=range(-3,4)" "game.universe=100" "game.num_signals=100" "simulation.dynamics.population_init_gamma=range(-3,4)" "seed=range(10)" "simulation.dynamics.max_its=1000"`

which produces 490 jobs.


## Bugs

- why is the efficiency loss negative? Shouldn't it be guaranteed to be positive?
  - isn't it guaranteed that F_lang >= F_optimal for any lang and any optimal encoder?

- line plot showing probability of signals with x-axis the meaning space.