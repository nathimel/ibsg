import sys
import omegaconf

import pandas as pd

from misc import util
from game.game import Game
from game.perception import generate_confusion_matrix
from altk.effcomm.util import joint, H

from pathlib import Path
from tqdm import tqdm

# We don't use the hydra.compose api, since we can't use sweeps with that anyways. Instead, we literally build a giant dataframe of all outputs in multirun.

def main():

    if len(sys.argv) != 2:
        print("Usage: python src/get_all_data.py. PATH_TO_ALL_DATA \nThis script does not use hydra; do not pass overrides.")
        sys.exit(1)

    # Where to save the giant dataframe
    save_fn = sys.argv[1]

    # helper
    concat = lambda list_of_dfs: pd.concat(list_of_dfs, axis=0, ignore_index=True)

    config_fp = "conf/config.yaml"
    cfg = omegaconf.OmegaConf.load(config_fp)

    # Assume we're interested in multisweeps
    root_dir = cfg.filepaths.hydra_sweep_root
    game_fn = cfg.filepaths.game_fn
    curve_metadata_fn = cfg.filepaths.curve_metadata_save_fn
    leaf_hydra_cfg_fn = ".hydra/config.yaml"

    # Collect all the results of individual simulations
    simulation_results = []
    print(f"collecting all simulation data from {root_dir}.")
    game_fns = list(Path(root_dir).rglob(game_fn))
    for path in tqdm(game_fns):

        parent = path.parent.absolute()

        # Load full config w/ overrides
        leaf_cfg = omegaconf.OmegaConf.load(parent / leaf_hydra_cfg_fn)

        # Create dataframes
        df_sim = pd.read_csv(parent / cfg.filepaths.simulation_points_save_fn)
        df_sim["point_type"] = "simulation"

        df_traj = pd.read_csv(parent / cfg.filepaths.trajectory_points_save_fn) # may need to check if exists
        df_traj["point_type"] = "trajectory"

        df_nearopt = pd.read_csv(parent / cfg.filepaths.nearest_optimal_points_save_fn)
        df_nearopt["point_type"] = "nearest_optimal"

        # Concat
        df = concat([df_sim, df_traj, df_nearopt])
        
        # Annotate with metadata
        df["universe"] = leaf_cfg.game.universe
        df["prior"] = leaf_cfg.game.prior
        df["num_signals"] = leaf_cfg.game.num_signals
        df["distance"] = leaf_cfg.game.distance
        df["discriminative_need_gamma"] = leaf_cfg.game.discriminative_need_gamma
        df["meaning_dist_gamma"] = leaf_cfg.game.meaning_dist_gamma

        df["dynamics"] = leaf_cfg.simulation.dynamics.name
        df["imprecise_imitation_gamma"] = leaf_cfg.simulation.dynamics.imprecise_imitation_gamma
        df["population_init_gamma"] = leaf_cfg.simulation.dynamics.population_init_gamma
        df["seed"] = leaf_cfg.seed

        # NOTE: this is slow.
        # conditional entropy of confusion distributions (imprecision)
        # H(Y|X) = H(X,Y) - H(X)
        confusion = generate_confusion_matrix(leaf_game.universe, leaf_cfg.simulation.dynamics.imprecise_imitation_gamma, leaf_game.dist_mat)
        pX = leaf_game.prior.numpy()        
        pXY = joint(confusion, pX)
        df["noise_cond_ent"] = H(pXY) - H(pX)

        df["ib_bound_function"] = None  # dummy value since all simulations are curve agnostic.

        simulation_results.append(df)
    
    # Collect all curves
    print(f"collecting all curve data from {root_dir}.")
    curves = []
    curve_fns = list(Path(root_dir).rglob(curve_metadata_fn))
    for path in tqdm(curve_fns):

        parent = path.parent.absolute()

        # load mdetadata
        curve_metadata = omegaconf.OmegaConf.load(parent / curve_metadata_fn)

        # Create dataframe
        df_ib = pd.read_csv(parent / cfg.filepaths.curve_points_save_fn)
        df_ib["point_type"] = "ib_bound"
        df_mse = pd.read_csv(parent / cfg.filepaths.mse_curve_points_save_fn)
        df_mse["point_type"] = "mse_bound"
        df = concat([df_ib, df_mse])

        # Annotate
        df["universe"] = curve_metadata.universe
        df["prior"] = curve_metadata.prior
        df["num_signals"] = curve_metadata.num_signals # this is a bit spurious
        df["distance"] = curve_metadata.distance
        df["meaning_dist_gamma"] = curve_metadata.meaning_dist_gamma
        df["ib_bound_function"] = curve_metadata.ib_bound_function
        # note that we aren't checking for seed, because we don't save diff curves based on random seed. maybe we should, but not for now.

        curves.append(df)

    # Concat all
    df_sims = concat(simulation_results)
    df_curves = concat(curves)
    df = concat([df_sims, df_curves])

    # Save
    df.to_csv(save_fn, index=False)

if __name__ == "__main__":
    main()