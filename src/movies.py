"""Collect all explored points, including those from simulation and the IB curve and get simple plot."""

import os
import hydra
import pandas as pd
import numpy as np
from misc import util, vis
from game.game import Game
from plot import generate_encoder_plots

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(config):
    util.set_seed(config.seed)

    # helper function to load the right files
    cwd = os.getcwd()
    fps = config.filepaths
    fullpath = lambda fn: os.path.join(cwd, fn)

    g = Game.from_hydra(config)

    moviepath = lambda run, fn: os.path.join(cwd, fps.trajectory_movies_dir, f"run_{run}", fn)

    trajectory_encoders: dict[str, np.ndarray] = np.load(fps.trajectory_encoders_save_fn)

    encoders_data_run = ...

    for run_num, (run_key, encoders) in enumerate(trajectory_encoders.items()):
        assert run_key == f"run_{run_num}"        
        run = run_num + 1 # to be consistent with other indexing
        
        # some hacky cleanup with this since assumes over runs
        # encoders_data_run = util.encoders_to_df(encoders, col="round")
        
        # Generate all pngs (both tiles and lines) for movie
        if config.plotting.generate_movie_plots:
            generate_encoder_plots(
                encoders,
                g.prior,
                faceted_lines_fn=None,
                faceted_tiles_fn=None,
                lines_dir=moviepath(run, fps.encoder_line_plots_dir),
                tiles_dir=moviepath(run, fps.encoder_tile_plots_dir),
                individual_file_prefix="round",
            )
        else:
            print("Skipping movie plot generation.")

        if config.plotting.generate_movies:
            movie_dir = f"movies/run_{run}"

            tiles_mp4_fn = os.path.join(movie_dir, "tiles.mp4")
            # Generate movies with ffmpeg
            os.system(f"ffmpeg -f image2 -framerate 10 -i movies/run_{run}/encoder_tile_plots/round_%d.png -vcodec mpeg4 -y {tiles_mp4_fn}")

            lines_mp4_fn = os.path.join(movie_dir, "lines.mp4")
            os.system(f"ffmpeg -f image2 -framerate 10 -i movies/run_{run}/encoder_line_plots/round_%d.png -vcodec mpeg4 -y {lines_mp4_fn}")

            print(f"Generated movie at {os.path.join(os.getcwd(), tiles_mp4_fn)}.")
            print(f"Generated movie at {os.path.join(os.getcwd(), lines_mp4_fn)}.")
        else:
            print("Skipping movie generation.")

if __name__ == "__main__":
    main()
