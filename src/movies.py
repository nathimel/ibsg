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

    moviepath = lambda run, fn: os.path.join(
        cwd, fps.trajectory_movies_dir, f"run_{run}", fn
    )

    trajectory_encoders: dict[str, np.ndarray] = np.load(
        fps.trajectory_encoders_save_fn
    )
    steps_recorded: dict[str, np.ndarray] = np.load(fps.steps_recorded_save_fn)

    # Maybe i should just completely refactor the saving logic :/
    for run_num, (run_key, encoders) in enumerate(trajectory_encoders.items()):
        assert run_key == f"run_{run_num}"
        run = run_num + 1  # to be consistent with other indexing

        # key into steps recorded
        steps_recorded_run = steps_recorded[run_key]  # pass this into plots

        # some hacky cleanup with this since assumes over runs
        # encoders_data_run = util.encoders_to_df(encoders, col="round")

        # Generate all pngs (both tiles and lines) for movie
        movie_dir = f"movies/run_{run}"

        if config.plotting.generate_movie_plots:
            # Clear out previously generated images

            if os.path.exists(movie_dir) and os.listdir(movie_dir):
                print(f"Removing all .pngs at {os.path.join(os.getcwd(), movie_dir)}.")

                os.system(f"rm -rf {movie_dir}/*")

            generate_encoder_plots(
                encoders,
                g.prior,
                faceted_lines_fn=None,
                faceted_tiles_fn=None,
                lines_dir=moviepath(run, fps.encoder_line_plots_dir),
                tiles_dir=moviepath(run, fps.encoder_tile_plots_dir),
                individual_file_prefix="step_record",
                title_nums=steps_recorded_run,
            )
        else:
            print("Skipping movie plot generation.")

        if config.plotting.generate_movies:
            # tiles_mp4_fn = os.path.join(movie_dir, "tiles.mp4")
            # Generate movies with ffmpeg
            # os.system(f"ffmpeg -f image2 -framerate 10 -i movies/run_{run}/encoder_tile_plots/round_%d.png -vcodec mpeg4 -y {tiles_mp4_fn}")

            lines_mp4_fn = os.path.join(movie_dir, "lines.mp4")
            os.system(
                f"ffmpeg -f image2 -framerate 10 -i movies/run_{run}/encoder_line_plots/step_record_%d.png -vcodec mpeg4 -y {lines_mp4_fn}"
            )

            # print(f"Generated movie at {os.path.join(os.getcwd(), tiles_mp4_fn)}.")
            print(f"Generated movie at {os.path.join(os.getcwd(), lines_mp4_fn)}.")
        else:
            print("Skipping movie generation.")


if __name__ == "__main__":
    main()
