"""Various methods for analyzing simulations, centered around studying the efficiency of emergent systems over their trajectories."""

import numpy as np
import pandas as pd

from game.game import Game
from rdot import distortions, information, probability
from ultk.effcomm.rate_distortion import ib_optimal_decoder, ib_encoder_to_point
from misc.tools import normalize_rows

from collections import namedtuple
from tqdm import tqdm

##########################################################################
# Loop over all encoders from trajectory and record their
# - complexity,
# - accuracy,
# etc.
##########################################################################

SimulationMeasurement = namedtuple(
    "SimulationMeasurement",
    [
        "trajectory_dataframe",
        "fitted_encoders",
        "final_F_deviation",
    ],
)

def measure_encoders(
    g: Game, 
    trajectory_encoders: dict[str, np.ndarray],
    trajectory_decoders: dict[str, np.ndarray],
    steps_recorded: dict[str, np.ndarray],
    ib_optimal_encoders: np.ndarray,
    ib_optimal_betas: np.ndarray,
    curve_data: pd.DataFrame,
    ) -> SimulationMeasurement:

    # Compute value of IB objective funcion achieved by optima
    curve_points = curve_data[["complexity", "accuracy"]].values
    F_opt = curve_points[:, 0] - ib_optimal_betas * curve_points[:, 1]

    # Setup main container for resulting dataframe
    observations: list[tuple[float]] = []
    fitted_encoders: list[np.ndarray] = [] # NOTE: you better never have run > 0

    # This outer loop is some extra cruft for run_num, which is always 0
    for run_num, (run_key, encoders) in enumerate(trajectory_encoders.items()):

        assert run_key == f"run_{run_num}"

        # key into steps recorded and decoders
        steps_recorded_run = steps_recorded[run_key]
        decoders = trajectory_decoders[run_key]

        # Main measuring loop
        for (encoder, decoder, recorded_step) in tqdm(

            zip(
                encoders,
                decoders,
                steps_recorded_run,
            ),
            desc="measuring trajectory encoders",
            total=len(encoders),            
        ):
            
            ####################################################################
            # Record complexity, accuracy, etc.
            ####################################################################
            
            bayesian_decoder = ib_optimal_decoder(
                encoder, g.prior, g.meaning_dists,
            )

            # IB Coordinates w.r.t. Bayesian receiver
            complexity, accuracy, distortion = ib_encoder_to_point(
                g.prior,
                g.meaning_dists,
                encoder,
                bayesian_decoder,
            )

            # Team MSE and EU w.r.t. emergent receiver
            system = normalize_rows(g.meaning_dists @ encoder @ decoder @ g.meaning_dists)
            # Mean squared error
            mse = distortions.expected_distortion(g.prior, system, g.dist_mat)
            # Expected Utility, relative to discriminative_need
            eu_gamma = np.sum(g.prior * (system * g.utility))

            # Expected KL between emergent receiver and the Bayesian optimal inverse of the Sender.
            # D[ R(\hat{x}_o | w) || S_bayes(\hat{x}_o | w) ], shape `(words, words)`
            kl_vec = information.kl_divergence(
                p=np.where(
                    (encoder * g.prior).T > 0., decoder, 0., # guard against underflows driving KL to infinity. TODO: more principled: implement numerically stable bayes rule in logspace.
                ),
                q=bayesian_decoder,
                axis=1,  # take entropy of meanings, i.e. sum over 2nd axis
                base=2,
            )
            # if np.any(np.isinf(kl_vec)):
                # breakpoint()
            # Take expectation over p(w)
            pw = probability.marginalize(encoder, g.meaning_dists @ g.prior)
            kl_eb = np.sum(pw * kl_vec)


            ####################################################################
            # Fit efficiency loss and gNID
            ####################################################################

            # Find lowest epsilon to any optimal system
            F_em = complexity - ib_optimal_betas * accuracy

            # Do we have F_[q] >= F_[q*] for all q?
            F_em_deviation = (F_em - F_opt) / ib_optimal_betas
            min_ind = np.argmin(F_em_deviation)

            fitted_beta = ib_optimal_betas[min_ind]
            fitted_eps = np.min(F_em_deviation)
            fitted_opt = ib_optimal_encoders[min_ind]

            if fitted_eps < 0:
                # raise Exception(
                import warnings
                warnings.warn(
                    f"A trajectory encoder has negative efficiency loss: epsilon={fitted_eps}, fitted_beta={fitted_beta}, iteration={recorded_step}."
                )

            ####################################################################
            # Record complexity, accuracy, etc.
            ####################################################################
                        

            # Find the gNID of the epsilon-fitted system
            gnids_to_curve = [
                information.gNID(encoder, opt_enc, g.prior) for opt_enc in ib_optimal_encoders
            ]
            optimal_encoder = ib_optimal_encoders[min_ind]
            gnid = information.gNID(encoder, optimal_encoder)

            ####################################################################
            # Append Observation 
            ####################################################################

            iteration = recorded_step

            # aliases
            min_epsilon = fitted_eps
            min_epsilon_beta = fitted_beta

            observation: tuple[float] = (
                run_num,
                iteration,
                complexity,
                accuracy,
                distortion,
                mse,
                eu_gamma,
                kl_eb,
                min_epsilon,
                min_epsilon_beta,
                gnid,
            )
            observations.append(observation)
            fitted_encoders.append(fitted_opt)

    columns = [

        "run_num",        
        "iteration",

        "complexity",
        "accuracy",
        "distortion",
        "mse",
        "eu_gamma",
        "kl_eb",
        "min_epsilon",
        "min_epsilon_beta",
        "min_gnid",
    ]

    trajectory_dataframe = pd.DataFrame(
        data=observations,
        columns=columns,
    )

    measurement = SimulationMeasurement(
        trajectory_dataframe,
        np.stack(fitted_encoders),
        F_em_deviation,
    )

    return measurement



##########################################################################
# Record expected utility of ib optimal encoders w.r.t the specific game
##########################################################################

def get_optimal_encoders_eu(g: Game, optimal_encoders: np.ndarray) -> list[float]:
    return  [
        # Expected Utility, relative to discriminative_need_gamma
        # eu_gamma = sum(prior * (optimal_team * utility))
        np.sum(
            g.prior * (
                g.meaning_dists @ (
                    encoder @ ib_optimal_decoder(
                        encoder, g.prior, g.meaning_dists
                    )
                ) @ g.meaning_dists 
            ) * g.utility
        ) for encoder in tqdm(optimal_encoders, desc="computing eu of optima")
    ]

