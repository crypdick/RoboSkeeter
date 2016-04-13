__author__ = 'richard'

import logging
from datetime import datetime

from scipy.optimize import minimize_scalar

from roboskeeter.experiment import start_simulation
from roboskeeter.scripts import load_mosquito_kde_data_dicts


# wrapper func for agent 3D
def fly_wrapper(BOUNCE_COEFF, *args):
    """
    :param bias_scale_BOUNCE_COEFF:
    :param mass_BOUNCE_COEFF:
        mean mosquito mass is 2.88e-6
    :param damping_BOUNCE_COEFF:
        estimated damping was 5e-6, # cranked up to get more noise #5e-6,#1e-6,  # 1e-5
    :return:
    """
    print """
    ####a##########################################################
    BOUNCE_COEFF: {}
    ##############################################################""".format(BOUNCE_COEFF)
    N_TRAJECTORIES = 100

    experiment_kwargs = {'condition': 'Control',
                         'time_max': 6.,
                         'bounded': True,
                         'number_trajectories': N_TRAJECTORIES
                         }

    agent_kwargs = {'random_f_strength': 6.55599224e-06,
                    'stim_f_strength': 0.,
                    'damping_coeff': 3.63674551e-07,
                    'collision_type': 'part_elastic',
                    'restitution_coeff': BOUNCE_COEFF,
                    'stimulus_memory': 100,
                    'decision_policy': 'cast_only',  # 'surge_only', 'cast_only', 'cast+surge', 'ignore'
                    'initial_position_selection': 'downwind_high',
                    'verbose': False
                    }

    (EXP_BINS, EXP_VALS) = args

    simulation, trajectory_s, windtunnel, plume, agent = start_simulation(agent_kwargs, experiment_kwargs)

    combined_score, score_components, _ = trajectory_s.calc_score(ref_ensemble=(EXP_BINS, EXP_VALS))

    global HIGH_SCORE
    global BEST_BOUNCE_COEFF
    if combined_score < HIGH_SCORE:
        HIGH_SCORE = combined_score
        BEST_BOUNCE_COEFF = BOUNCE_COEFF
        string = "{0} New high score: {1}. BOUNCE_COEFF: {2}. Score components = {3}".format(datetime.now(), HIGH_SCORE,
                                                                                             BOUNCE_COEFF,
                                                                                             score_components)
        print(string)
        logging.info(string)

    return combined_score




def main(x_0=None):
    logging.basicConfig(filename='optimize_restitution_coeff.log', level=logging.DEBUG)

    global HIGH_SCORE
    HIGH_SCORE = 1e10

    global BEST_BOUNCE_COEFF
    BEST_BOUNCE_COEFF = None

    print "Starting optimizer."


    result = minimize_scalar(
        fly_wrapper,
        bounds=(0., 1.),
        method='bounded',
        args=(load_mosquito_kde_data_dicts())
    )

    return BEST_BOUNCE_COEFF, HIGH_SCORE, result


if __name__ == '__main__':
    BEST_BOUNCE_COEFF, HIGH_SCORE, result = main()

    msg = """############################################################
    Trial end. FINAL BOUNCE_COEFF: {0}
    SCORE: {1}
     {2}
    ############################################################""".format(BEST_BOUNCE_COEFF, HIGH_SCORE, result)

    logging.info(msg)
    print msg
