__author__ = 'richard'

import logging
from datetime import datetime

from scipy.optimize import basinhopping
import numpy as np

from scripts.pickle_experiments import load_mosquito_kde_data_dicts
import agent


# wrapper func for agent 3D
def fly_wrapper(GUESS, *args):
    """
    :param bias_scale_GUESS:
    :param mass_GUESS:
        mean mosquito mass is 2.88e-6
    :param damping_GUESS:
        estimated damping was 5e-6, # cranked up to get more noise #5e-6,#1e-6,  # 1e-5
    :return:
    """
    print """
    ##############################################################
    Guess: {}
    ##############################################################""".format(GUESS)

    beta, F_rand_strength = GUESS
    windF_strength = 0.
    TEST_CONDITION = None
    K = 0  # no wall force for optimization
    F_STIM_SCALE = 0

    N_TRAJECTORIES, PLOTTER, (EXP_BINS, EXP_VALS) = args
    # import pdb; pdb.set_trace()

    simulation, skeeter = agent.gen_objects_and_fly(
        N_TRAJECTORIES,
        TEST_CONDITION,
        beta,
        F_rand_strength,
        windF_strength,
        F_STIM_SCALE,
        K,
        'downwind_high',
        bounded=False,
        verbose=False
    )

    combined_score, score_components, _ = simulation.calc_score(ref_ensemble=(EXP_BINS, EXP_VALS))

    global HIGH_SCORE
    global BEST_GUESS
    if combined_score < HIGH_SCORE:
        HIGH_SCORE = combined_score
        BEST_GUESS = GUESS
        string = "{0} New high score: {1}. Guess: {2}. Score components = {3}".format(datetime.now(), HIGH_SCORE, GUESS,
                                                                                      score_components)
        print(string)
        logging.info(string)

    # if PLOTTER is True:
    #     error_plotter(ensemble, GUESS, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed)
    #
    # if np.any(np.isinf(dkl_scores)):
    #     error_plotter(ensemble, GUESS, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed)

    return combined_score


# def error_plotter(ensemble, guess, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed):
#
#
#
#     print "{} New high score: {}. Guess: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, guess, dkl_scores, acf_score)
#     logging.info("{} New high score: {}. Guess: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, guess, dkl_scores, acf_score))
#
#     plt.ioff()
#     f, axarr = plt.subplots(2, sharex=False)
#     axarr[0].plot(aabs_bins[:-1], a_counts_n, label='RoboSkeeter')
#     axarr[0].plot(aabs_bins[:-1], a_observed, c='r', label='experiment')
#     axarr[0].set_title('accel score=> {}. High score = DKL(a)+DKL(v) = {}'.format(dkl_scores[1], HIGH_SCORE))
#     axarr[0].legend()
#
#     axarr[1].plot(vabs_bins[:-1], v_counts_n, label='RoboSkeeter')
#     axarr[1].plot(vabs_bins[:-1], v_observed, c='r', label='experiment')
#     axarr[1].set_title('velocity score=> {}. High score = DKL(a)+DKL(v) = {}'.format(dkl_scores[0], HIGH_SCORE))
#     axarr[1].legend()
#     plt.suptitle('Parameter Guess: {}'.format(guess))
#     plt.show()
#     plt.close()


class MyBounds(object):
    def __init__(self, xmax=[1e-4, 1e-4], xmin=[1e-7, 1e-7]):  # , 1e08]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def main(x_0=None):
    logging.basicConfig(filename='basin_hopping.log', level=logging.DEBUG)

    global HIGH_SCORE
    HIGH_SCORE = 1e10

    global BEST_GUESS
    BEST_GUESS = None

    OPTIM_ALGORITHM = 'SLSQP'
    PLOTTER = False
    # ACF_THRESH = 0.5
    GUESS_PARAMS = "[beta, F_rand_strength]"  #, F_WIND_SCALE]"  # [5e-6, 4.12405e-6, 5e-7]
    if x_0 is None:
        INITIAL_GUESS = [3.63674551e-07, 6.55599224e-06]
    else:
        INITIAL_GUESS = x_0
    N_TRAJECTORIES = 100

    print "Starting optimizer."

    logging.info("""\n ############################################################
        ############################################################
        {} Start optimization with {} algorithm
        # trajectories = {}
        Params = {}
        Initial Guess = {}
        ############################################################""".format(
        datetime.now(), OPTIM_ALGORITHM, N_TRAJECTORIES, GUESS_PARAMS, INITIAL_GUESS))

    mybounds = MyBounds()

    result = basinhopping(
        fly_wrapper,
        INITIAL_GUESS,
        stepsize=1e-6,
        T=1e-5,
        niter=100,  # number of basin hopping iterations, default 100
        niter_success=20,  # Stop the run if the global minimum candidate remains the same for this number of iterations
        minimizer_kwargs={
            "args": (N_TRAJECTORIES, PLOTTER, (load_mosquito_kde_data_dicts())),
            'method': OPTIM_ALGORITHM,
            "bounds": [(5e-7, 1e-3), (5e-7, 1e-2)],
            "tol": 0.02  # tolerance for considering a basin minimized, set to about the difference between re-scoring
            # same simulation
        },
        disp=True,
        accept_test=mybounds
    )

    return BEST_GUESS, HIGH_SCORE, result


if __name__ == '__main__':
    BEST_GUESS, HIGH_SCORE, result = main()

    msg = """############################################################
    Trial end. FINAL GUESS: {0}
    SCORE: {1}
     {2}
    ############################################################""".format(BEST_GUESS, HIGH_SCORE, result)

    logging.info(msg)
    print msg
