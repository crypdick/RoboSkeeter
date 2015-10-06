__author__ = 'richard'

from scipy.optimize import basinhopping
import logging
from datetime import datetime
import numpy as np

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

    beta, F_rand_strength, windF_strength = GUESS
    TEST_CONDITION = None
    K = 0  # no wall force for optimization
    F_STIM_SCALE = 0

    N_TRAJECTORIES, PLOTTER = args
    # import pdb; pdb.set_trace()

    simulation, skeeter = agent.gen_objects_and_fly(
        N_TRAJECTORIES,
        TEST_CONDITION,
        beta,
        F_rand_strength,
        windF_strength,
        F_STIM_SCALE,
        K,
        bounded=False,
        verbose=False
    )

    combined_score, _ = simulation.calc_score()

    # if combined_score < HIGH_SCORE:
    #     HIGH_SCORE = combined_score
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
    def __init__(self, xmax=[3., 3., 3.], xmin=[0., 0., 0.]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


if __name__ == '__main__':
    logging.basicConfig(filename='basin_hopping.log', level=logging.DEBUG)

    HIGH_SCORE = 1e10

    OPTIM_ALGORITHM = 'SLSQP'
    PLOTTER = False
    # ACF_THRESH = 0.5
    GUESS_PARAMS = "[beta, F_rand_strength, F_WIND_SCALE]"  # [5e-6, 4.12405e-6, 5e-7]
    INITIAL_GUESS = [1.37213380e-06, 1.39026239e-06, 7.06854777e-07]
    N_TRAJECTORIES = 10

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
        stepsize=1e-5,
        T=1e-2,
        minimizer_kwargs={"args": (N_TRAJECTORIES, PLOTTER),
                          'method': OPTIM_ALGORITHM,
                          "bounds": ((1e-8, 1), (1e-8, 1e-2), (1e-8, 1))},
        disp=True,
        accept_test=mybounds
        )



    logging.info("""############################################################
    Trial end. FINAL RESULT: {}
    ############################################################""".format(result))

    print result