__author__ = 'richard'

import logging
from datetime import datetime

import numpy as np
from scipy.optimize import basinhopping

import agent
from scripts.pickle_experimentsY import load_mosquito_kde_data_dicts


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
    ##############################################################
    BOUNCE_COEFF: {}
    ##############################################################""".format(BOUNCE_COEFF)

    agent_kwargs = {'initial_position_selection': 'downwind_high',
                    'windF_strength': 0.,
                    'randomF_strength': 6.55599224e-06,
                    'experimental_condition': None,  # {'Left', 'Right', None}
                    'stimF_stength': 0.,
                    'spring_const': 0,
                    'damping_coeff': 3.63674551e-07,
                    'time_max': 15.,
                    'bounded': True,
                    'collision_type': 'crash',
                    'crash_coeff': BOUNCE_COEFF}

    N_TRAJECTORIES, PLOTTER, (EXP_BINS, EXP_VALS) = args

    # import pdb; pdb.set_trace()

    simulation, skeeter = agent.gen_objects_and_fly(N_TRAJECTORIES, agent_kwargs, verbose=False)

    combined_score, score_components, _ = simulation.calc_score(ref_ensemble=(EXP_BINS, EXP_VALS))

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

    # if PLOTTER is True:
    #     error_plotter(ensemble, BOUNCE_COEFF, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed)
    #
    # if np.any(np.isinf(dkl_scores)):
    #     error_plotter(ensemble, BOUNCE_COEFF, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed)

    return combined_score


# def error_plotter(ensemble, BOUNCE_COEFF, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed):
#
#
#
#     print "{} New high score: {}. BOUNCE_COEFF: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, BOUNCE_COEFF, dkl_scores, acf_score)
#     logging.info("{} New high score: {}. BOUNCE_COEFF: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, BOUNCE_COEFF, dkl_scores, acf_score))
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
#     plt.suptitle('Parameter BOUNCE_COEFF: {}'.format(BOUNCE_COEFF))
#     plt.show()
#     plt.close()


class MyBounds(object):
    def __init__(self, xmax=[1.], xmin=[1e-7]):  # , 1e08]):
        self.xmax = np.array(xmax)
        self.xmin = np.array(xmin)

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        return tmax and tmin


def main(x_0=None):
    logging.basicConfig(filename='basin_hoppingY.log', level=logging.DEBUG)

    global HIGH_SCORE
    HIGH_SCORE = 1e10

    global BEST_BOUNCE_COEFF
    BEST_BOUNCE_COEFF = None

    OPTIM_ALGORITHM = 'SLSQP'
    PLOTTER = False
    # ACF_THRESH = 0.5
    BOUNCE_COEFF_PARAMS = "BOUNCE_COEFF"  # , F_WIND_SCALE]"  # [5e-6, 4.12405e-6, 5e-7]
    if x_0 is None:
        INITIAL_BOUNCE_COEFF = 0.9
    else:
        INITIAL_BOUNCE_COEFF = x_0
    N_TRAJECTORIES = 100

    print "Starting optimizer."

    logging.info("""\n ############################################################
        ############################################################
        {} Start optimization with {} algorithm
        # trajectories = {}
        Params = {}
        Initial BOUNCE_COEFF = {}
        ############################################################""".format(
        datetime.now(), OPTIM_ALGORITHM, N_TRAJECTORIES, BOUNCE_COEFF_PARAMS, INITIAL_BOUNCE_COEFF))

    mybounds = MyBounds()

    result = basinhopping(
        fly_wrapper,
        INITIAL_BOUNCE_COEFF,
        stepsize=1e-1,
        T=5e-1,
        niter=5,  # number of basin hopping iterations, default 100
        niter_success=3,  # Stop the run if the global minimum candidate remains the same for this number of iterations
        minimizer_kwargs={
            "args": (N_TRAJECTORIES, PLOTTER, (load_mosquito_kde_data_dicts())),
            'method': OPTIM_ALGORITHM,
            "bounds": [(5e-7, 1.)],
            "tol": 0.1  # tolerance for considering a basin minimized, set to about the difference between re-scoring
            # same simulation
        },
        disp=True,
        accept_test=mybounds
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
