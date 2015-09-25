__author__ = 'richard'

from scipy.optimize import basinhopping
import logging
from datetime import datetime

from matplotlib import pyplot as plt

import score
import agent
import plume
import windtunnel
import trajectory
from scripts import pickle_experiments


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
    # load  pickled experimental data just once
    experimental_trajs = pickle_experiments.load_mosquito_pickle()

    BETA, FORCES_AMPLITUDE, F_WIND_SCALE = GUESS
    HEATER = None
    K = 0  # no wall force for optimization
    F_STIM_SCALE = 0

    N_TRAJECTORIES = args
    #  F_WALL_SCALE aka k
    #  when we run this, agent3D is run and we return a score


    windtunnel_object = windtunnel.Windtunnel(None) # we are fitting for the control condition
    plume_object = plume.Plume(HEATER)
    #  temperature plume
    agent_trajectories = trajectory.Trajectory() # instantiate empty trajectories object
    skeeter = agent.Agent(
        experimental_condition=HEATER,
        windF_strength=F_WIND_SCALE,
        randomF_strength=FORCES_AMPLITUDE,
        stimF_stength=F_STIM_SCALE,
        damping_coeff=BETA,
        spring_const=K,
        bounded=False)

    skeeter.fly(total_trajectories=N_TRAJECTORIES, verbose=False)  # fix N trajectories in main

    ensemble = agent_trajectories.data

    combined_score, dkl_scores = score.score(agent_trajectories, experimental_trajs)

    if combined_score < HIGH_SCORE:
        HIGH_SCORE = combined_score
    # if PLOTTER is True:
    #     error_plotter(ensemble, GUESS, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed)
    #
    # if np.any(np.isinf(dkl_scores)):
    #     error_plotter(ensemble, GUESS, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed)

    return combined_score


def error_plotter(ensemble, guess, dkl_scores, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n, v_observed, a_observed):



    print "{} New high score: {}. Guess: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, guess, dkl_scores, acf_score)
    logging.info("{} New high score: {}. Guess: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, guess, dkl_scores, acf_score))

    plt.ioff()
    f, axarr = plt.subplots(2, sharex=False)
    axarr[0].plot(aabs_bins[:-1], a_counts_n, label='RoboSkeeter')
    axarr[0].plot(aabs_bins[:-1], a_observed, c='r', label='experiment')
    axarr[0].set_title('accel score=> {}. High score = DKL(a)+DKL(v) = {}'.format(dkl_scores[1], HIGH_SCORE))
    axarr[0].legend()

    axarr[1].plot(vabs_bins[:-1], v_counts_n, label='RoboSkeeter')
    axarr[1].plot(vabs_bins[:-1], v_observed, c='r', label='experiment')
    axarr[1].set_title('velocity score=> {}. High score = DKL(a)+DKL(v) = {}'.format(dkl_scores[0], HIGH_SCORE))
    axarr[1].legend()
    plt.suptitle('Parameter Guess: {}'.format(guess))
    plt.show()
    plt.close()


def main():
    print "Starting optimizer."


    guess_params = "[BETA, FORCES_AMPLITUDE, F_WIND_SCALE]"  # [5e-6, 4.12405e-6, 5e-7]

    logging.info("""\n ############################################################
        ############################################################
        {} Start optimization with {} algorithm
        # trajectories = {}
        Params = {}
        Initial Guess = {}
        ############################################################""".format(
        datetime.now(), OPTIM_ALGORITHM, N_TRAJECTORIES, guess_params, INITIAL_GUESS))

    result = basinhopping(
        fly_wrapper,
        INITIAL_GUESS,
        # stepsize=1e-5,
        # T=1e-4,
        minimizer_kwargs={"args": (N_TRAJECTORIES,), 'method': OPTIM_ALGORITHM,
                          "bounds": ((0, None), (0, None), (0, None))}
        # disp=True
        )

    return result


if __name__ == '__main__':
    logging.basicConfig(filename='basin_hopping.log',level=logging.DEBUG)

    HIGH_SCORE = 1e10

    OPTIM_ALGORITHM = 'SLSQP'
    PLOTTER = False
    ACF_THRESH = 0.5
    INITIAL_GUESS = [  1.37213380e-06 ,  1.39026239e-06 ,  2.06854777e-06]
    N_TRAJECTORIES = 20

    myplume = plume.Plume(None)

    result = main()


    logging.info("""############################################################
    Trial end. FINAL RESULT: {}
    ############################################################""".format(result))

    print result