__author__ = 'richard'

import score
import agent3D
import plume3D
import windtunnel
import trajectory3D
from scipy.optimize import basinhopping
import numpy as np
from matplotlib import pyplot as plt
import logging
from datetime import datetime

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
    print GUESS, "3"
    BETA, FORCES_AMPLITUDE, F_WIND_SCALE = GUESS
    HEATER = None
    K = 0  # no wall force for optimization
    #  F_WALL_SCALE aka k
    #  when we run this, agent3D is run and we return a score

    windtunnel_object = windtunnel.Windtunnel(None) # we are fitting for the control condition
    plume_object = plume3D.Plume(HEATER)
    #  temperature plume
    trajectories = trajectory3D.Trajectory() # instantiate empty trajectories object
    skeeter = agent3D.Agent(
        trajectories,
        plume_object,
        windtunnel_object,
        mass=2.88e-6,
        agent_pos="downwind_plane",
        heater=HEATER,
        wtf=F_WIND_SCALE,
        randomF_strength=FORCES_AMPLITUDE,
        stimF_str=0., # F_STIM_SCALE,
        beta=BETA,
        k=K, #
        Tmax=15.,
        dt=0.01,
        bounded=True)

    skeeter.fly(total_trajectories=N_TRAJECTORIES, verbose=False)  # fix N trajectories in main

    # ensemble = trajectories.ensemble
    # # trimmed_ensemble = ensemble.loc[
    # #     (ensemble['position_x'] >0.25) & (ensemble['position_x'] <0.95)]

    # we want to fit beta and rF
    combined_score, dkl_scores, dkl_a, dkl_v, acf_score, aabs_bins, a_counts_n, vabs_bins, v_counts_n \
        = score.score(trajectories.ensemble, ACF_THRESH)


    if (PLOTTER is True) or (np.isinf(dkl_score)):
        if combined_score < HIGH_SCORE:
            HIGH_SCORE = combined_score
            error_plotter()

    return combined_score


def error_plotter(ensemble, guess):



    print "{} New high score: {}. Guess: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, guess, dkl_score, acf_score)
    logging.info("{} New high score: {}. Guess: {}. DKL score = {}. ACF score = {}".format(datetime.now(), HIGH_SCORE, guess, dkl_score, acf_score))

    plt.ioff()
    f, axarr = plt.subplots(2, sharex=False)
    axarr[0].plot(aabs_bins[:-1], a_counts_n, label='RoboSkeeter')
    axarr[0].plot(aabs_bins[:-1], a_observed, c='r', label='experiment')
    axarr[0].set_title('accel score=> {}. High score = DKL(a)+DKL(v) = {}'.format(dkl_a, HIGH_SCORE))
    axarr[0].legend()

    axarr[1].plot(vabs_bins[:-1], v_counts_n, label='RoboSkeeter')
    axarr[1].plot(vabs_bins[:-1], v_observed, c='r', label='experiment')
    axarr[1].set_title('velocity score=> {}. High score = DKL(a)+DKL(v) = {}'.format(dkl_v, HIGH_SCORE))
    axarr[1].legend()
    plt.suptitle('Parameter Guess: {}'.format(guess))
    plt.show()
    plt.close()

    return combined_score


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
    # import pdb; pdb.set_trace()
    print INITIAL_GUESS, "2"
    result = basinhopping(
        fly_wrapper,
        INITIAL_GUESS,
        stepsize=1e-5,
        # T=1e-4,
        minimizer_kwargs={"args": (N_TRAJECTORIES,), 'method': OPTIM_ALGORITHM,  "bounds": ((0, None),(0,None),(0,None))},
        disp=True
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

    myplume = plume3D.Plume(None)
    print INITIAL_GUESS, "1"
    result = main()


    logging.info("""############################################################
    Trial end. FINAL RESULT: {}
    ############################################################""".format(result))

    print result