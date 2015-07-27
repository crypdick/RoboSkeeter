__author__ = 'richard'

import agent3D
import plume3D
import trajectory3D
from scipy.optimize import minimize, fminbound, basinhopping, fmin
import numpy as np
# from matplotlib import pyplot as plt
import pdb
import time


myplume = plume3D.Plume(None)

# load csv values
csv = np.genfromtxt('experimental_data/accelerationmag_raw.csv',delimiter=',')
csv = csv.T
observed = csv[4][:-1] # throw out last datum

# wrapper func for agent 3D
def wrapper(GUESS):
    """
    :param bias_scale_GUESS:
    :param mass_GUESS:
        mean mosquito mass is 2.88e-6
    :param damping_GUESS:
        estimated damping was 5e-6, # cranked up to get more noise #5e-6,#1e-6,  # 1e-5
    :return:
    """
    rescaling = 1e-8
    bias_scale_GUESS = GUESS[0] * rescaling
    mass_GUESS = GUESS[1] * rescaling
    damping_GUESS = GUESS[2] * rescaling

#    beta_prime, bias_scale_GUESS = param_vect
    # when we run this, agent3D is run and we return a score
    scalar = 1e-7

    wallF_params = [scalar]  #(4e-1, 1e-6, 1e-7, 250)

    # temperature plume

    trajectories = trajectory3D.Trajectory() # instantiate empty trajectories object
    myagent = agent3D.Agent(
        trajectories,
        myplume,
        agent_pos="downwind_plane",
        heater="l",
        v0_stdev=0.01,
        wtf=7e-07,
        wtf_scalar=.05,
        biasF_scale=bias_scale_GUESS, #4e-05,
        stimF_str=7e-7,
        beta=damping_GUESS, # 5e-6,
        Tmax=15.,
        mass=mass_GUESS,
        dt=0.01,
        detect_thresh=0.023175,
        bounded=True,
        wallF_params=wallF_params)
    myagent.fly(total_trajectories=10, verbose=False)

    ensemble = trajectories.ensemble
    trimmed_ensemble = ensemble.loc[
        (ensemble['position_x'] >0.25) & (ensemble['position_x'] <0.95)]

    # we want to fit beta and rF
    score = error_fxn(trimmed_ensemble)

    # end = time.time()
    # print end - start
    # print score
    return score


def error_fxn(ensemble):
    # compare ensemble to experiments, return score to wrapper

    # get histogram vals for ensemble
    # |a|
    amin, amax = -9.05, 11.
    # pdb.set
    accel_all_magn = ensemble['acceleration_3Dmagn'].values
    aabs_counts, aabs_bins = np.histogram(accel_all_magn, bins=np.linspace(amin, amax, 200))
    if np.isnan(np.sum(aabs_counts)) is True:
        print "NANANAN"
    aabs_counts = aabs_counts.astype(float)
    aabs_counts_n = aabs_counts / aabs_counts.sum()
    
    #plt.plot(aabs_bins[:-1], aabs_counts_n)
    #plt.plot(aabs_bins[:-1], observed, c='r')
    # print csv
    # print aabs_counts_n, 'counts',
    return np.sqrt(np.mean((aabs_counts_n - observed)**2)) * 100000  # multiply score to get better granularity

# # for nelder mead
# Nfeval = 1
# def callbackF(Xi):
#     global Nfeval
#     print '{0:4d}   {1: 3.25f}   {2: 3.25f}   {3: 3.8f}'.format(Nfeval, Xi[0], Xi[1], wrapper(Xi))
#     Nfeval += 1

# for basin hopping
Nfeval = 1
def callbackF(Xi, score, accept):
    global Nfeval
    print '{0:4d}  |  Basin found at: {1}  | Score: {2: 3.10f}  | Accepted: {3}'.format(Nfeval, Xi, score, accept)
    Nfeval += 1


def main():
#    result = minimize(
#        wrapper,
#        [1e-5, 4e-5],
#        method='Nelder-Mead',
#        #bounds=((1e-8, 1e-4), (1e-6, 1e-3)),
#        options={'xtol': 1e-7, 'disp':True},
#        callback=callbackF)

    # result = fminbound(
    #     wrapper, # bias_scale_GUESS, mass_GUESS, damping_GUESS)
    #     np.array([100, 200, 100]),
    #     np.array([1000, 1000, 1000]),
    #     xtol=100,
    #     full_output=True,
    #     disp=3)

    result = fmin(
        wrapper,
        [1020, 288, 500], # [1.0239e-5, 2.88e-6,  5e-6]
        xtol=20,
        full_output=1,
        disp=1,
        retall=1
        # ftol=0.0001
    )

    # result = basinhopping(
    #     wrapper,
    #     [640],
    #     stepsize=1000,
    #     T=4000,
    #     minimizer_kwargs={"bounds": ((1, 10000),)},
    #     callback=callbackF,
    #     disp=True)

    print result
    # fminbound(wrapper, )
    # wrapper(1e-7)


if __name__ == '__main__':
    main()