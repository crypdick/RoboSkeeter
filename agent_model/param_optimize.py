__author__ = 'richard'

import agent3D
import plume3D
import trajectory3D
from scipy.optimize import minimize, fminbound, basinhopping
import numpy as np
# from matplotlib import pyplot as plt
import pdb
import time


myplume = plume3D.Plume()

# load csv values
csv = np.genfromtxt ('experimental_data/accelerationmag_raw.csv', delimiter=",")
csv = csv.T

observed = csv[4][:-1] # throw out last datum

# wrapper func for agent 3D
def wrapper(bias_scale_prime):

    bias_scale_prime *= 1e-10
#    beta_prime, bias_scale_prime = param_vect
    # when we run this, agent3D is run and we return a score
    scalar = 1e-7

    wallF_params = [scalar]  #(4e-1, 1e-6, 1e-7, 250)

    # temperature plume

    trajectories = trajectory3D.Trajectory() # instantiate empty trajectories object
    myagent = agent3D.Agent(
        trajectories,
        myplume,
        agent_pos="door",
        heater="left",
        v0_stdev=0.01,
        wtf=7e-07,
        biasF_scale=bias_scale_prime, #4e-05,
        stimF_str=1e-4,
        beta=1e-5,
        Tmax=15.,
        dt=0.01,
        detect_thresh=0.023175,
        bounded=True,
        wallF_params=wallF_params)
    myagent.fly(total_trajectories=1)

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
    aabs_counts_n = aabs_counts / float(len(accel_all_magn))
    
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
    print '{0:4d}  |  Guess: {1}  | Score: {2: 3.10f}  | Accepted: {3}'.format(Nfeval, Xi, score, accept)
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
    #     wrapper,
    #     1,
    #     1500000,
    #     xtol=1,
    #     full_output=True,
    #     disp=3)

    result = basinhopping(
        wrapper,
        6e-6,
        stepsize=1e-7,
        T=1e-6,
        minimizer_kwargs={"bounds": ((1e-8, 1e-4),)},
        callback=callbackF,
        disp=True)

    print result
    # fminbound(wrapper, )
    # wrapper(1e-7)


if __name__ == '__main__':
    main()