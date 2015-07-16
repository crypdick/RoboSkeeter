__author__ = 'richard'

import agent3D
import plume3D
import trajectory3D
from scipy.optimize import minimize, fminbound
import numpy as np


myplume = plume3D.Plume()

# wrapper func for agent 3D
def wrapper(param_vect):
    beta_prime, bias_scale_prime = param_vect
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
        beta=beta_prime, #1e-5,
        Tmax=15.,
        dt=0.01,
        detect_thresh=0.023175,
        bounded=True,
        wallF_params=wallF_params)
    myagent.fly(total_trajectories=60)

    ensemble = trajectories.ensemble.loc[
        (trajectories.ensemble['position_x'] >0.25) & (trajectories.ensemble['position_x'] <0.95)]

    # we want to fit beta and rF
    score = error_fxn(ensemble)
    return score


def error_fxn(ensemble):
    # compare ensemble to experiments, return score to wrapper

    # get histogram vals for ensemble
    # |a|
    amin, amax = -4., 6.

    accel_stack = np.vstack((ensemble.acceleration_x.values, ensemble.acceleration_y.values, ensemble.acceleration_z.values))
    accel_all_magn = np.linalg.norm(accel_stack, axis=0)
    aabs_counts, aabs_bins = np.histogram(accel_all_magn, bins=np.linspace(amin, amax, 100))
    aabs_counts = aabs_counts.astype(float)
    aabs_counts_n = aabs_counts / aabs_counts.sum()

    # load csv values
    csv = np.genfromtxt ('experimental_data/accelerationmag_raw.csv', delimiter=",")
    csv = csv.T

    observed = csv[4][:-1] # throw out last datum
    # print csv
    return np.sqrt(np.mean((aabs_counts_n - observed)**2))

Nfeval = 1
def callbackF(Xi):
    global Nfeval
    print '{0:4d}   {1: 3.25f}   {2: 3.25f}   {3: 3.8f}'.format(Nfeval, Xi[0], Xi[1], wrapper(Xi))
    Nfeval += 1


def main():
    result = minimize(
        wrapper,
        [1e-5, 4e-5],
        method='Nelder-Mead',
        #bounds=((1e-8, 1e-4), (1e-6, 1e-3)),
        options={'xtol': 1e-7, 'disp':True},
        callback=callbackF)
    print result
    # fminbound(wrapper, )
    # wrapper(1e-7)


if __name__ == '__main__':
    main()