# -*- coding: utf-8 -*-
"""
Fork of flight_stats code to use the traj_gen script instead

@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
import traj_gen
import numpy as np

## define params
Tmax = 2e3
total_trajectories = 5

def flightStats(Tmax=Tmax, total_trajectories=total_trajectories):
    """
    TODO: how to do flight stats with t, r ,v, a of different lengths?
    """

###############  LEGACY CODE  ####################################################
    flight_states = np.zeros((total_trajectories, flight_runtime, 4))
    for i in range(total_trajectories):
        states = oscillator.main(runtime=flight_runtime, plotting=False)
        flight_states[i] = states
    
    x_positions = []
    x_velocities = []
    y_positions = []
    y_velocities = []
    
    for trajectory in range(total_trajectories):
        for time in range(int(flight_runtime)):
            x, xv, y, yv = flight_states[trajectory, :][time]
            x_positions.append(x)
            x_velocities.append(xv)
            y_positions.append(y)
            y_velocities.append(yv)
########################################################


if __name__ == '__main__':
    import matplotlib.pyplot as plt

##############  LEGACY CODE  ##########################################
    ## plot position distributions##
    xy_binwidth = 0.5
    position_lim = 5.0
    positional_bins = np.arange(-position_lim, position_lim + xy_binwidth, xy_binwidth) #specify bin locations
    
    #x dimension
    x_fig = plt.figure(3)
    plt.hist(x_positions, bins=positional_bins, normed=True)
    plt.title("x position distributions")
    
    #y dimension
    y_fig = plt.figure(4)
    plt.hist(y_positions, bins=positional_bins, orientation='horizontal', color='r', normed=True)
    plt.title("y position distributions")
    
    ##plot velocity distributions##
    velo_binwidth = 0.01
    velo_lim = 0.12
    velo_bins = np.arange(-velo_lim, velo_lim + velo_binwidth, velo_binwidth)
    
    #x velo dimension
    xv_fig = plt.figure(5)
    plt.hist(x_velocities, bins=velo_bins, color='g', normed=True)
    plt.title("x velocity distributions")
    
    #y velocity dim
    yv_fig = plt.figure(6)
    plt.hist(x_velocities, bins=velo_bins, orientation='horizontal', color='cyan', normed=True)
    plt.title("y velocity distributions")
    
    plt.show()
########################################################
