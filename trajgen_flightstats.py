# -*- coding: utf-8 -*-
"""
Fork of flight_stats code to use the traj_gen script instead

@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
import traj_gen
import numpy as np


def trajGenIter(Tmax, dt, total_trajectories):
    """
    run traj_gen total_trajectories times and return arrays
    """
    pos = []
    velos = []
    accels = []
    source_finds = []
    t_finds = []

    ## initial conditions TODO make vary
    r0 = [1., 0]
    v0 = [0, 0.4]
    for i in range(total_trajectories):
        t, r, v, a, source_found, tfound = traj_gen.traj_gen(r0, v0, Tmax=Tmax, dt=dt)
        pos += [r]
        velos += [v]
        accels += [a]
        source_finds += [source_found]
        t_finds += [tfound]

    return pos, velos, accels, source_finds, t_finds


def stateHistograms():
    pass    


def sourceFindHeatmap():
    pass


def main(Tmax=1.0, dt=0.01, total_trajectories=2):
    pos, velos, accels, source_finds, t_finds = trajGenIter(Tmax, dt, total_trajectories)

    return pos, velos, accels, source_finds, t_finds

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    pos, velos, accels, source_finds, t_finds = main()
    stateHistograms()
    sourceFindHeatmap()

    
##############  LEGACY CODE  ##########################################
    ## plot position distributions##
#    xy_binwidth = 0.5
#    position_lim = 5.0
#    positional_bins = np.arange(-position_lim, position_lim + xy_binwidth, xy_binwidth) #specify bin locations
#    
#    #x dimension
#    x_fig = plt.figure(3)
#    plt.hist(x_positions, bins=positional_bins, normed=True)
#    plt.title("x position distributions")
#    
#    #y dimension
#    y_fig = plt.figure(4)
#    plt.hist(y_positions, bins=positional_bins, orientation='horizontal', color='r', normed=True)
#    plt.title("y position distributions")
#    
#    ##plot velocity distributions##
#    velo_binwidth = 0.01
#    velo_lim = 0.12
#    velo_bins = np.arange(-velo_lim, velo_lim + velo_binwidth, velo_binwidth)
#    
#    #x velo dimension
#    xv_fig = plt.figure(5)
#    plt.hist(x_velocities, bins=velo_bins, color='g', normed=True)
#    plt.title("x velocity distributions")
#    
#    #y velocity dim
#    yv_fig = plt.figure(6)
#    plt.hist(x_velocities, bins=velo_bins, orientation='horizontal', color='cyan', normed=True)
#    plt.title("y velocity distributions")
#    
#    plt.show()
########################################################
