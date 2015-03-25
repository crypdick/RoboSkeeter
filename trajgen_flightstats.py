# -*- coding: utf-8 -*-
"""
Fork of flight_stats code to use the traj_gen script instead

TODO: make r0, v0 vary
TODO: make sure globals inside traj_gen get overriden (e.g. Tmax)

@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
import traj_gen
import numpy as np


def trajGenIter(r0, v0, k, beta, f0, wf0, rs, Tmax, dt, total_trajectories):
    """
    run traj_gen total_trajectories times and return arrays
    """
    pos = []
    velos = []
    accels = []
    source_finds = []
    t_finds = []

    for i in range(total_trajectories):
        t, r, v, a, source_found, tfound = traj_gen.traj_gen(r0=r0, v0=v0, k=k, beta=beta, f0=f0, wf0=wf0, rs=rs, Tmax=Tmax, dt=dt)
        pos += [r]
        velos += [v]
        accels += [a]
        source_finds += [source_found]
        t_finds += [tfound]

    return pos, velos, accels, source_finds, np.array(t_finds)


def stateHistograms(pos, velos, accels):
    pos_all = np.concatenate(pos, axis=0)
    posHistBinWidth = 0.1
    position_lim = 2.0
    positional_bins = np.arange(-position_lim-posHistBinWidth, position_lim + posHistBinWidth, posHistBinWidth) 
    pos_dist_fig = plt.figure(1)
    plt.hist(pos_all)
#    plt.hist(pos_all[:,0], bins=positional_bins, alpha=0.5, label='x', normed=True)
#    plt.hist(pos_all[:,1], bins=positional_bins, alpha=0.5, label='y', normed=True)
    plt.title("x,y position distributions")
    plt.legend()
    
    velo_all = np.concatenate(velos, axis=0)
    veloHistBinWidth = 0.1
    velo_lim = 0.5
    velo_bins = np.arange((-velo_lim - veloHistBinWidth), (velo_lim + veloHistBinWidth), veloHistBinWidth)
    velo_dist_fig = plt.figure(2)
    plt.hist(velo_all)
#    plt.hist(velo_all[:,0], bins=velo_bins, alpha=0.5, label='vx', normed=True)
#    plt.hist(velo_all[:,1], bins=velo_bins, alpha=0.5, label='vy', normed=True)
    plt.title("x,y velocity distributions")    
    plt.legend()
    
    accel_all = np.concatenate(accels, axis=0)
    accelHistBinWidth = 0.1
    accel_lim = 0.5
    accel_bins = np.arange((-accel_lim - accelHistBinWidth), (accel_lim + accelHistBinWidth), accelHistBinWidth)
    accel_dist_fig = plt.figure(3)
    plt.hist(accel_all)
#    plt.hist(accel_all[:,0], bins=accel_bins, alpha=0.5, label='ax', normed=True)
#    plt.hist(accel_all[:,1], bins=accel_bins, alpha=0.5, label='ay', normed=True)
    plt.title("x,y acceleration distributions")
    plt.legend()
    
    plt.show()
    

def probFindGrid(source_finds):
    pass


def T_find_average(t_finds):
    t_finds_NoNaNs = t_finds[~np.isnan(t_finds)]  # remove NaNs
    Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)
    print("<Time_find> = ", Tfind_avg)
    return Tfind_avg

def main(r0, v0, k, beta, f0, rs, Tmax, dt, total_trajectories):
    pos, velos, accels, source_finds, t_finds = trajGenIter(r0, v0, k, beta, f0, wf0, rs, Tmax, dt, total_trajectories)
    Tfind_avg = T_find_average(t_finds)

    return pos, velos, accels, source_finds, Tfind_avg

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    pos, velos, accels, source_finds, Tfind_avg = main(r0=[1., 0], v0=[0, 0.4], k=1e-6, beta=2e-7, f0=3e-6, rs = [0.13, 0.01], Tmax=30.0, dt=0.01, total_trajectories=15)
    stateHistograms(pos, velos, accels)
    probFindGrid(source_finds)

