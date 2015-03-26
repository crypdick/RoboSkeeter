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
import matplotlib.pyplot as plt


def trajGenIter(r0, v0, k, beta, f0, wf0, rs, Tmax, dt, total_trajectories):
    """
    run traj_gen total_trajectories times and return arrays
    """
    pos = []
    velos = []
    accels = []
    source_finds = []
    t_finds = []
    agent_paths_fig = plt.figure(1)

    for i in range(total_trajectories):
        v0 = np.random.normal(0, 0.3, 2)
        t, r, v, a, source_found, tfound = traj_gen.traj_gen(r0=r0, v0=v0, k=k, beta=beta, f0=f0, wf0=wf0, rs=rs, Tmax=Tmax, dt=dt)
        pos += [r]
        velos += [v]
        accels += [a]
        source_finds += [source_found]
        t_finds += [tfound]
        plt.plot(r[:, 0], r[:, 1], lw=2, alpha=0.5)
        plt.scatter(rs[0], rs[1], s=150, c='r', marker="*")
    plt.title("agent trajectories")  # TODO: list params
    plt.savefig("agent trajectories.png")
    plt.show()    

    return pos, velos, accels, source_finds, np.array(t_finds)


def stateHistograms(pos, velos, accels):
    pos_all = np.concatenate(pos, axis=0)
    posHistBinWidth = 0.05
    position_lim = 1.1
    positional_bins = np.arange(-0.2, position_lim + posHistBinWidth, posHistBinWidth) #set left bound just past 0
    pos_dist_fig = plt.figure(2)
#    plt.hist(pos_all)
    plt.hist(pos_all[:,0], bins=positional_bins, alpha=0.5, label='x', normed=True)
    plt.hist(pos_all[:,1], bins=positional_bins, alpha=0.5, label='y', normed=True)
    plt.title("x,y position distributions")
    plt.legend()
    plt.savefig("position distributions histo.png")

    velo_all = np.concatenate(velos, axis=0)
    veloHistBinWidth = 0.05
    velo_lim = 0.6
    velo_bins = np.arange((-velo_lim - veloHistBinWidth), (velo_lim + veloHistBinWidth), veloHistBinWidth)
    velo_dist_fig = plt.figure(3)
#    plt.hist(velo_all)
    plt.hist(velo_all[:,0], bins=velo_bins, alpha=0.5, label='vx', normed=True)
    plt.hist(velo_all[:,1], bins=velo_bins, alpha=0.5, label='vy', normed=True)
    plt.title("x,y velocity distributions")
    plt.legend()
    plt.savefig("velocity distributions histo.png")

    accel_all = np.concatenate(accels, axis=0)
    accelHistBinWidth = 0.6
    accel_lim = 9.
    accel_bins = np.arange((-accel_lim - accelHistBinWidth), (accel_lim + accelHistBinWidth), accelHistBinWidth)
    accel_dist_fig = plt.figure(4)
#    plt.hist(accel_all)
    plt.hist(accel_all[:,0], bins=accel_bins, alpha=0.5, label='ax', normed=True)
    plt.hist(accel_all[:,1], bins=accel_bins, alpha=0.5, label='ay', normed=True)
    plt.title("x,y acceleration distributions")
    plt.legend()
    plt.savefig("acceleration distributions histo.png")

    plt.show()


def T_find_average(t_finds, total_trajectories):
    t_finds_NoNaNs = t_finds[~np.isnan(t_finds)]  # remove NaNs
    if len(t_finds_NoNaNs) == 0:
        print "No successful source finds!"
    else:
        Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)
        print "<Time_find> = ", Tfind_avg, "sec. ", len(t_finds_NoNaNs), "finds out of ", total_trajectories, "trajectories"
        return Tfind_avg  # TODO: append to trajectory title


def main(r0, v0, k, beta, f0, wf0, rs, Tmax, dt, total_trajectories):
    pos, velos, accels, source_finds, t_finds = trajGenIter(r0=r0, v0=v0, k=k, beta=beta, f0=f0, wf0=wf0, rs=rs, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories)
    Tfind_avg = T_find_average(t_finds, total_trajectories)

    return pos, velos, accels, source_finds, Tfind_avg


if __name__ == '__main__':
    # set default params to send to main
    r0 = [1., 0]
    v0 = [0, 0.2]  # made this draw from a 2D gaussian above
    k = 1e-5
    beta = 2e-5
    f0 = 0. #5e-6
    wf0 = 0. #1e-6
    rs = [0.2, 0.05]
    Tmax = 5.0
    dt = 0.01
    total_trajectories = 50
    pos, velos, accels, source_finds, Tfind_avg = main(r0=r0, v0=v0, k=k, beta=beta, f0=f0, wf0=wf0, rs=rs, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories)
    stateHistograms(pos, velos, accels)
