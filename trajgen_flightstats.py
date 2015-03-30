# -*- coding: utf-8 -*-
"""
Fork of flight_stats code to use the traj_gen script instead

@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
import traj_gen
import numpy as np
import matplotlib.pyplot as plt


def trajGenIter(r0, v0_stdev, k, beta, f0, wf0, rs, Tmax, dt, total_trajectories):
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
        t, r, v, a, source_found, tfound = traj_gen.traj_gen(r0=r0, v0_stdev=v0_stdev, k=k, beta=beta, f0=f0, wf0=wf0, rs=rs, Tmax=Tmax, dt=dt)
        pos += [r]
        velos += [v]
        accels += [a]
        source_finds += [source_found]
        t_finds += [tfound]
        
    Tfind_avg = T_find_average(t_finds, total_trajectories)

    return pos, velos, accels, source_finds, t_finds, Tfind_avg


def T_find_average(t_finds, total_trajectories):
    t_finds = np.array(t_finds)
    t_finds_NoNaNs = t_finds[~np.isnan(t_finds)]  # remove NaNsT
    if len(t_finds_NoNaNs) == 0:
        return None
    else:
        Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)
        return Tfind_avg


def trajectory_plots(pos, rs, source_finds, Tmax, Tfind_avg, total_trajectories, beta, f0, wf0):
    for traj in pos:
        plt.plot(traj[:, 0], traj[:, 1], lw=2, alpha=0.5)
    plt.scatter(rs[0], rs[1], s=150, c='r', marker="*")
    title_append = """ for {0} secs. \n
                beta = {2}, f0 = {3}, wf = {4}. \n
                <Tfind> = {5}, Sourcefinds = {6}/(n = {1})
                """.format(Tmax, total_trajectories, beta, f0, wf0, Tfind_avg, sum(source_finds))
    plt.title("Agent trajectories" + title_append)  # TODO: list params
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.savefig("agent trajectories.png")
    plt.show()    
    

def stateHistograms(pos, velos, accels):
    pos_all = np.concatenate(pos, axis=0)
    posHistBinWidth = 0.05
    position_lim = 1.1
    positional_bins = np.arange(-0.2, position_lim + posHistBinWidth, posHistBinWidth) #set left bound just past 0
    pos_dist_fig = plt.figure(2)
    plt.hist(pos_all[:,0], bins=positional_bins, alpha=0.5, label='x', normed=True)
    plt.hist(pos_all[:,1], bins=positional_bins, alpha=0.5, label='y', normed=True)
    plt.title("x,y position distributions")
    plt.legend()
    plt.savefig("position distributions histo.png")

    velo_all = np.concatenate(velos, axis=0)
    veloHistBinWidth = 0.01
    velo_lim = 0.6
    velo_bins = np.arange((-velo_lim - veloHistBinWidth), (velo_lim + veloHistBinWidth), veloHistBinWidth)
    velo_dist_fig = plt.figure(3)
    plt.hist(velo_all[:,0], bins=velo_bins, alpha=0.5, label='vx', normed=True)
    plt.hist(velo_all[:,1], bins=20, alpha=0.5, label='vy', normed=True)
    plt.title("x,y velocity distributions")
    plt.legend()
    plt.savefig("velocity distributions histo.png")
    # absolute velo
    abs_velo_dist_fig = plt.figure(4)
    velo_all_magn = []
    for v in velo_all:
        velo_all_magn.append(np.linalg.norm(v))
    plt.hist(velo_all_magn, label='v_total', bins=30, normed=True)
    plt.title("absolute velocity distributions")
    plt.legend()
    plt.savefig("absolute velocity distributions histo.png")
    

    accel_all = np.concatenate(accels, axis=0)
    accelHistBinWidth = 0.3
    accel_lim = 9.
    accel_bins = np.arange((-accel_lim - accelHistBinWidth), (accel_lim + accelHistBinWidth), accelHistBinWidth)
    accel_dist_fig = plt.figure(5)
#    plt.hist(accel_all)
    plt.hist(accel_all[:,0], bins=accel_bins, alpha=0.5, label='ax', normed=True)
    plt.hist(accel_all[:,1], bins=accel_bins, alpha=0.5, label='ay', normed=True)
    plt.title("x,y acceleration distributions")
    plt.legend()
    plt.savefig("acceleration distributions histo.png")
    # absolute accel
    abs_accel_dist_fig = plt.figure(6)
    accel_all_magn = []
    for a in accel_all:
        accel_all_magn.append(np.linalg.norm(a))
    plt.hist(accel_all_magn, label='a_total', bins=30, normed=True)
    plt.title("absolute acceleration distributions")
    plt.legend()
    plt.savefig("absolute acceleration distributions histo.png")

    plt.show()


def main():
    # set default params to send to trajGenIter()
    r0 = [1., 0]
    v0_stdev = 0.01
    k = 0.
    beta = 2e-5
    f0 = 3e-6
    wf0 = 5e-6
    rs = [0.2, 0.05]
    Tmax = 4.0
    dt = 0.01
    total_trajectories = 15
    
    pos, velos, accels, source_finds, t_finds, Tfind_avg = trajGenIter(r0=r0, v0_stdev=v0_stdev, k=k, beta=beta, f0=f0, wf0=wf0, rs=rs, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories)
    
    trajectory_plots(pos, rs, source_finds, Tmax, Tfind_avg, total_trajectories, beta, f0, wf0)
    stateHistograms(pos, velos, accels)

    return pos, velos, accels, source_finds, t_finds, Tfind_avg


if __name__ == '__main__':
    main()
    