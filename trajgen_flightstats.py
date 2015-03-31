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


def trajGenIter(r0, v0_stdev, k, beta, f0, wf0, target_pos, Tmax, dt, total_trajectories, detect_thresh):
    """
    run traj_gen total_trajectories times and return arrays

    r0 = [1., 0]
    v0_stdev = 0.01
    k = 0.
    beta = 2e-5
    f0 = 3e-6
    wf0 = 5e-6
    target_pos = [0.2, 0.05]
    Tmax = 4.0
    dt = 0.01
    total_trajectories = 15
    """
    pos = []
    velos = []
    accels = []
    target_finds = []
    t_targfinds = []
    trajectory_objects_list = []

    for i in range(total_trajectories):
        trajectory = traj_gen.Trajectory(r0=r0, v0_stdev=v0_stdev, k=k, beta=beta, f0=f0, wf0=wf0, target_pos=target_pos, Tmax=Tmax, dt=dt, detect_thresh=detect_thresh)
        pos += [trajectory.positionList]
        velos += [trajectory.veloList]
        accels += [trajectory.accelList]
        target_finds += [trajectory.target_found]
        t_targfinds += [trajectory.t_targfound]
        trajectory_objects_list.append(trajectory)

    Tfind_avg, num_success = T_find_stats(t_targfinds, total_trajectories)

    return pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list


def T_find_stats(t_targfinds, total_trajectories):
    t_targfinds = np.array(t_targfinds)
    t_finds_NoNaNs = t_targfinds[~np.isnan(t_targfinds)]  # remove NaNs
    if len(t_finds_NoNaNs) == 0:
        return None, 0
    else:
        num_success = float(len(t_finds_NoNaNs))
        
        Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)
        return Tfind_avg, num_success


def trajectory_plots(pos, target_finds, Tfind_avg, trajectory_objects_list):
    traj_ex = trajectory_objects_list[0]
    agent_paths_fig = plt.figure(1)
    for traj in pos:
        plt.plot(traj[:, 0], traj[:, 1], lw=2, alpha=0.5)
    plt.scatter(traj_ex.target_pos[0], traj_ex.target_pos[1], s=150, c='r', marker="*")
    title_append = """ for {0} secs. \n
                beta = {2}, f0 = {3}, wf = {4}. \n
                <Tfind> = {5}, Sourcefinds = {6}/(n = {1})
                """.format(traj_ex.Tmax, len(trajectory_objects_list), traj_ex.beta, traj_ex.f0, traj_ex.wf0, Tfind_avg, sum(target_finds))
    plt.title("Agent trajectories" + title_append)
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.savefig("agent trajectories.png")
    plt.show()


def stateHistograms(pos, velos, accels):
    pos_all = np.concatenate(pos, axis=0)
    posHistBinWidth = 0.05
    position_lim = 1.1
    positional_bins = np.arange(-0.2, position_lim + posHistBinWidth, posHistBinWidth)  # set left bound just past 0
    pos_dist_fig = plt.figure(2)
    plt.hist(pos_all[:, 0], bins=positional_bins, alpha=0.5, label='x', normed=True)
    plt.hist(pos_all[:, 1], bins=positional_bins, alpha=0.5, label='y', normed=True)
    plt.title("x,y position distributions")
    plt.legend()
    plt.savefig("position distributions histo.png")

    velo_all = np.concatenate(velos, axis=0)
    veloHistBinWidth = 0.01
    velo_lim = 0.6
    velo_bins = np.arange((-velo_lim - veloHistBinWidth), (velo_lim + veloHistBinWidth), veloHistBinWidth)
    velo_dist_fig = plt.figure(3)
    plt.hist(velo_all[:, 0], bins=velo_bins, alpha=0.5, label='vx', normed=True)
    plt.hist(velo_all[:, 1], bins=20, alpha=0.5, label='vy', normed=True)
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
    plt.hist(accel_all[:, 0], bins=accel_bins, alpha=0.5, label='ax', normed=True)
    plt.hist(accel_all[:, 1], bins=accel_bins, alpha=0.5, label='ay', normed=True)
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


def main(r0=[1., 0.], v0_stdev=0.01, k=0., beta=2e-5, f0=3e-6, wf0=5e-6, target_pos=[0.2, 0.05], Tmax=4.0, dt=0.01, total_trajectories=20, detect_thresh=0.02, plotting = True):
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = trajGenIter(r0=r0, v0_stdev=v0_stdev, k=k, beta=beta, f0=f0, wf0=wf0, target_pos=target_pos, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories, detect_thresh=detect_thresh)

    if plotting is True:
        trajectory_plots(pos, target_finds, Tfind_avg, trajectory_objects_list)
        stateHistograms(pos, velos, accels)

    return pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list


if __name__ == '__main__':
    
    # following params only used if this function is being run on its own
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = main()
    
