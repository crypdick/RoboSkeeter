# -*- coding: utf-8 -*-
"""
Fork of flight_stats code to use the traj_gen script instead

@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
import generate_trajectory
import numpy as np
import plotting_funcs


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
        trajectory = generate_trajectory.Trajectory(r0=r0, v0_stdev=v0_stdev, k=k, beta=beta, f0=f0, wf0=wf0, target_pos=target_pos, Tmax=Tmax, dt=dt, detect_thresh=detect_thresh)
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


def main(r0=[1., 0.], v0_stdev=0.01, k=0., beta=2e-5, f0=3e-6, wf0=5e-6, target_pos=[0.2, 0.05], Tmax=4.0, dt=0.01, total_trajectories=20, detect_thresh=0.02, plotting = True):
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = trajGenIter(r0=r0, v0_stdev=v0_stdev, k=k, beta=beta, f0=f0, wf0=wf0, target_pos=target_pos, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories, detect_thresh=detect_thresh)

    if plotting is True:
        plotting_funcs.trajectory_plots(pos, target_finds, Tfind_avg, trajectory_objects_list)
        plotting_funcs.stateHistograms(pos, velos, accels)

    return pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list


if __name__ == '__main__':
    
    # following params only used if this function is being run on its own
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = main()
    
