# -*- coding: utf-8 -*-
"""
Use generate_trajectory script to generate a multitude of trajectory objects,
then do some basic stats on the flight trajectory.

Default params declared in main().

@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
import generate_trajectory
import numpy as np
import plotting_funcs


def trajGenIter(r0, v0_stdev, k, beta, f0, wf0, target_pos, Tmax, dt, total_trajectories, bounded, detect_thresh):
    """
    Run generate_trajectory total_trajectories times and return the results
    as arrays.
    
    Args:
        Takes all the Args generate_trajectory does.
        total_trajectories: # of times to run generate_trajectory (int)
        
    Returns:
        Arrays containing the different attributes of each trajectory object.
        pos: list of all positions (array)
        velos: list of all velocities (array)
        accels: list of all accelerations (array)
        target_finds: list of all whether trajectory found the source True/False (array)  # TODO: is this return neeeded? -rd
        t_targfinds: list of all time to the target (array) (sometimes NaN)
        Tfind_avg: list of all times the target was found (array) (some NaNs)
        num_success: fraction of total_trajectories that made it (int)
        trajectory_objects_list: list of all trajectory objects (array)
    """
    pos = []
    velos = []
    accels = []
    target_finds = []
    t_targfinds = []
    trajectory_objects_list = []

    for traj in range(total_trajectories):
        trajectory = generate_trajectory.Trajectory(r0=r0, v0_stdev=v0_stdev, k=k, beta=beta, f0=f0, wf0=wf0, target_pos=target_pos, Tmax=Tmax, dt=dt, detect_thresh=detect_thresh, bounded=bounded)
        # extract trajectory object attribs, append to our lists.
        pos += [trajectory.positionList]
        velos += [trajectory.veloList]
        accels += [trajectory.accelList]
        target_finds += [trajectory.target_found]
        t_targfinds += [trajectory.t_targfound]
        trajectory_objects_list.append(trajectory)

    Tfind_avg, num_success = T_find_stats(t_targfinds, total_trajectories)

    return pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list


def T_find_stats(t_targfinds, total_trajectories):
    """Target finding stats. Returns average time to find the target, and the
    number of successes"""
    t_targfinds = np.array(t_targfinds)
    t_finds_NoNaNs = t_targfinds[~np.isnan(t_targfinds)]  # remove NaNs
    if len(t_finds_NoNaNs) == 0:

        return None, 0  # no target finds, no stats to run
    else:
        num_success = float(len(t_finds_NoNaNs))
        Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)

        return Tfind_avg, num_success


def main(r0=[1., 0.], v0_stdev=0.01, k=0., beta=2e-5, f0=3e-6, wf0=5e-6, target_pos=None, Tmax=4.0, dt=0.01, total_trajectories=100, detect_thresh=0.02, bounded=True, plotting = True): # target_pos=[0.2, 0.04]
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = trajGenIter(r0=r0, v0_stdev=v0_stdev, k=k, beta=beta, f0=f0, wf0=wf0, target_pos=target_pos, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories, bounded=bounded, detect_thresh=detect_thresh)

    if plotting is True:
        # plot all trajectories
        plotting_funcs.trajectory_plots(pos, target_pos, target_finds, Tfind_avg, trajectory_objects_list)
        # plot histogram of pos, velo, accel distributions
#        plotting_funcs.stateHistograms(pos, velos, accels)

    return pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list


if __name__ == '__main__':
    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, trajectory_objects_list = main()
    
