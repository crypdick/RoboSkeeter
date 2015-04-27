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
import pandas as pd


def trajGenIter(agent_pos, target_pos, v0_stdev, k, beta, rf, wtf, Tmax, dt, total_trajectories, wallF, bounded, bounce, detect_thresh):
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
        ensemble: list of all trajectory objects (array)
    """
    
    ensemble = pd.DataFrame()

    for traj in range(total_trajectories):
        trajectory = generate_trajectory.Trajectory(agent_pos=agent_pos, target_pos=target_pos, v0_stdev=v0_stdev, k=k, beta=beta, rf=rf, wtf=wtf, Tmax=Tmax, dt=dt, detect_thresh=detect_thresh, wallF=wallF, bounded=bounded, bounce=bounce)
        # extract trajectory object attribs, append to our lists.
        trajectory.dynamics['trajectory'] = traj
        trajectory.dynamics.set_index('trajectory', append=True, inplace=True)
        ensemble = ensemble.append(trajectory.dynamics)
        trajectory.metadata['trajectory number'].append(traj)



    trajectory.metadata['time to target find average'], trajectory.metadata['number of successes'] = T_find_stats(trajectory.metadata['time to target find'])
    
    return ensemble, trajectory.metadata


def T_find_stats(t_targfinds):
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

    
def main(agent_pos="cage", v0_stdev=0.01, k=0., beta=4e-6, rf=3e-6, wtf=7e-7, target_pos="left", Tmax=15.0, dt=0.01, total_trajectories=2, detect_thresh=0.023175, wallF=None, bounded=True, bounce="crash", plotting = True):
#    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, ensemble = trajGenIter(r0=r0, target_pos="left", v0_stdev=v0_stdev, k=k, beta=beta, rf=rf, wtf=wtf, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories, bounded=bounded, detect_thresh=detect_thresh)   # defaults
    ensemble, metadata = trajGenIter(agent_pos=agent_pos, target_pos=target_pos, v0_stdev=v0_stdev, k=k, beta=beta, rf=rf, wtf=wtf, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories, wallF=wallF, bounded=bounded,  bounce=bounce, detect_thresh=detect_thresh)


#    if plotting is True:
#         plot all trajectories
#        plotting_funcs.trajectory_plots(ensemble, metadata, heatmap=True)
#         plot histogram of pos, velo, accel distributions
#        plotting_funcs.stateHistograms(ensemble, metadata)

    return ensemble, metadata


if __name__ == '__main__':
    # wallF params
    wallF_max=5e-7
    decay_const = 250
        
    # center repulsion params
    b = 4e-1  # determines shape
    shrink = 5e-7  # determines size/magnitude
    
    wallF = (b, shrink, wallF_max, decay_const)
    
    ensemble, metadata = main(wallF=wallF)
    
