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
import seaborn as sns


def trajGenIter(agent_pos, target_pos, v0_stdev, k, beta, rf, wtf, Tmax, dt, total_trajectories, wallF, stimF_str, bounded, bounce, detect_thresh):
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
    traj_count = 0
    while traj_count < total_trajectories:
        trajectory = generate_trajectory.Trajectory(agent_pos=agent_pos, target_pos=target_pos, v0_stdev=v0_stdev, k=k, beta=beta, rf=rf, wtf=wtf, Tmax=Tmax, dt=dt, detect_thresh=detect_thresh, wallF=wallF, stimF_str=stimF_str, bounded=bounded, bounce=bounce)
        #throw out trajectories with huge accelerations        
        if trajectory.dynamics['acceleration_y'].max(axis=0) > 30.:
            continue
        # extract trajectory object attribs, append to our lists.
        trajectory.dynamics['trajectory'] = traj_count
        traj_count += 1
#        trajectory.dynamics.set_index('trajectory', append=True, inplace=True)
        ensemble = ensemble.append(trajectory.dynamics)
        

    trajectory.metadata['total_trajectories'] = total_trajectories
    trajectory.metadata['time_target_find_avg'] = T_find_stats(trajectory.metadata['time_to_target_find'])
    
    return ensemble, trajectory.metadata


def T_find_stats(t_targfinds):
    """Target finding stats. Returns average time to find the target, and the
    number of successes"""
    t_targfinds = np.array(t_targfinds)
    t_finds_NoNaNs = t_targfinds[~np.isnan(t_targfinds)]  # remove NaNs
    if len(t_finds_NoNaNs) == 0:

        return np.nan  # no target finds, no stats to run
    else:
        num_success = float(len(t_finds_NoNaNs))
        Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)

        return Tfind_avg

    
def main(agent_pos="cage", v0_stdev=0.01, k=0., beta=4e-6, rf=3e-6, wtf=7e-7, target_pos="left", Tmax=15.0, dt=0.01, total_trajectories=2, detect_thresh=0.023175, wallF=(4e-1, 1e-6, 1e-7, 250), stimF_str = 0., bounded=True, bounce="crash", plot_kwargs={'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True}):
#    pos, velos, accels, target_finds, t_targfinds, Tfind_avg, num_success, ensemble = trajGenIter(r0=r0, target_pos="left", v0_stdev=v0_stdev, k=k, beta=beta, rf=rf, wtf=wtf, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories, bounded=bounded, detect_thresh=detect_thresh)   # defaults
    ensemble, metadata = trajGenIter(agent_pos=agent_pos, target_pos=target_pos, v0_stdev=v0_stdev, k=k, beta=beta, rf=rf, wtf=wtf, Tmax=Tmax, dt=dt, total_trajectories=total_trajectories, wallF=wallF, bounded=bounded,  bounce=bounce, detect_thresh=detect_thresh, stimF_str=stimF_str)

    if plot_kwargs['heatmap'] is True:
#         plot all trajectories
        plotting_funcs.trajectory_plots(ensemble, metadata, plot_kwargs=plot_kwargs)
    if plot_kwargs['singletrajectories'] is True:
        plotting_funcs.plot_single_trajectory(ensemble, metadata, plot_kwargs=plot_kwargs)
    if plot_kwargs['states'] is True:
#         plot histogram of pos, velo, accel distributions
        plotting_funcs.stateHistograms(ensemble, metadata, plot_kwargs=plot_kwargs)
    if plot_kwargs['force_scatter'] is True:
        plotting_funcs.force_scatter(ensemble)
        

    return ensemble, metadata


if __name__ == '__main__':
    # wallF params
    wallF_max=9e-6#1e-7
    decay_const = 90
        
    # center repulsion params
    b = 4e-1  # determines shape
    shrink = 1e-6  # determines size/magnitude
    
    wallF = (b, shrink, wallF_max, decay_const)
    
    # beta-- search for kinematic dynamic viscosity of air
    # 10-5
    # rads of mosquito is 2e-3
    # so .036 e -6
    #"stokes flow" occurs when 
    # don't let it go below aerodynamic damping? make it lower bound
    
    plot_kwargs={'trajectories':False, 'heatmap':True, 'states':False, 'singletrajectories':False, 'force_scatter':True}
    ensemble, metadata = main(wallF=wallF, plot_kwargs=plot_kwargs, beta = .4e-6)
    
#    
#    
#  normal beta=1e-5  
# undamped beta=0
#  critical damping beta=1
#       slightly more interesting beta=0.1
#
#    
    
