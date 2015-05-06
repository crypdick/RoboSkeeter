# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:15:28 2015

@author: richard

takes vectors from agent and adds it to ensemble

trajectory.ensemble
trajectory.append_ensemble(a trajectory)

trajectory.describe()
    agent info
    plots

trajectory.save
"""
import pandas as pd
import plotting_funcs
import numpy as np


def T_find_stats(t_targfinds):
    """Target finding stats. Returns average time to find the target, and the
    number of successes
    
    TODO:  find tfind stats for ensemble
    """
    t_targfinds = np.array(t_targfinds)
    t_finds_NoNaNs = t_targfinds[~np.isnan(t_targfinds)]  # remove NaNs
    if len(t_finds_NoNaNs) == 0:

        return np.nan  # no target finds, no stats to run
    else:
        num_success = float(len(t_finds_NoNaNs))
        Tfind_avg = sum(t_finds_NoNaNs)/len(t_finds_NoNaNs)

        return Tfind_avg


class Trajectory():
    def __init__(self):
        self.ensemble = pd.DataFrame()
        self.agent_info = {}
        
        
    def append_ensemble(self, arraydict):
        trajectory = pd.DataFrame(arraydict)
        self.ensemble = self.ensemble.append(trajectory)
        
        
    def plot_single_trajectory(self):
        plot_kwargs = {'title':"Individual agent trajectory", 'titleappend':''}
        plotting_funcs.plot_single_trajectory(self.ensemble.loc[self.ensemble['trajectory_num']==0], self.agent_info, plot_kwargs)
    

    def add_agent_info(self, data_dict):
        for key, value in data_dict.iteritems():
            self.agent_info[key] = value
            
            
    def describe(self, plot_kwargs={'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True}):
        print self.ensemble.describe()
        if plot_kwargs['heatmap'] is True:
  #         plot all trajectories
            plotting_funcs.trajectory_plots(self.ensemble, self.agent_info, plot_kwargs=plot_kwargs)
        if plot_kwargs['singletrajectories'] is True:
            plotting_funcs.plot_single_trajectory(self.ensemble, self.agent_info, plot_kwargs=plot_kwargs)
        if plot_kwargs['states'] is True:
    #         plot histogram of pos, velo, accel distributions
            plotting_funcs.stateHistograms(self.ensemble, self.agent_info, plot_kwargs=plot_kwargs)
        if plot_kwargs['force_scatter'] is True:
            plotting_funcs.force_violin(self.ensemble, self.agent_info)
    
#    def 