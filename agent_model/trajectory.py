# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:15:28 2015

@author: richard

creates a trajectory object that has a bunch of sweet methods.
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
        
        
    def plot_single_trajectory(self, trajectory_i=0):
        plot_kwargs = {'title':"Individual agent trajectory", 'titleappend':'( #{})'.format(trajectory_i)}
        plotting_funcs.plot_single_trajectory(self.ensemble.loc[self.ensemble['trajectory_num']==trajectory_i], self.agent_info, plot_kwargs)
    

    def add_agent_info(self, data_dict):
        for key, value in data_dict.iteritems():
            self.agent_info[key] = value
            
    def plot_force_violin(self):
        plotting_funcs.force_violin(self.ensemble, self.agent_info)
        
    
    def plot_posheatmap(self):
        plotting_funcs.heatmaps(self.ensemble, self.agent_info)
        
    
    def plot_kinematic_hists(self, ensemble='none', titleappend=''):
        print titleappend
        if type(ensemble) is str:
            ensemble = self.ensemble
            ensemble = ensemble.loc[(ensemble['position_x'] >0.25) & (ensemble['position_x'] <0.95)]
        plotting_funcs.stateHistograms(ensemble, self.agent_info, titleappend)
        
    def plot_door_velocity_compass(self, region='door', kind='bin_average'):
        
        if region == 'door':
            """plot the area """
            ensemble = self.ensemble.loc[((self.ensemble['position_x']>0.25) & (self.ensemble['position_x']<0.5)), ['velocity_x', 'velocity_y']]
        else:
            ensemble = self.ensemble
            
        plotting_funcs.compass_plots(ensemble, self.agent_info, kind)
      
      
        
        
    def plot_sliced_hists(self):
        """Plot histograms from 0.25 < x < 0.95, as well as that same space
        divided into 4 equal 0.15 m segments 
        """
        x_edges = np.linspace(.25,.95,5)
        
        print "full ensemble"
        full_ensemble = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[0]) \
            & (self.ensemble['position_x'] < x_edges[4])]
        self.plot_kinematic_hists(ensemble=full_ensemble, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[4]))
        
        print "downwind half ensemble"
        downwind_ensemble = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[0]) \
            & (self.ensemble['position_x'] < x_edges[2])]
        self.plot_kinematic_hists(ensemble=downwind_ensemble, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[2]))

        print "upwind half ensemble"
        upwind_ensemble = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[2]) \
            & (self.ensemble['position_x'] < x_edges[4])]
        self.plot_kinematic_hists(ensemble=upwind_ensemble, titleappend=' {} < x <= {}'.format(x_edges[2], x_edges[4]))    
#        
#        print "first quarter ensemble"
#        ensemble1 = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[0]) \
#            & (self.ensemble['position_x'] <= x_edges[1])]
#        self.plot_kinematic_hists(ensemble=ensemble1, titleappend=' {} < x <= {}'.format(x_edges[0], x_edges[1]))
#        
#        print "second quarter ensemble"
#        ensemble2 = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[1]) \
#            & (self.ensemble['position_x'] <= x_edges[2])]
#        self.plot_kinematic_hists(ensemble=ensemble2, titleappend=' {} < x <= {}'.format(x_edges[1], x_edges[2]))
#        
#        print "third quarter ensemble"
#        ensemble3 = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[2]) \
#            & (self.ensemble['position_x'] <= x_edges[3])]
#        self.plot_kinematic_hists(ensemble=ensemble3, titleappend=' {} < x <= {}'.format(x_edges[2], x_edges[3]))
#
#        print "fourth quarter ensemble"
#        ensemble4 = self.ensemble.loc[(self.ensemble['position_x'] > x_edges[3]) \
#            & (self.ensemble['position_x'] <= x_edges[4])]
#        self.plot_kinematic_hists(ensemble=ensemble4, titleappend=' {} < x <= {}'.format(x_edges[3], x_edges[4]))
#        
        print "full- + up- + down-wind"
        plotting_funcs.stateHistograms(full_ensemble, self.agent_info, titleappend = '', upw_ensemble = upwind_ensemble, downw_ensemble = downwind_ensemble)
        
            
            
    def describe(self, plot_kwargs={'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_violin':True}):
        print self.ensemble.describe()
        if plot_kwargs['heatmap'] is True:
            self.plot_posheatmap()
        if plot_kwargs['singletrajectories'] is True:
            self.plot_single_trajectory()
        if plot_kwargs['states'] is True:
    #         plot histogram of pos, velo, accel distributions
            self.plot_kinematic_hists()
        if plot_kwargs['force_violin'] is True:
            self.plot_force_violin()
            
    def dump2csvs(self):
        for trajectory_i in self.ensemble.trajectory_num.unique():
            temp_traj = self.ensemble[self.ensemble['trajectory_num'] == trajectory_i]
            temp_array = temp_traj[['position_x', 'position_y']].values
            np.savetxt(str(trajectory_i) + ".csv", temp_array, delimiter=",")
    
#    def 