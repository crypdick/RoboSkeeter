# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:21:42 2015

@author: richard
"""
import os
import pandas as pd
import numpy as np
import csv

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in

    
def load_csv(file_path):
#    file_path = os.path.join(script_dir, rel_data_path, filename)
    return pd.read_csv(file_path, header=None, na_values="NaN")
    
    
def load_histogram_csv(filepath):
    pass


def write_csv(trajectory_list, filepath):
    """Outputs x,y,z coords at each timestep to a csv file. These trajectories
    will still contain short NaN repeats, but Sharri will fix that downstream
    using her interpolating code. She will also Kalman filter.
    """
    dir = os.path.dirname(filepath)
    filename, extension = os.path.splitext ( os.path.basename(filepath) )
    for i, trajectory in enumerate(trajectory_list):
        trajectory = trajectory.fillna("NaN")  # hack to turn nan into string 
        # so that the csv doesn't have empty fields
        file_path = os.path.join(dir, "Processed/", filename + "_SPLIT_" + str(i))
        trajectory.to_csv(file_path, index=False)
        
        
def load_trajectory_dynamics_csv(data_fname):
    dyn_traj_reldir = "data/dynamical_trajectories/"
    file_path = os.path.join(os.path.dirname(__file__), dyn_traj_reldir, data_fname + ".csv")
    numrows = sum(1 for row in csv.reader(open(file_path)))
#    arrays = [np.tile(np.array([data_fname]), numrows),
#          np.arange(numrows)]
    dyn_trajectory_DF = pd.read_csv(file_path, header=None, na_values="NaN")#, index_col= arrays) index_col = ['Trajectory_ID', 'timestep'])#
    dyn_trajectory_DF.fillna(value=0, inplace=True)
#    dyn_trajectory_DF.index.name = 'timestep'
    dflen = len(dyn_trajectory_DF.index)
#    arrays=[np.array([data_fname]*numrows), np.arange(numrows)]
#    tuples = list(zip(*arrays))
#    index = MultiIndex.from_tuples(tuples, names=['first', 'second'])
    
    dyn_trajectory_DF.columns = ['pos_x','pos_y','pos_z', 'velo_x', 'velo_y',
        'velo_z', 'accel_x', 'accel_y', 'accel_z', 'angular_velo_xy',
        'angular_velo_yz', 'angular_velo_3D', 'heading_angle', 'curve']
    
    # add trajectory name to index
#    dyn_trajectory_DF['Trajectory'] = data_fname
#    dyn_trajectory_DF.set_index(index, append=True, inplac)
#    dyn_trajectory_DF.set_index('Trajectory', append=True)
    #'_x', '_y', '_z',
#    dyn_trajectory_DF.index.names[0] = 'Trajectory_ID'
#    dyn_trajectory_DF.index.names = ['timestep']
    
    new = pd.concat([dyn_trajectory_DF], keys=[data_fname])
    new.index.names = ['Trajectory', 'timestep']
    
    
    return new
#    return dyn_trajectory_DF
    
    
#    return dyn_trajectory_DF
a = load_trajectory_dynamics_csv('Control-27')