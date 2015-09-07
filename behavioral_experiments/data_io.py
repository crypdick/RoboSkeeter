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

    
def load_experiment_csv(file_path):
    """loads csv as-is"""
#    file_path = os.path.join(script_dir, rel_data_path, filename)
    return pd.read_csv(file_path, header=None, na_values="NaN")
    
    
def load_histogram_csv(filepath):
    pass


def save_processed_csv(trajectory_list, filepath):
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
        
        
def load_csv2DF(data_fname):
    dyn_traj_reldir = "data/processed_trajectories/"
    # print os.path.dirname(__file__)
    file_path = os.path.join(os.getcwd(), dyn_traj_reldir, data_fname + ".csv")

    col_labels = [
        'pos_x',
        'pos_y',
        'pos_z',
        'velo_x',
        'velo_y',
        'velo_z',
        'accel_x',
        'accel_y',
        'accel_z',
        'heading_angle',
        'angular_velo_xy',
        'angular_velo_yz',
        'curvature'
    ]

    dataframe = pd.read_csv(file_path, na_values="NaN", names=col_labels)  # recognize string as NaN
    dataframe.fillna(value=0, inplace=True)

    return dataframe


if __name__ == '__main__':
    a = load_csv2DF('Control-27')
