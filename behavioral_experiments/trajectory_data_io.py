# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 17:21:42 2015

@author: richard
"""
import os
import pandas as pd

script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
rel_data_path = "trajectory_data/"
    
    
def load_csv(filename):
    file_path = os.path.join(script_dir, rel_data_path, filename + ".csv")
    return pd.read_csv(file_path, header=None, na_values="NaN")


def write_csv(trajectory_list, filename_prefix):
    """Outputs x,y,z coords at each timestep to a csv file. These trajectories
    will still contain short NaN repeats, but Sharri will fix that downstream
    using her interpolating code. She will also Kalman filter.
    """
    processed = "Processed/"
    for i, trajectory in enumerate(trajectory_list):
        file_path = os.path.join(script_dir, rel_data_path, processed, filename_prefix + "_" + "SPLIT" + str(i))
        trajectory.to_csv(file_path)