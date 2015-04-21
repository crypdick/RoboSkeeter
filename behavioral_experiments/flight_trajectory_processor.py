# -*- coding: utf-8 -*-
"""
Process a single mosquito flight trajectory


Input:
Take .csv file

Processing::
File sanity check: make sure we have camxy, date, video, raw
^^ IGNORE for now ^^

trim ends

Split the trajectories: if the mosquito tracker loses the mosquito OR if the
tracker code bugs out and gets stuck. Break threshold is 0.5sec (timebins are 
10ms each, so we need 50 datapoints/NaNs in a row).

Output:
trajectory data as .csv file
** Output form? (x1, y1, z1), (x2, y2, z2), ...

Created on Fri Mar 13 14:30:42 2015
@author: Richard Decal, decal@uw.edu

Field info for Sharri's video mat files (ignore this for CSVs)
camxy - x and y positions in each camera (size n x 4, where n is the number of timesteps)
date - string of the recording date
video - string name of the recording file (avi)
raw - 3D position of the trajectory. (n x 3, where n is the number of timesteps)
"""

import os
import pandas as pd
import numpy as np


#def sanitychecks(full_trajectory):
#    """test if the csv file has camxy, date, video, raw
#    
#    TODO:Implement
#    """
#    return full_trajectory


#def trim_NaNs(filt, trim='fb'):
#    """
#    Trim the leading and/or trailing NaNs from a 1-D array or sequence.
#    Parameters
#    ----------
#    filt : 1-D array or sequence
#        Input array.
#    trim : str, optional
#        A string with 'f' representing trim from front and 'b' to trim from
#        back. Default is 'fb', trim NaNs from both front and back of the
#        array.
#    Returns
#    -------
#    trimmed : 1-D array or sequence
#        The result of trimming the input. The input data type is preserved.
#    Examples
#    --------
#    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
#    >>> np.trim_nans(a)
#    array([1, 2, 3, 0, 2, 1])
#    >>> np.trim_nans(a, 'b')
#    array([0, 0, 0, 1, 2, 3, 0, 2, 1])
#    The input data type is preserved, list/tuple in means list/tuple out.
#    >>> np.trim_nans([0, 1, 2, 0])
#    [1, 2]
#    """
#    first = 0
#    trim = trim.upper()
#    if 'F' in trim:
#        for i in filt:
#            if i != np.nan:
#                break
#            else:
#                first = first + 1
#    last = len(filt)
#    if 'B' in trim:
#        for i in filt[::-1]:
#            if i != np.nan:
#                break
#            else:
#                last = last - 1
#    return filt[first:last]
    
    
def trim_NaNs(filt, trim='fb'):
    """
    Trim the leading and/or trailing NaNs from a 1-D array or sequence.
    Parameters
    ----------
    filt : 1-D array or sequence
        Input array.
    trim : str, optional
        A string with 'f' representing trim from front and 'b' to trim from
        back. Default is 'fb', trim NaNs from both front and back of the
        array.
    Returns
    -------
    trimmed : 1-D array or sequence
        The result of trimming the input. The input data type is preserved.
    Examples
    --------
    >>> a = np.array((0, 0, 0, 1, 2, 3, 0, 2, 1, 0))
    >>> np.trim_nans(a)
    array([1, 2, 3, 0, 2, 1])
    >>> np.trim_nans(a, 'b')
    array([0, 0, 0, 1, 2, 3, 0, 2, 1])
    The input data type is preserved, list/tuple in means list/tuple out.
    >>> np.trim_nans([0, 1, 2, 0])
    [1, 2]
    """
    first = 0
    trim = trim.upper()
    if 'F' in trim:
        for i in filt:
            if np.isnan(i) == False:
                break
            else:
                first = first + 1
    last = len(filt)
    if 'B' in trim:
        for i in filt[::-1]:
            if np.isnan(i) == False:
                break
            else:
                last = last - 1
    return first, last
#    return filt[first:last]


def trim_ends(full_trajectory):
    trimmed_trajectory = full_trajectory
    return trimmed_trajectory    
    

def split_trajectories(full_trajectory):
    """split if we have too many NaNs or if the mosquito is stuck
    If len(NaN segment) >= threshold, split trajectory
    else, use cubic interpolator to estimate values and replace NaNs
    """
    split_thresh = 50
    min_trajectory_len = 20
    trajectory_list = [full_trajectory]  # TODO: fix
    # toss short trajectories
    trajectory_list = [ trajectory for trajectory in trajectory_list if len(trajectory) > min_trajectory_len ]
    return trajectory_list


def write_csv(trajectory_list):
    """Outputs x,y,z coords at each timestep to a csv file. These trajectories
    will still contain short NaN repeats, but Sharri will fix that downstream
    using her interpolating code. She will also Kalman filter.
    """
    pass


def main(filename):
    # load the csv
    script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
    rel_data_path = "trajectory_data/"
    file_path = os.path.join(script_dir, rel_data_path, filename)
    Data = pd.read_csv(file_path, header=None, na_values="nan")
    Data.columns = ['x','y','z']
    first, last = trim_NaNs(Data['x'])
    trimmed_Data = Data[first : last]
##        full_trajectory = sanitychecks(full_trajectory)
##        
#        
    return trimmed_Data
#        trajectory_list = split_trajectories(trimmed_trajectory)
#    write_csv(trajectory_list)


if __name__ == "__main__":
    thing = main("195511-1.csv")