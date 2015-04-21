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
#            if np.isnan(i) == False:
#                break
#            else:
#                first = first + 1
#    last = len(filt)
#    if 'B' in trim:
#        for i in filt[::-1]:
#            if np.isnan(i) == False:
#                break
#            else:
#                last = last - 1
#    return first, last
##    return filt[first:last]
    

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
    filtx = filt['x']  # we assume that if there's a NaN  in the x col the 
                    # rest of that row will also be NaNs
    first = 0
    trim = trim.upper()
    if 'F' in trim:
        for i in filtx:
            if np.isnan(i) == False:
                break
            else:
                first += 1
    last = len(filt)
    if 'B' in trim:
        for i in filtx[::-1]:
            if np.isnan(i) == False:
                break
            else:
                last -= 1
    return filt[first:last].reset_index()


def split_trajectories(full_trajectory, split_thresh=50, min_trajectory_len=20):
    """split if we have too many NaNs or if the mosquito is stuck
    If len(NaN segment) >= threshold, split trajectory
    else, use cubic interpolator to estimate values and replace NaNs
    
    use pandas.DataFrame.duplicated for stuck bug
    
    firstN = 0
    NaNcount=0
    inNaNs = False
    for i, x in enumerate(thing['x']):
        isNaN[i] == True:
            
            if 
            check if last position isNan was True. if so, record last NaN.
        loop through x until isNaN = true. set inNan=true
        if inNan[t-1]=False
            record where we encounter the first nan.
            start NaNcount=1
        
        while Nans, increment NaNcount, inNan=true
        if numer:
            record where we find the lastNaN, end NaNcount, inNan false
        if the first counter when we find our first nan.
    """
    xs = full_trajectory['x']  # we assume that if there's a NaN  in the x col the 
                    # rest of that row will also be NaNs
        # .reset_index() gets the index back at 0
    inNan = False
    firstN = None
    lastN = None
    firstNaN = None
    lastNaN = None
    Nancount = 0
    for i, x in enumerate(xs):
        if np.isnan(xs[i]) == False: # number
            if firstN = None:  # found our first number
                firstN = i+1  # +1 for python indexing
            if inNan is False: #
                pass
            if inNan is True:
                pass
        else: # found a NaN
            if inNan is False: # we discovered our first Nan, enter NaNs
                firstNaN = i + 1
                NaNcount += 1
                lastN = firstNaN - 1
                inNan = True
            if inNan is True: # we're in a Nansequence
                pass # TODO
            
    
    
    
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
    trimmed_Data = trim_NaNs(Data)
##    sanitychecks(trimmed_Data)
#    trajectory_list = split_trajectories(trimmed_Data)
#    for trajectory in trajectory_list:
#        # save a separate csv for each
#        pass
##
    thing = trimmed_Data['x']
    return thing
#        
#    write_csv(trajectory_list)


if __name__ == "__main__":
    thing = main("195511-1.csv")