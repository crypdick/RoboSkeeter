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
#    >>> np.trim_NaNs(a)
#    array([1, 2, 3, 0, 2, 1])
#    >>> np.trim_NaNs(a, 'b')
#    array([0, 0, 0, 1, 2, 3, 0, 2, 1])
#    The input data type is preserved, list/tuple in means list/tuple out.
#    >>> np.trim_NaNs([0, 1, 2, 0])
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
    >>> np.trim_NaNs(a)
    array([1, 2, 3, 0, 2, 1])
    >>> np.trim_NaNs(a, 'b')
    array([0, 0, 0, 1, 2, 3, 0, 2, 1])
    The input data type is preserved, list/tuple in means list/tuple out.
    >>> np.trim_NaNs([0, 1, 2, 0])
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
    return filt[first:last].reset_index()[['x', 'y', 'z']]


def split_trajectories(full_trajectory, NaN_split_thresh=50, min_trajectory_len=20):
    """split if we have too many NaNs or if the mosquito is stuck
    If len(NaN segment) >= threshold, split trajectory
    else, use cubic interpolator to estimate values and replace NaNs
    
    use pandas.DataFrame.duplicated for stuck bug
    
    firstN = 0
    NaNcount=0
    in_NaN_runs = False
    for i, x in enumerate(thing['x']):
        isnan[i] == True:
            
            if 
            check if last position isnan was True. if so, record last NaN.
        loop through x until isnan = true. set in_NaN_run=true
        if in_NaN_run[t-1]=False
            record where we encounter the first NaN.
            start NaNcount=1
        
        while NaNs, increment NaNcount, in_NaN_run=true
        if numer:
            record where we find the lastNaN, end NaNcount, in_NaN_run false
        if the first counter when we find our first NaN.
    """
    xs = full_trajectory['x']  # we assume that if there's a NaN  in the x col 
                    # the rest of that row will also be NaNs
    in_NaN_run = False
    firstN = None
    lastN = None
    firstNaN = None
    lastNaN = None
    NaNcount = 0
    split_trajectory_list = []
    for i, x in enumerate(xs):
        print "last N ", lastN
        # if x is a number
        if np.isnan(xs[i]) == False:
            # if starting a new trajectory
            if firstN is None:  # found our first number
                firstN = i
                print "firstN ",firstN
            # if continuing a number run
            if in_NaN_run is False:
                if i == len(xs)-1:  # loop reached end; grab last trajectory
                    if lastN is None:
                        lastN = i
                    print "the end!"
                    print full_trajectory[firstN:lastN+1]
                    split_trajectory_list.append(full_trajectory[firstN:lastN+1])
            # ending a NaN run, starting number run
            if in_NaN_run is True:
                lastNaN = i - 1
                # if splitting trajectory
                if NaNcount > NaN_split_thresh:
                    split_trajectory_list.append(full_trajectory[firstN:lastN+1])                 
                    print "new trajectory!"
                    in_NaN_run = False
                    firstN = i
                    lastN = None
                    firstNaN = None
                    lastNaN = None
                    NaNcount = 0
                else:
                    # not splitting; resume
                    lastN = None
                    firstNaN = None
                    lastNaN = None
                    NaNcount = 0
                    in_NaN_run = False
        # if x is a NaN
        else:
            # ending number run, starting NaN run
            if in_NaN_run is False:
                firstNaN = i
                NaNcount = 1
                lastN = firstNaN - 1  # exiting number run; storing lastN
                in_NaN_run = True
            # if continuing a NaN run
            if in_NaN_run is True:
                NaNcount += 1
        # for debugging
#        print "NaN_run {} firstN {} lastN {} firstNaN {} lastNaN {} Nancount {}".format(in_NaN_run, firstN, lastN, firstNaN, lastNaN, NaNcount)

    # toss short trajectories
    split_trajectory_list = [ trajectory for trajectory in split_trajectory_list if len(trajectory) > min_trajectory_len ]
    return split_trajectory_list


def write_csv(trajectory_list):
    """Outputs x,y,z coords at each timestep to a csv file. These trajectories
    will still contain short NaN repeats, but Sharri will fix that downstream
    using her interpolating code. She will also Kalman filter.
    """
    pass


def main(filename):
    if type(filename) == str:
        # load the csv
        script_dir = os.path.dirname(__file__) #<-- absolute dir the script is in
        rel_data_path = "trajectory_data/"
        file_path = os.path.join(script_dir, rel_data_path, filename)
        Data = pd.read_csv(file_path, header=None, na_values="NaN")
    else:
        # feed test Data into the module
        Data = filename
    Data.columns = ['x','y','z']
    trimmed_Data = trim_NaNs(Data)
##    sanitychecks(trimmed_Data)
    trajectory_list = split_trajectories(trimmed_Data)
#    for trajectory in trajectory_list:
#        # save a separate csv for each
#        pass
##
    thing = trajectory_list
    return thing
#        
#    write_csv(trajectory_list)


if __name__ == "__main__":
#    thing = main("195511-1.csv")
    thing = main(df(np.random.randn(40, 3), columns = ['x','y','z']))  # dummy data