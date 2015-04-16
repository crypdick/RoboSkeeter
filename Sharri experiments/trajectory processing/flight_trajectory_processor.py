# -*- coding: utf-8 -*-
"""
Process mosquito flight trajectories.

Sharri's .mat files have these variables:
['__version__', '__header__', 'store', '__globals__']

The flight data is inside 'store', stored as a few MatLab "classes" (unlabeled):
camxy - x and y positions in each camera (size n x 4, where n is the number of timesteps)
date - string of the recording date
video - string name of the recording file (avi)
raw - 3D position of the trajectory. (n x 3, where n is the number of timesteps)

Input:
Take .mat file, extract trajectory data. Eventually, take whole folder.

Processing:
Sanity check: test if the mat file has camxy, date, video, raw

If trajectory contains long string of repeated points, throw out that deta.  This is the tracker malfunctioning.

If trajectory contains NaNs:
    If len(NaN segment) >= threshold, split trajectory
    else, use cubic interpolator to estimate values and replace NaNs

Use filter to smooth out noise in continuous trajectories

Output:
trajectory data as .csv file

Created on Fri Mar 13 14:30:42 2015
@author: Richard Decal, decal@uw.edu
"""

from scipy import io

vid_data_path = "v192836.mat"

vid_data = io.loadmat(vid_data_path)  # load mat-file

#The trajectory as an Nx4 matrix. Each columns represent TODO
trajectory_data = vid_data['store'][0][0][0]

raw_trag_data = vid_data['store']['raw'][0][0]
