# -*- coding: utf-8 -*-
"""
Flight statistics heatmap.

Positional distributions in each plane (x, y, z)
Distributions of velocity components and magnitude  (x, y, z, |v|)
Distributions of acceleration components and magnitude (x,y,z,|a|)

Divide the flight arena into a grid, and see the statistics of P(find) and 
<time_find> given that the stimulus was placed in a given grid cell. 

Then, I'm going to plot an x-y heatmap of P(find) and <Time_find> given that the stimulus was placed in that grid cell.

top level: iterate through each row, column in the grid and place the stimulus
in it.

run the trajectory until it either runs out of time, or finds the stim.
for each trajectory, append [found(true or false) NaN]
 update
total trajectory counts and success count.



Created on Thu Mar 19 14:19:37 2015
@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
# set bin limits, bin centers
# store trajectories separateley
# then sum up, normalize by total counts

import oscillator

states = oscillator.main(runtime = 1e3, plotting=True) #work towards 2e3 

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter


#map np.array of dimensions

x_positions = states[:, 0]
x_velocities = states[:, 1]
y_positions = states[:, 2]
y_velocities = states[:, 3]

## plot position distributions##
xy_binwidth = 0.5
position_lim = 5.0
positional_bins = np.arange(-position_lim, position_lim + xy_binwidth, xy_binwidth) #specify bin locations

#x dimension
plt.figure(3)
plt.hist(x_positions, bins=positional_bins)
plt.title("x position distributions")

#y dimension
plt.figure(4)
plt.hist(y_positions, bins=positional_bins, orientation='horizontal', color = 'r')
plt.title("y position distributions")

##plot velocity distributions##
velo_binwidth = 0.01
velo_lim = 0.12
velo_bins = np.arange(-velo_lim, velo_lim + velo_binwidth, velo_binwidth)

#x velo dimension
plt.figure(5)
plt.hist(x_velocities, bins=velo_bins, color='g')
plt.title("x velocity distributions")

#y velocity dim
plt.figure(6)
plt.hist(x_velocities, bins=velo_bins, orientation='horizontal', color = 'cyan')
plt.title("y velocity distributions")

plt.show()