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
import numpy as np
import matplotlib.pyplot as plt
#from matplotlib.ticker import NullFormatter #will need for special plots -rd


#TODO: if trajectory is garbage or ODE solver throws "too much work done"
#                                       warning, discard that trajectory -rd

flight_runtime = 2e3  # careful! this is a float! -rd
total_trajectories = 5

flight_states = np.zeros((total_trajectories, flight_runtime, 4))
for i in range(total_trajectories):
    states = oscillator.main(runtime=flight_runtime, plotting=False)
    flight_states[i] = states

x_positions = []
x_velocities = []
y_positions = []
y_velocities = []

for trajectory in range(total_trajectories):
    for time in range(int(flight_runtime)):
        x, xv, y, yv = flight_states[trajectory, :][time]
        x_positions.append(x)
        x_velocities.append(xv)
        y_positions.append(y)
        y_velocities.append(yv)

## plot position distributions##
xy_binwidth = 0.5
position_lim = 5.0
positional_bins = np.arange(-position_lim, position_lim + xy_binwidth, xy_binwidth) #specify bin locations

#x dimension
x_fig = plt.figure(3)
plt.hist(x_positions, bins=positional_bins, normed=True)
plt.title("x position distributions")

#==============================================================================
# UNDER CONSTRUCTION: normalizing histograms
# #x dimension
# plt.figure(3)
# x_hist, bin_edges = np.histogram(x_positions, density = 1, bins=positional_bins)
# plt.bar(bin_edges[:-1], x_hist) #, width = 1)
# #sanity check: make sure sum is 1.0
# #np.sum(x_hist*np.diff(binedges))
# plt.title("x position distributions")
#==============================================================================


#y dimension
y_fig = plt.figure(4)
plt.hist(y_positions, bins=positional_bins, orientation='horizontal', color='r', normed=True)
plt.title("y position distributions")

##plot velocity distributions##
velo_binwidth = 0.01
velo_lim = 0.12
velo_bins = np.arange(-velo_lim, velo_lim + velo_binwidth, velo_binwidth)

#x velo dimension
xv_fig = plt.figure(5)
plt.hist(x_velocities, bins=velo_bins, color='g', normed=True)
plt.title("x velocity distributions")

#y velocity dim
yv_fig = plt.figure(6)
plt.hist(x_velocities, bins=velo_bins, orientation='horizontal', color='cyan', normed=True)
plt.title("y velocity distributions")

# TODO 

plt.show()
