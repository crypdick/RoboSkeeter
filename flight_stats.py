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

import oscillator

states = oscillator.main(plotting=True)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter




x_positions = states[:, 0]
x_velocities = states[:, 1]
y_positions = states[:, 2]
y_velocities = states[:, 3]


binwidth = 0.05
xymax = np.max( [np.max(x_positions), np.max(y_positions)] )
lim = ( int(xymax/binwidth) + 1) * binwidth

#plot x position distributions
plt.figure(3)
axScatter.set_xlim( (-lim, lim) )
axScatter.set_ylim( (-lim, lim) )
bins = np.arange(-lim, lim + binwidth, binwidth)
axHistx = plt.axes()
axHistx.hist(x_positions, bins=bins, normed=True)
plt.title("x position distributions")

#plot y position distributions
plt.figure(4)
axHisty = plt.axes()
axHisty.hist(y_positions, bins=bins, orientation='horizontal', color = 'r')
plt.title("y position distributions")

axHistx.set_xlim( axScatter.get_xlim() )
axHisty.set_ylim( axScatter.get_ylim() )

#plot x velocity distributions
plt.figure(5)
velo_binwidth = 0.001
velo_bins = np.arange(-velo_lim, velo_lim + velo_binwidth, velo_binwidth)
xy_velo_max = np.max( [np.max(x_velocities), np.max(y_velocities)] )
velo_lim = ( int(xy_velo_max/velo_binwidth) + 1) * velo_binwidth
axScatter.set_xlim( (-velo_lim, velo_lim) )
axScatter.set_ylim( (-velo_lim, velo_lim) )
axHistx = plt.axes()
axHistx.hist(x_velocities, bins=velo_bins, color='g')
plt.title("x velocity distributions")

#plot y velocity distributions
plt.figure(6)
axScatter.set_xlim( (-velo_lim, velo_lim) )
axScatter.set_ylim( (-velo_lim, velo_lim) )
axHistx = plt.axes()
axHistx.hist(x_velocities, bins=velo_bins, orientation='horizontal', color = 'cyan')
plt.title("y velocity distributions")

plt.show()