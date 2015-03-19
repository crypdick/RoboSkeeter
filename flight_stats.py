# -*- coding: utf-8 -*-
"""
Flight statistics heatmap.

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

