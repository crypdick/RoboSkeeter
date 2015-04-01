# -*- coding: utf-8 -*-
"""
Creates a grid and places the stimulus in each grid cell. Then, runs many flight
trajectories for each stimulus location and displays the probabiltiy P_find 
at each cell as a heatmap.

Created on Wed Mar 25 09:59:21 2015

@author: richard
"""
import trajectory_stats
import numpy as np
import matplotlib.pyplot as plt

# number of sections to divide flight arena into
Nx_bins = 30
Ny_bins = 10

# define boundaries of flight arena for which to make heatmap
xbounds = (0, 1)
ybounds = (0.08, -0.08)  # reverse sign to go from top left to bottom right
TRAJECTORIES_PER_BIN = 15

# initialize empty counts
src_counts = np.zeros((Ny_bins, Nx_bins))
src_probs = np.zeros((Ny_bins, Nx_bins))

# figure out spot locations. [1:-1] throws out the first and last items in the
# arrays, since we don't want to put the stimulus inside the walls.
xticks = np.linspace(*xbounds, num=Nx_bins+2)[1:-1]
yticks = np.linspace(*ybounds, num=Ny_bins+2)[1:-1]

# generate our list of target coordinates and save them along with their
# index.
spotCoordsList = []
for j in range(Ny_bins):
    for i in range(Nx_bins):
        spotCoordsList.append((i, j, xticks[i], yticks[j]))
        
# ...and reshape it.
spotCoordsGrid = np.reshape(spotCoordsList, (Ny_bins, Nx_bins, 4))

# detections are based on a radius around the target. This radius shrinks
# if we add more x,y bins to reduce overlap.
# Calculated based on distance between diagonal spots / 2
detect_thresh = (np.linalg.norm((spotCoordsGrid[0, 0][2:] - spotCoordsGrid[1, 1][2:]) / 2))

# iter through spotCoordsGrid, run trajectory_stats TRAJECTORIES_PER_BIN times
# for each spot, and fill in resulting stats into the src_counts and _probs grids
for row in spotCoordsGrid:
    for spot in row:
        x_index, y_index, x_coord, y_coord = spot
        _, _, _, target_finds, t_targfinds, _, num_success, trajectory_objects_list = trajectory_stats.main(target_pos=[x_coord, y_coord], total_trajectories=TRAJECTORIES_PER_BIN, detect_thresh=detect_thresh, plotting=False)
        src_counts[int(y_index), int(x_index)] += num_success
        src_probs[int(y_index), int(x_index)] += num_success / TRAJECTORIES_PER_BIN

# plot the heatmap
plt.pcolormesh(src_probs)
titleappend = str(TRAJECTORIES_PER_BIN)
plt.title("""Probabilty of flying to target for different target positions \n
n = """ + titleappend)
plt.xlabel("X bounds = " + str(xbounds))
plt.ylabel("Y bounds = " + str(ybounds))
plt.savefig("./figs/Pfind_heatmap.png")
plt.show()
