# -*- coding: utf-8 -*-
"""
Creates a grid and places the stimulus in each grid cell. Then, runs many flight
trajectories for each stimulus location and displays the probabiltiy P_find 
at each cell as a heatmap.

Created on Wed Mar 25 09:59:21 2015

@author: richard
"""
import traj_gen
import numpy as np
import matplotlib.pyplot as plt

# sources from 0 -> 0.5 in x,
# +- -.25 in y
Nx_bins = 3
Ny_bins = 4
xbounds = (0,20)#(0,1)
ybounds = (20,-20)#(0.08, -0.08)  # reverse sign to go from top left to bottom right

# initialize empty counts
src_counts = np.zeros((Nx_bins, Ny_bins))

# figure out spot locations
xticks = np.linspace(*xbounds, num=Nx_bins+2)[1:-1]
yticks = np.linspace(*ybounds, num=Ny_bins+2)[1:-1]

# generate spotCoordsList, spotsgrid
spotCoordsList = []
for j in range(Ny_bins):
    for i in range(Nx_bins):
        spotCoordsList.append([xticks[i], yticks[j]])
spotCoordsGrid = np.reshape(spotCoordsList, (Nx_bins, Ny_bins, 2))

detect_thresh = (np.linalg.norm((spotCoordsGrid[0, 0] - spotCoordsGrid[1, 1])/2))

gridIter = np.nditer(spotCoordsGrid, flags=['multi_index'])
while not gridIter.finished:
    print "%d <%s>" % (gridIter[0], gridIter.multi_index), gridIter.iternext()
#    
#    for trial in range(2):
#        _, _, _, _, source_found, tfound = traj_gen.traj_gen(
#            [1., 0], #r0
#            rs = rs,
#            k = 2e-5,
#            beta = 2e-5,
#            f0 = 7e-6,
#            wf0 = 1e-6,
#            Tmax = 4.0,
#            dt = 0.01,
#            detect_thresh = detect_thresh)
#        if source_found is True:
#            pass

# TODO: change detection limit
#given 

#def probFindGrid(source_finds):
#    for i in range(Nx_bins):
#        for j in range (Ny_bins):
#            foundCounts = trajGenIter()
#            src_counts[i,j] += foundCounts


#def trajGenIter(r0, v0, k, beta, f0, wf0, rs, Tmax, dt, total_trajectories):
#    """
#    run traj_gen total_trajectories times and return arrays
#    """
#    pos = []
#    velos = []
#    accels = []
#    source_finds = []
#    t_finds = []
#    agent_paths_fig = plt.figure(1)
#
#    for i in range(total_trajectories):
#        t, r, v, a, source_found, tfound = traj_gen.traj_gen(r0=r0, v0=v0, k=k, beta=beta, f0=f0, wf0=wf0, rs=rs, Tmax=Tmax, dt=dt)
#        pos += [r]
#        velos += [v]
#        accels += [a]
#        source_finds += [source_found]
#        t_finds += [tfound]
#        plt.plot(r[:, 0], r[:, 1], lw=2, alpha=0.5)
#        plt.scatter(rs[0], rs[1], s=150, c='r', marker="*")
#    plt.title("agent trajectories")  # TODO: list params
#    plt.savefig("agent trajectories.png")
#    plt.show()    
#
#    return pos, velos, accels, source_finds, np.array(t_finds)