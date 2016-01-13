# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 12:22:56 2015

@author: Sharri
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

tempdata_fname = '/home/richard/src/RoboSkeeter/data/experiments/plume_data/thermocouple/timeavg_right/timeavg_right.csv'
# opens data file with time averaged temperature data (m x 4 matrix: xpos, ypos, zpos, temp)
tempdata = np.genfromtxt(tempdata_fname, delimiter=',')

# TODO: unique along axis
# unique positions of recorded temperature
x_coords = np.sort(np.unique(tempdata[:, 0])) # 14
y_coords = np.sort(np.unique(tempdata[:, 1])) # 18
z_coords = np.sort(np.unique(tempdata[:, 2]))  # len 8
temps = tempdata[:, 3]  # len 1382

grid_x, grid_y, grid_z = np.meshgrid(x_coords, y_coords, z_coords)

# time averaged temperature in grid shape (18,14,8 for right data, which is what i'm building this with)
# This file is missing
grid_temp = np.genfromtxt(
    'C:/Users/Sharri/Dropbox/Le grand dossier du Sharri/Data/Temperature Data/grid_right_plume.csv',
    delimiter=',')
grid_temp = np.reshape(grid_temp, (18, 14, 8))

# create function to interpolate new data points
interp_func = RegularGridInterpolator((y_coords, x_coords, z_coords), grid_temp)  # todo: should y come before x?

# Before interpolating, get rid of NaNs. (This is the problem)

# Replace NaNs with interpolated values
# TODO: check if filling nans is needed before making finemesh, or we can just drop
nan_truthtable = np.isnan(grid_temp)  # bool of nan locations in gridspace
nan_locations = grid_temp[np.nonzero(grid_temp)]  # indices of nans # TODO: WTF?

filler = interp_func(nan_locations, method='nearest')  # cubic is best
grid_temp[nan_truthtable] = interp_func(nan_locations(nan_truthtable), nan_locations(~nan_truthtable),
                                        grid_temp[~nan_truthtable])

# Create finer mesh of regularly spaced points at which we'll interpolate, roughly 1" volume
# TODO: make actual min max bounds of windtunnel
finemesh_x = np.linspace(np.min(x_coords), np.max(x_coords), 40)
finemesh_y = np.linspace(np.min(y_coords), np.max(y_coords), 25)
finemesh_z = np.linspace(np.min(z_coords), np.max(z_coords), 20)
# TODO: wtf Nones?
uniform_x, uniform_y, uniform_z = interp_func(finemesh_x[:, None, None],
                                              finemesh_y[None, :, None],
                                              finemesh_z[None, None, :])

# Interpolate finemesh grid
interp_grid = interp_func((tempdata[:, 0].ravel(), tempdata[:, 1].ravel(), tempdata[:, 2].ravel()),
                          tempdata[:, 3].ravel(),
                          (uniform_x, uniform_y, uniform_z))
# TODO: join filler and gridtemp or find a way to get interp_func to work with grids

# Solve for the spatial gradient
spatial_grad_x, spatial_grad_y, spatial_grad_z = np.gradient(interp_grid,
                                                             uniform_x,
                                                             uniform_y,
                                                             uniform_z)

# Image spatial gradient
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.quiver(finemesh_x, finemesh_y, finemesh_z, spatial_grad_x, spatial_grad_y, spatial_grad_z)

plt.show()
