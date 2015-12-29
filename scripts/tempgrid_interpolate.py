# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 12:22:56 2015

@author: Sharri
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# opens data file with time averaged temperature data (m x 4 matrix: xpos, ypos, zpos, temp)
tempdata = np.genfromtxt(
    'C:/Users/Sharri/Dropbox/Le grand dossier du Sharri/Data/Temperature Data/Timeavg_right_plume.csv',
    delimiter=',')

# uniqe positions of recorded temperature
xpt = sort(np.unique(tempdata[:, 0]));
ypt = sort(np.unique(tempdata[:, 1]));
zpt = sort(np.unique(tempdata[:, 2]));

grid_x, grid_y, grid_z = np.meshgrid(xpt, ypt, zpt);

# time averaged temperature in grid shape (18,14,8 for right data, which is what i'm building this with)
grid_temp = np.genfromtxt(
    'C:/Users/Sharri/Dropbox/Le grand dossier du Sharri/Data/Temperature Data/grid_right_plume.csv',
    delimiter=',');
grid_temp = reshape(grid_temp, (18, 14, 8))

# create function to interpolate new data points
interp_func = RegularGridInterpolator((ypt, xpt, zpt), grid_temp)

# Before interpolating, get rid of NaNs. (This is the problem)
# Note: I had attempted to solve this problemt by interpolating planes along the length of the measured volume. No dice.

# Replace NaNs with interpolated values
nans = np.isnan(grid_temp)  # bool of nan locations in gridspace
nan_locations = grid_temp[nonzero(grid_temp)]  # indices of nans

filler = interp_func(nan_locations, method='nearest')  # cubic is best
grid_temp[nans] = interp(nan_locations(nans), nan_locations(~nans), grid_temp[~nans])

# Create finer mesh of regularly spaced points at which we'll interpolate, roughly 1" volume
finemesh_x = linspace(np.min(xpt), np.max(xpt), 40)
finemesh_y = linspace(np.min(ypt), np.max(ypt), 25)
finemesh_z = linspace(np.min(zpt), np.max(zpt), 20)
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

# Image spatial grdient
fig = plt.figure()
ax.fig.gca(projection='3d')

ax.quiver(finemesh_x, finemesh_y, finemesh_z, spatial_grad_x, spatial_grad_y, spatial_grad_z)

plt.show()
