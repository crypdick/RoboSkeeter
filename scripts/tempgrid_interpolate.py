# -*- coding: utf-8 -*-
"""
Created on Wed Dec 09 12:22:56 2015

@author: Sharri
"""

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator, griddata
import pandas as pd


tempdata_fname = '/home/richard/src/RoboSkeeter/data/experiments/plume_data/thermocouple/timeavg_right/timeavg_right.csv'
# time averaged temperature data (m x 4 matrix: xpos, ypos, zpos, temp)
tempdata = np.genfromtxt(tempdata_fname, delimiter=',')

# windtunnel dimensions
#      self.left = -0.127
#      self.right = 0.127
#      self.upwind = 1.0
#      self.downwind = 0.0
#      self.ceiling = 0.25
#      self.floor = 0.
grid_x, grid_y, grid_z = np.mgrid[0.:1.:100j, -0.127:0.127:25j, 0:0.254:25j]
points = tempdata[:,:3]  # (1382, 3)
temps = tempdata[:, 3]  # len 1382

interpolated_temps = griddata(points,
                              temps,
                              (grid_x, grid_y, grid_z),
                              method='nearest')

# Solve for the spatial gradient
spatial_grad_x, spatial_grad_y, spatial_grad_z = np.gradient(interpolated_temps,
                                                             grid_x,
                                                             grid_y,
                                                             grid_z)

# norm = spatial_grad_x **2 + spatial_grad_y**2 + spatial_grad_z **2

# Visualize spatial gradient
fig = plt.figure()
ax = fig.gca(projection='3d')

ax.quiver(grid_x, grid_y, grid_z, spatial_grad_x, spatial_grad_y, spatial_grad_z, length=0.01)

plt.show()
