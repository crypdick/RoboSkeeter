import numpy as np
from scipy.interpolate import Rbf
import pandas as pd
import mayavi
from mayavi import mlab
from scipy.stats import multivariate_normal


mlab.options.backend = 'envisage'

####### generate fake observations

# generate sampling coordinates
left = -3.
right = 3.
floor = -5
ceiling = 5
downwind = -2.
upwind = 2.

ground_tr_resolution = 0.7

x = np.arange(downwind, upwind, ground_tr_resolution)
y = np.arange(left, right, ground_tr_resolution)
z = np.arange(floor, ceiling, ground_tr_resolution)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
df_dict = dict()
df_dict['x'] = xx.ravel()
df_dict['y'] = yy.ravel()
df_dict['z'] = zz.ravel()
observations = pd.DataFrame(data=df_dict)

# generate temperatures with draws from 3d gaussian
xyz = np.column_stack([xx.flat, yy.flat, zz.flat])
mu = np.array([0.0, 0.0, 0.0])
sigma = np.array([1, 1, 1])
covariance = np.diag(sigma**2)
temp = multivariate_normal.pdf(xyz, mean=mu, cov=covariance)
observations['avg_temp'] = temp
temp_grid = temp.reshape(xx.shape)


# make interpolator
rbfi = Rbf(observations.x.values, observations.y.values, observations.z.values, observations.avg_temp.values,
           function='gaussian')

# define positions to interpolate (at higher resolution than ground truth, above)
interp_left = -2.
interp_right = 2.
interp_floor = -4
interp_ceiling = 4
interp_downwind = -1.
interp_upwind = 1.

interp_bounds = [interp_downwind, interp_upwind, interp_left, interp_right, interp_floor, interp_ceiling]

interp_res = ground_tr_resolution/2

# xi = np.arange(interp_downwind, interp_upwind, interp_res)
# yi = np.arange(interp_left, interp_right, interp_res)
# zi = np.arange(interp_floor, interp_ceiling, interp_res)
# xxi, yyi, zzi = np.meshgrid(xi, yi, zi, indexing='ij')
# xxi_flat = xxi.ravel()
# yyi_flat = yyi.ravel()
# zzi_flat = zzi.ravel()

# interpolate
# interp_temps = rbfi(xxi_flat, yyi_flat, zzi_flat)
interp_temps = rbfi(observations.x.values, observations.y.values, observations.z.values)
# tti = interp_temps.reshape((len(xi), len(yi), len(zi)))
tti = interp_temps.reshape(xx.shape)

# print """
#         Interpolated temp stats
#         min {}
#         max {}
#         avg {}
#         """.format(interp_temps.min(), interp_temps.max(), interp_temps.mean())
#

# Plot scatter with mayavi
"""
stackexchange question for cmaps: https://stackoverflow.com/questions/36946231/using-perceptually-uniform-colormaps-in-mayavi-volumetric-visualization
bug report for custom colormaps: https://github.com/enthought/mayavi/issues/371


"""


## visualize
gr_truth_fig = mlab.figure('Ground truth observations')


src = mlab.pipeline.scalar_field(temp_grid)
vol = mlab.pipeline.volume(src, vmin=temp.min(), vmax=temp.max()*.7)
mayavi.tools.pipeline.set_extent(mlab.pipeline.volume, interp_bounds)


# mlab.show()

###################
interp_fig = mlab.figure('Interpolation')

src = mlab.pipeline.scalar_field(tti)
vol = mlab.pipeline.volume(src, vmin=tti.min(), vmax=tti.max()*.7)

# mlab.axes(bounds=interp_bounds)

#########
mlab.axes()
mlab.show()
