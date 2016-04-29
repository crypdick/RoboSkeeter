import numpy as np
from scipy.interpolate import Rbf
import pandas as pd
from mayavi import mlab
from scipy.stats import multivariate_normal


# generate observations
left = -2.
right = 3.
floor = -5
ceiling = 5
bottom = -1.
top = 2.

resolution = 0.7

x = np.arange(bottom, top, resolution)
y = np.arange(left, right, resolution)
z = np.arange(floor, ceiling, resolution)

xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
df_dict = dict()
df_dict['x'] = xx.ravel()
df_dict['y'] = yy.ravel()
df_dict['z'] = zz.ravel()


observations = pd.DataFrame(data=df_dict)


xyz = np.column_stack([xx.flat, yy.flat, zz.flat])
mu = np.array([0.0, 0.0, 0.0])
sigma = np.array([1,1,1])
covariance = np.diag(sigma**2)

temp = multivariate_normal.pdf(xyz, mean=mu, cov=covariance)
temp = temp.reshape(xx.shape)



# # make some places hot
#
# temp = 19.
# observations['avg_temp'] = np.array([temp] * len(x) * len(y))  # make it the same temperature everywhere
# # observations.loc[((observations['x'] > -.5) & (observations['x'] < 0.5) & (observations['y'] > -.2) \
# #         & (observations['y'] < 0.6)), 'avg_temp'] = 50.
#
# # slant
# observations.loc[((observations['x'] > 0) &
#                   (observations['x'] < top) &
#                   (observations['y'] > 0) &
#                   (observations['y'] < right)),
#                  'avg_temp'] = observations.loc[((observations['x'] > 0) & (observations['x'] < top) & (observations['y'] > 0) & (observations['y'] < right)), 'x'] * 50 +\
#                     observations.loc[((observations['x'] > 0) & (observations['x'] < top) & (observations['y'] > 0) & (observations['y'] < right)), 'y'] / 50
# tt = np.reshape(observations.avg_temp, np.shape(xx))

### gauss

# observations['avg_temp'] = np.ravel(tt)


#
#
# # tt = np.reshape(observations.avg_temp, np.shape(xx))
#
# # make interpolator
# rbfi = Rbf(observations.x.values, observations.y.values, observations.z.values, observations.avg_temp.values,
#            function='gaussian')
#
# # define positions to interpolate at
# xi = np.linspace(bottom/2, top/2, 10)  # xmin * .8
# yi = np.linspace(left/2, right/2, 10)
# zi = np.linspace(floor/2, ceiling/2, 10)
# xxi, yyi = np.meshgrid(xi, yi, indexing='ij')
# xxi_flat = xxi.ravel()
# yyi_flat = yyi.ravel()
#
# # interpolate
# interp_temps = rbfi(xxi_flat, yyi_flat)
# tti = interp_temps.reshape((len(xi), len(yi)))
#
# print """
#         Interpolated temp stats
#         min {}
#         max {}
#         avg {}
#         """.format(interp_temps.min(), interp_temps.max(), interp_temps.mean())
#

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot', bgcolor=(1, 1, 1))



# grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
# min = density.min()
# max=density.max()
# s = mlab.pipeline.volume(grid, vmin=min, vmax=max,) #min + .5*(max-min))


src = mlab.pipeline.scalar_field(temp)
vol = mlab.pipeline.volume(src, vmin=temp.min(), vmax=temp.max()*.7)

mlab.axes()
mlab.show()
