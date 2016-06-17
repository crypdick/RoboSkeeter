import numpy as np
from scipy.interpolate import Rbf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# generate observations
left = -2.
right = 3.
# floor = 0.
# ceiling = 1.
bottom = -1.
top = 2.

resolution = 0.7

unique_xs = np.arange(bottom, top, resolution)
unique_ys = np.arange(left, right, resolution)
coords = (unique_xs, unique_ys)

coords_mesh = np.meshgrid(*coords, indexing="ij")

fn_value = np.power(coords_mesh[0], 2) + coords_mesh[1] * coords_mesh[2]  # F(x, y, z)

coords_array = np.vstack([x.flatten() for x in coords_mesh]).T  # Columns are x, y, z

x, y, z = [x.flatten() for x in coords_mesh]
d = fn_value.flatten()


# # xx, yy = np.meshgrid(x, y, indexing='ij')
# df_dict = dict()
# df_dict['x'] = np.concatenate((xx.ravel(), xx.ravel()))
# df_dict['y'] = np.concatenate((yy.ravel(), yy.ravel()))


observations = pd.DataFrame(data=df_dict)


# make some places hot

temp = 19.
observations['avg_temp'] = np.array([temp] * len(observations))  # make it the same temperature everywhere
# observations.loc[((observations['x'] > -.5) & (observations['x'] < 0.5) & (observations['y'] > -.2) \
#         & (observations['y'] < 0.6)), 'avg_temp'] = 50.

# slant
# observations.loc[((observations['x'] > 0) &
#                   (observations['x'] < top) &
#                   (observations['y'] > 0) &
#                   (observations['y'] < right)),
#                  'avg_temp'] = observations.loc[((observations['x'] > 0) & (observations['x'] < top) & (observations['y'] > 0) & (observations['y'] < right)), 'x'] * 50 +\
#                     observations.loc[((observations['x'] > 0) & (observations['x'] < top) & (observations['y'] > 0) & (observations['y'] < right)), 'y'] / 50
# tt = np.reshape(observations.avg_temp, np.shape(xx))

# ### gauss
# tt = plt.mlab.bivariate_normal(xx, yy)
# observations['avg_temp'] = np.ravel(tt)




# tt = np.reshape(observations.avg_temp, np.shape(xx))

# make interpolator
rbfi = Rbf(observations.x.values, observations.y.values, observations.avg_temp.values),
           # function='cubic')

# # define positions to interpolate at
# xi = np.linspace(bottom/2, top/2, 10)  # xmin * .8
# yi = np.linspace(left/2, right/2, 10)
# xxi, yyi = np.meshgrid(xi, yi, indexing='ij')
# xxi_flat = xxi.ravel()
# yyi_flat = yyi.ravel()

# interpolate
# interp_temps = rbfi(xxi_flat, yyi_flat)
# tti = interp_temps.reshape((len(xi), len(yi)))
interp_temps = rbfi(observations.x.values, observations.y.values)
tti = interp_temps.reshape(np.shape(xx))

print """
        Interpolated temp stats
        min {}
        max {}
        avg {}
        """.format(interp_temps.min(), interp_temps.max(), interp_temps.mean())

fig = plt.figure()
ax = fig.gca(projection='3d')
# plt.scatter(xxi_flat, yyi_flat, c=interp_temps, cmap='inferno', lw=0)
# ax.plot_wireframe(xx, yy, tt)
ax.plot_wireframe(xx, yy, tti, color='green')

# plt.scatter(observations.x.values, observations.y.values, c=observations.avg_temp.values, marker='x')
plt.show()


# # save to df
# df_dict = dict()
# df_dict['x'] = xxi_flat
# df_dict['y'] = yyi_flat
# df_dict['avg_temp'] = interp_temps
# df = pd.DataFrame(df_dict)
