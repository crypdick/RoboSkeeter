# import numpy as np
# from scipy import stats
# from mayavi import mlab
# import multiprocessing
# import matplotlib.pyplot as plt
#
# x, y = np.mgrid[-1.0:1.0:30j, -1.0:1.0:30j]
# # Need an (N, 2) array of (x, y) pairs.
# xy = np.column_stack([x.flat, y.flat])
#
# mu = np.array([0.0, 0.0])
#
# sigma = np.array([.025, .025])
# covariance = np.diag(sigma**2)
#
# z = stats.multivariate_normal.pdf(xy, mean=mu, cov=covariance)
#
# # Reshape back to a (30, 30) grid.
# z = z.reshape(x.shape)
#


import numpy as np
from scipy import stats
from mayavi import mlab
import multiprocessing
from matplotlib.cm import get_cmap

values = np.linspace(0., 1., 256)
lut_dict = {}
lut_dict['plasma'] = get_cmap('plasma')(values.copy())

def calc_kde(data):
    return kde(data.T)

mu, sigma = 0, 0.01
x = 10*np.random.normal(mu, sigma, 1000)
y = 10*np.random.normal(mu, sigma, 1000)
z = 10*np.random.normal(mu, sigma, 1000)

xyz = np.vstack([x,y,z])
kde = stats.gaussian_kde(xyz)

# Evaluate kde on a grid
xmin, ymin, zmin = x.min(), y.min(), z.min()
xmax, ymax, zmax = x.max(), y.max(), z.max()
xi, yi, zi = np.mgrid[xmin:xmax:30j, ymin:ymax:30j, zmin:zmax:30j]
coords = np.vstack([item.ravel() for item in [xi, yi, zi]])

# Multiprocessing
cores = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=cores)
results = pool.map(calc_kde, np.array_split(coords.T, 2))  # TODO: what is this 2?
density = np.concatenate(results).reshape(xi.shape)

# Plot scatter with mayavi
figure = mlab.figure('DensityPlot', bgcolor=(1, 1, 1))



# grid = mlab.pipeline.scalar_field(xi, yi, zi, density)
# min = density.min()
# max=density.max()
# s = mlab.pipeline.volume(grid, vmin=min, vmax=max,) #min + .5*(max-min))

x, y, z = np.ogrid[-10:10:20j, -10:10:20j, -10:10:20j]
s = np.sin(x*y*z)/(x*y*z)


src = mlab.pipeline.scalar_field(s)
vol = mlab.pipeline.volume(src)


lut = vol.module_manager.scalar_lut_manager.lut.table.to_array()

# The lut is a 256x4 array, with the columns representing RGBA
# (red, green, blue, alpha) coded with integers going from 0 to 255.

# We modify the alpha channel to add a transparency gradient
lut[:, -1] = np.linspace(0, 255, 256)
# and finally we put this LUT back in the surface object. We could have
# added any 255*4 array rather than modifying an existing LUT.
vol.module_manager.scalar_lut_manager.lut.table = lut






# lut = lut_dict['plasma']
# lut[:, -1] = np.linspace(0, 255, 256)
# # lut[:, 0] = np.linspace(0, 255, 256)
#
# vol.module_manager.scalar_lut_manager.lut.table = lut
#
#
#


# # Changing the ctf:
# from tvtk.util.ctf import ColorTransferFunction
# ctf = ColorTransferFunction()
# ctf.add_rgb_point(value, r, g, b)
# ctf.add_hsv_point(value, h, s, v)
# # ...
# vol._volume_property.set_color(ctf)
# vol._ctf = ctf
# vol.update_ctf = True
#
# # Changing the otf:
# from enthought.tvtk.util.ctf import PiecewiseFunction
# otf = PiecewiseFunction()
# otf.add_point(value, opacity)
# self._target._otf = otf
# self._target._volume_property.set_scalar_opacity(otf)
#

# grid.
# surf.module_manager.scalar_lut_manager.lut.tabl

mlab.axes()
mlab.show()
