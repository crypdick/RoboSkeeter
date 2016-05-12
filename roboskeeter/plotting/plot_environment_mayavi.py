import mayavi
from mayavi import mlab

mlab.options.backend = 'envisage'

def plot_plume_recordings_volume(bounds, grid_x, grid_y, grid_z, grid_temps):

    ## visualize
    fig = mlab.figure('Plume')


    src = mlab.pipeline.scalar_field(grid_temps)
    vol = mlab.pipeline.volume(src, vmin=grid_temps.min(), vmax=grid_temps.max()*.7)
    mayavi.tools.pipeline.set_extent(mlab.pipeline.volume)

    mlab.axes(bounds=bounds)
    mlab.show()
