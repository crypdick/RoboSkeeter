import matplotlib
matplotlib.use('Qt4Agg')
matplotlib.interactive(True)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse, Circle
# register Axes3D class with matplotlib by importing Axes3D
from mpl_toolkits.mplot3d import art3d
import numpy as np

def set_windtunnel_bounds(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.invert_xaxis()  # fix for y convention

    ax.set_ylabel("Crosswind/$y$ (meters)", fontsize=14)  # remember! x,y switched in plot() above!
    ax.set_xlabel("Upwind/$x$ (meters)", fontsize=14)
    ax.set_zlabel("Elevation/$z$ (meters)", fontsize=14)

    ax.grid(False)  # no grid
    ax.patch.set_visible(False)  # white background
    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        a.pane.set_visible(False)


def draw_windtunnel_border(ax):
    x_min = 0
    x_max = 1
    z_min = 0
    z_max = 0.254
    y_min = -0.127
    y_max = 0.127
    draw_rectangular_prism(ax, x_min, x_max, y_min, y_max, z_min, z_max)
    plt.draw()


def draw_heaters(ax, windtunnel):
    draw_heater(ax, windtunnel.heater_l)
    draw_heater(ax, windtunnel.heater_r)


def draw_heater(ax, heater):
    x_center, y_center = heater.x_position, heater.y_position
    elevation, height = heater.zmin, heater.zmax - heater.zmin
    radius = heater.diam / 2
    resolution = 101
    color = heater.color

    plot_3D_cylinder(ax, radius, height, elevation=elevation, resolution=resolution, color=color, x_center=x_center,
                     y_center=y_center)


def draw_rectangular_prism(ax, x_min, x_max, y_min, y_max, z_min, z_max):
    alpha = 1
    back = Rectangle((y_min, z_min), y_max - y_min, z_max, alpha=alpha, fill=None, linestyle='dotted')
    ax.add_patch(back)
    art3d.pathpatch_2d_to_3d(back, z=x_min, zdir="x")

    front = Rectangle((y_min, z_min), y_max - y_min, z_max, alpha=alpha, fill=None, linestyle='dotted')
    ax.add_patch(front)
    art3d.pathpatch_2d_to_3d(front, z=x_max, zdir="x")

    floor = Rectangle((x_min, y_min), x_max, z_max, alpha=alpha, fill=None, linestyle='dotted')
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=z_min, zdir="z")

    ceiling = Rectangle((x_min, y_min), x_max, z_max, alpha=alpha, fill=None, linestyle='dotted')
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=z_max, zdir="z")


def draw_bool_plume(plume, ax=None):
    """Note: tried  to add ells to Pathcollection and then plot that, but kept having a ValueError
    Also tried using an EclipseCollection but the patchcollection_2d_to_3d function does not work on it

    # xy is actually the yz plane
    # val headers: [u'x_position', u'z_position', u'small_radius', u'y_position']
    """
    if plume.condition in 'controlControlCONTROL':
        print "No plume in ", plume.condition
        return

    ells = [Ellipse(xy=(val[3], val[1]),
                    width=2 * val[2],
                    height=2 * val[2] * 3,
                    angle=0,
                    # linestyle='dotted',
                facecolor='none',
                    alpha=0.05,
                    edgecolor='r')
            for i, val in plume.data.iterrows()]

    for i, val in plume.data['x_position'].iteritems():
        ell = ells[i]
        ax.add_patch(ell)
        art3d.patch_2d_to_3d(ell, z=val, zdir="x")

    art3d.patch_collection_2d_to_3d(ells)

    plt.show()


def plot_3D_cylinder(ax, radius, height, elevation=0, resolution=200, color='r', x_center=0, y_center=0):
    x = np.linspace(x_center - radius, x_center + radius, resolution)
    z = np.linspace(elevation, elevation + height, resolution)
    X, Z = np.meshgrid(x, z)

    Y = np.sqrt(radius ** 2 - (X - x_center) ** 2) + y_center  # Pythagorean theorem

    ax.plot_surface(X, Y, Z, linewidth=0, color=color)
    ax.plot_surface(X, (2 * y_center - Y), Z, linewidth=0, color=color)

    floor = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(floor)
    art3d.pathpatch_2d_to_3d(floor, z=elevation, zdir="z")

    ceiling = Circle((x_center, y_center), radius, color=color)
    ax.add_patch(ceiling)
    art3d.pathpatch_2d_to_3d(ceiling, z=elevation + height, zdir="z")


def plot_windtunnel(windtunnel, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    # ax.axis('off')
    set_windtunnel_bounds(ax)
    draw_windtunnel_border(ax)
    ax.set_title(title, fontsize=20)  # FIXME parent functions aren't passing titles yet
    draw_heaters(ax, windtunnel)

    return fig, ax


def draw_trajectory(ax, trajectory, **kwargs):
    """TODO: red inside windtunnel"""
    highlight = kwargs.get("highlight_inside_plume")
    ax.plot(trajectory.position_x, trajectory.position_y, trajectory.position_z)
    if highlight:
        inside_pts = trajectory.loc[trajectory['inPlume'] == True]
        ax.scatter(inside_pts.position_x, inside_pts.position_y, inside_pts.position_z, c='r')


def plot_plume_recordings_scatter(plume_data, ax, temp_thresh = 0.):
    plume_data = plume_data[plume_data.avg_temp > temp_thresh]
    ax.scatter(plume_data.x, plume_data.y, plume_data.z, c=plume_data.avg_temp, cmap='Oranges', lw=0)


def plot_plume_gradient(plume, ax, thresh):  # TODO: plot inside windtunnel as in draw_bool_plume
    """
    Plot a quiverplot of the gradient
    Parameters
    ----------
    plume
        (gradient plume object)
    thresh
        (float)
        filters out plotting of gradient arrows smaller than this threshold
    """

    filtered = plume.data[plume.data.gradient_norm > thresh]

    ax.quiver(filtered.x, filtered.y, filtered.z, filtered.gradient_x, filtered.gradient_y, filtered.gradient_z, length=0.01)
    # ax.set_xlim3d(0, 1)
    # ax.set_ylim3d(-0.127, 0.127)
    # ax.set_zlim3d(0, 0.254)

    plt.title("Temperature gradient of interpolated time-averaged thermocouple recordings")
    plt.xlabel("Upwind/downwind")
    plt.ylabel("Crosswind")
    plt.clabel("Elevation")

    plt.show()


if __name__ is '__main__':
    ax = plot_windtunnel()