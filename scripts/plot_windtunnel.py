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


def draw_plume(plume, windtunnel, ax=None):
    # FIXME: switch to http://matplotlib.org/api/collections_api.html#matplotlib.collections.EllipseCollection

    # xy is actually the yz plane
    # v index: [u'x_position', u'z_position', u'small_radius', u'y_position']
    if plume.condition in 'controlControlCONTROL':
        print "No plume in ", plume.condition
        return

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        set_windtunnel_bounds(ax)
        draw_windtunnel_border(ax)
        draw_heaters(ax, windtunnel)

    ells = [
        Ellipse(xy=(v[3], v[1]),
                width=2 * v[2],
                height=2 * v[2] * 3,
                angle=0,
                # linestyle='dotted',
                facecolor='none',
                alpha=0.05,
                edgecolor='r')
        for i, v in plume.data.iterrows()]

    for i, v in plume.data['x_position'].iteritems():
        ell = ells[i]
        ax.add_patch(ell)
        art3d.patch_2d_to_3d(ell, z=v, zdir="x")

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





if __name__ is '__main__':
    ax = plot_windtunnel()

    # import experiment
    #
    # simulation, trajectory_s, windtunnel, plume, agent = experiment.run_simulation(None, None)
    # draw_trajectory(ax, trajectory_s.data)
