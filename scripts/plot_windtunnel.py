import matplotlib

reload(matplotlib)  # seaborn bugs
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
# register Axes3D class with matplotlib by importing Axes3D
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import m3d.art3d, Axes3D
import mpl_toolkits

reload(mpl_toolkits)
import mpl_toolkits.mplot3d as m3d


def mk_3Dfig():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    return ax


def set_windtunnel_bounds(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)

    ax.set_ylabel("Crosswind/$y$ (meters)", fontsize=14)  # remember! x,y switched in plot() above!
    ax.set_xlabel("Upwind/$x$ (meters)", fontsize=14)
    ax.set_zlabel("Elevation/$z$ (meters)", fontsize=14)

    ax.grid(False)  # no grid
    ax.patch.set_visible(False)  # white background
    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        a.pane.set_visible(False)


def draw_windtunnel(ax):
    x_min = 0
    x_max = 1
    z_min = 0
    z_max = 0.254
    y_min = -0.127
    y_max = 0.127
    draw_rectangular_prism(ax, x_min, x_max, y_min, y_max, z_min, z_max)


def draw_rectangular_prism(ax, x_min, x_max, y_min, y_max, z_min, z_max):
    alpha = 1
    back = Rectangle((y_min, z_min), y_max - y_min, z_max, alpha=alpha, fill=None, linestyle='dotted')
    ax.add_patch(back)
    m3d.art3d.pathpatch_2d_to_3d(back, z=x_min, zdir="x")

    front = Rectangle((y_min, z_min), y_max - y_min, z_max, alpha=alpha, fill=None, linestyle='dotted')
    ax.add_patch(front)
    m3d.art3d.pathpatch_2d_to_3d(front, z=x_max, zdir="x")

    floor = Rectangle((x_min, y_min), x_max, z_max, alpha=alpha, fill=None, linestyle='dotted')
    ax.add_patch(floor)
    m3d.art3d.pathpatch_2d_to_3d(floor, z=z_min, zdir="z")

    ceiling = Rectangle((x_min, y_min), x_max, z_max, alpha=alpha, fill=None, linestyle='dotted')
    ax.add_patch(ceiling)
    m3d.art3d.pathpatch_2d_to_3d(ceiling, z=z_max, zdir="z")


def draw_trajectory(ax, trajectory, **kwargs):
    ax.plot(trajectory.position_x, trajectory.position_y, trajectory.position_z)
    plt.show()


def draw_plume(ax, plume):
    # FIXME: switch to http://matplotlib.org/api/collections_api.html#matplotlib.collections.EllipseCollection

    # xy is actually the yz plane
    # v index: [u'x_position', u'z_position', u'small_radius', u'y_position']
    ells = [Ellipse(xy=(v[3], v[1]), width=v[2], height=v[2] * 3, angle=0)
            for i, v in plume.data.iterrows()]

    for i, v in plume.data['x_position'].iteritems():
        ell = ells[i]
        ax.add_patch(ell)
        m3d.art3d.patch_2d_to_3d(ell, z=v, zdir="x")

    plt.show()


if __name__ is '__main__':
    ax = mk_3Dfig()
    draw_windtunnel(ax)

    import experiment

    simulation, trajectory_s, windtunnel, plume, agent = experiment.run_simulation(None, None)
    draw_trajectory(ax, trajectory_s.data)
