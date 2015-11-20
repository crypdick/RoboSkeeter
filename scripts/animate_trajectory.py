from matplotlib import animation
from matplotlib import pyplot as plt

import scripts.plot_windtunnel as pwt

trajectory_i = None
highlight_inside_plume = False
show_plume = False

if trajectory_i is None:
    trajectory_i = trajectory_s.get_trajectory_numbers().min()

fig, ax = pwt.plot_windtunnel()
# ax.axis('off')
if show_plume:
    pwt.draw_plume(experiment.plume, experiment.windtunnel.heater, ax=ax)

p = trajectory_s.data[['position_x', 'position_y', 'position_z']].values
x_t = p.reshape((trajectory_s.data.trajectory_num.unique().__len__(), len(p), 3))

# set up lines and points
lines = sum([ax.plot([], [], [], '-', c='gray')
             ], [])
pts = sum([ax.plot([], [], [], '*', c='black')
           ], [])


# # prepare the axes limits
# ax.set_xlim((0, 1))
# ax.set_ylim((-.127, .127))
# ax.set_zlim((0, .254))

# set point-of-view: specified by (altitude degrees, azimuth degrees)
# ax.view_init(30, 0)

# initialization function: plot the background of each frame
def init():
    for line, pt in zip(lines, pts):
        line.set_data([], [])
        line.set_3d_properties([])

        pt.set_data([], [])
        pt.set_3d_properties([])
    return lines + pts


# animation function.  This will be called sequentially with the frame number
def animate(i):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    # ax.view_init(30, 0.3 * i)
    fig.canvas.draw()
    return lines + pts


# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               interval=1, blit=True, repeat_delay=8000)

# Save as mp4. This requires mplayer or ffmpeg to be installed
# anim.save('lorentz_attractor.mp4', fps=15, extra_args=['-vcodec', 'libx264'])

plt.show()
