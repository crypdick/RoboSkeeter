from matplotlib import animation

from analysis.plot_environment import plot_windtunnel as pwt

# Params
sim_or_exp = 'simulation'  # 'experiment', 'simulation'
experiment = eval(sim_or_exp)
highlight_inside_plume = False
show_plume = False
trajectory_i = None


if trajectory_i is None:
    trajectory_i = experiment.trajectories.get_trajectory_numbers().min()

# get df
df = experiment.trajectories.get_trajectory_i_df(trajectory_i)

p = df[['position_x', 'position_y', 'position_z']].values
x_t = p.reshape((1, len(p), 3))  # make into correct shape for Jake vdp's code


fig, ax = pwt.plot_windtunnel(experiment.windtunnel)
ax.axis('off')

if show_plume:
    pwt.draw_plume(experiment, ax=ax)

# # choose a different color for each trajectory
# colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

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
# ax.view_init(90, 0)

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
    print i

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        print xi.shape
        line.set_data(x, y)
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    # ax.view_init(30, 0.3 * i)
    ax.view_init(90, 0 * i)
    fig.canvas.draw()
    return lines + pts


# instantiate the animator.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               interval=1, blit=False, repeat_delay=8000,
                               frames=len(p))  # original:  frames=500, interval=30, blit=True)

# added writer b/c original func didn't work
Writer = animation.writers['mencoder']
writer = Writer(fps=100, metadata=dict(artist='Richard'), bitrate=1000)
anim.save('{}-{}.mp4'.format(sim_or_exp, trajectory_i), writer=writer)

# plt.show()
