import matplotlib.pyplot as plt
from matplotlib import animation


# Line3Dcollection

# Params
# sim_or_exp = 'simulation'  # 'experiment', 'simulation'
# experiment = eval(sim_or_exp)
# highlight_inside_plume = False
# show_plume = False
# trajectory_i = None


# initialization function: plot the background of each frame
# def init():
#     for line, pt in zip(lines, pts):
#         line.set_data([], [])
#         line.set_3d_properties([])
#
#         pt.set_data([], [])
#         pt.set_3d_properties([])
#     return lines + pts


# animation function.  This will be called sequentially with the frame number
def animate(i, fig, ax, x_t, lines, pts):
    # we'll step two time-steps per frame.  This leads to nice results.
    i = (2 * i) % x_t.shape[1]

    for line, pt, xi in zip(lines, pts, x_t):
        x, y, z = xi[:i].T
        line.set_data(x, y)
        # for line collection use
        # lc.set_segments
        line.set_3d_properties(z)

        pt.set_data(x[-1:], y[-1:])
        pt.set_3d_properties(z[-1:])

    # ax.view_init(30, 0.3 * i)
    ax.view_init(90, 0 * i)
    fig.canvas.draw()

    return lines + pts


# def local_init(x=x, t=t, data=data):
#     return init_func(x,t,data)



def wrapper(fig, ax, x_t):
    # # choose a different color for each trajectory
    # colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

    # set up lines and points
    lines = sum([ax.plot([], [], [], '-', c='gray')
                 ], [])
    pts = sum([ax.plot([], [], [], '*', c='black')
               ], [])

    # set point-of-view: specified by (altitude degrees, azimuth degrees)
    # ax.view_init(90, 0)

    animation_args = (fig, ax, x_t, lines, pts)

    # instantiate the animator.
    anim = animation.FuncAnimation(fig, animate, fargs=animation_args, init_func=init,
                                   # FIXME http://matplotlib.org/examples/animation/simple_3danim.html
                                   interval=1, blit=False, repeat_delay=8000,
                                   frames=x_t.shape[1])  # original:  frames=500, interval=30, blit=True)

    # added writer b/c original func didn't work
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=100, metadata=dict(artist='Richard'), bitrate=1000)
    # anim.save('{}-{}.mp4'.format(sim_or_exp, trajectory_i), writer=writer)

    plt.show()
