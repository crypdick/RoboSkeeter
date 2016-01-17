import matplotlib.pyplot as plt
from matplotlib import animation


# Line3Dcollection

# Params
# sim_or_exp = 'simulation'  # 'experiment', 'simulation'
# experiment = eval(sim_or_exp)
# highlight_inside_plume = False
# show_plume = False
# trajectory_i = None


# def local_init(x=x, t=t, data=data):
#     return init_func(x,t,data)





class Windtunnel_animation(object):
    def __init__(self, fig, ax, x_t,
                 metadata={"trajectory type": "simulation",
                           "trajectory index": 0},
                 save=False):
        print "hello1"
        self.fig = fig
        self.ax = ax
        self.x_t = x_t
        self.save = save
        self.metadata = metadata

        # # choose a different color for each trajectory
        # colors = plt.cm.jet(np.linspace(0, 1, N_trajectories))

        # set up lines and points
        self.lines = sum([ax.plot([], [], [], '-', c='gray')
                     ], [])
        self.pts = sum([ax.plot([], [], [], '*', c='black')
                   ], [])


    def start_animation(self):
        print "hello start"
        animation_args = (self.fig, self.ax, self.x_t, self.lines, self.pts)

        # instantiate the animator.
        print self.x_t

        anim = animation.FuncAnimation(self.fig, self.ani_update, init_func=self.ani_init,
                                       # FIXME http://matplotlib.org/examples/animation/simple_3danim.html
                                       interval=1, blit=False, repeat_delay=8000,
                                       frames=self.x_t.shape[1])  # original:  frames=500, interval=30, blit=True)

        # added writer b/c original func didn't work
        Writer_class = animation.writers['mencoder']
        writer = Writer_class(fps=100, metadata=dict(artist='Richard'), bitrate=10) #birate 1000 is good

        if self.save:
            anim.save('{}-{}.mp4'.format(self.metadata["trajectory type"], self.metadata["trajectory index"]), writer=writer)

        plt.show()


    def ani_init(self):
        print "initiating"
        for line, pt in zip(self.lines, self.pts):
            # init lines
            line.set_data([], [])
            line.set_3d_properties([])

            # init points
            pt.set_data([], [])
            pt.set_3d_properties([])

        return self.lines + self.pts


    # animation function.  This will be called sequentially with the frame number
    def ani_update(self, i):
        # we'll step two time-steps per frame.  This leads to nice results.  # TODO: test with 1
        i = (2 * i) % self.x_t.shape[1]
        print "i", i

        for line, pt, xi in zip(self.lines, self.pts, self.x_t):
            x, y, z = xi[:i].T
            print xi.shape
            # update lines
            line.set_data(x, y)
            # for line collection use
            # lc.set_segments
            line.set_3d_properties(z)

            # update points
            pt.set_data(x[-1:], y[-1:])
            pt.set_3d_properties(z[-1:])

        # ax.view_init(30, 0.3 * i)
        self.ax.view_init(90, 0 * i)
        self.fig.canvas.draw()

        return self.lines + self.pts

