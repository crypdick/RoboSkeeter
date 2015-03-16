# -*- coding: utf-8 -*-
"""
TODO: automatically save this video

stolen from https://github.com/stefanv/vibrations

Created on Mon Mar 16 11:40:53 2015
@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""

# Stefan van der Walt <stefan@sun.ac.za>, 2013
# License: CC0

from __future__ import division

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle


# Workaround to matplotlib bug in Functionheadingnimator that won't allow us
# to specify ``itertools.count(step=dt)`` as the frames argument.
# headinglso see ``set_time`` below.
dt = 0.02


#m is the mass, y the damping from the dash pot, k the restoring force of the spring, F the driving force and w the frequency.


class FreeUndamped(object):
    def __init__(self, k, m, x0, x0_prime):
        self.ang_freq0 = np.sqrt(k / m) #angular frequency
        self.start_pos = x0
        self.heading = x0_prime / self.ang_freq0

    def __call__(self, t):
        heading, start_pos, ang_freq0 = self.heading, self.start_pos, self.ang_freq0
        return heading * np.sin(ang_freq0 * t) + start_pos * np.cos(ang_freq0 * t)


class Agent(object):
    N = 100
    _hist_length = 100

    _agent_coords = np.zeros(N)
    _agent_coords[30:70] = 0.05 * (-1) ** np.arange(40)

    def __init__(self, axis, axis_history, k, m, gamma, F, x0, x0_prime):
        if gamma == 0 or F == 0: #gamma is damping
            self._method = FreeUndamped(k, m, x0, x0_prime)
        else:
            raise NotImplementedError() #damping is not implemented??
        
        #drawing the coords of all the moving bits
        self._t = 0
#        self._anchor = axis.vlines([0], -0.1, 0.1, linewidth=5, color='black') #draw the spring
        self._pot = Circle((self.x, 0), 0.05, color='black') #draw the mass
#        self._agent, = axis.plot(*self._agent_xy_positions(), color='black')  #draw spring thingie
#
        axis.vlines([1], -0.1, -0.2)
        axis.text(1, -0.25, '$x = 0$', horizontalalignment='center')
#
#        self._ax = axis
        axis.add_patch(self._pot)
        axis.set_xlim([0, 2])
        axis.set_ylim([-0.3, 0.2])
        axis.set_axis_off()
        axis.set_aspect('equal')

        # start of second plot
        self._history = [self.x - 1] * self._hist_length
        self._history_plot, = axis_history.plot(np.arange(self._hist_length) *
                                                dt, self._history)
#        axis_history.annotate('Now', #arrow schwag
#        
##                              (self._hist_length * dt, 1.5),
##                              (self._hist_length * dt, 1.8),
#                              arrowprops=dict(arrowstyle='->'),
#                              horizontalalignment='center')
        axis_history.set_ylim(-2, 1.5)
        axis_history.set_xticks([])
        axis_history.set_xlabel(r'$\mathrm{Time}$')
        axis_history.set_ylabel(r'$\mathrm{Position,\, x}$')

    def _agent_xy_positions(self):
        return np.linspace(0, self.x, self.N), self._agent_coords

    def set_time(self, t):  #TODO wtf
        self._t = t * dt
        self.update()

    @property  #  TODO wtf?
    def x(self):  #  TODO wtf?
        return 1 + self._method(self._t)

    def update(self):
        self._pot.center = (self.x, 0)
        
        #these are the handles for plotting
        x_positions, y_positions = self._agent_xy_positions() #y_positions unused
#        self._agent.set_xdata(x_positions)

        self._history.append(self.x - 1)
        self._history = self._history[-self._hist_length:]

        self._history_plot.set_ydata(self._history)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Illustration simple mechanical vibrations')
    parser.add_argument('-o', '--output',
                        help='Filename for output video.  Requires ffmpeg.')
    parser.add_argument('-m', '--mass', type=float, default=0.5)
    parser.add_argument('-gamma', '--damping', type=float, default=0)
    parser.add_argument('-k', '--spring', type=float, default=1)
    parser.add_argument('-F', '--force', type=float, default=0)
    parser.add_argument('--x0', type=float, default=0)
    parser.add_argument('--x0_prime', type=float, default=0.5)
    args = parser.parse_args()

    m, gamma, k, F, x0, x0_prime = (getattr(args, name) for name in
                                    ('mass', 'damping', 'spring', 'force',
                                     'x0', 'x0_prime'))

    fig, (ax0, ax1) = plt.subplots(2, 1) #ax0 is handle for pot, ax1 is handle for the lower plot
    inst_agent = Agent(axis=ax0, axis_history=ax1,
               k=k, m=m, gamma=gamma, F=F, x0=x0, x0_prime=x0_prime)

    anim = animation.FuncAnimation(fig, inst_agent.set_time, interval=dt * 10, #intervial draws new frame every "interval" ms and redraws image
                                  )#save_count=400)

    if args.output:
        print "Saving video output to %s (this may take a while)." % args.output
        anim.save(args.output, fps=40)

    plt.show()
