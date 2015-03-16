# -*- coding: utf-8 -*-
"""
stolen from https://github.com/stefanv/vibrations

Created on Mon Mar 16 13:10:09 2015
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


# Workaround to matplotlib bug in FunctionAnimator that won't allow us
# to specify ``itertools.count(step=delta)`` as the frames argument.
# Also see ``set_time`` below.
delta = 0.02


class FreeUndamped(object):
    def __init__(self, k, m, u0, u0_prime):
        self.w0 = np.sqrt(k / m)
        self.B = u0
        self.A = u0_prime / self.w0

    def __call__(self, t):
        A, B, w0 = self.A, self.B, self.w0
        return A * np.sin(w0 * t) + B * np.cos(w0 * t)


class Spring(object):
    N = 100
    _hist_length = 100

    _spring_coords = np.zeros(N)
    _spring_coords[30:70] = 0.05 * (-1) ** np.arange(40)

    def __init__(self, axis, axis_history, k, m, gamma, F, u0, u0_prime):
        if gamma == 0 or F == 0:
            self._method = FreeUndamped(k, m, u0, u0_prime)
        else:
            raise NotImplementedError()

        self._t = 0
        self._anchor = axis.vlines([0], -0.1, 0.1, linewidth=5, color='black')
        self._pot = Circle((self.u, 0), 0.05, color='black')
        self._spring, = axis.plot(*self._spring_xy(), color='black')

        axis.vlines([1], -0.1, -0.2)
        axis.text(1, -0.25, '$u = 0$', horizontalalignment='center')

        self._ax = axis
        axis.add_patch(self._pot)
        axis.set_xlim([0, 2])
        axis.set_ylim([-0.3, 0.2])
        axis.set_axis_off()
        axis.set_aspect('equal')

        self._history = [self.u - 1] * self._hist_length
        self._history_plot, = axis_history.plot(np.arange(self._hist_length) *
                                                delta, self._history)
        axis_history.annotate('Now',
                              (self._hist_length * delta, 1.5),
                              (self._hist_length * delta, 1.8),
                              arrowprops=dict(arrowstyle='->'),
                              horizontalalignment='center')
        axis_history.set_ylim(-2, 1.5)
        axis_history.set_xticks([])
        axis_history.set_xlabel(r'$\mathrm{Time}$')
        axis_history.set_ylabel(r'$\mathrm{Position,\, u}$')

    def _spring_xy(self):
        return np.linspace(0, self.u, self.N), self._spring_coords

    def set_time(self, t):
        self._t = t * delta
        self.update()

    @property
    def u(self):
        return 1 + self._method(self._t)

    def update(self):
        self._pot.center = (self.u, 0)

        x, y = self._spring_xy()
        self._spring.set_xdata(x)

        self._history.append(self.u - 1)
        self._history = self._history[-self._hist_length:]

        self._history_plot.set_ydata(self._history)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Illustration simple mechanical vibrations')
    parser.add_argument('-o', '--output',
                        help='Filename for output video.  Requires ffmpeg.')
    parser.add_argument('-m', '--mass', type=float, default=0.5)
    parser.add_argument('-y', '--damping', type=float, default=0)
    parser.add_argument('-k', '--spring', type=float, default=1)
    parser.add_argument('-F', '--force', type=float, default=0)
    parser.add_argument('--u0', type=float, default=0)
    parser.add_argument('--u0_prime', type=float, default=0.5)
    args = parser.parse_args()

    m, gamma, k, F, u0, u0_prime = (getattr(args, name) for name in
                                    ('mass', 'damping', 'spring', 'force',
                                     'u0', 'u0_prime'))

    f, (ax0, ax1) = plt.subplots(2, 1)
    s = Spring(axis=ax0, axis_history=ax1,
               k=k, m=m, gamma=gamma, F=F, u0=u0, u0_prime=u0_prime)

    anim = animation.FuncAnimation(f, s.set_time, interval=delta * 1000,
                                  save_count=400)

    if args.output:
        print "Saving video output to %s (this may take a while)." % args.output
        anim.save(args.output, fps=25)

    plt.show()
