# -*- coding: utf-8 -*-
"""
Flight trajectory plotting functions

Created on Fri Mar 20 12:27:04 2015
@author: Richard Decal, decal@uw.edu
https://staff.washington.edu/decal/
https://github.com/isomerase/
"""
from matplotlib import pyplot as plt
plt.close('all')

def StatesOverTimeGraph(t, states, dim = 2):
    plt.figure(1) #linewidth=2
    plt.plot(t, states[:, 0], color = 'b', linewidth=2)
    plt.plot(t, states[:, 1], color = 'g', linewidth=1, linestyle=':')
    plt.plot(t, states[:, 2], color = 'r', linewidth=2)
    plt.plot(t, states[:, 3], color = 'cyan', linewidth=1, linestyle=':')
    plt.xlabel('time')
    plt.ylabel('states')
    plt.title('mass-agent oscillating system')
    if dim == 1:
        plt.legend(('x', 'vx'))
    elif dim == 2:
        plt.legend(('x', 'vx', 'y', 'vy'))


def StateSpaceDraw(states, dim=2, animate=False, box=False):
    """Plot state-space over course of experiment.
    
    Credit to Paul Gribble (email: paul [at] gribblelab [dot] org), code based
    on a function in his "Computational Modelling in Neuroscience" course:
    http://www.gribblelab.org/compneuro/index.html
    """
    plt.figure(2)
    if dim == 1:
        pb, = plt.plot(states[:, 0], states[:, 1], 'b-')
        plt.xlabel('x')
        plt.ylabel('vx')
#        p, = plt.plot(states1[0:10, 0], states1[0:10, 1], 'b-')
#        pp, = plt.plot(states1[10, 0], states1[10, 1], 'b.', markersize=10)
        tt = plt.title("State-space graph for 1 dimensions")
    elif dim == 2:
        pb, = plt.plot(states[:, 0], states[:, 2], 'black', linewidth=2)  # select cols 0,2
                                                            # for x,y pos
        plt.xlabel('x position')
        plt.ylabel('y position')
#        p, = plt.plot(states1[0:10, 0], states1[0:10, 2], 'b-')
#        pp, = plt.plot(states1[10, 0], states1[10, 2], 'b.', markersize=10)
        tt = plt.title("State-space graph for 2 dimensions")
        if box is True:
#                plt.ylim(-0,5, 0.5)
#                plt.xlim(-0,5, 0.5)
                plt.grid()
#                gca().add_patch(pylab.Rectangle((-0.5, -0.5), 1, 1, linewidth=3, fill=False))
        plt.show()

    # animate
    if animate is True:
        step = 2
        for i in xrange(1, np.shape(states1)[0]-10, step):
            p.set_xdata(states1[10+i:20+i, 0])
            p.set_ydata(states1[10+i:20+i, 1])
            pp.set_xdata(states1[10+i, 0])
            pp.set_ydata(states1[10+i, 1])
            tt.set_text("State-space trajectory animation - step # %d" % (i))
            plt.draw()
