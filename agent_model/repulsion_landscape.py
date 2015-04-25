# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

Given an agent crosswind position, return force.

@author: Richard Decal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as deriv


def landscape(wallF_max=8e-8, decay_const = 90, mu=0., stdev=0.04, centerF_max=5e-8):
    """exponential decay params: wallF_max, decay_const
    gaussian params: mu, stdev, centerF_max
    """
    stdev2, centerF_max2 = 0.004, 3e-8
    fx = lambda pos : (wallF_max * np.exp(-1 * decay_const * (pos+0.127))) + ( centerF_max * np.exp(-1*(pos-mu)**2 / (2*stdev**2)) ) + ( (centerF_max2) * np.exp(-1*(pos-mu)**2 / (2*(stdev2)**2)) ) + (wallF_max * np.exp(1 * decay_const * (pos-0.127)))
    
    return fx


def plot_landscape(repulsion_fxn, wallF_max, decay_const, _, stdev, centerF_max):
    fig, axarr = plt.subplots(2, sharex=True)
    
    ycoords = np.linspace(-0.127, 0.127, 200)
    scariness = repulsion_fxn(ycoords)
    
    axarr[0].plot(ycoords, scariness)
    axarr[0].set_title("Crosswind repulsion landscape", fontsize = 14)
    axarr[0].set_xlim(-0.127, 0.127)
    axarr[0].set_xlabel("crosswind position")
    axarr[0].set_ylabel("scariness")
    axarr[0].xaxis.grid(True)
#    axarr[0].savefig("repulsion_landscape.png")
    
    # plot derivative
    slopes = deriv(repulsion_fxn, ycoords, dx=0.00001)
    
    axarr[1].axhline(y=0, color="grey", lw=1, alpha=0.4)
#    axarr[1].plot(ycoords, slopes, label="derivative")
    axarr[1].plot(ycoords, -1* slopes, label="-derivative")
    axarr[1].set_title("Strength of Repulsive Force (-slope of scariness)", fontsize = 14)
    axarr[1].set_xlim(-0.127, 0.127)
    axarr[1].set_xlabel("crosswind position")
    axarr[1].set_ylabel("slope")
    axarr[1].xaxis.grid(True)
    axarr[1].legend()
    
    plt.tight_layout()
    
    plt.savefig("./figs/repulsion_landscape wallF_max{wallF_max} decay_const{decay_const} stdev{stdev} centerF_max{centerF_max}.png".format(wallF_max=wallF_max, decay_const=decay_const, stdev= stdev, centerF_max=centerF_max))
    plt.show()


def main(wallF, pos_y=None, plotting=False):
    repulsion_fxn = landscape(*wallF)
    
    if pos_y != None:
        repF = deriv(repulsion_fxn, pos_y, dx=0.00001)
    
    if plotting is True:
        plot_landscape(repulsion_fxn, *wallF)
        
    return -1*repF  # this is the force on the agent in the y component


if __name__ == '__main__':
    # wallF params
    wallF_max=8e-8
    decay_const = 120
    mu=0.
    stdev=0.01
    centerF_max=8e-8
    
    wallF = (wallF_max, decay_const, mu, stdev, centerF_max)
    force_y = main(wallF, 0, plotting=True)