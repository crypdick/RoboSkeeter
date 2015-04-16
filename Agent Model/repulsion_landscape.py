# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

Given an agent crosswind position, return force.

@author: Richard Decal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as deriv

def landscape(pos, wallF_max=8e-8, decay_const = 90, mu=0., stdev=0.04, centerF_max=5e-8):
    # exp params
    wallF_max = wallF_max
    decay_const = decay_const
    # gaussian params
    mu, stdev, centerF_max = mu, stdev, centerF_max
    
    fx = lambda pos : (wallF_max * np.exp(-1 * decay_const * (pos+0.15))) + ( centerF_max * np.exp(-1*(pos-mu)**2 / (2*stdev**2)) )  + (wallF_max * np.exp(1 * decay_const * (pos-0.15)))
    
    f_at_x = fx(pos)
    slope = deriv(fx, pos, dx=0.00001)
    
    return slope, f_at_x, (wallF_max, decay_const, stdev, centerF_max)


def plot_landscape(f_at_x, wallF_max, decay_const, stdev, centerF_max):
    fig, axarr = plt.subplots(2, sharex=True)
    
    xcoords = np.linspace(-.15,0.15,200)
    slope_at_x, f_at_x,  _ = landscape(xcoords)
    axarr[0].plot(xcoords, f_at_x)
    axarr[0].set_title("Crosswind repulsion landscape", fontsize = 14)
    axarr[0].set_xlim(-.15,0.15)
    axarr[0].set_xlabel("crosswind position")
    axarr[0].set_ylabel("scariness")
    axarr[0].xaxis.grid(True)
#    axarr[0].savefig("repulsion_landscape.png")
    
    # plot derivative
    axarr[1].axhline(y=0, color="grey", lw=1, alpha=0.4)
#    axarr[1].plot(xcoords, slope_at_x, label="derivative")
    axarr[1].plot(xcoords, -1* slope_at_x, label="-derivative")
    axarr[1].set_title("Slope of crosswind repulsion landscape", fontsize = 14)
    axarr[1].set_xlim(-.15,0.15)
    axarr[1].set_xlabel("crosswind position")
    axarr[1].set_ylabel("slope")
    axarr[1].xaxis.grid(True)
    axarr[1].legend()
    
    plt.tight_layout()
    
    plt.savefig("./figs/repulsion_landscape wallF_max{wallF_max} decay_const{decay_const} stdev{stdev} centerF_max{centerF_max}.png".format(wallF_max=wallF_max, decay_const=decay_const, stdev= stdev, centerF_max=centerF_max))
    plt.show()


def main(pos, plotting=False):
    slope_at_x, f_at_x, plot_title_args = landscape(pos)
    
    if plotting is True:
        plot_landscape(f_at_x, *plot_title_args)
        
    return -1*slope_at_x  # this is the force on the agent in the y component


if __name__ == '__main__':
    force_y = main(0.149, plotting=True)