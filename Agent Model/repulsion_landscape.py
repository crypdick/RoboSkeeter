# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

@author: norepinefriend
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as deriv

def landscape(x):
    # exp params
    y0 = 1e-1
    decay_const = 90
    # gaussian params
    mu, stdev, a = 0, 0.04, 0.04
    
    fx = lambda x : (y0 * np.exp(-1 * decay_const * (x+0.15))) + ( a * np.exp(-1*(x-mu)**2 / (2*stdev**2)) )  + (y0 * np.exp(1 * decay_const * (x-0.15)))
    
    y = fx(x)
    slope = deriv(fx,x, dx=0.00001)
    
    return y, slope

def plot_landscape():
    f, axarr = plt.subplots(2, sharex=True)
    
    xcoords = np.linspace(-.15,0.15,200)
    y, slope_at_x = landscape(xcoords)
    axarr[0].plot(xcoords, y)
    axarr[0].set_title("Crosswind repulsion landscape", fontsize = 14)
    axarr[0].set_xlim(-.15,0.15)
    axarr[0].set_xlabel("crosswind position")
    axarr[0].set_ylabel("scariness")
#    axarr[0].savefig("repulsion_landscape.png")
    
    # plot derivative
    axarr[1].plot(xcoords, slope_at_x)
    axarr[1].set_title("Slope of crosswind repulsion landscape", fontsize = 14)
    axarr[1].set_xlim(-.15,0.15)
    axarr[1].set_xlabel("crosswind position")
    axarr[1].set_ylabel("slope")
    
    plt.tight_layout()
    
    plt.savefig("repulsion_landscape.png")
    plt.show()


def main(x, plotting=False):
    y, slope_at_x = landscape(x)
    print slope_at_x
    
    if plotting is True:
        plot_landscape()
        
    return slope_at_x, fx


#y0 = 1
#decay_const = 90
#x = np.linspace(-.15,0.06,200)
##fx = y0 * np.exp(-1 * decay_const * (x))
#plt.plot(x,fx)
#fx = y0 * np.exp(-1 * decay_const * (x+0.15))
#plt.plot(x,fx, c='r')

if __name__ == '__main__':
    slope_at_x, fx = main(.13, plotting=True)