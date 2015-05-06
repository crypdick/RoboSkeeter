# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

Given an agent crosswind position, return force.

@author: Richard Decal
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as deriv
import scipy.integrate as integr


def landscape(b, shrink, wallF_max=8e-8, decay_const = 90, mu=0.):
    """exponential decay params: wallF_max, decay_const
    gaussian params: mu, stdev, centerF_max
    """
#    wallF_max = decay_const*(1/250)
#    
#    # landscape with repulsion in center
#    fx = lambda pos : (wallF_max * np.exp(-1 * decay_const * (pos+0.127))) +  shrink * ( 1/(2*b) * np.exp(-1 * abs(pos-mu)/b) ) + (wallF_max * np.exp(1 * decay_const * (pos-0.127)))

    # no center repulsion; only wall repulsion
    fx = lambda pos : (wallF_max * np.exp(-1 * decay_const * (pos+0.127))) + (wallF_max * np.exp(1 * decay_const * (pos-0.127)))

#    fx = lambda pos : (wallF_max * np.exp(-1 * decay_const * (pos+0.127))) +  shrink * ( 1/(2*b) * np.exp(-1 * abs(pos-mu)/b) ) + (wallF_max * np.exp(1 * decay_const * (pos-0.127)))
#    fx = lambda pos : ( 1/(2*b) * np.exp(-1 * abs(pos-mu)/b) ) * .01
    return fx


def plot_landscape(repulsion_fxn, wallF):
    fig, axarr = plt.subplots(2, sharex=True)
    fig.set_size_inches(10,10)
    
    ycoords = np.linspace(-0.127, 0.127, 200)
    scariness = repulsion_fxn(ycoords)
    
    axarr[1].plot(ycoords, scariness)
    axarr[1].set_title("Crosswind repulsion landscape", fontsize = 14)
    axarr[1].set_xlim(-0.127, 0.127)
    axarr[1].set_ylabel("Scariness")
    axarr[1].xaxis.grid(True)
#    axarr[0].savefig("repulsion_landscape.png")
    
    # plot derivative
    slopes = deriv(repulsion_fxn, ycoords, dx=0.00001)
    
    axarr[0].axhline(y=0, color="grey", lw=1, alpha=0.4)
#    axarr[1].plot(ycoords, slopes, label="derivative")
    axarr[0].plot(ycoords, -1* slopes, label="-derivative of landscape")
    axarr[0].set_title("Crosswind force (-slope of scariness)", fontsize = 14)
    axarr[0].set_xlim(-0.127, 0.127)
    axarr[1].set_xlabel("Agent crosswind position (meters)")
    axarr[0].set_ylabel("Force in the $+y$ direction")
    axarr[0].xaxis.grid(True)
    axarr[0].legend()
    
    plt.tight_layout()
    
    b, shrink, wallF_max, decay_const = wallF
    plt.savefig("./figs/repulsion_landscape wallF_max{wallF_max} decay_const{decay_const} b{b} shrink{shrink}.svg".format(wallF_max=wallF_max, decay_const=decay_const, b=b, shrink=shrink), format='svg')
    plt.show()
    
    
def solve_lamba2a_ratio(wallF_max=8e-8, decay_const = 90):
#    ycoords = np.linspace(-0.127, 0.127, 200)
#    new_x = np.linspace(8e-7, 8e-9, 200)
    
#    fxa = lambda a: lambda wallF_max : wallF_max * np.exp(-1 * 90 * (x+0.127))
#    fx_lamb = lambda decay_const : 8e-8 * np.exp(-1 * decay_const * (0+0.127))
#    fxa_func = fxa(new_x)
##    print fxa_func
#    fx_lamb_func = fx_lamb(ycoords)
##    fig3, ax3 = plt.figure(3)
#    plt.plot(fx_lamb_func, fxa_func)
#    plt.plot(new_x, fxa_func, c='red')
#    area_lamb = integr.quad(fx_lamb, 8e-8, 8e-9)
#    area_a = integr.quad(fxa, 8e-7, 8e-9)
#    print "area lambda ", area_lamb, "\n area a = ", area_a
    
    xs = np.linspace(-.127, .127, 200)
    lambdas = np.linspace(0, 4e-6)
    fxlambda = []
    for i in lambdas:
        fxlambda.append(np.sum(wallF_max * np.exp(-1 * i * (xs+0.127) )))
    
    consts = np.linspace(0, 100)
    fxdecays = []
    for const in consts:
        fxdecays.append(np.sum(const * np.exp(-1 * 90 * (xs+0.127) )))
        
    plt.plot(fxlambda, fxdecays)
        
#    print fxlambda
#    print fxdecays

def main(wallF, pos_y=None, plotting=False):
    repulsion_fxn = landscape(*wallF)
    
    if pos_y != None:
        repF = deriv(repulsion_fxn, pos_y, dx=0.00001)
    
    if plotting is True:
        plot_landscape(repulsion_fxn, wallF)
        
    return -1*repF  # this is the force on the agent in the y component


if __name__ == '__main__':
    # wallF params
    wallF_max=9e-6#1e-7
    decay_const = 90
    
    # center repulsion params
    b = 4e-1  # determines shape
    shrink = 1e-6  # determines size/magnitude
    
    wallF = (b, shrink, wallF_max, decay_const)
    force_y = main(wallF, 0, plotting=True)
    
    solve_lamba2a_ratio(wallF_max, decay_const)