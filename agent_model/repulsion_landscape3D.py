# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

Given an agent crosswind position, return force.

@author: Richard Decal

TODO: create fxn once,
then solve for forces as needed
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as deriv
from scipy.integrate import quad


def landscape(normed=True):
    """exponential decay params: wallF_max, decay_const
    gaussian params: mu, stdev, centerF_max
    
    repulsion_type: (str)
        "walls_only", 'walls+trenches', 'walls+center', 'center_only'
    """
#    if repulsion_type == 'gaussian_sum':
    # gass func has the form: np.exp(-np.power(pos - mu, 2.) / (2 * np.power(sig, 2.)))
    
    # set up repulsion field in y dim
    A_left = 1
    mu_left = -.09 # TODO fit
    sig_left = .01   # TODO fit
    A_center = .2
    mu_center = 0 # TODO fit
    sig_center = .05 # TODO fit
    A_right = 1
    mu_right = .09 # TODO fit
    sig_right = .01 # TODO fit
    repulsion_y = lambda pos_y: A_left * np.exp(-np.power(pos_y - mu_left, 2.) / (2 * np.power(sig_left, 2.))) +\
        A_center * np.exp(-np.power(pos_y - mu_center, 2.) / (2 * np.power(sig_center, 2.))) +\
            A_right * np.exp(-np.power(pos_y - mu_right, 2.) / (2 * np.power(sig_right, 2.)))
    if normed is True:
        # divide function by its area to get normalized function
        (area, err) = quad(repulsion_y, -1, 1)
        normed_repulsion_y = lambda pos_y: repulsion_y(pos_y) / area
        
    # TODO: implement
    repulsion_x = lambda pos_x: pos_x * 0
    normed_repulsion_x = repulsion_x
    
    repulsion_z = lambda pos_z: pos_z * 0
    normed_repulsion_z = repulsion_z
    
    if normed is True:
        return [normed_repulsion_x, normed_repulsion_y, normed_repulsion_z]
    else:
        return [repulsion_x, repulsion_y, repulsion_z]


def plot_landscape(repulsion_fxn):
    fig, axarr = plt.subplots(2, sharex=True)
    fig.set_size_inches(10,10)
#    plt.suptitle("repulsion_type", y=1.05)
    
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



def main(plotting=False):
    
#    repulsion_type = 'gaussian_sum'
    repulsion_funcs = landscape(normed=True)
    

    
    
#    if pos_y != None:
#        repF = deriv(fy, pos_y, dx=0.00001)
#        return -1*repF  # this is the force on the agent in the y component
    
    if plotting is True:
        plot_landscape(repulsion_funcs[1]) # plot y repulsion landscape
        
    return repulsion_funcs
        
    

if __name__ == '__main__':
#    force_y = main(plotting=True)
    repulsion_funcs = main(plotting=True)
    
#    solve_lamba2a_ratio(wallF_max, decay_const)