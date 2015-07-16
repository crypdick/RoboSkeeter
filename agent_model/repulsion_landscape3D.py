# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 16:41:26 2015

Given an agent crosswind position, return force.

@author: Richard Decal

create fxn once,
then solve for forces as needed

# TODO: subtract global minimum from the weighting function so that at some places the force is 0?
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative as deriv
from scipy.integrate import quad

INT_DIST = 0.003 / 2. # integratal distance in mm
DX = 0.00001
BOUNDS = [0.0, 1.0, 0.127, -0.127, 0., 0.254]
SCALING = 9.5

def x_dist(pos_x):
    # Up/downwind (x) repulsion # TODO: @Sharri this means that agent is pushed downwind by this force
    # Fit to a line:
    # Upwind(x) = p1x + p2
    # where coefficients are -0.011364	0.015125

    # max at 0, value 0.015125
    scalar = 1. / 0.015125  # scalar to bring peak of dist to 1
    return ((-0.011364) * pos_x + 0.015125 ) * scalar

(total_area_x, err_x) = quad(x_dist, BOUNDS[0], BOUNDS[1])

def x_prob(pos_x):
    prob_pos_x = quad(x_dist, pos_x-INT_DIST, pos_x+INT_DIST)[0] / total_area_x
    return prob_pos_x

def x_weight_fxn(pos_x):
    return 1. - x_dist(pos_x)

def y_dist(pos_y):
    # Crosswind (y) repulsion
    # Fit with a 6th order polynomial (r**2= 0.9359)
    # Crosswind(y) = p1x**6 + p2x**5 + p3x4 + p4x**3 + p5x**2 + p6x
    # -80641	-204.09	1516.4	-17.248	-3.2367	0.16778	0.0089941

    #max 0.05689599204037 at x=-.111362001508, therefore
    scalar = 1./0.05689599204037

    return (
         -80641 * pos_y **6 + -204.09 * pos_y **5 + 1516.4 * pos_y **4 + -17.248 * pos_y **3
         + (-3.2367 * pos_y **2) + 0.16778 * pos_y + 0.0089941) * scalar

(total_area_y, err_y) = quad(y_dist, BOUNDS[3], BOUNDS[2]) # boundaries +/- switched in our convention

def y_prob(pos_y):
    # divide by its area to get normalized distribution
    prob_pos_y = quad(y_dist, pos_y-INT_DIST, pos_y+INT_DIST)[0] / total_area_y
    return prob_pos_y

def y_weight_fxn(pos_y):
    return 1. - y_dist(pos_y)

def z_dist(pos_z):
    # Elevation (z) repulsion
    # fit with sum of 3 Gaussians (r**2 = 0.995):
    # Elevation(x) = a1*exp(-((x-b1)/c1)^2) + a2*exp(-((x-b2)/c2)^2) + a3*exp(-((x-b3)/c3)^2)
    # 0.13083	0.23437	0.0052473	0.086796	0.2245 0.0132	0.036283	0.19091	0.035445
    # NOTE: c parameters have "absorbed" the "2 x _ " from the Gaussian function

    # max 0.190997 at z=0.233635, scalar is therefore
    scalar = 1. / 0.190997

    return scalar * (0.13083 * np.exp(-np.power(pos_z - 0.23437, 2.) / (np.power(0.0052473, 2.))) +
        0.086796 * np.exp(-np.power(pos_z - 0.2245, 2.) / (np.power(0.0132, 2.))) +
        0.036283 * np.exp(-np.power(pos_z - 0.19091, 2.) / (np.power(0.035445, 2.))))

(total_area_z, err_z) = quad(z_dist, BOUNDS[4], BOUNDS[5])

def z_prob(pos_z):
    prob_pos_z = quad(z_dist, pos_z-INT_DIST, pos_z+INT_DIST)[0] / total_area_z
    return prob_pos_z

def z_weight_fxn(pos_z):
    return 1. - z_prob(pos_z)

def solve_direction(function, pos):
    """
    :param function: probability distribution for some dimension
    :param pos: the position along the dimension
    :return: which direction to apply the repulsion force.
    """
    slope = deriv(function, pos, dx=DX)
    # if function.__name__ in ["y_prob", "y_dist"]:
    #     # hack to fix slope calculation issues with our y dimension, which has a flipped sign convention
    #     slope *= -1
    if slope < 0:
        direction = -1.
    elif slope > 0:
        direction = 1.
    else: # this shouldn't happen
        direction = 0.

    return direction



def xyz_to_weights(xyz):
    """
    given position, return the weighting of the repulsion force and direction for each dimension
    :param xyz:
    :return:
    """
    x, y, z = xyz
    x_weight, y_weight, z_weight = x_weight_fxn(x), y_weight_fxn(y), z_weight_fxn(z)
    dir_x, dir_y, dir_z = solve_direction(x_prob, x), solve_direction(y_prob, y), solve_direction(z_prob, z)

    return np.array([x_weight * dir_x, y_weight* dir_y, z_weight * dir_z])



def plot_landscape(BOUNDS):
    for i, func_list in enumerate([[x_dist, x_prob, x_weight_fxn],
                                   [y_dist, y_prob, y_weight_fxn],
                                   [z_dist, z_prob, z_weight_fxn]]):
        resolution = 200
        # fig.set_size_inches(10,10)
    #    plt.suptitle("repulsion_type", y=1.05)
        if i == 0:
            dim = 'x'
            bounds = BOUNDS[0:2]
            coords = np.linspace(BOUNDS[0], BOUNDS[1], resolution)
        elif i == 1:
            dim = 'y'
            print BOUNDS
            bounds = BOUNDS[2:4][::-1] # flip order to compensate for our convention
            print bounds
            coords = np.linspace(bounds[0], bounds[1], resolution) 
            coords = coords[::-1] # and re-reverse to get proper figures
        elif i == 2:
            dim = 'z'
            bounds = BOUNDS[4:]
            coords = np.linspace(BOUNDS[4], BOUNDS[5], resolution)

        position_dist = np.array([func_list[0](coord) for coord in coords])
        probability_dist = np.array([func_list[1](coord) for coord in coords])
        scariness = np.array([func_list[2](coord) for coord in coords])
        directions = np.array([solve_direction(func_list[1], coord) for coord in coords])

        fig, axarr = plt.subplots(4, sharex=True)
        # position distributions fit
        axarr[0].plot(coords, position_dist)
        axarr[0].set_xlim(*bounds)
        axarr[0].set_title("{} position distribution (model)".format(dim), fontsize = 14)
        axarr[0].set_ylabel("Fit")

        # probability distributions
        axarr[1].plot(coords, probability_dist)
        axarr[1].set_title("{} probability distr landscape".format(dim), fontsize = 14)
        axarr[1].set_xlim(*bounds)
        axarr[1].set_ylabel("P({} position)".format(dim))

        # weighting function
        axarr[2].plot(coords, scariness)
        axarr[2].set_title("{} repulsion landscape".format(dim), fontsize = 14)
        axarr[2].set_xlim(*bounds)
        axarr[2].set_ylabel("Scariness")
        axarr[2].xaxis.grid(True)
    #    axarr[0].savefig("repulsion_landscape.png")
        
        # plot directions
        axarr[3].plot(coords, directions, lw=3, c='g')
        axarr[3].set_title("Force direction (relative to +{})".format(dim), fontsize = 14)
        axarr[3].set_xlim(*bounds)
        axarr[3].set_ylim([-1,1])
        axarr[3].set_ylabel("Direction of the force".format(dim))
        axarr[3].xaxis.grid(True)

        axarr[3].set_xlabel("Agent {} position (meters)".format(dim))# bottom lable of shared axis
        
        plt.tight_layout()
        
        plt.show()
    
    
# def solve_lamba2a_ratio(wallF_max=8e-8, decay_const = 90):
# #    ycoords = np.linspace(-0.127, 0.127, 200)
# #    new_x = np.linspace(8e-7, 8e-9, 200)
#
# #    fxa = lambda a: lambda wallF_max : wallF_max * np.exp(-1 * 90 * (x+0.127))
# #    fx_lamb = lambda decay_const : 8e-8 * np.exp(-1 * decay_const * (0+0.127))
# #    fxa_func = fxa(new_x)
# ##    print fxa_func
# #    fx_lamb_func = fx_lamb(ycoords)
# ##    fig3, ax3 = plt.figure(3)
# #    plt.plot(fx_lamb_func, fxa_func)
# #    plt.plot(new_x, fxa_func, c='red')
# #    area_lamb = integr.quad(fx_lamb, 8e-8, 8e-9)
# #    area_a = integr.quad(fxa, 8e-7, 8e-9)
# #    print "area lambda ", area_lamb, "\n area a = ", area_a
#
#     xs = np.linspace(-.127, .127, 200)
#     lambdas = np.linspace(0, 4e-6)
#     fxlambda = []
#     for i in lambdas:
#         fxlambda.append(np.sum(wallF_max * np.exp(-1 * i * (xs+0.127) )))
#
#     consts = np.linspace(0, 100)
#     fxdecays = []
#     for const in consts:
#         fxdecays.append(np.sum(const * np.exp(-1 * 90 * (xs+0.127) )))
#
#     plt.plot(fxlambda, fxdecays)
#
# #    print fxlambda
# #    print fxdecays



def main(plotting=False):
    
    if plotting is True:
        plot_landscape(BOUNDS) # plot repulsion landscapes

if __name__ == '__main__':
    print solve_direction(y_weight_fxn, 0.)
#    force_y = main(plotting=True)
    main(plotting=True)
    #repulsion_funcs = main(plotting=True)
    
#    solve_lamba2a_ratio(wallF_max, decay_const)