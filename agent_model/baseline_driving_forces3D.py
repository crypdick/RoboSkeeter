# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:16:16 2015

Our driving forces.

@author: Richard Decal
"""
import numpy as np
from numpy import linalg, newaxis
from scipy.integrate import quad
from scipy.misc import derivative as deriv


def bias_force(rf, bias='uniform'):
    """Generate random-direction force vector at each timestep from double-
    exponential distribution given exponent term rf.

    Args:
        rf: random force distribution exponent (float)

    Returns:
        random force x and y components (array)
    """
    mag_thresh = 1e-5
    ends = gen_symm_vecs(3)
    # the following params were fitted from Sharri's flight data
    mu = -0.405632480939
    sigma = 0.932352661694 * 0.5
    scale = 0.666555094117
    # # following params were fitted from the Dickinson flight data
    # mu = 0.600023812816
    # sigma = 0.719736466122
    # scale = 1.82216219069
    mag = np.random.lognormal(mean=mu, sigma=sigma, size=1)
    if mag * rf > mag_thresh: # filter out huge magnitudes
        return ends * 0. * rf

#    print mag*rf  # check the order of the biasF
    return mag * ends * rf
        
#==============================================================================
### depreciated
# def random_force(rf, dim=2):
#     """Generate random-direction force vector at each timestep from double-
#     exponential distribution given exponent term rf.
# 
#     Args:
#         rf: random force distribution exponent (float)
# 
#     Returns:
#         random force x and y components (array)
#     """
#     mag_thresh = 3e-6
#     if dim == 2:
#         ends = gen_symm_vecs(2)
#         # following params were fitted from the Dickinson fligt data
#         mu = 0.600023812816
#         sigma = 0.719736466122
#         scale = 1.82216219069
#         mag = np.random.lognormal(mean=mu, sigma=sigma, size=1)
#         return mag * ends * rf
#     if dim == 3: # FIXME: 3d symmetric
#         ends = gen_symm_vecs(3)
#         # following params were fitted from the Dickinson fligt data
#         mu = 0.600023812816
#         sigma = 0.719736466122
#         scale = 1.82216219069
#         mag = np.random.lognormal(mean=mu, sigma=sigma, size=1)
#         if mag * rf > mag_thresh: # filter out huge magnitudes
#             return ends * mag_thresh
#         return mag * ends * rf
#     else:
#==============================================================================


def gen_symm_vecs(dims):
    """generate radially-symmetric vectors sampled from the unit circle.  These
    can then be scaled by a force to make radially symmetric distributions.
    
    making a scatter plot of many draws from this function makes a unit circle.
    """
    vecs = np.random.normal(size=dims)
    mags = linalg.norm(vecs, axis=-1)

    ends = vecs / mags[..., newaxis]
    
    return ends  # vector


def upwindBiasForce(wtf, upwind_direction=0.):
    """Biases the agent to fly upwind. Constant push with strength wtf
    
    [formerly]: Picks the direction +- pi/2 rads from
    the upwind direction and scales it by accelList constant magnitude, "wtf".

    Args:
        wtf: bias distribution exponent
        upwind_direction: direction of upwind (in radians)

    Returns:
        upwind bias force x and y components as an array
    """
    if wtf == None:
        return [0., 0., 0.]
    else:
        return [wtf, 0., 0.]  # wf is constant, directly upwind

#def wall_force_field(current_pos, current_velo, wallF, wallX_pos=[0., 1.], wallY_pos=[-0.15, 0.15]):
#    """If agent gets too close to wall, inflict it with repulsive forces as a function of how close it is to the wall and it's speed towards the wall.
#    
#    Args:
#        current_pos: (x,y) coords of agent right now (array)
#        wallF: (None | tuple)
#            None- disabled
#            tuple- (lambda (float), y0 (float). Lambda is the decay constant in the exponentially decaying wall repulsive force, y0 is where the decaying starts.
#        wallX_pos: Wall X coords (array)
#        wallY_pos: Wall Y coords (array)
#    
#    Returns:
#        wall_force: (array)
#    """
#    if wallF is None:
#        return [0., 0.]
##    elif type(wallF) == "tuple":
#    else:
#        wallF_lambda, wallF_y0 = wallF
#        wallF_x = 0.
#        wallF_y = 0.
#        posx, posy = current_pos
#        velox, veloy = current_velo
#        y_dist = 0.15 - abs(posy) # distance of agent from wall
#        force_mag_y = wallF_magnitude(y_dist, wallF_lambda, wallF_y0)
#        if (posy < 0 and veloy < 0) or (posy > 0 and veloy > 0): # heading towards top OR bottom
#            wallF_y = -1. * veloy * force_mag_y  # equal an opposite force, scaled by force mag
#        return np.array([wallF_x, wallF_y])
#
#def wallF_magnitude(distance, wallF, y0):
#    """magnitude of wall repulsion, a function of position
#    
#    """
#    return y0 * np.exp(-1 * wallF * distance)


def repulsionF(position, repulsion_funcs, wallF_params):
    """repulsion as a function of position.
    """
    scalar = wallF_params[0]
    intd = 0.003 / 2 # integratal distance in mm
    pos_x, pos_y, pos_z = position
    repulsion_x, repulsion_y, repulsion_z = repulsion_funcs
    dx = 0.00001
    
    # solve direction of forces
    slope_rep_x, slope_rep_y, slope_rep_z = deriv(repulsion_x, pos_x, dx=dx), deriv(repulsion_y, pos_y, dx=dx), deriv(repulsion_z, pos_z, dx=dx)
    directions = []
    for slope in [slope_rep_x, slope_rep_y, slope_rep_z]:
        if slope < 0:
            directions.append(1.)
        elif slope > 0:
            directions.append(-1.)
        else: # this shouldn't happen
            directions.append(0.)
    
    # [0] to discard error term
    repulsionF =  scalar * np.array([directions[0] * quad(repulsion_x, pos_x-intd, pos_x+intd)[0], \
        scalar * directions[1] * quad(repulsion_y, pos_y-intd, pos_y+intd)[0],\
        scalar * directions[2] * quad(repulsion_z, pos_z-intd, pos_z+intd)[0]])
        
#    print "pos", position
#    print "repF", repulsionF
    return repulsionF


#def brakingF(candidate_pos, totalF_x, totalF_y, boundary):
#    # check x dim
#    if candidate_pos[0] < boundary[0]:  # too far left
#        outside_correct = True
#        brakeF_x = -6 * totalF_x
#        brakeF_y = 0.
#    # check y dim
#    if candidate_pos[1] > boundary[2]:  # too far up
#        outside_correct = True
#        brakeF_x = 0.
#        brakeF_y = -6 * totalF_y
##                        print "crash! top wall"
#    elif candidate_pos[1] < boundary[3]:  # too far down
#        outside_correct = True
#        brakeF_x = 0.
#        brakeF_y = -6 * totalF_y
#    else:
#        outside_correct, brakeF_x, brakeF_y = False, 0., 0.
#        
#    return outside_correct, brakeF_x, brakeF_y
