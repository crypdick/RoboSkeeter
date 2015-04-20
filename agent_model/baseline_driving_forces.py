# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 17:16:16 2015

Our driving forces.

@author: Richard Decal
"""
import numpy as np
from numpy import linalg, newaxis, random
from matplotlib import collections
import repulsion_landscape


def random_force(rf, dim=2):
    """Generate random-direction force vector at each timestep from double-
    exponential distribution given exponent term rf.

    Args:
        rf: random force distribution exponent (float)

    Returns:
        random force x and y components (array)
    """
    if rf == 0.:
        return np.array([0., 0.])
    if dim == 2:
        ends = gen_symm_vecs(2)
        # following params were fitted from the Dickinson fligt data
        mu = 0.600023812816
        sigma = 0.719736466122
        scale = 1.82216219069
        mag = np.random.lognormal(mean=mu, sigma=sigma, size=1)
        return mag * ends * rf
    else:
        raise NotImplementedError('Too many dimensions!')


def gen_symm_vecs(dims):
    """generate radially-symmetric vectors sampled from the unit circle.  These
    can then be scaled by a force to make radially symmetric distributions.
    
    making a scatter plot of many draws from this function makes a unit circle.
    """
    vecs = np.random.normal(size=dims)
    mags = linalg.norm(vecs, axis=-1)

    ends = vecs / mags[..., newaxis]
    return ends  # vector


def upwindBiasForce(wtf, upwind_direction=0, dim=2):
    """Biases the agent to fly upwind. Constant push with strength wtf
    
    [formerly]: Picks the direction +- pi/2 rads from
    the upwind direction and scales it by accelList constant magnitude, "wtf".

    Args:
        wtf: bias distribution exponent
        upwind_direction: direction of upwind (in radians)

    Returns:
        upwind bias force x and y components as an array
    """
    if dim == 2:
        if wtf == 0:
            return [0, 0]
        else:
            return [wtf, 0]  # wf is constant, directly to right
    else:
        raise NotImplementedError('wind bias only works in 2D right now!')


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


def repulsionF(position, wallF):
    """repulsion as a function of position.
    """
    if wallF is None:
        return np.array([0., 0.])
    else:
        posx, posy = position
        force_y = repulsion_landscape.main(posy, wallF)
        return np.array([0., force_y])


def stimulusDrivingForce():
    """[PLACEHOLDER]
    Force driving agegent towards stimulus source, determined by
    temperature-stimulus at the current position at the current time: b(timeList(x,timeList))

    TODO: Make two biased functions for this: accelList spatial-context dependent 
    bias (e.g. to drive mosquitos away from walls), and accelList temp 
    stimulus-dependent driving force.
    """
    pass