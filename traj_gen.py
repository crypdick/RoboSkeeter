"""
Created on Mon Mar 23 15:07:40 2015

@author: rkp, rbd

Generate "mosquito" trajectories using harmonic oscillator equations.
"""

import numpy as np
from numpy.linalg import norm

## define params
# These are the default params. They will get reassigned if this script is
# instantiated with user input
#Tmax = 3  # maximum flight time (s)
## Behavioral data: <control flight duration> = 4.4131 +- 4.4096
#dt = 0.01  # (s) =10ms
m = 2.5e-6  # mass (kg) =2.6 mg
#k = 0.  # spring constant (kg/s^2)
#beta = 1e-6  # damping force (kg/s) NOTE: if beta is too big, things blow up
#f0 = 5e-6  # random driving force exp term for exp distribution (not N)
#wf0 = 5e-6  # upwind bias force magnitude (N)
#
#rdetect = 0.02  # distance from which mozzie can detect source (m) =2 cm


def random_force(f0, dim=2):
    """Generate a random-direction force vector at each timestep from double-
    exponential distribution with exponent f0.

    Args:
        f0: random force distribution exponent

    Returns:
        random force x and y components as an array
    """
    if dim == 2:
#        return  [0, 0]
        return np.random.laplace(0, f0, size=2)
    else:
        raise NotImplementedError('Too many dimensions!')


def upwindBiasForce(wf0, upwind_direction=np.pi, dim=2):
    """Biases the agent to fly upwind. Picks the direction +- pi/2 rads from
    the upwind direction and scales it by a constant magnitude, "wf0".

    Args:
        wf0: bias distribution exponent
        upwind_direction: direction of upwind (in radians)

    Returns:
        upwind bias force x and y components as an array
    """
    if dim == 2:
        if wf0 == 0:
            return [0, 0]
        else:
            return [-wf0, 0]
#==============================================================================
        # Code for randomized wind force
#             # chose direction
#             force_direction = np.random.uniform(upwind_direction - (np.pi/2), upwind_direction + (np.pi/2))
#             w = np.random.normal(wf0, size=2)
#             #return x and y components of bias force as an array
#     #        return wf0 * np.array([np.cos(force_direction), np.sin(force_direction)]) #constant wf0
#     #        return w * force_direction #force mag drawn from dist
#             fx = 1
#             while fx > 0:  # hack to only select negative fx
#                 fx, fy = np.random.laplace(0, wf0, size=2)    
#             return [fx, fy]
#==============================================================================
    else:
        raise NotImplementedError('wind bias only works in 2D right now!')


def stimulusDrivingForce():
    """[PLACEHOLDER]
    Force driving agegent towards stimulus source, determined by
    temperature-stimulus at the current position at the current time: b(T(x,t))

    TODO: Make two biased functions for this: a spatial-context dependent 
    bias (e.g. to drive mosquitos away from walls), and a temp 
    stimulus-dependent driving force.
    """
    pass


def traj_gen(r0=[1., 0.], v0_stdev=0.01, Tmax=4., dt=0.01, rs=None, k=0., beta=2e-5, f0=7e-7, wf0=5e-6, detect_thresh=0.02):
    """Generate a single trajectory.

    Args:
        r0: initial position (list/array)
        v0_stdev: stdev of initial velocity distribution (float)
        rs: source position (list/array) (set to None if no source)
        Tmax: max length of a trajector (float)
        dt: length of timebins (float)

    Returns:
        t: time vector
        r: trajectory positions at all times (array)
        v: trajectory velocities at all times (array)
        a: trajectory accelerations at all times (array)
        source_found: boolean that is True if source is found
        tfound: time till source found (None if not found)
    """

    ## get dimension
    dim = len(r0)

    ## initialize all arrays
    ts_max = int(np.ceil(Tmax / dt))  # maximum time step
    t = np.arange(0, Tmax, dt)
    r = np.zeros((ts_max, dim), dtype=float)
    v = np.zeros((ts_max, dim), dtype=float)
    a = np.zeros((ts_max, dim), dtype=float)
    
    # generate random intial velocity condition    
    v0 = np.random.normal(0, v0_stdev, 2)

    ## insert initial position and velocity into r,v arrays
    r[0] = r0
    v[0] = v0

    ## loop through timesteps
    for ts in range(ts_max-1):

        # calculate current force
        force = -k*r[ts] - beta*v[ts] + random_force(f0) + upwindBiasForce(wf0)
        # calculate current acceleration
        a[ts] = force/m

        # update velocity in next timestep
        v[ts+1] = v[ts] + a[ts]*dt

        # update position in next timestep
        r[ts+1] = r[ts] + v[ts+1]*dt  # why not use v[ts]? -rd

        # if source, check if source has been found
        if rs is not None:
            if norm(r[ts+1] - rs) < detect_thresh:
                source_found = True
                tfound = t[ts]  # should this be t[ts+1]? -rd
                # trim excess timebins in arrays
                t = t[:ts+1]
                r = r[:ts+1]
                v = v[:ts+1]
                a = a[:ts+1]
                break  # stop flying at source
    else:  # why is this else at this indentation? -rd
        source_found = False
        tfound = np.nan

    return t, r, v, a, source_found, tfound


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    t, r, v, a, source_found, tfound = traj_gen(rs = [0.2, 0.05])
    plt.plot(r[:, 0], r[:, 1], lw=2, alpha=0.5)
    plt.scatter(rs[0], rs[1], s=150, c='r', marker="*")