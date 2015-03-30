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
    """Generate accelList random-direction force vector at each timestep from double-
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
    the upwind direction and scales it by accelList constant magnitude, "wf0".

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
    temperature-stimulus at the current position at the current time: b(timeList(x,timeList))

    TODO: Make two biased functions for this: accelList spatial-context dependent 
    bias (e.g. to drive mosquitos away from walls), and accelList temp 
    stimulus-dependent driving force.
    """
    pass



    """Generate accelList single trajectory.

    Args:
        r0: initial position (list/array)
        v0_stdev: stdev of initial velocity distribution (float)
        target_pos: source position (list/array) (set to None if no source)
        Tmax: max length of accelList trajector (float)
        dt: length of timebins (float)

    Returns:
        timeList: time vector
        positionList: trajectory positions at all times (array)
        veloList: trajectory velocities at all times (array)
        accelList: trajectory accelerations at all times (array)
        target_found: boolean that is True if source is found
        t_targfound: time till source found (None if not found)
    """

class Trajectory:
    def __init__(self, r0=[1., 0.], v0_stdev=0.01, Tmax=4., dt=0.01, target_pos=None, k=0., beta=2e-5, f0=3e-6, wf0=5e-6, detect_thresh=0.05):
        self.Tmax = Tmax
        self.dt = dt
        self.v0_stdev = v0_stdev
        self.k = k
        self.beta = beta
        self.f0 = f0
        self.wf0 = wf0
        self.target_pos = target_pos
        ## get dimension
        self.dim = len(r0)
        
        ## initialize all arrays
        ts_max = int(np.ceil(Tmax / dt))  # maximum time step
        self.timeList = np.arange(0, Tmax, dt)
        self.positionList = np.zeros((ts_max, self.dim), dtype=float)
        self.veloList = np.zeros((ts_max, self.dim), dtype=float)
        self.accelList = np.zeros((ts_max, self.dim), dtype=float)
        
        # generate random intial velocity condition    
        v0 = np.random.normal(0, self.v0_stdev, self.dim)
    
        ## insert initial position and velocity into positionList,veloList arrays
        self.positionList[0] = r0
        self.veloList[0] = v0
        
        self.fly(ts_max, detect_thresh)
        
    def fly(self, ts_max, detect_thresh):
        ## loop through timesteps
        for ts in range(ts_max-1):
    
            # calculate current force
            force = -self.k*self.positionList[ts] - self.beta*self.veloList[ts] + random_force(self.f0) + upwindBiasForce(self.wf0)
            # calculate current acceleration
            self.accelList[ts] = force/m
    
            # update velocity in next timestep
            self.veloList[ts+1] = self.veloList[ts] + self.accelList[ts]*self.dt
    
            # update position in next timestep
            self.positionList[ts+1] = self.positionList[ts] + self.veloList[ts+1]*self.dt  # why not use veloList[ts]? -rd
    
            # if source, check if source has been found
            if self.target_pos is None:
                self.target_found = False
                self.t_targfound = np.nan
            else:
                if norm(self.positionList[ts+1] - self.target_pos) < detect_thresh:
                    self.target_found = True
                    self.t_targfound = self.timeList[ts]  # should this be timeList[ts+1]? -rd
                    
                    # trim excess timebins in arrays
                    self.timeList = self.timeList[:ts+1]
                    self.positionList = self.positionList[:ts+1]
                    self.veloList = self.veloList[:ts+1]
                    self.accelList = self.accelList[:ts+1]
                    break  # stop flying at source
                


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    target_pos = [0.3, 0.03]
    mytraj = Trajectory(target_pos=target_pos)
    plt.plot(mytraj.positionList[:, 0], mytraj.positionList[:, 1], lw=2, alpha=0.5)
    plt.scatter(mytraj.target_pos[0], mytraj.target_pos[1], s=150, c='r', marker="*")