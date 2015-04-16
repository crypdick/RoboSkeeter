"""
Created on Mon Mar 23 15:07:40 2015

@author: rkp, rbd

Generate "mosquito" trajectories (class object) using harmonic oscillator equations.
"""

import numpy as np
from numpy.linalg import norm
import plotting_funcs
import baseline_driving_forces

## define params
# population weight data: 2.88 +- 0.35mg
m = 3.0e-6 #2.88e-6  # mass (kg) =2.88 mg


def place_heater(target_pos):
    if target_pos is None:
            return None
    elif target_pos == "left":
        return [0.86, -0.0507]
    elif target_pos == "right":
        return [0.86, 0.0507]
    elif type(target_pos) is list:
        return target_pos
    else:
        raise Exception('invalid heater type specified')


def place_agent(agent_pos):
    if type(agent_pos) is list:
        return agent_pos
    if agent_pos == "center":
        return [0.1524, 0.]  # center of box
    if agent_pos == "cage":  # bounds of cage
        return [np.random.uniform(0.1143, 0.1909), np.random.uniform(-0.0381, 0.0381)]
    else:
        raise Exception('invalid agent position specified')


class Trajectory:
    """Generate single trajectory, forbidding agent from leaving bounds

    Args:
        agent_position: (list/array, "cage", "center")
            sets initial position r0 (meters)
        v0_stdev: (float)
            stdev of initial velocity distribution 
        target_pos: (list/array, "left", "right", or "None")
            heater position  (set to None if no source)
        Tmax: (float)
            max length of accelList trajectory 
        dt: (float)
            length of timebins to divide Tmax by 
        k: (float)
            spring constant, disabled 
        beta: (float)
            damping force (kg/s)  NOTE: if beta is too big, things blow up
        rf: (float)
            random driving force exp term for exp distribution 
        wtf: (float)
            upwind bias force magnitude  # TODO units
        detect_thresh: (float)
            distance mozzie can detect target in (m), 2 cm + radius of heaters,
            (0.00635m/2= 0.003175)
        bounce: (None, "crash")
            "crash" sets whether agent velocity gets set to zero in the corresponding component if it crashes into a wall
        boundary: (array)
            specify where walls are  (minx, maxx, miny, maxy)
        

    All args are in SI units and based on behavioral data:
    Tmax, dt: seconds (data: <control flight duration> = 4.4131 +- 4.4096
    

    Returns:
        trajectory object
    """
    boundary = [0.0, 1.0, 0.15, -0.15]  # these are real dims of our wind tunnel
    def __init__(self, agent_pos, v0_stdev, Tmax, dt, target_pos, beta, rf, wtf, detect_thresh, bounded, bounce, wallF, plotting=False, title="Individual trajectory", titleappend = '', k=0.):
        """ Initialize object with instant variables, and trigger other funcs. 
        """
        self.Tmax = Tmax
        self.dt = dt
        self.v0_stdev = v0_stdev
        self.k = k
        self.beta = beta
        self.rf = rf
        self.wtf = wtf
        self.detect_thresh = detect_thresh     
        self.bounce = bounce
        self.wallF = wallF
        
        # place heater
        self.target_pos = place_heater(target_pos)
        
        # place agent
        r0 = place_agent(agent_pos)
        self.dim = len(r0)  # get dimension
        
        self.target_found = False
        self.t_targfound = np.nan
        
        ## initialize all arrays
        ts_max = int(np.ceil(Tmax / dt))  # maximum time step
        self.timeList = np.arange(0, Tmax, dt)
        self.positionList = np.zeros((ts_max, self.dim), dtype=float)
        self.veloList = np.zeros((ts_max, self.dim), dtype=float)
        self.accelList = np.zeros((ts_max, self.dim), dtype=float)
        self.ForcesList = np.zeros((ts_max, 3, self.dim), dtype=float)  # we have 3 drivers
        
        # generate random intial velocity condition    
        v0 = np.random.normal(0, self.v0_stdev, self.dim)
    
        ## insert initial position and velocity into positionList,veloList arrays
        self.positionList[0] = r0
        self.veloList[0] = v0
        
        self.fly(ts_max, detect_thresh, self.boundary, bounded)
        
        if plotting is True:
            self.title = title
            self.titleappend = titleappend
            plotting_funcs.plot_single_trajectory(self.positionList, self.target_pos, self.detect_thresh, self.boundary, self.title, self.titleappend)
        
    def fly(self, ts_max, detect_thresh, boundary, bounded):
        """Run the simulation using Euler's method"""
        ## loop through timesteps
        for ts in range(ts_max-1):
            # calculate drivers
            randF = baseline_driving_forces.random_force(self.rf)
            self.ForcesList[ts][0] = randF
            upwindF = baseline_driving_forces.upwindBiasForce(self.wtf)
            self.ForcesList[ts][1] = upwindF
            wallRepulsiveF = baseline_driving_forces.repulsionF(self.positionList[ts])
            self.ForcesList[ts][2] = wallRepulsiveF
            # calculate current force
            force = -self.k*self.positionList[ts] - self.beta*self.veloList[ts] + randF + upwindF + wallRepulsiveF
            # calculate current acceleration
            self.accelList[ts] = force/m
    
            # update velocity in next timestep
            self.veloList[ts+1] = self.veloList[ts] + self.accelList[ts]*self.dt
            # update position in next timestep
            candidate_pos = self.positionList[ts] + self.veloList[ts+1]*self.dt  # why not use veloList[ts]? -rd
            
            if bounded is True:  # if walls are enabled
                ## forbid mosquito from going out of bounds
                # check x dim
                if candidate_pos[0] < boundary[0]:  # too far left
                    candidate_pos[0] = boundary[0] + 1e-4
                    if self.bounce is "crash":
                        self.veloList[ts+1][0] = 0.
                elif candidate_pos[0] > boundary[1]:  # too far right
                    candidate_pos[0] = boundary[1] - 1e-4
                    self.land(ts)  # end trajectory when reach end of tunnel
                    break  # stop flying at end  
                # check y dim
                if candidate_pos[1] > boundary[2]:  # too far up
                    candidate_pos[1] = boundary[2] + 1e-4
                    if self.bounce is "crash":
#                        print "teleport!"
                        self.veloList[ts+1][1] = 0.
                elif candidate_pos[1] < boundary[3]:  # too far down
                    candidate_pos[1] = boundary[3] - 1e-4
                    if self.bounce is "crash":
#                        print "teleport!"
                        self.veloList[ts+1][1] = 0.
                
            self.positionList[ts+1] = candidate_pos
    
            # if there is a target, check if we are finding it
            if self.target_pos is None:
                self.target_found = False
                self.t_targfound = np.nan
            else:
                if norm(self.positionList[ts+1] - self.target_pos) < self.detect_thresh:
                    self.target_found = True
                    self.t_targfound = self.timeList[ts]  # should this be timeList[ts+1]? -rd
                    self.land(ts)
                    break  # stop flying at source  
                    
    def land(self, ts):    
        # trim excess timebins in arrays
        self.timeList = self.timeList[:ts+1]
        self.positionList = self.positionList[:ts+1]
        self.veloList = self.veloList[:ts+1]
        self.accelList = self.accelList[:ts+1]


if __name__ == '__main__':
    target_pos = "left"
    
    mytraj = Trajectory(agent_pos="cage", target_pos="left", plotting = True, v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=(80, 1e-4))