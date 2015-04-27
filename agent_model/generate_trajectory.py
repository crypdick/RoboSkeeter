"""
Created on Mon Mar 23 15:07:40 2015

@author: rkp, rbd

Generate "mosquito" trajectories (class object) using harmonic oscillator equations.
"""

import numpy as np
from numpy.linalg import norm
import pandas as pd
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
    def __init__(self, agent_pos, v0_stdev, Tmax, dt, target_pos, beta, rf, wtf, detect_thresh, bounded, bounce, wallF, plotting=False, title="Individual trajectory", titleappend = '', k=0.):
        """ Initialize object with instant variables, and trigger other funcs. 
        """
        self.boundary = [0.0, 1.0, 0.127, -0.127]  # these are real dims of our wind tunnel
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
        
        self.metadata = dict()
        self.metadata['time_max'] = Tmax
        self.metadata['boundary'] = self.boundary
        self.metadata['target_position'] = self.target_pos
        self.metadata['detection_threshold'] = detect_thresh
        self.metadata['initial_position'] = agent_pos
        self.metadata['initial_velo_stdev'] = v0_stdev
        self.metadata['k'] = k
        self.metadata['beta'] = beta
        self.metadata['rf'] = rf
        self.metadata['wtf'] = wtf
        self.metadata['bounce'] = bounce
        self.metadata['wallF'] = wallF
        # for stats, later
        self.metadata['total_trajectories'] = 0
        self.metadata['time_target_find_avg'] = []
        self.metadata['total_finds'] = 0
        self.metadata['target_found'] = [False]
        self.metadata['time_to_target_find'] = [np.nan]
        
        ## initialize all arrays
        self.dynamics = pd.DataFrame()
        
        ts_max = int(np.ceil(Tmax / dt))  # maximum time step
        
        self.timeList = np.arange(0, Tmax, dt)
        self.kinetics = dict()
        self.kinetics['randF_x'] = []
        self.kinetics['randF_y'] = []
        self.kinetics['wallRepulsiveF_x'] = []
        self.kinetics['wallRepulsiveF_y'] = []
        self.kinetics['upwindF_x'] = []
        self.kinetics['upwindF_y'] = []
        self.kinetics['totalF_x'] = []
        self.kinetics['totalF_y'] = []
        self.kinetics['velocity_x'] = []
        self.kinetics['velocity_y'] = []
        self.kinetics['acceleration_x'] = []
        self.kinetics['acceleration_y'] = []
        self.kinetics['totalF_x'] = []
        self.kinetics['totalF_y'] = []
        self.kinetics['position_x'] = []
        self.kinetics['position_y'] = []
        self.kinetics['times'] = []
        
        # generate random intial velocity condition    
        v0 = np.random.normal(0, self.v0_stdev, self.dim)
    
        ## insert initial position and velocity into positionList,veloList arrays
        self.kinetics['position_x'].append(r0[0])
        self.kinetics['position_y'].append(r0[1])
        self.kinetics['velocity_x'].append(v0[0])
        self.kinetics['velocity_y'].append(v0[1])
        
        self.fly(ts_max, detect_thresh, self.boundary, bounded)
        
        if plotting is True:
            plot_kwargs = {'title':"Individual trajectory", 'titleappend':''}
#            print self.dynamics.keys()
            plotting_funcs.plot_single_trajectory(self.dynamics, self.metadata, plot_kwargs)
        
        
    def fly(self, ts_max, detect_thresh, boundary, bounded):
        """Run the simulation using Euler's method"""
        ## loop through timesteps
        for ts in range(ts_max-1):
            
            # calculate drivers
            randF = baseline_driving_forces.random_force(self.rf)
            self.kinetics['randF_x'].append(randF[0])
            self.kinetics['randF_y'].append(randF[1])
            
            upwindF = baseline_driving_forces.upwindBiasForce(self.wtf)
            self.kinetics['upwindF_x'].append(upwindF[0])
            self.kinetics['upwindF_y'].append(upwindF[1])
            
            wallRepulsiveF = baseline_driving_forces.repulsionF([self.kinetics['position_x'][ts], self.kinetics['position_y'][ts]], self.wallF)
            self.kinetics['wallRepulsiveF_x'].append(wallRepulsiveF[0])
            self.kinetics['wallRepulsiveF_y'].append(wallRepulsiveF[1])
            
            # calculate current force
            totalF = -self.k*np.array([self.kinetics['position_x'][ts], self.kinetics['position_y'][ts]]) \
                      -self.beta*np.array([self.kinetics['velocity_x'][ts], self.kinetics['velocity_y'][ts]]) \
                      + randF + upwindF + wallRepulsiveF
            self.kinetics['totalF_x'].append(totalF[0])
            self.kinetics['totalF_y'].append(totalF[1])
            
            # calculate current acceleration
            accel = totalF/m
            self.kinetics['acceleration_x'].append(accel[0])
            self.kinetics['acceleration_y'].append(accel[1])
    
            # update velocity in next timestep
            velo_future = np.array([self.kinetics['velocity_x'][ts], self.kinetics['velocity_y'][ts]]) + accel*self.dt
            self.kinetics['velocity_x'].append(velo_future[0])
            self.kinetics['velocity_y'].append(velo_future[1])
            
            # update position in next timestep
            candidate_pos = np.array([self.kinetics['position_x'][ts], self.kinetics['position_y'][ts]]) + np.array(velo_future)*self.dt  # why not use veloList[ts]? -rd
#            print candidate_pos
            
            if bounded is True:  # if walls are enabled
                ## forbid mosquito from going out of bounds
                # check x dim
#                print candidate_pos[0]
                if candidate_pos[0] < boundary[0]:  # too far left
                    candidate_pos[0] = boundary[0] + 1e-4
                    if self.bounce is "crash":
                        self.kinetics['velocity_x'][ts+1] = 0.
#                        print "teleport! left wall"
                elif candidate_pos[0] > boundary[1]:  # reached upwind wall
                    candidate_pos[0] = boundary[1] - 1e-4
                    self.land(ts)  # end trajectory when reach end of tunnel
                    break  # stop flying at end  
                # check y dim
                if candidate_pos[1] > boundary[2]:  # too far up
                    candidate_pos[1] = boundary[2] + 1e-4
                    if self.bounce is "crash":
#                        print "teleport!"
                        self.kinetics['velocity_y'][ts+1] = 0.
#                        print "crash! top wall"
                elif candidate_pos[1] < boundary[3]:  # too far down
                    candidate_pos[1] = boundary[3] - 1e-4
                    if self.bounce is "crash":
                        self.kinetics['velocity_y'][ts+1] = 0.
#                        print "crash! bottom wall"
                
            self.kinetics['position_x'].append(candidate_pos[0])
            self.kinetics['position_y'].append(candidate_pos[1])
    
            # if there is a target, check if we are finding it
            if self.target_pos is None:
                self.metadata['target_found'][0]  = False
                self.metadata['time_to_target_find'][0] = np.nan
            elif norm(candidate_pos - self.target_pos) < self.detect_thresh:
                    self.metadata['target_found'][0]  = True
                    self.metadata['total_finds'] += 1
                    self.metadata['time_to_target_find'][0] = self.timeList[ts]  # should this be timeList[ts+1]? -rd
                    self.land(ts)
                    break  # stop flying at source
            elif ts == ts_max-1:  # ran out of time
                self.metadata['target_found'][0]  = False
                self.metadata['time_to_target_find'][0] = np.nan
                self.land(ts)
                break  # stop flying at source
                
                
                    
    def land(self, ts):    
        # trim excess timebins in arrays
        for key, value in self.kinetics.iteritems():
            self.dynamics[key] = pd.Series(value[:ts+1])
            


if __name__ == '__main__':
    # wallF params
    wallF_max=1e-7
    decay_const = 250
    
    # center repulsion params
    b = 4e-1  # determines shape
    shrink = 1e-6  # determines size/magnitude
    
    wallF = (b, shrink, wallF_max, decay_const)  #(4e-1, 1e-6, 1e-7, 250)
    
    mytraj = Trajectory(agent_pos="cage", target_pos="left", plotting = True, v0_stdev=0.01, wtf=7e-07, rf=4e-06, beta=1e-5, Tmax=15, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=wallF)  #, wallF=(80, 1e-4)