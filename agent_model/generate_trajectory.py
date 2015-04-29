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
import stim_biasF


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
    def __init__(self, agent_pos, v0_stdev, Tmax, dt, target_pos, beta, rf, wtf, stimF_str, detect_thresh, bounded, bounce, wallF, plotting=False, title="Individual trajectory", titleappend = '', k=0.):
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
        self.stimF_str = stimF_str
        
        # place heater
        self.target_pos = place_heater(target_pos)
        
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
        self.metadata['time_to_target_find'] = [np.nan] # TODO: make sure these lists are concating correctly

        ## initialize all arrays
        
        
#        ts_max = int(np.ceil(Tmax / dt))  # maximum time step
        
        self.dynamics = pd.DataFrame(columns=['times', 'position_x', 'position_y',\
        'velocity_x', 'velocity_y', 'acceleration_x', 'acceleration_y',\
        'totalF_x', 'totalF_y', 'randF_x', 'randF_y', 'wallRepulsiveF_x',
        'wallRepulsiveF_y', 'upwindF_x', 'upwindF_y', 'stimF_x', 'stimF_y', 'inPlume'])
        # insert time
        self.dynamics['times'] = np.arange(0, Tmax+dt, dt)
        self.times = np.arange(0,Tmax+dt,dt) 
        ########  Run simulation ##########
        # place agent
        r0 = place_agent(agent_pos)
        self.dim = len(r0)  # get dimension
        
        # generate random intial velocity condition
        v0 = np.random.normal(0, self.v0_stdev, self.dim)
        
        ## insert initial position and velocity into positionList,veloList arrays # TODO fix loc
        self.dynamics.loc[0, 'position_x'] = r0[0]
        self.dynamics.loc[0, 'position_y'] = r0[1]
        self.dynamics.loc[0, 'velocity_x'] = v0[0]
        self.dynamics.loc[0, 'velocity_y'] = v0[1]
        
        self.fly(self.dt, self.dynamics, detect_thresh, self.boundary, bounded)
        
        if plotting is True:
            plot_kwargs = {'title':"Individual trajectory", 'titleappend':''}
            plotting_funcs.plot_single_trajectory(self.dynamics, self.metadata, plot_kwargs)
#    def fly2(self):
#        dt = 0.01
#        for tsi,ts in enumerate(self.times[:10]):
#            #self.dynamics.loc[self.dynamics['times'] == ts+dt, 'position_y'] = candidate_pos[1]
#            tot = ts + dt
#            print self.dynamics['times'].iloc[tsi], (self.times.astype(float)==float(ts+0.01)).sum()
        
    def fly(self, dt, dataframe, detect_thresh, boundary, bounded):
        """Run the simulation using Euler's method
        """
        
                
        
        for tsi, row in self.dynamics.iterrows():
            ts = row['times']
            print ts
            
            # calculate drivers
            randF = baseline_driving_forces.random_force(self.rf)
            self.dynamics['randF_x'].iloc[tsi] = randF[0]
            self.dynamics['randF_y'].iloc[tsi] = randF[1]
            
            upwindF = baseline_driving_forces.upwindBiasForce(self.wtf)
            self.dynamics['upwindF_x'].iloc[tsi] = upwindF[0]
            self.dynamics['upwindF_y'].iloc[tsi] = upwindF[1]
            
            wallRepulsiveF = baseline_driving_forces.repulsionF(self.dynamics[['position_x', 'position_y']].iloc[tsi].values, self.wallF)
            self.dynamics['wallRepulsiveF_x'].iloc[tsi] = wallRepulsiveF[0]
            self.dynamics['wallRepulsiveF_y'].iloc[tsi] = wallRepulsiveF[1]
#            print "velo", self.dynamics.loc[self.dynamics.times == ts, ['velocity_x', 'velocity_y']].values
            
            if ts == 0.0:
                stimF, inPlume = stim_biasF.abs_plume(self.dynamics[['position_x', 'position_y']].iloc[tsi].values[0],\
                    self.stimF_str, False)
            else:
                stimF, inPlume = stim_biasF.abs_plume(self.dynamics[['position_x', 'position_y']].iloc[tsi].values[0],\
                    self.stimF_str, self.dynamics['inPlume'].iloc[tsi-1])
            self.dynamics['inPlume'].iloc[tsi] = inPlume
            self.dynamics['stimF_x'].iloc[tsi] = stimF[0]
            self.dynamics['stimF_y'].iloc[tsi] = stimF[1]
            
            
            # calculate current force
            #spring graveyard ==> # -self.k*np.array([self.dynamics.loc[self.dynamics.times == ts, 'position_x'],
            totalF = -self.beta*self.dynamics[['velocity_x','velocity_y']].iloc[tsi].values[0]\
                      + randF + upwindF + wallRepulsiveF + stimF
            self.dynamics['totalF_x'].iloc[tsi] = totalF[0]
            self.dynamics['totalF_y'].iloc[tsi] = totalF[1]
            
            # calculate current acceleration
            accel = totalF/m
            self.dynamics['acceleration_x'].iloc[tsi] = accel[0]
            self.dynamics['acceleration_y'].iloc[tsi] = accel[1]
    
            # update velocity in next timestep
            velo_future = self.dynamics[['velocity_x','velocity_y']].iloc[tsi].values[0] + accel*dt
            self.dynamics['velocity_x'].iloc[tsi+1] = velo_future[0]
            self.dynamics['velocity_y'].iloc[tsi+1] = velo_future[1]
            
            # create candidate position for next timestep # TODO check
            candidate_pos = self.dynamics[['position_x', 'position_y']].iloc[tsi].values\
                + self.dynamics[['velocity_x','velocity_y']].iloc[tsi].values*dt  # why not use veloList.loc[ts]? -rd
            
            if bounded is True:  # if walls are enabled
                ## forbid mosquito from going out of bounds
                # check x dim
                if candidate_pos[0] < boundary[0]:  # too far left
                    candidate_pos[0] = boundary[0] + 1e-4
                    if self.bounce is "crash":
                        self.dynamics['velocity_x'].iloc[tsi+1] = 0.
#                        print "teleport! left wall"
                elif candidate_pos[0] > boundary[1]:  # reached upwind wall
                    candidate_pos[0] = boundary[1] - 1e-4
                    break  # stop flying at end  
                # check y dim
                if candidate_pos[1] > boundary[2]:  # too far up
                    candidate_pos[1] = boundary[2] + 1e-4
                    if self.bounce is "crash":
#                        print "teleport!"
                        self.dynamics['velocity_y'].iloc[tsi+1] = 0.
#                        print "crash! top wall"
                elif candidate_pos[1] < boundary[3]:  # too far down
                    candidate_pos[1] = boundary[3] - 1e-4
                    if self.bounce is "crash":
                        self.dynamics['velocity_y'].iloc[tsi+1] = 0.
#                        print "crash! bottom wall"
            
            # assign candidate_pos to future position
            self.dynamics['position_x'].iloc[tsi+1] = candidate_pos[0]
            self.dynamics['position_y'].iloc[tsi+1] = candidate_pos[1]
            
#            print "\n"
#            print self.dynamics.loc[self.dynamics.times <= ts+dt]            
            
            # if there is a target, check if we are finding it
            if self.target_pos is None:
                self.metadata['target_found'][0]  = False
                self.metadata['time_to_target_find'][0] = np.nan
            elif norm(candidate_pos - self.target_pos) < self.detect_thresh:
                    self.metadata['target_found'][0]  = True
                    self.metadata['total_finds'] += 1
                    self.metadata['time_to_target_find'][0] = self.timeList.loc[ts]  # should this be timeList[i+1]? -rd # TODO fix .loc
                    break  # stop flying at source
            elif ts == self.Tmax-dt:  # ran out of time
                self.metadata['target_found'][0]  = False
                self.metadata['time_to_target_find'][0] = np.nan
                break  # stop flying at source
            


if __name__ == '__main__':
    # wallF params
    wallF_max=1e-7
    decay_const = 250
    
    # center repulsion params
    b = 4e-1  # determines shape
    shrink = 1e-6  # determines size/magnitude
    
    wallF = (b, shrink, wallF_max, decay_const)  #(4e-1, 1e-6, 1e-7, 250)
    
    
    mytraj = Trajectory(agent_pos="cage", target_pos="left", plotting = True, v0_stdev=0.01, wtf=7e-07, rf=4e-06, stimF_str=1e-4, beta=1e-5, Tmax=0.3, dt=0.01, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=wallF)