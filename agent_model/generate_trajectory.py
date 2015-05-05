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
    if agent_pos == "door":  # start trajectories when they exit the front door
        return [0.1909, np.random.uniform(-0.0381, 0.0381)]
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

        """
        First put everything into np arrays, then at end put it into a Pandas DF
        this speeds up the code thousands of times
        """
        tsi_max = int(np.ceil(Tmax / dt))  # N bins/maximum time step
        self._times = np.linspace(0, Tmax, tsi_max)
        self._position_x = np.full(tsi_max, np.nan)
        self._position_y = np.full(tsi_max, np.nan)
        self._velocity_x = np.full(tsi_max, np.nan)
        self._velocity_y = np.full(tsi_max, np.nan)
        self._acceleration_x = np.full(tsi_max, np.nan)
        self._acceleration_y = np.full(tsi_max, np.nan)
        self._totalF_x = np.full(tsi_max, np.nan)
        self._totalF_y = np.full(tsi_max, np.nan)
        self._randF_x = np.full(tsi_max, np.nan)
        self._randF_y = np.full(tsi_max, np.nan)
        self._wallRepulsiveF_x = np.full(tsi_max, np.nan)
        self._wallRepulsiveF_y = np.full(tsi_max, np.nan)
        self._upwindF_x = np.full(tsi_max, np.nan)
        self._upwindF_y = np.full(tsi_max, np.nan)
        self._stimF_x = np.full(tsi_max, np.nan)
        self._stimF_y = np.full(tsi_max, np.nan)
        self._inPlume = np.full(tsi_max, np.nan)
        
        ########  Run simulation ##########
        # place agent
        r0 = place_agent(agent_pos)
        self.dim = len(r0)  # get dimension
        
        # generate random intial velocity condition
        v0 = np.random.normal(0, self.v0_stdev, self.dim)
        
        ## insert initial position and velocity into positionList,veloList arrays
        self._position_x[0] = r0[0]
        self._position_y[0] = r0[1]
        self._velocity_x[0] = v0[0]
        self._velocity_y[0] = v0[1]
        
        self.arraydict = {'times': self._times, 'position_x': self._position_x, 'position_y': self._position_y,\
        'velocity_x': self._velocity_x, 'velocity_y': self._velocity_y, 'acceleration_x': self._acceleration_x, 'acceleration_y': self._acceleration_y,\
        'totalF_x': self._totalF_x, 'totalF_y': self._totalF_y, 'randF_x': self._randF_x, 'randF_y': self._randF_y, 'wallRepulsiveF_x': self._wallRepulsiveF_x,
        'wallRepulsiveF_y': self._wallRepulsiveF_y, 'upwindF_x': self._upwindF_x, 'upwindF_y': self._upwindF_y, 'stimF_x': self._stimF_x, 'stimF_y': self._stimF_y, 'inPlume': self._inPlume}
        
        self.fly(self.dt, tsi_max, detect_thresh, self.boundary, bounded)        

        self.dynamics = pd.DataFrame(self.arraydict)
#        print self.dynamics        
        
        if plotting is True:
            plot_kwargs = {'title':"Individual agent trajectory", 'titleappend':''}
            plotting_funcs.plot_single_trajectory(self.dynamics, self.metadata, plot_kwargs)
#    def fly2(self):
#        dt = 0.01
#        for tsi,ts in enumerate(self.times[:10]):
#            #self.dynamics.loc[self.dynamics['times'] == ts+dt, 'position_y'] = candidate_pos[1]
#            tot = ts + dt
#            print self.dynamics['times'].iloc[tsi], (self.times.astype(float)==float(ts+0.01)).sum()
        
    def fly(self, dt, tsi_max, detect_thresh, boundary, bounded):
        """Run the simulation using Euler's method
        """
        
        
        for tsi in range(tsi_max):
#            print tsi
            # calculate drivers
            randF = baseline_driving_forces.random_force(self.rf)
            self._randF_x[tsi] = randF[0]
            self._randF_y[tsi] = randF[1]
            
            upwindF = baseline_driving_forces.upwindBiasForce(self.wtf)
            self._upwindF_x[tsi] = upwindF[0]
            self._upwindF_y[tsi] = upwindF[1]
            
            wallRepulsiveF = baseline_driving_forces.repulsionF(np.array([self._position_x[tsi], self._position_y[tsi]]), self.wallF)
            self._wallRepulsiveF_x[tsi] = wallRepulsiveF[0]
            self._wallRepulsiveF_y[tsi] = wallRepulsiveF[1]
#            print "velo", self.dynamics.loc[self.dynamics.times == ts, ['velocity_x', 'velocity_y']].values
            
#            # this may get updated if we find outselves crashing into the wall
#            self._brakeF_x[tsi] = 0.
#            self._brakeF_y[tsi] = 0.
            
            # assume that in the first timestep we are not in the plume
            if tsi == 0:
                stimF, inPlume = stim_biasF.abs_plume(np.array([self._position_x[tsi], self._position_y[tsi]]),\
                    self.stimF_str, False)
            else:
                stimF, inPlume = stim_biasF.abs_plume(np.array([self._position_x[tsi], self._position_y[tsi]]),\
                    self.stimF_str, self._inPlume[tsi-1])
            self._inPlume[tsi] = inPlume
            self._stimF_x[tsi] = stimF[0]
            self._stimF_y[tsi] = stimF[1]
            
            # calculate current force
            #spring graveyard ==> # -self.k*np.array([self.dynamics.loc[self.dynamics.times == ts, 'position_x'],
            totalF = -self.beta*np.array([self._velocity_x[tsi], self._velocity_y[tsi]])\
                  + randF + upwindF + wallRepulsiveF + stimF
            self._totalF_x[tsi] = totalF[0]
#                print outside_correct
#                print totalF
            self._totalF_y[tsi] = totalF[1]
            
            # calculate current acceleration
            accel = totalF/m
            self._acceleration_x[tsi] = accel[0]
            self._acceleration_y[tsi] = accel[1]
            
            # if time is out, end loop
            # TODO: check that landing behavior is right
            if tsi == tsi_max-1:
                self.metadata['target_found'][0]  = False
                self.metadata['time_to_target_find'][0] = np.nan
                self.land(tsi)
                break
            
            # update velocity in next timestep
            velo_future = np.array([self._velocity_x[tsi], self._velocity_y[tsi]]) + accel*dt
            self._velocity_x[tsi+1] = velo_future[0]
            self._velocity_y[tsi+1] = velo_future[1]
            
            # create candidate position for next timestep
            candidate_pos = np.array([self._position_x[tsi], self._position_y[tsi]]) \
                + np.array([self._velocity_x[tsi], self._velocity_y[tsi]])*dt  # why not use veloList.loc[ts]? -rd
            
            if bounded is True:  # if walls are enabled
                ## forbid mosquito from going out of bounds
                if candidate_pos[0] > boundary[1]:  # reached far (upwind) wall
                    self.metadata['target_found'][0]  = False
                    self.metadata['time_to_target_find'][0] = np.nan
                    self.land(tsi-1)  # stop flying at end, throw out last row
                    break
#                toggle, brakeF_x, brakeF_y = baseline_driving_forces.brakingF(candidate_pos, self._totalF_x[tsi], self._totalF_y[tsi], boundary)
#                outside_correct = toggle
#                if toggle is True:
#                    print tsi, "bounce~!"
#                    print outside_correct
            
                # assign candidate_pos to future position
                self._position_x[tsi+1] = candidate_pos[0]
                self._position_y[tsi+1] = candidate_pos[1]
            
#            print "\n"
#            print self.dynamics.loc[self.dynamics.times <= ts+dt]            
            
            # if there is a target, check if we are finding it                
            if norm(candidate_pos - self.target_pos) < self.detect_thresh:
                    self.metadata['target_found'][0]  = True
                    self.metadata['total_finds'] += 1
                    self.metadata['time_to_target_find'][0] = self._times[tsi]  # should this be timeList[i+1]? -rd # TODO fix .loc
                    self.land(tsi)  # stop flying at source
                    break

    def land(self, tsi):
        # trim excess timebins in arrays
        for key, array in self.arraydict.items():
            self.arraydict[key] = array[:tsi+1]
            


if __name__ == '__main__':
    # wallF params
    wallF_max=9e-6#1e-7
    decay_const = 250
    
    # center repulsion params
    b = 4e-1  # determines shape
    shrink = 1e-6  # determines size/magnitude
    
    wallF = (b, shrink, wallF_max, decay_const)  #(4e-1, 1e-6, 1e-7, 250)
    
    
    mytraj = Trajectory(agent_pos="door", target_pos="left", plotting = True, v0_stdev=0.01, wtf=7e-07, rf=4e-06, stimF_str=1e-4, beta=1e-5, Tmax=10, dt=0.001, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=wallF)