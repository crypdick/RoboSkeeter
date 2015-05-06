# -*- coding: utf-8 -*-
"""
creates agent object, with metadata, and methods such as fly, land.
when you use agent attribute fly, it generates trajectories

Created on Tue May  5 21:08:51 2015

@author: richard

agent:
-describe() [metadata]
-traj_gen iter(N=1)
    load_plume
    load trajectory object
    -fly()
        -temp sensing <--> plume.get_temp()
        -land --> mytrajectory.append(vectors)
-store agent as pickle
    http://www.rafekettler.com/magicmethods.html#callable
    
"""

import numpy as np
from numpy.linalg import norm
import pandas as pd
import plotting_funcs
import baseline_driving_forces
import stim_biasF
import plume
import trajectory


## define params
# population weight data: 2.88 +- 0.35mg
m = 3.0e-6 #2.88e-6  # mass (kg) =2.88 mg


def place_heater(target_pos):
    if target_pos is None:
            return None
    elif target_pos == "left":
        return [0.8651, -0.0507]
    elif target_pos == "right":
        return [0.8651, 0.0507]
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


class Agent():
    """Generate agent

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
        
    TODO: would it make more sense to make fly and land separate functions, since we never need to use
    those methods with the trajectory object in other scripts? Or maybe an agent class that takes
    a trajectory class?
    """
    def __init__(self, agent_pos, v0_stdev, Tmax, dt, target_pos, beta, rf, wtf, stimF_str, detect_thresh, bounded, bounce, wallF, k=0.):
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

    def fly(self, total_trajectories=1):
        trajectories.add_agent_info(self.metadata)
        
        traj_count = 0
        while traj_count < total_trajectories:
            try:
                array_dict = self._fly_single(self.dt, self.detect_thresh, self.boundary)
            except ValueError:  # throw out trajectories with impossible accels
                continue
            # extract trajectory object attribs, append to our lists.
            array_dict['trajectory_num'] = traj_count
            traj_count += 1
    #        trajectory.dynamics.set_index('trajectory', append=True, inplace=True)
            trajectories.append_ensemble(array_dict)  # TODO: check this is correct
            
        # concluding stats
        trajectories.add_agent_info({'total_trajectories': total_trajectories, 'time_target_find_avg': trajectory.T_find_stats(trajectories.agent_info['time_to_target_find'])})
    
        
        if __name__ == '__main__' and total_trajectories == 1:
            trajectories.plot_single_trajectory()

    
    def _fly_single(self, dt, detect_thresh, boundary, bounded=True):
        """Run the simulation using Euler's method
    
        First put everything into np arrays, then at end put it into a Pandas DF
        this speeds up the code thousands of times
        """
        tsi_max = int(np.ceil(self.metadata['time_max'] / dt))  # N bins/maximum time step
        _times = np.linspace(0, self.metadata['time_max'], tsi_max)
        _position_x = np.full(tsi_max, np.nan)
        _position_y = np.full(tsi_max, np.nan)
        _velocity_x = np.full(tsi_max, np.nan)
        _velocity_y = np.full(tsi_max, np.nan)
        _acceleration_x = np.full(tsi_max, np.nan)
        _acceleration_y = np.full(tsi_max, np.nan)
        _totalF_x = np.full(tsi_max, np.nan)
        _totalF_y = np.full(tsi_max, np.nan)
        _randF_x = np.full(tsi_max, np.nan)
        _randF_y = np.full(tsi_max, np.nan)
        _wallRepulsiveF_x = np.full(tsi_max, np.nan)
        _wallRepulsiveF_y = np.full(tsi_max, np.nan)
        _upwindF_x = np.full(tsi_max, np.nan)
        _upwindF_y = np.full(tsi_max, np.nan)
        _stimF_x = np.full(tsi_max, np.nan)
        _stimF_y = np.full(tsi_max, np.nan)
        _temperature = np.full(tsi_max, np.nan)
        _inPlume = np.full(tsi_max, np.nan)
        
        # place agent
        r0 = place_agent(self.metadata['initial_position'])
        self.dim = len(r0)  # get dimension
        
        # generate random intial velocity condition
        v0 = np.random.normal(0, self.v0_stdev, self.dim)
        
        ## insert initial position and velocity into positionList,veloList arrays
        _position_x[0] = r0[0]
        _position_y[0] = r0[1]
        _velocity_x[0] = v0[0]
        _velocity_y[0] = v0[1]
        
        arraydict = {'times': _times, 'position_x': _position_x, 'position_y': _position_y,\
        'velocity_x': _velocity_x, 'velocity_y': _velocity_y, 'acceleration_x': _acceleration_x,\
        'acceleration_y': _acceleration_y, 'totalF_x': _totalF_x, 'totalF_y': _totalF_y,\
        'randF_x': _randF_x, 'randF_y': _randF_y, 'wallRepulsiveF_x': _wallRepulsiveF_x,\
        'wallRepulsiveF_y': _wallRepulsiveF_y, 'upwindF_x': _upwindF_x, 'upwindF_y': _upwindF_y,\
        'stimF_x': _stimF_x, 'stimF_y': _stimF_y, 'temperature': _temperature,\
        'inPlume': _inPlume}
        
        for tsi in range(tsi_max):
            # sense the temperature
            _temperature[tsi] = myplume.temp_lookup([_position_x[tsi], _position_y[tsi]])
            
            # calculate drivers
            randF = baseline_driving_forces.random_force(self.rf)
            _randF_x[tsi] = randF[0]
            _randF_y[tsi] = randF[1]
            
            upwindF = baseline_driving_forces.upwindBiasForce(self.wtf)
            _upwindF_x[tsi] = upwindF[0]
            _upwindF_y[tsi] = upwindF[1]
            
            wallRepulsiveF = baseline_driving_forces.repulsionF(np.array([_position_x[tsi], _position_y[tsi]]), self.wallF)
            _wallRepulsiveF_x[tsi] = wallRepulsiveF[0]
            _wallRepulsiveF_y[tsi] = wallRepulsiveF[1]
            
#            # this may get updated if we find outselves crashing into the wall
#            _brakeF_x[tsi] = 0.
#            _brakeF_y[tsi] = 0.
            
            try:
                stimF, inPlume = stim_biasF.main(_temperature[tsi], _velocity_y[tsi],\
                    _inPlume[tsi-1], self.stimF_str)
            except UnboundLocalError:
                stimF, inPlume = stim_biasF.main(_temperature[tsi], _velocity_y[tsi],\
                    False, self.stimF_str)
            _inPlume[tsi] = inPlume
            _stimF_x[tsi] = stimF[0]
            _stimF_y[tsi] = stimF[1]
            
            # calculate current force
            #spring graveyard ==> # -self.k*np.array([self.dynamics.loc[self.dynamics.times == ts, 'position_x'],
            totalF = -self.beta*np.array([_velocity_x[tsi], _velocity_y[tsi]])\
                  + randF + upwindF + wallRepulsiveF# + stimF
            _totalF_x[tsi] = totalF[0]
            _totalF_y[tsi] = totalF[1]
            
            # calculate current acceleration
            accel = totalF/m
            _acceleration_x[tsi] = accel[0]
            _acceleration_y[tsi] = accel[1]
            if accel[1] > 50. or accel[0] > 50.:
                self.metadata['target_found'][0]  = False
                self.metadata['time_to_target_find'][0] = np.nan
                arraydict = self.land(tsi-1, arraydict)
                print "Throwing out trajectory, impossible acceleration!"
                raise ValueError('Impossible acceleration! ', accel[0], accel[1])
            
            # if time is out, end loop
            # TODO: check that landing behavior is right
            if tsi == tsi_max-1:
                self.metadata['target_found'][0]  = False
                self.metadata['time_to_target_find'][0] = np.nan
                arraydict = self.land(tsi, arraydict)
                break
            
            # update velocity in next timestep
            velo_future = np.array([_velocity_x[tsi], _velocity_y[tsi]]) + accel*dt
            _velocity_x[tsi+1] = velo_future[0]
            _velocity_y[tsi+1] = velo_future[1]
            
            # create candidate position for next timestep
            candidate_pos = np.array([_position_x[tsi], _position_y[tsi]]) \
                + np.array([_velocity_x[tsi], _velocity_y[tsi]])*dt  # why not use veloList.loc[ts]? -rd
            
            if bounded is True:  # if walls are enabled
                ## forbid mosquito from going out of bounds
                if candidate_pos[0] > boundary[1]:  # reached far (upwind) wall
                    self.metadata['target_found'][0]  = False
                    self.metadata['time_to_target_find'][0] = np.nan
                    arraydict = self.land(tsi-1, arraydict)  # stop flying at end, throw out last row
                    break
#                toggle, brakeF_x, brakeF_y = baseline_driving_forces.brakingF(candidate_pos, _totalF_x[tsi], _totalF_y[tsi], boundary)
#                outside_correct = toggle
#                if toggle is True:
#                    print tsi, "bounce~!"
#                    print outside_correct
            
                # assign candidate_pos to future position
                _position_x[tsi+1] = candidate_pos[0]
                _position_y[tsi+1] = candidate_pos[1]
            
            # if there is a target, check if we are finding it                
            if norm(candidate_pos - self.target_pos) < self.detect_thresh:
                    self.metadata['target_found'][0]  = True
                    self.metadata['total_finds'] += 1
                    self.metadata['time_to_target_find'][0] = _times[tsi]  # should this be timeList[i+1]? -rd # TODO fix .loc
                    arraydict = self.land(tsi, arraydict)  # stop flying at source
                    break
        
        return arraydict


    def land(self, tsi, arraydict):
        # trim excess timebins in arrays
        for key, array in arraydict.items():
            arraydict[key] = array[:tsi+1]
            
        return arraydict


if __name__ == '__main__':
    # wallF params
    wallF_max=9e-6#1e-7
    decay_const = 250
    
    # center repulsion params
    b = 4e-1  # determines shape
    shrink = 1e-6  # determines size/magnitude
    
    wallF = (b, shrink, wallF_max, decay_const)  #(4e-1, 1e-6, 1e-7, 250)
    
    # temperature plume
    myplume = plume.Plume()
    trajectories = trajectory.Trajectory()
    myagent = Agent(agent_pos="door", target_pos="left", v0_stdev=0.01, wtf=7e-07, rf=4e-06, stimF_str=1e-4, beta=1e-5, Tmax=15, dt=0.001, detect_thresh=0.023175, bounded=True, bounce="crash", wallF=wallF)
    myagent.fly(total_trajectories=4)
    
    plot_kwargs = {'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True}
    
#    trajectories.describe(plot_kwargs)
    trajectories.plot_single_trajectory()
    