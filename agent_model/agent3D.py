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
import baseline_driving_forces3D
import stim_biasF3D
import plume3D
import trajectory3D
import repulsion_landscape3D


def place_heater(target_pos):
    ''' puts a heater in the correct position in the wind tunnel
    '''
    zmin = 0.03800
    zmax = 0.11340
    diam = 0.01905
    if target_pos is None:
            return None
    elif target_pos == "left":
        return [0.8651, -0.0507, zmin, zmax, diam]
    elif target_pos == "right":
        return [0.8651, 0.0507, zmin, zmax, diam]
    elif type(target_pos) is list:
        return target_pos
    else:
        raise Exception('invalid heater type specified')


def place_agent(agent_pos):
    ''' puts the agent in an initial position, usually within the bounds of the
    cage
    '''
    if type(agent_pos) is list:
        return agent_pos
    if agent_pos == "center":
        return [0.1524, 0., 0.]  # center of box
        # FIXME cage height
    if agent_pos == "cage":  # bounds of cage
        return [np.random.uniform(0.1143, 0.1909), np.random.uniform(-0.0381, 0.0381), np.random.uniform(0., 0.1016)]
        # FIXME cage height
    if agent_pos == "door":  # start trajectories when they exit the front door
        return [0.1909, np.random.uniform(-0.0381, 0.0381), np.random.uniform(0., 0.1016)]
        # FIXME cage height
    else:
        raise Exception('invalid agent position specified')


class Agent():
    """Generate agent

    Args:
        Trajectory_object: (trajectory object)
            
        Plume_object: (plume object)
            
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
            damping force (kg/s). Undamped = 0, Critical damping = 1,
            normal range ~1e-5. NOTE: if beta is too big, things blow up
        rf: (float)
            random driving force exp term for exp distribution 
        wtf: (float)
            upwind bias force magnitude  # TODO units
        detect_thresh: (float)
            distance mozzie can detect target in (m), 2 cm + radius of heaters,
            (0.00635m/2= 0.003175)
        boundary: (array)
            specify where walls are  (minx, maxx, miny, maxy)
        

    All args are in SI units and based on behavioral data:
    Tmax, dt: seconds (data: <control flight duration> = 4.4131 +- 4.4096
    

    Returns:
        Agent object

    """
    def __init__(self, Trajectory_object, Plume_object, agent_pos='door', v0_stdev=0.01, Tmax=15, dt=0.01,\
        target_pos='left', beta=1e-5, rf=4e-06, wtf=7e-07, stimF_str=1e-4, \
        detect_thresh=0.023175, bounded=True, \
        wallF_params=(4e-1, 1e-6, 1e-7, 250, "walls_only"), k=0.):
        """ Initialize object with instant variables, and trigger other funcs. 
        """
        
        self.boundary = [0.0, 1.0, 0.127, -0.127, 0., 0.254]  # these are real dims of our wind tunnel
        self.Tmax = Tmax
        self.dt = dt
        self.v0_stdev = v0_stdev
        self.k = k
        self.beta = beta
        self.rf = rf
        self.wtf = wtf
        self.detect_thresh = detect_thresh     
        self.wallF_params = wallF_params
        self.stimF_str = stimF_str
        
        
        # place heater
        self.target_pos = place_heater(target_pos)
        
        self.target_found = False
        self.t_targfound = np.nan
        
        self.metadata = dict()
        # population weight data: 2.88 +- 0.35mg
        self.metadata['mass'] = 3.0e-6 # 2.88e-6  # mass (kg) =2.88 mg
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
        self.metadata['wallF_params'] = wallF_params
        # for stats, later
        self.metadata['time_target_find_avg'] = []
        self.metadata['total_finds'] = 0
        self.metadata['target_found'] = [False]
        self.metadata['time_to_target_find'] = [np.nan] # TODO: make sure these lists are concating correctly
        
        self.trajectory_obj = Trajectory_object
        self.Plume_object = Plume_object
        
        # create repulsion landscape
        self._repulsion_funcs = repulsion_landscape3D.landscape(normed=True)


    def fly(self, total_trajectories=1):
        ''' iterates _fly_single total_trajectories times
        '''
        traj_count = 0
        while traj_count < total_trajectories:
            try:
                array_dict = self._fly_single(self.dt, self.metadata['mass'], self.detect_thresh, self.boundary)
            except ValueError:  # throw out trajectories with impossible accels
                continue
            # extract trajectory object attribs, append to our lists.
            array_dict['trajectory_num'] = traj_count
            traj_count += 1
    #        trajectory.dynamics.set_index('trajectory', append=True, inplace=True)
            self.trajectory_obj.append_ensemble(array_dict)
        
        
        self.metadata['total_trajectories'] = total_trajectories
        self.trajectory_obj.add_agent_info(self.metadata)
        # concluding stats
        self.trajectory_obj.add_agent_info({'time_target_find_avg': trajectory3D.T_find_stats(self.trajectory_obj.agent_info['time_to_target_find'])})
    
        
#        if __name__ == '__main__' and total_trajectories == 1:
#            self.trajectory_obj.plot_single_trajectory()

    
    def _fly_single(self, dt, m, detect_thresh, boundary, bounded=True):
        """Run the simulation using Euler's method
    
        First put everything into np arrays, then at end put it into a Pandas DF
        this speeds up the code thousands of times
        """
        BOUNCE = "elastic"        
        
        # initialize np arrays
        tsi_max = int(np.ceil(self.metadata['time_max'] / dt))  # N bins/maximum time step
        _times = np.linspace(0, self.metadata['time_max'], tsi_max)
        _position_x = np.full(tsi_max, np.nan)
        _position_y = np.full(tsi_max, np.nan)
        _position_z = np.full(tsi_max, np.nan)
        _velocity_x = np.full(tsi_max, np.nan)
        _velocity_y = np.full(tsi_max, np.nan)
        _velocity_z = np.full(tsi_max, np.nan)
        _acceleration_x = np.full(tsi_max, np.nan)
        _acceleration_y = np.full(tsi_max, np.nan)
        _acceleration_z = np.full(tsi_max, np.nan)
        _totalF_x = np.full(tsi_max, np.nan)
        _totalF_y = np.full(tsi_max, np.nan)
        _totalF_z = np.full(tsi_max, np.nan)
        _randF_x = np.full(tsi_max, np.nan)
        _randF_y = np.full(tsi_max, np.nan)
        _randF_z = np.full(tsi_max, np.nan)
        _wallRepulsiveF_x = np.full(tsi_max, np.nan)
        _wallRepulsiveF_y = np.full(tsi_max, np.nan)
        _wallRepulsiveF_z = np.full(tsi_max, np.nan)
        _upwindF_x = np.full(tsi_max, np.nan)
        _upwindF_y = np.full(tsi_max, np.nan)
        _upwindF_z = np.full(tsi_max, np.nan)
        _stimF_x = np.full(tsi_max, np.nan)
        _stimF_y = np.full(tsi_max, np.nan)
        _stimF_z = np.full(tsi_max, np.nan)        
        _temperature = np.full(tsi_max, np.nan)
        _inPlume = np.full(tsi_max, np.nan)
        
        # place agent
        r0 = place_agent(self.metadata['initial_position'])
        self.dim = len(r0)  # get dimension
        _position_x[0], _position_y[0], _position_z[0] = r0 # store initial position
        
        # generate random intial velocity condition
        v0 = np.random.normal(0, self.v0_stdev, self.dim)
        
        
        # make dictionary for creating Pandas df, later
        arraydict = {'times': _times, 'position_x': _position_x, 'position_y': _position_y,\
        'position_z': _position_z,\
        'velocity_x': _velocity_x, 'velocity_y': _velocity_y, 'velocity_z': _velocity_z,\
        'acceleration_x': _acceleration_x, 'acceleration_y': _acceleration_y, 'acceleration_z': _acceleration_z,\
        'totalF_x': _totalF_x, 'totalF_y': _totalF_y, 'totalF_z': _totalF_z,\
        'randF_x': _randF_x, 'randF_y': _randF_y, 'randF_z': _randF_z, 'wallRepulsiveF_x': _wallRepulsiveF_x,\
        'wallRepulsiveF_y': _wallRepulsiveF_y, 'wallRepulsiveF_z': _wallRepulsiveF_z,\
        'upwindF_x': _upwindF_x, 'upwindF_y': _upwindF_y, 'upwindF_z': _upwindF_z,\
        'stimF_x': _stimF_x, 'stimF_y': _stimF_y, 'stimF_z': _stimF_z, 'temperature': _temperature,\
        'inPlume': _inPlume}
        
        # generate a flight!
        for tsi in xrange(tsi_max):
            # sense the temperature
            _temperature[tsi] = self.Plume_object.temp_lookup([_position_x[tsi], _position_y[tsi]])
            
            # calculate driving forces
            randF = baseline_driving_forces3D.random_force(self.rf, dim=self.dim)
            _randF_x[tsi], _randF_y[tsi], _randF_z[tsi] = randF
            
            upwindF = baseline_driving_forces3D.upwindBiasForce(self.wtf, dim=self.dim)
            _upwindF_x[tsi], _upwindF_y[tsi], _upwindF_z[tsi] = upwindF

            
            wallRepulsiveF = baseline_driving_forces3D.repulsionF(\
                np.array([_position_x[tsi], _position_y[tsi], _position_z[tsi]]),\
                self._repulsion_funcs, self.wallF_params)
            _wallRepulsiveF_x[tsi], _wallRepulsiveF_y[tsi], _wallRepulsiveF_z[tsi] = wallRepulsiveF
            
            if tsi == 0:
                stimF, inPlume = stim_biasF3D.main(_temperature[tsi], v0[1],\
                    _inPlume[tsi-1], self.stimF_str)
            else:
                try:
                    stimF, inPlume = stim_biasF3D.main(_temperature[tsi], _velocity_y[tsi],\
                        _inPlume[tsi-1], self.stimF_str)
                except UnboundLocalError: # TODO: wtf?
                    stimF, inPlume = stim_biasF3D.main(_temperature[tsi], _velocity_y[tsi],\
                        False, self.stimF_str)
                    print "erorr"
            _inPlume[tsi] = inPlume
            _stimF_x[tsi], _stimF_y[tsi], _stimF_z[tsi] = stimF
            
            # calculate current force
            if tsi == 0:
                totalF = -self.beta*v0 + randF + upwindF + wallRepulsiveF# + stimF # FIXME
            else:
                totalF = -self.beta*np.array([_velocity_x[tsi-1], _velocity_y[tsi-1], _velocity_z[tsi-1]])\
                  + randF + upwindF + wallRepulsiveF# + stimF # FIXME
            _totalF_x[tsi], _totalF_y[tsi], _totalF_z[tsi] = totalF
            
            # calculate current acceleration
            accel = totalF / m
            _acceleration_x[tsi], _acceleration_y[tsi], _acceleration_z[tsi] = accel
            
            # calculate velocity
            if tsi == 0:
                _velocity_x[tsi], _velocity_y[tsi], _velocity_z[tsi] =\
                v0 + accel*dt
            else:
                _velocity_x[tsi], _velocity_y[tsi], _velocity_z[tsi] =\
                np.array([_velocity_x[tsi-1], _velocity_y[tsi-1], _velocity_z[tsi-1]]) + accel*dt

            
            # if time is out, end loop before we solve for future position
            if tsi == tsi_max-1:
                self.metadata['target_found'][0]  = False
                self.metadata['time_to_target_find'][0] = np.nan
                arraydict = self.land(tsi, arraydict)
                break            
            
            # solve candidate position for next timestep
            candidate_pos = np.array([_position_x[tsi], _position_y[tsi], _position_z[tsi]]) \
                + np.array([_velocity_x[tsi], _velocity_y[tsi], _velocity_z[tsi]])*dt
#            
            # if walls are enabled, forbid mosquito from going out of bounds
            if bounded is True:  
                # x dim
                if candidate_pos[0] > boundary[1]:  # reached far (upwind) wall (end)
                    self.metadata['target_found'][0]  = False
                    self.metadata['time_to_target_find'][0] = np.nan
                    arraydict = self.land(tsi-1, arraydict)  # stop flying at end, throw out last row
                    break
                if candidate_pos[0] < boundary[0]:  # too far left
                    print "too far left"
                    candidate_pos[0] = boundary[0] + 1e-4
                    if BOUNCE == 'elastic':
                        _velocity_x[tsi+1] = _velocity_x[tsi+1] * -1
                        print "boom! left"
                    elif BOUNCE == 'crash':
                        _velocity_x[tsi+1] = 0.
            
                #y dim
                if candidate_pos[1] > boundary[2]:  # too left
                    print "too left"
                    candidate_pos[1] = boundary[2] + 1e-4 # note, left is going more negative in our convention
                    if BOUNCE == 'elastic':
                        _velocity_y[tsi+1] = _velocity_y[tsi+1] * -1
                    elif BOUNCE == "crash":
#                        print "teleport!"
                        _velocity_y[tsi+1] = 0.
#                        print "crash! top wall"
                if candidate_pos[1] < boundary[3]:  # too far right
                    print "too far right"
                    candidate_pos[1] = boundary[3] - 1e-4
                    if BOUNCE == 'elastic':
                        _velocity_y[tsi+1] = _velocity_y[tsi+1] * -1
                    elif BOUNCE == 'crash':
                        _velocity_y[tsi+1] = 0.
                
                # z dim
                if candidate_pos[2] > boundary[5]:  # too far above
                    print "too far above"
                    candidate_pos[2] = boundary[5] - 1e-4
                    if BOUNCE == 'elastic':
                        _velocity_z[tsi+1] = _velocity_z[tsi+1] * -1
                        print "boom! top"
                    elif BOUNCE == "crash":
#                        print "teleport!"
                        _velocity_z[tsi+1] = 0.
#                        print "crash! top wall"
                if candidate_pos[2] < boundary[4]:  # too far below
                    print "too far below"
                    candidate_pos[2] = boundary[4] + 1e-4
                    if BOUNCE == 'elastic':
                        _velocity_z[tsi+1] = _velocity_z[tsi+1] * -1
                    elif BOUNCE == 'crash':
                        _velocity_z[tsi+1] = 0.
                        
                # save screened candidate_pos to future position
            _position_x[tsi+1], _position_y[tsi+1], _position_z[tsi+1] = candidate_pos
            
#            # test the kinematics ex-post facto
#            real_velo = (candidate_pos - np.array([_position_x[tsi], _position_y[tsi], _position_z[tsi]])) / dt
#            _velocity_x[tsi], _velocity_y[tsi], _velocity_z[tsi] = real_velo
#            
#            if tsi == 0:
#                real_accel = (real_velo - v0) / dt
#            else:
#                real_accel = (real_velo - np.array([_velocity_x[tsi-1], _velocity_y[tsi-1], _velocity_z[tsi-1]])) / dt
#            _acceleration_x[tsi], _acceleration_y[tsi], _acceleration_z[tsi] = real_accel
            
            # if there is a target, check if we are finding it                
            if norm(candidate_pos - self.target_pos[0:3]) < self.detect_thresh:
                    self.metadata['target_found'][0]  = True
                    self.metadata['total_finds'] += 1
                    self.metadata['time_to_target_find'][0] = _times[tsi]  # should this be timeList[i+1]? -rd # TODO fix .loc
                    arraydict = self.land(tsi, arraydict)  # stop flying at source
                    print "source found"
                    break
        
        return arraydict


    def land(self, tsi, arraydict):
        ''' trim excess timebins in arrays
        '''
        for key, array in arraydict.items():
            arraydict[key] = array[:tsi+1]
            
        return arraydict
        
        
    def show_landscape(self):
        import repulsion_landscape
        repulsion_landscape.main(self.wallF_params, None, plotting=True)


if __name__ == '__main__':
    # wallF params
    scalar = 1e-7
    
    wallF_params = [scalar]  #(4e-1, 1e-6, 1e-7, 250)

    # temperature plume
    myplume = plume3D.Plume()
    trajectories = trajectory3D.Trajectory() # instantiate empty trajectories object
    myagent = Agent(trajectories, myplume, agent_pos="door", target_pos="left", v0_stdev=0.01, wtf=7e-07,\
        rf=4e-05, stimF_str=1e-4, beta=1e-5, Tmax=15., dt=0.01, detect_thresh=0.023175, \
        bounded=True, wallF_params=wallF_params)
    myagent.fly(total_trajectories=1)
    
   
#    trajectories.describe(plot_kwargs = {'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True, 'force_violin':True})
    trajectories.plot_single_3Dtrajectory()
    