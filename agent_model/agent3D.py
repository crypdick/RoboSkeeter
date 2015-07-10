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
   
   
GOALS
#
Plotting
#
#####GOAL: Have same figures for both agent/ model

#
model building
#
!implement repulsion functions for x, y, z dimensions
    polyfit each distribution? -- waiting for sharri
Kalman filter
upwind downwind decision policy?

plume triggered stats:
if in-> out, grab 1s
check velocity_y component. sort left, right.
plot compass plots


#
monte carlo fitting
#
!fit "prior" kinematics-- waiting for sharri
!make cost function
!run monte carlo for each kinematic to fit params of model to the kinematic fit

#
EVERYTHING PLUME
#
! add plume interaction forces (decision policies)


#
Analysis
#
--lowest--
autocorrelation, PACF of the heading angle
make color coding 3D trajectories
do plume trigger analysis 
turn ratio
downwind bouts --
"""

import numpy as np
from numpy.linalg import norm
from math import atan2
import baseline_driving_forces3D
import stim_biasF3D
import plume3D
import trajectory3D
import repulsion_landscape3D


def place_heater(target_pos):
    ''' puts a heater in the correct position in the wind tunnel
    
    returns [x,y, zmin, zmax, diam]
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
    
    
def score_output(desired_output, output):
    """take the root mean square error of two kinematic distributions    
    """
    pass

def solve_heading(velo_x, velo_y):
    theta = atan2(velo_y, velo_x)
    return theta*180/np.pi


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
        biasF_scale: (float)
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
        target_pos='left', beta=1e-5, biasF_scale=4e-06, wtf=7e-07, stimF_str=1e-4, \
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
        self.biasF_scale = biasF_scale
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
        self.metadata['biasF_scale'] = biasF_scale
        self.metadata['wtf'] = wtf
        self.metadata['wallF_params'] = wallF_params
        # for stats, later
        self.metadata['time_target_find_avg'] = []
        self.metadata['total_finds'] = 0
#        self.metadata['target_found'] = [False]
#        self.metadata['time_to_target_find'] = [np.nan] # TODO: make sure these lists are concating correctly
        
        self.metadata['kinematic_vals'] = ['velocity', 'acceleration']
        self.metadata['forces'] = ['totalF', 'biasF', 'wallRepulsiveF', 'upwindF', 'stimF']
        
        
        # turn thresh, in units deg s-1.
        # From Sharri:
        # it is the stdevof the broader of two Gaussians that fit the distribution of angular velocity
        self.metadata['turn_threshold'] = 433.5         
        
        self.trajectory_obj = Trajectory_object
        self.Plume_object = Plume_object
        
        # create repulsion landscape
        self._repulsion_funcs = repulsion_landscape3D.landscape(normed=True)


    def fly(self, total_trajectories=1):
        ''' iterates _fly_single total_trajectories times
        '''
        traj_count = 0
        while traj_count < total_trajectories:
            vectors_object = self._fly_single(self.dt, self.metadata['mass'], self.detect_thresh, self.boundary)
            # extract trajectory object attribs, append to our lists.
            value = setattr(vectors_object, 'trajectory_num', traj_count)
#            trajectory.dynamics.set_index('trajectory', append=True, inplace=True)
#            self._calc_polar_kinematics(vectors_object)
            self.trajectory_obj.append_ensemble(vars(vectors_object))
            
            traj_count += 1
        
        
        self.metadata['total_trajectories'] = total_trajectories
        self.trajectory_obj.add_agent_info(self.metadata)
        # concluding stats
#        self.trajectory_obj.add_agent_info({'time_target_find_avg': trajectory3D.T_find_stats(self.trajectory_obj.agent_info['time_to_target_find'])})
    
        
#        if __name__ == '__main__' and total_trajectories == 1:
#            self.trajectory_obj.plot_single_trajectory()

    
    def _fly_single(self, dt, m, detect_thresh, boundary, bounded=True):
        """Run the simulation using Euler's method
    
        First put everything into np arrays, then at end put it into a Pandas DF
        this speeds up the code thousands of times
        """
        BOUNCE = "elastic"        
        tsi_max = int(np.ceil(self.metadata['time_max'] / dt))  # N bins/maximum time step
        
        # V for vector
        # retrieve dict w/ vars(V)
        V = Object()
        
        for name in self.metadata['kinematic_vals']+self.metadata['forces']:
            # adds attribute to V
            for ext in ['_x', '_y', '_z', '_xy_theta', '_xy_mag']:
                value = setattr(V, name+ext, np.full(tsi_max, np.nan))
        
        
        
        # initialize np arrays
        
        V.times = np.linspace(0, self.metadata['time_max'], tsi_max)
        V.position_x = np.full(tsi_max, np.nan)
        V.position_y = np.full(tsi_max, np.nan)
        V.position_z = np.full(tsi_max, np.nan)
#        _post_velocity_x = np.full(tsi_max, np.nan)
#        _temperature = np.full(tsi_max, np.nan) # depreciated
        V.inPlume = np.full(tsi_max, np.nan)
        V.plume_experience = [np.nan]*tsi_max
        V.turning = [np.nan]*tsi_max
        V.heading_angle = np.full(tsi_max, np.nan)
        V.velocity_angular = np.full(tsi_max, np.nan)
        
        # place agent
        r0 = place_agent(self.metadata['initial_position'])
        self.dim = len(r0)  # get dimension
        V.position_x[0], V.position_y[0], V.position_z[0] = r0 # store initial position
        
        # generate random intial velocity condition
        v0 = np.random.normal(0, self.v0_stdev, self.dim)       

        # generate a flight!
        for tsi in xrange(tsi_max):
            # sense the temperature
#            _temperature[tsi] = self.Plume_object.temp_lookup([_position_x[tsi], _position_y[tsi]]) # depreciated
            V.inPlume[tsi] = self.Plume_object.check_for_plume([V.position_x[tsi], V.position_y[tsi], V.position_z[tsi]])
            if tsi == 0:
                V.plume_experience[tsi] = 'searching'
            else:
                V.plume_experience[tsi] = self._check_crossing_state(tsi, V.inPlume)
            
            # calculate driving forces
            biasF = baseline_driving_forces3D.bias_force(self.biasF_scale)
            V.biasF_x[tsi], V.biasF_y[tsi], V.biasF_z[tsi] = biasF
            
            upwindF = baseline_driving_forces3D.upwindBiasForce(self.wtf)
            V.upwindF_x[tsi], V.upwindF_y[tsi], V.upwindF_z[tsi] = upwindF

            
            wallRepulsiveF = baseline_driving_forces3D.repulsionF(\
                np.array([V.position_x[tsi], V.position_y[tsi], V.position_z[tsi]]),\
                self._repulsion_funcs, self.wallF_params)
            V.wallRepulsiveF_x[tsi], V.wallRepulsiveF_y[tsi], V.wallRepulsiveF_z[tsi] = wallRepulsiveF
            
#            if tsi == 0:
#                stimF, inPlume = stim_biasF3D.main(_temperature[tsi], v0[1],\
#                    V.inPlume[tsi-1], self.stimF_str)
#            else:
#                try:
#                    stimF, inPlume = stim_biasF3D.main(_temperature[tsi], V.velocity_y[tsi],\
#                        V.inPlume[tsi-1], self.stimF_str)
#                except UnboundLocalError: # TODO: wtf?
#                    stimF, inPlume = stim_biasF3D.main(_temperature[tsi], V.velocity_y[tsi],\
#                        False, self.stimF_str)
#                    print "erorr"
#                else:
#                    print 
#            _stimF_x[tsi], _stimF_y[tsi], _stimF_z[tsi] = stimF
            
            # calculate current force
            if tsi == 0:
                totalF = -self.beta*v0 + biasF + upwindF + wallRepulsiveF# + stimF # FIXME
            else:
                totalF = -self.beta*np.array([V.velocity_x[tsi-1], V.velocity_y[tsi-1], V.velocity_z[tsi-1]])\
                  + biasF + upwindF + wallRepulsiveF# + stimF # FIXME
            V.totalF_x[tsi], V.totalF_y[tsi], V.totalF_z[tsi] = totalF
            
            # calculate current acceleration
            accel = totalF / m
            V.acceleration_x[tsi], V.acceleration_y[tsi], V.acceleration_z[tsi] = accel
            
            # calculate velocity
            if tsi == 0:
                V.velocity_x[tsi], V.velocity_y[tsi], V.velocity_z[tsi] =\
                v0 + accel*dt
            else:
                V.velocity_x[tsi], V.velocity_y[tsi], V.velocity_z[tsi] =\
                np.array([V.velocity_x[tsi-1], V.velocity_y[tsi-1], V.velocity_z[tsi-1]]) + accel*dt


            
            # if time is out, end loop before we solve for future position
            if tsi == tsi_max-1:
#                self.metadata['target_found'][0]  = False
#                self.metadata['time_to_target_find'][0] = np.nan
                V = self.land(tsi, V)
                break            
            
            # solve candidate position for next timestep
            candidate_pos = np.array([V.position_x[tsi], V.position_y[tsi], V.position_z[tsi]]) \
                + np.array([V.velocity_x[tsi], V.velocity_y[tsi], V.velocity_z[tsi]])*dt
#            
            # if walls are enabled, forbid mosquito from going out of bounds
            if bounded is True:  
                # x dim
                if candidate_pos[0] > boundary[1]:  # reached far (upwind) wall (end)
#                    self.metadata['target_found'][0]  = False
#                    self.metadata['time_to_target_find'][0] = np.nan
                    V = self.land(tsi-1, V)  # stop flying at end, throw out last row
                    break
                if candidate_pos[0] < boundary[0]:  # too far left
                    print "too far left"
                    candidate_pos[0] = boundary[0] + 1e-4
                    if BOUNCE == 'elastic':
                        V.velocity_x[tsi+1] = V.velocity_x[tsi+1] * -1
                        print "boom! left"
                    elif BOUNCE == 'crash':
                        V.velocity_x[tsi+1] = 0.
            
                #y dim
                if candidate_pos[1] > boundary[2]:  # too left
                    print "too left"
                    candidate_pos[1] = boundary[2] + 1e-4 # note, left is going more negative in our convention
                    if BOUNCE == 'elastic':
                        V.velocity_y[tsi+1] = V.velocity_y[tsi+1] * -1
                    elif BOUNCE == "crash":
#                        print "teleport!"
                        V.velocity_y[tsi+1] = 0.
#                        print "crash! top wall"
                if candidate_pos[1] < boundary[3]:  # too far right
                    print "too far right"
                    candidate_pos[1] = boundary[3] - 1e-4
                    if BOUNCE == 'elastic':
                        V.velocity_y[tsi+1] = V.velocity_y[tsi+1] * -1
                    elif BOUNCE == 'crash':
                        V.velocity_y[tsi+1] = 0.
                
                # z dim
                if candidate_pos[2] > boundary[5]:  # too far above
                    print "too far above"
                    candidate_pos[2] = boundary[5] - 1e-4
                    if BOUNCE == 'elastic':
                        V.velocity_z[tsi+1] = V.velocity_z[tsi+1] * -1
                        print "boom! top"
                    elif BOUNCE == "crash":
#                        print "teleport!"
                        V.velocity_z[tsi+1] = 0.
#                        print "crash! top wall"
                if candidate_pos[2] < boundary[4]:  # too far below
                    print "too far below"
                    candidate_pos[2] = boundary[4] + 1e-4
                    if BOUNCE == 'elastic':
                        V.velocity_z[tsi+1] = V.velocity_z[tsi+1] * -1
                    elif BOUNCE == 'crash':
                        V.velocity_z[tsi+1] = 0.
                        
                # save screened candidate_pos to future position
            V.position_x[tsi+1], V.position_y[tsi+1], V.position_z[tsi+1] = candidate_pos
            
            # solve final heading
            V.heading_angle[tsi] = solve_heading(V.velocity_y[tsi], V.velocity_x[tsi])
            # ... and angular velo
            if tsi == 0:
                V.velocity_angular[tsi] = 0
            else:
                V.velocity_angular[tsi] = (V.heading_angle[tsi] - V.heading_angle[tsi-1]) / dt
                
            # turning state
            if tsi in [0, 1]:
                V.turning[tsi] = 0
            else:
                turn_min = 3
                if abs(V.velocity_angular[tsi-turn_min:tsi]).sum() > self.metadata['turn_threshold']*turn_min:
                    V.turning[tsi] = 1
                else:
                    V.turning[tsi] = 0
            
            
##            # test the kinematics ex-post facto
#            real_velo = (candidate_pos - np.array([V.position_x[tsi], V.position_y[tsi], V.position_z[tsi]])) / dt
#            _postV.velocity_x[tsi] = real_velo[0]
#            V.velocity_x[tsi], V.velocity_y[tsi], V.velocity_z[tsi] = real_velo
#            
#            if tsi == 0:
#                real_accel = (real_velo - v0) / dt
#            else:
#                real_accel = (real_velo - np.array([V.velocity_x[tsi-1], V.velocity_y[tsi-1], V.velocity_z[tsi-1]])) / dt
#            V.acceleration_x[tsi], V.acceleration_y[tsi], V.acceleration_z[tsi] = real_accel
            
            # if there is a target, check if we are finding it                
#            if norm(candidate_pos - self.target_pos[0:3]) < self.detect_thresh:
#                    self.metadata['target_found'][0]  = True
#                    self.metadata['total_finds'] += 1
#                    self.metadata['time_to_target_find'][0] = V.times[tsi]  # should this be timeList[i+1]?
#                    V = self.land(tsi, V)  # stop flying at source
#                    print "source found"
#                    break
        
        return V


    def land(self, tsi, V):
        ''' trim excess timebins in arrays
        '''
        for key, array in vars(V).items():
            value = setattr(V, key, array[:tsi+1])
            
        V = self._calc_polar_kinematics(V)
        
        return V
        
        
    def show_landscape(self):
        import repulsion_landscape
        repulsion_landscape.main(self.wallF_params, None, plotting=True)
        
    def _check_crossing_state(self, tsi, inPlume):
        """
        out2out - searching
            TODO: differentiate b/w plume-trigger exit and just searching
        out2in - entering plume
        in2in - staying
        in2out - exiting
        """
        current_state, past_state = inPlume[tsi], inPlume[tsi-1]
        if current_state == False and past_state == False:
            # we are not in plume and weren't in last ts
            return 'searching'
        if current_state == True and past_state == False:
            # entering plume
            return 'entering'
        if current_state == True and past_state == True:
            # we stayed in the plume
            return 'staying'
        if current_state == False and past_state == True:
            # exiting the plume
            return "exiting"
            
            
    def _calc_polar_kinematics(self, V):
#        # linalg norm basically does sqrt( sum(x**2) )
#        self.ensemble['magnitude'] = [np.linalg.norm(x) for x in self.ensemble.values]
#        self.ensemble['angle'] = np.tan(self.ensemble['velocity_y'] / self.ensemble['velocity_x']) % (2*np.pi)
#        
        for name in self.metadata['kinematic_vals']+self.metadata['forces']: #['velocity', 'acceleration', 'biasF', 'wallRepulsiveF', 'upwindF', 'stimF']
            # adds attribute to V
            x_component, y_component = getattr(V, name+'_x'), getattr(V, name+'_y')
            angle = np.arctan2(y_component, x_component)
            angle[angle<0] += 2*np.pi # get vals b/w [0,2pi]
#            print angle
            setattr(V, name+'_xy_theta', angle)
            setattr(V, name+'_xy_mag', np.sqrt((y_component)**2 + (x_component)**2))
        
        return V
#                for ext in ['_x', '_y', '_z', '_xy_theta', '_xy_mag']:
#                    value = setattr(V, name+ext, np.full(tsi_max, np.nan))
                
        
    
class Object(object):
        pass
    
        


if __name__ == '__main__':
    # wallF params
    scalar = 1e-7
    
    wallF_params = [scalar]  #(4e-1, 1e-6, 1e-7, 250)

    # temperature plume
    myplume = plume3D.Plume()
    trajectories = trajectory3D.Trajectory() # instantiate empty trajectories object
    myagent = Agent(trajectories, myplume, agent_pos="door", target_pos="left", v0_stdev=0.01, wtf=7e-07,\
        biasF_scale=4e-05, stimF_str=1e-4, beta=1e-5, Tmax=15., dt=0.01, detect_thresh=0.023175, \
        bounded=True, wallF_params=wallF_params)
    myagent.fly(total_trajectories=100)
    
   
#    trajectories.describe(plot_kwargs = {'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True, 'force_violin':True})
    trajectories.plot_single_3Dtrajectory()
    