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
from math import atan2
import baseline_driving_forces3D
import stim_biasF3D
import plume3D
import trajectory3D
import repulsion_landscape3D
import sys
import pandas as pd


def place_heater(location):
    ''' given {left, right, none, custom location} place heater in the coordinates in our convention, as well as height
    and diameter location

    Args:
    location

    returns [x,y, zmin, zmax, diam]
    '''
    zmin = 0.03800
    zmax = 0.11340
    diam = 0.01905
    if location is None:
            return None
    elif location in "leftLeftLEFT":
        return [0.8651, -0.0507, zmin, zmax, diam, 'left']
    elif location in "rightRightRIGHT":
        return [0.8651, 0.0507, zmin, zmax, diam, 'right']
    elif type(location) is list:
        return location
    else:
        raise Exception('invalid location type specified')


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
    if agent_pos == 'downwind_plane':
        return [0.1, np.random.uniform(-0.127, 0.127), np.random.uniform(0., 0.254)]
    else:
        raise Exception('invalid agent position specified')




def solve_heading(velo_x, velo_y):
    theta = atan2(velo_y, velo_x)
    return theta*180/np.pi


def fly_wrapper(agent_obj, args, traj_count):
    """wrapper fxn for fly_single to use for multithreading
    """
    vectors_object = agent_obj._fly_single(*args)
    setattr(vectors_object, 'trajectory_num', traj_count)
    df = pd.DataFrame(vars(vectors_object))
    
    
    return df


class Agent():
    """Generate agent

    Args:
        Trajectory_object: (trajectory object)
            
        Plume_object: (plume object)
            
        agent_position: (list/array, "cage", "center")
            sets initial position r0 (meters)
        v0_stdev: (float)
            stdev of initial velocity distribution 
        heater: (list/array, "left", "right", or "None")
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
    def __init__(
        self,
        Trajectory_object,
        Plume_object,
        agent_pos='door',
        v0_stdev=0.01,
        Tmax=15.,
        dt=0.01,
        heater='left',
        beta=1e-5,
        F_baseline_scale=4e-06,
        wtf=7e-07,
        wtf_scalar=0.05,
        stimF_str=1e-4,
        detect_thresh=0.023175,
        bounded=True,
        wallF_params=(4e-1, 1e-6, 1e-7, 250, "walls_only"),
        mass = 3e-6, #3.0e-6 # 2.88e-6  # mass (kg) =2.88 mg,
        k=0.):
        """ Initialize object with instant variables, and trigger other funcs.
        """
        
        self.boundary = [0.0, 1.0, 0.127, -0.127, 0., 0.254]  # these are real dims of our wind tunnel
        self.Tmax = Tmax
        self.dt = dt
        self.v0_stdev = v0_stdev
        self.k = k
        self.beta = beta
        self.biasF_scale = F_baseline_scale
        # self.wtf = wtf
        # the mean of a lognorm dist is E[X] = e ** (mu + 1/2 omega **2)
        # in our case, mu = -0.405632480939 and sigma = 0.932352661694 * 0.5
        # which is then scaled by F_baseline_scale
        self.wtf = np.exp(-0.405632480939 + 0.5 * 0.932352661694 * 0.5 ** 2) * F_baseline_scale * wtf_scalar
        # print wtf, self.wtf_scalar, F_baseline_scale
        self.detect_thresh = detect_thresh     
        self.wallF_params = wallF_params
        self.stimF_str = stimF_str
        
        
        # place heater
        self.heater = place_heater(heater)
        
        self.metadata = dict()
        # population weight data: 2.88 +- 0.35mg
        self.metadata['mass'] = mass
        self.metadata['time_max'] = Tmax
        self.metadata['boundary'] = self.boundary
        self.metadata['heater_position'] = self.heater
        self.metadata['detection_threshold'] = detect_thresh
        self.metadata['initial_position'] = agent_pos
        self.metadata['initial_velo_stdev'] = v0_stdev
        self.metadata['k'] = k
        self.metadata['beta'] = beta
        self.metadata['F_baseline_scale'] = F_baseline_scale
        self.metadata['wtf'] = self.wtf
        self.metadata['wallF_params'] = wallF_params
        # for stats, later
        self.metadata['time_target_find_avg'] = []
        self.metadata['total_finds'] = 0
        self.metadata['target_found'] = [False]  # TODO: shouldn't this be attached to Trajectories() instead?
        self.metadata['time_to_target_find'] = [np.nan] # TODO: make sure these lists are concating correctly
        
        self.metadata['kinematic_vals'] = ['velocity', 'acceleration']
        self.metadata['forces'] = ['totalF', 'biasF', 'wallRepulsiveF', 'upwindF', 'stimF']
        
        
        # turn thresh, in units deg s-1.
        # From Sharri:
        # it is the stdevof the broader of two Gaussians that fit the distribution of angular velocity
        self.metadata['turn_threshold'] = 433.5         
        
        self.trajectory_obj = Trajectory_object
        self.plume_obj = Plume_object
        
        # # create repulsion landscape
        # self._repulsion_funcs = repulsion_landscape3D.landscape(boundary=self.boundary)


    def fly(self, total_trajectories=1, verbose=True):
        ''' iterates _fly_single total_trajectories times
        '''
        traj_count = 0
        args = (self.dt, self.metadata['mass'], self.detect_thresh, self.boundary)
        df_list = []
        while traj_count < total_trajectories:
            if verbose is True:
                sys.stdout.write("\rTrajectory {}/{}".format(traj_count+1, total_trajectories))
                sys.stdout.flush()
            df = fly_wrapper(self, args, traj_count)
            df_list.append(df)
#            vectors_object = self._fly_single(self.dt, self.metadata['mass'], self.detect_thresh, self.boundary)
#            # extract trajectory object attribs, append to our lists.
#            setattr(vectors_object, 'trajectory_num', traj_count)
            
#            self.trajectory_obj.append_ensemble(vars(vectors_object))
            
            traj_count += 1
            if traj_count == total_trajectories:
                if verbose is True:
                    sys.stdout.write("\rSimulations finished. Performing deep magic.")
                    sys.stdout.flush()
        

        
        self.metadata['total_trajectories'] = total_trajectories
        self.trajectory_obj.add_agent_info(self.metadata)
        self.trajectory_obj.ensemble = pd.concat(df_list)
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
        PLUME_TRIGGER_TIME = int(1.0/dt)
        tsi_max = int(np.ceil(self.metadata['time_max'] / dt))  # N bins/maximum time step
        
        # V for vector
        # retrieve dict w/ vars(V)
        V = Object()
        
        for name in self.metadata['kinematic_vals']+self.metadata['forces']:
            # adds attribute to V
            for ext in ['_x', '_y', '_z', '_xy_theta', '_xy_mag']:
                setattr(V, name+ext, np.full(tsi_max, np.nan))
        
        
        
        # initialize np arrays
        
        V.times = np.linspace(0, self.metadata['time_max'], tsi_max)
        V.position_x = np.full(tsi_max, np.nan)
        V.position_y = np.full(tsi_max, np.nan)
        V.position_z = np.full(tsi_max, np.nan)
#        _post_velocity_x = np.full(tsi_max, np.nan)
#        _temperature = np.full(tsi_max, np.nan) # depreciated
        V.inPlume = np.full(tsi_max, -1, dtype=np.uint8)
        V.plume_experience = [None]*tsi_max
        V.turning = [None]*tsi_max
        V.heading_angle = np.full(tsi_max, np.nan)
        V.velocity_angular = np.full(tsi_max, np.nan)

        ######## INITIAL CONDITIONS
        # place agent
        V.position_x[0], V.position_y[0], V.position_z[0] = place_agent(self.metadata['initial_position'])
        self.dim = 3
        
        # generate random intial velocity condition
        V.velocity_x[0], V.velocity_y[0], V.velocity_z[0] = np.random.normal(0, self.v0_stdev, self.dim)

        # generate a flight!
        for tsi in xrange(tsi_max):
            # import pdb; pdb.set_trace()
            # sense the temperature
#            _temperature[tsi] = self.plume_obj.temp_lookup([_position_x[tsi], _position_y[tsi]]) # depreciated
            
            # are we in the plume?
            if self.plume_obj.condition is None:  # skip if no plume
                V.inPlume[tsi] = 0
            else:
                V.inPlume[tsi] = self.plume_obj.check_for_plume(
                    [V.position_x[tsi], V.position_y[tsi], V.position_z[tsi]])
            
            ############ what is our behavior given our plume interactions thus far?
            if tsi == 0:
                V.plume_experience[tsi] = 'searching'
            else:
                if self.plume_obj.condition is None:  # hack for no plume condition
                    V.plume_experience[tsi] = 'searching'
                else:
                    if V.plume_experience[tsi] is None: # need to find state
                        V.plume_experience[tsi] = self._check_crossing_state(tsi, V.inPlume, V.velocity_y[tsi-1])
                        # import pdb; pdb.set_trace()
                        if V.plume_experience[tsi] in (
                                'Left_plume Exit leftLeft_plume Exit rightRight_plume Exit leftRight_plume Exit right'):
                            try:
                                for i in range(PLUME_TRIGGER_TIME): # store experience for the following timeperiod
                                    V.plume_experience[tsi+i] = str(V.plume_experience[tsi])
                            except IndexError: # can't store whole snapshot, so save 'truncated' label instead
                               # print "plume trigger turn lasted less than threshold of {} timesteps "\
                               # "before trajectory ended, so adding _truncated suffix".format(tsi_max)
                               for i in range(tsi_max - tsi):
                                    V.plume_experience[tsi+i] = (str(V.plume_experience[tsi])+'_truncated')

                    else: # if we're already orienting from past memory
                        if V.inPlume is False:  # we haven't re-entered the plume
                            pass
                        else:  # found the plume again!
                            V.plume_experience[tsi:]
            ######################################################################

            
            # calculate driving forces
            F_stim = stim_biasF3D.stimF(V.plume_experience[tsi], self.stimF_str)
            V.stimF_x[tsi], V.stimF_y[tsi], V.stimF_z[tsi] = F_stim
            
            F_base = baseline_driving_forces3D.bias_force(self.biasF_scale)
            V.biasF_x[tsi], V.biasF_y[tsi], V.biasF_z[tsi] = F_base
            
            F_upwind = baseline_driving_forces3D.upwindBiasForce(self.wtf)
            V.upwindF_x[tsi], V.upwindF_y[tsi], V.upwindF_z[tsi] = F_upwind

            F_wall_repulsion = self.wallF_params[0] * repulsion_landscape3D.xyz_to_weights([V.position_x[tsi], V.position_y[tsi], V.position_z[tsi]])
            # F_wall_repulsion = baseline_driving_forces3D.repulsionF(\
            #     np.array([V.position_x[tsi], V.position_y[tsi], V.position_z[tsi]]),\
            #     self._repulsion_funcs, self.wallF_params)
            V.wallRepulsiveF_x[tsi], V.wallRepulsiveF_y[tsi], V.wallRepulsiveF_z[tsi] = F_wall_repulsion
            
#            else:
#                try:
            
#                except UnboundLocalError: # TODO: wtf?
#                    F_stim, inPlume = stim_biasF3D.main(_temperature[tsi], V.velocity_y[tsi],\
#                        False, self.stimF_str)
#                    print "erorr"
#                else:
#                    print 
#            _stimF_x[tsi], _stimF_y[tsi], _stimF_z[tsi] = F_stim
            
            ########################### calculate current force
            F_total = -self.beta*np.array([V.velocity_x[tsi], V.velocity_y[tsi], V.velocity_z[tsi]])\
              + F_base + F_upwind + F_wall_repulsion + F_stim
            V.totalF_x[tsi], V.totalF_y[tsi], V.totalF_z[tsi] = F_total
            ###############################
            
            # calculate current acceleration
            V.acceleration_x[tsi], V.acceleration_y[tsi], V.acceleration_z[tsi] = F_total / m

            # if time is out, end loop before we solve for future velo, position
            if tsi == tsi_max-1: # -1 because of how range() works
#                self.metadata['target_found'][0]  = False
#                self.metadata['time_to_target_find'][0] = np.nan
                V = self.land(tsi, V)
                break

            # calculate future velocity given the forces we just got
            V.velocity_x[tsi+1], V.velocity_y[tsi+1], V.velocity_z[tsi+1] = \
                np.array([V.velocity_x[tsi], V.velocity_y[tsi], V.velocity_z[tsi]]) \
                + np.array([V.acceleration_x[tsi], V.acceleration_y[tsi], V.acceleration_z[tsi]]) * dt

            
            # solve candidate position for next timestep
            candidate_pos = np.array([V.position_x[tsi], V.position_y[tsi], V.position_z[tsi]]) \
                + np.array([V.velocity_x[tsi], V.velocity_y[tsi], V.velocity_z[tsi]])*dt # shouldnt position in future be affected by the forces in this timestep? aka use velo[i+1]
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
#                    print "too far left"
                    candidate_pos[0] = boundary[0] + 1e-4 # teleport back inside
                    if BOUNCE == 'elastic':
                        V.velocity_x[tsi+1] *= -1
#                        print "boom! left"
                    elif BOUNCE == 'crash':
                        V.velocity_x[tsi+1] = 0.
            
                #y dim
                if candidate_pos[1] > boundary[2]:  # too left
#                    print "too left"
                    candidate_pos[1] = boundary[2] + 1e-4 # note, left is going more negative in our convention
                    if BOUNCE == 'elastic':
                        V.velocity_y[tsi+1] *= -1
                    elif BOUNCE == "crash":
#                        print "teleport!"
                        V.velocity_y[tsi+1] = 0.
#                        print "crash! top wall"
                if candidate_pos[1] < boundary[3]:  # too far right
#                    print "too far right"
                    candidate_pos[1] = boundary[3] - 1e-4
                    if BOUNCE == 'elastic':
                        V.velocity_y[tsi+1] *= -1
                    elif BOUNCE == 'crash':
                        V.velocity_y[tsi+1] = 0.
                
                # z dim
                if candidate_pos[2] > boundary[5]:  # too far above
#                    print "too far above"
                    candidate_pos[2] = boundary[5] - 1e-4
                    if BOUNCE == 'elastic':
                        V.velocity_z[tsi+1] *= -1
#                        print "boom! top"
                    elif BOUNCE == "crash":
#                        print "teleport!"
                        V.velocity_z[tsi+1] = 0.
#                        print "crash! top wall"
                if candidate_pos[2] < boundary[4]:  # too far below
#                    print "too far below"
                    candidate_pos[2] = boundary[4] + 1e-4
                    if BOUNCE == 'elastic':
                        V.velocity_z[tsi+1] *= -1
                    elif BOUNCE == 'crash':
                        V.velocity_z[tsi+1] = 0.
                        
                # save screened candidate_pos to future position
            V.position_x[tsi+1], V.position_y[tsi+1], V.position_z[tsi+1] = candidate_pos

            # solve final heading #TODO: also solve zy?
            V.heading_angle[tsi] = solve_heading(V.velocity_y[tsi], V.velocity_x[tsi])
            # ... and angular velo
            if tsi == 0: # hack to prevent index error
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
            if self.heater is not None and np.linalg.norm(candidate_pos - self.heater[0:3]) < self.detect_thresh:
                   self.metadata['target_found'][0]  = True
                   self.metadata['total_finds'] += 1
                   self.metadata['time_to_target_find'][0] = V.times[tsi]  # should this be timeList[i+1]?
                   V = self.land(tsi, V)  # stop flying at source
                   print "source found"
                   break
        
        return V


    def land(self, tsi, V):
        ''' trim excess timebins in arrays
        '''
        for key, array in vars(V).items():
            value = setattr(V, key, array[:tsi+1])

        V = self._calc_polar_kinematics(V)

        # absolute magnitude of velocity, accel vectors in 3D
        velo_mag_stack = np.vstack((V.velocity_x, V.velocity_y, V.velocity_z))
        V.velocity_3Dmagn = np.linalg.norm(velo_mag_stack, axis=0)

        accel_mag_stack = np.vstack((V.acceleration_x, V.acceleration_y, V.acceleration_z))
        V.acceleration_3Dmagn = np.linalg.norm(accel_mag_stack, axis=0)
        
        return V
        
        
    def show_landscape(self):
        import repulsion_landscape
        repulsion_landscape.main(self.wallF_params, None, plotting=True)
        
        
    def _check_crossing_state(self, tsi, inPlume, vy):
        """
        out2out - searching
         (or orienting, but then this func shouldn't be called)
        out2in - entering plume
        in2in - staying
        in2out - exiting
            {Left_plume Exit left, Left_plume Exit right
#           Right_plume Exit left, Right_plume Exit right}
        """
        current_state, past_state = inPlume[tsi], inPlume[tsi-1]
        if current_state == 0 and past_state == 0:
            # we are not in plume and weren't in last ts
            return 'searching'
        if current_state == 1 and past_state == 0:
            # entering plume
            return 'entering'
        if current_state == 1 and past_state == 1:
            # we stayed in the plume
            return 'staying'
        if current_state == 0 and past_state == 1:
            # exiting the plume
            if self.heater[5] == 'left':
                if vy <= 0:
                    return 'Left_plume Exit left'
                else:
                    return 'Left_plume Exit right'
            else:
                if vy <= 0:
                    return "Right_plume Exit left"
                else:
                    return "Right_plume Exit right"
            
            
    def _calc_polar_kinematics(self, V):
        # TODO: also solve for yz
#        # linalg norm basically does sqrt( sum(x**2) )
#        self.ensemble['magnitude'] = [np.linalg.norm(x) for x in self.ensemble.values]
#        self.ensemble['angle'] = np.tan(self.ensemble['velocity_y'] / self.ensemble['velocity_x']) % (2*np.pi)
#        
        for name in self.metadata['kinematic_vals']+self.metadata['forces']: #['velocity', 'acceleration', 'biasF', 'wallRepulsiveF', 'upwindF', 'stimF']
            # adds attribute to V
            x_component, y_component = getattr(V, name+'_x'), getattr(V, name+'_y')
            angle = np.arctan2(y_component, x_component)
            angle[angle < 0] += 2*np.pi  # get vals b/w [0,2pi]
#            print angle
            setattr(V, name+'_xy_theta', angle)
            setattr(V, name+'_xy_mag', np.sqrt(y_component**2 + x_component**2))
        
        return V
#                for ext in ['_x', '_y', '_z', '_xy_theta', '_xy_mag']:
#                    value = setattr(V, name+ext, np.full(tsi_max, np.nan))
                
        
    
class Object(object):
        pass


def main(heater):
    """
    Params fitted using scipy.optimize
    
    biasF_scale => 4.12405e-6 using fminbound
    """
    # wallF params
    scalar = 3e-8 #6e-8

    wallF_params = [scalar]  # (4e-1, 1e-6, 1e-7, 250)

    # temperature plume
    plume_object = plume3D.Plume(heater)
    # import pdb; pdb.set_trace()
    trajectories = trajectory3D.Trajectory() # instantiate empty trajectories object
    skeeter = Agent(
        trajectories,
        plume_object,
        agent_pos="downwind_plane",  # [0.250180, -0.050700, 0.144400],
        heater=heater,
        v0_stdev=0.01,
        wtf=3.5e-7, #7e-07,
        F_baseline_scale=4.12405e-6,
        stimF_str=3.5e-7, #7e-7,
        beta=5e-6,#1e-6,  # 1e-5
        Tmax=15.,
        dt=0.01,
        detect_thresh=0.023175,
        bounded=True,
        wallF_params=wallF_params)
    sys.stdout.write("\rAgent born")
    skeeter.fly(total_trajectories=200)
    
    # trajectories.plot_kinematic_hists()
    
    return plume_object, skeeter.trajectory_obj, skeeter
    
   
#    trajectories.describe(plot_kwargs = {'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True, 'force_violin':True})
    #trajectories.plot_single_3Dtrajectory()
#    
    

if __name__ == '__main__':
    HEATER = 'r' #None # 'l', 'r'
    myplume, trajectories, skeeter = main(HEATER)
    print "\nDone."
    # trajectories.plot_single_3Dtrajectory()
    #
    # # # csv dump for Sharri
    # print "dumping to csvs"
    # e = trajectories.ensemble
    # r = e.trajectory_num.iloc[-1]
    #
    # for i in range(r+1):
    #     e1 = e.loc[e['trajectory_num'] == i, ['position_x', 'position_y', 'position_z', 'inPlume']]
    #     e1.to_csv("l"+str(i)+'.csv', index=False)
    #
    # # trajectories.plot_kinematic_hists()
    # # trajectories.plot_posheatmap()
    # # trajectories.plot_force_violin()
    #
    # # # for plume stats
    # # g = e.loc[e['plume_experience'].isin(['Left_plume Exit left', 'Left_plume Exit right', 'Right_plume Exit left', 'Right_plume Exit right'])]