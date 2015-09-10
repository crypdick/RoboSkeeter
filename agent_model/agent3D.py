# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:08:51 2015

@author: richard

TODO: implemement unit tests with nose
"""

import numpy as np
import baseline_driving_forces3D
import stim_biasF3D
import plume3D
import trajectory3D
import windtunnel
import sys
import pandas as pd


def solve_heading(vector_x, vector_y):  # TODO: export to trajectory
    from math import atan2

    theta = atan2(vector_y, vector_x)

    return theta*180/np.pi


# def fly_wrapper(agent_obj, args, traj_count):
#     """wrapper fxn for fly_single to use for multithreading
#     """
#     vectors_object = agent_obj._fly_single(*args)
#     setattr(vectors_object, 'trajectory_num', traj_count)
#     df = pd.DataFrame(vars(vectors_object))
#
#
#     return df


def attraction_basin(k, pos, y_spring_center=0.12, z_spring_center=0.2):
    """given y position, determine if the agent is currently flighing in the left or right half of the windtunnel.
    then, figures out which spring center to use,
     and returns the spring force.

    TODO: solve for correct spring centers
    TODO: make separate k for y, z?
    """
    x_pos, y_pos, z_pos = pos
    if y_pos >= 0:
        y_sign = 1
    else:
        y_sign = -1
    return np.array([
        0.,
        k * ((y_sign * y_spring_center) -  y_pos),
        k * (z_spring_center -  z_pos)
        ])



class Agent():
    """Generate agent (our simulated mosquito) which can fly.

    TODO: separate windtunnel environment into its own class

    Args:
        Trajectory_object: (trajectory object)
            pandas dataframe which we store all our simulated trajectories and their kinematics
        Plume_object: (plume object)
            the temperature data for our thermal, convective plume
        windtunnel_object: (windtunnel object)
            our virtual wind tunnel
        agent_position: (list/array, "cage", "center")
            how to pick the initial position coordinates of a flight
        v0_stdev: (float)
            stdev of initial velocity distribution (default: experimentally observed v0, fit to a normal distribution)
        heater: (list/array, "left", "right", or "None")
            the heater condition: left/right heater turned on (set to None if no heater is on)
        Tmax: (float)
            max time an agent is allowed to fly
            (data: <control flight duration> = 4.4131 +- 4.4096)  # TODO: recheck when Sharri's new data comes in
        dt: (float)
            width of our time steps
        k: (float)
            spring constant for our wall attraction flow
        beta: (float)
            damping force (kg/s). Undamped = 0, Critical damping = 1,t
            NOTE: if beta is too big, the agent accelerates to infinity!
        randomF_scale: (float)
            parameter to scale the main driving force
        wtf: (float)
            magnitude of upwind bias force   # TODO units
        

    All args are in SI units and based on behavioral data:

    

    Returns:
        Agent object

    """
    def __init__(self,
        Trajectory_object,
        Plume_object,
        Windtunnel_object,
        initial_position_selection='downwind_plane',
        initial_velocity_stdev=0.01,
        time_max=15.,
        dt=0.01,
        experimental_condition=None,  #FIXME export
        damping_coeff=1e-5,
        randomF_strength=4e-06,
        windF_strength=5e-06,
        stimF_stength=1e-4,
        bounded=True,  # fixme export
        mass = 2.88e-6, #3.0e-6 # 2.88e-6  # mass (kg) =2.88 mg,
        spring_const=0.):
        """generate the agent"""
        # TODO: load wind tunnel, plume into here

        self.mass = mass # population weight data: 2.88 +- 0.35mg
        self.time_max = time_max
        self.time_bin_width = dt
        self.max_bins = int(np.ceil(self.time_max / dt))  # N bins
        self.stimF_strength = stimF_stength

        # windtunnel
        self.boundary = Windtunnel_object.boundary
        self.experimental_condition = Windtunnel_object.test_condition
        self.collision = "elastic"

        self.initial_position_selection = initial_position_selection
        self.initial_velocity_stdev = initial_velocity_stdev
        self.spring_const = spring_const
        self.damping_coeff = damping_coeff
        self.randomF_strength= randomF_strength
        self.windF_strength = windF_strength
        
        self.kinematics_list = ['position', 'velocity', 'acceleration']
        self.forces_list = ['totalF', 'randomF', 'wallRepulsiveF', 'upwindF', 'stimF']
        
        
        # turn thresh, in units deg s-1.
        # From Sharri:
        # it is the stdevof the broader of two Gaussians that fit the distribution of angular velocity
        self.turn_threshold = 433.5  # TODO: move this to trajectories       
        
        self.trajectory_obj = Trajectory_object
        self.plume_obj = Plume_object
        
        # # create repulsion landscape
        # self._repulsion_funcs = repulsion_landscape3D.landscape(boundary=self.boundary)



    def fly(self, total_trajectories=1, verbose=True):
        ''' iterates self._fly_single() total_trajectories times
        '''
        args = (self.time_bin_width, self.mass)
        df_list = []

        traj_count = 0
        while traj_count < total_trajectories:
            if verbose is True:
                sys.stdout.write("\rTrajectory {}/{}".format(traj_count+1, total_trajectories))
                # sys.stdout.flush()

            array_dict = self._generate_flight(*args)
            array_dict['trajectory_num'] = [traj_count] * len(array_dict['velocity_x'])  # enumerate the trajectories
            df = pd.DataFrame(array_dict)

            df_list.append(df)
            
            traj_count += 1
            if traj_count == total_trajectories:
                if verbose is True:
                    sys.stdout.write("\rSimulations finished. Performing deep magic.")
                    sys.stdout.flush()
        

        
        self.total_trajectories = total_trajectories
        self.trajectory_obj.load_ensemble(df_list)  # concatinate all the dataframes at once instead of one at a
                                                          # time for performance boost.
        # concluding stats
#        self.trajectory_obj.add_agent_info({'time_target_find_avg': trajectory3D.T_find_stats(self.trajectory_obj.agent_info['time_to_target_find'])})
    
    def _generate_flight(self, dt, m, bounded=True):
        """Generate a single trajectory using our model.
    
        First put everything into np arrays stored inside of a dictionary
        """
        V = self._initialize_vectors()
        V['position'][0], V['velocity'][0] = self._set_init_pos_and_velo()

        for tsi in xrange(self.max_bins):
            V['inPlume'][tsi] = self._check_in_plume(V)
            V = self._calc_current_behavioral_state(tsi, V)
            V = self._calc_forces(V, tsi)


            # calculate current acceleration
            V['acceleration'][tsi] = V['totalF'][tsi] / m

            # check if time is out, end loop before we solve for future velo, position
            if tsi == self.max_bins-1: # -1 because of how range() works
#                self.metadata['target_found'][0]  = False
#                self.metadata['time_to_target_find'][0] = np.nan
                V = self._land(tsi, V)
                break

            ################################################
            # Calculate candidate velocity and positions
            ################################################
            V['velocity'][tsi+1]= V['velocity'][tsi] + V['acceleration'][tsi] * dt

            candidate_pos = V['position'][tsi] + V['velocity'][tsi]*dt # shouldnt position in future be affected by the forces in this timestep? aka use velo[i+1]

            ################################################
            # if walls are enabled, check if candidate velocity and position is illegal
            ################################################
            if bounded is True:
                answer = self._check_candidate_position(V, tsi, candidate_pos)
                if type(answer) is str:  # means that we reached the end of the wind tunnel
                    V = self._land(tsi-1, V)  # discard last row
                    break
                else:
                    V = answer

            V['position'][tsi+1] = candidate_pos

            # ################################################
            # # solve for heading angle, turning state, etc. TODO: export to trajectories class
            # ################################################
            # # solve final heading #TODO: also solve zy?
            # V['heading_angle'][tsi] = solve_heading(V['velocity_y'][tsi], V['velocity_x'][tsi])
            # # ... and angular velo
            # if tsi == 0: # hack to prevent index error
            #     V['velocity_angular'][tsi] = 0
            # else:
            #     V['velocity_angular'][tsi] = (V['heading_angle'][tsi] - V['heading_angle'][tsi-1]) / dt
            #
            # # turning state
            # if tsi in [0, 1]:
            #     V['turning'][tsi] = 0
            # else:
            #     turn_min = 3
            #     if abs(V['velocity_angular'][tsi-turn_min:tsi]).sum() > self.turn_threshold*turn_min:
            #         V['turning'][tsi] = 1
            #     else:
            #         V['turning'][tsi] = 0

        # dataframe only accepts 1D vectors
        # split xyz arrays into separate x, y, z vectors for dataframe
        for kinematic in self.kinematics_list+self.forces_list:
            V[kinematic+'_x'], V[kinematic+'_y'], V[kinematic+'_z'] = np.split(V[kinematic], 3, axis=1)
            del V[kinematic]  # delete 3D array

        for key, array in V.iteritems():
            V[key] = V[key].reshape(len(array))  # fix pandas bug when trying to load (R,1) arrays


        return V


    def _initialize_vectors(self):
        """
        # initialize np arrays, store in dictionary
        """
        V = {}

        for name in self.kinematics_list+self.forces_list:
            V[name] = np.full((self.max_bins, 3), np.nan)

        V['times'] = np.linspace(0, self.time_max, self.max_bins)
        V['inPlume'] = np.full(self.max_bins, -1, dtype=np.uint8)
        V['behavior_state'] = np.array([None]*self.max_bins)
        V['turning'] = np.array([None]*self.max_bins)
        V['heading_angle'] = np.full(self.max_bins, np.nan)
        V['velocity_angular'] = np.full(self.max_bins, np.nan)

        return V


    def _set_init_pos_and_velo(self, agent_pos='downwind_plane'):
        ''' puts the agent in an initial position, usually within the bounds of the
        cage

        Options: [the cage] door, or anywhere in the plane at x=.1 meters

        set initial velocity from fitted distribution
        '''

        # generate random intial velocity condition using normal distribution fitted to experimental data
        initial_velocity = np.random.normal(0, self.initial_velocity_stdev, 3)

        if type(agent_pos) is list:
            initial_position = np.array(agent_pos)
        if agent_pos == "door":  # start trajectories as they exit the front door
            initial_position = np.array([0.1909, np.random.uniform(-0.0381, 0.0381), np.random.uniform(0., 0.1016)])
            # FIXME cage is actually suspending above floor
        if agent_pos == 'downwind_plane':
            initial_position =  np.array([0.1, np.random.uniform(-0.127, 0.127), np.random.uniform(0., 0.254)])
        else:
            raise Exception('invalid agent position specified')


        return initial_position, initial_velocity


    def _check_in_plume(self, V):
        if self.plume_obj.condition is None:  # skip if no plume  # TODO: always check for plume
            inPlume = 0
        else:
            inPlume = self.plume_obj.check_for_plume(V['position'][tsi])

        return inPlume


    def _calc_current_behavioral_state(self, tsi, V):
        if tsi == 0:  # always start searching
            V['behavior_state'][tsi] = 'searching'
        else:
            if self.plume_obj.condition is None:  # hack for no plume condition
                V['behavior_state'][tsi] = 'searching'
            else:
                if V['behavior_state'][tsi] is None: # need to find state
                    V['behavior_state'][tsi] = self._check_crossing_state(tsi, V['inPlume'], V['velocity_y'][tsi-1])
                    if V['behavior_state'][tsi] in (
                            'Left_plume Exit leftLeft_plume Exit rightRight_plume Exit leftRight_plume Exit right'):
                        try:
                            for i in range(PLUME_TRIGGER_TIME): # store experience for the following timeperiod
                                V['behavior_state'][tsi+i] = str(V['behavior_state'][tsi])
                        except IndexError: # can't store whole snapshot, so save 'truncated' label instead
                           # print "plume trigger turn lasted less than threshold of {} timesteps "\
                           # "before trajectory ended, so adding _truncated suffix".format(self.max_bins)  # TODO: DOCUMENT
                           for i in range(self.max_bins - tsi):
                                V['behavior_state'][tsi+i] = (str(V['behavior_state'][tsi])+'_truncated')

                else: # state is still "plume exit" because we haven't re-entered the plume
                    if V['inPlume'] is False:  # plume not found :'( better luck next time, Roboskeeter.
                        pass
                    else:  # found the plume again!
                        V['behavior_state'][tsi:] = None  # reset memory
                        V['behavior_state'][tsi] = 'entering'

        return V

    def _calc_forces(self, V, tsi):
        ################################################
        # Calculate driving forces at this timestep
        ################################################
        V['stimF'][tsi] = stim_biasF3D.stimF(V['behavior_state'][tsi], self.stimF_strength)

        V['randomF'][tsi] = baseline_driving_forces3D.random_force(self.randomF_strength)

        V['upwindF'][tsi] = baseline_driving_forces3D.upwindBiasForce(self.windF_strength)

        V['wallRepulsiveF'][tsi] = attraction_basin(self.damping_coeff, V['position'][tsi])

        ################################################
        # calculate total force
        ################################################
        V['totalF'][tsi] = -self.damping_coeff * V['velocity'][tsi] +  V['randomF'][tsi] + V['upwindF'][tsi] + V['wallRepulsiveF'][tsi] + V['stimF'][tsi]
        ###############################

        return V


    def _land(self, tsi, array_dict):
        ''' trim excess timebins in arrays
        '''
        for array in array_dict.itervalues():
            array = array[:tsi]

        # array_dict = self._calc_polar_kinematics(array_dict)  # TODO: export to trajectories

        # absolute magnitude of velocity, accel vectors in 3D
        array_dict['velocity_3Dmagn'] = np.linalg.norm(array_dict['velocity'], axis=1)
        array_dict['acceleration_3Dmagn'] = np.linalg.norm(array_dict['acceleration'], axis=1)
        
        return array_dict
        
        
    def _check_crossing_state(self, tsi, inPlume, velocity_y):
        """
        out2out - searching
         (or orienting, but then this func shouldn't be called)
        out2in - entering plume
        in2in - staying
        in2out - exiting
            {Left_plume Exit left, Left_plume Exit right
            Right_plume Exit left, Right_plume Exit right}
        TODO: export to trajectory class
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
            if self.experimental_condition[5] == 'left':
                if velocity_y <= 0:
                    return 'Left_plume Exit left'
                else:
                    return 'Left_plume Exit right'
            else:
                if velocity_y <= 0:
                    return "Right_plume Exit left"
                else:
                    return "Right_plume Exit right"

    def _check_candidate_position(self, V, tsi, candidate_pos):
        # x dim
        if candidate_pos[0] > self.boundary[1]:  # reached far (upwind) wall (end)
#                    self.metadata['target_found'][0]  = False
#                    self.metadata['time_to_target_find'][0] = np.nan
            return 'land'
        if candidate_pos[0] < self.boundary[0]:  # too far left
#                    print "too far left"
            candidate_pos[0] = self.boundary[0] + 1e-4 # teleport back inside
            if self.collision == 'elastic':
                V['velocity_x'][tsi+1] *= -1
#                        print "boom! left"
            elif self.collision == 'crash':
                V['velocity_x'][tsi+1] = 0.

        #y dim
        if candidate_pos[1] > self.boundary[2]:  # too left
#                    print "too left"
            candidate_pos[1] = self.boundary[2] + 1e-4 # note, left is going more negative in our convention
            if self.collision == 'elastic':
                V['velocity'][tsi+1][1] *= -1
            elif self.collision == "crash":
#                        print "teleport!"
                V['velocity'][tsi+1][1] = 0.
#                        print "crash! top wall"
        if candidate_pos[1] < self.boundary[3]:  # too far right
#                    print "too far right"
            candidate_pos[1] = self.boundary[3] - 1e-4
            if self.collision == 'elastic':
                V['velocity'][tsi+1][1] *= -1
            elif self.collision == 'crash':
                V['velocity'][tsi+1][1] = 0.

        # z dim
        if candidate_pos[2] > self.boundary[5]:  # too far above
#                    print "too far above"
            candidate_pos[2] = self.boundary[5] - 1e-4
            if self.collision == 'elastic':
                V['velocity'][tsi+1][2] *= -1
#                        print "boom! top"
            elif self.collision == "crash":
#                        print "teleport!"
                V['velocity'][tsi+1][2] = 0.
#                        print "crash! top wall"
        if candidate_pos[2] < self.boundary[4]:  # too far below
#                    print "too far below"
            candidate_pos[2] = self.boundary[4] + 1e-4
            if self.collision == 'elastic':
                V['velocity'][tsi+1][2] *= -1
            elif self.collision == 'crash':
                V['velocity'][tsi+1][2] = 0.

        V['position'][tsi+1] = candidate_pos

        return V

    def _calc_polar_kinematics(self, array_dict):
        """append polar kinematics to vectors dictionary TODO: export to trajectory class"""
        for name in self.kinematics_list+self.forces_list:  # ['velocity', 'acceleration', 'randomF', 'wallRepulsiveF', 'upwindF', 'stimF']
            x_component, y_component = array_dict[name+'_x'], array_dict[name+'_y']
            angle = np.arctan2(y_component, x_component)
            angle[angle < 0] += 2*np.pi  # get vals b/w [0,2pi]
            array_dict[name+'_xy_theta'] = angle
            array_dict[name+'_xy_mag'] = np.sqrt(y_component**2 + x_component**2)
        
        return array_dict



def gen_objects_and_fly(N_TRAJECTORIES, TEST_CONDITION, BETA, FORCES_AMPLITUDE, F_WIND_SCALE, K):
    """
    Params fitted using scipy.optimize

    """
    # generate environment
    windtunnel_object = windtunnel.Windtunnel(TEST_CONDITION)
    # generate temperature plume
    plume_object = plume3D.Plume(TEST_CONDITION)
    # instantiate empty trajectories class
    trajectories_object = trajectory3D.Trajectory()
    # instantiate a Roboskeeter
    skeeter = Agent(
        trajectories_object,
        plume_object,
        windtunnel_object,
        mass=MASS,
        initial_position_selection="downwind_plane",
        windF_strength=F_WIND_SCALE,
        randomF_strength=FORCES_AMPLITUDE,
        stimF_stength=F_STIM_SCALE,
        spring_const=K,
        damping_coeff=BETA,
        time_max=4.,
        dt=0.01,
        bounded=True)
    sys.stdout.write("\rAgent born")

    # make the skeeter fly. this updates the trajectories_object
    skeeter.fly(total_trajectories=N_TRAJECTORIES)
    
    return plume_object, skeeter.trajectory_obj, skeeter
    
   
#    trajectories.describe(plot_kwargs = {'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True, 'force_violin':True})
    #trajectories.plot_single_3Dtrajectory()
#    
    

if __name__ == '__main__':
    N_TRAJECTORIES = 30
    TEST_CONDITION = None  # {'Left', 'Right', None}
    # old beta- 5e-5, forces 4.12405e-6, fwind = 5e-7
    BETA, FORCES_AMPLITUDE, F_WIND_SCALE =  [  1.37213380e-06  , 1.39026239e-06 ,  7.06854777e-07]
    MASS = 2.88e-6
    F_STIM_SCALE = 0.  #7e-7,   # set to zero to disable tracking hot air
    K = 0.  #1e-7               # set to zero to disable wall attraction

    myplume, trajectories, skeeter = gen_objects_and_fly(N_TRAJECTORIES, TEST_CONDITION, BETA, FORCES_AMPLITUDE, F_WIND_SCALE, K)

    print "\nDone."

    ######################### plotting methods
    trajectories.plot_single_3Dtrajectory(0)  # plot ith trajectory in the ensemble of trajectories

    # trajectories.plot_kinematic_hists()
    # trajectories.plot_posheatmap()
    # trajectories.plot_force_violin()
    # trajectories.plot_kinematic_compass()
    # trajectories.plot_sliced_hists()

    ######################### dump data for csv for Sharri
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
    # # g = e.loc[e['behavior_state'].isin(['Left_plume Exit left', 'Left_plume Exit right', 'Right_plume Exit left', 'Right_plume Exit right'])]