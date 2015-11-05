# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:08:51 2015

@author: richard

TODO: implemement unit tests with nose
"""

import sys

import numpy as np
import pandas as pd

import plume
import trajectory
import windtunnel
import forces


# def fly_wrapper(agent_obj, args, traj_count):
#     """wrapper fxn for fly_single to use for multithreading
#     """
#     vectors_object = agent_obj._fly_single(*args)
#     setattr(vectors_object, 'trajectory_num', traj_count)
#     df = pd.DataFrame(vars(vectors_object))
#
#
#     return df


class Agent():
    """Generate agent (our simulated mosquito) which can fly.

    TODO: separate windtunnel environment into its own class

    Args:
        Trajectory_object: (trajectory object)
            pandas dataframe which we store all our simulated trajectories and their data
        Plume_object: (plume object)
            the temperature data for our thermal, convective plume
        windtunnel_obj: (windtunnel object)
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
        initial_position_selection='downwind_plane',
        initial_velocity_stdev=0.01,
        time_max=15.,
        dt=0.01,
        experimental_condition=None,
        damping_coeff=1e-5,
        randomF_strength=4e-06,
        windF_strength=5e-06,
        stimF_stength=1e-4,
        bounded=True,
                 mass=2.88e-6,  # avg. mass of our colony (kg) =2.88 mg,
                 spring_const=0.,
                 collision_type='crash'):
        """generate the agent"""

        self.time_max = time_max
        self.time_bin_width = dt
        self.max_bins = int(np.ceil(self.time_max / dt))  # N bins

        self.mass = mass # population weight data: 2.88 +- 0.35mg
        self.stimF_strength = stimF_stength
        self.initial_position_selection = initial_position_selection
        self.initial_velocity_stdev = initial_velocity_stdev
        self.spring_const = spring_const
        self.damping_coeff = damping_coeff
        self.randomF_strength= randomF_strength
        self.windF_strength = windF_strength

        self.experimental_condition = experimental_condition
        self.collision_type = collision_type
        
        self.kinematics_list = ['position', 'velocity', 'acceleration']
        self.forces_list = ['totalF', 'randomF', 'wallRepulsiveF', 'upwindF', 'stimF']
        
        self.windtunnel_obj, self.plume_obj, self.trajectory_obj, self.forces = self._gen_environment_objects()

        # windtunnel
        self.bounded = bounded
        self.boundary = self.windtunnel_obj.boundary

        # turn thresh, in units deg s-1.
        # From Sharri:
        # it is the stdevof the broader of two Gaussians that fit the distribution of angular velocity
        self.turn_threshold = 433.5  # TODO: move this to trajectories       

        
        # # create repulsion landscape
        # self._repulsion_funcs = repulsion_landscape3D.landscape(boundary=self.boundary)

    def _gen_environment_objects(self):
        """generate environment"""
        # we make a windtunnel even in the unbounded case b/c plotting functions use bounds
        windtunnel_object = windtunnel.Windtunnel(self.experimental_condition)
        # generate temperature plume
        plume_object = plume.Plume(self.experimental_condition)
        # instantiate empty trajectories class
        trajectories_object = trajectory.Agent_Trajectory()
        trajectories_object.add_agent_info(self)
        forces_object = forces.Forces()

        return windtunnel_object, plume_object, trajectories_object, forces_object



    def fly(self, total_trajectories=1, verbose=True):
        ''' iterates self._fly_single() total_trajectories times
        '''
        args = (self.time_bin_width, self.mass)
        df_list = []

        traj_i = 0
        while traj_i < total_trajectories:
            if verbose is True:
                sys.stdout.write("\rTrajectory {}/{}".format(traj_i + 1, total_trajectories))
                sys.stdout.flush()


            array_dict = self._generate_flight(*args)

            # if len(array_dict['velocity_x']) < 5:  # hack to catch when optimizer makes trajectories explode
            #     print "catching explosion"
            #     break

            # add label column to enumerate the trajectories
            array_len = len(array_dict['tsi'])
            array_dict['trajectory_num'] = [traj_i] * array_len

            # mk df, add to list of dfs
            df = pd.DataFrame(array_dict)
            df = df.set_index(['trajectory_num'])
            df_list.append(df)

            traj_i += 1
            if traj_i == total_trajectories:
                if verbose is True:
                    sys.stdout.write("\rSimulations finished. Performing deep magic.")
                    sys.stdout.flush()

        self.total_trajectories = total_trajectories
        self.trajectory_obj.load_ensemble_and_analyze(df_list)  # concatinate all the dataframes at once instead of one at a
                                                          # time for performance boost.
        # add agent to trajectory object for plotting funcs

    def _generate_flight(self, dt, m):
        """Generate a single trajectory using our model.
    
        First put everything into np arrays stored inside of a dictionary
        """
        V = self._initialize_vectors()

        # dynamically create easy-to-read aliases for the contents of V
        for key, value in V.iteritems():
            exec(key + " = V['" + key + "']")

        position[0], velocity[0] = self._set_init_pos_and_velo()

        for tsi in V['tsi']:
            inPlume[tsi] = self.plume_obj.is_in_plume(position[tsi])
            V = self._calc_current_behavioral_state(tsi, V)
            stimF[tsi], randomF[tsi], upwindF[tsi], wallRepulsiveF[tsi], totalF[tsi] =\
                self._calc_forces(position[tsi], velocity[tsi], behavior_state[tsi], tsi)


            # calculate current acceleration
            acceleration[tsi] = totalF[tsi] / m

            # check if time is out, end loop before we solve for future velo, position
            if tsi == self.max_bins-1: # -1 because of how range() works
#                self.metadata['target_found'][0]  = False
#                self.metadata['time_to_target_find'][0] = np.nan
                V = self._land(tsi, V)
                break

            ################################################
            # Calculate candidate velocity and positions
            ################################################
            candidate_velo = velocity[tsi] + acceleration[tsi] * dt

            # make sure velocity doesn't diverge to infinity if system is unstable
            # this stops the optimizer from crashing
            candidate_velo = self._velocity_ceiling(candidate_velo)

            candidate_pos = position[tsi] + candidate_velo * dt

            ################################################
            # test candidates
            ################################################
            if self.bounded:
                candidate_pos, candidate_velo = self._collide_with_wall(candidate_pos, candidate_velo)
            #                if candidate_pos is None:  # _collide_with_wall returns None if reaches end
            #                    self._land(tsi, V)  # end flight if reach end of tunnel
            #                    break

            position[tsi + 1] = candidate_pos
            velocity[tsi + 1] = candidate_velo

        # TODO: put in "fix dict" func
        # prepare dict for loading into pandas (dataframe only accepts 1D vectors)
        # split xyz arrays into separate x, y, z vectors for dataframe
        V2 = {'tsi': V['tsi'], 'times': V['times']}
        for kinematic in self.kinematics_list+self.forces_list:
            V2[kinematic+'_x'], V2[kinematic+'_y'], V2[kinematic+'_z'] = np.split(V[kinematic], 3, axis=1)

        # fix pandas bug when trying to load (R,1) arrays when it expects (R,) arrays
        for key, array in V2.iteritems():
            V2[key] = V2[key].reshape(len(array))
            if V2[key].size == 0:
                V2[key] = np.array([0.])  # hack so that kde calculation doesn't freeze on empty arrays


        return V2


    def _initialize_vectors(self):
        """
        # initialize np arrays, store in dictionary
        """
        V = {}

        for name in self.kinematics_list+self.forces_list:
            V[name] = np.full((self.max_bins, 3), np.nan)

        V['tsi'] = np.arange(self.max_bins)
        V['times'] = np.linspace(0, self.time_max, self.max_bins)
        V['inPlume'] = np.full(self.max_bins, -1, dtype=np.uint8)
        V['behavior_state'] = np.array([None]*self.max_bins)
        V['turning'] = np.array([None]*self.max_bins)
        V['heading_angle'] = np.full(self.max_bins, np.nan)
        V['velocity_angular'] = np.full(self.max_bins, np.nan)

        return V

    def _set_init_pos_and_velo(self):
        ''' puts the agent in an initial position, usually within the bounds of the
        cage

        Options: [the cage] door, or anywhere in the plane at x=.1 meters

        set initial velocity from fitted distribution
        '''

        # generate random intial velocity condition using normal distribution fitted to experimental data
        initial_velocity = np.random.normal(0, self.initial_velocity_stdev, 3)
        agent_pos = self.initial_position_selection

        if type(agent_pos) is list:
            initial_position = np.array(agent_pos)
        if agent_pos == "door":  # start trajectories as they exit the front door
            initial_position = np.array([0.1909, np.random.uniform(-0.0381, 0.0381), np.random.uniform(0., 0.1016)])
            # FIXME cage is actually suspending above floor
        if agent_pos == 'downwind_plane':
            initial_position =  np.array([0.1, np.random.uniform(-0.127, 0.127), np.random.uniform(0., 0.254)])
        if agent_pos == 'downwind_high':
            initial_position = np.array(
                [0.05, np.random.uniform(-0.127, 0.127), 0.2373])  # 0.2373 is mode of z pos distribution
        else:
            raise Exception('invalid agent position specified: {}'.format(agent_pos))


        return initial_position, initial_velocity


    def _calc_current_behavioral_state(self, tsi, V):
        if tsi == 0:  # always start searching
            V['behavior_state'][tsi] = 'searching'
        else:
            if self.plume_obj.condition is None:  # hack for no plume condition FIXME
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

    def _calc_forces(self, position, velocity, behavior_state, tsi):
        ################################################
        # Calculate driving forces at this timestep
        ################################################
        stimF = self.forces.stimF(behavior_state, self.stimF_strength)

        randomF = self.forces.randomF(self.randomF_strength)

        upwindF = self.forces.upwindBiasF(self.windF_strength)

        wallRepulsiveF = self.forces.attraction_basin(self.spring_const, position)

        ################################################
        # calculate total force
        ################################################
        totalF = -self.damping_coeff * velocity +  randomF + upwindF + wallRepulsiveF + stimF
        ###############################

        return stimF, randomF, upwindF, wallRepulsiveF, totalF


    def _land(self, tsi, V):
        ''' trim excess timebins in arrays
        '''
        if tsi == 0:  # hack for if we need to chop a trajectory at the very start
            for k, array in V.iteritems():
                V[k] = array[:1]
        else:
            for k, array in V.iteritems():
                V[k] = array[:tsi - 1]
                V[k] = array[:tsi - 1]

        
        return V
        
        
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

    def _collide_with_wall(self, candidate_pos, candidate_velo):
        walls = self.windtunnel_obj.walls
        xpos, ypos, zpos = candidate_pos
        xvelo, yvelo, zvelo = candidate_velo

        crash = False

        # x dim
        if xpos > walls.upwind:  # reached far (upwind) wall (end)
#                    self.metadata['target_found'][0]  = False
#                    self.metadata['time_to_target_find'][0] = np.nan
#            return None, None  # this is needed to end at end
xpos = walls.upwind - 0.02  # teleport back inside
if self.collision_type == 'elastic':
    xvelo *= -1.
elif self.collision_type == 'crash':
    xvelo *= -1.
    crash = True
        if xpos < walls.downwind:  # too far behind
            xpos = walls.downwind + 0.02  # teleport back inside
            if self.collision_type == 'elastic':
                xvelo *= -1.
            elif self.collision_type == 'crash':
                xvelo *= -1.
                crash = True

        #y dim
        if ypos < walls.left:  # too left
            if self.collision_type == 'elastic':
                yvelo *= -1.
            elif self.collision_type == "crash":
                yvelo *= -1.
                crash = True
        if ypos > walls.right:  # too far right
            ypos = walls.right - 0.01
            if self.collision_type == 'elastic':
                yvelo *= -1.
            elif self.collision_type == 'crash':
                yvelo *= -1.
                crash = True

        # z dim
        if zpos > walls.ceiling:  # too far above
            zpos = walls.ceiling - 0.01
            if self.collision_type == 'elastic':
                zvelo *= -1.
            elif self.collision_type == "crash":
                zvelo *= -1.
                crash = True
        if zpos < walls.floor:  # too far below
            zpos = walls.floor + 0.01
            if self.collision_type == 'elastic':
                zvelo *= -1.
            elif self.collision_type == 'crash':
                zvelo *= -1.
                crash = True

        candidate_pos, candidate_velo = np.array([xpos, ypos, zpos]), np.array([xvelo, yvelo, zvelo])

        if crash is True:
            candidate_velo *= 0.2


        return candidate_pos, candidate_velo


    def _velocity_ceiling(self, candidate_velo):
        """check if we're seeing enormous velocities, which sometimes happens when running the optimization
         algoirithm. if so, cap the velocity instead of landing. this allows the optimizer to keep running.
        """
        for i, velo in enumerate(candidate_velo):
            if velo > 20:
                candidate_velo[i] = 20.
            elif velo < -20:
                candidate_velo[i] = -20.

        return candidate_velo



def gen_objects_and_fly(N_TRAJECTORIES,
                        TEST_CONDITION,
                        BETA,
                        RANDF_STRENGTH,
                        Fwind_strength,
                        F_stim_scale,
                        K,
                        initial_position_selection,
                        bounded=True,
                        verbose=True,
                        collision_type='crash'
                        ):
    """
    Params fitted using scipy.optimize

    """
    # instantiate a Roboskeeter
    skeeter = Agent(
        initial_position_selection=initial_position_selection,
        windF_strength=Fwind_strength,
        randomF_strength=RANDF_STRENGTH,
        experimental_condition=TEST_CONDITION,
        stimF_stength=F_stim_scale,
        spring_const=K,
        damping_coeff=BETA,
        time_max=15.,
        dt=0.01,
        bounded=bounded,
        collision_type=collision_type)

    # make the skeeter fly. this updates the trajectory_obj
    skeeter.fly(total_trajectories=N_TRAJECTORIES, verbose=verbose)
    
    return skeeter.trajectory_obj, skeeter
    
   
#    trajectories.describe(plot_kwargs = {'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True, 'force_violin':True})
    #trajectories.plot_single_3Dtrajectory()
#    
