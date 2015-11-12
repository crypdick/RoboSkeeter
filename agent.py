# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:08:51 2015

@author: richard

TODO: implemement unit tests with nose
"""

import sys

import numpy as np
import pandas as pd

import forces


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

    def __init__(self, experiment, agent_kwargs):
        """generate the agent"""
        # defaults
        self.mass = 2.88e-6  # avg. mass of our colony (kg) =2.88 mg,
        self.time_max = 15.
        self.dt = 0.01
        self.max_bins = int(np.ceil(self.time_max / self.dt))  # N bins
        self.initial_velocity_stdev = 0.01

        for key in agent_kwargs:
            setattr(self, key, agent_kwargs[key])

        self.kinematics_list = ['position', 'velocity', 'acceleration']
        self.forces_list = ['totalF', 'randomF', 'stimF']
        self.other_list = ['tsi', 'times', 'inPlume', 'plume_interaction', 'turning', 'heading_angle',
                           'velocity_angular']
        
        # things fed to the class
        self.experiment = experiment

        # useful aliases
        self.windtunnel_obj = self.experiment.windtunnel
        self.bounded = self.experiment.bounded
        self.boundary = self.windtunnel_obj.boundary
        self.plume_obj = self.experiment.plume
        self.trajectory_obj = self.experiment.trajectories

        # mk forces
        self.forces = forces.Forces(self.randomF_strength, self.stimF_strength, self.stimulus_memory,
                                    self.decision_policy)

        # turn thresh, in units deg s-1.
        # From Sharri:
        # it is the stdevof the broader of two Gaussians that fit the distribution of angular velocity
        self.turn_threshold = 433.5  # TODO: move this to trajectories       

        
        # # create repulsion landscape
        # self._repulsion_funcs = repulsion_landscape3D.landscape(boundary=self.boundary)



    def fly(self, total_trajectories=1, verbose=True):
        ''' iterates self._fly_single() total_trajectories times
        '''
        df_list = []
        traj_i = 0
        while traj_i < total_trajectories:
            if verbose is True:
                sys.stdout.write("\rTrajectory {}/{}".format(traj_i + 1, total_trajectories))
                sys.stdout.flush()

            array_dict = self._generate_flight()

            # if len(array_dict['velocity_x']) < 5:  # hack to catch when optimizer makes trajectories explode
            #     print "catching explosion"
            #     break

            # add label column to enumerate the trajectories
            array_len = len(array_dict['tsi'])
            array_dict['trajectory_num'] = [traj_i] * array_len

            # mk df, add to list of dfs
            df = pd.DataFrame(array_dict)
            # df = df.set_index(['trajectory_num'])
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

    def _generate_flight(self):
        """Generate a single trajectory using our model.
    
        First put everything into np arrays stored inside of a dictionary
        """
        dt = self.dt
        m = self.mass
        V = self._initialize_vectors()

        # dynamically create easy-to-read aliases for the contents of V
        for key, value in V.iteritems():
            exec(key + " = V['" + key + "']")

        position[0], velocity[0] = self._set_init_pos_and_velo()

        tsi_plume_last_sighted = -10000000  # a long time ago

        for tsi in V['tsi']:
            inPlume[tsi] = self.plume_obj.in_plume(position[tsi])

            plume_interaction[tsi] = self._plume_interaction(tsi, inPlume, velocity[tsi][1])
            if plume_interaction[tsi] is 'inside':
                tsi_plume_last_sighted = tsi
            stimF[tsi], randomF[tsi], totalF[tsi] = \
                self._calc_forces(tsi, velocity[tsi], plume_interaction, tsi_plume_last_sighted)


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

        V = self._fix_vector_dict(V)

        return V

    def _calc_forces(self, tsi, velocity_now, plume_interaction_history, tsi_plume_last_sighted):
        ################################################
        # Calculate driving forces at this timestep
        ################################################
        randomF = self.forces.randomF(self.randomF_strength)
        stimF = self.forces.stimF((tsi, tsi_plume_last_sighted, plume_interaction_history))

        ################################################
        # calculate total force
        ################################################
        totalF = -self.damping_coeff * velocity_now + randomF + stimF
        ###############################

        return stimF, randomF, totalF


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

    def _plume_interaction(self, tsi, inPlume, velocity_y_now):
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
        if tsi == 0:  # always start searching
            state = 'outside'
        elif current_state == 0 and past_state == 0:
            # we are not in plume and weren't in last ts
            state = 'outside'
        elif current_state == 1 and past_state == 0:
            # entering plume
            state = 'inside'
        elif current_state == 1 and past_state == 1:
            # we stayed in the plume
            state = 'inside'
        elif current_state == 0 and past_state == 1:
            # exiting the plume
            if velocity_y_now <= 0:
                state = 'Exit left'
            else:
                state = 'Exit right'

        return state


    def _collide_with_wall(self, candidate_pos, candidate_velo):
        # TODO: move to wall
        walls = self.windtunnel_obj.walls
        xpos, ypos, zpos = candidate_pos
        xvelo, yvelo, zvelo = candidate_velo
        teleport_distance = 0.005
        crash = False

        # x dim
        if xpos < walls.downwind:  # too far behind
            xpos = walls.downwind + teleport_distance  # teleport back inside
            if self.collision_type == 'elastic':
                xvelo *= -1.
            elif self.collision_type == 'crash':
                # xvelo *= -1.
                crash = True
        if xpos > walls.upwind:  # reached far (upwind) wall (end)
            xpos = walls.upwind - teleport_distance  # teleport back inside
            if self.collision_type == 'elastic':
                xvelo *= -1.
            elif self.collision_type == 'crash':
                # xvelo *= -1.
                crash = True

        #y dim
        if ypos < walls.left:  # too left
            ypos = walls.left + teleport_distance
            if self.collision_type == 'elastic':
                yvelo *= -1.
            elif self.collision_type == "crash":
                # yvelo *= -1.
                crash = True
        if ypos > walls.right:  # too far right
            ypos = walls.right - teleport_distance
            if self.collision_type == 'elastic':
                yvelo *= -1.
            elif self.collision_type == 'crash':
                # yvelo *= -1.
                crash = True

        # z dim
        if zpos > walls.ceiling:  # too far above
            zpos = walls.ceiling - teleport_distance
            if self.collision_type == 'elastic':
                zvelo *= -1.
            elif self.collision_type == "crash":
                # zvelo *= -1.
                crash = True
        if zpos < walls.floor:  # too far below
            zpos = walls.floor + teleport_distance
            if self.collision_type == 'elastic':
                zvelo *= -1.
            elif self.collision_type == 'crash':
                # zvelo *= -1.
                crash = True

        candidate_pos, candidate_velo = np.array([xpos, ypos, zpos]), np.array([xvelo, yvelo, zvelo])

        if crash is True and self.collision_type is 'crash':
            candidate_velo *= self.crash_coeff


        return candidate_pos, candidate_velo

    def _initialize_vectors(self):
        """
        initialize np arrays, store in dictionary
        """
        V = {}

        for name in self.kinematics_list + self.forces_list:
            V[name] = np.full((self.max_bins, 3), np.nan)

        V['tsi'] = np.arange(self.max_bins)
        V['times'] = np.linspace(0, self.time_max, self.max_bins)
        V['inPlume'] = np.full(self.max_bins, -1, dtype=np.uint8)
        V['plume_interaction'] = np.array([None] * self.max_bins)
        V['turning'] = np.array([None] * self.max_bins)
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
        selection = self.experiment.initial_position_selection

        if type(selection) is list:
            initial_position = np.array(selection)
        if selection == "door":  # start trajectories as they exit the front door
            initial_position = np.array([0.1909, np.random.uniform(-0.0381, 0.0381), np.random.uniform(0., 0.1016)])
            # FIXME cage is actually suspending above floor
        if selection == 'downwind_plane':
            initial_position = np.array([0.1, np.random.uniform(-0.127, 0.127), np.random.uniform(0., 0.254)])
        if selection == 'downwind_high':
            initial_position = np.array(
                [0.05, np.random.uniform(-0.127, 0.127), 0.2373])  # 0.2373 is mode of z pos distribution
        else:
            raise Exception('invalid agent position specified: {}'.format(selection))

        return initial_position, initial_velocity

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

    def _fix_vector_dict(self, dct):
        # prepare dict for loading into pandas (dataframe only accepts 1D vectors)
        # split xyz dicts into separate x, y, z vectors for dataframe

        fixed_dct = {}
        for kinematic in self.kinematics_list + self.forces_list:
            fixed_dct[kinematic + '_x'], fixed_dct[kinematic + '_y'], fixed_dct[kinematic + '_z'] = np.split(
                dct[kinematic], 3, axis=1)

        # migrate rest of dict
        for v in self.other_list:
            fixed_dct[v] = dct[v]

        # fix pandas bug when trying to load (R,1) arrays when it expects (R,) arrays
        for key, dct in fixed_dct.iteritems():
            fixed_dct[key] = fixed_dct[key].reshape(len(dct))
            if fixed_dct[key].size == 0:
                fixed_dct[key] = np.array([0.])  # hack so that kde calculation doesn't freeze on empty arrays

        return fixed_dct

    
   
#    trajectories.describe(plot_kwargs = {'trajectories':False, 'heatmap':True, 'states':True, 'singletrajectories':False, 'force_scatter':True, 'force_violin':True})
    #trajectories.plot_single_3Dtrajectory()
#    
