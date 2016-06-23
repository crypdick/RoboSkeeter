# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:08:51 2015

@author: richard

TODO: implemement unit tests with nose
"""

import sys
import numpy as np
import pandas as pd
from flight import Flight
from decisions import Decisions
from observations import Observations
from random import choice as choose
from roboskeeter.math.math_toolbox import gen_symm_vecs

class Simulator:
    """Our simulated mosquito.
    """

    def __init__(self, experiment, agent_kwargs):
        """ Load params
        """
        # dump kwarg dictionary into the agent object
        for key, value in agent_kwargs.iteritems():
            setattr(self, key, value)

        self.decisions = Decisions(self.decision_policy, self.stimulus_memory_n_timesteps)

        if experiment.is_simulation:
            self._simulated_agent_init(experiment)

    def _simulated_agent_init(self, experiment):
        # defaults
        self.mass = 2.88e-6  # avg. mass of our colony (kg) =2.88 mg,
        self.time_max = 15.
        self.dt = 0.01
        self.max_bins = int(np.ceil(self.time_max / self.dt))  # N bins

        # from gassian fit to experimental control data
        self.initial_velocity_mu = 0.18
        self.initial_velocity_stdev = 0.08

        # useful aliases
        self.experiment = experiment
        self.windtunnel = self.experiment.environment.windtunnel
        self.bounded = self.experiment.experiment_conditions['bounded']
        self.boundary = self.windtunnel.boundary
        self.plume = self.experiment.environment.plume

        # useful lists TODO: get rid of?
        self.kinematics_list = ['position', 'velocity', 'acceleration']  # curvature?
        self.forces_list = ['total_f', 'random_f', 'stim_f']
        self.other_list = ['tsi', 'times', 'decision', 'plume_signal', 'in_plume']

        # mk forces
        self.flight = Flight(self.random_f_strength,
                             self.stim_f_strength,
                             self.damping_coeff)

        # turn thresh, in units deg s-1.
        # From Sharri:
        # it is the stdev of the broader of two Gaussians that fit the distribution of angular velocity

        # # create repulsion landscape
        # self._repulsion_funcs = repulsion_landscape3D.landscape(boundary=self.boundary)

    def fly(self, n_trajectories=1):
        """ runs _generate_flight n_trajectories times
        """
        df_list = []
        traj_i = 0
        try:
            if self.verbose:
                print """Starting simulations with {} plume model and {} decision policy.
                If you run out of patience, press <CTL>-C to stop generating simulations and
                cut to the chase scene.""".format(
                self.plume.plume_model, self.decision_policy)

            while traj_i < n_trajectories:
                # print updates
                if self.verbose:
                    sys.stdout.write("\rTrajectory {}/{}".format(traj_i + 1, n_trajectories))
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

                if traj_i == n_trajectories:
                    if self.verbose:
                        sys.stdout.write("\rSimulations finished. Performing deep magic.")
                        sys.stdout.flush()

        except KeyboardInterrupt:
            print "\n Simulations interrupted at iteration {}. Moving along...".format(traj_i)
            pass

        observations = Observations()
        observations.kinematics = pd.concat(df_list)  # concatenate all the data frames at once for performance boost.

        return observations

    def _generate_flight(self):
        """Generate a single trajectory using our model.
    
        First put everything into np arrays stored inside of a dictionary
        """
        dt = self.dt
        m = self.mass
        vector_dict = self._initialize_vector_dict()

        # # dynamically create easy-to-read aliases for the contents of vector_dict
        # for key, value in vector_dict.iteritems():
        #     exec(key + " = vector_dict['" + key + "']")
        # unpack vector dict into nicer aliases
        in_plume = vector_dict['in_plume']
        plume_signal = vector_dict['plume_signal']
        position = vector_dict['position']
        velocity = vector_dict['velocity']
        acceleration = vector_dict['acceleration']
        random_f = vector_dict['random_f']
        stim_f = vector_dict['stim_f']
        total_f = vector_dict['total_f']
        decision = vector_dict['decision']

        position[0] = self._set_init_position()
        velocity[0] = self._set_init_velocity()

        for tsi in vector_dict['tsi']:
            in_plume[tsi] = self.plume.check_in_plume_bounds(position[tsi])  # returns False for non-Bool plume

            decision[tsi], plume_signal[tsi] = self.decisions.make_decision(in_plume[tsi], velocity[tsi][1])

            if plume_signal[tsi] == 'X':  # this is an awful hack telling us to look up the gradient
                plume_signal[tsi] = self.plume.get_nearest_gradient(position[tsi])

            stim_f[tsi], random_f[tsi], total_f[tsi] = self.flight.calc_forces(velocity[tsi], decision[tsi], plume_signal[tsi])

            # calculate current acceleration
            acceleration[tsi] = total_f[tsi] / m

            # check if time is out, end loop before we solve for future velo, position
            if tsi == self.max_bins-1: # -1 because of how range() works
                vector_dict = self._land(tsi, vector_dict)
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

            position[tsi + 1] = candidate_pos
            velocity[tsi + 1] = candidate_velo

        # once flight is finished, make dictionary ready to be loaded into DF
        vector_dict = self._fix_vector_dict(vector_dict)

        return vector_dict

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


    def _collide_with_wall(self, candidate_pos, candidate_velo):
        walls = self.windtunnel.walls
        xpos, ypos, zpos = candidate_pos
        xvelo, yvelo, zvelo = candidate_velo
        teleport_distance = 0.005
        crash = False

        # print "test", candidate_velo

        # x dim
        if xpos < walls.downwind:  # too far behind
            xpos = walls.downwind + teleport_distance  # teleport back inside
            if self.collision_type == 'elastic':
                xvelo *= -1.
            elif self.collision_type == 'part_elastic':
                xvelo *= -self.restitution_coeff
            elif self.collision_type == 'crash':
                # xvelo *= -1.
                crash = True
            else:
                raise ValueError("unknown collision type {}".format(self.collision_type))
        if xpos > walls.upwind:  # reached far (upwind) wall (end)
            xpos = walls.upwind - teleport_distance  # teleport back inside
            if self.collision_type == 'elastic':
                xvelo *= -1.
            elif self.collision_type == 'part_elastic':
                xvelo *= -self.restitution_coeff
            elif self.collision_type == 'crash':
                # xvelo *= -1.
                crash = True

        # y dim
        if ypos < walls.left:  # too left
            ypos = walls.left + teleport_distance
            if self.collision_type == 'elastic':
                yvelo *= -1.
            elif self.collision_type == 'part_elastic':
                yvelo *= -self.restitution_coeff
            elif self.collision_type == "crash":
                # yvelo *= -1.
                crash = True
        if ypos > walls.right:  # too far right
            ypos = walls.right - teleport_distance
            if self.collision_type == 'elastic':
                yvelo *= -1.
            if self.collision_type == 'part_elastic':
                yvelo *= -self.restitution_coeff
            elif self.collision_type == 'crash':
                # yvelo *= -1.
                crash = True

        # z dim
        if zpos > walls.ceiling:  # too far above
            zpos = walls.ceiling - teleport_distance
            if self.collision_type == 'elastic':
                zvelo *= -1.
            if self.collision_type == 'part_elastic':
                zvelo *= -self.restitution_coeff
            elif self.collision_type == "crash":
                # zvelo *= -1.
                crash = True
        if zpos < walls.floor:  # too far below
            zpos = walls.floor + teleport_distance
            if self.collision_type == 'elastic':
                zvelo *= -1.
            if self.collision_type == 'part_elastic':
                zvelo *= -self.restitution_coeff
            elif self.collision_type == 'crash':
                # zvelo *= -1.
                crash = True

        try:
            candidate_pos, candidate_velo = np.array([xpos, ypos, zpos]), np.array([xvelo, yvelo, zvelo])
        except:
            print " cand velo", [xvelo, yvelo, zvelo], "before", candidate_velo

        if crash is True and self.collision_type is 'crash':
            candidate_velo *= self.crash_coeff


        return candidate_pos, candidate_velo

    def _initialize_vector_dict(self):
        """
        initialize np arrays, store in dictionary
        """
        V = {}

        for name in self.kinematics_list + self.forces_list:
            V[name] = np.full((self.max_bins, 3), np.nan)

        V['tsi'] = np.arange(self.max_bins)
        V['times'] = np.linspace(0, self.time_max, self.max_bins)
        V['in_plume'] = np.zeros(self.max_bins, dtype=bool)
        V['plume_signal'] = np.array([None] * self.max_bins)
        V['decision'] = np.array([None] * self.max_bins)

        return V

    def _set_init_velocity(self):
        initial_velocity_norm = np.random.normal(self.initial_velocity_mu, self.initial_velocity_stdev, 1)

        unit_vector = gen_symm_vecs(3)
        velocity_vec = initial_velocity_norm * unit_vector

        return velocity_vec

    def _set_init_position(self):
        ''' puts the agent in an initial position, usually within the bounds of the
        cage

        Options: [the cage] door, or anywhere in the plane at x=.1 meters

        set initial velocity from fitted distribution
        '''

        # generate random intial velocity condition using normal distribution fitted to experimental data
        if self.initial_position_selection == 'realistic':
            """these were calculated by taking selecting the initial positions of all observed trajectories in all
            conditions. Then, for each dimension, I calculated the distance of each initial position to the nearest wall
            in that dimension (i.e. for each z I calculated the distance to the floor and ceiling and selected
            the smallest distance. Then, Decisions aand"""
            downwind, upwind, left, right, floor, ceiling = self.boundary
            x_avg_dist_to_wall = 0.268
            y_avg_dist_to_wall = 0.044
            z_avg_dist_to_wall = 0.049
            x = choose([(downwind + x_avg_dist_to_wall), (upwind - x_avg_dist_to_wall)])
            y = choose([left + y_avg_dist_to_wall, (right - y_avg_dist_to_wall)])
            z = ceiling - z_avg_dist_to_wall
            initial_position = np.array([x,y,z])
        elif self.initial_position_selection == 'downwind_high':
            initial_position = np.array(
                [0.05, np.random.uniform(-0.127, 0.127), 0.2373])  # 0.2373 is mode of z pos distribution
        elif type(self.initial_position_selection) is list:
            initial_position = np.array(self.initial_position_selection)
        elif self.initial_position_selection == "door":  # start trajectories as they exit the front door
            initial_position = np.array([0.1909, np.random.uniform(-0.0381, 0.0381), np.random.uniform(0., 0.1016)])
            # FIXME cage is actually suspending above floor
        elif self.initial_position_selection == 'downwind_plane':
            initial_position = np.array([0.1, np.random.uniform(-0.127, 0.127), np.random.uniform(0., 0.254)])
        else:
            raise Exception('invalid agent position specified: {}'.format(self.initial_position_selection))

        return initial_position

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