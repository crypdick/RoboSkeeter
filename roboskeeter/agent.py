# -*- coding: utf-8 -*-
"""
Created on Tue May  5 21:08:51 2015

@author: richard

TODO: implemement unit tests with nose
"""

import sys
import numpy as np
import pandas as pd
from flights import Flights
from forces import Forces


class Agent:
    """Our simulated mosquito.
    """

    def __init__(self, experiment, agent_kwargs):
        """ Load params
        """
        # dump kwarg dictionary into the agent object
        for key, value in agent_kwargs.iteritems():
            setattr(self, key, value)

        if experiment.is_simulation:
            self._simulated_agent_init(experiment)

    def _simulated_agent_init(self, experiment):
        # defaults
        self.mass = 2.88e-6  # avg. mass of our colony (kg) =2.88 mg,
        self.time_max = 15.
        self.dt = 0.01
        self.max_bins = int(np.ceil(self.time_max / self.dt))  # N bins
        self.initial_velocity_stdev = 0.01

        # useful aliases
        self.experiment = experiment
        self.windtunnel = self.experiment.environment.windtunnel
        self.bounded = self.experiment.experiment_conditions['bounded']
        self.boundary = self.windtunnel.boundary
        self.plume = self.experiment.environment.plume

        # useful lists TODO: get rid of?
        self.kinematics_list = ['position', 'velocity', 'acceleration']  # curvature?
        self.forces_list = ['totalF', 'randomF', 'stimF']
        self.other_list = ['tsi', 'times', 'in_plume', 'plume_interaction']

        # mk forces
        self.forces = Forces(self.randomF_strength,
                             self.stimF_strength,
                             self.stimulus_memory_N_timesteps,
                             self.decision_policy)

        # turn thresh, in units deg s-1.
        # From Sharri:
        # it is the stdev of the broader of two Gaussians that fit the distribution of angular velocity

        # # create repulsion landscape
        # self._repulsion_funcs = repulsion_landscape3D.landscape(boundary=self.boundary)

    def fly(self, total_trajectories=1):
        """ runs _generate_flight total_trajectories times
        """
        df_list = []
        traj_i = 0
        try:
            if self.verbose:
                print 'Starting simulations. If you run out of patience, hit <CTL>-C to end the loop early.'

            while traj_i < total_trajectories:
                if self.verbose:
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
                    if self.verbose:
                        sys.stdout.write("\rSimulations finished. Performing deep magic.")
                        sys.stdout.flush()

        except KeyboardInterrupt:
            print "\n Simulations interrupted at iteration {}. Moving along...".format(traj_i)
            pass

        flights = Flights()
        flights.kinematics = pd.concat(df_list)  # concatenate all the data frames at once for performance boost.

        return flights

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

        triggered_tsi = {'stimulus': -10000000, 'exit': -10000000}  # a long time ago

        for tsi in V['tsi']:
            in_plume[tsi] = self.plume.check_for_plume(position[tsi])  # TODO: export to behavior module

            plume_interaction[tsi], triggered_tsi = self._plume_interaction(tsi, in_plume, velocity[tsi][1],
                                                                            triggered_tsi)  # TODO: export

            stimF[tsi], randomF[tsi], totalF[tsi] = \
                self._calc_forces(tsi, velocity[tsi], plume_interaction, triggered_tsi, position[tsi])

            # calculate current acceleration
            acceleration[tsi] = totalF[tsi] / m

            # check if time is out, end loop before we solve for future velo, position
            if tsi == self.max_bins-1: # -1 because of how range() works
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

    def _calc_forces(self, tsi, velocity_now, plume_interaction_history, triggered_tsi, position_now):
        ################################################
        # Calculate driving forces at this timestep
        ################################################
        randomF = self.forces.randomF()

        if "gradient" in self.decision_policy:
            raise NotImplementedError
            kwargs = {"gradient": self.plume.get_nearest_data(position_now)}
        elif "surge" in self.decision_policy or "cast" in self.decision_policy:
            kwargs = {"tsi": tsi,
                      "plume_interaction_history": plume_interaction_history,
                      "triggered_tsi": triggered_tsi,
                      "position_now": position_now}
        else:
            raise ValueError("unknown decision policy {}".format(self.decision_policy))

        if self.experiment.experiment_conditions['condition'] in 'controlControlCONTROL' or self.decision_policy == 'ignore':
            stimF = np.array([0.,0.,0.])
        else:
            stimF = self.forces.stimF(kwargs)

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

    def _plume_interaction(self, tsi, in_plume, velocity_y_now, last_triggered):
        """
        out2out - searching
         (or orienting, but then this func shouldn't be called)
        out2in - entering plume
        in2in - staying
        in2out - exiting
            {Left_plume Exit left, Left_plume Exit right
            Right_plume Exit left, Right_plume Exit right}
        """
        # TODO: only run this stuff if running a relevant decision policy
        current_in_plume, past_in_plume = in_plume[tsi], in_plume[tsi - 1]

        if tsi == 0:  # always start searching
            state = 'outside'
        elif current_in_plume == False and past_in_plume == False:
            # we are not in plume and weren't in last ts
            state = 'outside'
        elif current_in_plume == True and past_in_plume == False:
            # entering plume
            state = 'inside'
        elif current_in_plume == 1 and past_in_plume == True:
            # we stayed in the plume
            state = 'inside'
        elif current_in_plume == False and past_in_plume == True:
            # exiting the plume
            if velocity_y_now <= 0:
                state = 'Exit left'
            else:
                state = 'Exit right'
        else:
            raise Exception("This error shouldn't ever run")

        if state is 'outside':
            pass
        elif state is 'inside':
            last_triggered['stimulus'] = tsi
        elif 'Exit' in state:
            last_triggered['exit'] = tsi

        return state, last_triggered


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

    def _initialize_vectors(self):
        """
        initialize np arrays, store in dictionary
        """
        V = {}

        for name in self.kinematics_list + self.forces_list:
            V[name] = np.full((self.max_bins, 3), np.nan)

        V['tsi'] = np.arange(self.max_bins)
        V['times'] = np.linspace(0, self.time_max, self.max_bins)
        V['in_plume'] = np.zeros(self.max_bins, dtype=bool)
        V['plume_interaction'] = np.array([None] * self.max_bins)

        return V

    def _set_init_pos_and_velo(self):
        ''' puts the agent in an initial position, usually within the bounds of the
        cage

        Options: [the cage] door, or anywhere in the plane at x=.1 meters

        set initial velocity from fitted distribution
        '''

        # generate random intial velocity condition using normal distribution fitted to experimental data
        initial_velocity = np.random.normal(0, self.initial_velocity_stdev, 3)
        selection = self.initial_position_selection

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

class Behavior():
    # TODO: implement behavior class
    # TODO: implement plume boundary interaction
    # TODO: implement plume memory
    def __init__(self, decision_policy):
        self.decision_policy = decision_policy


class Gradient_Following(Behavior):
    pass # TODO implement gradient following