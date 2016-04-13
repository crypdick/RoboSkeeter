__author__ = 'richard'
import numpy as np
from roboskeeter.math import math_toolbox


class Flight():
    def __init__(self, random_f_strength, stim_f_strength, damping_coeff, stimulus_memory_n_timesteps, decision_policy):
        self.random_f_strength = random_f_strength
        self.stim_f_strength = stim_f_strength
        self.damping_coeff = damping_coeff

    def random(self):
        """Generate random-direction force vector at each timestep from double-
        exponential distribution given exponent term rf.
        """
        # TODO: make randomF draw from the canonical eqn for random draws Rich taught you
        ends = math_toolbox.gen_symm_vecs(3)
        force = self.random_f_strength * ends

        return force

    def stimulus(self, kwargs):
        """given force direction and strength, return a force vector
        decision policies: 'cast_only', 'surge_only', 'cast+surge'

        TODO: review this func to make sure castsurge would work
        """
        force = np.array([0., 0., 0.])

        if 'cast' in self.decision_policy:
            force += self.behaviors.cast(kwargs)
        if 'surge' in self.decision_policy:
            force += self.behaviors.surge_upwind(kwargs)
        if 'gradient' in self.decision_policy:
            force += self.behaviors.surge_up_gradient(kwargs)
        if 'ignore' in self.decision_policy:
            pass

        return force

    def calc_forces(self, current_velocity, stim_f_kwargs):
        # TODO: make sure all the args can be the same for all the different behavioral policies
        ################################################
        # Calculate driving forces at this timestep
        ################################################
        random_f = self.random()

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
            stim_f = np.array([0.,0.,0.])
        else:
            stim_f = self.forces.stimulus(kwargs)

        ################################################
        # calculate total force
        ################################################
        total_f = -self.damping_coeff * current_velocity + random_f + stim_f
        ###############################

        return stim_f, random_f, total_f

class Behaviors:  # return a decision
    def __init__(self, decision_policy):
        self.decision_policy = decision_policy

    def cast(self, kwargs):
        # TODO: review cast
        tsi = kwargs['tsi']
        plume_interaction_history = kwargs['plume_interaction_history']
        triggered_tsi = kwargs['triggered_tsi']

        empty = np.array([0., 0., 0.])  # FIXME naming
        # TODO: check if this is updating triggers, and if it should be
        inside_ago = abs(tsi - triggered_tsi['stimulus'])
        exit_ago = abs(tsi - triggered_tsi['exit'])
        cast_strength = self.stimF_strength / 10
        if tsi == 0:
            return empty  # FIXME naming
        elif inside_ago < exit_ago:  # if we re-encounter the plume, stop casting
            return empty
        elif exit_ago <= self.stimulus_memory:  # stimulus encountered recently
            # print "we have a memory!"
            # print "currently {tsi}, last {last}, difference {diff}".format(tsi=tsi, last=last_triggered['exit'], diff=exit_ago)
            experience = plume_interaction_history[tsi - exit_ago]
            # if experience in 'outside':
            #     pass # keep going back
            # elif experience is 'inside':
            #     pass # keep going back
            if experience == 'Exit left':
                return np.array([0., cast_strength, 0.])  # cast right
            elif experience == 'Exit right':
                return np.array([0., -cast_strength, 0.])  # cast left
            else:
                # print "valueerror! experience", experience, "tsi", tsi
                # print experience == 'Exit right', experience
                raise ValueError('no such experience known: {}'.format(experience))
                # except ValueError:
                #     print "tsi", tsi, "memory", memory[:tsi], plume_interaction_history
                # except TypeError:
                #     print "memory type", memory, type(memory)


        else:  # no recent memory of stimulus
            current_experience = plume_interaction_history[tsi]
            if current_experience in ['outside', 'inside']:
                force = empty
            else:
                print "plume_interaction_history", plume_interaction_history, plume_interaction_history[:tsi]
                print "current_experience", current_experience
                raise ValueError("no such experience {} at tsi {}".format(current_experience, tsi))

            return force

    def surge_upwind(self, kwargs):
        tsi = kwargs['tsi']
        plume_interaction_history = kwargs['plume_interaction_history']

        if plume_interaction_history[tsi] is 'inside':
            force = np.array([self.stimF_strength, 0., 0.])
        else:
            force = np.array([0., 0., 0.])

        return force

    def surge_up_gradient(self, kwargs):
        """gradient vector norm * stimF strength"""
        df = kwargs['gradient']

        scalar = self.stimF_strength
        vector = df[['gradient_x', "gradient_y", "gradient_z"]].values
        force = scalar * vector

        # stimF here is proportional to norm of gradient. in order to avoid huge stimFs, we put a ceiling on the
        # size of stimF
        ceiling = 1e-5  # TODO: parameterize in function call
        norm = np.linalg.norm(force)
        if norm > ceiling:
            force *= 1e-5 / norm

        # catch problems in stimF
        if np.isnan(force).any():
            raise ValueError("Nans in stimF!! {} {}".format(force, vector))
        if np.isinf(force).any():
            raise ValueError("infs in stimF! {} {}".format(force, vector))

        return force

