__author__ = 'richard'

import numpy as np

from analysis import math_toolbox


class Forces():
    def __init__(self, randomF_strength, stimF_strength, stimulus_memory, decision_policy):
        # TODO: save coefficients here
        self.randomF_strength = randomF_strength
        self.stimF_strength = stimF_strength
        self.stimulus_memory = stimulus_memory
        self.decision_policy = decision_policy

    def randomF(self, dim=3):
        """Generate random-direction force vector at each timestep from double-
        exponential distribution given exponent term rf.

        Args:
            rf: random force distribution exponent (float)

        Returns:
            random force x and y components (array)
        """
        # TODO: make randomF draw from the canonical eqn for random draws Rich taught you
        ends = math_toolbox.gen_symm_vecs(3)
        force = self.randomF_strength * ends

        return force

    def stimF(self, kwargs):
        """given force direction and strength, return a force vector
        decision policies: 'cast_only', 'surge_only', 'cast+surge'

        FIXME: if cast in decision_policy
        """
        if self.decision_policy is 'cast_only':
            force = self._stimF_cast_only(kwargs)
        elif self.decision_policy is 'surge_only':
            force = self._stimF_surge_only(kwargs)
        elif self.decision_policy is 'cast+surge':
            raise NotImplementedError
        elif self.decision_policy is 'gradient':
            force = self._stimF_temp_gradient(kwargs)
        elif self.decision_policy is 'ignore':
            force = np.array([0., 0., 0.])
        else:
            print "stimF error", self.decision_policy
            force = np.array([0., 0., 0.])

        return force

    def _stimF_cast_only(self, kwargs):
        """
        :return:
        """
        tsi = kwargs['tsi']
        plume_interaction_history = kwargs['plume_interaction_history']
        triggered_tsi = kwargs['triggered_tsi']

        empty = np.array([0., 0., 0.])
        inside_ago = abs(tsi - triggered_tsi['stimulus'])
        exit_ago = abs(tsi - triggered_tsi['exit'])
        cast_strength = self.stimF_strength / 10
        if tsi == 0:
            return empty
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

    def _stimF_surge_only(self, kwargs):
        tsi = kwargs['tsi']
        plume_interaction_history = kwargs['plume_interaction_history']
        triggered_tsi = kwargs['triggered_tsi']

        if plume_interaction_history[tsi] is 'inside':
            force = np.array([self.stimF_strength, 0., 0.])
        else:
            force = np.array([0., 0., 0.])

        return force

    def _stimF_temp_gradient(self, kwargs):
        """gradient vector * stimFstrength"""
        df = kwargs['gradient']

        scalar = self.stimF_strength
        vector = df[['gradient_x', "gradient_y", "gradient_z"]].values
        force = scalar * vector
        norm = np.linalg.norm(force)
        if norm > 1e-5:
            force *= 1e-5/norm
        if np.isnan(force).any():
            raise ValueError("Nans in stimF!! {} {}".format(force, vector))
        if np.isinf(force).any():
            raise ValueError("infs in stimF! {} {}".format(force, vector))


        return force
