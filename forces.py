__author__ = 'richard'

import numpy as np

from scripts import math_sorcery


class Forces():
    def __init__(self, randomF_strength, stimF_strength, stimulus_memory, decision_policy):
        # TODO: save coefficients here
        self.randomF_strength = randomF_strength
        self.stimF_strength = stimF_strength
        self.stimulus_memory = stimulus_memory
        self.decision_policy = decision_policy

    def randomF(self, dim=3, kind='constant'):
        """Generate random-direction force vector at each timestep from double-
        exponential distribution given exponent term rf.

        Args:
            rf: random force distribution exponent (float)

        Returns:
            random force x and y components (array)
        """
        if kind is 'lognorm':
            MAG_THRESH = 3e-4

            ends = math_sorcery.gen_symm_vecs(3)
            # following params were fitted from the Dickinson fligt data
            mu = 0.600023812816
            sigma = 0.719736466122
            # scale = 1.82216219069
            mag = np.random.lognormal(mean=mu, sigma=sigma, size=1)
            if mag * self.randomF_strength > MAG_THRESH:  # filter out huge magnitudes
                force = ends * MAG_THRESH
            else:
                force = mag * self.randomF_strength * ends

        elif kind is 'constant':
            ends = math_sorcery.gen_symm_vecs(3)
            force = self.randomF_strength * ends

        return force

    def stimF(self, args):
        """given force direction and strength, return a force vector
        decision policies: 'cast_only', 'surge_only', 'cast+surge'
        """
        if self.decision_policy is 'cast_only':
            force = self._stimF_cast_only(*args)
        elif self.decision_policy is 'surge_only':
            force = self._stimF_surge_only(*args)
        elif self.decision_policy is 'cast+surge':
            raise NotImplementedError
        else:
            print "stimF error", self.decision_policy
            force = np.array([0., 0., 0.])

        return force

    def _stimF_cast_only(self, tsi, plume_interaction_history, last_triggered):
        """
        :param tsi:
        :param tsi_plume_last_sighted:
        :param plume_interaction_history: timestep at w hich
        :return:
        """
        empty = np.array([0., 0., 0.])
        exit_ago = abs(tsi - last_triggered['exit'])
        cast_strength = self.stimF_strength / 10
        if tsi == 0:
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

    def _stimF_surge_only(self, tsi, plume_interaction_history, _):
        if plume_interaction_history[tsi] is 'inside':
            force = np.array([self.stimF_strength, 0., 0.])
        else:
            force = np.array([0., 0., 0.])

        return force
